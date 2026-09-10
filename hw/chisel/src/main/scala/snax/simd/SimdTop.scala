package snax.simd

import chisel3._
import chisel3.util._

import snax.DataPathExtension.DataPathExtensionHost
import snax.readerWriter.AddressGenUnitCfgIO
import snax.readerWriter.Reader
import snax.readerWriter.ReaderWriterCfgIO
import snax.readerWriter.Writer
import snax.reqRspManager.ReqRspManager
import snax.reqRspManager.SnaxReqRspIO
import snax.utils.BasicCounter
import snax.utils.SparseTCDMReq
import snax.utils.SparseTCDMRsp

/** One SIMD task, as decoded from the read-write CSR bank.
  *
  * The field ORDER here is the CSR order, and it deliberately mirrors the xDMA's: reader AGU, reader channel mask,
  * extension enable + user CSRs, writer AGU, writer channel/byte mask. Keeping the order identical is what makes
  * the snax-xdma-lib -> snax-simd-lib port mechanical.
  */
class SimdTaskCfg(param: SimdParam) extends Bundle {
  val readerAgu = new AddressGenUnitCfgIO(param.readerParam.aguParam)
  val readerRw  = new ReaderWriterCfgIO(param.readerParam)
  // Sized to at least 1 so the Bundle stays legal for a build with no extension / no user CSR; the surplus
  // entries are tied off and never reach the extension host.
  val extEnable = UInt(math.max(1, param.extParam.length).W)
  val extCsr    = Vec(math.max(1, param.extUserCsrNum), UInt(32.W))
  val writerAgu = new AddressGenUnitCfgIO(param.writerParam.aguParam)
  val writerRw  = new ReaderWriterCfgIO(param.writerParam)

  /** Split a flat CSR vector into this bundle, returning what is left (which must be the single start CSR). */
  def connectWithList(csrList: IndexedSeq[UInt]): IndexedSeq[UInt] = {
    var remaining = csrList
    remaining = readerAgu.connectWithList(remaining)
    remaining = readerRw.connectWithList(remaining)
    if (param.extParam.nonEmpty) {
      extEnable := remaining.head
      remaining = remaining.tail
      for (i <- 0 until param.extUserCsrNum) {
        extCsr(i) := remaining.head
        remaining = remaining.tail
      }
      for (i <- param.extUserCsrNum until extCsr.length) extCsr(i) := 0.U
    } else {
      extEnable := 0.U
      extCsr.foreach(_ := 0.U)
    }
    remaining = writerAgu.connectWithList(remaining)
    remaining = writerRw.connectWithList(remaining)
    remaining
  }
}

class SimdTopIO(param: SimdParam) extends Bundle {
  val csrIO = new SnaxReqRspIO(addrWidth = param.cfgParam.addrWidth, dataWidth = param.cfgParam.dataWidth)

  val tcdmReader = new Bundle {
    val req = Vec(
      param.readerParam.tcdmParam.numChannel,
      Decoupled(
        new SparseTCDMReq(
          param.readerParam.tcdmParam.addrWidth,
          param.readerParam.tcdmParam.dataWidth
        )
      )
    )
    val rsp = Vec(
      param.readerParam.tcdmParam.numChannel,
      Flipped(Valid(new SparseTCDMRsp(dataWidth = param.readerParam.tcdmParam.dataWidth)))
    )
  }

  val tcdmWriter = new Bundle {
    val req = Vec(
      param.writerParam.tcdmParam.numChannel,
      Decoupled(
        new SparseTCDMReq(
          param.writerParam.tcdmParam.addrWidth,
          param.writerParam.tcdmParam.dataWidth
        )
      )
    )
  }

  val status = new Bundle {
    val busy = Output(Bool())
  }
}

/** The standalone SIMD engine: CSR in, TCDM in and out, and the DataPathExtension chain in between.
  *
  * `reader -> extension chain -> writer`, which is exactly the reader half of `XDMADataPath` with the data switch
  * removed. Everything cross-cluster (AXI, remote cfg, junctions, chain roles, multicast) is absent by
  * construction rather than tied off.
  */
class SimdTop(param: SimdParam, clusterName: String = "unnamed_cluster") extends Module with RequireAsyncReset {
  override val desiredName = s"${clusterName}_simd"

  private val namePrefix = s"${clusterName}_simd"

  val io = IO(new SimdTopIO(param))

  // ------------------------------------------------------------------ CSR
  val csrManager = Module(
    new ReqRspManager(
      numReadWriteReg = param.totalRwCsrNum,
      numReadOnlyReg  = param.totalRoCsrNum,
      addrWidth       = param.cfgParam.addrWidth,
      ioDataWidth     = param.cfgParam.dataWidth,
      regDataWidth    = 32,
      moduleTagName   = s"${namePrefix}_"
    )
  )
  csrManager.io.reqRspIO <> io.csrIO

  // ------------------------------------------------------------------ cfg decode + task queue
  val cfgIn = Wire(Decoupled(new SimdTaskCfg(param)))
  private val leftover = cfgIn.bits.connectWithList(csrManager.io.readWriteRegIO.bits.toIndexedSeq)
  require(
    leftover.length == 1,
    s"SIMD CSR map mismatch: ${leftover.length} CSRs left after decoding a task, expected exactly 1 (the start " +
      s"register). SimdParam.totalRwCsrNum (${param.totalRwCsrNum}) disagrees with what connectWithList consumes."
  )
  cfgIn.valid                        := csrManager.io.readWriteRegIO.valid
  csrManager.io.readWriteRegIO.ready := cfgIn.ready

  // Two entries so the core can stage the next task while one runs -- the multi-pass chains (softmax T1/T2/T3,
  // and the FlashAttention inner loop) otherwise stall on CSR writes between passes.
  val taskQueue = Module(new Queue(new SimdTaskCfg(param), entries = 2) {
    override val desiredName = s"${namePrefix}_taskQueue"
  })
  taskQueue.io.enq <> cfgIn

  // ------------------------------------------------------------------ datapath
  val reader = Module(new Reader(param.readerParam, moduleNamePrefix = namePrefix))
  val writer = Module(new Writer(param.writerParam, moduleNamePrefix = namePrefix))

  val extHost = Module(
    new DataPathExtensionHost(
      param.extParam,
      dataWidth        = param.dataWidth,
      headCut          = false,
      tailCut          = false,
      halfCut          = false,
      moduleNamePrefix = namePrefix
    )
  )

  reader.io.tcdmReq <> io.tcdmReader.req
  reader.io.tcdmRsp <> io.tcdmReader.rsp
  writer.io.tcdmReq <> io.tcdmWriter.req

  extHost.io.data.in <> reader.io.data
  extHost.io.data.out <> writer.io.data

  // The task's cfg is held by the queue for the whole task (it is popped only at retire), which is what the
  // reader needs: it reads `aguCfg.temporalStrides(0)` / `temporalBounds(0)` combinationally to drive its
  // stride-0 broadcast repeater, not just at the start pulse.
  //
  // Between tasks `Queue.io.deq.bits` presents the last popped entry. That is harmless here because nothing acts
  // on the cfg without a start pulse -- unlike the xDMA, where a stale junction enable was read combinationally
  // by the data switch and banked a bogus finish credit.
  private val curCfg = taskQueue.io.deq.bits
  reader.io.aguCfg          := curCfg.readerAgu
  reader.io.readerwriterCfg := curCfg.readerRw
  writer.io.aguCfg          := curCfg.writerAgu
  writer.io.readerwriterCfg := curCfg.writerRw

  if (param.extParam.nonEmpty) {
    extHost.io.cfg.enable := curCfg.extEnable
    extHost.io.cfg.userCsr.zipWithIndex.foreach { case (csr, i) => csr := curCfg.extCsr(i) }
  } else {
    extHost.io.cfg := DontCare
  }

  // ------------------------------------------------------------------ busy
  // Copied deliberately from XDMADataPath.scala:149. A reader is not done when its AGU is done -- data is still
  // in flight in the buffer, and the extension chain may still be draining. Getting this wrong is the failure
  // mode that hid the streaming-FSM hang: the task retires, the next one starts on top of it.
  val readerBusy = reader.io.busy | (~reader.io.bufferEmpty) | extHost.io.busy
  val writerBusy = writer.io.busy | (~writer.io.bufferEmpty)
  val busy       = readerBusy | writerBusy
  io.status.busy := busy

  // ------------------------------------------------------------------ task FSM
  // sIdle -> sWaitBusy -> sBusy -> sIdle, the same shape as the xDMA's src/dst FSMs (XDMACtrl.scala:533).
  // sWaitBusy exists because `busy` does not rise on the start pulse itself: the AGU needs a cycle. Retiring on
  // `!busy` straight out of sIdle would retire every task instantly.
  val sIdle :: sWaitBusy :: sBusy :: Nil = Enum(3)
  sIdle.suggestName("sIdle")
  sWaitBusy.suggestName("sWaitBusy")
  sBusy.suggestName("sBusy")

  val nextState    = Wire(chiselTypeOf(sIdle))
  val currentState = RegNext(nextState, sIdle)
  dontTouch(currentState)

  // A task whose temporal bounds are degenerate never makes its AGU busy, so sWaitBusy would never be left. Bound
  // the wait and report it in the status CSR instead of hanging: a visible bad-config bit is debuggable, a hung
  // engine is not. `stuckLimit` is far above any real start latency (the AGU asserts busy within a cycle or two).
  val stuckLimit   = 32
  val stuckCounter = RegInit(0.U(log2Ceil(stuckLimit + 1).W))
  val stuckSticky  = RegInit(false.B)

  val startPulse = WireDefault(false.B)
  taskQueue.io.deq.ready := false.B
  nextState              := currentState

  switch(currentState) {
    is(sIdle) {
      when(taskQueue.io.deq.valid) {
        startPulse := true.B
        nextState  := sWaitBusy
      }
    }
    is(sWaitBusy) {
      when(busy) {
        nextState := sBusy
      }.elsewhen(stuckCounter === stuckLimit.U) {
        // The engine never started: drop the task so the queue drains and software's finish counter still moves.
        nextState              := sIdle
        taskQueue.io.deq.ready := true.B
      }
    }
    is(sBusy) {
      when(~busy) {
        nextState              := sIdle
        taskQueue.io.deq.ready := true.B
      }
    }
  }

  when(currentState =/= sWaitBusy) {
    stuckCounter := 0.U
  }.elsewhen(stuckCounter =/= stuckLimit.U) {
    stuckCounter := stuckCounter + 1.U
  }
  when(currentState === sWaitBusy && stuckCounter === stuckLimit.U && !busy) { stuckSticky := true.B }
  when(startPulse) { stuckSticky := false.B }

  reader.io.start  := startPulse
  writer.io.start  := startPulse
  extHost.io.start := startPulse

  // ------------------------------------------------------------------ read-only CSRs
  // 0: tasks accepted, 1: tasks finished. Software takes the task id from (0) after a start and polls (1) --
  // the same contract as snax_xdma_start / snax_xdma_local_wait.
  private def counter(tag: String, tick: Bool, reset: Bool = false.B): UInt = {
    // `tag`, not `name`: inside the anonymous Module subclass `name` binds to BaseModule.name, so a parameter
    // called `name` would silently name every counter after the module itself.
    val c = Module(new BasicCounter(32, hasCeil = false) {
      override val desiredName = s"${namePrefix}_$tag"
    })
    c.io.ceil  := DontCare
    c.io.reset := reset
    c.io.tick  := tick
    c.io.value
  }

  csrManager.io.readOnlyReg(0) := counter("submittedTaskCounter", cfgIn.fire)
  csrManager.io.readOnlyReg(1) := counter("finishedTaskCounter", taskQueue.io.deq.fire)

  // Cycle counters for the last task / last reader burst / last writer burst, mirroring the xDMA's
  // XDMA_PERF_CTR_* so the app-side profiling code ports unchanged.
  private val pcIdle :: pcRunning :: Nil = Enum(2)

  private def perfCounter(tag: String, run: Bool, startCond: Bool): UInt = {
    val c = Module(new BasicCounter(width = 32, hasCeil = false) {
      override val desiredName = s"${namePrefix}_$tag"
    })
    val state     = RegInit(pcIdle)
    val nextPcState = WireDefault(state)
    state      := nextPcState
    c.io.ceil  := DontCare
    c.io.reset := false.B
    c.io.tick  := false.B
    switch(state) {
      is(pcIdle) {
        when(startCond) {
          c.io.reset  := true.B
          nextPcState := pcRunning
        }
      }
      is(pcRunning) {
        c.io.tick := true.B
        when(~run) {
          c.io.tick   := false.B
          nextPcState := pcIdle
        }
      }
    }
    c.io.value
  }

  csrManager.io.readOnlyReg(2) := perfCounter("perfCounterTask", busy, startPulse)
  csrManager.io.readOnlyReg(3) := perfCounter("perfCounterReader", readerBusy, readerBusy)
  csrManager.io.readOnlyReg(4) := perfCounter("perfCounterWriter", writerBusy, writerBusy)

  // [0] busy now, [1] a task never started (sticky since the last start pulse), [2] the task queue is full.
  csrManager.io.readOnlyReg(5) := Cat(0.U(29.W), ~taskQueue.io.enq.ready, stuckSticky, busy)
}
