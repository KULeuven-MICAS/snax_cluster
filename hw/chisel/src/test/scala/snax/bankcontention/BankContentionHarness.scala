package snax.bankcontention

import chisel3._
import chisel3.util._

import snax.readerWriter._

/** A stream: where it starts and how its address generator walks. Mirrors the streamer CSRs a
  * kernel writes (`ptr`, `Xtlbound*`, `Xtlstride*`).
  */
case class StreamSpec(ptr: Int, temporalBounds: Seq[Int], temporalStrides: Seq[Int]) {
  def nSteps: Int = temporalBounds.product
}

/** Several real `Reader`s competing for one bank array, with an engine draining them in step.
  *
  * WHAT IS RTL HERE AND WHAT IS NOT. The readers are the shipped modules -- address generator,
  * `DataRequestor` (which HOLDS an ungranted address until it fires), `DataResponser` with its
  * in-flight counter, and the `ComplexQueue` whose fill level drives the TCDM priority bit.
  * That is the whole mechanism the Python model claims to reproduce, and none of it is
  * re-implemented here.
  *
  * The bank array below IS testbench code, deliberately. The split cluster sets
  * `tcdm.sparse_interconnect: false`, so `SparseInterconnect` is not in its datapath; what
  * arbitrates there is the cluster's dense TCDM crossbar. The policy is small and fully
  * specified -- one grant per bank per cycle, highest asserted priority first, round-robin
  * among equals (`PriorityRoundRobinArbiter`) -- so it is written out here rather than pulled
  * in, and the same policy is what the Python arbiter implements. The thing under test is
  * everything upstream of it.
  *
  * The engine consumes ONE step from every reader per cycle, which is what an output-stationary
  * array does with its A and B operands: it advances only when all of them have data.
  */
class BankContentionHarness(
  specs:       Seq[StreamSpec],
  numBanks:    Int,
  numChannel:  Int     = 8,
  bufferDepth: Int     = 8,
  tcdmSizeKiB: Int     = 1024,
  dataWidth:   Int     = 64
) extends Module
    with RequireAsyncReset {           // `Reader` demands it, so the harness must match

  require(specs.nonEmpty, "need at least one stream")
  require(specs.map(_.temporalBounds.length).distinct.length == 1,
          "all streams must declare the same temporal dimension")
  require(specs.map(_.nSteps).distinct.length == 1,
          "this harness drives the streams in step, so they must be the same length")

  val nStreams  = specs.length
  val nSteps    = specs.head.nSteps
  val byteOffW  = log2Ceil(dataWidth / 8)
  val bankSelW  = log2Ceil(numBanks)

  val io = IO(new Bundle {
    val start  = Input(Bool())
    val done   = Output(Bool())
    val cycles = Output(UInt(32.W))
    val steps  = Output(UInt(32.W))
    /** Cycles the engine was blocked AND this stream had no data -- the "all" attribution, so
      * the figures are unambiguous rather than depending on a tie-break. */
    val stall  = Output(Vec(nStreams, UInt(32.W)))
    val blocked = Output(UInt(32.W))
    /** Requests that asked for a bank and did not get it. */
    val retries = Output(Vec(nStreams, UInt(32.W)))
  })

  val param = new ReaderWriterParam(
    spatialBounds     = List(numChannel),
    temporalDimension = specs.head.temporalBounds.length,
    tcdmDataWidth     = dataWidth,
    tcdmSize          = tcdmSizeKiB,
    numChannel        = numChannel,
    dataBufferDepth   = bufferDepth,
    dynamicPriority   = true
  )

  val readers = specs.map(_ => Module(new Reader(param, moduleNamePrefix = "bc")))

  // ---- configure each reader's AGU from its spec, and hold it ---------------------------
  readers.zip(specs).foreach { case (rd, spec) =>
    rd.io.aguCfg.ptr := spec.ptr.U
    rd.io.aguCfg.spatialStrides(0) := (dataWidth / 8).U        // channels are contiguous words
    spec.temporalBounds.zipWithIndex.foreach { case (b, i) => rd.io.aguCfg.temporalBounds(i) := b.U }
    spec.temporalStrides.zipWithIndex.foreach { case (s, i) => rd.io.aguCfg.temporalStrides(i) := s.U }
    rd.io.aguCfg.addressRemapIndex := 0.U
    rd.io.readerwriterCfg.enabledChannel := ~0.U(numChannel.W)
    rd.io.readerwriterCfg.enabledByte    := ~0.U((dataWidth / 8).W)
    rd.io.start := io.start
  }

  // ---- the bank array: one grant per bank, priority first, round-robin on ties -----------
  val nReq   = nStreams * numChannel
  def flat(s: Int, c: Int): Int = s * numChannel + c

  val reqValid = Wire(Vec(nReq, Bool()))
  val reqBank  = Wire(Vec(nReq, UInt(bankSelW.W)))
  val reqPrio  = Wire(Vec(nReq, Bool()))
  val granted  = Wire(Vec(nReq, Bool()))

  readers.zipWithIndex.foreach { case (rd, s) =>
    for (c <- 0 until numChannel) {
      val i = flat(s, c)
      reqValid(i) := rd.io.tcdmReq(c).valid
      reqBank(i)  := rd.io.tcdmReq(c).bits.addr(bankSelW + byteOffW - 1, byteOffW)
      reqPrio(i)  := rd.io.tcdmReq(c).bits.priority
      rd.io.tcdmReq(c).ready := granted(i)
      // one-cycle bank latency, exactly as the cluster's crossbar assumes
      rd.io.tcdmRsp(c).valid     := RegNext(granted(i) && reqValid(i), false.B)
      rd.io.tcdmRsp(c).bits.data := 0.U
    }
  }

  granted.foreach(_ := false.B)
  val rr = RegInit(VecInit(Seq.fill(numBanks)(0.U(log2Ceil(nReq).W))))

  for (b <- 0 until numBanks) {
    val asking   = VecInit((0 until nReq).map(i => reqValid(i) && reqBank(i) === b.U))
    val urgent   = VecInit((0 until nReq).map(i => asking(i) && reqPrio(i)))
    val anyUrgent = urgent.reduce(_ || _)
    // highest priority masks the rest, then round-robin among the survivors
    val eligible = VecInit((0 until nReq).map(i => Mux(anyUrgent, urgent(i), asking(i))))
    val rotated  = VecInit((0 until nReq).map { k =>
      val idx = Wire(UInt(log2Ceil(nReq).W))       // narrow: `+&` widens, the modulo does not
      idx := (rr(b) +& k.U) % nReq.U
      eligible(idx)
    })
    val hit      = rotated.reduce(_ || _)
    val firstK   = PriorityEncoder(rotated)
    // `+&` widens, and the modulo keeps the value in range but not the WIDTH -- indexing a Vec
    // with a too-wide signal is a silently dropped write in Chisel, so narrow it explicitly.
    val winner   = Wire(UInt(log2Ceil(nReq).W))
    winner := (rr(b) +& firstK) % nReq.U
    when(hit) {
      granted(winner) := true.B
      rr(b) := (winner +& 1.U) % nReq.U
    }
  }

  // ---- the engine: one step from every reader per cycle ----------------------------------
  val allReady = readers.map(_.io.data.valid).reduce(_ && _)
  readers.foreach(_.io.data.ready := allReady)

  val cycles  = RegInit(0.U(32.W))
  val steps   = RegInit(0.U(32.W))
  val blocked = RegInit(0.U(32.W))
  val stalls  = RegInit(VecInit(Seq.fill(nStreams)(0.U(32.W))))
  val retries = RegInit(VecInit(Seq.fill(nStreams)(0.U(32.W))))
  val running = RegInit(false.B)

  when(io.start) { running := true.B }
  when(steps === nSteps.U) { running := false.B }

  when(running) {
    cycles := cycles + 1.U
    when(allReady) {
      steps := steps + 1.U
    }.otherwise {
      blocked := blocked + 1.U
      readers.zipWithIndex.foreach { case (rd, s) =>
        when(!rd.io.data.valid) { stalls(s) := stalls(s) + 1.U }
      }
    }
    readers.zipWithIndex.foreach { case (_, s) =>
      val missed = (0 until numChannel).map(c => reqValid(flat(s, c)) && !granted(flat(s, c)))
      retries(s) := retries(s) + PopCount(VecInit(missed))
    }
  }

  io.done    := steps === nSteps.U
  io.cycles  := cycles
  io.steps   := steps
  io.blocked := blocked
  io.stall   := stalls
  io.retries := retries
}
