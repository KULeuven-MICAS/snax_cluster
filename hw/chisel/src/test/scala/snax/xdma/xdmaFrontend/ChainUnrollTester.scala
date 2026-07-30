package snax.xdma.xdmaFrontend

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathJunction.HasMonoidJunction
import snax.readerWriter.ReaderWriterParam
import snax.xdma.DesignParams._
import snax.xdma.xdmaIO._

/** Harness for the CFG PLANE alone: an `XDMACtrl` with a deserializer on its outgoing cfg port, so a test can
  * submit one task over CSRs and read back the exact sequence of frames the unroll dispatched.
  *
  * The data plane is left idle. What is under test is orchestration: how many frames a task emits, which side
  * (reader / writer) each configures, and what position each one stamps.
  */
class ChainUnrollHarness(readerParam: XDMAParam, writerParam: XDMAParam) extends Module with RequireAsyncReset {
  val ctrl = Module(new XDMACtrl(readerParam, writerParam, "chain"))
  val des  = Module(new XDMAInterClusterCfgIODeserializer(writerParam))

  val io = IO(new Bundle {
    val csrIO              = chiselTypeOf(ctrl.io.csrIO)
    val clusterBaseAddress = Input(UInt(writerParam.axiParam.addrWidth.W))
    val frame              = Decoupled(new XDMAInterClusterCfgIO(readerParam, writerParam))
  })

  io.csrIO <> ctrl.io.csrIO
  ctrl.io.clusterBaseAddress := io.clusterBaseAddress
  ctrl.io.junctionStarved := false.B
  ctrl.io.junctionCfgErr  := false.B
  des.io.cfgIn <> ctrl.io.remoteXDMACfg.toRemote
  io.frame <> des.io.cfgOut

  // no inbound cfg, and the local datapath never goes busy: a submitted task retires immediately, so the whole
  // unroll is dispatched without needing a data plane
  ctrl.io.remoteXDMACfg.fromRemote.valid := false.B
  ctrl.io.remoteXDMACfg.fromRemote.bits  := 0.U
  ctrl.io.localXDMACfg.readerBusy        := false.B
  ctrl.io.localXDMACfg.writerBusy        := false.B
  ctrl.io.remoteTaskFinished             := false.B
}

/** The gather unroll, checked frame by frame.
  *
  * A frame crossing the wire carries `isWriterSide` and the receiver demuxes on it, so one frame configures a
  * reader OR a writer, never both. A ChainWrite middle only ever writes, so one frame per hop suffices. A
  * ChainGather hop must both read its own partial and fold what arrives, so it needs two -- and the head needs
  * only the reader one, because nothing arrives at a head.
  *
  * The two tests are the same task shape with the junction off / on, which is exactly the difference between a
  * chained write and a gather. The ChainWrite case doubles as the regression that the unroll still emits one
  * frame per hop with the positions it always did.
  */
class ChainUnrollTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  private val CLUSTER = BigInt("10000000", 16)
  private val STEP    = BigInt(1) << 22 // one cluster tag step (tcdmSize 4096 KiB -> tag above bit 21)
  private val S1      = CLUSTER + STEP
  private val S2      = CLUSTER + 2 * STEP
  private val LOCAL_DST = CLUSTER + 0x1000
  private val LOCAL_SRC = CLUSTER

  private def readerParam = new XDMAParam(
    cfgParam          = new XDMAConfigParam(addrWidth = 32, dataWidth = 32),
    axiParam          = new XDMAAXIParam,
    crossClusterParam = new XDMACrossClusterParam,
    rwParam           = new ReaderWriterParam(configurableByteMask = false, configurableChannel = true)
  )
  private def writerParam = new XDMAParam(
    cfgParam          = new XDMAConfigParam(addrWidth = 32, dataWidth = 32),
    axiParam          = new XDMAAXIParam,
    crossClusterParam = new XDMACrossClusterParam,
    rwParam           = new ReaderWriterParam(configurableByteMask = true, configurableChannel = true),
    extParam          = Seq(),
    junctionParam     = Seq(new HasMonoidJunction(dataWidth = 512))
  )

  /** One decoded frame as the receiver would see it. */
  private case class Frame(writerSide: Boolean, readerPtr: BigInt, wp0: BigInt, wp1: BigInt,
                           chainRole: Int, collective: Boolean) {
    private def nm(a: BigInt) =
      if (a == S1) "S1" else if (a == S2) "S2" else if (a == LOCAL_DST) "DST" else if (a == LOCAL_SRC) "SRC"
      else if (a == 0) "0" else f"0x$a%x"
    def role = chainRole match { case 0 => "HEAD"; case 1 => "MIDDLE"; case 2 => "TAIL"; case x => s"?$x" }
    override def toString =
      f"${if (writerSide) "WRITER" else "READER"}%-6s rd=${nm(readerPtr)}%-5s wp0=${nm(wp0)}%-5s " +
        f"wp1=${nm(wp1)}%-5s $role%-6s collective=$collective"
  }

  /** Submit one task over the CSR port and collect every frame the unroll puts on the wire. */
  private def runTask(dut: ChainUnrollHarness, writerPtrs: Seq[BigInt], junctionEnable: Int,
                      maxCyc: Int = 4000): Seq[Frame] = {
    dut.io.clusterBaseAddress.poke(CLUSTER.U)
    dut.io.frame.ready.poke(true)

    var addr = 0
    def writeCsr(data: BigInt): Unit = {
      dut.io.csrIO.req.bits.write.poke(true.B)
      dut.io.csrIO.req.bits.strb.poke("b1111".U)
      dut.io.csrIO.req.bits.data.poke((data & BigInt("ffffffff", 16)).U)
      dut.io.csrIO.req.bits.addr.poke(addr.U)
      dut.io.csrIO.req.valid.poke(true.B)
      var w = 0
      while (!dut.io.csrIO.req.ready.peekBoolean() && w < 200) { dut.clock.step(1); w += 1 }
      dut.clock.step(1)
      dut.io.csrIO.req.valid.poke(false.B)
      addr += 1
    }
    def writePtr(a: BigInt): Unit = { writeCsr(a & BigInt("ffffffff", 16)); writeCsr(a >> 32) }

    writePtr(LOCAL_SRC)                                   // readerPtr: this node's own partial
    writerPtrs.foreach(writePtr)                          // writerPtr[]: the chain, in data order
    for (_ <- writerPtrs.length until 4) writePtr(0)      // pad the unused multicast slots
    // reader AGU: one beat
    writeCsr(8); writeCsr(1); for (_ <- 0 until 4) writeCsr(1); writeCsr(64); for (_ <- 0 until 4) writeCsr(0)
    writeCsr(0xff)                                        // reader enabledChannel
    // writer AGU: one beat
    writeCsr(8); writeCsr(1); for (_ <- 0 until 4) writeCsr(1); writeCsr(64); for (_ <- 0 until 4) writeCsr(0)
    writeCsr(0xff); writeCsr(0xff)                        // writer enabledChannel + enabledByte
    writeCsr(junctionEnable)                              // junction enable bitmask -> this is what makes it a gather
    writeCsr(0)                                           // junction user CSR (the geometry word)
    writeCsr(1)                                           // start

    val out = collection.mutable.ArrayBuffer[Frame]()
    var idle = 0
    var cyc  = 0
    while (cyc < maxCyc && idle < 300) {
      if (dut.io.frame.valid.peekBoolean()) {
        out += Frame(
          writerSide = dut.io.frame.bits.isWriterSide.peekBoolean(),
          readerPtr  = dut.io.frame.bits.readerPtr.peekInt(),
          wp0        = dut.io.frame.bits.writerPtr(0).peekInt(),
          wp1        = dut.io.frame.bits.writerPtr(1).peekInt(),
          chainRole  = dut.io.frame.bits.chainRole.peekInt().toInt,
          collective = dut.io.frame.bits.collectiveMode.peekBoolean()
        )
        idle = 0
      } else idle += 1
      dut.clock.step(1); cyc += 1
    }
    out.toSeq
  }

  "ChainUnroll_ChainWrite" should "emit one writer-side frame per hop, with the positions it always did" in {
    test(new ChainUnrollHarness(readerParam, writerParam))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        // multicast/chained write: this node's buffer -> S1 -> S2 -> local dst
        val frames = runTask(dut, Seq(S1, S2, LOCAL_DST), junctionEnable = 0)
        frames.zipWithIndex.foreach { case (f, i) => println(f"[ChainWrite] frame $i: $f") }

        assert(frames.forall(_.writerSide), "a chained write configures writers only -- no reader companions")
        assert(frames.forall(!_.collective), "collectiveMode must stay clear when no junction is armed")
        assert(frames.length == 2, s"expected one frame per REMOTE hop (S1, S2), got ${frames.length}")
        assert(frames(0).wp0 == S1 && frames(0).wp1 == S2, s"frame 0 addresses: ${frames(0)}")
        assert(frames(0).chainRole == XDMAChainRole.MIDDLE, s"S1 forwards, so MIDDLE: ${frames(0)}")
        assert(frames(1).wp0 == S2 && frames(1).wp1 == LOCAL_DST, s"frame 1 addresses: ${frames(1)}")
        assert(frames(1).chainRole == XDMAChainRole.MIDDLE, s"S2 forwards, so MIDDLE: ${frames(1)}")
        println("[ChainWrite] one writer-side frame per remote hop, roles unchanged")
      }
  }

  "ChainUnroll_ChainGather" should "emit a reader companion per hop and no writer frame for the head" in {
    test(new ChainUnrollHarness(readerParam, writerParam))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        // gather: S1 -> S2 -> this node's local dst, folding at every hop
        val frames = runTask(dut, Seq(S1, S2, LOCAL_DST), junctionEnable = 1)
        frames.zipWithIndex.foreach { case (f, i) => println(f"[ChainGather] frame $i: $f") }

        val readers = frames.filter(!_.writerSide)
        val writers = frames.filter(_.writerSide)

        assert(frames.forall(_.collective), "every gather frame must carry collectiveMode")

        // Each remote hop is told to read its OWN partial: the frame is routed by readerPtr, so readerPtr names
        // the hop itself.
        assert(readers.length == 2, s"expected a reader companion for S1 and S2, got ${readers.length}")
        assert(readers(0).readerPtr == S1 && readers(0).wp0 == S2, s"S1 reads itself, sends to S2: ${readers(0)}")
        assert(readers(1).readerPtr == S2 && readers(1).wp0 == LOCAL_DST, s"S2 reads itself: ${readers(1)}")

        // THE HEAD IS THE POINT: S1 sources the chain, so it gets a reader frame stamped HEAD and NO writer
        // frame. A writer frame would arm its junction on a stream that never arrives and starve the join.
        assert(readers(0).chainRole == XDMAChainRole.HEAD, s"S1 is the chain head: ${readers(0)}")
        assert(!writers.exists(_.wp0 == S1), "the head must NOT be given a writer-side frame")

        // Only the middle gets a writer frame: that is what arms its junction and its forward.
        assert(writers.length == 1, s"expected exactly one writer-side frame (S2), got ${writers.length}")
        assert(writers(0).wp0 == S2 && writers(0).wp1 == LOCAL_DST, s"S2 folds and forwards: ${writers(0)}")
        assert(writers(0).chainRole == XDMAChainRole.MIDDLE, s"S2 forwards, so MIDDLE: ${writers(0)}")

        // The tail is this node itself, configured locally -- so it never appears on the wire.
        assert(!frames.exists(f => f.wp0 == LOCAL_DST && f.writerSide),
               "the collector configures its own writer locally; no frame for it crosses the wire")
        println("[ChainGather] head: reader-only + HEAD; middle: reader + writer; tail: local")
      }
  }
}
