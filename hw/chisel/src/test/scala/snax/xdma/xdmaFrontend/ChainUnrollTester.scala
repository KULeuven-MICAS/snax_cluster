package snax.xdma.xdmaFrontend

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import chiseltest.WriteVcdAnnotation
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
  // The inbound half: a test can push a frame back IN, through the real serializer, so a captured frame can be
  // REPLAYED at the node it was addressed to. That closes the loop -- what the unroll emits is fed to a receiver
  // and the cfg it presents to its own datapath is observable.
  val ser = Module(new XDMAInterClusterCfgIOSerializer(writerParam))

  val io = IO(new Bundle {
    val csrIO              = chiselTypeOf(ctrl.io.csrIO)
    val clusterBaseAddress = Input(UInt(writerParam.axiParam.addrWidth.W))
    val frame              = Decoupled(new XDMAInterClusterCfgIO(readerParam, writerParam))
    // inbound: a frame arriving from another cluster
    val inFrame            = Flipped(Decoupled(chiselTypeOf(ser.io.cfgIn.bits)))
    // what this node hands its own datapath
    val writerCfg          = Output(chiselTypeOf(ctrl.io.localXDMACfg.writerCfg))
    val readerCfg          = Output(chiselTypeOf(ctrl.io.localXDMACfg.readerCfg))
    val writerStart        = Output(Bool())
    val readerStart        = Output(Bool())
    // driveable busy levels, so a task can be made to retire and the node re-armed for the next one
    val readerBusy         = Input(Bool())
    val writerBusy         = Input(Bool())
  })

  io.csrIO <> ctrl.io.csrIO
  ctrl.io.clusterBaseAddress := io.clusterBaseAddress
  ctrl.io.junctionStarved := false.B
  ctrl.io.junctionCfgErr  := false.B
  des.io.cfgIn <> ctrl.io.remoteXDMACfg.toRemote
  io.frame <> des.io.cfgOut

  ser.io.cfgIn <> io.inFrame
  ctrl.io.remoteXDMACfg.fromRemote <> ser.io.cfgOut

  ctrl.io.localXDMACfg.readerBusy := io.readerBusy
  ctrl.io.localXDMACfg.writerBusy := io.writerBusy
  ctrl.io.remoteTaskFinished      := false.B

  io.writerCfg   := ctrl.io.localXDMACfg.writerCfg
  io.readerCfg   := ctrl.io.localXDMACfg.readerCfg
  io.writerStart := ctrl.io.localXDMACfg.writerStart
  io.readerStart := ctrl.io.localXDMACfg.readerStart
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

  /** Every field of one frame, enough to REPLAY it into a receiving node. */
  private case class RawFrame(writerSide: Boolean, taskID: BigInt, readerPtr: BigInt, wp: Seq[BigInt],
                              beats: BigInt, spatialStride: BigInt, bounds: Seq[BigInt], strides: Seq[BigInt],
                              enabledChannel: BigInt, enabledByte: BigInt, chainRole: Int,
                              collective: Boolean, writerExtCfg: Seq[BigInt]) {
    def toFrame = Frame(writerSide, readerPtr, wp(0), wp(1), chainRole, collective)
  }

  private def capture(dut: ChainUnrollHarness): RawFrame = {
    val b = dut.io.frame.bits
    RawFrame(
      writerSide     = b.isWriterSide.peekBoolean(),
      taskID         = b.taskID.peekInt(),
      readerPtr      = b.readerPtr.peekInt(),
      wp             = b.writerPtr.map(_.peekInt()).toSeq,
      beats          = b.axiTransferBeatSize.peekInt(),
      spatialStride  = b.spatialStride.peekInt(),
      bounds         = b.temporalBounds.map(_.peekInt()).toSeq,
      strides        = b.temporalStrides.map(_.peekInt()).toSeq,
      enabledChannel = b.enabledChannel.peekInt(),
      enabledByte    = b.enabledByte.peekInt(),
      chainRole      = b.chainRole.peekInt().toInt,
      collective     = b.collectiveMode.peekBoolean(),
      writerExtCfg   = b.writerExtCfg.map(_.peekInt()).toSeq
    )
  }

  /** Drive a captured frame back in through the real serializer, as if it arrived from another cluster. */
  private def replay(dut: ChainUnrollHarness, f: RawFrame, tick: () => Unit, maxCyc: Int = 400): Unit = {
    val b = dut.io.inFrame.bits
    b.isWriterSide.poke(f.writerSide.B)
    b.taskID.poke(f.taskID.U)
    b.readerPtr.poke(f.readerPtr.U)
    b.writerPtr.zip(f.wp).foreach { case (p, v) => p.poke(v.U) }
    b.axiTransferBeatSize.poke(f.beats.U)
    b.spatialStride.poke(f.spatialStride.U)
    b.temporalBounds.zip(f.bounds).foreach { case (p, v) => p.poke(v.U) }
    b.temporalStrides.zip(f.strides).foreach { case (p, v) => p.poke(v.U) }
    b.enabledChannel.poke(f.enabledChannel.U)
    b.enabledByte.poke(f.enabledByte.U)
    b.chainRole.poke(f.chainRole.U)
    b.collectiveMode.poke(f.collective.B)
    b.writerExtCfg.zip(f.writerExtCfg).foreach { case (p, v) => p.poke(v.U) }
    dut.io.inFrame.valid.poke(true.B)
    var c = 0
    while (c < maxCyc && !dut.io.inFrame.ready.peekBoolean()) { tick(); c += 1 }
    tick()
    dut.io.inFrame.valid.poke(false.B)
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

  /** Submit a task while MODELLING a datapath that actually goes busy, so the local start FSMs retire and the
    * node can be re-armed. `tick` is the caller's clock step, which also drives the busy levels.
    */
  private def runTaskModelled(dut: ChainUnrollHarness, writerPtrs: Seq[BigInt], junctionEnable: Int,
                              tick: () => Unit, maxCyc: Int = 4000): Seq[RawFrame] = {
    dut.io.frame.ready.poke(true)

    var addr = 0
    def writeCsr(data: BigInt): Unit = {
      dut.io.csrIO.req.bits.write.poke(true.B)
      dut.io.csrIO.req.bits.strb.poke("b1111".U)
      dut.io.csrIO.req.bits.data.poke((data & BigInt("ffffffff", 16)).U)
      dut.io.csrIO.req.bits.addr.poke(addr.U)
      dut.io.csrIO.req.valid.poke(true.B)
      var w = 0
      while (!dut.io.csrIO.req.ready.peekBoolean() && w < 200) { tick(); w += 1 }
      tick()
      dut.io.csrIO.req.valid.poke(false.B)
      addr += 1
    }
    def writePtr(a: BigInt): Unit = { writeCsr(a & BigInt("ffffffff", 16)); writeCsr(a >> 32) }

    writePtr(LOCAL_SRC)
    writerPtrs.foreach(writePtr)
    for (_ <- writerPtrs.length until 4) writePtr(0)
    writeCsr(8); writeCsr(1); for (_ <- 0 until 4) writeCsr(1); writeCsr(64); for (_ <- 0 until 4) writeCsr(0)
    writeCsr(0xff)
    writeCsr(8); writeCsr(1); for (_ <- 0 until 4) writeCsr(1); writeCsr(64); for (_ <- 0 until 4) writeCsr(0)
    writeCsr(0xff); writeCsr(0xff)
    writeCsr(junctionEnable)
    writeCsr(0)
    writeCsr(1)

    val out = collection.mutable.ArrayBuffer[RawFrame]()
    var idle = 0
    var cyc  = 0
    while (cyc < maxCyc && idle < 300) {
      if (dut.io.frame.valid.peekBoolean()) { out += capture(dut); idle = 0 } else idle += 1
      tick(); cyc += 1
    }
    out.toSeq
  }

  /** WHAT DOES A REAL GATHER MIDDLE HOP ACTUALLY RECEIVE?
    *
    * Every other test here watches one side of the wire. This one closes the loop: it unrolls a gather at the
    * initiator, captures the frames, then RE-ADDRESSES the same `XDMACtrl` as the middle hop S2 and replays the
    * frames that were addressed to S2 back into it through the real serializer / deserializer. What comes out is
    * the cfg that node hands its own datapath -- `io.writerCfg`, sampled on the cycle `writerStart` pulses.
    *
    * That is the measurement the bug writeup asked for and could not get without a 20-minute HeMAiA build:
    * whether `collectiveMode && remoteLoopback` (the predicate of the rejected fix attempt) is even true at a
    * real middle hop, and which fields ARE trustworthy for identifying one.
    */
  "ChainUnroll_MiddleHopWriterCfg" should "hand a middle hop a writer cfg that identifies it as a gather middle" in {
    test(new ChainUnrollHarness(readerParam, writerParam))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>

      // a datapath model: any start pulse makes the corresponding side busy for a few cycles, so the ctrl's
      // start FSMs run their full sIdle -> sWaitBusy -> sBusy -> sIdle cycle and pop the cfg
      var wBusy = 0
      var rBusy = 0
      var sawWriterStart = false
      var writerCfgAtStart: Option[(Boolean, Boolean, Boolean, BigInt, BigInt, BigInt, Int, Boolean)] = None
      val tick: () => Unit = () => {
        if (dut.io.writerStart.peekBoolean()) {
          wBusy = 8
          if (!sawWriterStart) {
            sawWriterStart = true
            writerCfgAtStart = Some((
              dut.io.writerCfg.collectiveMode.peekBoolean(),
              dut.io.writerCfg.remoteLoopback.peekBoolean(),
              dut.io.writerCfg.localLoopback.peekBoolean(),
              dut.io.writerCfg.writerPtr(0).peekInt(),
              dut.io.writerCfg.writerPtr(1).peekInt(),
              dut.io.writerCfg.extCfg(0).peekInt(),
              dut.io.writerCfg.chainRole.peekInt().toInt,
              dut.io.writerCfg.origination.peekBoolean()
            ))
          }
        }
        if (dut.io.readerStart.peekBoolean()) rBusy = 8
        dut.io.writerBusy.poke((wBusy > 0).B)
        dut.io.readerBusy.poke((rBusy > 0).B)
        dut.clock.step(1)
        if (wBusy > 0) wBusy -= 1
        if (rBusy > 0) rBusy -= 1
      }

      // ---- phase 1: unroll the gather at the initiator ----
      dut.io.clusterBaseAddress.poke(CLUSTER.U)
      dut.io.inFrame.valid.poke(false.B)
      val frames = runTaskModelled(dut, Seq(S1, S2, LOCAL_DST), junctionEnable = 1, tick)
      frames.foreach(f => println("[MiddleHopCfg] emitted " + f.toFrame))

      val toMiddleWriter = frames.filter(f => f.writerSide && f.wp(0) == S2)
      val toMiddleReader = frames.filter(f => !f.writerSide && f.readerPtr == S2)
      assert(toMiddleWriter.length == 1, "expected exactly one writer-side frame addressed to the middle hop S2")
      assert(toMiddleReader.length == 1, "expected exactly one reader companion addressed to the middle hop S2")

      // ---- phase 2: BE the middle hop, and take delivery of those frames ----
      sawWriterStart = false
      dut.io.clusterBaseAddress.poke(S2.U)
      for (_ <- 0 until 40) tick()

      // the cfg plane sends the reader companion first (higher arbiter priority), then the writer frame
      replay(dut, toMiddleReader.head, tick)
      replay(dut, toMiddleWriter.head, tick)
      var c = 0
      while (c < 1500 && !sawWriterStart) { tick(); c += 1 }

      assert(sawWriterStart, "the middle hop never started its writer at all -- the replayed frame did not land")
      val (collective, remoteLb, localLb, wp0, wp1, ext0, role, origination) = writerCfgAtStart.get
      println(f"[MiddleHopCfg] writerCfg at writerStart: collectiveMode=$collective remoteLoopback=$remoteLb " +
              f"localLoopback=$localLb writerPtr0=0x$wp0%x writerPtr1=0x$wp1%x extCfg(0)=$ext0 " +
              f"chainRole=$role originationIsFromLocal=$origination")

      assert(wp0 == S2, f"the middle hop's writer cfg should point at itself, got 0x$wp0%x")
      assert(remoteLb, "a gather MIDDLE hop must have remoteLoopback set -- it forwards the fold to a next hop")
      assert(ext0 != 0,
             "the junction ENABLE word did not survive to the middle hop's writer cfg. Without it the switch " +
             "never derives a gather at all, so extCfg is not a usable middle-hop predicate.")
      assert(collective,
             "collectiveMode is FALSE on a real gather middle hop's writer cfg. That is exactly why the " +
             "rejected fix attempt (writer.io.start := io.writerStart && !(collectiveMode && remoteLoopback)) " +
             "was a no-op at middle hops and fired somewhere else instead.")
      println("[MiddleHopCfg] collectiveMode && remoteLoopback && junction-enable all hold at a real middle hop")
    }
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

  /** REGRESSION: submit the SAME gather task TWICE.
    *
    * Measured on HeMAiA: the first ChainGather through a chain with a middle node folds
    * correctly (P=4 linear, 551 cycles, byte-exact); every subsequent one retires in ~26
    * cycles having written nothing, with the junction reporting neither cfgerr nor starved.
    * A chain with no middle node (P=2) repeats fine. `XDMADataSwitch` handles back-to-back
    * gathers correctly in isolation, so the stuck state is in the CFG PLANE, not the switch.
    *
    * Every other test here submits exactly ONE task, which is why this was never caught.
    * The second call rewrites the full CSR set and starts again -- exactly what software does.
    */
  "ChainUnroll_ChainGather_TWICE" should "dispatch the same frames for a second identical task" in {
    test(new ChainUnrollHarness(readerParam, writerParam))
      .withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation, flags)) { dut =>
      val chain = Seq(S1, S2, LOCAL_DST) // head S1 -> middle S2 -> tail (here)

      val first = runTask(dut, chain, junctionEnable = 1)
      println("[Unroll/x2] round 1 emitted " + first.length + " frames:")
      first.foreach(f => println("    " + f))

      dut.clock.step(50) // let the task fully retire

      val second = runTask(dut, chain, junctionEnable = 1)
      println("[Unroll/x2] round 2 emitted " + second.length + " frames:")
      second.foreach(f => println("    " + f))

      assert(second.nonEmpty,
             "the SECOND gather task dispatched NO cfg frames at all -- the chain unroll does " +
             "not re-arm, so no hop is configured and the transfer retires having done nothing")
      assert(second.length == first.length,
             s"second task dispatched ${second.length} frames but the first dispatched ${first.length}")
      assert(second.map(_.toString) == first.map(_.toString),
             "the second task dispatched a DIFFERENT frame sequence:\n" +
             "  round 1:\n" + first.map("    " + _).mkString("\n") +
             "\n  round 2:\n" + second.map("    " + _).mkString("\n"))
      println("[Unroll/x2] both rounds dispatched identical frame sequences")
    }
  }
}
