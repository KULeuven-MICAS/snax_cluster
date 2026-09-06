package snax.xdma.xdmaFrontend

import chisel3._
import chiseltest._
import chiseltest.WriteVcdAnnotation
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathJunction.HasMonoidJunction
import snax.readerWriter.ReaderWriterParam
import snax.xdma.DesignParams._
import snax.xdma.xdmaIO.XDMAChainRole

/** ChainGather at the DATAPATH level: does a MIDDLE hop become re-usable after a fold?
  *
  * `XDMADataSwitchTester` covers the switch alone and `ChainUnrollTester` covers the cfg plane alone; both pass
  * back-to-back gathers. Neither instantiates the READER and WRITER engines, which is exactly where the defect
  * lives, so nothing in this repo caught it.
  *
  * ---- THE DEFECT ----
  * A gather MIDDLE hop is handed a WRITER frame, so `writer.io.start` fires and its address generator produces a
  * full local-write address stream. But the fold leaves through the data switch's `toRemote` port -- a middle hop
  * writes NOTHING to local TCDM, which `XDMADataSwitchTester` asserts explicitly. So `dataBuffer` is never filled,
  * the TCDM requestors never issue, and nothing drains the AGU's address buffer. `Writer.scala`:
  * {{{
  *   io.busy := addressgen.io.busy | (~addressgen.io.bufferEmpty)
  * }}}
  * leaves `writer.io.busy` asserted forever. That single stuck level is self-contained in this repo -- it depends
  * on no AXI response, no adapter handshake and nothing outside snax.
  *
  * It then jams the whole hop: `XDMADataSwitch` leaves `gActive` only on `~gatherWorking`, and
  * `gatherWorking = (readerBusy || writerBusyRaw) || junctionBusy`. With `writerBusyRaw` pinned the hop stays in
  * `gActive` with `isGather` asserted forever and can never be armed again.
  *
  * ---- MEASURED ON SILICON-EQUIVALENT RTL (HeMAiA, 2x2 and 4x4 chiplet grids) ----
  * The first gather through a chain WITH a middle hop folds correctly (P=4 linear, 505-551 cycles, byte-exact);
  * every subsequent one retires in ~26 cycles having moved nothing. A chain with NO middle hop (P=2) repeats
  * indefinitely. Waveform on the 2x2 chain `0x01 -> 0x11 -> 0x10 -> 0x00`:
  * {{{
  *   chip 0_0 (tail)   gActive @638us -> IDLE @644us -> re-arms @1258us   correct
  *   chip 1_1 (middle) gActive @638us -> never returns
  *   chip 1_0 (middle) gActive @640us -> never returns   (writerBusyRaw high 639us -> end of run)
  * }}}
  *
  * The test below is the unit-level statement of that: run ONE fold through a middle hop and require the hop to
  * go idle afterwards. It fails on the unfixed RTL at the `writerBusy` assertion.
  */
class XDMADataPathGatherTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  private val dataWidth = 512
  private val pairSlots = 8

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
    junctionParam     = Seq(new HasMonoidJunction(dataWidth = dataWidth))
  )

  private def f32(f: Double): BigInt = BigInt(java.lang.Float.floatToIntBits(f.toFloat).toLong & 0xffffffffL)
  private def packPairs(p: Seq[(Double, Double)]): BigInt = {
    var b = BigInt(0)
    for ((v, k) <- p.zipWithIndex) { b |= f32(v._1) << (32 * k); b |= f32(v._2) << (32 * (pairSlots + k)) }
    b
  }
  // MonoidJunction geometry word: [7:0] nValid | [11:8] n | [21:18] nExp | [27:26] sigma.
  // (m, l) with one twisted value coordinate, 8 partials per beat.
  private def csrMoment(nValid: Int): BigInt =
    (BigInt(3) << 26) | (BigInt(1) << 18) | (BigInt(1) << 8) | BigInt(nValid)

  // The three cluster tags a middle hop sees: the hop before it in the chain, itself, and its next hop. Only
  // their being distinct and non-zero matters here -- they make `writerPtr(1) =/= 0` true at a middle hop, so a
  // fix predicated on the pointers rather than on remoteLoopback is testable against this bench too.
  private val PREV_HOP = BigInt("10000000", 16)
  private val SELF_HOP = BigInt("10400000", 16)
  private val NEXT_HOP = BigInt("10800000", 16)

  private val localBeat  = packPairs(Seq.tabulate(pairSlots)(k => (1.0 + k, 2.0)))
  private val remoteBeat = packPairs(Seq.tabulate(pairSlots)(k => (3.0 + k, 0.5)))

  /** Arm one side's cfg for a single 64 B beat. `collective` + `middle` make it a gather MIDDLE hop:
    * `collectiveMode` says the payload is chain traffic and `remoteLoopback` says a next hop exists.
    */
  private def armCfg(dut: XDMADataPath, writerSide: Boolean, collective: Boolean, middle: Boolean,
                     junctionEnable: Int, junctionCsr: BigInt, beats: Int = 1): Unit = {
    val cfg = if (writerSide) dut.io.writerCfg else dut.io.readerCfg
    cfg.taskID.poke(1.U)
    // A hop in a chain is CONFIGURED REMOTELY: the initiator unrolls the chain and every hop receives its frames
    // over the link, so convertToXDMACfgIO stamps origination = fromRemote (XDMACfgIO.scala) and isInitiator
    // stays clear. Poking these the other way round would make the hop look like the node that issued the task.
    cfg.origination.poke(cfg.originationIsFromRemote.B)
    cfg.isInitiator.poke(false.B)
    // The pointers a real middle hop is given: its writer cfg points at ITSELF (that is what routed the frame
    // here) and carries its next hop in writerPtr(1) -- which is exactly what postRoute_dst_local turns into
    // remoteLoopback. The reader companion is told to read this hop and send on to the next.
    cfg.readerPtr.poke((if (writerSide) PREV_HOP else SELF_HOP).U)
    cfg.writerPtr.foreach(_.poke(0.U))
    cfg.writerPtr(0).poke((if (writerSide) SELF_HOP else NEXT_HOP).U)
    if (writerSide && middle) cfg.writerPtr(1).poke(NEXT_HOP.U)
    cfg.axiTransferBeatSize.poke(beats.U)
    cfg.localLoopback.poke(false.B)
    // Only the WRITER side ever carries remoteLoopback: XDMACtrl hard-wires postRoute_src_local.remoteLoopback
    // to false, so a reader cfg with it set is a state no real hop can be in.
    cfg.remoteLoopback.poke((writerSide && middle).B)
    cfg.chainRole.poke((if (middle) XDMAChainRole.MIDDLE else XDMAChainRole.TAIL).U)
    cfg.collectiveMode.poke(collective.B)

    // `beats` beats: the outermost temporal bound carries the count, the rest are 1. With a single-beat
    // transfer the AGU sweep is one vector, so `addressgen.busy` is never the half of `Writer.io.busy` that
    // stays stuck -- only `~bufferEmpty` is. A multi-beat transfer exercises both halves.
    cfg.aguCfg.ptr.poke(0.U)
    cfg.aguCfg.spatialStrides.zipWithIndex.foreach { case (s, i) => s.poke((if (i == 0) 8 else 0).U) }
    cfg.aguCfg.temporalBounds.zipWithIndex.foreach { case (b, i) => b.poke((if (i == 0) beats else 1).U) }
    cfg.aguCfg.temporalStrides.foreach(_.poke(0.U))
    cfg.aguCfg.addressRemapIndex.poke(0.U)

    cfg.readerwriterCfg.enabledChannel.poke(((BigInt(1) << 8) - 1).U)
    // enabledByte is tcdmParam.dataWidth/8 = 8 bits wide, not the full 64 B beat
    if (writerSide) cfg.readerwriterCfg.enabledByte.poke(((BigInt(1) << 8) - 1).U)

    // Junction bank layout: the ENABLE BITMASK FIRST, then the per-junction user CSRs
    // (DataPathJunctionHostIO.connectCfgWithList). It sits after the extension region, and
    // XDMACfgIO derives `junctionEnabled := extCfg(extCsrNum).orR` -- with no extensions
    // configured that is extCfg(0). Getting this order wrong arms no junction at all, so the
    // transfer is a plain chained write and the gather is never exercised.
    cfg.extCfg.foreach(_.poke(0.U))
    if (writerSide && cfg.extCfg.length >= 2) {
      cfg.extCfg(0).poke(junctionEnable.U) // junction enable bitmask
      cfg.extCfg(1).poke(junctionCsr.U)    // junction user CSR (the geometry word)
    }
  }

  /** Serve TCDM read responses so the local reader's own partial can flow, drain `toRemote`, and offer the
    * arriving partial on `fromRemote`. Returns the beat forwarded to the next hop, if any.
    */
  private def pump(dut: XDMADataPath, remote: Option[BigInt], maxCyc: Int): Option[BigInt] =
    pumpN(dut, remote, beats = 1, maxCyc = maxCyc).headOption

  /** The multi-beat form: offer `beats` copies of the arriving partial and collect every beat forwarded. */
  private def pumpN(dut: XDMADataPath, remote: Option[BigInt], beats: Int, maxCyc: Int): Seq[BigInt] = {
    val forwardedAll = collection.mutable.ArrayBuffer[BigInt]()
    var remoteLeft = if (remote.isEmpty) 0 else beats
    remote.foreach { b => dut.io.remoteXDMAData.fromRemote.bits.poke(b.U); dut.io.remoteXDMAData.fromRemote.valid.poke(true.B) }
    dut.io.remoteXDMAData.toRemote.ready.poke(true.B)
    dut.io.tcdmWriter.req.foreach(_.ready.poke(true.B))

    val nCh = dut.io.tcdmReader.req.length
    var cyc = 0
    while (cyc < maxCyc) {
      // model the TCDM: a read request fires -> a response one cycle later carrying the local partial's slice
      val fired = (0 until nCh).map { i =>
        dut.io.tcdmReader.req(i).ready.poke(true.B)
        dut.io.tcdmReader.req(i).valid.peekBoolean()
      }
      val frFire = remoteLeft > 0 && dut.io.remoteXDMAData.fromRemote.ready.peekBoolean()
      val trNow  = dut.io.remoteXDMAData.toRemote.valid.peekBoolean()
      val trBits = if (trNow) dut.io.remoteXDMAData.toRemote.bits.peekInt() else BigInt(0)

      dut.clock.step(1); cyc += 1

      (0 until nCh).foreach { i =>
        if (fired(i)) {
          dut.io.tcdmReader.rsp(i).valid.poke(true.B)
          dut.io.tcdmReader.rsp(i).bits.data.poke(((localBeat >> (64 * i)) & ((BigInt(1) << 64) - 1)).U)
        } else dut.io.tcdmReader.rsp(i).valid.poke(false.B)
      }
      if (frFire) {
        remoteLeft -= 1
        if (remoteLeft == 0) dut.io.remoteXDMAData.fromRemote.valid.poke(false.B)
      }
      if (trNow) forwardedAll += trBits
    }
    dut.io.tcdmReader.rsp.foreach(_.valid.poke(false.B))
    forwardedAll.toSeq
  }

  "XDMADataPath_GATHER_MIDDLE" should "return writerBusy to idle after a fold, so the hop can gather again" in {
    test(new XDMADataPath(readerParam, writerParam, "dp"))
      .withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation, flags)) { dut =>

      // quiescent
      dut.io.readerStart.poke(false.B)
      dut.io.writerStart.poke(false.B)
      dut.io.remoteXDMAData.fromRemote.valid.poke(false.B)
      dut.io.remoteXDMAData.toRemote.ready.poke(false.B)
      dut.io.tcdmReader.req.foreach(_.ready.poke(false.B))
      dut.io.tcdmReader.rsp.foreach(_.valid.poke(false.B))
      dut.io.tcdmWriter.req.foreach(_.ready.poke(false.B))
      armCfg(dut, writerSide = false, collective = true, middle = true, 0, 0)
      armCfg(dut, writerSide = true, collective = true, middle = true, 1, csrMoment(pairSlots))
      dut.clock.step(4)

      assert(!dut.io.writerBusy.peekBoolean(), "writerBusy asserted before the transfer even started")

      // start the hop exactly as the cfg plane does: reader companion frame and writer frame
      dut.io.readerStart.poke(true.B)
      dut.io.writerStart.poke(true.B)
      dut.clock.step(1)
      dut.io.readerStart.poke(false.B)
      dut.io.writerStart.poke(false.B)

      val fwd = pump(dut, Some(remoteBeat), maxCyc = 400)
      println(s"[DataPath/GATHER_MIDDLE] forwarded a beat to the next hop: ${fwd.isDefined}")

      // The forward must be the FOLD of the two operands, not a copy of either. Without this the
      // test would silently pass on a plain chained write and prove nothing about gathers.
      assert(fwd.isDefined, "no beat was forwarded to the next hop at all")
      assert(fwd.get != localBeat && fwd.get != remoteBeat,
             "the hop forwarded a RAW operand instead of the fold -- the junction was never armed, " +
             "so this is not exercising a gather")
      println("[DataPath/GATHER_MIDDLE] the forwarded beat IS the fold (junction engaged)")

      // Let everything drain. A middle hop writes nothing locally, so once the fold has been forwarded the hop
      // has no work left and MUST go idle -- otherwise it can never serve another gather.
      dut.io.remoteXDMAData.toRemote.ready.poke(true.B)
      var settle = 0
      while (settle < 600 && dut.io.writerBusy.peekBoolean()) { dut.clock.step(1); settle += 1 }

      println(s"[DataPath/GATHER_MIDDLE] after the fold: writerBusy=${dut.io.writerBusy.peekBoolean()} " +
              s"readerBusy=${dut.io.readerBusy.peekBoolean()} (settled after $settle cycles)")

      assert(!dut.io.writerBusy.peekBoolean(),
             "writerBusy is STILL asserted long after the fold was forwarded. A ChainGather middle hop writes " +
             "nothing to local TCDM, so its writer is armed with an address stream nothing ever drains and " +
             "Writer.io.busy (addressgen.busy | ~addressgen.bufferEmpty) never falls. That pins " +
             "gatherTrigger, so XDMADataSwitch never leaves gActive and this hop can never gather again -- " +
             "which is why only the FIRST gather through a chain with a middle hop works.")
    }
  }

  /** THE GATE: a middle hop must serve a SECOND gather, with the identical result.
    *
    * The test above catches the stuck `writerBusy` level. This one catches everything that would still make the
    * second gather wrong once the level is unstuck -- a junction left half-armed, a skid FIFO holding a leftover
    * operand, a switch FSM that never returned to idle, an AGU address buffer that carried entries across tasks.
    * It is the unit-level statement of the HeMAiA symptom: round 1 byte-exact, round 2 moves nothing.
    */
  "XDMADataPath_GATHER_MIDDLE_TWICE" should "fold identically on the second gather through the same hop" in {
    test(new XDMADataPath(readerParam, writerParam, "dp"))
      .withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation, flags)) { dut =>

      def quiesce(): Unit = {
        dut.io.readerStart.poke(false.B)
        dut.io.writerStart.poke(false.B)
        dut.io.remoteXDMAData.fromRemote.valid.poke(false.B)
        dut.io.remoteXDMAData.toRemote.ready.poke(false.B)
        dut.io.tcdmReader.req.foreach(_.ready.poke(false.B))
        dut.io.tcdmReader.rsp.foreach(_.valid.poke(false.B))
        dut.io.tcdmWriter.req.foreach(_.ready.poke(false.B))
      }

      def oneGather(round: Int): BigInt = {
        quiesce()
        armCfg(dut, writerSide = false, collective = true, middle = true, 0, 0)
        armCfg(dut, writerSide = true, collective = true, middle = true, 1, csrMoment(pairSlots))
        dut.clock.step(4)
        assert(!dut.io.writerBusy.peekBoolean(),
               s"round $round started with writerBusy already asserted -- the hop never retired from the " +
               "previous gather, so it can never be armed again")

        dut.io.readerStart.poke(true.B)
        dut.io.writerStart.poke(true.B)
        dut.clock.step(1)
        dut.io.readerStart.poke(false.B)
        dut.io.writerStart.poke(false.B)

        val fwd = pump(dut, Some(remoteBeat), maxCyc = 400)
        assert(fwd.isDefined, s"round $round forwarded NO beat to the next hop")
        assert(fwd.get != localBeat && fwd.get != remoteBeat,
               s"round $round forwarded a RAW operand instead of the fold -- the junction was not armed")

        // let it drain and go idle again
        dut.io.remoteXDMAData.toRemote.ready.poke(true.B)
        var settle = 0
        while (settle < 600 && dut.io.writerBusy.peekBoolean()) { dut.clock.step(1); settle += 1 }
        println(s"[DataPath/GATHER_MIDDLE_TWICE] round $round: fold=0x${fwd.get.toString(16)} " +
                s"writerBusy=${dut.io.writerBusy.peekBoolean()} after $settle cycles")
        assert(!dut.io.writerBusy.peekBoolean(),
               s"round $round never retired: writerBusy still asserted after the fold was forwarded")
        dut.clock.step(20)
        fwd.get
      }

      val first  = oneGather(1)
      val second = oneGather(2)
      assert(second == first,
             "the SECOND gather through this middle hop produced a different fold:\n" +
             s"  round 1: 0x${first.toString(16)}\n  round 2: 0x${second.toString(16)}")
      println("[DataPath/GATHER_MIDDLE_TWICE] both gathers folded identically")
    }
  }

  /** THE REALISTIC SKEW: at a real middle hop the two cfg frames do NOT arrive together.
    *
    * The unroll emits the reader companion BEFORE the writer frame (it takes the lower arbiter index in
    * XDMACtrl, so a hop is told what to read before it is told to fold), and the two cross the link separately.
    * So the local reader can already be running when the writer frame -- the one carrying the junction enable
    * bitmask, and therefore the only thing that makes this node a gather at all -- finally lands.
    *
    * Until it lands the switch sees no junction, so `isGather` is low and the reader's stream is routed straight
    * at `toRemote` as a plain local read. This test pins down what the hop does in that window, and that it still
    * ends up folding and retiring once the writer frame arrives.
    */
  "XDMADataPath_GATHER_MIDDLE_LATE_WRITER_CFG" should "still fold and retire when the writer frame lands after the reader is running" in {
    test(new XDMADataPath(readerParam, writerParam, "dp"))
      .withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation, flags)) { dut =>

      dut.io.readerStart.poke(false.B)
      dut.io.writerStart.poke(false.B)
      dut.io.remoteXDMAData.fromRemote.valid.poke(false.B)
      dut.io.remoteXDMAData.toRemote.ready.poke(false.B)
      dut.io.tcdmReader.req.foreach(_.ready.poke(false.B))
      dut.io.tcdmReader.rsp.foreach(_.valid.poke(false.B))
      dut.io.tcdmWriter.req.foreach(_.ready.poke(false.B))

      // the reader companion has landed; the writer frame has NOT (no junction armed yet)
      armCfg(dut, writerSide = false, collective = true, middle = true, 0, 0)
      armCfg(dut, writerSide = true, collective = false, middle = false, 0, 0)
      dut.clock.step(4)
      dut.io.readerStart.poke(true.B)
      dut.clock.step(1)
      dut.io.readerStart.poke(false.B)

      // ... a few cycles later the writer frame arrives and arms the junction
      dut.clock.step(6)
      armCfg(dut, writerSide = true, collective = true, middle = true, 1, csrMoment(pairSlots))
      dut.clock.step(1)
      dut.io.writerStart.poke(true.B)
      dut.clock.step(1)
      dut.io.writerStart.poke(false.B)

      val fwd = pump(dut, Some(remoteBeat), maxCyc = 400)
      println(s"[DataPath/LATE_WRITER_CFG] forwarded: ${fwd.map(b => "0x" + b.toString(16)).getOrElse("nothing")}")
      assert(fwd.isDefined, "nothing was forwarded to the next hop at all")
      assert(fwd.get != localBeat,
             "the hop pushed its RAW local operand to the next hop instead of the fold: the local reader ran " +
             "before the writer frame armed the junction, so the switch routed it out as a plain local read " +
             "and the chain carries an operand where a partial fold belongs")
      assert(fwd.get != remoteBeat, "the hop forwarded the arriving partial unchanged")

      dut.io.remoteXDMAData.toRemote.ready.poke(true.B)
      var settle = 0
      while (settle < 600 && dut.io.writerBusy.peekBoolean()) { dut.clock.step(1); settle += 1 }
      println(s"[DataPath/LATE_WRITER_CFG] after the fold: writerBusy=${dut.io.writerBusy.peekBoolean()} " +
              s"(settled after $settle cycles)")
      assert(!dut.io.writerBusy.peekBoolean(), "the hop never retired after a skewed start")
    }
  }

  /** THE OTHER HALF OF THE CONTRACT: a middle hop must advertise itself as participating from the instant it is
    * configured, without a single idle cycle in between.
    *
    * `fromRemoteAccompaniedCfg.readyToTransfer` is derived from this level, and the AXI grant manager keys its
    * IDLE -> WRITE_MIDDLE transition on it. A hop whose level dips low is simply not in the chain while it is
    * low: the upstream hop finds no one to grant to, and a P=4 gather quietly returns the P=2 answer. Since a
    * middle hop's writer engine is deliberately not started, that level cannot come from `writer.io.busy`, and
    * the local reader needs several cycles to raise `readerBusy` -- so the switch has to bridge the gap itself.
    *
    * This is the assertion that stops the stuck-writer fix from being "fixed" by simply dropping the hop.
    */
  "XDMADataPath_GATHER_MIDDLE_PARTICIPATION" should "hold writerBusy high continuously from the start pulse until the fold leaves" in {
    test(new XDMADataPath(readerParam, writerParam, "dp"))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>

      dut.io.readerStart.poke(false.B)
      dut.io.writerStart.poke(false.B)
      dut.io.remoteXDMAData.fromRemote.valid.poke(false.B)
      dut.io.remoteXDMAData.toRemote.ready.poke(false.B)
      dut.io.tcdmReader.req.foreach(_.ready.poke(false.B))
      dut.io.tcdmReader.rsp.foreach(_.valid.poke(false.B))
      dut.io.tcdmWriter.req.foreach(_.ready.poke(false.B))
      armCfg(dut, writerSide = false, collective = true, middle = true, 0, 0)
      armCfg(dut, writerSide = true, collective = true, middle = true, 1, csrMoment(pairSlots))
      dut.clock.step(4)

      dut.io.readerStart.poke(true.B)
      dut.io.writerStart.poke(true.B)
      assert(dut.io.writerBusy.peekBoolean(),
             "writerBusy is LOW on the very cycle the hop is started. The hop is not in the chain yet, so the " +
             "upstream grant manager cannot transition IDLE -> WRITE_MIDDLE for it.")
      dut.clock.step(1)
      dut.io.readerStart.poke(false.B)
      dut.io.writerStart.poke(false.B)

      // walk to the fold, checking the level never dips
      var cyc      = 0
      var dippedAt = -1
      var folded   = false
      val nCh      = dut.io.tcdmReader.req.length
      dut.io.remoteXDMAData.fromRemote.bits.poke(remoteBeat.U)
      dut.io.remoteXDMAData.fromRemote.valid.poke(true.B)
      dut.io.remoteXDMAData.toRemote.ready.poke(true.B)
      dut.io.tcdmWriter.req.foreach(_.ready.poke(true.B))
      var remoteDone = false
      while (cyc < 400 && !folded) {
        val fired = (0 until nCh).map { i =>
          dut.io.tcdmReader.req(i).ready.poke(true.B)
          dut.io.tcdmReader.req(i).valid.peekBoolean()
        }
        val frFire = !remoteDone && dut.io.remoteXDMAData.fromRemote.ready.peekBoolean()
        if (dut.io.remoteXDMAData.toRemote.valid.peekBoolean()) folded = true
        if (!dut.io.writerBusy.peekBoolean() && dippedAt < 0) dippedAt = cyc
        dut.clock.step(1); cyc += 1
        (0 until nCh).foreach { i =>
          if (fired(i)) {
            dut.io.tcdmReader.rsp(i).valid.poke(true.B)
            dut.io.tcdmReader.rsp(i).bits.data.poke(((localBeat >> (64 * i)) & ((BigInt(1) << 64) - 1)).U)
          } else dut.io.tcdmReader.rsp(i).valid.poke(false.B)
        }
        if (frFire) { remoteDone = true; dut.io.remoteXDMAData.fromRemote.valid.poke(false.B) }
      }
      println(s"[DataPath/GATHER_MIDDLE_PARTICIPATION] fold reached toRemote after $cyc cycles, " +
              s"writerBusy dip at ${if (dippedAt < 0) "never" else dippedAt.toString}")
      assert(folded, "the fold never reached the next hop")
      assert(dippedAt < 0,
             s"writerBusy went LOW at cycle $dippedAt, before the fold was forwarded. The hop drops out of the " +
             "chain for that window and the gather truncates.")
    }
  }

  /** THE STALE-CFG SCENARIO, which is what a real hop actually sees between tasks.
    *
    * `XDMACtrl` drives `io.writerCfg` from `currentCfgDst.bits` unconditionally, and that wire comes out of a
    * one-entry `-|>` cut whose `io.deq.bits := ram(deq_ptr.value)` is NOT gated on the queue being non-empty
    * (CustomOperators.scala). So after a gather retires and its cfg is popped, the datapath keeps being shown
    * round 1's writer cfg -- junction enable word included -- until a new writer frame lands.
    *
    * That matters because the two cfg frames of the NEXT gather do not arrive together: the reader companion is
    * dispatched at higher arbiter priority, so the local reader can already be producing its operand while the
    * only writer cfg on show is the stale one. The junction is therefore armed by a dequeued frame, and this
    * hop's operand `b` goes into the junction's skid FIFO with no partner in sight. `jct_start_i` resets the
    * credit and the retire pipe but does not drain those FIFOs, so whatever was parked early is still there when
    * the real cfg finally arrives.
    *
    * The question this test answers: does the second fold still come out right?
    */
  "XDMADataPath_GATHER_MIDDLE_STALE_CFG_SKEW" should "fold correctly on a second gather whose reader starts before the writer frame lands" in {
    test(new XDMADataPath(readerParam, writerParam, "dp"))
      .withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation, flags)) { dut =>

      def quiesce(): Unit = {
        dut.io.readerStart.poke(false.B)
        dut.io.writerStart.poke(false.B)
        dut.io.remoteXDMAData.fromRemote.valid.poke(false.B)
        dut.io.remoteXDMAData.toRemote.ready.poke(false.B)
        dut.io.tcdmReader.req.foreach(_.ready.poke(false.B))
        dut.io.tcdmReader.rsp.foreach(_.valid.poke(false.B))
        dut.io.tcdmWriter.req.foreach(_.ready.poke(false.B))
      }

      // ---- round 1: an ordinary gather through this middle hop ----
      quiesce()
      armCfg(dut, writerSide = false, collective = true, middle = true, 0, 0)
      armCfg(dut, writerSide = true, collective = true, middle = true, 1, csrMoment(pairSlots))
      dut.clock.step(4)
      dut.io.readerStart.poke(true.B)
      dut.io.writerStart.poke(true.B)
      dut.clock.step(1)
      dut.io.readerStart.poke(false.B)
      dut.io.writerStart.poke(false.B)
      val first = pump(dut, Some(remoteBeat), maxCyc = 400)
      assert(first.isDefined, "round 1 forwarded nothing")
      dut.io.remoteXDMAData.toRemote.ready.poke(true.B)
      var s1 = 0
      while (s1 < 600 && dut.io.writerBusy.peekBoolean()) { dut.clock.step(1); s1 += 1 }
      assert(!dut.io.writerBusy.peekBoolean(), "round 1 never retired")
      println(s"[DataPath/STALE_CFG_SKEW] round 1 fold=0x${first.get.toString(16)} (retired after $s1)")

      // ---- round 2: the reader companion lands first; the writer cfg on show is still round 1's ----
      // deliberately NOT re-arming the writer cfg here: that IS the stale cut.
      quiesce()
      dut.clock.step(4)
      dut.io.readerStart.poke(true.B)
      dut.clock.step(1)
      dut.io.readerStart.poke(false.B)

      // serve the local reader for a while, so its operand is really in flight before the writer frame arrives
      val nCh = dut.io.tcdmReader.req.length
      for (_ <- 0 until 12) {
        val fired = (0 until nCh).map { i =>
          dut.io.tcdmReader.req(i).ready.poke(true.B)
          dut.io.tcdmReader.req(i).valid.peekBoolean()
        }
        dut.clock.step(1)
        (0 until nCh).foreach { i =>
          if (fired(i)) {
            dut.io.tcdmReader.rsp(i).valid.poke(true.B)
            dut.io.tcdmReader.rsp(i).bits.data.poke(((localBeat >> (64 * i)) & ((BigInt(1) << 64) - 1)).U)
          } else dut.io.tcdmReader.rsp(i).valid.poke(false.B)
        }
      }
      dut.io.tcdmReader.rsp.foreach(_.valid.poke(false.B))

      // ... and only now does the writer frame land and re-arm the junction
      armCfg(dut, writerSide = true, collective = true, middle = true, 1, csrMoment(pairSlots))
      dut.clock.step(1)
      dut.io.writerStart.poke(true.B)
      dut.clock.step(1)
      dut.io.writerStart.poke(false.B)

      val second = pump(dut, Some(remoteBeat), maxCyc = 400)
      println(s"[DataPath/STALE_CFG_SKEW] round 2 fold=" +
              second.map(b => "0x" + b.toString(16)).getOrElse("NOTHING"))
      assert(second.isDefined,
             "round 2 forwarded NOTHING. The local reader ran while the writer cfg on show was the popped " +
             "round-1 frame, so the junction was armed by a dequeued cfg and swallowed this hop's operand " +
             "before the real cfg arrived.")
      assert(second.get == first.get,
             "round 2 folded a DIFFERENT value:\n" +
             s"  round 1: 0x${first.get.toString(16)}\n  round 2: 0x${second.get.toString(16)}\n" +
             "The operand parked in the junction's skid FIFO during the stale-cfg window is still there: " +
             "jct_start_i resets the credit and the retire pipe but does not drain aQ/bQ.")

      dut.io.remoteXDMAData.toRemote.ready.poke(true.B)
      var s2 = 0
      while (s2 < 600 && dut.io.writerBusy.peekBoolean()) { dut.clock.step(1); s2 += 1 }
      assert(!dut.io.writerBusy.peekBoolean(), "round 2 never retired")
      println(s"[DataPath/STALE_CFG_SKEW] round 2 retired after $s2 cycles; both folds identical")
    }
  }

  /** THE MULTI-BEAT GATE.
    *
    * Every other test here moves ONE beat, which leaves half the defect unexercised:
    * `Writer.io.busy = addressgen.busy | ~addressgen.bufferEmpty` and a single-vector AGU sweep only ever pins
    * the second term. A real transfer is many beats, the AGU is still sweeping while the data stalls, and both
    * terms are live. This runs a 4-beat gather twice through a middle hop.
    */
  "XDMADataPath_GATHER_MIDDLE_MULTIBEAT_TWICE" should "fold every beat and retire, on both of two 4-beat gathers" in {
    test(new XDMADataPath(readerParam, writerParam, "dp"))
      .withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation, flags)) { dut =>

      val BEATS = 4

      def oneGather(round: Int): Seq[BigInt] = {
        dut.io.readerStart.poke(false.B)
        dut.io.writerStart.poke(false.B)
        dut.io.remoteXDMAData.fromRemote.valid.poke(false.B)
        dut.io.remoteXDMAData.toRemote.ready.poke(false.B)
        dut.io.tcdmReader.req.foreach(_.ready.poke(false.B))
        dut.io.tcdmReader.rsp.foreach(_.valid.poke(false.B))
        dut.io.tcdmWriter.req.foreach(_.ready.poke(false.B))
        armCfg(dut, writerSide = false, collective = true, middle = true, 0, 0, beats = BEATS)
        armCfg(dut, writerSide = true, collective = true, middle = true, 1, csrMoment(pairSlots), beats = BEATS)
        dut.clock.step(4)
        assert(!dut.io.writerBusy.peekBoolean(), s"round $round started with writerBusy already asserted")

        dut.io.readerStart.poke(true.B)
        dut.io.writerStart.poke(true.B)
        dut.clock.step(1)
        dut.io.readerStart.poke(false.B)
        dut.io.writerStart.poke(false.B)

        val fwd = pumpN(dut, Some(remoteBeat), beats = BEATS, maxCyc = 600)
        dut.io.remoteXDMAData.toRemote.ready.poke(true.B)
        var settle = 0
        while (settle < 900 && dut.io.writerBusy.peekBoolean()) { dut.clock.step(1); settle += 1 }
        println(s"[DataPath/MULTIBEAT] round $round forwarded ${fwd.length} beats, " +
                s"writerBusy=${dut.io.writerBusy.peekBoolean()} after $settle cycles")
        assert(fwd.length == BEATS,
               s"round $round forwarded ${fwd.length} beats, expected $BEATS -- a multi-beat fold did not " +
               "complete through this hop")
        assert(fwd.forall(_ == fwd.head), s"round $round forwarded beats are not all the same fold: $fwd")
        assert(fwd.head != localBeat && fwd.head != remoteBeat,
               s"round $round forwarded a RAW operand instead of the fold")
        assert(!dut.io.writerBusy.peekBoolean(),
               s"round $round never retired: writerBusy still asserted after $BEATS beats were forwarded. " +
               "With a multi-beat transfer BOTH halves of Writer.io.busy are live, so a writer engine armed " +
               "for a local write that never happens pins addressgen.busy as well as ~bufferEmpty.")
        dut.clock.step(20)
        fwd
      }

      val first  = oneGather(1)
      val second = oneGather(2)
      assert(second == first,
             s"the second 4-beat gather produced different beats:\n  round 1: $first\n  round 2: $second")
      println("[DataPath/MULTIBEAT] both 4-beat gathers folded identically")
    }
  }
}
