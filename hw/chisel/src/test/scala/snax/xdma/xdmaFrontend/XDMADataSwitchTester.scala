package snax.xdma.xdmaFrontend

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathJunction.HasMonoidJunction
import snax.readerWriter.ReaderWriterParam
import snax.xdma.DesignParams._

/** Tier-1 for `XDMADataSwitch` -- one test per mode of the crossing.
  *
  * Two things are pinned down here:
  *
  *   1. LOCAL / READ / WRITE / CHAINWRITE route as specified. CHAINWRITE in particular must deliver the arriving
  *      beat to BOTH the local writer and the next hop.
  *   2. The gather modes route through the junction instead: CHAINGATHER folds and forwards (writing nothing
  *      locally), GATHERROOT folds and lands the answer locally (forwarding nothing).
  *
  * The CHAINGATHER test also checks the busy-level property. `writerBusy` arms the grant manager and the finish
  * FSM, and a gather middle node's local writer never goes busy -- so if `writerBusy` were derived from the writer
  * alone, the remote stream could never enter the switch and the chain would stall with no diagnostic. The test
  * holds `writerBusyRaw` LOW throughout and requires the transfer to complete anyway.
  */
class XDMADataSwitchTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))
  private val dataWidth = 512
  private val pairSlots = 8

  private def param = new XDMAParam(
    cfgParam          = new XDMAConfigParam(addrWidth = 32, dataWidth = 32),
    axiParam          = new XDMAAXIParam,
    crossClusterParam = new XDMACrossClusterParam,
    rwParam           = new ReaderWriterParam,
    extParam          = Seq(),
    junctionParam     = Seq(new HasMonoidJunction(dataWidth = dataWidth))
  )

  private def f32(f: Double): BigInt = BigInt(java.lang.Float.floatToIntBits(f.toFloat).toLong & 0xffffffffL)
  private def dec(b: BigInt): Double = java.lang.Float.intBitsToFloat(b.toInt).toDouble
  private def laneF32(beat: BigInt, i: Int): Double = dec((beat >> (32 * i)) & ((BigInt(1) << 32) - 1))

  private def packPairs(p: Seq[(Double, Double)]): BigInt = {
    var b = BigInt(0)
    for ((v, k) <- p.zipWithIndex) { b |= f32(v._1) << (32 * k); b |= f32(v._2) << (32 * (pairSlots + k)) }
    b
  }
  // the monoid geometry word: [7:0] nValid | [11:8] n | [21:18] nExp | [27:26] sigma. (m, l), one twisted
  // value coordinate, 8 partials per beat.
  private def csrMoment(nValid: Int): BigInt =
    (BigInt(3) << 26) | (BigInt(1) << 18) | (BigInt(1) << 8) | BigInt(nValid)

  /** Park every input at a quiescent, non-gather, non-chained state. */
  private def idle(dut: XDMADataSwitch): Unit = {
    dut.io.readerLocalLoopback.poke(false)
    dut.io.writerLocalLoopback.poke(false)
    dut.io.writerRemoteLoopback.poke(false)
    dut.io.writerStart.poke(false)
    dut.io.writerBusyRaw.poke(false)
    dut.io.readerBusy.poke(false)
    dut.io.junctionCfg.enable.poke(0)
    dut.io.junctionCfg.userCsr(0).poke(0)
    dut.io.localIn.valid.poke(false)
    dut.io.fromRemote.valid.poke(false)
    dut.io.localOut.ready.poke(false)
    dut.io.toRemote.ready.poke(false)
    dut.clock.step(2)
  }

  /** Offer `localIn` and/or `fromRemote` for up to `maxCyc` cycles while draining both outputs, and report what
    * each output port produced. `None` means the port never fired.
    */
  private def pump(dut: XDMADataSwitch, localIn: Option[BigInt], fromRemote: Option[BigInt],
                   maxCyc: Int = 300): (Option[BigInt], Option[BigInt]) = {
    var outLocal: Option[BigInt]  = None
    var outRemote: Option[BigInt] = None
    var liDone = localIn.isEmpty
    var frDone = fromRemote.isEmpty
    localIn.foreach { b => dut.io.localIn.bits.poke(b.U); dut.io.localIn.valid.poke(true) }
    fromRemote.foreach { b => dut.io.fromRemote.bits.poke(b.U); dut.io.fromRemote.valid.poke(true) }
    dut.io.localOut.ready.poke(true)
    dut.io.toRemote.ready.poke(true)
    var cyc = 0
    while (cyc < maxCyc) {
      val liFire = !liDone && dut.io.localIn.ready.peekBoolean()
      val frFire = !frDone && dut.io.fromRemote.ready.peekBoolean()
      val loNow  = dut.io.localOut.valid.peekBoolean()
      val loBits = if (loNow) dut.io.localOut.bits.peekInt() else BigInt(0)
      val trNow  = dut.io.toRemote.valid.peekBoolean()
      val trBits = if (trNow) dut.io.toRemote.bits.peekInt() else BigInt(0)
      dut.clock.step(1); cyc += 1
      if (liFire) { liDone = true; dut.io.localIn.valid.poke(false) }
      if (frFire) { frDone = true; dut.io.fromRemote.valid.poke(false) }
      if (loNow && outLocal.isEmpty) outLocal = Some(loBits)
      if (trNow && outRemote.isEmpty) outRemote = Some(trBits)
      // stop once everything that could still move has moved
      if (liDone && frDone && cyc > 40) return (outLocal, outRemote)
    }
    (outLocal, outRemote)
  }

  private val beatA = packPairs(Seq.tabulate(pairSlots)(k => (1.0 + k, 2.0)))
  private val beatB = packPairs(Seq.tabulate(pairSlots)(k => (3.0 + k, 0.5)))

  "XDMADataSwitch_LOCAL_READ_WRITE" should "route the three non-chained modes" in {
    test(new XDMADataSwitch(param, dataWidth, "sw")).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      // LOCAL: (1) -> (3), the reader's stream loops back to the writer
      idle(dut)
      dut.io.readerLocalLoopback.poke(true); dut.io.writerLocalLoopback.poke(true)
      var (lo, tr) = pump(dut, Some(beatA), None)
      assert(lo.contains(beatA), s"LOCAL: localOut=$lo")
      assert(tr.isEmpty, "LOCAL must not drive toRemote")

      // READ: (1) -> (2), the reader's stream heads for the next hop (this is also the gather-chain head)
      idle(dut)
      val r2 = pump(dut, Some(beatA), None); lo = r2._1; tr = r2._2
      assert(tr.contains(beatA), s"READ: toRemote=$tr")
      assert(lo.isEmpty, "READ must not drive localOut")

      // WRITE: (4) -> (3), a plain destination. The remote stream may only enter while the writer is busy.
      idle(dut)
      dut.io.writerBusyRaw.poke(true)
      val r3 = pump(dut, None, Some(beatB)); lo = r3._1; tr = r3._2
      assert(lo.contains(beatB), s"WRITE: localOut=$lo")
      assert(tr.isEmpty, "WRITE must not drive toRemote")
      println("[Switch] LOCAL / READ / WRITE routed as specified")
    }
  }

  "XDMADataSwitch_CHAINWRITE" should "deliver the arriving beat to BOTH the writer and the next hop" in {
    test(new XDMADataSwitch(param, dataWidth, "sw")).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      idle(dut)
      dut.io.writerRemoteLoopback.poke(true) // a next hop exists
      dut.io.writerBusyRaw.poke(true)        // ... and this node writes locally too
      dut.clock.step(2)
      assert(dut.io.isChainedWrite.peekBoolean(), "CHAINWRITE state never entered")
      assert(!dut.io.isGather.peekBoolean(), "a junction-less transfer must not be a gather")
      val (lo, tr) = pump(dut, None, Some(beatB))
      assert(lo.contains(beatB), s"CHAINWRITE: localOut=$lo (must receive a copy)")
      assert(tr.contains(beatB), s"CHAINWRITE: toRemote=$tr (must forward a copy)")
      println("[Switch/CHAINWRITE] broadcast middle hop: the beat is copied to (2) and (3)")
    }
  }

  "XDMADataSwitch_CHAINGATHER" should "fold and forward without writing locally, and without a busy writer" in {
    test(new XDMADataSwitch(param, dataWidth, "sw")).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      idle(dut)
      dut.io.writerRemoteLoopback.poke(true) // a next hop exists -> middle of the chain
      dut.io.junctionCfg.enable.poke(1)
      dut.io.junctionCfg.userCsr(0).poke(csrMoment(pairSlots).U)
      // The local writer stays idle for the entire transfer; only the local reader runs.
      dut.io.writerBusyRaw.poke(false)
      dut.io.readerBusy.poke(true)
      dut.clock.step(2)
      assert(dut.io.isGather.peekBoolean(), "gather state never entered")
      assert(dut.io.isGatherMid.peekBoolean(), "a gather with a next hop must be CHAINGATHER")
      assert(dut.io.writerBusy.peekBoolean(),
             "a gather middle node must raise writerBusy without a busy writer, or the chain stalls")
      assert(!dut.io.isChainedWrite.peekBoolean(), "a gather must not also assert isChainedWrite")

      val (lo, tr) = pump(dut, Some(beatA), Some(beatB))
      assert(lo.isEmpty, s"CHAINGATHER must write NOTHING locally, got $lo")
      assert(tr.isDefined, "CHAINGATHER produced no forwarded beat")
      // the forwarded beat is the FOLD of the two operands, not a copy of either
      val out = tr.get
      assert(out != beatA && out != beatB, "CHAINGATHER forwarded a raw operand instead of the fold")
      for (k <- 0 until pairSlots) {
        val (ma, la, mb, lb) = (1.0 + k, 2.0, 3.0 + k, 0.5)
        val gm = math.max(ma, mb)
        val gl = la * math.exp(ma - gm) + lb * math.exp(mb - gm)
        assert(math.abs(laneF32(out, k) - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"slot $k m")
        assert(math.abs(laneF32(out, pairSlots + k) - gl) / gl <= 1.5e-2, s"slot $k l")
      }
      println("[Switch/CHAINGATHER] reduce middle hop: ((4),(1)) -> junction -> (2), no local write, writer idle")
    }
  }

  "XDMADataSwitch_GATHERROOT" should "fold and land the answer locally without forwarding" in {
    test(new XDMADataSwitch(param, dataWidth, "sw")).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      idle(dut)
      dut.io.writerRemoteLoopback.poke(false) // no next hop -> this node is the tail
      dut.io.junctionCfg.enable.poke(1)
      dut.io.junctionCfg.userCsr(0).poke(csrMoment(pairSlots).U)
      dut.io.writerBusyRaw.poke(true)
      dut.io.readerBusy.poke(true)
      dut.clock.step(2)
      assert(dut.io.isGatherRoot.peekBoolean(), "a gather with no next hop must be GATHERROOT")

      val (lo, tr) = pump(dut, Some(beatA), Some(beatB))
      assert(tr.isEmpty, s"GATHERROOT is the tail and must forward nothing, got $tr")
      assert(lo.isDefined, "GATHERROOT produced no local beat")
      val out = lo.get
      assert(out != beatA && out != beatB, "GATHERROOT landed a raw operand instead of the fold")
      for (k <- 0 until pairSlots) {
        val (ma, la, mb, lb) = (1.0 + k, 2.0, 3.0 + k, 0.5)
        val gm = math.max(ma, mb)
        val gl = la * math.exp(ma - gm) + lb * math.exp(mb - gm)
        assert(math.abs(laneF32(out, k) - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"slot $k m")
        assert(math.abs(laneF32(out, pairSlots + k) - gl) / gl <= 1.5e-2, s"slot $k l")
      }
      println("[Switch/GATHERROOT] reduce tail: ((4),(1)) -> junction -> (3), the answer lands here")
    }
  }
}
