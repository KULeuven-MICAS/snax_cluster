package snax.DataPathJunction

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathExtension.FpHelpers
import snax.DataPathJunction.JunctionTestUtils._
import snax.utils.DecoupledCut._

/** Harness for the SOCKET itself -- `DataPathJunctionHost` with a list of operators, not a single junction. */
class JunctionHostHarness(junctions: Seq[HasDataPathJunction], dataWidth: Int = 512)
    extends Module
    with RequireAsyncReset {
  val dut = Module(new DataPathJunctionHost(junctions, dataWidth = dataWidth, moduleNamePrefix = "host_dut"))
  val io  = IO(chiselTypeOf(dut.io))

  io.busy    := dut.io.busy
  io.active  := dut.io.active
  io.starved := dut.io.starved
  io.cfgerr  := dut.io.cfgerr
  dut.io.cfg   := io.cfg
  dut.io.start := io.start

  io.data.a -||> dut.io.data.a
  io.data.b -||> dut.io.data.b
  dut.io.data.out -||> io.data.out
}

/** THE SOCKET TEST. Everything here is about the host, not about either operator's arithmetic.
  *
  * The socket's promise is that an operator is a PLUG-IN: you hand the host a list, each entry gets its own user
  * CSR words, and the enable bitmask arms exactly one per transfer. That promise has never been exercised with
  * more than one entry -- the deployed configuration has always been a single-element list -- so the CSR
  * allocation in `connectCfgWithList` has never actually had to distribute anything.
  *
  * The words below are chosen so that a MIS-ROUTED CSR IS DETECTABLE. The monoid word read as a linear one
  * decodes to an FP16 ADD instead of the BF16 MUL that was asked for; the linear word read as a monoid one
  * decodes to a different nValid and a different field count than the geometry that was asked for. Both are
  * plainly different from the right answer, which a lazier choice of words would not have been.
  */
class JunctionHostTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))
  private val dHead     = 8
  private val pairSlots = 8
  private val elemWidth = 16
  private val fpPipe    = 1

  import ElementwiseJunction._
  import MonoidCombine._

  // the operator list, in the same order as the cfg's `writer_junctions`
  private def operators: Seq[HasDataPathJunction] = Seq(
    new HasElementwiseJunction(elemWidth = elemWidth, fpPipe = fpPipe),
    new HasMonoidJunction(fpPipe = fpPipe)
  )
  private val EW_SLOT  = 0
  private val MON_SLOT = 1

  private def csrLinear(op: Int, fmt: Int): BigInt = (BigInt(fmt) << 4) | BigInt(op)
  /** the monoid geometry word: [7:0] nValid | [11:8] n | [21:18] nExp | [25:22] nAdd | [27:26] sigma */
  private def csrMonoid(n: Int, nExp: Int, nAdd: Int, sigma: Int, nValid: Int): BigInt =
    (BigInt(sigma) << 26) | (BigInt(nAdd) << 22) | (BigInt(nExp) << 18) | (BigInt(n) << 8) | BigInt(nValid)

  private def encBf16(d: Double): BigInt = {
    val b = java.lang.Float.floatToIntBits(d.toFloat)
    BigInt(((b + 0x7fff + ((b >>> 16) & 1)) >>> 16) & 0xffff)
  }
  private def decBf16(b: BigInt): Double = java.lang.Float.intBitsToFloat((b.toInt & 0xffff) << 16).toDouble

  private def packBf16(v: Seq[Double]): BigInt = {
    var x = BigInt(0)
    for ((e, i) <- v.zipWithIndex) x |= encBf16(e) << (16 * i)
    x
  }
  private def packPairs(p: Seq[(Double, Double)]): BigInt = {
    var b = BigInt(0)
    for ((v, k) <- p.zipWithIndex) { b |= f32(v._1) << (32 * k); b |= f32(v._2) << (32 * (pairSlots + k)) }
    b
  }

  /** drive one operand pair through the HOST with a given enable bitmask, both CSR slots loaded */
  private def runHost(dut: JunctionHostHarness, enable: Int, csrs: Seq[BigInt],
                      aBeat: BigInt, bBeat: BigInt): BigInt = {
    for ((c, i) <- csrs.zipWithIndex) dut.io.cfg.userCsr(i).poke(c.U)
    dut.io.cfg.enable.poke(enable.U)
    dut.io.start.poke(true); dut.clock.step(1); dut.io.start.poke(false)
    var out = BigInt(0)
    var th  = new chiseltest.internal.TesterThreadList(Seq())
    th = th.fork {
      dut.io.data.a.bits.poke(aBeat.U); dut.io.data.a.valid.poke(true)
      while (!dut.io.data.a.ready.peekBoolean()) dut.clock.step(1)
      dut.clock.step(1); dut.io.data.a.valid.poke(false)
    }
    th = th.fork {
      dut.io.data.b.bits.poke(bBeat.U); dut.io.data.b.valid.poke(true)
      while (!dut.io.data.b.ready.peekBoolean()) dut.clock.step(1)
      dut.clock.step(1); dut.io.data.b.valid.poke(false)
    }
    th = th.fork {
      while (!dut.io.data.out.valid.peekBoolean()) dut.clock.step(1)
      out = dut.io.data.out.bits.peekInt()
      dut.io.data.out.ready.poke(true); dut.clock.step(1); dut.io.data.out.ready.poke(false)
    }
    th.joinAndStep()
    dut.io.data.out.ready.poke(true)
    var g = 0; while (dut.io.busy.peekBoolean() && g < 400) { dut.clock.step(1); g += 1 }
    dut.io.data.out.ready.poke(false)
    out
  }

  "JunctionHost_twoOperators" should "give each operator its own CSR words and arm exactly one" in {
    val rng = new Random(0x50c1)
    // BOTH words loaded at once, for every transfer -- that is the whole point
    val wLin = csrLinear(OP_MUL, FpHelpers.FMT_BF16)
    val wMon = csrMonoid(n = 1, nExp = 1, nAdd = 0, sigma = 3, nValid = pairSlots)
    val csrs = Seq(wLin, wMon)

    val lanes = 512 / 16
    val la    = Seq.fill(lanes)(rng.between(-4.0, 4.0))
    val lb    = Seq.fill(lanes)(rng.between(-4.0, 4.0))
    val pa    = Seq.fill(pairSlots)((rng.between(-3.0, 6.0), rng.between(1.0, 5.0)))
    val pb    = Seq.fill(pairSlots)((rng.between(-3.0, 6.0), rng.between(1.0, 5.0)))

    test(new JunctionHostHarness(operators)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      // ---- arm operator 0 (linear) -- BF16 MUL ----
      val outL = runHost(dut, 1 << EW_SLOT, csrs, packBf16(la), packBf16(lb))
      for (i <- 0 until lanes) {
        val gold = decBf16(encBf16(la(i))) * decBf16(encBf16(lb(i)))
        val got  = decBf16(lane(outL, i, 16))
        assert(math.abs(got - gold) <= math.abs(gold) * 1e-2 + 1e-3,
               f"lane $i: BF16 MUL gave $got%.5f, expected $gold%.5f -- a mis-routed CSR decodes as FP16 ADD")
      }

      // ---- arm operator 1 (monoid) -- the softmax normalizer ----
      val outM = runHost(dut, 1 << MON_SLOT, csrs, packPairs(pa), packPairs(pb))
      for (k <- 0 until pairSlots) {
        val gm = math.max(pa(k)._1, pb(k)._1)
        val gl = pa(k)._2 * math.exp(pa(k)._1 - gm) + pb(k)._2 * math.exp(pb(k)._1 - gm)
        assert(math.abs(laneF32(outM, k) - gm) <= math.abs(gm) * 1e-6 + 1e-9,
               s"slot $k: m = ${laneF32(outM, k)} vs $gm -- a mis-routed CSR decodes as nValid=17, SUM")
        assert(math.abs(laneF32(outM, pairSlots + k) - gl) / gl <= 1.5e-2, s"slot $k: l")
      }
      println("[Host] two operators, one socket: each reads its OWN csr word and the enable bitmask arms one")
    }
  }

  "JunctionHost_noneArmed" should "forward stream A untouched when no operator is enabled" in {
    // With an empty enable the host is a wire and B is never consumed -- the property that lets a non-collective
    // transfer share the same datapath.
    val rng   = new Random(0xd15a)
    val beats = Seq.fill(6)(packPairs(Seq.fill(pairSlots)((rng.between(-4.0, 4.0), rng.between(0.5, 3.0)))))
    test(new JunctionHostHarness(operators)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      dut.io.cfg.userCsr(0).poke(csrLinear(OP_MUL, FpHelpers.FMT_BF16).U)
      dut.io.cfg.userCsr(1).poke(csrMonoid(n = 1, nExp = 1, nAdd = 0, sigma = 3, nValid = pairSlots).U)
      dut.io.cfg.enable.poke(0.U)
      dut.io.start.poke(true); dut.clock.step(1); dut.io.start.poke(false)
      assert(!dut.io.active.peekBoolean(), "no operator armed, yet the host reports active")
      dut.io.data.out.ready.poke(true)
      dut.io.data.b.valid.poke(false)
      dut.io.data.a.valid.poke(true); dut.io.data.a.bits.poke(beats(0).U)
      var got = List[BigInt]()
      var cyc = 0
      var fed = 0
      while (got.length < beats.length && cyc < 600) {
        val canFeed = fed < beats.length && dut.io.data.a.ready.peekBoolean()
        val outNow  = dut.io.data.out.valid.peekBoolean()
        val bits    = if (outNow) dut.io.data.out.bits.peekInt() else BigInt(0)
        dut.clock.step(1); cyc += 1
        if (canFeed) {
          fed += 1
          if (fed < beats.length) dut.io.data.a.bits.poke(beats(fed).U) else dut.io.data.a.valid.poke(false)
        }
        if (outNow) got = got :+ bits
      }
      assert(got == beats.toList, "an unarmed socket must forward stream A byte-identically")
      println("[Host] no operator armed => a transparent wire, and B is never consumed")
    }
  }

  "JunctionHost_activeFlag" should "report active only for the operator that is armed" in {
    test(new JunctionHostHarness(operators)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      dut.io.cfg.userCsr(0).poke(0.U); dut.io.cfg.userCsr(1).poke(0.U)
      for ((en, want) <- Seq(0 -> false, 1 -> true, 2 -> true)) {
        dut.io.cfg.enable.poke(en.U)
        dut.clock.step(1)
        assert(dut.io.active.peekBoolean() == want, s"enable=$en: active should be $want")
      }
      println("[Host] `active` tracks the enable bitmask -- it is what arms the collective dataflow")
    }
  }

  "JunctionHost_cfgErrIsPerArmedOperator" should "report only the armed operator's verdict" in {
    // O5 through the socket. Both operators hold a CSR word at all times, and a word that is nonsense for one is
    // routine for the other -- so the host must report the ARMED operator's verdict and nobody else's, or every
    // transfer would raise an error on behalf of an operator that is not running.
    test(new JunctionHostHarness(operators)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      def check(en: Int, lin: BigInt, mon: BigInt): Boolean = {
        dut.io.cfg.userCsr(0).poke(lin.U)
        dut.io.cfg.userCsr(1).poke(mon.U)
        dut.io.cfg.enable.poke(en.U)
        dut.clock.step(2)
        dut.io.cfgerr.peekBoolean()
      }
      val linOk  = csrLinear(OP_MUL, FpHelpers.FMT_BF16)
      val monOk  = csrMonoid(n = 1, nExp = 1, nAdd = 0, sigma = 3, nValid = pairSlots)
      val monBad = csrMonoid(n = 1, nExp = 1, nAdd = 0, sigma = 3, nValid = 0) // no live slot
      assert(!check(1 << EW_SLOT, linOk, monOk), "two good words must not raise")
      assert(!check(1 << MON_SLOT, linOk, monOk), "two good words must not raise, either operator armed")
      // the monoid word is nonsense, but the LINEAR operator is the one armed -- silence is correct
      assert(!check(1 << EW_SLOT, linOk, monBad), "a bad word for an operator that is not armed must be silent")
      assert(check(1 << MON_SLOT, linOk, monBad), "the armed operator's bad word must be reported")
      assert(!check(0, linOk, monBad), "with nothing armed the socket has no verdict to report")
      println("[Host/O5] the socket reports the armed operator's configuration verdict and no one else's")
    }
  }
}
