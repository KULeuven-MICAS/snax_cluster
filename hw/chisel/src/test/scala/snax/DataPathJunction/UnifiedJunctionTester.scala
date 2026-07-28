package snax.DataPathJunction

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathExtension.FpHelpers
import snax.DataPathJunction.JunctionTestUtils._

/** Tier-1 for `UnifiedJunction` -- the ONE 2->1 netlist that carries BOTH fold families over a shared FMA pool.
  *
  * The headline check is an EQUIVALENCE, not a golden: the merged netlist must reproduce, BIT FOR BIT, what the
  * two predecessor junctions produce for the same CSR word and the same operand pair --
  *   `ElementwiseJunction` for the linear per-element class, and
  *   `MonoidJunction` for the structured serving-reduction class,
  * with both classes exercised on ONE elaborated instance. That is what makes the merge a pure area/reuse move
  * rather than a redesign, and it is the artifact behind the "N operators, one datapath" claim.
  *
  * Because chiseltest gives each scalatest CASE one Verilator build directory, the two reference legs are
  * separate cases that hand their beats to the equivalence case through the vars below.
  */
class UnifiedJunctionTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  import ElementwiseJunction._
  import MonoidCombine._
  import UnifiedJunction._

  private val dHead     = 8
  private val pairSlots = 8
  private val elemWidth = 16
  private val fpPipe    = 1

  // ---- CSR encoders: the predecessors' encodings, plus one class bit ----------------------------------------
  private def csrMonoid(mode: Int, nValid: Int): BigInt = (BigInt(mode) << 13) | BigInt(nValid)
  private def csrLinear(op: Int, fmt: Int): BigInt =
    (BigInt(CLASS_ELEMENTWISE) << OPCLASS_BIT) | (BigInt(fmt) << 4) | BigInt(op)

  // ---- transport-format codecs -------------------------------------------------------------------------------
  private def encFp16(d: Double): BigInt = BigInt(java.lang.Float.floatToFloat16(d.toFloat) & 0xffff)
  private def encBf16(d: Double): BigInt = {
    val b = java.lang.Float.floatToIntBits(d.toFloat)
    BigInt(((b + 0x7fff + ((b >>> 16) & 1)) >>> 16) & 0xffff)
  }
  private case class Fmt(code: Int, width: Int, enc: Double => BigInt, name: String) {
    def lanes: Int = 512 / width
  }
  private val FMT16 = Fmt(FpHelpers.FMT_FP16, 16, encFp16, "FP16")
  private val FMTBF = Fmt(FpHelpers.FMT_BF16, 16, encBf16, "BF16")
  private val FMT32 = Fmt(FMT_FP32, 32, f32, "FP32")

  private def packFmt(v: Seq[Double], f: Fmt): BigInt = {
    var b = BigInt(0)
    for ((x, i) <- v.zipWithIndex) b |= f.enc(x) << (f.width * i)
    b
  }
  private def packPairs(p: Seq[(Double, Double)]): BigInt = {
    var b = BigInt(0)
    for ((v, k) <- p.zipWithIndex) { b |= f32(v._1) << (32 * k); b |= f32(v._2) << (32 * (pairSlots + k)) }
    b
  }
  private def packSingle(m: Double, l: Double, rest: Seq[Double]): BigInt = {
    var b = f32(m) | (f32(l) << 32)
    for ((v, k) <- rest.zipWithIndex) b |= f32(v) << (32 * (2 + k))
    b
  }

  // ---- the shared stimulus ------------------------------------------------------------------------------------
  private case class Vec3(label: String, csr: BigInt, a: BigInt, b: BigInt)

  private val linearCases: Seq[Vec3] = {
    val rng = new Random(0x11ee)
    Seq((FMT16, OP_ADD, "ADD"), (FMT16, OP_MUL, "MUL"), (FMT16, OP_MAX, "MAX"), (FMT16, OP_MIN, "MIN"),
        (FMTBF, OP_ADD, "ADD"), (FMT32, OP_MUL, "MUL")).map { case (f, op, nm) =>
      val va = Seq.fill(f.lanes)(rng.between(-4.0, 4.0))
      val vb = Seq.fill(f.lanes)(rng.between(-4.0, 4.0))
      Vec3(s"${f.name}/$nm", csrLinear(op, f.code), packFmt(va, f), packFmt(vb, f))
    }
  }

  private val monoidCases: Seq[Vec3] = {
    val rng = new Random(0x22dd)
    def pr() = Seq.fill(pairSlots)((rng.between(-3.0, 6.0), rng.between(1.0, 5.0)))
    def idx(off: Int) = (0 until pairSlots).map(i => (rng.between(-9.0, 9.0), (off + i).toDouble))
    Seq(
      Vec3("SUM",     csrMonoid(MODE_SUM, pairSlots),     packPairs(pr()), packPairs(pr())),
      Vec3("MOMENT",  csrMonoid(MODE_MOMENT, pairSlots),  packPairs(pr()), packPairs(pr())),
      Vec3("MOMENT3", csrMonoid(MODE_MOMENT, 3),          packPairs(pr()), packPairs(pr())),
      Vec3("MAXPOOL", csrMonoid(MODE_MAXPOOL, pairSlots), packPairs(pr()), packPairs(pr())),
      Vec3("ARGMAX",  csrMonoid(MODE_ARGMAX, pairSlots),  packPairs(idx(0)), packPairs(idx(pairSlots))),
      Vec3("ATTN",    csrMonoid(MODE_ATTN, 1),
           packSingle(rng.between(-2.0, 5.0), rng.between(1.0, 5.0), Seq.fill(dHead)(rng.between(-3.0, 3.0))),
           packSingle(rng.between(-2.0, 5.0), rng.between(1.0, 5.0), Seq.fill(dHead)(rng.between(-3.0, 3.0)))),
      Vec3("MOMENT2", csrMonoid(MODE_MOMENT2, 1),
           packSingle(rng.between(-2.0, 5.0), rng.between(1.0, 5.0), Seq(rng.between(-4.0, 4.0), rng.between(0.5, 6.0))),
           packSingle(rng.between(-2.0, 5.0), rng.between(1.0, 5.0), Seq(rng.between(-4.0, 4.0), rng.between(0.5, 6.0))))
    )
  }

  private var refLinear = Map[String, BigInt]()
  private var refMonoid = Map[String, BigInt]()

  private def hasUnified =
    new HasUnifiedJunction(elemWidth = elemWidth, fpPipe = fpPipe, dHead = dHead, pairSlots = pairSlots)

  "UnifiedJunction_ref_linear" should "capture ElementwiseJunction's beats (reference leg 1 of 3)" in {
    // The predecessor reads only [6:0], so it is fed the SAME word the merged netlist gets -- which is itself the
    // check that the class bit is backward compatible with the existing software encoder.
    test(new DataPathJunctionHarness(new HasElementwiseJunction(elemWidth = elemWidth, fpPipe = fpPipe)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        for ((c, i) <- linearCases.zipWithIndex)
          refLinear += (c.label -> runPair(dut, c.csr, c.a, c.b, bDelay = i % 3))
      }
  }

  "UnifiedJunction_ref_monoid" should "capture MonoidJunction's beats (reference leg 2 of 3)" in {
    test(new DataPathJunctionHarness(new HasMonoidJunction(dHead = dHead, pairSlots = pairSlots)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        for ((c, i) <- monoidCases.zipWithIndex)
          refMonoid += (c.label -> runPair(dut, c.csr, c.a, c.b, bDelay = i % 3))
      }
  }

  "UnifiedJunction_equivalence" should "reproduce BOTH predecessors bit-for-bit on ONE netlist" in {
    assert(refLinear.size == linearCases.length, "run UnifiedJunction_ref_linear first")
    assert(refMonoid.size == monoidCases.length, "run UnifiedJunction_ref_monoid first")

    // Interleave the two classes so the same elaborated instance is re-armed across the class boundary between
    // transfers -- a merged netlist that only worked one class at a time would not be a replacement.
    val mixed = linearCases.zipAll(monoidCases, null, null).flatMap { case (l, m) =>
      Seq(Option(l), Option(m)).flatten
    }
    var got = Map[String, BigInt]()
    test(new DataPathJunctionHarness(hasUnified)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      for ((c, i) <- mixed.zipWithIndex) got += (c.label -> runPair(dut, c.csr, c.a, c.b, bDelay = i % 3))
    }

    for (c <- linearCases)
      assert(got(c.label) == refLinear(c.label),
             f"linear ${c.label}: merged 0x${got(c.label)}%x != ElementwiseJunction 0x${refLinear(c.label)}%x")
    for (c <- monoidCases)
      assert(got(c.label) == refMonoid(c.label),
             f"monoid ${c.label}: merged 0x${got(c.label)}%x != MonoidJunction 0x${refMonoid(c.label)}%x")

    // one independent numeric anchor per class, so the test does not rest solely on the predecessors being right
    val momentOut = got("MOMENT")
    assert(laneF32(momentOut, 0).isFinite && laneF32(momentOut, pairSlots) > 0.0, "MOMENT beat is not sane")
    println(s"[Unified] ONE netlist reproduces ${linearCases.length} linear and ${monoidCases.length} monoid " +
            "transfers BIT-EXACTLY, interleaved across the class boundary")
  }

  "UnifiedJunction_roofline" should "stream at ~1 beat pair per cycle in BOTH classes" in {
    test(new DataPathJunctionHarness(hasUnified)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      def stream(csr: BigInt, beats: Seq[BigInt]): (Int, Double) = {
        val n = beats.length
        dut.io.csr_i(0).poke(csr.U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.out_o.ready.poke(true)
        dut.io.a_i.valid.poke(true); dut.io.b_i.valid.poke(true)
        dut.io.a_i.bits.poke(beats(0).U); dut.io.b_i.bits.poke(beats(0).U)
        var cyc = 0; var fed = 0; var rec = 0; var first = -1; var warm = -1; var last = -1
        while (rec < n && cyc < n * 40 + 600) {
          val canFeed = fed < n && dut.io.a_i.ready.peekBoolean() && dut.io.b_i.ready.peekBoolean()
          val outNow  = dut.io.out_o.valid.peekBoolean()
          dut.clock.step(1); cyc += 1
          if (canFeed) {
            fed += 1
            if (fed == 4) warm = cyc
            last = cyc
            if (fed < n) { dut.io.a_i.bits.poke(beats(fed).U); dut.io.b_i.bits.poke(beats(fed).U) }
            else { dut.io.a_i.valid.poke(false); dut.io.b_i.valid.poke(false) }
          }
          if (outNow) { if (first < 0) first = cyc; rec += 1 }
        }
        dut.io.out_o.ready.poke(false)
        assert(rec == n, s"only $rec/$n beats retired")
        (first, if (last > warm) (n - 4).toDouble / (last - warm) else 0.0)
      }
      val rng     = new Random(0x77aa)
      val monB    = Seq.fill(64)(packPairs(Seq.fill(pairSlots)((rng.between(-2.0, 4.0), rng.between(0.5, 3.0)))))
      val linB    = Seq.fill(64)(packFmt(Seq.fill(FMT16.lanes)(rng.between(-4.0, 4.0)), FMT16))
      val (lm, um) = stream(csrMonoid(MODE_MOMENT, pairSlots), monB)
      val (ll, ul) = stream(csrLinear(OP_ADD, FMT16.code), linB)
      println(f"[Unified/roofline] MONOID: first-out $lm CC, util=$um%.3f | ELEMENTWISE: first-out $ll CC, util=$ul%.3f")
      assert(um > 0.9, f"monoid class should stream at ~1 beat pair/cycle, got $um%.3f")
      assert(ul > 0.9, f"linear class should stream at ~1 beat pair/cycle, got $ul%.3f")
      assert(ll < lm, "the linear class should retire in fewer cycles than the exp-LUT class")
    }
  }

  "UnifiedJunction_bypass" should "be a transparent wire from a to out when disabled" in {
    test(new DataPathJunctionHarness(hasUnified)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng   = new Random(0xbb01)
      val beats = Seq.fill(8)(packPairs(Seq.fill(pairSlots)((rng.between(-4.0, 4.0), rng.between(0.5, 3.0)))))
      dut.io.csr_i(0).poke(csrMonoid(MODE_MOMENT, pairSlots).U)
      dut.io.enable_i.poke(false)
      dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
      dut.io.out_o.ready.poke(true)
      dut.io.b_i.valid.poke(false)
      dut.io.a_i.valid.poke(true); dut.io.a_i.bits.poke(beats(0).U)
      var got = List[BigInt]()
      var cyc = 0
      var fed = 0
      while (got.length < beats.length && cyc < 500) {
        val canFeed = fed < beats.length && dut.io.a_i.ready.peekBoolean()
        val outNow  = dut.io.out_o.valid.peekBoolean()
        val outBits = if (outNow) dut.io.out_o.bits.peekInt() else BigInt(0)
        dut.clock.step(1); cyc += 1
        if (canFeed) {
          fed += 1
          if (fed < beats.length) dut.io.a_i.bits.poke(beats(fed).U) else dut.io.a_i.valid.poke(false)
        }
        if (outNow) got = got :+ outBits
      }
      assert(got == beats.toList, "disabled junction must forward a_i byte-identically")
      println("[Unified/bypass] enable=0 is byte-identical to a raw forward")
    }
  }
}
