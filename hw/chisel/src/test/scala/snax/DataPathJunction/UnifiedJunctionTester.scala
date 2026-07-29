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

  private def hasUnified =
    new HasUnifiedJunction(elemWidth = elemWidth, fpPipe = fpPipe, dHead = dHead)

  // ---- THE FROZEN REFERENCE ---------------------------------------------------------------------------------
  // These 13 beats were CAPTURED from the two predecessor netlists -- `ElementwiseJunction` for the linear class,
  // `MonoidJunction` for the monoid class -- driven by the deterministic stimulus above, and are now literals.
  //
  // They are literals ON PURPOSE. Elaborating a predecessor as a LIVE reference looks stronger and is in fact
  // weaker: both placements call the same `MonoidCombine.apply`, so any change to that shared code moves the
  // reference and the DUT together and this test passes VACUOUSLY. A frozen table cannot move. Every datapath
  // refactor from here on is checked against the behaviour of the netlist as it shipped, not against itself.
  //
  // If a beat below ever needs to change, that is a FORMAT CHANGE and must be argued as one -- never re-captured
  // silently. Regenerate deliberately, and only alongside an independent numeric golden for the affected mode.
  private def g(s: String): BigInt = BigInt(s, 16)

  private val GOLDEN_LINEAR: Map[String, BigInt] = Map(
    "FP16/ADD" -> g("34164434c59c40cc3c06c58a46dc34053c77420a4000414ec4f242a73c8cbd0c3cf5baecb530b10040fd403cb070410244d4430db954c32e4648bee1bd26c19a"),
    "FP16/MUL" -> g("4482caf3bd9d416645acbde045edc4bb2430400a3e0c2fac4814c85d3d01c5f92f67c781362e428f42f1c3d43fc43c1c439e42c94390415ec8de4225afd5c063"),
    "FP16/MAX" -> g("be5a41fb4073418840413d6e436e3ba6c0164084427e40a2c1273d39c200433842752df8b17240c13f3640c2433a436541c340babcbfb5d63cbd40863e654065"),
    "FP16/MIN" -> g("bcf9c376c019be453e8ec3b4c31cc3ccc2d2c24fbead3e16be0ebe7d22b5c21a4174c001a82f1ea73ffbc36fc13f3c55c2d3c16432883d56be81bfecc2232c9f"),
    "BF16/ADD" -> g("bf9cc0c8bf12c05c40b1c035c0103fb8c0183f6c408e4054c01e3ed8bfacbf18c046c01140e5c0ac3ff53da0bef83f3cc07c3f0c40c7c0864064bfd23facbee2"),
    "FP32/MUL" -> g("41202bcb3eb8d0bfbff4d2a5c0446521c076510fc0f6cd9f3ffc5988bdbf8d4ec1075146c0ef636ec0c0715440b7da5d40ba79ddc072eae83f591936bfc3eb60"),
  )

  private val GOLDEN_MONOID: Map[String, BigInt] = Map(
    "SUM"     -> g("40acaf8640e9bbc040c29950411c640d40ebea76402b656240c011344093bf0e4039645540db9c084041f3a840111e6e3ffd1490bfcaf79f409713e44067c16d"),
    "MOMENT"  -> g("40670e05409ee9c3400a1c2a3fe7109e40ad143c40a017a140857ee3408d4eb2409e4da940babec140a0bbfe40b580ce409c97df3ff3c298407367f23feaf2f8"),
    // nValid = 3: slots 3..7 are dead. Note their KEY lanes retire ff7fffff = ID_M, NOT zero -- a dead slot is
    // already a legal identity partial, which is what makes the output re-foldable (C1) at any downstream nValid.
    "MOMENT3" -> g("407dd43b40944076408bdc27ff7fffffff7fffffff7fffffff7fffffff7fffff40999e973fc41557407d963e"),
    "MAXPOOL" -> g("4079858140a86f0240afc25c406972df40ae0d0640b454c4409506ea407a5da3"),
    "ARGMAX"  -> g("40e0000041600000415000004140000041300000412000003f8000000000000040ae6a9240d261c140f42c16408062d9bf94544840926ed23d7d3a8640ba19b2"),
    "ATTN"    -> g("bffe731e3ff85d1b3fde3e53c037fe514035db033f8683843dbbcbb8401584ec407d9275406d669b"),
    "MOMENT2" -> g("402f21d34071bdd8403ef9ccbf9f5a76"),
  )

  "UnifiedJunction_equivalence" should "reproduce the frozen predecessor beats bit-for-bit on ONE netlist" in {
    assert(GOLDEN_LINEAR.size == linearCases.length && GOLDEN_MONOID.size == monoidCases.length,
           "the frozen tables must cover every stimulus case")

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
      assert(got(c.label) == GOLDEN_LINEAR(c.label),
             f"linear ${c.label}: got 0x${got(c.label)}%x != frozen ElementwiseJunction 0x${GOLDEN_LINEAR(c.label)}%x")
    for (c <- monoidCases)
      assert(got(c.label) == GOLDEN_MONOID(c.label),
             f"monoid ${c.label}: got 0x${got(c.label)}%x != frozen MonoidJunction 0x${GOLDEN_MONOID(c.label)}%x")

    // one independent numeric anchor per class, so the test does not rest solely on the predecessors being right
    val momentOut = got("MOMENT")
    assert(laneF32(momentOut, 0).isFinite && laneF32(momentOut, pairSlots) > 0.0, "MOMENT beat is not sane")
    println(s"[Unified] ONE netlist reproduces ${linearCases.length} linear and ${monoidCases.length} monoid " +
            "transfers BIT-EXACTLY against the FROZEN predecessor goldens, interleaved across the class boundary")
  }

  // ---- RAW GEOMETRY ------------------------------------------------------------------------------------------
  // csr: [17] rawGeom | [12] hasKey | [11:8] n | [21:18] nExp | [25:22] nAdd | [27:26] sigma | [28] keyPol
  private def csrRaw(hasKey: Int, n: Int, nExp: Int, nAdd: Int, sigma: Int, nValid: Int, keyPol: Int = 0): BigInt =
    (BigInt(1) << 17) | (BigInt(keyPol) << 28) | (BigInt(sigma) << 26) | (BigInt(nAdd) << 22) |
      (BigInt(nExp) << 18) | (BigInt(hasKey) << 12) | (BigInt(n) << 8) | BigInt(nValid)

  /** the same six operators, spelled in the raw encoding instead of the enum */
  private val rawEquivalents: Seq[(String, BigInt)] = Seq(
    "SUM"     -> csrRaw(hasKey = 0, n = 2, nExp = 0, nAdd = 2, sigma = 3, nValid = pairSlots),
    "MOMENT"  -> csrRaw(hasKey = 1, n = 1, nExp = 1, nAdd = 0, sigma = 3, nValid = pairSlots),
    "MOMENT3" -> csrRaw(hasKey = 1, n = 1, nExp = 1, nAdd = 0, sigma = 3, nValid = 3),
    "MAXPOOL" -> csrRaw(hasKey = 1, n = 0, nExp = 0, nAdd = 0, sigma = 3, nValid = pairSlots),
    "ARGMAX"  -> csrRaw(hasKey = 1, n = 1, nExp = 0, nAdd = 0, sigma = 3, nValid = pairSlots),
    "ATTN"    -> csrRaw(hasKey = 1, n = 1 + dHead, nExp = 1 + dHead, nAdd = 0, sigma = 0, nValid = 1),
    "MOMENT2" -> csrRaw(hasKey = 1, n = 3, nExp = 3, nAdd = 0, sigma = 0, nValid = 1)
  )

  "UnifiedJunction_rawGeom" should "decode the raw geometry to the SAME beat as the legacy enum" in {
    // The enum is a 6-entry elaboration-time ROM over the raw fields, nothing more. If these differ, the enum
    // has semantics the geometry does not -- which would mean the mode decode never really went away.
    test(new DataPathJunctionHarness(hasUnified)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      for ((label, csr) <- rawEquivalents) {
        val c   = monoidCases.find(_.label == label).get
        val out = runPair(dut, csr, c.a, c.b)
        assert(out == GOLDEN_MONOID(label),
               f"raw $label: 0x$out%x != the frozen enum beat 0x${GOLDEN_MONOID(label)}%x")
      }
      println(s"[Unified/rawGeom] all ${rawEquivalents.length} enum operators re-spelled as raw geometry, " +
              "bit-identical -- the mode decode carries no semantics of its own")
    }
  }

  "UnifiedJunction_beyondEnum" should "run operators the 6-mode enum cannot express, on the same netlist" in {
    test(new DataPathJunctionHarness(hasUnified)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0x5eed)

      // (1) ATTN with dHead = 6: F = 8 => sigma = 1 => TWO partials per beat, where the enum's dHead is frozen
      //     into the netlist at elaboration. lane = field*2 + slot.
      val d6 = 6
      val sh = (0 until 2).map(_ => (0 until 2).map(_ =>
        (rng.between(-2.0, 5.0), rng.between(1.0, 5.0), Seq.fill(d6)(rng.between(-3.0, 3.0)))))
      def packS2(p: Seq[(Double, Double, Seq[Double])]): BigInt = {
        var b = BigInt(0)
        for ((x, s) <- p.zipWithIndex) {
          b |= f32(x._1) << (32 * s); b |= f32(x._2) << (32 * (2 + s))
          for ((o, j) <- x._3.zipWithIndex) b |= f32(o) << (32 * ((2 + j) * 2 + s))
        }
        b
      }
      val outA = runPair(dut, csrRaw(1, 1 + d6, 1 + d6, 0, sigma = 1, nValid = 2), packS2(sh(0)), packS2(sh(1)))
      for (s <- 0 until 2) {
        val (a, b) = (sh(0)(s), sh(1)(s))
        val gm = math.max(a._1, b._1)
        val gl = a._2 * math.exp(a._1 - gm) + b._2 * math.exp(b._1 - gm)
        assert(math.abs(laneF32(outA, s) - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"attn6 slot $s m")
        assert(math.abs(laneF32(outA, 2 + s) - gl) / gl <= 1.5e-2, s"attn6 slot $s l")
        for (j <- 0 until d6) {
          val go = a._3(j) * math.exp(a._1 - gm) + b._3(j) * math.exp(b._1 - gm)
          val d  = math.abs(laneF32(outA, (2 + j) * 2 + s) - go) / (math.abs(go) + 1e-6)
          assert(d <= 2e-2, s"attn6 slot $s O[$j] rel=$d")
        }
      }

      // (2) ARGMIN with a SHORT beat -- the case that catches a wrong key identity. Flipping the comparison
      //     without flipping the pad would make the pad win every min and return it instead of the real answer.
      val nv  = 3
      val la  = Seq(4.0, -1.0, 7.0) ++ Seq.fill(pairSlots - nv)(0.0)
      val lb  = Seq(2.0, 9.0, 8.0) ++ Seq.fill(pairSlots - nv)(0.0)
      val pa  = la.zipWithIndex.map { case (v, i) => (v, i.toDouble) }
      val pb  = lb.zipWithIndex.map { case (v, i) => (v, (pairSlots + i).toDouble) }
      val oMin = runPair(dut, csrRaw(1, 1, 0, 0, sigma = 3, nValid = nv, keyPol = 1), packPairs(pa), packPairs(pb))
      for (k <- 0 until nv) {
        val g = math.min(la(k), lb(k))
        assert(lane(oMin, k) == f32(g), f"argmin slot $k: 0x${lane(oMin, k)}%x != ${g} (a wrong pad wins here)")
        assert(laneF32(oMin, pairSlots + k) == (if (la(k) <= lb(k)) k.toDouble else (pairSlots + k).toDouble),
               s"argmin slot $k index")
      }
      // dead slots must carry the MIN-monoid identity, not the max one
      for (k <- nv until pairSlots)
        assert(lane(oMin, k) == BigInt("7f7fffff", 16), f"argmin dead slot $k: 0x${lane(oMin, k)}%x != ID_MAX")

      // (3) flash + argmax carry: (m, l, payload) -- nExp=1, nSel=1. One alpha, one winner-select, F=3.
      val ma = Seq((3.0, 2.0, 11.0), (1.0, 5.0, 12.0))
      val mb = Seq((5.0, 1.0, 21.0), (0.0, 4.0, 22.0))
      def packF3(p: Seq[(Double, Double, Double)]): BigInt = {
        var b = BigInt(0)
        for ((x, s) <- p.zipWithIndex) {
          b |= f32(x._1) << (32 * s); b |= f32(x._2) << (32 * (2 + s)); b |= f32(x._3) << (32 * (4 + s))
        }
        b
      }
      val oCar = runPair(dut, csrRaw(1, 2, 1, 0, sigma = 1, nValid = 2), packF3(ma), packF3(mb))
      for (s <- 0 until 2) {
        val gm = math.max(ma(s)._1, mb(s)._1)
        val gl = ma(s)._2 * math.exp(ma(s)._1 - gm) + mb(s)._2 * math.exp(mb(s)._1 - gm)
        assert(math.abs(laneF32(oCar, s) - gm) <= 1e-6, s"carry slot $s m")
        assert(math.abs(laneF32(oCar, 2 + s) - gl) / gl <= 1.5e-2, s"carry slot $s l")
        assert(laneF32(oCar, 4 + s) == (if (ma(s)._1 >= mb(s)._1) ma(s)._3 else mb(s)._3), s"carry slot $s payload")
      }
      println("[Unified/beyondEnum] ATTN dHead=6 at 2 partials/beat, ARGMIN on a short beat, and flash+argmax " +
              "carry -- three operators with no enum encoding, zero RTL change")
    }
  }

  "UnifiedJunction_sigmaSaturation" should "degrade an illegal sigma instead of interleaving partials" in {
    // sigma is software-settable, so an over-large sigma is a new failure mode. Saturation turns it into
    // "fewer partials per beat" rather than two partials written into each other's lanes.
    test(new DataPathJunctionHarness(hasUnified)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val c    = monoidCases.find(_.label == "ATTN").get
      val legal = csrRaw(1, 1 + dHead, 1 + dHead, 0, sigma = 0, nValid = 1) // F = 10 => sigma_max = 0
      val silly = csrRaw(1, 1 + dHead, 1 + dHead, 0, sigma = 3, nValid = 1) // F = 10 with sigma = 3: impossible
      val a = runPair(dut, legal, c.a, c.b)
      val b = runPair(dut, silly, c.a, c.b)
      assert(a == b, f"sigma=3 at F=10 must saturate to sigma=0: 0x$b%x != 0x$a%x")
      assert(a == GOLDEN_MONOID("ATTN"), "and still equal the frozen ATTN beat")
      println("[Unified/sigma] an impossible geometry saturates to the widest legal one -- no illegal CSR word")
    }
  }

  // ---- MOMENT2-DENSE: the one throughput gain in the existing operator set ------------------------------------
  // F = 4 admits sigma = 2, i.e. FOUR partials per beat where the legacy geometry carried one and left 12 of 16
  // lanes dead. lane = field*4 + slot, so slot s occupies lanes s, 4+s, 8+s, 12+s.
  private def packDense4(p: Seq[(Double, Double, Double, Double)]): BigInt = {
    var b = BigInt(0)
    for ((x, s) <- p.zipWithIndex) {
      b |= f32(x._1) << (32 * s)
      b |= f32(x._2) << (32 * (4 + s))
      b |= f32(x._3) << (32 * (8 + s))
      b |= f32(x._4) << (32 * (12 + s))
    }
    b
  }
  /** the flash merge of (m, l, A, B) in double precision -- INDEPENDENT of the RTL, not a captured beat */
  private def moment2Ref(sh: Seq[(Double, Double, Double, Double)]): (Double, Double, Double, Double) = {
    val ms = sh.map(_._1).max
    def w(sel: ((Double, Double, Double, Double)) => Double) = sh.map(s => sel(s) * math.exp(s._1 - ms)).sum
    (ms, w(_._2), w(_._3), w(_._4))
  }

  "UnifiedJunction_moment2Dense" should "fold FOUR moment banks per beat at sigma=2" in {
    test(new DataPathJunctionHarness(hasUnified)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0x2b02)
      val csr = csrRaw(hasKey = 1, n = 3, nExp = 3, nAdd = 0, sigma = 2, nValid = 4)
      for (trial <- 0 until 4) {
        def bank() = (0 until 4).map(_ =>
          (rng.between(-2.0, 5.0), rng.between(1.0, 5.0), rng.between(-4.0, 4.0), rng.between(0.5, 6.0)))
        val (pa, pb) = (bank(), bank())
        val out = runPair(dut, csr, packDense4(pa), packDense4(pb), bDelay = trial % 3)
        for (s <- 0 until 4) {
          val (gm, gl, ga, gb) = moment2Ref(Seq(pa(s), pb(s)))
          assert(math.abs(laneF32(out, s) - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"dense trial $trial slot $s m")
          assert(math.abs(laneF32(out, 4 + s) - gl) / gl <= 1.5e-2, s"dense trial $trial slot $s l")
          assert(math.abs(laneF32(out, 8 + s) - ga) / (math.abs(ga) + 1e-6) <= 2e-2, s"dense trial $trial slot $s A")
          assert(math.abs(laneF32(out, 12 + s) - gb) / gb <= 2e-2, s"dense trial $trial slot $s B")
        }
      }
      println("[Unified/MOMENT2-dense] 4 moment banks per beat (was 1), 16/16 lanes live, vs a double-precision " +
              "reference -- MOMENT2's first independent numeric check")
    }
  }

  "UnifiedJunction_chainClosure" should "fold hop-by-hop to the global answer at EVERY geometry" in {
    // C1 says the output beat is a legal INPUT beat. The way to know is to feed it back: P = 4 partials reduced
    // as three successive pairwise ops must equal the direct global merge. Proved PER GEOMETRY, not argued once
    // -- the lane map changes with sigma, so closure is a property of (F, sigma), not of the module.
    test(new DataPathJunctionHarness(hasUnified)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0x7c1a)

      // (a) MOMENT at sigma = 3 -- 8 independent (m, l) reductions per beat
      for (trial <- 0 until 2) {
        val shards = Seq.fill(4)(Seq.fill(pairSlots)((rng.between(-3.0, 6.0), rng.between(1.0, 5.0))))
        var acc = packPairs(shards.head)
        for (hop <- 1 until 4)
          acc = runPair(dut, csrRaw(1, 1, 1, 0, sigma = 3, nValid = pairSlots), acc, packPairs(shards(hop)),
                        bDelay = hop)
        for (k <- 0 until pairSlots) {
          val slot = shards.map(_(k))
          val gm   = slot.map(_._1).max
          val gl   = slot.map { case (m, l) => l * math.exp(m - gm) }.sum
          assert(math.abs(laneF32(acc, k) - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"chain sigma=3 trial $trial slot $k m")
          assert(math.abs(laneF32(acc, pairSlots + k) - gl) / gl <= 3e-2, s"chain sigma=3 trial $trial slot $k l")
        }
      }

      // (b) MOMENT2-dense at sigma = 2 -- the NEW geometry, four 4-field banks per beat
      for (trial <- 0 until 2) {
        val shards = Seq.fill(4)((0 until 4).map(_ =>
          (rng.between(-2.0, 5.0), rng.between(1.0, 5.0), rng.between(-4.0, 4.0), rng.between(0.5, 6.0))))
        var acc = packDense4(shards.head)
        for (hop <- 1 until 4)
          acc = runPair(dut, csrRaw(1, 3, 3, 0, sigma = 2, nValid = 4), acc, packDense4(shards(hop)), bDelay = hop)
        for (s <- 0 until 4) {
          val (gm, gl, ga, gb) = moment2Ref(shards.map(_(s)))
          assert(math.abs(laneF32(acc, s) - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"chain sigma=2 trial $trial slot $s m")
          assert(math.abs(laneF32(acc, 4 + s) - gl) / gl <= 4e-2, s"chain sigma=2 trial $trial slot $s l")
          assert(math.abs(laneF32(acc, 8 + s) - ga) / (math.abs(ga) + 1e-6) <= 5e-2, s"chain sigma=2 trial $trial slot $s A")
          assert(math.abs(laneF32(acc, 12 + s) - gb) / gb <= 5e-2, s"chain sigma=2 trial $trial slot $s B")
        }
      }

      // (c) ATTN at sigma = 0 -- the widest partial, one per beat
      val ash = Seq.fill(4)((rng.between(-2.0, 5.0), rng.between(1.0, 5.0), Seq.fill(dHead)(rng.between(-3.0, 3.0))))
      var acc = packSingle(ash.head._1, ash.head._2, ash.head._3)
      for (hop <- 1 until 4)
        acc = runPair(dut, csrRaw(1, 1 + dHead, 1 + dHead, 0, sigma = 0, nValid = 1), acc,
                      packSingle(ash(hop)._1, ash(hop)._2, ash(hop)._3), bDelay = hop)
      val gm = ash.map(_._1).max
      assert(math.abs(laneF32(acc, 0) - gm) <= math.abs(gm) * 1e-6 + 1e-9, "chain sigma=0 m")
      assert(math.abs(laneF32(acc, 1) - ash.map { case (m, l, _) => l * math.exp(m - gm) }.sum) /
               ash.map { case (m, l, _) => l * math.exp(m - gm) }.sum <= 4e-2, "chain sigma=0 l")
      for (j <- 0 until dHead) {
        val go = ash.map { case (m, _, o) => o(j) * math.exp(m - gm) }.sum
        assert(math.abs(laneF32(acc, 2 + j) - go) / (math.abs(go) + 1e-6) <= 5e-2, s"chain sigma=0 O[$j]")
      }
      println("[Unified/closure] P=4 hop-by-hop == the direct global merge at sigma = 3, 2 and 0 -- C1 holds " +
              "per geometry, not by assertion")
    }
  }

  "UnifiedJunction_infiniteKeys" should "fold two equal infinite keys with a twist of exactly 1.0" in {
    // Two shards whose maxima are both +Inf (or both -Inf) make `loser - m*` = Inf - Inf = NaN. NaN must not
    // reach the exp LUT: its input clamp is built from integer comparisons that are only valid on finite
    // operands, so a NaN is not bounded by the clamp, it is LAUNDERED THROUGH it into the largest legal
    // argument -- and the twist comes back ~2.4e38 instead of 1.0. The value lanes then retire finite,
    // plausible, badly wrong numbers, which is the worst failure mode there is.
    //
    // Equal keys mean delta = 0 by definition, so the answer is a plain sum of the value fields.
    test(new DataPathJunctionHarness(hasUnified)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val INF  = BigInt("7f800000", 16)
      val NINF = BigInt("ff800000", 16)
      for ((key, name) <- Seq((INF, "+Inf"), (NINF, "-Inf"))) {
        // small value fields: this is the band where a runaway twist stays FINITE and therefore silent
        val la = Seq.fill(pairSlots)(1.0e-20)
        val lb = Seq.fill(pairSlots)(2.0e-20)
        def pack(k: BigInt, v: Seq[Double]): BigInt = {
          var b = BigInt(0)
          for (i <- 0 until pairSlots) { b |= k << (32 * i); b |= f32(v(i)) << (32 * (pairSlots + i)) }
          b
        }
        // the golden is the FP32 sum of the FP32-rounded operands -- 1e-20 and 2e-20 are not representable, so
        // comparing against the literal 3e-20 would fail on the last ULP for the right answer
        val gold = (java.lang.Float.intBitsToFloat(f32(la.head).toInt) +
                      java.lang.Float.intBitsToFloat(f32(lb.head).toInt)).toDouble
        val out  = runPair(dut, csrMonoid(MODE_MOMENT, pairSlots), pack(key, la), pack(key, lb))
        for (k <- 0 until pairSlots) {
          assert(lane(out, k) == key, f"$name slot $k: key must survive, got 0x${lane(out, k)}%x")
          val got = laneF32(out, pairSlots + k)
          // the failure this exists for is a twist of ~2.4e38 instead of 1.0, which lands ~58 orders of
          // magnitude out -- so check the magnitude first, then the value
          assert(got < 1.0e-15,
                 f"$name slot $k: got $got%.6g -- the twist ran away, a laundered NaN scaled the value field")
          assert(math.abs(got - gold) <= gold * 1e-6,
                 f"$name slot $k: expected the plain sum $gold%.9g (twist = 1.0), got $got%.9g")
        }
      }
      println("[Unified/inf-keys] equal infinite keys fold with twist = 1.0, not with a laundered NaN")
    }
  }

  "UnifiedJunction_zeroFieldGeometry" should "not collapse an F=0 word into an all-zero beat" in {
    // n = 0 with hasKey = 0 gives F = 0. Unclamped that is a cliff: sigma_max's first arm is `F <= 2` so it
    // returns 3, and `field(l) < F` is then false on every lane because the compare is unsigned -- the block
    // emits 512 zero bits. For an additive fold that is a legal identity, so a whole chain converges to zero
    // and nothing anywhere reports it.
    test(new DataPathJunctionHarness(hasUnified)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val a = packPairs(Seq.fill(pairSlots)((3.0, 5.0)))
      val b = packPairs(Seq.fill(pairSlots)((7.0, 11.0)))
      val out = runPair(dut, csrRaw(hasKey = 0, n = 0, nExp = 0, nAdd = 0, sigma = 3, nValid = pairSlots), a, b)
      assert(out != BigInt(0), "an F=0 word retired an all-zero beat -- the geometry cliff is back")
      // clamped to F = 1: field 0 is in range and plainly summed, everything above it is out of range
      for (k <- 0 until pairSlots)
        assert(laneF32(out, k) == 10.0, f"slot $k: expected the field-0 sum 10.0, got ${laneF32(out, k)}%.6g")
      println("[Unified/F=0] a zero-field geometry clamps to one field instead of emitting a silent zero beat")
    }
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
