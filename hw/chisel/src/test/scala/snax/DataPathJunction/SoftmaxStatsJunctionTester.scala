package snax.DataPathJunction

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathJunction.JunctionTestUtils._
import snax.utils.TestFp16

/** \============================================================================================================
  * `SoftmaxStatsJunction` -- the fixed-function (m, l) merge, held to the general operator and to the algebra.
  * \============================================================================================================
  *
  * The unit reads what the SIMD engine leaves in the scratchpad: a group is a key beat and a value beat of 32 FP16
  * rows. It folds with the same exponential and the same FMA as `MonoidJunction`, rounds the value to FP16 once, and
  * keeps the winning key exactly. So the checks are:
  *
  *   - BIT-EXACT against `MonoidJunction` (the FP32 build, whose own tests pin it to double-precision goldens): the
  *     same FP16 inputs folded there in FP32 and rounded to FP16 must be the unit's output, bit for bit;
  *   - the `scale` CSR against a double-precision golden, with a control that the golden WITHOUT the scale is wrong;
  *   - short groups (rows at or above nValid leave as the identity), identity substitution of either operand, and a
  *     four-cluster chain against the global merge;
  *   - `cfgerr`, the rate and the latency, and that `rowsPerCycle` changes the cycle count and nothing else.
  *
  * Two `test(...)` builds in one scalatest case collide on the Verilator build directory, so the reference and each
  * instance under test are separate cases that share their results through the class.
  */
class SoftmaxStatsJunctionTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  private val ROWS   = 32
  private val ID_K   = 0xfbff // the lowest finite FP16
  private val TOL    = 6e-4   // 2^-11 for the FP16 rounding, plus the 128-entry exponential
  private val F32ONE = BigInt(0x3f800000L)

  private def enc(d: Double): Int    = TestFp16.enc(d) & 0xffff
  private def dec(b: Int):    Double = TestFp16.dec(b).toDouble
  private def beat(rows: Seq[Int]): BigInt =
    rows.zipWithIndex.map { case (x, i) => BigInt(x) << (16 * i) }.sum
  private def rowsOf(b: BigInt): Seq[Int] = (0 until ROWS).map(i => ((b >> (16 * i)) & 0xffff).toInt)

  /** a group's two operand partials: keys and values, one FP16 number per row */
  private case class Part(k: Seq[Int], v: Seq[Int]) {
    def beats: Seq[BigInt] = Seq(beat(k), beat(v))
  }
  private def identityPart = Part(Seq.fill(ROWS)(ID_K), Seq.fill(ROWS)(ID_K)) // what the socket substitutes

  private def randomPart(rng: Random): Part =
    Part(Seq.fill(ROWS)(enc(rng.nextGaussian() * 3.0)), Seq.fill(ROWS)(enc(rng.between(1.0, 600.0))))

  /** random operands with the corner rows the arithmetic has to get right */
  private def cornerPairs(rng: Random): Seq[(Part, Part)] = {
    val plain = Seq.fill(12)((randomPart(rng), randomPart(rng)))
    // row 0: equal keys; row 1: a gap so wide the loser is wiped out; row 2: a zero loser; row 3: identity against
    // a real row; row 4: identity against identity; row 5: a key gap of about one nat
    val a = randomPart(rng); val b = randomPart(rng)
    def put(p: Part, row: Int, k: Int, v: Int) = Part(p.k.updated(row, k), p.v.updated(row, v))
    var pa = a; var pb = b
    pa = put(pa, 0, enc(2.0), enc(10.0));  pb = put(pb, 0, enc(2.0), enc(30.0))
    pa = put(pa, 1, enc(-2000.0), enc(7.0)); pb = put(pb, 1, enc(3.0), enc(9.0))
    pa = put(pa, 2, enc(5.0), enc(40.0));  pb = put(pb, 2, enc(1.0), 0)
    pa = put(pa, 3, ID_K, 0);              pb = put(pb, 3, enc(-4.0), enc(12.0))
    pa = put(pa, 4, ID_K, 0);              pb = put(pb, 4, ID_K, 0)
    pa = put(pa, 5, enc(1.0), enc(100.0)); pb = put(pb, 5, enc(0.0), enc(20.0))
    plain :+ ((pa, pb))
  }

  /** the exact answer for one row, in double precision, from the FP16 inputs */
  private def golden(ka: Int, kb: Int, la: Int, lb: Int, scale: Double): (Int, Double) = {
    val a = dec(ka); val b = dec(kb); val m = math.max(a, b)
    (if (a >= b) ka else kb, math.exp(scale * (a - m)) * dec(la) + math.exp(scale * (b - m)) * dec(lb))
  }

  /** Feed `aBeats` and `bBeats` (B after `bDelay` cycles), collect `aBeats.length` output beats with the sink always
    * ready. Returns the beats, the cycle of the first one, and the cycle after the last one, counted from the start.
    */
  private def drive(
    dut:    DataPathJunctionHarness,
    csr0:   Int,
    csr1:   BigInt,
    aBeats: Seq[BigInt],
    bBeats: Seq[BigInt],
    bDelay: Int = 0
  ): (Seq[BigInt], Int, Int) = {
    dut.io.csr_i(0).poke(csr0.U); dut.io.csr_i(1).poke(csr1.U)
    dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    def feed(io: chisel3.util.DecoupledIO[UInt], beats: Seq[BigInt], delay: Int): Unit = {
      if (delay > 0) dut.clock.step(delay)
      for (bt <- beats) {
        io.bits.poke(bt.U); io.valid.poke(true)
        while (!io.ready.peekBoolean()) dut.clock.step(1)
        dut.clock.step(1)
      }
      io.valid.poke(false)
    }
    val out  = scala.collection.mutable.ArrayBuffer[BigInt]()
    var cyc  = 0
    var first = -1
    var th   = new chiseltest.internal.TesterThreadList(Seq())
    th = th.fork { feed(dut.io.a_i, aBeats, 0) }
    th = th.fork { feed(dut.io.b_i, bBeats, bDelay) }
    th = th.fork {
      dut.io.out_o.ready.poke(true)
      for (_ <- aBeats.indices) {
        while (!dut.io.out_o.valid.peekBoolean()) { dut.clock.step(1); cyc += 1 }
        if (first < 0) first = cyc
        out += dut.io.out_o.bits.peekInt()
        dut.clock.step(1); cyc += 1
      }
    }
    th.joinAndStep()
    dut.io.out_o.ready.poke(true)
    var w = 0; while (dut.io.busy_o.peekBoolean() && w < 400) { dut.clock.step(1); w += 1 }
    assert(!dut.io.busy_o.peekBoolean(), "the junction must go idle once the group has left")
    dut.io.out_o.ready.poke(false)
    (out.toSeq, first, cyc)
  }

  private def foldGroup(dut: DataPathJunctionHarness, nValid: Int, csr1: BigInt, a: Part, b: Part, bDelay: Int = 0): Part = {
    val (o, _, _) = drive(dut, nValid, csr1, a.beats, b.beats, bDelay)
    Part(rowsOf(o(0)), rowsOf(o(1)))
  }

  private def relErr(got: Double, want: Double): Double =
    if (want == 0.0) math.abs(got) else math.abs(got - want) / math.abs(want)

  // ---------------------------------------------------------------------------------------------------------------
  // the reference: the FP32 MonoidJunction folds the same FP16 inputs, eight rows at a time
  private val refRng   = new Random(0x57a7)
  private val refPairs = cornerPairs(refRng)
  private var refOut: Seq[Part] = Seq()

  "SoftmaxStats_reference" should "fold the corner groups on MonoidJunction in FP32 and round them to FP16" in {
    test(new DataPathJunctionHarness(new HasMonoidJunction(hasScan = false)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val word = BigInt(0x0c043108L) // (m, l), 8 slots, FP32
        refOut = refPairs.map { case (a, b) =>
          val k = Array.fill(ROWS)(0); val v = Array.fill(ROWS)(0)
          for (g <- 0 until ROWS / 8) {
            def beat32(p: Part): BigInt = {
              var x = BigInt(0)
              for (j <- 0 until 8) { x |= f32(dec(p.k(8 * g + j))) << (32 * j); x |= f32(dec(p.v(8 * g + j))) << (32 * (8 + j)) }
              x
            }
            val out = runPair(dut, word, beat32(a), beat32(b))
            for (j <- 0 until 8) { k(8 * g + j) = enc(laneF32(out, j)); v(8 * g + j) = enc(laneF32(out, 8 + j)) }
          }
          Part(k.toSeq, v.toSeq)
        }
        println(s"[SoftmaxStats] reference: ${refOut.length} groups of $ROWS rows folded by MonoidJunction")
      }
  }

  private var r4Out: Seq[Part] = Seq()
  private var r4Time = (0, 0) // (first-out, total) of one full group

  "SoftmaxStats_r4" should "match the general operator bit for bit and keep the algebra, the identity and the rate" in {
    assert(refOut.nonEmpty, "run SoftmaxStats_reference first")
    test(new DataPathJunctionHarness(new HasSoftmaxStatsJunction(rowsPerCycle = 4)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0xbeef)

        // ---- 1. bit-exact against the general operator, scale CSR left at zero (= 1.0) ----------------------
        val got = refPairs.map { case (a, b) => foldGroup(dut, ROWS, 0, a, b) }
        r4Out = got
        for ((((a, b), g), r) <- refPairs.zip(got).zip(refOut); row <- 0 until ROWS) {
          assert(g.k(row) == r.k(row), f"key row $row: got 0x${g.k(row)}%04x, MonoidJunction 0x${r.k(row)}%04x")
          assert(g.v(row) == r.v(row), f"value row $row: got 0x${g.v(row)}%04x, MonoidJunction 0x${r.v(row)}%04x")
        }
        println(s"[SoftmaxStats] ${got.length} groups x $ROWS rows are bit-identical to MonoidJunction folded in FP32")

        // ---- 2. the scale CSR against a double-precision golden, and the control that it matters -------------
        val scale     = 0.5
        var worst     = 0.0
        var worstNone = 0.0
        for (_ <- 0 until 10) {
          val a = randomPart(rng); val b = randomPart(rng)
          val g = foldGroup(dut, ROWS, f32(scale), a, b)
          for (row <- 0 until ROWS) {
            val (kw, lw)  = golden(a.k(row), b.k(row), a.v(row), b.v(row), scale)
            val (_, lw1)  = golden(a.k(row), b.k(row), a.v(row), b.v(row), 1.0)
            assert(g.k(row) == kw, f"key row $row")
            worst     = math.max(worst, relErr(dec(g.v(row)), lw))
            worstNone = math.max(worstNone, relErr(dec(g.v(row)), lw1))
          }
        }
        assert(worst <= TOL, f"scale 0.5: worst relative error $worst%.2e exceeds $TOL%.0e")
        assert(worstNone > 20 * TOL, f"control: the unscaled golden agrees to $worstNone%.2e, so the test cannot see the scale")
        println(f"[SoftmaxStats] scale 0.5: worst error $worst%.2e of the golden; against the unscaled golden $worstNone%.2e (control)")

        // ---- 3. short groups: rows at or above nValid leave as the identity -------------------------------------
        for (nv <- Seq(1, 3, 4, 5, 17, 31, 32)) {
          val a = randomPart(rng); val b = randomPart(rng)
          val g = foldGroup(dut, nv, 0, a, b)
          for (row <- 0 until ROWS)
            if (row < nv) {
              val (kw, lw) = golden(a.k(row), b.k(row), a.v(row), b.v(row), 1.0)
              assert(g.k(row) == kw, s"nValid $nv key row $row")
              assert(relErr(dec(g.v(row)), lw) <= TOL, s"nValid $nv value row $row")
            } else {
              assert(g.k(row) == ID_K && g.v(row) == 0, f"nValid $nv: row $row must be the identity, got 0x${g.k(row)}%04x / 0x${g.v(row)}%04x")
            }
        }
        println("[SoftmaxStats] nValid 1..32: live rows fold, the others leave as the identity (0xFBFF, 0)")

        // ---- 4. the published identity beat, substituted for either operand, changes nothing ---------------------
        assert(dut.io.identity_o.peekInt() == beat(Seq.fill(ROWS)(ID_K)), "the published identity beat")
        for (_ <- 0 until 4) {
          val b  = randomPart(rng)
          val ga = foldGroup(dut, ROWS, 0, identityPart, b)
          assert(ga.k == b.k && ga.v == b.v, "identity as operand A must return B exactly")
          val gb = foldGroup(dut, ROWS, 0, b, identityPart)
          assert(gb.k == b.k && gb.v == b.v, "identity as operand B must return A exactly")
        }
        println("[SoftmaxStats] an all-0xFBFF operand, in either role, returns the other operand bit for bit")

        // ---- 5. four clusters folded hop by hop against the global merge -----------------------------------------
        var worstChain = 0.0
        for (_ <- 0 until 6) {
          val parts = Seq.fill(4)(randomPart(rng))
          val acc   = parts.tail.foldLeft(parts.head)((x, y) => foldGroup(dut, ROWS, f32(0.5), x, y))
          for (row <- 0 until ROWS) {
            val ms   = parts.map(p => dec(p.k(row))).max
            val want = parts.map(p => math.exp(0.5 * (dec(p.k(row)) - ms)) * dec(p.v(row))).sum
            assert(acc.k(row) == parts.map(_.k(row)).maxBy(dec), s"chain key row $row")
            worstChain = math.max(worstChain, relErr(dec(acc.v(row)), want))
          }
        }
        assert(worstChain <= 3 * TOL, f"4-cluster chain: worst relative error $worstChain%.2e")
        println(f"[SoftmaxStats] 4 clusters, 3 folds: worst error $worstChain%.2e of the global merge (FP16 rounding at every hop)")

        // ---- 6. cfgerr --------------------------------------------------------------------------------------------
        def verdict(c0: Int, c1: BigInt): Boolean = {
          dut.io.csr_i(0).poke(c0.U); dut.io.csr_i(1).poke(c1.U); dut.io.enable_i.poke(true); dut.clock.step(2)
          dut.io.cfgerr_o.peekBoolean()
        }
        assert(!verdict(32, 0) && !verdict(1, f32(0.5)) && !verdict(8, F32ONE), "legal words must not be flagged")
        assert(verdict(0, 0) && verdict(33, 0) && verdict(255, 0), "nValid 0 or above 32 must be reported")
        assert(verdict(8, BigInt(0xbf800000L)), "a negative scale must be reported")
        assert(verdict(8, BigInt(0x7f800000L)) && verdict(8, BigInt(0x7fc00000L)), "an infinite or NaN scale must be reported")
        println("[SoftmaxStats] cfgerr: nValid 0 / > 32 and a negative, infinite or NaN scale are reported")

        // ---- 7. a stream of groups with skew between the operands, then the timing --------------------------------
        val groups = Seq.fill(5)((randomPart(rng), randomPart(rng)))
        val (sOut, _, sTot) = drive(dut, ROWS, 0, groups.flatMap(_._1.beats), groups.flatMap(_._2.beats), bDelay = 3)
        for (((a, b), gi) <- groups.zipWithIndex; row <- 0 until ROWS) {
          val (kw, lw) = golden(a.k(row), b.k(row), a.v(row), b.v(row), 1.0)
          assert(rowsOf(sOut(2 * gi))(row) == kw, s"stream group $gi key row $row")
          assert(relErr(dec(rowsOf(sOut(2 * gi + 1))(row)), lw) <= TOL, s"stream group $gi value row $row")
        }
        val (_, f32rows, t32) = drive(dut, ROWS, 0, groups.head._1.beats, groups.head._2.beats)
        val (_, f1row, t1)    = drive(dut, 1, 0, groups.head._1.beats, groups.head._2.beats)
        r4Time = (f32rows, t32)
        assert(f1row + (ROWS / 4 - 1) <= f32rows, s"one row ($f1row cycles) must finish ${ROWS / 4 - 1} chunks sooner than 32 rows ($f32rows)")
        println(s"[SoftmaxStats] 5 groups streamed with skew, all correct (${sTot} cycles); first output after $f32rows cycles for 32 rows, $f1row for 1 row")
      }
  }

  private def checkAgainstR4(tag: String, rpc: Int, slowerThan: Option[Int]) = {
    assert(r4Out.nonEmpty, "run SoftmaxStats_r4 first")
    test(new DataPathJunctionHarness(new HasSoftmaxStatsJunction(rowsPerCycle = rpc)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val got = refPairs.map { case (a, b) => foldGroup(dut, ROWS, 0, a, b) }
        for (((g, r), gi) <- got.zip(r4Out).zipWithIndex) assert(g == r, s"$tag: group $gi differs from the 4-row instance")
        val (_, first, total) = drive(dut, ROWS, 0, refPairs.head._1.beats, refPairs.head._2.beats)
        slowerThan.foreach(t => assert(if (rpc < 4) first > t else first < t,
          s"$tag: first output after $first cycles, against ${t} for 4 rows a cycle"))
        println(s"[SoftmaxStats] rowsPerCycle $rpc: same beats as 4 rows a cycle; first output after $first cycles, $total in all")
      }
  }

  "SoftmaxStats_r1" should "give the same beats one row a cycle, only later" in {
    checkAgainstR4("r1", 1, Some(r4Time._1))
  }

  "SoftmaxStats_r8" should "give the same beats eight rows a cycle, only sooner" in {
    checkAgainstR4("r8", 8, Some(r4Time._1))
  }
}
