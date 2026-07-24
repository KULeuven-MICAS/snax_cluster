package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** UnifiedMonoidMergeRt: ONE configurable combine cell driven by the 3-bit combineMode CSR field, exercised in
  * all four modes on the SAME netlist:
  *   - MOMENT  : the softmax normalizer (m, ℓ) paired-tree fold == StreamMomentMergeRt
  *   - SUM     : the LayerNorm/RMSNorm (Σx, Σx²) paired-tree fold == StreamNormStatMergeRt
  *   - MAXPOOL : max-reduce over the beat's field0 lanes (degenerate KEY-only)
  *   - ATTN    : the flash-attention (m, ℓ, O) single-partial accEn fold == StreamAttnMergeRt
  * Proves the representation holds -- one datapath computes every combine in the family, mode-selected -- and
  * that the shared chassis streams (roofline util ~ 1.0) in the paired modes.
  */
class UnifiedMonoidMergeRtTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))
  private val dHead = 8

  private val MODE_SUM = 0; private val MODE_MOMENT = 1; private val MODE_ATTN = 2; private val MODE_MAXPOOL = 3

  private def f32(f: Double): BigInt = BigInt(java.lang.Float.floatToIntBits(f.toFloat).toLong & 0xffffffffL)
  private def dec(b: BigInt): Double = java.lang.Float.intBitsToFloat(b.toInt).toDouble
  private def lane(beat: BigInt, i: Int): Double = dec((beat >> (32 * i)) & ((BigInt(1) << 32) - 1))

  // csr(0): [7:0] nValid | [8] accEn | [9] accInit | [12:10] accSlot | [15:13] combineMode
  private def csrWord(mode: Int, nValid: Int, accEn: Int = 0, accInit: Int = 0, accSlot: Int = 0): BigInt =
    (BigInt(mode) << 13) | (BigInt(accSlot) << 10) | (BigInt(accInit) << 9) | (BigInt(accEn) << 8) | BigInt(nValid)

  // pack 8 (field0, field1) partials into a 512-bit beat: field0_k = lane k, field1_k = lane 8+k
  private def packPairs(pairs: Seq[(Double, Double)]): BigInt = {
    var b = BigInt(0)
    for ((p, k) <- pairs.zipWithIndex) { b |= f32(p._1) << (32 * k); b |= f32(p._2) << (32 * (8 + k)) }
    b
  }
  // pack one (m, ℓ, O[dHead]) partial: lane0=m, lane1=ℓ, lanes 2..2+dHead-1 = O
  private def packAttn(m: Double, l: Double, o: Seq[Double]): BigInt = {
    var b = f32(m) | (f32(l) << 32)
    for ((ov, k) <- o.zipWithIndex) b |= f32(ov) << (32 * (2 + k))
    b
  }

  // drive one beat through the harness, return the raw output beat
  private def runBeat(dut: DataPathExtensionHarness, csr: BigInt, beat: BigInt): BigInt = {
    dut.io.csr_i(0).poke(csr.U)
    dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    var out = BigInt(0)
    var th  = new chiseltest.internal.TesterThreadList(Seq())
    th = th.fork {
      dut.io.data_i.bits.poke(beat.U); dut.io.data_i.valid.poke(true)
      while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
      dut.clock.step(1); dut.io.data_i.valid.poke(false)
    }
    th = th.fork {
      while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
      out = dut.io.data_o.bits.peekInt()
      dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false)
    }
    th.joinAndStep()
    dut.io.data_o.ready.poke(true)
    var w = 0; while (dut.io.busy_o.peekBoolean() && w < 200) { dut.clock.step(1); w += 1 }
    dut.io.data_o.ready.poke(false)
    out
  }

  "UnifiedMonoidMergeRt_MOMENT" should "fold the softmax (m, ℓ) normalizer on the paired tree" in {
    test(new DataPathExtensionHarness(new HasUnifiedMonoidMergeRt(dHead = dHead)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x2b2b)
        var worst = 0.0
        for (trial <- 0 until 10) {
          // each shard supplies a local (m, ℓ) = (max, Σexp(x-max)); merged == direct global (max, Σexp)
          val shards = Seq.fill(8)(Seq.fill(4)(rng.between(-3.0, 6.0)))
          val pairs  = shards.map { s => val m = s.max; (m, s.map(x => math.exp(x - m)).sum) }
          val out    = runBeat(dut, csrWord(MODE_MOMENT, nValid = 8), packPairs(pairs))
          val flat   = shards.flatten; val gm = flat.max; val gl = flat.map(x => math.exp(x - gm)).sum
          val (m, l) = (lane(out, 0), lane(out, 1))
          val rel = math.abs(l - gl) / gl; if (rel > worst) worst = rel
          assert(math.abs(m - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"trial $trial m=$m vs $gm")
          assert(rel <= 1.5e-2, s"trial $trial l=$l vs $gl (rel=$rel)")
        }
        println(f"[Unified/MOMENT] softmax normalizer fold worst rel err=$worst%.3g over 10 trials")
      }
  }

  "UnifiedMonoidMergeRt_SUM" should "fold the norm (Σx, Σx²) stats on the same paired tree" in {
    test(new DataPathExtensionHarness(new HasUnifiedMonoidMergeRt(dHead = dHead)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x5011)
        for (trial <- 0 until 10) {
          val pairs = Seq.fill(8)((rng.between(-4.0, 4.0), rng.between(0.0, 8.0))) // (S1, S2) partials
          val out   = runBeat(dut, csrWord(MODE_SUM, nValid = 8), packPairs(pairs))
          val (gs1, gs2) = (pairs.map(_._1).sum, pairs.map(_._2).sum)
          val (s1, s2)   = (lane(out, 0), lane(out, 1))
          assert(math.abs(s1 - gs1) / (math.abs(gs1) + 1e-6) <= 1e-4, s"trial $trial S1=$s1 vs $gs1")
          assert(math.abs(s2 - gs2) / (math.abs(gs2) + 1e-6) <= 1e-4, s"trial $trial S2=$s2 vs $gs2")
        }
        // short beat: only 3 live pairs, lanes 3..7 masked to the additive identity (0)
        val p3  = Seq((1.5, 2.0), (-2.0, 3.0), (0.5, 1.0))
        val out = runBeat(dut, csrWord(MODE_SUM, nValid = 3), packPairs(p3))
        assert(math.abs(lane(out, 0) - p3.map(_._1).sum) <= 1e-4 && math.abs(lane(out, 1) - p3.map(_._2).sum) <= 1e-4,
               "short SUM beat")
        println(f"[Unified/SUM] norm (Σx, Σx²) fold matches plain sum; short-beat(3) masking OK")
      }
  }

  "UnifiedMonoidMergeRt_MAXPOOL" should "max-reduce the beat's field0 lanes (KEY-only degenerate)" in {
    test(new DataPathExtensionHarness(new HasUnifiedMonoidMergeRt(dHead = dHead)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x3a11)
        for (trial <- 0 until 10) {
          val vals  = Seq.fill(8)(rng.between(-9.0, 9.0))
          val pairs = vals.map(v => (v, rng.between(-1.0, 1.0))) // field1 is don't-care for MAXPOOL
          val out   = runBeat(dut, csrWord(MODE_MAXPOOL, nValid = 8), packPairs(pairs))
          assert(math.abs(lane(out, 0) - vals.max) <= math.abs(vals.max) * 1e-6 + 1e-9,
                 s"trial $trial max=${lane(out, 0)} vs ${vals.max}")
        }
        println(f"[Unified/MAXPOOL] max-reduce over 8 field0 lanes exact over 10 trials")
      }
  }

  "UnifiedMonoidMergeRt_ATTN" should "fold the (m, ℓ, O) triple as the single-partial accEn mode" in {
    test(new DataPathExtensionHarness(new HasUnifiedMonoidMergeRt(dHead = dHead)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x0a44)
        def refMerge(sh: Seq[(Double, Double, Seq[Double])]): (Double, Double, Seq[Double]) = {
          val ms = sh.map(_._1).max
          val ls = sh.map { case (m, l, _) => l * math.exp(m - ms) }.sum
          val os = (0 until dHead).map(j => sh.map { case (m, _, o) => o(j) * math.exp(m - ms) }.sum)
          (ms, ls, os)
        }
        for (trial <- 0 until 6) {
          val shards = Seq.fill(4)((rng.between(-2.0, 5.0), rng.between(1.0, 5.0), Seq.fill(dHead)(rng.between(-3.0, 3.0))))
          runBeat(dut, csrWord(MODE_ATTN, nValid = 1, accEn = 1, accInit = 1), packAttn(shards(0)._1, shards(0)._2, shards(0)._3))
          var out = BigInt(0)
          for (i <- 1 until 4)
            out = runBeat(dut, csrWord(MODE_ATTN, nValid = 1, accEn = 1, accInit = 0), packAttn(shards(i)._1, shards(i)._2, shards(i)._3))
          val (gm, gl, go) = refMerge(shards)
          assert(math.abs(lane(out, 0) - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"attn trial $trial m")
          assert(math.abs(lane(out, 1) - gl) / gl <= 1.5e-2, s"attn trial $trial ℓ=${lane(out, 1)} vs $gl")
          for (j <- 0 until dHead) {
            val d = math.abs(lane(out, 2 + j) - go(j)) / (math.abs(go(j)) + 1e-6)
            assert(d <= 2e-2, s"attn trial $trial O[$j] rel=$d")
          }
        }
        println(f"[Unified/ATTN] (m, ℓ, O) 4-shard accEn fold matches flash-attention reference (dHead=$dHead)")
      }
  }

  "UnifiedMonoidMergeRt_roofline" should "stream the collective at ~1 beat/cycle in the paired MOMENT mode" in {
    test(new DataPathExtensionHarness(new HasUnifiedMonoidMergeRt(dHead = dHead)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x77aa)
        val nIn = 200
        val beats = Seq.fill(nIn)(packPairs(Seq.fill(8)((rng.between(-2.0, 4.0), rng.between(0.5, 3.0)))))
        dut.io.csr_i(0).poke(csrWord(MODE_MOMENT, nValid = 8).U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.data_o.ready.poke(true)
        dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(beats(0).U)
        var cyc = 0L; var fed = 0; var got = 0; var warm = -1L; var last = -1L
        while (got < nIn && cyc < nIn * 40 + 400) {
          val canFeed = fed < nIn && dut.io.data_i.ready.peekBoolean(); val outNow = dut.io.data_o.valid.peekBoolean()
          dut.clock.step(1); cyc += 1
          if (canFeed) { fed += 1; if (fed == 8) warm = cyc; last = cyc
            if (fed < nIn) dut.io.data_i.bits.poke(beats(fed).U) else dut.io.data_i.valid.poke(false) }
          if (outNow) got += 1
        }
        assert(got == nIn, s"only $got/$nIn")
        val util = (nIn - 8).toDouble / (last - warm)
        println(f"[Unified/roofline] MOMENT-mode util=$util%.3f (the configurable collective at 512 b/cyc)")
        assert(util > 0.9, s"unified roofline should be ~1.0, got $util")
      }
  }
}
