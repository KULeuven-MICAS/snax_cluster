package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** StreamMomentMergeRt: the in-transit nonlinear collective as a STREAMING op (doc 13 F2). One beat carries 8
  * flash-statistic pairs (m_k, l_k); the module folds them into the global (m*, l*) via the moment-merge
  * monoid, at the 512 b/cyc roofline. Proves correctness (merged == direct global (max, Sexp)), the short-beat
  * (nValid<8) masking, and that beats stream at util ~ 1.0 (the fold is fully pipelined).
  */
class StreamMomentMergeRtTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  private def f32(f: Double): BigInt = BigInt(java.lang.Float.floatToIntBits(f.toFloat).toLong & 0xffffffffL)
  private def dec(b: BigInt): Double = java.lang.Float.intBitsToFloat(b.toInt).toDouble
  private def local(s: Seq[Double]): (Double, Double) = { val m = s.max; (m, s.map(x => math.exp(x - m)).sum) }
  private def global(a: Seq[Double]): (Double, Double) = { val m = a.max; (m, a.map(x => math.exp(x - m)).sum) }

  // pack up to 8 (m,l) pairs into a 512-bit beat: m_k in lane k, l_k in lane 8+k
  private def packPairs(pairs: Seq[(Double, Double)]): BigInt = {
    var b = BigInt(0)
    for ((p, k) <- pairs.zipWithIndex) {
      b |= f32(p._1) << (32 * k)
      b |= f32(p._2) << (32 * (8 + k))
    }
    b
  }

  private def runBeat(dut: DataPathExtensionHarness, pairs: Seq[(Double, Double)]): (Double, Double) = {
    dut.io.csr_i(0).poke(BigInt(pairs.length).U)
    dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    var out = BigInt(0)
    var th = new chiseltest.internal.TesterThreadList(Seq())
    th = th.fork {
      dut.io.data_i.bits.poke(packPairs(pairs).U); dut.io.data_i.valid.poke(true)
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
    var w = 0; while (dut.io.busy_o.peekBoolean() && w < 100) { dut.clock.step(1); w += 1 }
    dut.io.data_o.ready.poke(false)
    (dec(out & ((BigInt(1) << 32) - 1)), dec((out >> 32) & ((BigInt(1) << 32) - 1)))
  }

  "StreamMomentMergeRt_correct" should "merge 8 shard statistics to the global (m*, Sexp*)" in {
    test(new DataPathExtensionHarness(new HasStreamMomentMergeRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x2b2b)
        var worst = 0.0
        for (trial <- 0 until 10) {
          val shards = Seq.fill(8)(Seq.fill(4)(rng.between(-3.0, 6.0)))
          val (m, l)   = runBeat(dut, shards.map(local))
          val (gm, gl) = global(shards.flatten)
          val rel = math.abs(l - gl) / gl; if (rel > worst) worst = rel
          assert(math.abs(m - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"trial $trial m=$m vs $gm")
          assert(rel <= 1.5e-2, s"trial $trial l=$l vs $gl (rel=$rel)")
        }
        // short beat: only 3 live partials, the rest masked to the identity (-inf, 0)
        val sh3 = Seq(Seq(1.0, 3.0), Seq(4.0, 0.0), Seq(2.5, -1.0))
        val (m3, l3) = runBeat(dut, sh3.map(local))
        val (gm3, gl3) = global(sh3.flatten)
        assert(math.abs(m3 - gm3) <= 1e-6 && math.abs(l3 - gl3) / gl3 <= 1.5e-2, s"short beat m=$m3 l=$l3")
        println(f"[MomentMergeRt] 8-shard merge worst rel err=$worst%.3g; short-beat(3) OK")
      }
  }

  "StreamMomentMergeRt_roofline" should "stream the collective at ~1 beat/cycle (512 b/cyc)" in {
    test(new DataPathExtensionHarness(new HasStreamMomentMergeRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x77aa)
        val nIn = 200
        val beats = Seq.fill(nIn)(packPairs(Seq.fill(8)((rng.between(-2.0, 4.0), rng.between(0.5, 3.0)))))
        dut.io.csr_i(0).poke(BigInt(8).U)
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
        println(f"[MomentMergeRt roofline] util=$util%.3f (the nonlinear collective at 512 b/cyc)")
        assert(util > 0.9, s"moment-merge roofline should be ~1.0, got $util")
      }
  }
}
