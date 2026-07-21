package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** StreamTopKMergeRt: the THIRD collective-algebra monoid -- distributed MoE top-k routing. One beat carries
  * up to 8 candidate (logit, expert-id) pairs; the module sorts+merges them, in accEn mode into a persistent
  * top-8 slot, so P shard-lists fold into the global top-k table in-transit. Proves: (1) the sorted top-M is
  * exactly the numpy global top-k with correct indices, (2) short beats (nValid<8) sort real candidates above
  * the identity, (3) accEn folds two producers' local lists into the global top-k, (4) roofline stream rate.
  */
class StreamTopKMergeRtTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))
  private val M = 8

  private def f32(f: Double): BigInt = BigInt(java.lang.Float.floatToIntBits(f.toFloat).toLong & 0xffffffffL)
  private def dec(b: BigInt): Double = java.lang.Float.intBitsToFloat(b.toInt).toDouble

  // pack up to 8 (value, index) candidates into a 512-bit beat: value_k in lane k, index_k in lane 8+k
  private def packCand(cand: Seq[(Double, Int)]): BigInt = {
    var b = BigInt(0)
    for (((v, idx), k) <- cand.zipWithIndex) {
      b |= f32(v) << (32 * k)
      b |= (BigInt(idx) & ((BigInt(1) << 32) - 1)) << (32 * (M + k))
    }
    b
  }
  // numpy-style global reference: sort DESCENDING by value, ties -> smaller index; take top-k
  private def globalTopK(cand: Seq[(Double, Int)], k: Int): Seq[(Double, Int)] =
    cand.sortBy { case (v, idx) => (-v, idx) }.take(k)

  // decode the output beat -> M (value, index) pairs in lane order
  private def unpack(out: BigInt): Seq[(Double, Int)] =
    (0 until M).map { k =>
      val v   = dec((out >> (32 * k)) & ((BigInt(1) << 32) - 1))
      val idx = ((out >> (32 * (M + k))) & ((BigInt(1) << 32) - 1)).toInt
      (v, idx)
    }

  // drive ONE beat (with CSR bits) and return the emitted top-M table. Slots survive start_i, so a caller can
  // arm then fold across two calls.
  private def runBeat(dut: DataPathExtensionHarness, cand: Seq[(Double, Int)],
                      accEn: Int = 0, accInit: Int = 0, accSlot: Int = 0): Seq[(Double, Int)] = {
    val csr = BigInt(cand.length & 0xff) | (BigInt(accEn) << 8) | (BigInt(accInit) << 9) | (BigInt(accSlot) << 10)
    dut.io.csr_i(0).poke(csr.U)
    dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    var out = BigInt(0)
    var th  = new chiseltest.internal.TesterThreadList(Seq())
    th = th.fork {
      dut.io.data_i.bits.poke(packCand(cand).U); dut.io.data_i.valid.poke(true)
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
    unpack(out)
  }

  private def assertTopKMatches(got: Seq[(Double, Int)], gold: Seq[(Double, Int)], k: Int, msg: String): Unit =
    for (r <- 0 until k) {
      assert(math.abs(got(r)._1 - gold(r)._1) <= math.abs(gold(r)._1) * 1e-6 + 1e-9,
             s"$msg rank $r value ${got(r)._1} vs ${gold(r)._1}")
      assert(got(r)._2 == gold(r)._2, s"$msg rank $r index ${got(r)._2} vs ${gold(r)._2}")
    }

  "StreamTopKMergeRt_sort" should "emit the global top-8 (value,index) table sorted descending" in {
    test(new DataPathExtensionHarness(new HasStreamTopKMergeRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x70b1)
        for (trial <- 0 until 12) {
          // 8 DISTINCT logits (distinct-value invariant -> unambiguous ordering), indices = expert ids 0..7 shuffled
          val vals = rng.shuffle((0 until M).toList).map(i => -3.0 + i * 0.9 + rng.between(-0.3, 0.3))
          val idxs = rng.shuffle((10 until 10 + M).toList)
          val cand = vals.zip(idxs)
          val got  = runBeat(dut, cand)
          assertTopKMatches(got, globalTopK(cand, M), M, s"trial $trial")
        }
        println("[TopKMergeRt] 8-candidate full sort (value+index) OK over 12 trials")
      }
  }

  "StreamTopKMergeRt_shortbeat" should "sort real candidates above the identity for nValid<8" in {
    test(new DataPathExtensionHarness(new HasStreamTopKMergeRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val cand = Seq((2.0, 5), (7.5, 3), (1.0, 9))   // 3 live; the other 5 lanes are identity
        val got  = runBeat(dut, cand)
        assertTopKMatches(got, globalTopK(cand, 3), 3, "short-beat")
        // the top-3 must be the 3 real experts, not the identity sentinel
        assert(got.take(3).map(_._2).toSet == Set(3, 5, 9), s"short-beat top-3 ids ${got.take(3).map(_._2)}")
        println("[TopKMergeRt] short-beat(3) real>identity OK")
      }
  }

  "StreamTopKMergeRt_tiebreak" should "break value ties by SMALLER index first (matches the numpy golden)" in {
    test(new DataPathExtensionHarness(new HasStreamTopKMergeRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        // exact value ties across several experts; the sort must resolve them by smaller index first, exactly
        // as globalTopK's stable sortBy (-v, idx) does. Indices deliberately out of order to force the tiebreak.
        val cand = Seq((5.0, 6), (5.0, 1), (5.0, 4), (2.0, 9), (2.0, 0), (7.0, 3), (7.0, 8), (2.0, 2))
        val got  = runBeat(dut, cand)
        assertTopKMatches(got, globalTopK(cand, M), M, "tiebreak")
        // spell out the expected order: 7.0@3, 7.0@8, 5.0@1, 5.0@4, 5.0@6, 2.0@0, 2.0@2, 2.0@9
        assert(got.map(_._2) == Seq(3, 8, 1, 4, 6, 0, 2, 9), s"tiebreak order ${got.map(_._2)}")
        println("[TopKMergeRt] value-tie -> smaller-index-first OK")
      }
  }

  "StreamTopKMergeRt_accEn" should "fold two shards' local lists into the global top-k" in {
    test(new DataPathExtensionHarness(new HasStreamTopKMergeRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x2c0f)
        for (trial <- 0 until 8) {
          // two shards, each with 8 distinct local experts (disjoint id ranges), distinct values across both
          val base  = rng.between(-1.0, 1.0)
          val vA    = (0 until M).map(i => base + i * 0.7 + rng.between(-0.2, 0.2))
          val vB    = (0 until M).map(i => base + 0.35 + i * 0.7 + rng.between(-0.2, 0.2))
          val shardA = vA.zip(100 until 100 + M)
          val shardB = vB.zip(200 until 200 + M)
          val slot   = trial % 8
          runBeat(dut, shardA, accEn = 1, accInit = 1, accSlot = slot) // arm with shard A's local top-8
          val got    = runBeat(dut, shardB, accEn = 1, accInit = 0, accSlot = slot) // fold shard B
          val gold   = globalTopK(shardA ++ shardB, M)
          // check the top-2 (canonical MoE routing) exactly -- value AND expert id
          assertTopKMatches(got, gold, 2, s"accEn trial $trial")
        }
        println("[TopKMergeRt] accEn 2-shard fold -> global top-2 routing OK over 8 trials")
      }
  }

  "StreamTopKMergeRt_roofline" should "stream the sort at ~1 beat/cycle (512 b/cyc)" in {
    test(new DataPathExtensionHarness(new HasStreamTopKMergeRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng   = new Random(0x51de)
        val nIn   = 200
        val beats = Seq.fill(nIn)(packCand((0 until M).map(i => (rng.between(-4.0, 4.0), i))))
        dut.io.csr_i(0).poke(BigInt(M).U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.data_o.ready.poke(true)
        dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(beats(0).U)
        var cyc = 0L; var fed = 0; var got = 0; var warm = -1L; var last = -1L
        while (got < nIn && cyc < nIn * 40 + 400) {
          val canFeed = fed < nIn && dut.io.data_i.ready.peekBoolean(); val outNow = dut.io.data_o.valid.peekBoolean()
          dut.clock.step(1); cyc += 1
          if (canFeed) { fed += 1; if (fed == 1) warm = cyc; last = cyc
            if (fed < nIn) dut.io.data_i.bits.poke(beats(fed).U) else dut.io.data_i.valid.poke(false) }
          if (outNow) got += 1
        }
        assert(got == nIn, s"only $got/$nIn")
        val util = (nIn - 1).toDouble / (last - warm)
        println(f"[TopKMergeRt roofline] util=$util%.3f (the top-k sort at 512 b/cyc)")
        assert(util > 0.9, s"top-k roofline should be ~1.0, got $util")
      }
  }
}
