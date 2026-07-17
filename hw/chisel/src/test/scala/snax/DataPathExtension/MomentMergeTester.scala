package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathExtension.FpHelpers._

/** F2 — the online-softmax MOMENT-MERGE monoid (the in-transit nonlinear collective, doc 13 §2). Combining
  * flash statistics (m, l) = (running max, Sexp) under the max-rescaled monoid must reproduce the GLOBAL
  * (max, Sexp) exactly (algebraically), and be order-independent (associative + commutative). Proven here:
  * doc 13's worked example, a random P-shard fold vs the direct global, and a shuffle-order associativity
  * check. This is the collective SHARP cannot express (a max coupled to a rescaled sum).
  */
class MomentMergeTester extends AnyFlatSpec with ChiselScalatestTester {

  // A balanced fold tree over P (m, l) pairs via momentMerge (combinational; exp LUT depth 128).
  private class MomentFoldDUT(p: Int) extends Module with RequireAsyncReset {
    val io = IO(new Bundle {
      val m  = Input(Vec(p, UInt(32.W)))
      val l  = Input(Vec(p, UInt(32.W)))
      val mo = Output(UInt(32.W))
      val lo = Output(UInt(32.W))
    })
    var ms: Seq[UInt] = (0 until p).map(io.m(_))
    var ls: Seq[UInt] = (0 until p).map(io.l(_))
    while (ms.length > 1) {
      val nm = scala.collection.mutable.ArrayBuffer[UInt]()
      val nl = scala.collection.mutable.ArrayBuffer[UInt]()
      var i = 0
      while (i + 1 < ms.length) {
        val (mm, ll, _) = momentMerge(ms(i), ls(i), ms(i + 1), ls(i + 1))
        nm += mm; nl += ll; i += 2
      }
      if (i < ms.length) { nm += ms(i); nl += ls(i) } // odd leftover passes through (identity fold)
      ms = nm.toSeq; ls = nl.toSeq
    }
    io.mo := ms.head; io.lo := ls.head
  }

  private def f32(f: Double): BigInt = BigInt(java.lang.Float.floatToIntBits(f.toFloat).toLong & 0xffffffffL)
  private def dec(b: BigInt): Double = java.lang.Float.intBitsToFloat(b.toInt).toDouble

  // local flash statistic of a shard of scores
  private def local(scores: Seq[Double]): (Double, Double) = {
    val m = scores.max; (m, scores.map(s => math.exp(s - m)).sum)
  }
  private def global(all: Seq[Double]): (Double, Double) = {
    val m = all.max; (m, all.map(s => math.exp(s - m)).sum)
  }

  private def runFold(dut: MomentFoldDUT, shards: Seq[Seq[Double]]): (Double, Double) = {
    for ((sh, k) <- shards.zipWithIndex) { val (m, l) = local(sh); dut.io.m(k).poke(f32(m).U); dut.io.l(k).poke(f32(l).U) }
    dut.clock.step(1)
    (dec(dut.io.mo.peekInt()), dec(dut.io.lo.peekInt()))
  }

  "MomentMerge_docexample" should "merge two shards to the exact global (m, Sexp)" in {
    test(new MomentFoldDUT(2)) { dut =>
      val shards = Seq(Seq(1.0, 3.0, 2.0), Seq(4.0, 0.0, 2.5))
      val (m, l) = runFold(dut, shards)
      val (gm, gl) = global(shards.flatten)
      println(f"[MomentMerge doc-example] merged=($m%.4f, $l%.6f)  global=($gm%.4f, $gl%.6f)")
      assert(m == gm, s"m: merged $m vs global $gm")
      assert(math.abs(l - gl) / gl <= 1e-2, s"l: merged $l vs global $gl") // exp-LUT tolerance
    }
  }

  "MomentMerge_random" should "fold P random shards to the global, order-independent" in {
    test(new MomentFoldDUT(8)) { dut =>
      val rng = new Random(0x9a11)
      var worst = 0.0
      for (trial <- 0 until 20) {
        val shards = Seq.fill(8)(Seq.fill(4)(rng.between(-3.0, 6.0)))
        val (m, l)   = runFold(dut, shards)
        val (gm, gl) = global(shards.flatten)
        val rel = math.abs(l - gl) / gl
        if (rel > worst) worst = rel
        assert(math.abs(m - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"trial $trial m: merged $m vs global $gm") // FP32 max
        assert(rel <= 1.5e-2, s"trial $trial l: merged $l vs global $gl (rel=$rel)")
        // associativity: a shuffled fold order must match (same monoid, order-independent)
        val (m2, l2) = runFold(dut, rng.shuffle(shards))
        assert(m2 == m && math.abs(l2 - l) <= math.abs(l) * 1e-6, s"trial $trial not order-independent")
      }
      println(f"[MomentMerge random P=8] worst rel err vs global = $worst%.3g (exp LUT), order-independent OK")
    }
  }
}
