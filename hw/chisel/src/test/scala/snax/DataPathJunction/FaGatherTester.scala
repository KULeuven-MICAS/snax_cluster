package snax.DataPathJunction

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathExtension.FpHelpers
import snax.DataPathJunction.JunctionTestUtils._

/** \============================================================================================================
  * `FaGatherTester` -- the 4-cluster FlashAttention gather, as a bench for CHOOSING an encoding.
  * \============================================================================================================
  *
  * The junction's own testers prove the operator is right. This one asks the next question: given that it is, WHAT
  * SHOULD SOFTWARE PUT IN THE FIELDS? Four clusters each hold a partial `(m_c, l_c, O_c)` for the same query tile,
  * having attended a different slice of the keys; the chain folds them hop by hop and the answer is `O* / l*`.
  * Several encodings reach it, they differ by orders of magnitude in accuracy and by 2x in payload, and one of them
  * is wrong in a way that looks entirely plausible.
  *
  * Every method runs on the SAME data through the SAME RTL against the SAME float64 golden, so the columns are
  * comparable by construction.
  *
  * ---- WHY THE DATA IS SHAPED LIKE THIS ----
  * The shards get genuinely different MASSES (`l` from 1 to ~600 matched keys), not just different maxima. That is
  * what separates the encodings: when every shard attended a similar number of keys the naive encoding looks fine,
  * and only a lopsided split exposes that weighting by `alpha` alone instead of `alpha . l` is wrong. A bench built
  * on uniform shards would rank the methods the wrong way round.
  *
  * The raw numerator is `sum P8.V8`, i.e. ~`127 . l . V`, reaching ~1e7 -- 48x past FP16's ceiling and inside
  * FP32's exact-integer range. Both facts are load-bearing.
  *
  * ---- HOW ERROR IS MEASURED, WHICH IS ITSELF A RESULT ----
  * `O*` is a SUM OVER SHARDS WITH SIGNS, and it cancels: measured on this data the largest term is typically 1.0x
  * the sum but reaches 140x in the worst lanes. Scoring "relative to the answer" therefore measures the
  * cancellation, not the encoding -- an early version of this bench did exactly that and ranked FP16 far worse
  * than it deserves. The column that decides is `relative to max|term|`, the standard bound for a floating-point
  * summation; `rel-to-gold` is reported alongside as a median and a p99 because it is what a user actually feels.
  */
class FaGatherTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  private val P     = 4   // clusters in the chain
  private val Br    = 8   // query rows folded (payload columns are reported per Br = 32)
  private val dHead = 128
  private val CH    = 9   // `n` is 4 bits so F <= 16, and 1 + dHead values need ceil(129/15) chunks

  // ---- the shard data, and the float64 answer -------------------------------------------------------------
  private val rng = new Random(0xfa4c)
  private val m   = Array.tabulate(P, Br)((_, _) => rng.between(-2.0, 8.0))
  private val l   = Array.tabulate(P, Br)((_, _) => rng.between(1.0, 600.0))
  private val O   = Array.tabulate(P, Br, dHead)((c, r, _) => l(c)(r) * rng.between(-127.0, 127.0))

  private val mStar = Array.tabulate(Br)(r => (0 until P).map(m(_)(r)).max)
  private val alpha = Array.tabulate(P, Br)((c, r) => math.exp(m(c)(r) - mStar(r)))
  private val lStar = Array.tabulate(Br)(r => (0 until P).map(c => alpha(c)(r) * l(c)(r)).sum)
  private val gNum  = Array.tabulate(Br, dHead)((r, j) => (0 until P).map(c => alpha(c)(r) * O(c)(r)(j)).sum)
  // the scale a floating-point summation's error is judged against: the largest term that entered it
  private val gScal = Array.tabulate(Br, dHead)((r, j) => (0 until P).map(c => math.abs(alpha(c)(r) * O(c)(r)(j))).max)
  private val gold  = Array.tabulate(Br, dHead)((r, j) => gNum(r)(j) / lStar(r))

  // ---- codecs ----------------------------------------------------------------------------------------------
  private def encF16(d: Double): BigInt = BigInt(java.lang.Float.floatToFloat16(d.toFloat) & 0xffff)
  private def decF16(b: BigInt): Double = java.lang.Float.float16ToFloat(b.toShort).toDouble
  private def pOf(x: Double):    Int    = math.floor(math.log(x) / math.log(2.0)).toInt // = l's exponent field

  private def word(n: Int, nExp: Int, sigma: Int, nValid: Int, fmt: Int): BigInt =
    (BigInt(sigma) << 26) | (BigInt(nExp) << 18) | (BigInt(fmt) << 12) | (BigInt(n) << 8) | BigInt(nValid)
  private val FP32 = ElementwiseJunction.FMT_FP32
  private val FP16 = FpHelpers.FMT_FP16

  /** 129 value coordinates sliced into CH chunks of 15, the key replicated into every one */
  private def chunk(key: Double, vals: Seq[Double]): Seq[(Double, Seq[Double])] =
    (0 until CH).map { k =>
      val s = vals.slice(k * 15, math.min((k + 1) * 15, vals.length))
      (key, s ++ Seq.fill(15 - s.length)(0.0))
    }

  /** one shard's partial for query row `r`, as (key, 129 values), under a chosen embedding */
  private def partial(c: Int, r: Int, pow2: Boolean, naive: Boolean): (Double, Seq[Double]) = {
    if (naive)     (m(c)(r), 1.0 +: (0 until dHead).map(j => O(c)(r)(j) / l(c)(r)))
    else if (pow2) {
      val p = pOf(l(c)(r)); val s = math.pow(2.0, -p)
      (m(c)(r) + p * math.log(2.0), (l(c)(r) * s) +: (0 until dHead).map(j => O(c)(r)(j) * s))
    } else         (m(c)(r), l(c)(r) +: (0 until dHead).map(j => O(c)(r)(j)))
  }

  private case class Enc(name: String, fmt: Int, pow2: Boolean, naive: Boolean, tolScale: Double) {
    val eW      = if (fmt == FP32) 32 else 16
    val sigma   = if (fmt == FP32) 0 else 1   // FP32 beat = 16 elements, so F*S <= 16 forces S = 1
    val slots   = 1 << sigma
    val csr     = word(15, 15, sigma, slots, fmt)
    def beats32 = 32 / slots * CH
    def bytes32 = beats32 * 64
    def enc(d: Double): BigInt = if (fmt == FP32) f32(d) else encF16(d)
    def dec(b: BigInt): Double = if (fmt == FP32) JunctionTestUtils.dec(b) else decF16(b)

    /** the CH beats carrying query rows [g*slots .. g*slots+slots-1] of shard c */
    def beatsOf(c: Int, g: Int): Seq[BigInt] = {
      val per = (0 until slots).map(s => chunk _ tupled partial(c, g * slots + s, pow2, naive))
      (0 until CH).map { k =>
        var b = BigInt(0)
        for (s <- 0 until slots) {
          val (key, vs) = per(s)(k)
          b |= enc(key) << (eW * s)
          for ((v, f) <- vs.zipWithIndex) b |= enc(v) << (eW * ((f + 1) * slots + s))
        }
        b
      }
    }
    /** folded beats -> (denominator field, dHead numerator fields) for one slot */
    def readBack(beats: Seq[BigInt], s: Int): (Double, Seq[Double]) = {
      val flat = beats.flatMap(b => (1 until 16).map(f => dec(lane(b, f * slots + s, eW))))
      (flat.head, flat.slice(1, 1 + dHead))
    }
  }

  private val methods = Seq(
    Enc("A   FP32  raw        key=m,         vals=(l,O)",        FP32, pow2 = false, naive = false, 1e-3),
    Enc("B   FP16  pow2       key=m+p.ln2,   vals=(l,O)/2^p",    FP16, pow2 = true,  naive = false, 5e-2),
    Enc("B32 FP32  pow2       (isolates the EMBEDDING)",          FP32, pow2 = true,  naive = false, 1e-3),
    Enc("C   FP16  O/l        key=m          (control: WRONG)",  FP16, pow2 = false, naive = true,  9e9)
  )

  "FaGather" should "rank the candidate encodings on one bench" in {
    test(new DataPathJunctionHarness(new HasMonoidJunction(elemWidth = 16)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { implicit dut =>
        println("%-46s | %-10s | %-9s | %-9s | payload".format("encoding", "worst/term", "med/gold", "p99/gold"))
        println("-" * 104)
        val verdicts = methods.map { e =>
          var worstScale = 0.0
          val relGold    = scala.collection.mutable.ArrayBuffer[Double]()
          for (g <- 0 until Br / e.slots) {
            // hop by hop: A is what arrived from upstream, B is this node's own partial.
            var acc = e.beatsOf(0, g)
            for (hop <- 1 until P) acc = acc.zip(e.beatsOf(hop, g)).map { case (a, b) => runPair(dut, e.csr, a, b) }
            for (s <- 0 until e.slots) {
              val r           = g * e.slots + s
              val (den, nums) = e.readBack(acc, s)
              for (j <- 0 until dHead) {
                val got = nums(j) / den
                val ae  = math.abs(got - gold(r)(j))
                worstScale = math.max(worstScale, ae * lStar(r) / gScal(r)(j)) // FP-summation bound
                relGold += ae / math.abs(gold(r)(j))
              }
            }
          }
          val sorted = relGold.sorted
          println(f"${e.name}%-46s | ${worstScale}%10.2e | ${sorted(sorted.length / 2)}%9.2e | " +
            f"${sorted((0.99 * sorted.length).toInt)}%9.2e | ${e.beats32}%4d beats ${e.bytes32}%6d B")
          (e, worstScale)
        }
        println("-" * 104)
        for ((e, w) <- verdicts)
          assert(w <= e.tolScale, f"${e.name}: worst/term $w%.2e exceeded its budget ${e.tolScale}%.0e")
        val ctlErr = verdicts.last._2
        assert(ctlErr > 1.0, s"the control came out RIGHT (${ctlErr}) -- the bench cannot tell the methods apart")
        println(f"control C is wrong by ${ctlErr}%.0fx max|term|, so the bench discriminates")
      }
  }
}

/** The remaining serious candidate: DON'T fold with the monoid at all.
  *
  * Fold `(m, l)` first -- 4 beats, cheap -- broadcast `m*, l*` back to every shard, let each shard pre-scale its own
  * numerator by `alpha_c / l*`, and then the numerators are already on a common basis and the collective is a plain
  * SUM. That is `ElementwiseJunction` ADD: no key, no twist, no chunking and no key replication, so the beats carry
  * only payload.
  *
  * The question this answers is whether FP16's cost in `FaGatherTester` is the FORMAT or the TWIST PATH. If a
  * pre-scaled FP16 ADD lands near the monoid's FP16 number, the loss is the format and there is no cheaper FP16
  * route; if it lands much closer to FP32, the twist is what FP16 cannot carry and the two-pass shape buys accuracy
  * back at half the payload.
  *
  * The price, which this bench cannot measure, is a SERIALISING round trip: no shard can pre-scale until `l*` comes
  * back to it.
  */
class FaGatherElementwiseTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  private val P     = 4
  private val Br    = 8
  private val dHead = 128

  private val rng = new Random(0xfa4c) // the SAME stream as FaGatherTester, so the rows are the same rows
  private val m   = Array.tabulate(P, Br)((_, _) => rng.between(-2.0, 8.0))
  private val l   = Array.tabulate(P, Br)((_, _) => rng.between(1.0, 600.0))
  private val O   = Array.tabulate(P, Br, dHead)((c, r, _) => l(c)(r) * rng.between(-127.0, 127.0))

  private val mStar = Array.tabulate(Br)(r => (0 until P).map(m(_)(r)).max)
  private val alpha = Array.tabulate(P, Br)((c, r) => math.exp(m(c)(r) - mStar(r)))
  private val lStar = Array.tabulate(Br)(r => (0 until P).map(c => alpha(c)(r) * l(c)(r)).sum)
  private val gNum  = Array.tabulate(Br, dHead)((r, j) => (0 until P).map(c => alpha(c)(r) * O(c)(r)(j)).sum)
  private val gScal = Array.tabulate(Br, dHead)((r, j) => (0 until P).map(c => math.abs(alpha(c)(r) * O(c)(r)(j))).max)
  private val gold  = Array.tabulate(Br, dHead)((r, j) => gNum(r)(j) / lStar(r))

  private def encF16(d: Double): BigInt = BigInt(java.lang.Float.floatToFloat16(d.toFloat) & 0xffff)
  private def decF16(b: BigInt): Double = java.lang.Float.float16ToFloat(b.toShort).toDouble

  "FaGather_elementwise" should "score the pre-scaled ADD route at both transports" in {
    test(new DataPathJunctionHarness(new HasElementwiseJunction(elemWidth = 16)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        import ElementwiseJunction._
        println("%-46s | %-10s | %-9s | %-9s | payload".format("encoding", "worst/term", "med/gold", "p99/gold"))
        println("-" * 104)
        for ((name, fmt, eW) <- Seq(
               ("E16 FP16  pre-scaled ADD  (2 passes, no key)", FpHelpers.FMT_FP16, 16),
               ("E32 FP32  pre-scaled ADD  (2 passes, no key)", FMT_FP32, 32)
             )) {
          val perBeat = 512 / eW
          val csr     = (BigInt(fmt) << 4) | BigInt(OP_ADD)
          def enc(d: Double): BigInt = if (eW == 32) f32(d) else encF16(d)
          def dc(b:  BigInt): Double = if (eW == 32) dec(b) else decF16(b)
          var worst   = 0.0
          val relGold = scala.collection.mutable.ArrayBuffer[Double]()
          for (r <- 0 until Br) {
            // each shard ships alpha_c . O_c / l*, already on the common basis
            def shardBeats(c: Int): Seq[BigInt] =
              (0 until dHead).map(j => alpha(c)(r) * O(c)(r)(j) / lStar(r)).grouped(perBeat).map { g =>
                var b = BigInt(0); for ((v, i) <- g.zipWithIndex) b |= enc(v) << (eW * i); b
              }.toSeq
            var acc = shardBeats(0)
            for (hop <- 1 until P) acc = acc.zip(shardBeats(hop)).map { case (a, b) => runPair(dut, csr, a, b) }
            val got = acc.flatMap(b => (0 until perBeat).map(i => dc(lane(b, i, eW)))).take(dHead)
            for (j <- 0 until dHead) {
              val ae = math.abs(got(j) - gold(r)(j))
              worst = math.max(worst, ae * lStar(r) / gScal(r)(j))
              relGold += ae / math.abs(gold(r)(j))
            }
          }
          val s     = relGold.sorted
          val beats = 32 * (dHead / perBeat) + 4 // + the (m, l) fold that has to precede it
          println(f"$name%-46s | $worst%10.2e | ${s(s.length / 2)}%9.2e | ${s((0.99 * s.length).toInt)}%9.2e | " +
            f"$beats%4d beats ${beats * 64}%6d B")
        }
        println("-" * 104)
        println("(+ a serialising round trip to broadcast l* back, which this bench does not model)")
      }
  }
}
