package snax.DataPathJunction

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathJunction.JunctionTestUtils._

/** Tier-1 for `TopKJunction` -- the operator from OUTSIDE the classified monoid family.
  *
  * The point of this operator is not that top-k is useful. It is that the socket carries an operator with a
  * different algebra, a different beat layout and no arithmetic in common with the monoid cell -- no key front
  * end, no exponential, no FMA -- while satisfying the same four obligations. So the tests below are deliberately
  * the CONTRACT tests, not a feature list:
  *
  *   T-O2 closure   the output of one instance is a legal input to the next, checked against a 3-input reference
  *   T-O3 identity  a short list is padded with the identity and the live entries are unaffected
  *   T-O1 latency   retire is fire delayed by the published constant
  *   T-O4 stateless a pair's result does not depend on the pairs before it
  */
class TopKJunctionTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))
  private val k      = 8
  private val ID_M_D = java.lang.Float.intBitsToFloat(0xff7fffff).toDouble

  private def hasTopK = new HasTopKJunction(k = k)

  private def csrWord(nValid: Int): BigInt = BigInt(nValid)

  /** pack one descending sorted list: values on lanes 0..k-1, payloads on k..2k-1 */
  private def pack(vals: Seq[Double], pays: Seq[Double]): BigInt = {
    var b = BigInt(0)
    for (i <- vals.indices) { b |= f32(vals(i)) << (32 * i); b |= f32(pays(i)) << (32 * (k + i)) }
    b
  }
  private def unpackV(beat: BigInt): Seq[Double] = (0 until k).map(i => laneF32(beat, i))
  private def unpackP(beat: BigInt): Seq[Double] = (0 until k).map(i => laneF32(beat, k + i))

  /** the software reference: the k largest of the union, descending, payload following its value */
  private def topK(entries: Seq[(Double, Double)]): Seq[(Double, Double)] =
    entries.sortBy(-_._1).take(k)

  /** a random DESCENDING sorted list of `live` distinct values, padded to k with the identity.
    *
    * Values are ROUND-TRIPPED THROUGH FP32 first. The beat carries FP32, so a double-precision stimulus would
    * make the software reference disagree with the hardware in the last few bits of the mantissa for the right
    * answer -- and this operator only ever selects and reorders values, never computes new ones, so with an
    * exactly-representable stimulus the comparison can and should be exact.
    */
  private def randList(rng: Random, live: Int, tag: Int): Seq[(Double, Double)] = {
    val vs = Seq.fill(live)(rng.between(-50.0, 50.0))
      .map(d => java.lang.Float.intBitsToFloat(f32(d).toInt).toDouble)
      .distinct.sorted.reverse
    val ps = vs.indices.map(i => (tag * 100 + i).toDouble)
    (vs.zip(ps) ++ Seq.fill(k - vs.length)((ID_M_D, 0.0))).take(k)
  }

  "TopK_merge" should "produce the k largest of the union, descending" in {
    test(new DataPathJunctionHarness(hasTopK)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0x70b1)
      for (trial <- 0 until 8) {
        val A = randList(rng, k, 1)
        val B = randList(rng, k, 2)
        val out = runPair(dut, csrWord(k), pack(A.map(_._1), A.map(_._2)), pack(B.map(_._1), B.map(_._2)),
                          bDelay = trial % 3)
        val gold = topK(A ++ B)
        val gotV = unpackV(out)
        val gotP = unpackP(out)
        for (i <- 0 until k) {
          assert(gotV(i) == gold(i)._1, f"trial $trial entry $i: value ${gotV(i)}%.4f vs ${gold(i)._1}%.4f")
          assert(gotP(i) == gold(i)._2, f"trial $trial entry $i: payload ${gotP(i)} vs ${gold(i)._2}")
        }
        // and the result really is sorted -- the merge, not just the set
        for (i <- 1 until k) assert(gotV(i - 1) >= gotV(i), s"trial $trial: output not descending at $i")
      }
      println(s"[TopK] k=$k merge of two sorted lists == the k largest of the union, payloads carried, 8 trials")
    }
  }

  "TopK_O2_closure" should "let its own output be a legal input to the next hop" in {
    // THE differentiating obligation. A 1->1 extension transforms a stream and stops; a junction's output IS the
    // next hop's input, so a chain collective only composes if the operator is closed under its own format.
    // Feed instance 1's output back in against a third list and compare with a 3-input software reference.
    test(new DataPathJunctionHarness(hasTopK)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0xc105)
      for (trial <- 0 until 6) {
        val A = randList(rng, k, 1)
        val B = randList(rng, k, 2)
        val C = randList(rng, k, 3)
        val ab = runPair(dut, csrWord(k), pack(A.map(_._1), A.map(_._2)), pack(B.map(_._1), B.map(_._2)))
        // the SAME beat, unmodified, is fed straight back in -- no repack, no re-sort, no software in between
        val abc = runPair(dut, csrWord(k), ab, pack(C.map(_._1), C.map(_._2)), bDelay = trial % 3)
        val gold = topK(A ++ B ++ C)
        val gotV = unpackV(abc)
        val gotP = unpackP(abc)
        for (i <- 0 until k) {
          assert(gotV(i) == gold(i)._1, f"trial $trial entry $i: hop-2 value ${gotV(i)}%.4f vs ${gold(i)._1}%.4f")
          assert(gotP(i) == gold(i)._2, f"trial $trial entry $i: hop-2 payload ${gotP(i)} vs ${gold(i)._2}")
        }
      }
      println("[TopK/O2] the output beat re-enters the SAME operator unchanged and 3 lists fold to the true top-k")
    }
  }

  "TopK_O3_identity" should "leave the live entries untouched when the lists are short" in {
    test(new DataPathJunctionHarness(hasTopK)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0x1d11)
      for (live <- Seq(1, 3, 5, k)) {
        val A = randList(rng, live, 1)
        val B = randList(rng, live, 2)
        val out  = runPair(dut, csrWord(live), pack(A.map(_._1), A.map(_._2)), pack(B.map(_._1), B.map(_._2)))
        val gold = topK(A.take(live) ++ B.take(live) ++ Seq.fill(2 * k - 2 * live)((ID_M_D, 0.0)))
        val gotV = unpackV(out)
        for (i <- 0 until k)
          assert(gotV(i) == gold(i)._1, f"live=$live entry $i: ${gotV(i)}%.4f vs ${gold(i)._1}%.4f")
        // the dead tail is the identity, which is what keeps the short beat a legal input partial
        for (i <- math.min(2 * live, k) until k)
          assert(gotV(i) == ID_M_D, s"live=$live entry $i should be the identity, got ${gotV(i)}")
      }
      println("[TopK/O3] short lists pad with the identity; live entries are exactly the top-k of the live union")
    }
  }

  "TopK_O1_latency" should "retire at the declared constant, payload-independent" in {
    test(new DataPathJunctionHarness(hasTopK)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      def firstOut(n: Int): (Int, Double) = {
        val rng   = new Random(0x900d + n)
        val beats = Seq.fill(n) {
          val L = randList(rng, k, 1)
          pack(L.map(_._1), L.map(_._2))
        }
        dut.io.csr_i(0).poke(csrWord(k).U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.out_o.ready.poke(true)
        dut.io.a_i.valid.poke(true); dut.io.b_i.valid.poke(true)
        dut.io.a_i.bits.poke(beats(0).U); dut.io.b_i.bits.poke(beats(0).U)
        var cyc = 0; var fed = 0; var got = 0; var first = -1; var warm = -1; var last = -1
        while (got < n && cyc < n * 40 + 600) {
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
          if (outNow) { if (first < 0) first = cyc; got += 1 }
        }
        dut.io.out_o.ready.poke(false)
        assert(got == n, s"only $got/$n retired")
        (first, if (last > warm) (n - 4).toDouble / (last - warm) else 0.0)
      }
      val (l1, _)   = firstOut(1)
      val (l64, u)  = firstOut(64)
      println(f"[TopK/O1] first-out: N=1 -> $l1 CC, N=64 -> $l64 CC; util=$u%.3f (declared latency ${1 + 3})")
      assert(l64 <= l1 + 1, s"NOT cut-through: first output moved from $l1 to $l64 as the payload grew")
      assert(u > 0.9, f"a feed-forward merge should stream at one pair per cycle, got $u%.3f")
    }
  }

  // O4 needs two elaborations -- one isolated, one interleaved -- so it is split across two scalatest cases:
  // chiseltest gives each CASE one Verilator build directory and two `test(...)` calls in a case collide on it.
  private val o4Cases = {
    val rng = new Random(0x5747)
    Seq.fill(6)((randList(rng, k, 1), randList(rng, k, 2)))
  }
  private var o4Alone = Seq[BigInt]()

  "TopK_O4_isolated" should "record each pair run on its own (leg 1 of 2)" in {
    test(new DataPathJunctionHarness(hasTopK)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      o4Alone = o4Cases.map { case (la, lb) =>
        runPair(dut, csrWord(k), pack(la.map(_._1), la.map(_._2)), pack(lb.map(_._1), lb.map(_._2)))
      }
    }
  }

  "TopK_O4_stateless" should "give each pair the same result when interleaved with others" in {
    assert(o4Alone.length == o4Cases.length, "run TopK_O4_isolated first (leg 1 of 2)")
    test(new DataPathJunctionHarness(hasTopK)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      // same pairs, reversed order, varying operand skew -- nothing about a pair may depend on its neighbours
      val got = o4Cases.indices.reverse.map { i =>
        val (la, lb) = o4Cases(i)
        i -> runPair(dut, csrWord(k), pack(la.map(_._1), la.map(_._2)), pack(lb.map(_._1), lb.map(_._2)),
                     bDelay = i % 4)
      }.toMap
      for (i <- o4Cases.indices)
        assert(got(i) == o4Alone(i), f"pair $i differs when interleaved: 0x${got(i)}%x vs 0x${o4Alone(i)}%x")
      println("[TopK/O4] results are independent of what ran before them, and of the operand skew")
    }
  }
}
