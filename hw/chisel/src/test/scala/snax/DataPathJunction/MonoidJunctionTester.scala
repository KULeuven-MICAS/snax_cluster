package snax.DataPathJunction

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathJunction.JunctionTestUtils._

/** Tier-1 for `MonoidJunction` -- the nonlinear 2->1 collective fold, and the socket's twisted-family operator.
  *
  * This is the operator's own test, and it carries what `JunctionSocketTester` cannot: the tree's only
  * INDEPENDENT double-precision goldens. The socket tester compares against FROZEN beats, which proves agreement
  * with a captured result rather than correctness of the arithmetic; here every expected value is recomputed from
  * the algebra. It also holds the only end-to-end C1 chain-closure proof.
  *
  * What these tests pin down, beyond "the arithmetic is right":
  *
  *   1. Every geometry folds a live operand PAIR against a numeric golden computed in double precision.
  *   2. CHAINING: folding P = 4 partials as three successive pairwise junction ops reproduces the direct global
  *      answer, which is the reduce-along-the-route property at component level.
  *   3. CUT-THROUGH: the first output appears after the SAME number of cycles whether the payload is 1 beat or 64.
  *      A store-and-forward join would make first-output latency scale with payload.
  *   4. The fold streams at ~1 beat pair per cycle, since the combine holds no state between pairs.
  *   5. Bypass discipline and the starvation watchdog -- the two properties the Junction CLASS guarantees.
  */
class MonoidJunctionTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))
  private val dHead     = 8
  private val pairSlots = 8

  import MonoidCombine._

  // csr(0): [7:0] nValid | [11:8] n | [21:18] nExp | [25:22] nAdd | [27:26] sigma | [28] keyPol
  private def csrWord(n: Int, nExp: Int, nAdd: Int, sigma: Int, nValid: Int, keyMul: Int = 0): BigInt =
    (BigInt(keyMul) << 29) | (BigInt(sigma) << 26) | (BigInt(nAdd) << 22) | (BigInt(nExp) << 18) |
      (BigInt(n) << 8) | BigInt(nValid)

  private val MOMENT  = (1, 1, 0, 3)          // (m, l)               one twisted coordinate
  private val ATTN    = (1 + dHead, 1 + dHead, 0, 0) // (m, l, O[dHead])  one alpha across every value lane
  private val MAXPOOL = (0, 0, 0, 3)          // (m)                  a bare key
  private val ARGMAX  = (1, 0, 0, 3)          // (m, payload)         field 1 is past nExp+nAdd => winner-select
  private def csr(g: (Int, Int, Int, Int), nValid: Int): BigInt = csrWord(g._1, g._2, g._3, g._4, nValid)

  /** pack `pairSlots` (field0, field1) partials: field0_k = lane k, field1_k = lane pairSlots+k */
  private def packPairs(p: Seq[(Double, Double)]): BigInt = {
    var b = BigInt(0)
    for ((v, k) <- p.zipWithIndex) { b |= f32(v._1) << (32 * k); b |= f32(v._2) << (32 * (pairSlots + k)) }
    b
  }

  /** pack one (m, l, O[dHead]) partial across lanes 0..1+dHead */
  private def packAttn(m: Double, l: Double, o: Seq[Double]): BigInt = {
    var b = f32(m) | (f32(l) << 32)
    for ((ov, k) <- o.zipWithIndex) b |= f32(ov) << (32 * (2 + k))
    b
  }

  // These run against whatever hardware actually ships, since they are the tree's only INDEPENDENT
  // double-precision goldens and its only C1 chain-closure proof.
  private def hasMonoid = new HasMonoidJunction()

  // ---- goldens ----
  private def momentGolden(a: (Double, Double), b: (Double, Double)): (Double, Double) = {
    val m = math.max(a._1, b._1)
    (m, a._2 * math.exp(a._1 - m) + b._2 * math.exp(b._1 - m))
  }

  "MonoidJunction_plainSumFields" should "sum the value coordinates that lie past nExp, on the winner's key" in {
    // `nExp` twisted coordinates, then `nAdd` plainly summed ones. Here nExp = 0 and nAdd = 1, so the key still
    // decides the winner and the single value coordinate is added with no rescale -- the same FMA as a twisted
    // lane with its scale port held at 1.0.
    test(new DataPathJunctionHarness(hasMonoid)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0x5011)
      for (trial <- 0 until 6) {
        val pa  = Seq.fill(pairSlots)((rng.between(-4.0, 4.0), rng.between(0.0, 8.0)))
        val pb  = Seq.fill(pairSlots)((rng.between(-4.0, 4.0), rng.between(0.0, 8.0)))
        val w   = csrWord(n = 1, nExp = 0, nAdd = 1, sigma = 3, nValid = pairSlots)
        val out = runPair(dut, w, packPairs(pa), packPairs(pb), bDelay = trial % 3)
        for (k <- 0 until pairSlots) {
          val gm = math.max(pa(k)._1, pb(k)._1)
          val gv = pa(k)._2 + pb(k)._2
          // the key lane SELECTS an operand, so compare the encodings: the beat carries FP32 and a
          // double-precision golden differs from the right answer in the last mantissa bits.
          assert(lane(out, k) == f32(gm), s"trial $trial slot $k key")
          assert(math.abs(laneF32(out, pairSlots + k) - gv) <= math.abs(gv) * 1e-5 + 1e-6,
                 s"trial $trial slot $k value")
        }
      }
      println(s"[Junction/nAdd] $pairSlots keyed partials with an untwisted value coordinate, 6 trials exact")
    }
  }

  "MonoidJunction_MOMENT" should "fold two streams of softmax (m, l) normalizers slot-wise" in {
    test(new DataPathJunctionHarness(hasMonoid)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng   = new Random(0x2b2b)
      var worst = 0.0
      for (trial <- 0 until 6) {
        val pa  = Seq.fill(pairSlots)((rng.between(-3.0, 6.0), rng.between(1.0, 5.0)))
        val pb  = Seq.fill(pairSlots)((rng.between(-3.0, 6.0), rng.between(1.0, 5.0)))
        val out = runPair(dut, csr(MOMENT, pairSlots), packPairs(pa), packPairs(pb), bDelay = trial % 4)
        for (k <- 0 until pairSlots) {
          val (gm, gl) = momentGolden(pa(k), pb(k))
          val rel      = math.abs(laneF32(out, pairSlots + k) - gl) / gl
          if (rel > worst) worst = rel
          assert(math.abs(laneF32(out, k) - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"trial $trial slot $k m")
          assert(rel <= 1.5e-2, s"trial $trial slot $k l rel=$rel")
        }
      }
      println(f"[Junction/MOMENT] the NONLINEAR fold (max coupled to a rescaled sum) -- worst rel err=$worst%.3g")
    }
  }

  "MonoidJunction_ATTN" should "fold the single-partial (m, l, O) flash-attention triple" in {
    test(new DataPathJunctionHarness(hasMonoid)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0x0a44)
      for (trial <- 0 until 5) {
        val (ma, la, oa) = (rng.between(-2.0, 5.0), rng.between(1.0, 5.0), Seq.fill(dHead)(rng.between(-3.0, 3.0)))
        val (mb, lb, ob) = (rng.between(-2.0, 5.0), rng.between(1.0, 5.0), Seq.fill(dHead)(rng.between(-3.0, 3.0)))
        val out = runPair(dut, csr(ATTN, 1), packAttn(ma, la, oa), packAttn(mb, lb, ob), bDelay = trial % 3)
        val gm  = math.max(ma, mb)
        val gl  = la * math.exp(ma - gm) + lb * math.exp(mb - gm)
        assert(math.abs(laneF32(out, 0) - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"attn trial $trial m")
        assert(math.abs(laneF32(out, 1) - gl) / gl <= 1.5e-2, s"attn trial $trial l")
        for (j <- 0 until dHead) {
          val go = oa(j) * math.exp(ma - gm) + ob(j) * math.exp(mb - gm)
          val d  = math.abs(laneF32(out, 2 + j) - go) / (math.abs(go) + 1e-6)
          assert(d <= 2e-2, s"attn trial $trial O[$j] rel=$d")
        }
      }
      println(s"[Junction/ATTN] (m, l, O[$dHead]) triple folded from a live operand pair -- one shared alpha")
    }
  }

  "MonoidJunction_MAXPOOL_ARGMAX" should "max-reduce and carry the winner's index" in {
    test(new DataPathJunctionHarness(hasMonoid)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0x3a11)
      for (_ <- 0 until 4) {
        val va  = Seq.fill(pairSlots)(rng.between(-9.0, 9.0))
        val vb  = Seq.fill(pairSlots)(rng.between(-9.0, 9.0))
        val out = runPair(dut, csr(MAXPOOL, pairSlots), packPairs(va.map((_, 0.0))), packPairs(vb.map((_, 0.0))))
        for (k <- 0 until pairSlots) {
          val g = math.max(va(k), vb(k))
          assert(math.abs(laneF32(out, k) - g) <= math.abs(g) * 1e-6 + 1e-9, s"maxpool slot $k")
        }
      }
      // ARGMAX: field1 carries the index of the winning logit (distinct logits => unambiguous)
      for (trial <- 0 until 4) {
        val la  = Seq.fill(pairSlots)(rng.between(-9.0, 0.0))
        val lb  = Seq.fill(pairSlots)(rng.between(0.1, 9.0))
        val pa  = la.zipWithIndex.map { case (v, i) => (v, i.toDouble) }
        val pb  = lb.zipWithIndex.map { case (v, i) => (v, (pairSlots + i).toDouble) }
        val out = runPair(dut, csr(ARGMAX, pairSlots), packPairs(pa), packPairs(pb))
        for (k <- 0 until pairSlots) {
          val aWins = la(k) >= lb(k)
          val g     = math.max(la(k), lb(k))
          assert(math.abs(laneF32(out, k) - g) <= math.abs(g) * 1e-6 + 1e-9, s"argmax trial $trial slot $k max")
          assert(laneF32(out, pairSlots + k) == (if (aWins) k.toDouble else (pairSlots + k).toDouble),
                 s"argmax trial $trial slot $k idx")
        }
      }
      println("[Junction/MAXPOOL+ARGMAX] max-reduce and top-1 index carry over a live pair")
    }
  }


  // ================================================================================================
  // THE ORDERED SCAN -- keyMul = 1. The key monoid is (R, x) and the twist is the B-side key:
  //     (k_A, v_A) (+) (k_B, v_B)  =  ( k_A*k_B ,  k_B*v_A + v_B )
  // This is the chunked recurrence of linear attention / RetNet / GLA / Mamba-2, on the same lanes and
  // the same FMA as the softmax family. It is associative and DELIBERATELY NOT commutative.
  // ================================================================================================

  /** one scan step, in double precision, on one slot */
  private def scanStep(a: (Double, Double), b: (Double, Double)): (Double, Double) =
    (a._1 * b._1, b._1 * a._2 + b._2)

  /** The state update is a SUM of two terms, so its error must be judged against the larger term, not against
    * the result: `k_B*v_A + v_B` can cancel to nearly zero, and relative error on a cancelling sum is unbounded
    * in any finite precision. This is the standard way to state a floating-point summation bound.
    */
  private def sumScale(a: (Double, Double), b: (Double, Double)): Double =
    math.max(math.abs(b._1 * a._2), math.abs(b._2))

  "MonoidJunction_scan" should "fold a chunked linear-attention recurrence slot-wise" in {
    test(new DataPathJunctionHarness(hasMonoid)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng   = new Random(0x5ca7)
      var worst = 0.0
      for (trial <- 0 until 6) {
        // decays in (0,1) and states of order 1 -- the regime a real gated recurrence runs in
        val pa  = Seq.fill(pairSlots)((rng.between(0.1, 0.99), rng.between(-2.0, 2.0)))
        val pb  = Seq.fill(pairSlots)((rng.between(0.1, 0.99), rng.between(-2.0, 2.0)))
        val csr = csrWord(n = 1, nExp = 1, nAdd = 0, sigma = 3, nValid = pairSlots, keyMul = 1)
        val out = runPair(dut, csr, packPairs(pa), packPairs(pb), bDelay = trial % 3)
        for (k <- 0 until pairSlots) {
          val (gk, gv) = scanStep(pa(k), pb(k))
          val ek = math.abs(laneF32(out, k) - gk) / math.abs(gk)
          val ev = math.abs(laneF32(out, pairSlots + k) - gv) / sumScale(pa(k), pb(k))
          assert(ek <= 1e-6, f"trial $trial slot $k decay: ${laneF32(out, k)}%.8f vs $gk%.8f")
          assert(ev <= 1e-6, f"trial $trial slot $k state: ${laneF32(out, pairSlots + k)}%.8f vs $gv%.8f")
          worst = math.max(worst, math.max(ek, ev))
        }
      }
      println(f"[Junction/scan] $pairSlots independent (decay, state) recurrences per beat pair, " +
              f"worst rel err=$worst%.2e")
    }
  }

  "MonoidJunction_scan_chain" should "reproduce the global recurrence when 4 chunks fold along a chain" in {
    // Associativity in hardware, for the NON-commutative operator: folding chunk 0..3 hop by hop must equal
    // running the recurrence straight through. This is the property that makes the scan a legal collective.
    test(new DataPathJunctionHarness(hasMonoid)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng    = new Random(0x0117)
      val chunks = Seq.fill(4)(Seq.fill(pairSlots)((rng.between(0.2, 0.95), rng.between(-2.0, 2.0))))
      val csr    = csrWord(n = 1, nExp = 1, nAdd = 0, sigma = 3, nValid = pairSlots, keyMul = 1)
      // A = what arrived from the previous hop (the EARLIER chunks), B = this node's own. Order is the route.
      var acc = packPairs(chunks(0))
      for (hop <- 1 until 4) acc = runPair(dut, csr, acc, packPairs(chunks(hop)), bDelay = hop)
      var worst = 0.0
      for (k <- 0 until pairSlots) {
        val gold  = (1 until 4).foldLeft(chunks(0)(k))((st, h) => scanStep(st, chunks(h)(k)))
        // the chain accumulates three sums, so scale by the largest term seen at any hop
        val scale = (1 until 4).foldLeft((chunks(0)(k), 0.0)) { case ((st, mx), h) =>
          (scanStep(st, chunks(h)(k)), math.max(mx, sumScale(st, chunks(h)(k))))
        }._2
        val ek   = math.abs(laneF32(acc, k) - gold._1) / math.abs(gold._1)
        val ev   = math.abs(laneF32(acc, pairSlots + k) - gold._2) / scale
        assert(ek <= 5e-6, f"slot $k decay: ${laneF32(acc, k)}%.8f vs ${gold._1}%.8f")
        assert(ev <= 5e-6, f"slot $k state: ${laneF32(acc, pairSlots + k)}%.8f vs ${gold._2}%.8f")
        worst = math.max(worst, math.max(ek, ev))
      }
      println(f"[Junction/scan] 4 chunks folded hop-by-hop == the direct recurrence, worst rel err=$worst%.2e")
    }
  }

  "MonoidJunction_scan_ordered" should "be deliberately NOT commutative, with A the earlier chunk" in {
    // The obligation this operator adds to the socket. Every other operator here is commutative, so the chassis
    // has never had to promise an operand ORDER. This one does: A is what arrived from upstream, i.e. earlier in
    // the sequence. Swapping the operands must change the answer -- if it did not, the decay would be applied to
    // the wrong side and the recurrence would be silently wrong rather than loudly different.
    test(new DataPathJunctionHarness(hasMonoid)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0x0dd1)
      val pa  = Seq.fill(pairSlots)((rng.between(0.2, 0.9), rng.between(-2.0, 2.0)))
      val pb  = Seq.fill(pairSlots)((rng.between(0.2, 0.9), rng.between(-2.0, 2.0)))
      val csr = csrWord(n = 1, nExp = 1, nAdd = 0, sigma = 3, nValid = pairSlots, keyMul = 1)
      val ab  = runPair(dut, csr, packPairs(pa), packPairs(pb))
      val ba  = runPair(dut, csr, packPairs(pb), packPairs(pa))
      assert(ab != ba, "the scan returned the same beat with the operands swapped -- it is not ordered")
      // the key field IS commutative (a product), so only the state lanes may differ
      for (k <- 0 until pairSlots)
        assert(lane(ab, k) == lane(ba, k), s"slot $k: the decay product must not depend on operand order")
      // and A is the earlier chunk: A's state is the one that gets scaled
      for (k <- 0 until pairSlots) {
        val (_, gv) = scanStep(pa(k), pb(k))
        assert(math.abs(laneF32(ab, pairSlots + k) - gv) / sumScale(pa(k), pb(k)) <= 1e-6,
               s"slot $k: A is not being treated as the earlier chunk")
      }
      println("[Junction/scan] operand order is semantic: A = upstream = earlier; the decay product is not")
    }
  }

  "MonoidJunction_scan_identity" should "pad a dead slot with (1, 0), the scan's identity" in {
    // O3 for a monoid whose key identity is NOT the losing extreme. A max-monoid pad here would multiply the
    // running decay by -3.4e38 and annihilate the chain; the neutral element of (R, x) is 1.0.
    test(new DataPathJunctionHarness(hasMonoid)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val live = 3
      val rng  = new Random(0x1de1)
      val pa   = Seq.fill(pairSlots)((rng.between(0.2, 0.9), rng.between(-2.0, 2.0)))
      val pb   = Seq.fill(pairSlots)((rng.between(0.2, 0.9), rng.between(-2.0, 2.0)))
      val csr  = csrWord(n = 1, nExp = 1, nAdd = 0, sigma = 3, nValid = live, keyMul = 1)
      val out  = runPair(dut, csr, packPairs(pa), packPairs(pb))
      for (k <- 0 until live) {
        val (gk, gv) = scanStep(pa(k), pb(k))
        assert(math.abs(laneF32(out, k) - gk) / math.abs(gk) <= 1e-6, s"live slot $k decay")
        assert(math.abs(laneF32(out, pairSlots + k) - gv) / sumScale(pa(k), pb(k)) <= 1e-6,
               s"live slot $k state")
      }
      // a dead slot folds identity against identity: 1*1 = 1, and 1*0 + 0 = 0. Both must be exact.
      for (k <- live until pairSlots) {
        assert(laneF32(out, k) == 1.0, s"dead slot $k decay should be exactly 1.0, got ${laneF32(out, k)}")
        assert(laneF32(out, pairSlots + k) == 0.0, s"dead slot $k state should be exactly 0.0")
      }
      println(s"[Junction/scan] slots >= $live carry (1.0, 0.0) exactly -- the neutral element of (R, x)")
    }
  }

  "MonoidJunction_chain" should "reproduce the global fold when P=4 partials are reduced along a chain" in {
    test(new DataPathJunctionHarness(hasMonoid)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      // Node i folds (arriving partial, local partial) and forwards. The output beat uses the SAME layout as the
      // operands, so the chain is closed under its own format: a 4-node chain is three successive junction ops and
      // must equal the direct global merge.
      val rng = new Random(0x7c1a)
      for (trial <- 0 until 4) {
        val shards = Seq.fill(4)(Seq.fill(pairSlots)((rng.between(-3.0, 6.0), rng.between(1.0, 5.0))))
        var acc    = packPairs(shards.head)
        for (hop <- 1 until 4)
          acc = runPair(dut, csr(MOMENT, pairSlots), acc, packPairs(shards(hop)), bDelay = hop)
        for (k <- 0 until pairSlots) {
          val slot = shards.map(_(k))
          val gm   = slot.map(_._1).max
          val gl   = slot.map { case (m, l) => l * math.exp(m - gm) }.sum
          assert(math.abs(laneF32(acc, k) - gm) <= math.abs(gm) * 1e-6 + 1e-9, s"chain trial $trial slot $k m")
          assert(math.abs(laneF32(acc, pairSlots + k) - gl) / gl <= 3e-2, s"chain trial $trial slot $k l")
        }
      }
      println("[Junction/chain] P=4 hop-by-hop fold == the direct global merge (associativity, in hardware)")
    }
  }

  "MonoidJunction_cutthrough" should "emit its first beat after a payload-INDEPENDENT latency" in {
    test(new DataPathJunctionHarness(hasMonoid)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      // Feed N pairs back-to-back and time the FIRST output. A store-and-forward join would assemble the whole
      // payload first, so this number would grow with N.
      def firstOutLatency(n: Int): (Int, Double) = {
        val rng   = new Random(0x99 + n)
        val beats = Seq.fill(n)(packPairs(Seq.fill(pairSlots)((rng.between(-2.0, 4.0), rng.between(0.5, 3.0)))))
        dut.io.csr_i(0).poke(csr(MOMENT, pairSlots).U)
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
        assert(got == n, s"only $got/$n beats retired at N=$n")
        (first, if (last > warm) (n - 4).toDouble / (last - warm) else 0.0)
      }
      val (lat1, _)     = firstOutLatency(1)
      val (lat64, util) = firstOutLatency(64)
      println(f"[Junction/cut-through] first-output latency: N=1 -> $lat1 CC, N=64 -> $lat64 CC; stream util=$util%.3f")
      assert(lat64 <= lat1 + 1,
             s"NOT cut-through: first output slipped from $lat1 CC at N=1 to $lat64 CC at N=64 (store-and-forward)")
      assert(lat1 <= 12, s"per-hop latency $lat1 CC exceeds budget (the harness adds 2 cut stages)")
      // The fold is stateless in the operand pair, so it streams at one beat pair per cycle.
      assert(util > 0.9, f"junction fold should stream at ~1 beat pair/cycle, got $util%.3f")
    }
  }

  "MonoidJunction_bypass" should "be a transparent wire from a to out when disabled" in {
    test(new DataPathJunctionHarness(hasMonoid)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng   = new Random(0xbb01)
      val beats = Seq.fill(8)(packPairs(Seq.fill(pairSlots)((rng.between(-4.0, 4.0), rng.between(0.5, 3.0)))))
      dut.io.csr_i(0).poke(csr(MOMENT, pairSlots).U)
      dut.io.enable_i.poke(false) // the bypass property every Junction inherits from the class
      dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
      dut.io.out_o.ready.poke(true)
      dut.io.b_i.valid.poke(false)
      var got = List[BigInt]()
      dut.io.a_i.valid.poke(true); dut.io.a_i.bits.poke(beats(0).U)
      var cyc = 0; var fed = 0
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
      println("[Junction/bypass] enable=0 is byte-identical to a raw forward")
    }
  }

  "MonoidJunction_watchdog" should "flag a starved join instead of hanging silently" in {
    test(new DataPathJunctionHarness(new HasMonoidJunction(starveLimit = 64)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        // A two-stream join stalls indefinitely if the operand beat counts disagree, which is a software contract
        // on two independently dispatched cfgs. The watchdog turns that silent stall into an observable flag.
        dut.io.csr_i(0).poke(csr(MOMENT, pairSlots).U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.out_o.ready.poke(true)
        dut.io.b_i.valid.poke(false) // the local operand never arrives
        dut.io.a_i.valid.poke(true); dut.io.a_i.bits.poke(BigInt(1).U)
        var cyc = 0
        while (!dut.io.starved_o.peekBoolean() && cyc < 400) { dut.clock.step(1); cyc += 1 }
        assert(dut.io.starved_o.peekBoolean(), "starvation watchdog never fired on a one-sided join")
        // and it clears once the missing operand shows up
        dut.io.b_i.valid.poke(true); dut.io.b_i.bits.poke(BigInt(1).U)
        dut.clock.step(4)
        assert(!dut.io.starved_o.peekBoolean(), "watchdog must clear once the join makes progress")
        println(s"[Junction/watchdog] starved join flagged after $cyc CC and cleared on progress")
      }
  }
}
