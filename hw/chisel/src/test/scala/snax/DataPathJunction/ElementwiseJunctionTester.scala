package snax.DataPathJunction

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import snax.DataPathExtension.FpHelpers
import snax.DataPathJunction.JunctionTestUtils._

/** Tier-1 for `ElementwiseJunction` -- the linear 2->1 fold.
  *
  * Covers the four ops at FP16, the runtime format select (BF16 and FP32 out of one netlist), the INTEGER grid
  * and the three properties that only it has -- exactness, route-independence and idempotent max -- the streaming
  * and cut-through behaviour, and the class-level bypass property.
  *
  * The last test is the reason the monoid class exists at all: it shows directly that a per-element
  * reduction applied to online-softmax statistics computes the WRONG answer, because the softmax merge needs a
  * rescale derived from the operand pair and is not a per-element operation at any granularity.
  */
class ElementwiseJunctionTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  import ElementwiseJunction._

  private def csrWord(op: Int, fmt: Int): BigInt = (BigInt(fmt) << 4) | BigInt(op)

  // ---- transport-format codecs (goldens must be computed on the DECODED operands, not the doubles) ----
  private def encFp16(d: Double): BigInt = BigInt(java.lang.Float.floatToFloat16(d.toFloat) & 0xffff)
  private def decFp16(b: BigInt): Double = java.lang.Float.float16ToFloat(b.toShort).toDouble
  private def encBf16(d: Double): BigInt = {
    val b = java.lang.Float.floatToIntBits(d.toFloat)
    BigInt(((b + 0x7fff + ((b >>> 16) & 1)) >>> 16) & 0xffff)
  }
  private def decBf16(b: BigInt): Double = java.lang.Float.intBitsToFloat((b.toInt & 0xffff) << 16).toDouble

  private case class Fmt(code: Int, width: Int, enc: Double => BigInt, dec: BigInt => Double, name: String) {
    def lanes: Int = 512 / width
  }
  private val FMT16  = Fmt(FpHelpers.FMT_FP16, 16, encFp16, decFp16, "FP16")
  private val FMTBF  = Fmt(FpHelpers.FMT_BF16, 16, encBf16, decBf16, "BF16")
  private val FMT32  = Fmt(FMT_FP32, 32, f32, dec, "FP32")

  private def pack(v: Seq[Double], f: Fmt): BigInt = {
    var b = BigInt(0)
    for ((x, i) <- v.zipWithIndex) b |= f.enc(x) << (f.width * i)
    b
  }
  private def unpack(beat: BigInt, f: Fmt): Seq[Double] =
    (0 until f.lanes).map(i => f.dec(lane(beat, i, f.width)))

  private def hasEw = new HasElementwiseJunction(elemWidth = 16, fpPipe = 1)

  private def checkOp(dut: DataPathJunctionHarness, f: Fmt, op: Int, opName: String, ref: (Double, Double) => Double,
                      relTol: Double, seed: Int): Double = {
    val rng   = new Random(seed)
    var worst = 0.0
    for (trial <- 0 until 4) {
      val va  = Seq.fill(f.lanes)(rng.between(-4.0, 4.0))
      val vb  = Seq.fill(f.lanes)(rng.between(-4.0, 4.0))
      val (ba, bb) = (pack(va, f), pack(vb, f))
      val out = runPair(dut, csrWord(op, f.code), ba, bb, bDelay = trial % 3)
      val got = unpack(out, f)
      // golden on the DECODED operands: the transport round-trip is not part of what is under test
      val gold = unpack(ba, f).zip(unpack(bb, f)).map { case (x, y) => ref(x, y) }
      for (i <- 0 until f.lanes) {
        val rel = math.abs(got(i) - gold(i)) / (math.abs(gold(i)) + 1e-6)
        if (rel > worst) worst = rel
        assert(rel <= relTol, s"${f.name}/$opName trial $trial lane $i: got ${got(i)} vs ${gold(i)} (rel=$rel)")
      }
    }
    worst
  }

  "ElementwiseJunction_FP16" should "reduce two streams with ADD / MUL / MAX / MIN" in {
    test(new DataPathJunctionHarness(hasEw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val wAdd = checkOp(dut, FMT16, OP_ADD, "ADD", _ + _, 1.5e-3, 0x1001)
      val wMul = checkOp(dut, FMT16, OP_MUL, "MUL", _ * _, 1.5e-3, 0x1002)
      val wMax = checkOp(dut, FMT16, OP_MAX, "MAX", math.max, 1e-9, 0x1003)
      val wMin = checkOp(dut, FMT16, OP_MIN, "MIN", math.min, 1e-9, 0x1004)
      println(f"[EW/FP16] ${FMT16.lanes} lanes/beat -- ADD $wAdd%.2g MUL $wMul%.2g MAX $wMax%.2g MIN $wMin%.2g (rel)")
    }
  }

  "ElementwiseJunction_runtimeFmt" should "serve BF16 and FP32 from the SAME netlist" in {
    test(new DataPathJunctionHarness(hasEw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      // Internal math is FP32, so the transport format is an EDGE problem: one lane grid, three formats,
      // selected per transfer by the `fmt` CSR field. The reduction itself is unaffected.
      val wBf = checkOp(dut, FMTBF, OP_ADD, "ADD", _ + _, 1e-2, 0x2001)
      val w32 = checkOp(dut, FMT32, OP_ADD, "ADD", _ + _, 1e-6, 0x2002)
      val m32 = checkOp(dut, FMT32, OP_MUL, "MUL", _ * _, 1e-6, 0x2003)
      println(f"[EW/runtime-fmt] BF16 add $wBf%.2g | FP32 add $w32%.2g mul $m32%.2g -- one netlist, fmt by CSR")
    }
  }

  "ElementwiseJunction_stream" should "reduce at one beat pair per cycle with cut-through latency" in {
    test(new DataPathJunctionHarness(hasEw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      def run(n: Int): (Int, Double) = {
        val rng   = new Random(0x55 + n)
        val beats = Seq.fill(n)(pack(Seq.fill(FMT16.lanes)(rng.between(-2.0, 2.0)), FMT16))
        dut.io.csr_i(0).poke(csrWord(OP_ADD, FMT16.code).U)
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
      val (lat1, _)     = run(1)
      val (lat64, util) = run(64)
      println(f"[EW/stream] first-output latency N=1 -> $lat1 CC, N=64 -> $lat64 CC; util=$util%.3f")
      assert(lat64 <= lat1 + 1, s"NOT cut-through: $lat1 CC at N=1 vs $lat64 CC at N=64")
      assert(util > 0.9, f"linear fold should stream at ~1 beat pair/cycle, got $util%.3f")
    }
  }

  "ElementwiseJunction_bypass" should "be a transparent wire from a to out when disabled" in {
    test(new DataPathJunctionHarness(hasEw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng   = new Random(0xcc02)
      val beats = Seq.fill(8)(pack(Seq.fill(FMT16.lanes)(rng.between(-3.0, 3.0)), FMT16))
      dut.io.csr_i(0).poke(csrWord(OP_ADD, FMT16.code).U)
      dut.io.enable_i.poke(false)
      dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
      dut.io.out_o.ready.poke(true)
      dut.io.b_i.valid.poke(false)
      dut.io.a_i.valid.poke(true); dut.io.a_i.bits.poke(beats(0).U)
      var got = List[BigInt](); var cyc = 0; var fed = 0
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
      println("[EW/bypass] enable=0 is byte-identical to a raw forward")
    }
  }


  // ================================================================================================
  // THE INTEGER GRID -- exact, native-width, and the merge rule of the standard mergeable summaries.
  // Every golden below is computed in Scala from the same integers the hardware sees, so these are
  // INDEPENDENT numeric checks, not frozen beats: integer arithmetic has no rounding to agree about.
  // ================================================================================================

  /** pack `512/w` two's-complement lanes, low lane first */
  private def packInt(v: Seq[BigInt], w: Int): BigInt = {
    val mask = (BigInt(1) << w) - 1
    v.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (x, i)) => acc | ((x & mask) << (w * i)) }
  }
  /** unpack as SIGNED lanes */
  private def unpackInt(beat: BigInt, w: Int): Seq[BigInt] = {
    val mask = (BigInt(1) << w) - 1
    (0 until 512 / w).map { i =>
      val u = (beat >> (w * i)) & mask
      if (u.testBit(w - 1)) u - (BigInt(1) << w) else u
    }
  }
  private def wrap(x: BigInt, w: Int): BigInt = {
    val u = x & ((BigInt(1) << w) - 1)
    if (u.testBit(w - 1)) u - (BigInt(1) << w) else u
  }

  private def checkInt(dut: DataPathJunctionHarness, code: Int, w: Int, op: Int, opName: String,
                       ref: (BigInt, BigInt) => BigInt, seed: Int): Unit = {
    val rng  = new Random(seed)
    val n    = 512 / w
    val lo   = -(BigInt(1) << (w - 1))
    val hi   = (BigInt(1) << (w - 1)) - 1
    def draw() = BigInt(rng.between(lo.toLong, hi.toLong + 1))
    for (trial <- 0 until 4) {
      // trial 0 pins the corners on lane 0 and 1 so wraparound is exercised deliberately, not by luck
      val va = if (trial == 0) Seq(hi, lo) ++ Seq.fill(n - 2)(draw()) else Seq.fill(n)(draw())
      val vb = if (trial == 0) Seq(hi, lo) ++ Seq.fill(n - 2)(draw()) else Seq.fill(n)(draw())
      val out = unpackInt(runPair(dut, csrWord(op, code), packInt(va, w), packInt(vb, w), bDelay = trial % 3), w)
      for (i <- 0 until n) {
        val g = wrap(ref(va(i), vb(i)), w)
        assert(out(i) == g, s"INT$w/$opName trial $trial lane $i: ${out(i)} != $g  (a=${va(i)} b=${vb(i)})")
      }
    }
    println(s"[EW/INT$w] $opName exact on $n native lanes, 4 trials incl. the wraparound corners")
  }

  "ElementwiseJunction_integer" should "reduce INT8 / INT16 / INT32 exactly, at native width" in {
    test(new DataPathJunctionHarness(hasEw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      for ((code, w, seed) <- Seq((FMT_INT8, 8, 0x1a), (FMT_INT16, 16, 0x2b), (FMT_INT32, 32, 0x3c))) {
        checkInt(dut, code, w, OP_ADD, "ADD", (a, b) => a + b, seed)
        checkInt(dut, code, w, OP_MUL, "MUL", (a, b) => a * b, seed + 1)
        checkInt(dut, code, w, OP_MAX, "MAX", (a, b) => a.max(b), seed + 2)
        checkInt(dut, code, w, OP_MIN, "MIN", (a, b) => a.min(b), seed + 3)
      }
    }
  }

  "ElementwiseJunction_routeIndependence" should "return ONE answer over every reduction order, in integer" in {
    // THE PROPERTY FLOATING-POINT ADDITION DOES NOT HAVE. An in-network reducer that adds in FP returns an
    // answer that depends on the reduction tree, because FP addition is not associative. Wrapping integer
    // addition is the group operation of Z_2^w -- exactly associative, exactly commutative -- so every one of
    // the 4! chain orders of the same four shards must be BIT-IDENTICAL.
    //
    // Both arms run here so the contrast is measured rather than asserted: same shards, same operator, same
    // netlist, only `fmt` differs.
    test(new DataPathJunctionHarness(hasEw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val orders = Seq(0, 1, 2, 3).permutations.toSeq
      def foldAll(csr: BigInt, beat: Int => BigInt): Seq[BigInt] =
        orders.map(o => o.tail.foldLeft(beat(o.head))((acc, i) => runPair(dut, csr, acc, beat(i))))

      // ---- arm 1: INT32, wrapping ------------------------------------------------------------------------
      val w    = 32
      val n    = 512 / w
      val rngI = new Random(0xa550c)
      val si   = Seq.fill(4)(Seq.fill(n)(BigInt(rngI.between(-(1L << 30), 1L << 30))))
      val ri   = foldAll(csrWord(OP_ADD, FMT_INT32), i => packInt(si(i), w)).distinct
      assert(ri.length == 1,
             s"wrapping integer add is not route-independent: ${ri.length} distinct answers over ${orders.length}")
      val goldI = (0 until n).map(i => wrap(si.map(_(i)).sum, w))
      assert(unpackInt(ri.head, w) == goldI, "route-independent, but not equal to the reference sum")

      // ---- arm 2: FP32, the same shape ---------------------------------------------------------------------
      // Magnitudes are spread deliberately: cancellation is where FP associativity fails, and a reducer that
      // never sees it would look reproducible by luck.
      val rngF = new Random(0xf10a7)
      val sf   = Seq.fill(4)(Seq.fill(16)(rngF.between(-1.0, 1.0) * math.pow(2.0, rngF.between(-12, 13))))
      val fp32 = Fmt(FMT_FP32, 32, d => BigInt(java.lang.Float.floatToIntBits(d.toFloat).toLong & 0xffffffffL),
                     b => java.lang.Float.intBitsToFloat(b.toInt).toDouble, "FP32")
      val rf   = foldAll(csrWord(OP_ADD, FMT_FP32), i => pack(sf(i), fp32)).distinct

      println(f"[EW/route] ${orders.length} reduction orders of 4 shards: " +
              f"INT32 -> ${ri.length} distinct beat(s), FP32 -> ${rf.length}")
      assert(rf.length > 1,
             "FP32 addition returned one answer over all orders -- the stimulus is not exercising cancellation, " +
               "so the integer arm's route-independence is not evidence of anything")
      println("[EW/route] the integer grid is certified route-independent; the FP grid is measurably not")
    }
  }


  "ElementwiseJunction_hyperloglog" should "merge HyperLogLog registers by per-register max, over a chain" in {
    // The most widely deployed mergeable summary there is. An HLL's registers merge by elementwise max, so the
    // union of four shards' sketches is one INT8 MAX fold along the chain -- and because max is idempotent and
    // commutative the answer cannot depend on the route or on a duplicated beat.
    test(new DataPathJunctionHarness(hasEw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val w   = 8
      val n   = 512 / w // 64 registers per beat
      val rng = new Random(0x1177)
      val shard = Seq.fill(4)(Seq.fill(n)(BigInt(rng.between(0, 40)))) // HLL registers: small, non-negative
      val csr = csrWord(OP_MAX, FMT_INT8)
      var acc = packInt(shard(0), w)
      for (h <- 1 until 4) acc = runPair(dut, csr, acc, packInt(shard(h), w), bDelay = h)
      val gold = (0 until n).map(i => shard.map(_(i)).max)
      assert(unpackInt(acc, w) == gold, "chained HLL register merge != the elementwise max of the four shards")
      // idempotence: folding a shard in twice cannot change the result
      val again = runPair(dut, csr, acc, packInt(shard(2), w))
      assert(again == acc, "a duplicated beat changed the result -- max is not being applied idempotently")
      println(s"[EW/HLL] 64 registers merged over a 4-shard chain; a duplicate beat is a no-op")
    }
  }

  "ElementwiseJunction_intClosure" should "let an integer output beat re-enter as an input beat" in {
    // O2 on the integer grid. Nothing about the format changes across the operator, so the third shard folds in
    // with no repack -- the same obligation the monoid and top-k operators are checked against.
    test(new DataPathJunctionHarness(hasEw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val w = 16; val n = 512 / w
      val rng = new Random(0xc105e)
      val s = Seq.fill(3)(Seq.fill(n)(BigInt(rng.between(-3000, 3000))))
      val csr = csrWord(OP_ADD, FMT_INT16)
      val ab  = runPair(dut, csr, packInt(s(0), w), packInt(s(1), w))
      val abc = runPair(dut, csr, ab, packInt(s(2), w), bDelay = 2)
      val gold = (0 until n).map(i => wrap(s.map(_(i)).sum, w))
      assert(unpackInt(abc, w) == gold, "the output beat is not a legal input beat on the integer grid")
      println("[EW/O2] an INT16 sum beat re-enters the same operator unchanged and a third shard folds in")
    }
  }

  "ElementwiseJunction_vs_monoid" should "be provably unable to express the softmax merge" in {
    // Feed the SAME (m, l) partial pair to the linear reduction and check it against the online-softmax golden.
    // A per-element ADD gets `m` wrong (it sums two maxima) and `l` wrong (it omits the exp(m_loser - m*)
    // rescale). No choice of per-element op fixes this, because the correct `l` depends on BOTH fields of BOTH
    // operands. The monoid class computes it at the same arity, hop count and beat rate -- see MonoidJunctionTester.
    test(new DataPathJunctionHarness(hasEw)).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val (ma, la, mb, lb) = (1.0, 2.0, 3.0, 0.5)
      val mStar = math.max(ma, mb)
      val lStar = la * math.exp(ma - mStar) + lb * math.exp(mb - mStar)
      val va    = Seq(ma, la) ++ Seq.fill(FMT32.lanes - 2)(0.0)
      val vb    = Seq(mb, lb) ++ Seq.fill(FMT32.lanes - 2)(0.0)
      val out   = unpack(runPair(dut, csrWord(OP_ADD, FMT32.code), pack(va, FMT32), pack(vb, FMT32)), FMT32)
      assert(math.abs(out(0) - (ma + mb)) <= 1e-5, "the linear reduction did add the two maxima")
      assert(math.abs(out(0) - mStar) > 1e-3, "expected the linear m to differ from max(ma, mb)")
      assert(math.abs(out(1) - lStar) > 1e-3, "expected the linear l to differ from the rescaled sum")
      println(f"[EW vs Monoid] linear 2-input reduction on (m,l): got (${out(0)}%.4f, ${out(1)}%.4f), " +
              f"softmax merge needs ($mStar%.4f, $lStar%.4f) -- not expressible per element")
    }
  }
}
