package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** StreamReduceRt: one elaborated netlist reduces a streamed row to a scalar (ADD / SUMSQ / MAX) at FP16 /
  * BF16 / FP8 chosen at runtime by the `fmt` CSR field. Proves (a) ADD/SUMSQ/MAX correctness at FP16 (32
  * lanes), (b) the critical MAX-over-an-all-negative-FP16-row test (the high 32 padding lanes must be masked
  * to -inf, else MAX picks the padding 0.0), (c) an FP8 reduce on the SAME instance (64 lanes), and (d) full
  * 512 b/cyc input throughput at computeLanes=maxLanes with the parallel fold.
  */
class StreamReduceRtTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  def f16bitsToF32(h: Int): Float = {
    val sign = if ((h & 0x8000) != 0) -1.0 else 1.0
    val exp  = (h >> 10) & 0x1f; val mant = h & 0x3ff
    (if (exp == 0) sign * mant * math.pow(2, -24)
     else if (exp == 0x1f) if (mant == 0) sign * Double.PositiveInfinity else Double.NaN
     else sign * (1024 + mant) * math.pow(2, exp - 25)).toFloat
  }
  def f32ToF16bits(f: Float): Int = {
    val bits = java.lang.Float.floatToIntBits(f); val sign = (bits >>> 16) & 0x8000
    val rawe = (bits >>> 23) & 0xff; val mant = bits & 0x7fffff
    if (rawe == 0xff) return sign | 0x7c00 | (if (mant != 0) 0x200 else 0)
    val exp = rawe - 127 + 15
    if (exp >= 0x1f) return sign | 0x7c00
    if (exp <= 0) { if (exp < -10) return sign
      val m = mant | 0x800000; val shift = 14 - exp; val half = m >>> shift
      val rem = m & ((1 << shift) - 1); val hw = 1 << (shift - 1)
      var r = half; if (rem > hw || (rem == hw && (half & 1) == 1)) r += 1; return sign | r }
    var h = sign | (exp << 10) | (mant >>> 13); val rem = mant & 0x1fff
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1)) h += 1; h
  }
  // FP8 e5m2 (1 sign, 5 exp bias 15, 2 mant) — matches the widenRt FP8 grid
  def f8e5m2ToF32(b: Int): Float = {
    val sign = if ((b & 0x80) != 0) -1.0 else 1.0
    val exp  = (b >> 2) & 0x1f; val mant = b & 0x3
    (if (exp == 0) sign * mant * math.pow(2, -16)
     else sign * (4 + mant) * math.pow(2, exp - 17)).toFloat
  }
  def packF16(v: Seq[Int]): BigInt = v.zipWithIndex.foldLeft(BigInt(0)) { case (a, (h, i)) => a | (BigInt(h & 0xffff) << (16 * i)) }
  def packF8(v: Seq[Int]):  BigInt = v.zipWithIndex.foldLeft(BigInt(0)) { case (a, (x, i)) => a | (BigInt(x & 0xff) << (8 * i)) }
  def csr0(oc: Int, fmt: Int): BigInt = BigInt(oc) | (BigInt(fmt) << 16)
  def normalF8(rng: Random): Int = (if (rng.nextBoolean()) 0x80 else 0) | ((1 + rng.nextInt(30)) << 2) | rng.nextInt(4)
  val OP_MAX = 0; val OP_ADD = 1; val OP_SUMSQ = 2

  /** Drive `beats.length` operand beats through the reduce and return the single (splatted) output beat. */
  def runReduce(dut: DataPathExtensionHarness, fmt: Int, op: Int, beats: Seq[BigInt]): BigInt = {
    dut.io.csr_i(0).poke(csr0(beats.length, fmt).U); dut.io.csr_i(1).poke(BigInt(op).U)
    dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    var out = BigInt(0)
    var threads = new chiseltest.internal.TesterThreadList(Seq())
    threads = threads.fork {
      for (bt <- beats) {
        dut.io.data_i.bits.poke(bt.U); dut.io.data_i.valid.poke(true)
        while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
        dut.clock.step(1)
      }
      dut.io.data_i.valid.poke(false)
    }
    threads = threads.fork {
      while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
      out = dut.io.data_o.bits.peekInt()
      dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false)
    }
    threads.joinAndStep()
    dut.io.data_o.ready.poke(true)
    var w = 0; while (dut.io.busy_o.peekBoolean() && w < 300) { dut.clock.step(1); w += 1 }
    dut.io.data_o.ready.poke(false)
    out
  }

  "StreamReduceRt_runtime" should "reduce ADD/SUMSQ/MAX at FP16 and MAX/ADD at FP8 on ONE netlist (cl=64)" in {
    test(new DataPathExtensionHarness(new HasStreamReduceRt(computeLanes = 64, op = Seq("FMA", "MAX"))))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0xd0)
        // --- FP16 ADD: Σ over 32 lanes × 2 beats ---
        val b0 = Seq.fill(32)(f32ToF16bits(rng.between(-2f, 2f)))
        val b1 = Seq.fill(32)(f32ToF16bits(rng.between(-2f, 2f)))
        val oAdd  = runReduce(dut, 0, OP_ADD, Seq(packF16(b0), packF16(b1)))
        val gAdd  = (b0 ++ b1).map(f16bitsToF32).sum
        val rAdd  = f16bitsToF32((oAdd & 0xffff).toInt)
        assert(math.abs(rAdd - gAdd) <= math.max(math.abs(gAdd) * 0.03f, 0.05f), s"FP16 ADD hw=$rAdd gold=$gAdd")
        // --- FP16 SUMSQ: Σ x² over 32 lanes ---
        val sq   = Seq.fill(32)(f32ToF16bits(rng.between(-2f, 2f)))
        val oSq  = runReduce(dut, 0, OP_SUMSQ, Seq(packF16(sq)))
        val gSq  = sq.map(f16bitsToF32).map(x => x * x).sum
        val rSq  = f16bitsToF32((oSq & 0xffff).toInt)
        assert(math.abs(rSq - gSq) <= math.max(math.abs(gSq) * 0.03f, 0.05f), s"FP16 SUMSQ hw=$rSq gold=$gSq")
        // --- FP16 MAX over an ALL-NEGATIVE row: the high 32 padding lanes MUST be masked to -inf ---
        val neg   = Seq.fill(32)(f32ToF16bits(-(rng.between(0.5f, 8f))))
        val oNeg  = runReduce(dut, 0, OP_MAX, Seq(packF16(neg)))
        val gNeg  = neg.map(f16bitsToF32).max
        val rNeg  = f16bitsToF32((oNeg & 0xffff).toInt)
        assert(rNeg == gNeg, s"FP16 MAX(all-neg) hw=$rNeg gold=$gNeg (padding-lane masking broken?)")
        // --- FP8 MAX over 64 lanes on the SAME instance ---
        val f8   = Seq.fill(64)(normalF8(rng))
        val oF8  = runReduce(dut, 2, OP_MAX, Seq(packF8(f8)))
        val gF8  = f8.map(f8e5m2ToF32).max
        val rF8  = f8e5m2ToF32((oF8 & 0xff).toInt)
        assert(rF8 == gF8, s"FP8 MAX hw=$rF8 gold=$gF8")
        // --- FP8 ADD all-ones (0x3C = 1.0): Σ over 64 = 64.0 (exact, 0x54), back-to-back with the FP16 tasks ---
        val ones = Seq.fill(64)(0x3c)
        val oOne = runReduce(dut, 2, OP_ADD, Seq(packF8(ones)))
        val rOne = f8e5m2ToF32((oOne & 0xff).toInt)
        assert(rOne == 64.0f, s"FP8 ADD all-ones hw=$rOne gold=64.0")
        println(f"[StreamReduceRt] ONE netlist: FP16 ADD=$rAdd%.3f SUMSQ=$rSq%.3f MAX(neg)=$rNeg%.3f, FP8 MAX=$rF8%.3f ADD1s=$rOne%.1f")
      }
  }

  "StreamReduceRt_roofline" should "reach ~1 input-beat/cycle (512 b/cyc) at cl=64 with the parallel fold" in {
    test(new DataPathExtensionHarness(new HasStreamReduceRt(computeLanes = 64, op = Seq("FMA", "MAX"), foldParallel = 1)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng   = new Random(0x9e)
        val oc    = 4
        val nRows = 64
        val nIn   = oc * nRows
        val beats = Seq.fill(nIn)(packF8(Seq.fill(64)(normalF8(rng))))
        dut.io.csr_i(0).poke(csr0(oc, 2).U); dut.io.csr_i(1).poke(BigInt(OP_ADD).U) // FP8, ADD
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.data_o.ready.poke(true)
        dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(beats(0).U)
        var cyc = 0L; var fed = 0; var warmFeed = -1L; var lastFeed = -1L
        val maxCyc = nIn.toLong * 40 + 400
        while (fed < nIn && cyc < maxCyc) {
          val canFeed = dut.io.data_i.ready.peekBoolean()
          dut.clock.step(1); cyc += 1
          if (canFeed) { fed += 1; if (fed == 8) warmFeed = cyc
            if (fed < nIn) dut.io.data_i.bits.poke(beats(fed).U) else dut.io.data_i.valid.poke(false); lastFeed = cyc }
        }
        assert(fed == nIn, s"only fed $fed/$nIn after $cyc cyc")
        val beatSpan = (lastFeed - warmFeed).toDouble / (nIn - 8)
        val util = 1.0 / beatSpan
        println(f"[StreamReduceRt roofline cl=64] beatSpan=$beatSpan%.2f util=$util%.3f effBW=${util * 512}%.0f b/cyc")
        assert(util > 0.9, s"expected ~1.0 input util at cl=64, got $util")
      }
  }

  val OP_ARGMAX = 3
  "StreamReduceRt_argmax" should "reduce to the argmax lane index (MoE gating) at FP16+FP8 and hold the roofline" in {
    test(new DataPathExtensionHarness(
      new HasStreamReduceRt(computeLanes = 64, op = Seq("FMA", "MAX", "ARGMAX"), foldParallel = 1)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng   = new Random(0x33)
        val mask  = (BigInt(1) << 32) - 1
        // --- FP16: argmax over 32 expert logits (one beat) ---
        val v16   = Seq.fill(32)(f32ToF16bits(rng.between(-4f, 4f) + rng.nextInt(100) * 0.013f))
        val hw16  = (runReduce(dut, 0, OP_ARGMAX, Seq(packF16(v16))) & mask).toInt
        val gold16 = v16.map(f16bitsToF32).zipWithIndex.maxBy(_._1)._2
        assert(hw16 == gold16, s"FP16 argmax hw=$hw16 gold=$gold16")
        // --- FP8: argmax over 64 experts on the SAME netlist ---
        val v8    = Seq.fill(64)(normalF8(rng))
        val hw8   = (runReduce(dut, 2, OP_ARGMAX, Seq(packF8(v8))) & mask).toInt
        val gold8 = v8.map(f8e5m2ToF32).zipWithIndex.maxBy(_._1)._2
        assert(hw8 == gold8, s"FP8 argmax hw=$hw8 gold=$gold8")
        // --- regression: MAX value still correct through the wantMax path ---
        val rMax  = f16bitsToF32((runReduce(dut, 0, OP_MAX, Seq(packF16(v16))) & 0xffff).toInt)
        assert(rMax == v16.map(f16bitsToF32).max, s"MAX regressed: hw=$rMax")
        // --- roofline: argmax at cl=64, one-beat rows ---
        val nIn   = 128
        val beats = Seq.fill(nIn)(packF8(Seq.fill(64)(normalF8(rng))))
        dut.io.csr_i(0).poke(csr0(1, 2).U); dut.io.csr_i(1).poke(BigInt(OP_ARGMAX).U) // FP8, ARGMAX, oc=1
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.data_o.ready.poke(true)
        dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(beats(0).U)
        var cyc = 0L; var fed = 0; var warmFeed = -1L; var lastFeed = -1L
        val maxCyc = nIn.toLong * 40 + 400
        while (fed < nIn && cyc < maxCyc) {
          val canFeed = dut.io.data_i.ready.peekBoolean()
          dut.clock.step(1); cyc += 1
          if (canFeed) { fed += 1; if (fed == 8) warmFeed = cyc
            if (fed < nIn) dut.io.data_i.bits.poke(beats(fed).U) else dut.io.data_i.valid.poke(false); lastFeed = cyc }
        }
        assert(fed == nIn, s"only fed $fed/$nIn")
        val util = (nIn - 8).toDouble / (lastFeed - warmFeed)
        println(f"[StreamReduceRt ARGMAX] FP16 idx=$hw16 (gold $gold16), FP8 idx=$hw8 (gold $gold8); roofline util=$util%.3f")
        assert(util > 0.9, s"argmax roofline broke: util=$util")
      }
  }
}
