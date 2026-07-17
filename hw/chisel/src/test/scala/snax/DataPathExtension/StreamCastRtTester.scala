package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** StreamCastRt: the in-transit QUANTIZER (the Cast primitive). One netlist casts a 16-bit row (FP16/BF16)
  * DOWN to FP8 or an MX element type, deriving the OCP E8M0 block scale per input beat (one beat = one 32-elem
  * MX block). Proves the block-scaled quantize is EXACT for values representable at the dst grid (widen +
  * per-beat MAXABS block scale + narrow + pack + trailing scale beat), across per-block scales.
  */
class StreamCastRtTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  def bf16bits(f: Float): Int = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xffff
  def packHalf(v: Seq[Int]): BigInt =
    v.zipWithIndex.foldLeft(BigInt(0)) { case (a, (h, i)) => a | (BigInt(h & 0xffff) << (16 * i)) }

  // FP16 <-> FP32 (for the FP16-source path); copied from the proven StreamMapRtTester helpers.
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

  /** Decode one MX element code (sign|exp|mant, finite) to its unscaled real value. */
  def decodeMX(bits: Int, expW: Int, sigW: Int, bias: Int): Double = {
    val s   = (bits >> (expW + sigW)) & 1
    val e   = (bits >> sigW) & ((1 << expW) - 1)
    val m   = bits & ((1 << sigW) - 1)
    val mag =
      if (e == 0) (m.toDouble / (1 << sigW)) * math.pow(2, 1 - bias)
      else (1.0 + m.toDouble / (1 << sigW)) * math.pow(2, e - bias)
    if (s == 1) -mag else mag
  }

  /** Feed `beats` (each one 32-elem MX block) and return (dataBeat, scaleBeat). */
  def runCast(dut: DataPathExtensionHarness, srcFmt: Int, dstFmt: Int, emitScl: Boolean, beats: Seq[BigInt])
    : (BigInt, BigInt) = {
    val csr = BigInt(srcFmt | (dstFmt << 2) | ((if (emitScl) 1 else 0) << 5))
    dut.io.csr_i(0).poke(csr.U)
    dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    dut.io.data_o.ready.poke(true)
    val want = if (emitScl) 2 else 1
    var fed = 0; var got = 0; var cyc = 0
    var dataB = BigInt(0); var sclB = BigInt(0)
    while (got < want && cyc < 1000) {
      val ready = dut.io.data_i.ready.peekBoolean()
      if (fed < beats.length && ready) { dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(beats(fed).U) }
      else dut.io.data_i.valid.poke(false)
      val ov = dut.io.data_o.valid.peekBoolean()
      val ob = if (ov) dut.io.data_o.bits.peekInt() else BigInt(0)
      dut.clock.step(1); cyc += 1
      if (fed < beats.length && ready) fed += 1
      if (ov) { if (got == 0) dataB = ob else sclB = ob; got += 1 }
    }
    assert(got == want, s"StreamCastRt: expected $want output beats, got $got")
    (dataB, sclB)
  }

  "StreamCastRt_mxfp8" should "quantize BF16 -> MXFP8-E5M2 with an in-transit block scale (exact for representable values)" in {
    test(new DataPathExtensionHarness(new HasStreamCastRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x0c)
        // exactly E5M2-representable magnitudes (2 mantissa bits): {1,1.25,1.5,1.75} x 2^k
        def repVal(): Float = (4 + rng.nextInt(4)).toFloat * (1 << rng.nextInt(2)) // {4,5,6,7,8,10,12,14}
        val blocks = Seq(Seq.fill(32)(repVal()), Seq.fill(32)(repVal()))           // R=2 for an 8-bit dst
        val beats  = blocks.map(b => packHalf(b.map(bf16bits)))
        val (data, scl) = runCast(dut, srcFmt = 1, dstFmt = 3, emitScl = true, beats)
        var maxErr = 0.0
        for (b <- 0 until 2) {
          val s = ((scl >> (8 * b)) & 0xff).toInt
          for (i <- 0 until 32) {
            val code = ((data >> (8 * (b * 32 + i))) & 0xff).toInt
            val deq  = decodeMX(code, 5, 2, 15) * math.pow(2, s - 127)
            val ref  = blocks(b)(i).toDouble
            maxErr = math.max(maxErr, math.abs(deq - ref) / math.max(math.abs(ref), 1e-9))
          }
        }
        println(f"[StreamCastRt MXFP8-E5M2] BF16->MXFP8 block-scaled, 2 blocks, max rel err = $maxErr%.2e")
        assert(maxErr < 1e-2, s"MXFP8 roundtrip rel err too high: $maxErr")
      }
  }

  "StreamCastRt_mxfp4" should "quantize BF16 -> MXFP4-E2M1 at 128 elems/beat with a PER-BLOCK scale (4:1 beats, exact for representable values)" in {
    test(new DataPathExtensionHarness(new HasStreamCastRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        // E2M1-representable base {2,3,4,6} = {1.0,1.5}x2^k; each block scaled by 2^b so the 4 blocks carry
        // DIFFERENT E8M0 scales -> exercises the per-beat block-scale derivation + emission.
        val base   = Seq(2f, 3f, 4f, 6f)
        val blocks = (0 until 4).map(b => Seq.tabulate(32)(i => base(i % 4) * (1 << b))) // R=4 for MXFP4
        val beats  = blocks.map(bk => packHalf(bk.map(bf16bits)))
        val (data, scl) = runCast(dut, srcFmt = 1, dstFmt = 7, emitScl = true, beats)
        var maxErr = 0.0
        val scales = (0 until 4).map(b => ((scl >> (8 * b)) & 0xff).toInt)
        for (b <- 0 until 4) {
          for (i <- 0 until 32) {
            val code = ((data >> (4 * (b * 32 + i))) & 0xf).toInt
            val deq  = decodeMX(code, 2, 1, 1) * math.pow(2, scales(b) - 127)
            val ref  = blocks(b)(i).toDouble
            maxErr = math.max(maxErr, math.abs(deq - ref) / math.max(math.abs(ref), 1e-9))
          }
        }
        println(f"[StreamCastRt MXFP4-E2M1] BF16->MXFP4 128 elems/beat, per-block scales=$scales, max rel err = $maxErr%.2e")
        assert(maxErr < 1e-2, s"MXFP4 roundtrip rel err too high: $maxErr")
      }
  }

  "StreamCastRt_signs" should "preserve sign through BF16 -> MXFP4 quantize (exact for representable magnitudes)" in {
    test(new DataPathExtensionHarness(new HasStreamCastRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng  = new Random(0x51)
        val mags = Seq(2f, 3f, 4f, 6f) // E2M1-representable
        val blocks = (0 until 4).map(_ => Seq.tabulate(32)(i => (if (rng.nextBoolean()) 1 else -1) * mags(i % 4)))
        val beats  = blocks.map(bk => packHalf(bk.map(bf16bits)))
        val (data, scl) = runCast(dut, srcFmt = 1, dstFmt = 7, emitScl = true, beats)
        val scales = (0 until 4).map(b => ((scl >> (8 * b)) & 0xff).toInt)
        var maxErr = 0.0
        for (b <- 0 until 4; i <- 0 until 32) {
          val code = ((data >> (4 * (b * 32 + i))) & 0xf).toInt
          val deq  = decodeMX(code, 2, 1, 1) * math.pow(2, scales(b) - 127)
          maxErr = math.max(maxErr, math.abs(deq - blocks(b)(i)) / math.max(math.abs(blocks(b)(i)), 1e-9))
        }
        println(f"[StreamCastRt signs] BF16->MXFP4 with +/- magnitudes, max rel err = $maxErr%.2e")
        assert(maxErr < 1e-2, s"sign roundtrip err $maxErr")
      }
  }

  "StreamCastRt_ftz_zero" should "flush a far-below-range element to zero and encode an all-zero block as zero (no garbage)" in {
    test(new DataPathExtensionHarness(new HasStreamCastRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        // block0: max 6 + one tiny (2^-5) far below the E2M1 in-block range -> must FTZ.
        // block1: all zero -> block scale clamps to 0 -> all codes must be 0 (this is the adversarial case).
        val b0 = Seq.tabulate(32)(i => if (i == 0) 0.03125f else Seq(2f, 3f, 4f, 6f)(i % 4))
        val b1 = Seq.fill(32)(0f)
        val b2 = Seq.tabulate(32)(i => Seq(2f, 3f, 4f, 6f)(i % 4))
        val blocks = Seq(b0, b1, b2, b2)
        val beats  = blocks.map(bk => packHalf(bk.map(bf16bits)))
        val (data, scl) = runCast(dut, srcFmt = 1, dstFmt = 7, emitScl = true, beats)
        val scales = (0 until 4).map(b => ((scl >> (8 * b)) & 0xff).toInt)
        val tinyDeq = decodeMX((data & 0xf).toInt, 2, 1, 1) * math.pow(2, scales(0) - 127)
        assert(math.abs(tinyDeq) < 0.5, s"tiny element should FTZ, got $tinyDeq")
        var zeroBad = 0
        for (i <- 0 until 32) if (((data >> (4 * (32 + i))) & 0xf).toInt != 0) zeroBad += 1
        assert(zeroBad == 0, s"all-zero block produced $zeroBad nonzero codes (scale=${scales(1)})")
        var maxErr = 0.0
        for (i <- 0 until 32) {
          val deq = decodeMX(((data >> (4 * (64 + i))) & 0xf).toInt, 2, 1, 1) * math.pow(2, scales(2) - 127)
          maxErr = math.max(maxErr, math.abs(deq - b2(i)) / math.max(b2(i), 1e-9))
        }
        println(f"[StreamCastRt ftz/zero] tiny->$tinyDeq%.3f (FTZ), all-zero block clean, block2 rel err=$maxErr%.2e")
        assert(maxErr < 1e-2)
      }
  }

  "StreamCastRt_fp8_noscale" should "quantize BF16 -> plain FP8 (no MX scale), R=2, exactly one output beat" in {
    test(new DataPathExtensionHarness(new HasStreamCastRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x07)
        def rep(): Float = (4 + rng.nextInt(4)).toFloat * (1 << rng.nextInt(2)) // E5M2-representable
        val blocks = Seq(Seq.fill(32)(rep()), Seq.fill(32)(rep()))
        val beats  = blocks.map(b => packHalf(b.map(bf16bits)))
        val (data, _) = runCast(dut, srcFmt = 1, dstFmt = 2, emitScl = false, beats) // dstFmt=2=FP8, no scale beat
        var maxErr = 0.0
        for (b <- 0 until 2; i <- 0 until 32) {
          val deq = decodeMX(((data >> (8 * (b * 32 + i))) & 0xff).toInt, 5, 2, 15) // plain FP8=E5M2, scale 2^0
          maxErr = math.max(maxErr, math.abs(deq - blocks(b)(i)) / math.max(blocks(b)(i), 1e-9))
        }
        println(f"[StreamCastRt FP8 no-scale] BF16->FP8, 1 output beat (no scale), rel err=$maxErr%.2e")
        assert(maxErr < 1e-2)
      }
  }

  "StreamCastRt_rounding_fp16src" should "bound the quantization error for non-representable values from an FP16 source" in {
    test(new DataPathExtensionHarness(new HasStreamCastRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x3d)
        def v(): Float = 8f + rng.nextFloat() * 7.9f // non-representable, one binade [8,16) -> E5M2 <=1/8 rel step
        val blocks = Seq(Seq.fill(32)(v()), Seq.fill(32)(v()))
        val beats  = blocks.map(b => packHalf(b.map(f32ToF16bits))) // FP16 source (srcFmt=0)
        val (data, scl) = runCast(dut, srcFmt = 0, dstFmt = 3, emitScl = true, beats) // MXFP8-E5M2
        val scales = (0 until 2).map(b => ((scl >> (8 * b)) & 0xff).toInt)
        var maxErr = 0.0
        for (b <- 0 until 2; i <- 0 until 32) {
          val deq = decodeMX(((data >> (8 * (b * 32 + i))) & 0xff).toInt, 5, 2, 15) * math.pow(2, scales(b) - 127)
          val ref = f16bitsToF32(f32ToF16bits(blocks(b)(i))).toDouble // the FP16-rounded value the DUT sees
          maxErr = math.max(maxErr, math.abs(deq - ref) / math.max(math.abs(ref), 1e-9))
        }
        println(f"[StreamCastRt rounding/FP16-src] non-representable, max rel err=$maxErr%.3f (<= ~1/8 E5M2 step)")
        assert(maxErr < 0.15, s"rel err $maxErr exceeds the E5M2 half-step bound")
      }
  }

  "StreamCastRt_saturate" should "SATURATE the block-max element (round-up past the format max) instead of overflowing to inf/NaN" in {
    test(new DataPathExtensionHarness(new HasStreamCastRt()))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        // block max = 1.96875*2^3 = 15.75; its mantissa (1.96875) rounds up past E5M2's max normal (1.75)
        // -> would become 2.0*2^15 = overflow. The OCP scale aligns it to the top normal binade, so it must
        // SATURATE to max-normal, not produce inf. Rest of the block sits in the same binade [8,16).
        val blk = Seq.tabulate(32)(i => if (i == 0) 15.75f else 8f + (i % 4))
        // --- E5M2 (fpnew narrow): assert NO inf/NaN code (exp field != all-ones) ---
        val (dE5, sE5) = runCast(dut, srcFmt = 1, dstFmt = 3, emitScl = true, Seq(blk, blk).map(b => packHalf(b.map(bf16bits))))
        var infE5 = 0; var errE5 = 0.0
        for (b <- 0 until 2; i <- 0 until 32) {
          val code = ((dE5 >> (8 * (b * 32 + i))) & 0xff).toInt
          if ((code & 0x7c) == 0x7c) infE5 += 1 // E5M2 exp field (bits 6:2) all-ones = inf/NaN
          val deq = decodeMX(code, 5, 2, 15) * math.pow(2, ((sE5 >> (8 * b)) & 0xff).toInt - 127)
          errE5 = math.max(errE5, math.abs(deq - blk(i)) / blk(i))
        }
        assert(infE5 == 0, s"E5M2 block-max round-up produced $infE5 inf/NaN codes -- must saturate")
        // --- MXFP4 (narrowFin saturates by construction): same block, assert finite + bounded ---
        val (dF4, sF4) = runCast(dut, srcFmt = 1, dstFmt = 7, emitScl = true, Seq.fill(4)(blk).map(b => packHalf(b.map(bf16bits))))
        var errF4 = 0.0
        for (b <- 0 until 4; i <- 0 until 32) {
          val code = ((dF4 >> (4 * (b * 32 + i))) & 0xf).toInt
          val deq  = decodeMX(code, 2, 1, 1) * math.pow(2, ((sF4 >> (8 * b)) & 0xff).toInt - 127)
          errF4 = math.max(errF4, math.abs(deq - blk(i)) / blk(i))
        }
        println(f"[StreamCastRt saturate] E5M2: 0 inf codes, rel err=$errE5%.3f; MXFP4 rel err=$errF4%.3f (both saturate the top element)")
        assert(errE5 < 0.2 && errF4 < 0.4, s"saturation error too high: E5M2=$errE5 MXFP4=$errF4")
      }
  }
}
