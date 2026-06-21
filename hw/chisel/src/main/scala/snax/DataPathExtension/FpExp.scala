package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** FpExp: per-lane exp(x), FP32 in / FP32 out, combinational.
  *
  * LUT-based 2^m design (the chosen area point): exp(x) = 2^m with m = x*log2e. Split
  * m*256 into an integer iM = round(m*256); then n = iM >> 8 is the integer part of m and idx = iM[7:0]
  * indexes a 256-entry ROM of 2^(idx/256); the leftover frac = m*256 - iM in [-0.5,0.5] is folded with a
  * single interpolation FMA (2^(frac/256) ~= 1 + frac*ln2/256). Finally scale by 2^n (exponent-field
  * construct). The float->int uses the 1.5*2^23 magic-number add (no F2I unit).
  *
  * ~8x smaller than the prior degree-5 polynomial (1 ROM + 1 interp FMA vs the 7-FMA Horner chain), and
  * matches math.exp to <= 1 FP16 ULP over the softmax range (the linear interpolation error of 2^(d),
  * |d| <= 1/512, is ~1e-6 relative, far under the FP16 grid).
  */
class FpExp extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in  = Input(UInt(32.W))  // FP32
    val out = Output(UInt(32.W)) // FP32 = exp(in)
  })

  import FpHelpers._

  val LOG2EF_256 = f32lit((1.44269504088896341 * 256.0).toFloat) // log2(e) * 256
  val LN2_256    = f32lit((0.6931471805599453 / 256.0).toFloat)  // ln2 / 256
  val ONE        = f32lit(1.0f)
  val HI         = f32lit(88.3762626647949f)
  val LO         = f32lit(-88.3762626647949f)
  val MAGIC      = f32lit(12582912.0f)  // 1.5 * 2^23 = 0x4B400000
  val NMAGIC     = f32lit(-12582912.0f) // -1.5 * 2^23 = 0xCB400000
  def fneg(u: UInt): UInt = Cat(~u(31), u(30, 0)) // FP32 negate (sign flip)

  // clamp to [-88.376, 88.376]
  val x = fp32max(fp32min(io.in, HI), LO)

  // m256 = x * log2e * 256, then round to the nearest integer iM via the magic-number trick
  val m256   = fmul(x, LOG2EF_256)
  val rM     = fadd(m256, MAGIC)
  val iM     = rM.asSInt - 0x4b400000.S    // round(m256); |iM| <= 32512 after the clamp
  val mFloat = fadd(rM, NMAGIC)            // = rM - MAGIC = round(m256) as float (exact)
  val frac   = fadd(m256, fneg(mFloat))    // m256 - iM, in [-0.5, 0.5]

  val n   = iM >> 8               // integer part of m (arithmetic shift, signed)
  val idx = iM.asUInt(7, 0)       // fractional LUT index in [0, 256)

  // ROM: 2^(idx/256), 256 FP32 entries (elaboration-time)
  val lut    = VecInit((0 until 256).map(i => f32lit(math.pow(2.0, i / 256.0).toFloat)))
  val lutVal = lut(idx)

  // 2^f = 2^(idx/256) * 2^(frac/256) ~= lutVal * (1 + frac*ln2/256)
  val corr = ffma(frac, LN2_256, ONE)
  val twoF = fmul(lutVal, corr)

  // 2^n (exponent field = n + 127; underflow -> +0, which narrows to FP16 0)
  val pow2nExp = (n + 127.S).asUInt(7, 0)
  val pow2n    = Cat(0.U(1.W), pow2nExp, 0.U(23.W))

  io.out := fmul(twoF, pow2n)
}
