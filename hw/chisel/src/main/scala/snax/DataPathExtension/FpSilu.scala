package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** FpSilu: per-lane silu(x) = x * sigmoid(x), FP32 in / FP32 out, combinational.
  *
  * sigmoid(x) = 1/(1+e^-x) is read from a 512-entry FP32 ROM sampled uniformly over x in [-16, 16]
  * (step h = 1/16) and refined with a single central-difference interpolation FMA; the result is then
  * multiplied by the ORIGINAL x to form silu. 512 nodes (vs 256 for FpExp) are needed because silu's
  * negative tail, silu ~ x*e^x, lands small normal values on a very fine FP16 grid where linear
  * interpolation must be denser to stay within 1 ULP. The LUT tabulates sigmoid (not silu directly) because
  * sigmoid'' vanishes at the origin, exactly where silu is curviest, so a uniform table is far more
  * accurate there; the x-multiply also makes the tails exact (x>=16: sigmoid~1 => silu~x; x<=-16:
  * sigmoid~0 => silu~0) without tabulating the unbounded positive ramp. The float->index conversion uses
  * the same 1.5*2^23 magic-number round as FpExp (no F2I unit).
  *
  * Same shape and role as FpExp (one ROM family + one interp FMA): it plugs into StreamMap as func=SILU
  * exactly where EXP routes through FpExp. Matches host silu to <= 1 FP16 ULP after narrowing (FpSiluTester).
  */
class FpSilu extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in  = Input(UInt(32.W))  // FP32
    val out = Output(UInt(32.W)) // FP32 = silu(in)
  })

  import FpHelpers._

  // sigmoid sampled on [XLO, XHI] at N uniform nodes (node i at XLO + i*H); idx = (x - XLO)/H = INVH*x + BIAS.
  val N    = 512
  val XLO  = -16.0
  val XHI  = 16.0
  val H    = (XHI - XLO) / N // 0.125
  val INVH = 1.0 / H         // 8.0

  val SCALE  = f32lit(INVH.toFloat)          // 1/H
  val BIAS   = f32lit((-INVH * XLO).toFloat) // -INVH*XLO = 128.0  =>  idxReal = INVH*x + BIAS
  val HI     = f32lit(XHI.toFloat)
  val LO     = f32lit(XLO.toFloat)
  val MAGIC  = f32lit(12582912.0f)           // 1.5 * 2^23 = 0x4B400000
  val NMAGIC = f32lit(-12582912.0f)
  def fneg(u: UInt): UInt = Cat(~u(31), u(30, 0)) // FP32 negate (sign flip)

  def sigmoid(xx: Double): Double = 1.0 / (1.0 + math.exp(-xx))
  def nodeX(i: Int): Double = XLO + i * H
  // base(i) = sigmoid(node i); slope(i) = central d(sigmoid)/d(idx) ~= (sig(i+1) - sig(i-1)) / 2.
  val base  = VecInit((0 until N).map(i => f32lit(sigmoid(nodeX(i)).toFloat)))
  val slope = VecInit((0 until N).map(i =>
    f32lit(((sigmoid(nodeX(i + 1)) - sigmoid(nodeX(i - 1))) / 2.0).toFloat)))

  // clamp x into the table domain for indexing; the final multiply uses the true io.in (exact linear tail)
  val x = fp32max(fp32min(io.in, HI), LO)

  // idxReal = x/H + 128, rounded to the nearest node via the magic-number trick
  val idxReal = ffma(x, SCALE, BIAS)
  val rM      = fadd(idxReal, MAGIC)
  val iM      = rM.asSInt - 0x4b400000.S            // round(idxReal), in [0, N]
  val nodeF   = fadd(rM, NMAGIC)                     // round(idxReal) as float (exact)
  val frac    = fadd(idxReal, fneg(nodeF))           // idxReal - node, in [-0.5, 0.5]
  val idx     = Mux(iM > (N - 1).S, (N - 1).U, iM.asUInt(log2Ceil(N) - 1, 0))

  val sig = ffma(frac, slope(idx), base(idx))        // sigmoid(x) with one interp FMA
  io.out := fmul(sig, io.in)                          // silu = x * sigmoid(x)
}
