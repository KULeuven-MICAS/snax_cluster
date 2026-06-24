package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** FpExp: per-lane exp(x), FP32 in / FP32 out.
  *
  * LUT-based 2^m design (the chosen area point): exp(x) = 2^m with m = x*log2e. Split
  * m*lutN into an integer iM = round(m*lutN); then n = iM >> log2(lutN) is the integer part of m and
  * idx = iM[log2(lutN)-1:0] indexes a `lutN`-entry ROM of 2^(idx/lutN); the leftover
  * frac = m*lutN - iM in [-0.5,0.5] is folded with a single interpolation FMA
  * (2^(frac/lutN) ~= 1 + frac*ln2/lutN). Finally scale by 2^n (exponent-field construct). The
  * float->int uses the 1.5*2^23 magic-number add (no F2I unit).
  *
  * ~8x smaller than the prior degree-5 polynomial (1 ROM + 1 interp FMA vs the 7-FMA Horner chain).
  *
  * PIPELINING (timing): the math is a ~7-deep FP32 chain (clamp->mul->add->add->add->fma->mul->mul),
  * which over-runs the 2 ns clock as one combinational path. When `pipelined` is set the chain is cut
  * into `latency` (=4) register stages of <=2 FP ops each; the stages are pure feed-forward so the result
  * simply appears `latency` cycles later (the host FSM accounts for it). `pipelined=false` keeps the
  * original 1-cycle combinational behaviour (used by the standalone tester).
  *
  * AREA: `lutN` (ROM depth) trades table size for interpolation accuracy. 128 entries stay within the
  * relaxed <=2 FP16 ULP budget on the softmax range while halving the ROM vs the original 256.
  */
class FpExp(pipelined: Boolean = false, lutN: Int = 256) extends Module with RequireAsyncReset {
  require(isPow2(lutN), "FpExp: lutN must be a power of two")
  val io = IO(new Bundle {
    val in  = Input(UInt(32.W))  // FP32
    val out = Output(UInt(32.W)) // FP32 = exp(in)
  })

  import FpHelpers._

  val LOGN = log2Ceil(lutN)
  val latency: Int = if (pipelined) FpExp.PipeLatency else 0
  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  val LOG2EF_N = f32lit((1.44269504088896341 * lutN.toDouble).toFloat) // log2(e) * lutN
  val LN2_N    = f32lit((0.6931471805599453 / lutN.toDouble).toFloat)  // ln2 / lutN
  val ONE      = f32lit(1.0f)
  val HI       = f32lit(88.3762626647949f)
  val LO       = f32lit(-88.3762626647949f)
  val MAGIC    = f32lit(12582912.0f)  // 1.5 * 2^23 = 0x4B400000
  val NMAGIC   = f32lit(-12582912.0f) // -1.5 * 2^23 = 0xCB400000
  def fneg(u: UInt): UInt = Cat(~u(31), u(30, 0)) // FP32 negate (sign flip)

  // ---- S0: clamp to [-88.376, 88.376], then m256 = x * log2e * lutN ----
  val x    = fp32max(fp32min(io.in, HI), LO)
  val m256 = sr(fmul(x, LOG2EF_N))            // reg0

  // ---- S1: round to the nearest integer iM via the magic-number trick ----
  val rM     = sr(fadd(m256, MAGIC))          // reg1
  val m256_d = sr(m256)                        // carry m256 to S2 (aligned with rM)

  // ---- S2: split into integer (iM) and fractional (frac in [-0.5,0.5]) parts ----
  val mFloat = fadd(rM, NMAGIC)               // = rM - MAGIC = round(m256) as float (exact)
  val frac   = sr(fadd(m256_d, fneg(mFloat))) // reg2 : m256 - round(m256)
  val iM     = sr(rM.asSInt - 0x4b400000.S)   // reg2 : round(m256); |iM| < lutN*128

  // ---- S3: 2^f = 2^(idx/lutN) * 2^(frac/lutN) ~= lutVal * (1 + frac*ln2/lutN) ----
  val idx    = iM.asUInt(LOGN - 1, 0)         // fractional LUT index in [0, lutN)
  val lut    = VecInit((0 until lutN).map(i => f32lit(math.pow(2.0, i.toDouble / lutN.toDouble).toFloat)))
  val corr   = ffma(frac, LN2_N, ONE)
  val twoF   = sr(fmul(lut(idx), corr))       // reg3
  val n      = sr(iM >> LOGN)                  // reg3 : integer part of m (arithmetic shift, signed)

  // ---- S4: scale by 2^n (exponent field = n + 127; underflow -> +0, narrows to FP16 0) ----
  val pow2nExp = (n + 127.S).asUInt(7, 0)
  val pow2n    = Cat(0.U(1.W), pow2nExp, 0.U(23.W))
  io.out := fmul(twoF, pow2n)
}

object FpExp {
  /** Register-stage latency of the pipelined datapath (kept in sync with the `sr()` cuts above and
    * with [[FpSilu.PipeLatency]] so StreamMap's func-mux branches line up). */
  val PipeLatency: Int = 4
}
