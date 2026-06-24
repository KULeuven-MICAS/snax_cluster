package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** FpSilu: per-lane silu(x) = x * sigmoid(x), FP32 in / FP32 out.
  *
  * sigmoid(x) = 1/(1+e^-x) is read from an `N`-entry FP32 ROM sampled uniformly over x in [-16, 16]
  * (step h = 32/N) and refined with a single central-difference interpolation FMA; the result is then
  * multiplied by the ORIGINAL x to form silu. The LUT tabulates sigmoid (not silu directly) because
  * sigmoid'' vanishes at the origin, exactly where silu is curviest, so a uniform table is far more
  * accurate there; the x-multiply also makes the tails exact (x>=16: sigmoid~1 => silu~x; x<=-16:
  * sigmoid~0 => silu~0) without tabulating the unbounded positive ramp. The float->index conversion uses
  * the same 1.5*2^23 magic-number round as FpExp (no F2I unit).
  *
  * PIPELINING (timing): like FpExp, the ~7-deep FP32 chain is cut into `latency` (=4) register stages of
  * <=2 FP ops each when `pipelined` is set (the original x is carried down the pipe for the final multiply);
  * the host FSM accounts for the latency. `pipelined=false` keeps the original combinational behaviour.
  *
  * AREA: `N` (ROM depth, ×2 for base+slope) trades table size for accuracy. 256 nodes stay within the
  * relaxed <=2 FP16 ULP budget while halving the two ROMs vs the original 512.
  *
  * Same shape and role as FpExp (one ROM family + one interp FMA): it plugs into StreamMap as func=SILU
  * exactly where EXP routes through FpExp.
  */
class FpSilu(pipelined: Boolean = false, N: Int = 512) extends Module with RequireAsyncReset {
  require(isPow2(N), "FpSilu: N must be a power of two")
  val io = IO(new Bundle {
    val in  = Input(UInt(32.W))  // FP32
    val out = Output(UInt(32.W)) // FP32 = silu(in)
  })

  import FpHelpers._

  val latency: Int = if (pipelined) FpSilu.PipeLatency else 0
  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  // sigmoid sampled on [XLO, XHI] at N uniform nodes (node i at XLO + i*H); idx = (x - XLO)/H = INVH*x + BIAS.
  val XLO  = -16.0
  val XHI  = 16.0
  val H    = (XHI - XLO) / N
  val INVH = 1.0 / H

  val SCALE  = f32lit(INVH.toFloat)          // 1/H
  val BIAS   = f32lit((-INVH * XLO).toFloat) // -INVH*XLO  =>  idxReal = INVH*x + BIAS
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

  // ---- S0: clamp x into the table domain, then idxReal = x/H + (-INVH*XLO) ----
  val x       = fp32max(fp32min(io.in, HI), LO)
  val idxReal = sr(ffma(x, SCALE, BIAS))     // reg0
  val in0     = sr(io.in)                      // reg0 : carry original x for the final multiply

  // ---- S1: round idxReal to the nearest node via the magic-number trick ----
  val rM       = sr(fadd(idxReal, MAGIC))     // reg1
  val idxReal1 = sr(idxReal)                   // reg1 (aligned with rM)
  val in1      = sr(in0)                        // reg1

  // ---- S2: integer node index (iM) and fractional offset (frac in [-0.5,0.5]) ----
  val nodeF = fadd(rM, NMAGIC)                 // round(idxReal) as float (exact)
  val frac  = sr(fadd(idxReal1, fneg(nodeF))) // reg2 : idxReal - node
  val iM    = sr(rM.asSInt - 0x4b400000.S)    // reg2 : round(idxReal), in [0, N]
  val in2   = sr(in1)                           // reg2

  // ---- S3: sigmoid(x) via one interpolation FMA ----
  val idx = Mux(iM > (N - 1).S, (N - 1).U, iM.asUInt(log2Ceil(N) - 1, 0))
  val sig = sr(ffma(frac, slope(idx), base(idx))) // reg3
  val in3 = sr(in2)                                // reg3

  // ---- S4: silu = x * sigmoid(x) ----
  io.out := fmul(sig, in3)
}

object FpSilu {
  /** Register-stage latency of the pipelined datapath (kept in sync with the `sr()` cuts above and
    * with [[FpExp.PipeLatency]] so StreamMap's func-mux branches line up). */
  val PipeLatency: Int = 4
}
