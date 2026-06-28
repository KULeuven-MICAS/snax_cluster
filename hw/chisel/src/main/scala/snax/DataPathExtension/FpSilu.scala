package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** FpSilu: per-lane silu(x) = x * sigmoid(x), FP32 in / FP32 out.
  *
  * sigmoid is read from an `N`-entry FP32 ROM using ODD SYMMETRY to halve the table: we tabulate only
  * g(m) = sigmoid(-m) for m = |x| in [0, 16] (step h = 16/N) — the small, fully-accurate side — and
  * recover sigmoid(x) = (x<=0) ? g(|x|) : 1 - g(|x|) (sigmoid(-m)=1-sigmoid(m)). A single central-
  * difference interpolation FMA refines g (sigmoid' is even, so one slope table serves both halves); the
  * result is then multiplied by the ORIGINAL signed x to form silu. Tabulating g(|x|) (not raw sigmoid
  * over [-16,16]) keeps the deep-negative tail — where silu ~ x*e^x lands small normals on a fine FP16
  * grid — at full FP32 precision: it is read DIRECTLY from the ROM, never via a cancelling subtraction.
  * The only subtraction (1 - g) is on the x>0 side where g is the tiny operand, so it is benign. The
  * x-multiply makes the tails exact (x>=16: sigmoid~1 => silu~x; x<=-16: sigmoid~0 => silu~0). The
  * float->index conversion uses the same 1.5*2^23 magic-number round as FpExp (no F2I unit).
  *
  * PIPELINING (timing): like FpExp, the ~7-deep FP32 chain is cut into `latency` (=4) register stages of
  * <=2 FP ops each when `pipelined` is set (the original x is carried down the pipe for the final multiply);
  * the host FSM accounts for the latency. `pipelined=false` keeps the original combinational behaviour.
  *
  * AREA: `N` (ROM depth, ×2 for base+slope) is the node count over [0,16]. Odd symmetry gives the same
  * node density as the old 512-over-[-16,16] table at N=256, halving both ROMs while staying within <=1
  * FP16 ULP (the prior uniform 256-node [-16,16] table measured 3 ULP at x~-10.8; symmetry fixes that).
  *
  * Same shape and role as FpExp (one ROM family + one interp FMA): it plugs into StreamMap as func=SILU
  * exactly where EXP routes through FpExp.
  */
class FpSilu(pipelined: Boolean = false, N: Int = 256) extends Module with RequireAsyncReset {
  require(isPow2(N), "FpSilu: N must be a power of two")
  val io = IO(new Bundle {
    val in  = Input(UInt(32.W))  // FP32
    val out = Output(UInt(32.W)) // FP32 = silu(in)
  })

  import FpHelpers._

  val latency: Int = if (pipelined) FpSilu.PipeLatency else 0
  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  // |x| folded onto [XLO, XHI] = [0, 16] at N uniform nodes (node i at i*H); idx = |x|/H = INVH*|x| + BIAS.
  // The ROM tabulates g(m) = sigmoid(-m) (the small, accurate side); sigmoid(x) = (x<=0) ? g(|x|) : 1 - g(|x|).
  val XLO  = 0.0
  val XHI  = 16.0
  val H    = (XHI - XLO) / N
  val INVH = 1.0 / H

  val SCALE  = f32lit(INVH.toFloat)          // 1/H
  val BIAS   = f32lit((-INVH * XLO).toFloat) // -INVH*XLO = 0  =>  idxReal = INVH*|x|
  val HI     = f32lit(XHI.toFloat)
  val ONE    = f32lit(1.0f)                  // for the 1 - g(|x|) reflection on x > 0
  val MAGIC  = f32lit(12582912.0f)           // 1.5 * 2^23 = 0x4B400000
  val NMAGIC = f32lit(-12582912.0f)
  def fneg(u: UInt): UInt = Cat(~u(31), u(30, 0)) // FP32 negate (sign flip)
  def fabs(u: UInt): UInt = Cat(0.U(1.W), u(30, 0)) // FP32 abs (sign clear)

  def sigmoid(xx: Double): Double = 1.0 / (1.0 + math.exp(-xx))
  def g(m: Double): Double = sigmoid(-m)     // tabulated side: small & fully accurate in FP32
  def nodeX(i: Int): Double = XLO + i * H
  // base(i) = g(node i); slope(i) = central d(g)/d(idx) ~= (g(i+1) - g(i-1)) / 2.
  val base  = VecInit((0 until N).map(i => f32lit(g(nodeX(i)).toFloat)))
  val slope = VecInit((0 until N).map(i =>
    f32lit(((g(nodeX(i + 1)) - g(nodeX(i - 1))) / 2.0).toFloat)))

  // ---- S0: fold to |x|, clamp to [0,16], then idxReal = |x|/H ----
  val m       = fp32min(fabs(io.in), HI)
  val idxReal = sr(ffma(m, SCALE, BIAS))     // reg0
  val in0     = sr(io.in)                      // reg0 : carry original (signed) x for the final multiply
  val sgn0    = sr(io.in(31))                  // reg0 : sign of x -> picks g (x<=0) vs 1-g (x>0)

  // ---- S1: round idxReal to the nearest node via the magic-number trick ----
  val rM       = sr(fadd(idxReal, MAGIC))     // reg1
  val idxReal1 = sr(idxReal)                   // reg1 (aligned with rM)
  val in1      = sr(in0)                        // reg1
  val sgn1     = sr(sgn0)                        // reg1

  // ---- S2: integer node index (iM) and fractional offset (frac in [-0.5,0.5]) ----
  val nodeF = fadd(rM, NMAGIC)                 // round(idxReal) as float (exact)
  val frac  = sr(fadd(idxReal1, fneg(nodeF))) // reg2 : idxReal - node
  val iM    = sr(rM.asSInt - 0x4b400000.S)    // reg2 : round(idxReal), in [0, N]
  val in2   = sr(in1)                           // reg2
  val sgn2  = sr(sgn1)                           // reg2

  // ---- S3: g(|x|) via one interpolation FMA, then reflect to sigmoid(x): x<=0 -> g, x>0 -> 1-g ----
  val idx  = Mux(iM > (N - 1).S, (N - 1).U, iM.asUInt(log2Ceil(N) - 1, 0))
  val gpos = ffma(frac, slope(idx), base(idx))         // g(|x|) = sigmoid(-|x|)
  val sig  = sr(Mux(sgn2, gpos, fadd(ONE, fneg(gpos)))) // reg3 : sigmoid(x)
  val in3  = sr(in2)                                    // reg3

  // ---- S4: silu = x * sigmoid(x) ----
  io.out := fmul(sig, in3)
}

object FpSilu {
  /** Register-stage latency of the pipelined datapath (kept in sync with the `sr()` cuts above and
    * with [[FpExp.PipeLatency]] so StreamMap's func-mux branches line up). */
  val PipeLatency: Int = 4
}
