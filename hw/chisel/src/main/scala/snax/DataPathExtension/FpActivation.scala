package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** FpActivation: one per-lane activation core that computes exp(x) OR silu(x) (FP32 in/out), selected at runtime by
  * `io.func` (false = exp, true = silu). It MERGES the former standalone FpExp and FpSilu, which were both instantiated
  * per StreamMap lane yet are mutually exclusive at runtime. The two share the exact same 4-stage skeleton —
  * input-affine -> magic-round -> fractional split -> LUT + interpolation FMA -> post-transform — so all seven FP units
  * are shared (operands muxed by `func`); only the ROMs stay per-function (exp: one 2^(i/lutN) table; silu: base+slope
  * sigmoid tables). This halves the activation FP area of StreamMap versus the two separate modules, bit-exactly (see
  * the fmul==ffma(_,_,+0) and 1-g==ffma(g,-1,+1) identities below), while keeping the same PipeLatency=4 the func-mux
  * relies on.
  *
  * When only one function is built the mux conditions constant-fold, degenerating to the single pipeline.
  */
class FpActivation(
  pipelined: Boolean = false,
  hasExp:    Boolean = true,
  hasSilu:   Boolean = true,
  hasGelu:   Boolean = false,
  expLutN:   Int     = 128,
  siluN:     Int     = 256
) extends Module
    with RequireAsyncReset {
  require(hasExp || hasSilu || hasGelu, "FpActivation: at least one of exp/silu/gelu must be built")
  require(!hasExp || isPow2(expLutN), "FpActivation: expLutN must be a power of two")
  require(!(hasSilu || hasGelu) || isPow2(siluN), "FpActivation: siluN must be a power of two")
  val io = IO(new Bundle {
    val in   = Input(UInt(32.W))  // FP32
    val func = Input(Bool())      // false = exp, true = the g-family (silu/gelu); ignored when only one built
    val gelu = Input(Bool())      // within the g-family: true = gelu, false = silu; ignored unless both built
    val out  = Output(UInt(32.W)) // FP32
  })

  import FpHelpers._

  // SiLU and GELU are the SAME datapath: x*g(x) with an S-curve g that obeys g(x) = 1 - g(-x) (sigmoid for
  // SiLU, the Gaussian CDF Phi for GELU). They share every stage (input-affine -> index -> LUT interp ->
  // 1-g reflect -> multiply by x); only the base/slope ROM differs, muxed by `gelu`. So GELU is a pure ROM
  // add on the SiLU pipeline — no new FP units, same PipeLatency=4, roofline-neutral.
  val hasG   = hasSilu || hasGelu
  val both   = hasExp && hasG
  val isExp  = if (both) !io.func else hasExp.B          // compile-time constant when the g-family is absent
  val isGelu = if (hasSilu && hasGelu) io.gelu else hasGelu.B // within the g-family (constant if one built)
  val latency: Int = if (pipelined) FpActivation.PipeLatency else 0
  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  // ---- shared + per-function constants (mirrors FpExp / FpSilu exactly) ----
  val ONE    = f32lit(1.0f)
  val ZERO   = FP32_ZERO
  val MAGIC  = f32lit(12582912.0f) // 1.5 * 2^23 = 0x4B400000 (float->int round)
  val NMAGIC = f32lit(-12582912.0f)
  def fneg(u: UInt): UInt = Cat(~u(31), u(30, 0))
  def fabs(u: UInt): UInt = Cat(0.U(1.W), u(30, 0))

  // exp: m = x*log2e ; scale = log2e*lutN so the fractional index lands in the ROM
  val LOGN_E   = if (hasExp) log2Ceil(expLutN) else 0
  val LOG2EF_N = f32lit((1.44269504088896341 * expLutN.toDouble).toFloat)
  val LN2_N    = f32lit((0.6931471805599453 / expLutN.toDouble).toFloat)
  val HI_E     = f32lit(88.3762626647949f)
  val LO_E     = f32lit(-88.3762626647949f)
  val lut      =
    if (hasExp) VecInit((0 until expLutN).map(i => f32lit(math.pow(2.0, i.toDouble / expLutN.toDouble).toFloat)))
    else VecInit(Seq(ZERO))

  // g-family: idx = |x|/H over [0,16]; g(m) tabulated (base) with central-difference slope. SiLU uses
  // g=sigmoid(-m); GELU uses g=Phi(-m), the Gaussian CDF (Phi(t)=0.5(1+erf(t/sqrt2))). Both saturate well
  // before |x|=16 and both reflect via 1-g for x>0, so the whole S0..S4 chain is shared; only the ROM changes.
  val LOGN_S = if (hasG) log2Ceil(siluN) else 0
  // SiLU tabulates |x| over [0,16] (sigmoid tail decays ~exp(-m)); GELU over [0,5] — Phi's Gaussian tail
  // decays ~exp(-m^2/2), FAR faster, so a 16-wide grid over-coarsens the tail (18 FP16 ULP at x~-4 where
  // gelu is still a normal ~1.3e-4). A 5-wide grid at the SAME node count (siluN) restores 1 ULP; the clip
  // at 5 only drops x<-5.7 where gelu < FP16 min-subnormal (FTZ anyway). Node count is shared so the index
  // path (idxS mask, LOGN_S, siluN-1 clamp) is common; only the affine scale + clamp differ, muxed by isGelu.
  val XHI    = 16.0
  val H      = XHI / siluN
  val SCALE  = f32lit((1.0 / H).toFloat) // SiLU affine scale 1/H
  val HI_S   = f32lit(XHI.toFloat)
  val XHI_G  = 5.0
  val H_G    = XHI_G / siluN
  val SCALE_G = f32lit((1.0 / H_G).toFloat) // GELU affine scale 1/H_G
  val HI_G   = f32lit(XHI_G.toFloat)
  val BIAS   = ZERO // -INVH*XLO with XLO=0
  // S0 affine picks the g-func's own scale/clamp (constant-folds when only one g-func is built)
  val gScale = if (hasSilu && hasGelu) Mux(isGelu, SCALE_G, SCALE) else if (hasGelu) SCALE_G else SCALE
  val gClamp = if (hasSilu && hasGelu) Mux(isGelu, HI_G, HI_S) else if (hasGelu) HI_G else HI_S
  def sigmoid(xx: Double): Double = 1.0 / (1.0 + math.exp(-xx))
  // erf via Abramowitz-Stegun 7.1.26 (max abs err 1.5e-7, far below FP16 ULP -> LUT-node-accurate)
  def erf(xx: Double): Double = {
    val t = 1.0 / (1.0 + 0.3275911 * math.abs(xx))
    val y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t * math.exp(-xx * xx)
    if (xx >= 0) y else -y
  }
  def phi(t:      Double): Double = 0.5 * (1.0 + erf(t / math.sqrt(2.0)))
  def gnodeS(i:   Int):    Double = sigmoid(-(i * H))   // SiLU g-node over [0,16]
  def gnodeG(i:   Int):    Double = phi(-(i * H_G))     // GELU g-node over [0,5] (own, tighter grid)
  val base   = if (hasSilu) VecInit((0 until siluN).map(i => f32lit(gnodeS(i).toFloat))) else VecInit(Seq(ZERO))
  val slope  =
    if (hasSilu) VecInit((0 until siluN).map(i => f32lit(((gnodeS(i + 1) - gnodeS(i - 1)) / 2.0).toFloat)))
    else VecInit(Seq(ZERO))
  val gbase  = if (hasGelu) VecInit((0 until siluN).map(i => f32lit(gnodeG(i).toFloat))) else VecInit(Seq(ZERO))
  val gslope =
    if (hasGelu) VecInit((0 until siluN).map(i => f32lit(((gnodeG(i + 1) - gnodeG(i - 1)) / 2.0).toFloat)))
    else VecInit(Seq(ZERO))

  // ---- S0: input-affine. exp: fmul(clamp(x),LOG2EF_N) == ffma(_, _, +0); silu: ffma(|x|clamped, SCALE, 0) ----
  val preExp = if (hasExp) fp32max(fp32min(io.in, HI_E), LO_E) else io.in
  val preSil = if (hasG) fp32min(fabs(io.in), gClamp) else io.in
  val s0     = sr(ffma(Mux(isExp, preExp, preSil), Mux(isExp, LOG2EF_N, gScale), Mux(isExp, ZERO, BIAS))) // reg0
  val xin0 = sr(io.in)     // reg0 : original signed x (silu final multiply)
  val sgn0 = sr(io.in(31)) // reg0 : sign of x (silu g/1-g reflection)

  // ---- S1: round to nearest integer/node via the magic add ----
  val rM   = sr(fadd(s0, MAGIC)) // reg1
  val s0d  = sr(s0)              // reg1 (carry the scaled value for frac)
  val xin1 = sr(xin0)
  val sgn1 = sr(sgn0)

  // ---- S2: integer index iM and fractional part frac in [-0.5,0.5] ----
  val roundF = fadd(rM, NMAGIC)             // round(s0) as float (exact)
  val frac   = sr(fadd(s0d, fneg(roundF)))  // reg2 : s0 - round(s0)
  val iM     = sr(rM.asSInt - 0x4b400000.S) // reg2 : round(s0) as int
  val xin2   = sr(xin1)
  val sgn2   = sr(sgn1)

  // ---- S3: LUT lookup + interpolation FMA, then post: exp -> lut*corr ; silu -> reflect (1-g on x>0) ----
  val idxE      = if (hasExp) iM.asUInt(LOGN_E - 1, 0) else 0.U
  val idxS      = if (hasG) Mux(iM > (siluN - 1).S, (siluN - 1).U, iM.asUInt(LOGN_S - 1, 0)) else 0.U
  val lutV      = lut(idxE)
  // g-family ROM read, muxed SiLU vs GELU by `isGelu` (a constant when only one g-func is built)
  val baseV     =
    if (hasSilu && hasGelu) Mux(isGelu, gbase(idxS), base(idxS))
    else if (hasGelu) gbase(idxS)
    else if (hasSilu) base(idxS)
    else ZERO
  val slopeV    =
    if (hasSilu && hasGelu) Mux(isGelu, gslope(idxS), slope(idxS))
    else if (hasGelu) gslope(idxS)
    else if (hasSilu) slope(idxS)
    else ZERO
  // interp: exp corr = frac*LN2_N + 1 ; silu gpos = frac*slope + base  (one shared ffma; same as the old
  // FpExp/FpSilu first S3 op)
  val interp    = ffma(frac, Mux(isExp, LN2_N, slopeV), Mux(isExp, ONE, baseV))
  // post — KEEP the original per-func op (fmul for exp, fadd for silu), NOT a shared ffma. The S3 critical
  // path stays ffma->{fmul|fadd} (2 FP ops/stage) exactly like the old FpExp/FpSilu; folding both into one
  // ffma would make S3 ffma->ffma (a deeper combinational path) and risk timing. Bit-identical either way
  // (fmul(a,b)==ffma(a,b,+0); fadd(1,-g)==ffma(g,-1,+1)).
  val expTwoF   = if (hasExp) fmul(lutV, interp) else ZERO              // exp: lut * corr
  val siluOneMg = if (hasG) fadd(ONE, fneg(interp)) else ZERO          // g-family: 1 - g (reflect for x>0)
  // exp keeps twoF; silu picks g (x<=0) vs 1-g (x>0)
  val r3        = sr(Mux(isExp, expTwoF, Mux(sgn2, interp, siluOneMg))) // reg3
  val nE        = if (hasExp) sr(iM >> LOGN_E) else sr(0.S)             // reg3 : exp integer part of m
  val xin3      = sr(xin2)

  // ---- S4: exp scales by 2^n (exp-field construct); silu multiplies by the original x ----
  val pow2n = if (hasExp) Cat(0.U(1.W), (nE + 127.S).asUInt(7, 0), 0.U(23.W)) else ZERO
  io.out := fmul(r3, Mux(isExp, pow2n, xin3))
}

object FpActivation {

  /** Register-stage latency of the pipelined datapath (matches the former FpExp/FpSilu PipeLatency so StreamMap's
    * LINEAR/EXP/SILU func-mux branches stay aligned).
    */
  val PipeLatency: Int = 4
}
