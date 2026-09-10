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
  expLutN:   Int     = 128,
  siluN:     Int     = 256
) extends Module
    with RequireAsyncReset {
  // Parameter-derived name: see the note in fp_native.FpAdd. Two blocks in one design can hold different
  // FpActivation builds (exp-only vs exp+silu, different LUT depths) and must not share a module name.
  override def desiredName =
    s"FpActivation" + (if (hasExp) s"_exp$expLutN" else "") + (if (hasSilu) s"_silu$siluN" else "") +
      (if (pipelined) "_pipe" else "")
  require(hasExp || hasSilu, "FpActivation: at least one of exp/silu must be built")
  require(!hasExp || isPow2(expLutN), "FpActivation: expLutN must be a power of two")
  require(!hasSilu || isPow2(siluN), "FpActivation: siluN must be a power of two")
  val io = IO(new Bundle {
    val in   = Input(UInt(32.W))  // FP32
    val func = Input(Bool())      // false = exp, true = silu (ignored when only one is built)
    val out  = Output(UInt(32.W)) // FP32
  })

  import FpHelpers._

  val both  = hasExp && hasSilu
  val isExp = if (both) !io.func else hasExp.B // compile-time constant when a single function is built
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

  // silu: idx = |x|/H over [0,16]; g(m)=sigmoid(-m) tabulated (base) with central-difference slope
  val LOGN_S = if (hasSilu) log2Ceil(siluN) else 0
  val XHI    = 16.0
  val H      = XHI / siluN
  val SCALE  = f32lit((1.0 / H).toFloat)
  val BIAS   = ZERO // -INVH*XLO with XLO=0
  val HI_S   = f32lit(XHI.toFloat)
  def sigmoid(xx: Double): Double = 1.0 / (1.0 + math.exp(-xx))
  def gnode(i:    Int):    Double = sigmoid(-(i * H))
  val base  = if (hasSilu) VecInit((0 until siluN).map(i => f32lit(gnode(i).toFloat))) else VecInit(Seq(ZERO))
  val slope =
    if (hasSilu) VecInit((0 until siluN).map(i => f32lit(((gnode(i + 1) - gnode(i - 1)) / 2.0).toFloat)))
    else VecInit(Seq(ZERO))

  // ---- S0: input-affine. exp: fmul(clamp(x),LOG2EF_N) == ffma(_, _, +0); silu: ffma(|x|clamped, SCALE, 0) ----
  val preExp = if (hasExp) fp32max(fp32min(io.in, HI_E), LO_E) else io.in
  val preSil = if (hasSilu) fp32min(fabs(io.in), HI_S) else io.in
  val s0     = sr(ffma(Mux(isExp, preExp, preSil), Mux(isExp, LOG2EF_N, SCALE), Mux(isExp, ZERO, BIAS))) // reg0
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
  val idxS      = if (hasSilu) Mux(iM > (siluN - 1).S, (siluN - 1).U, iM.asUInt(LOGN_S - 1, 0)) else 0.U
  val lutV      = lut(idxE)
  val baseV     = base(idxS)
  val slopeV    = slope(idxS)
  // interp: exp corr = frac*LN2_N + 1 ; silu gpos = frac*slope + base  (one shared ffma; same as the old
  // FpExp/FpSilu first S3 op)
  val interp    = ffma(frac, Mux(isExp, LN2_N, slopeV), Mux(isExp, ONE, baseV))
  // post — KEEP the original per-func op (fmul for exp, fadd for silu), NOT a shared ffma. The S3 critical
  // path stays ffma->{fmul|fadd} (2 FP ops/stage) exactly like the old FpExp/FpSilu; folding both into one
  // ffma would make S3 ffma->ffma (a deeper combinational path) and risk timing. Bit-identical either way
  // (fmul(a,b)==ffma(a,b,+0); fadd(1,-g)==ffma(g,-1,+1)).
  val expTwoF   = if (hasExp) fmul(lutV, interp) else ZERO              // exp: lut * corr
  val siluOneMg = if (hasSilu) fadd(ONE, fneg(interp)) else ZERO        // silu: 1 - g
  // exp keeps twoF; silu picks g (x<=0) vs 1-g (x>0)
  val r3        = sr(Mux(isExp, expTwoF, Mux(sgn2, interp, siluOneMg))) // reg3
  val nE        = if (hasExp) sr(iM >> LOGN_E) else sr(0.S)             // reg3 : exp integer part of m
  val xin3      = sr(xin2)

  // ---- S4: exp scales by 2^n (exp-field construct); silu multiplies by the original x ----
  // `nE + 127` is written straight into the 8-bit exponent field, so an `nE` below -127 WRAPS: at the input
  // clamp edge (`in <= -88.035`, i.e. `iM <= -16257`) `nE = -128` gives `(nE+127) & 0xFF = 0xFF` = +Inf --
  // the LARGEST representable value where the true result is ~0. Flush to zero instead. That is the correct
  // limit, and it is what makes exp TOTAL (`exp(-inf) = 0`), which the monoid fold's identity padding depends
  // on: two shards whose maxima differ by more than ~88 otherwise rescale by +Inf. `nE` is registered at S3,
  // so this mux lands in S4 in front of a single fmul, off the deep S0/S3 cones.
  val pow2n = if (hasExp) Mux(nE < (-127).S, ZERO, Cat(0.U(1.W), (nE + 127.S).asUInt(7, 0), 0.U(23.W))) else ZERO
  io.out := fmul(r3, Mux(isExp, pow2n, xin3))
}

object FpActivation {

  /** Register-stage latency of the pipelined datapath (matches the former FpExp/FpSilu PipeLatency so StreamMap's
    * LINEAR/EXP/SILU func-mux branches stay aligned).
    */
  val PipeLatency: Int = 4
}
