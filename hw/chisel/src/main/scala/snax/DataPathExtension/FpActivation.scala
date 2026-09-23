package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** FpActivation: one per-lane activation core that computes exp(x), silu(x) OR rsqrt(x) (FP32 in/out), selected at
  * runtime by `io.func` (1 = exp, 2 = silu, 3 = rsqrt — the same encoding StreamMap's func CSR uses). It MERGES the
  * former standalone FpExp and FpSilu, which were both instantiated per StreamMap lane yet are mutually exclusive at
  * runtime. All three share the exact same 4-stage skeleton — input-affine -> magic-round -> fractional split -> LUT +
  * interpolation FMA -> post-transform — so all seven FP units are shared (operands muxed by `func`); only the ROMs
  * stay per-function (exp: one 2^(i/lutN) table; silu and rsqrt: base+slope tables). This halves the activation FP
  * area of StreamMap versus separate modules, bit-exactly (see the fmul==ffma(_,_,+0) and 1-g==ffma(g,-1,+1)
  * identities below), while keeping the same PipeLatency=4 the func-mux relies on.
  *
  * RSQRT IS WHY THIS MODULE EARNED A THIRD FUNCTION. Every per-row normalisation on this block — rmsnorm's
  * 1/sqrt(mean), softmax's 1/Sexp — reduces to a scalar, and until now that scalar had to leave the datapath for an
  * FPU-less core to invert. Measured on the [32,128] rmsnorm the core's integer sqrt+reciprocal was 52% of the WHOLE
  * kernel (six serial `divu` per row). It does not need a pass of its own: the broadcast that follows the reduce is
  * already a StreamMap, so `a = 1/D, func = RSQRT` turns that pass's identity multiply into the rsqrt and the scalar
  * round trip disappears for free.
  *
  * THE RSQRT PATH IS NOT AN EXP/SILU-STYLE TABLE OF THE INPUT VALUE — it is a table of the SIGNIFICAND, because
  * rsqrt's exponent is exact and only the significand needs approximating. Split x = 2^e * m with m in [1,2), fold
  * the exponent's odd bit into the significand so m' = 2^p * m lies in [1,4) with (e-p) even, and
  *
  *     rsqrt(x) = 2^(-(e-p)/2) * rsqrt(m')
  *
  * The left factor is a shift and an exponent-field write — exact, no error, no table. The right factor is what the
  * shared skeleton computes: the affine maps m' in [1,4) onto the table index, the magic round splits it into node +
  * frac, and one interpolation FMA evaluates base[i] + frac*slope[i]. That is exactly silu's S3 shape, so the ffma is
  * shared; and S4's multiply by a constructed power of two is exactly exp's, so the fmul is shared too. The only new
  * hardware is the two ROMs, the m' bit-build, and a 9-bit exponent carried alongside `xin`.
  *
  * TOTAL, like exp. x <= 0, a subnormal, an infinity or a NaN all flush the output to zero rather than producing a
  * wrapped exponent. rsqrt's true domain is x > 0, and the caller that matters here is a sum of squares, so the only
  * reachable edge is an all-zero row: 0 gives y = x*0 = 0, which is the answer an eps-guarded rmsnorm approximates,
  * where +Inf would poison the whole row.
  *
  * When only one function is built the mux conditions constant-fold, degenerating to the single pipeline.
  */
class FpActivation(
  pipelined: Boolean = false,
  hasExp:    Boolean = true,
  hasSilu:   Boolean = true,
  expLutN:   Int     = 128,
  siluN:     Int     = 256,
  // hasRsqrt/rsqN go LAST and default off so the existing positional call sites keep their meaning.
  hasRsqrt:  Boolean = false,
  rsqN:      Int     = 64
) extends Module
    with RequireAsyncReset {
  // Parameter-derived name: see the note in fp_native.FpAdd. Two blocks in one design can hold different
  // FpActivation builds (exp-only vs exp+silu, different LUT depths) and must not share a module name.
  override def desiredName =
    s"FpActivation" + (if (hasExp) s"_exp$expLutN" else "") + (if (hasSilu) s"_silu$siluN" else "") +
      (if (hasRsqrt) s"_rsq$rsqN" else "") + (if (pipelined) "_pipe" else "")
  require(hasExp || hasSilu || hasRsqrt, "FpActivation: at least one of exp/silu/rsqrt must be built")
  require(!hasExp || isPow2(expLutN), "FpActivation: expLutN must be a power of two")
  require(!hasSilu || isPow2(siluN), "FpActivation: siluN must be a power of two")
  require(!hasRsqrt || isPow2(rsqN), "FpActivation: rsqN must be a power of two")
  val io = IO(new Bundle {
    val in   = Input(UInt(32.W))  // FP32
    // 1 = exp, 2 = silu, 3 = rsqrt; ignored (and constant-folded away) when only one is built.
    val func = Input(UInt(2.W))
    val out  = Output(UInt(32.W)) // FP32
  })

  import FpHelpers._

  // A build with exactly one function folds its select to a constant; with several, `func` picks.
  val nBuilt = Seq(hasExp, hasSilu, hasRsqrt).count(identity)
  val isExp  = if (nBuilt > 1) io.func === FpActivation.EXP.U else hasExp.B
  val isRsq  = if (nBuilt > 1) io.func === FpActivation.RSQRT.U else hasRsqrt.B
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

  // rsqrt: the reduced significand m' runs over [1,4), so node i sits at 1 + 3i/rsqN and the affine that
  // maps m' onto the index is (m' - 1) * rsqN/3. `slopeR` is the ANALYTIC derivative d(rsqrt)/d(index) at
  // the node, not silu's central difference: rsqrt is smooth and convex here, so the tangent line is the
  // better of the two and needs no neighbour. Error of the tangent over |frac| <= 1/2 is
  // (1/8)(3/rsqN)^2 * max|f''| = 0.09375 * (3/rsqN)^2; at rsqN=64 that is 2.1e-4, versus an FP16 ULP of
  // 9.8e-4 at 1.0 -- a fifth of an ULP, with a 64x2x32b ROM instead of silu's 256x2.
  val LOGN_R  = if (hasRsqrt) log2Ceil(rsqN) else 0
  val HR      = 3.0 / rsqN.toDouble                       // node spacing in the m' domain
  val SCALE_R = f32lit((1.0 / HR).toFloat)                // m' -> index
  val BIAS_R  = f32lit((-1.0 / HR).toFloat)               // ... minus the m'=1 origin
  def rnode(i: Int): Double = 1.0 + i.toDouble * HR
  val baseR   =
    if (hasRsqrt) VecInit((0 until rsqN).map(i => f32lit((1.0 / math.sqrt(rnode(i))).toFloat)))
    else VecInit(Seq(ZERO))
  val slopeR  =
    if (hasRsqrt) VecInit((0 until rsqN).map(i => f32lit((-0.5 * math.pow(rnode(i), -1.5) * HR).toFloat)))
    else VecInit(Seq(ZERO))

  // m' = 2^p * m, built by OVERWRITING x's exponent field with 127+p and forcing the sign positive. The
  // bias is odd, so e = ex-127 is odd exactly when ex is EVEN, hence p = ~ex[0]. The result is always a
  // positive normal in [1,4) whatever x was, which is what keeps the FP units downstream out of trouble on
  // the flushed inputs below.
  val exIn   = io.in(30, 23)
  val parity = ~exIn(0)                                       // p
  val mPrime = Cat(0.U(1.W), (127.U(8.W) + parity), io.in(22, 0))
  // Result exponent field: -(e - p)/2 + 127. (e-p) is even by construction, so the shift is exact; e spans
  // [-126,127] so the field lands in [63,190] and can never wrap.
  val eUnb   = (exIn.zext - 127.S).asSInt                     // e
  val nR     = -((eUnb - parity.zext) >> 1)                   // -(e-p)/2
  val eFieldR = (nR + 127.S)(7, 0).asUInt
  // TOTAL: x <= 0, subnormal, inf or NaN -> 0. See the header.
  val flushR = io.in(31) || (exIn === 0.U) || (exIn === 255.U)

  // ---- S0: input-affine. exp: fmul(clamp(x),LOG2EF_N) == ffma(_, _, +0); silu: ffma(|x|clamped, SCALE, 0) ----
  val preExp = if (hasExp) fp32max(fp32min(io.in, HI_E), LO_E) else io.in
  val preSil = if (hasSilu) fp32min(fabs(io.in), HI_S) else io.in
  // rsqrt shares silu's slot in the affine: both are "scale the reduced argument onto a node index".
  val preNL  = if (hasRsqrt) Mux(isRsq, mPrime, preSil) else preSil
  val sclNL  = if (hasRsqrt) Mux(isRsq, SCALE_R, SCALE) else SCALE
  val biaNL  = if (hasRsqrt) Mux(isRsq, BIAS_R, BIAS) else BIAS
  val s0     = sr(ffma(Mux(isExp, preExp, preNL), Mux(isExp, LOG2EF_N, sclNL), Mux(isExp, ZERO, biaNL))) // reg0
  val xin0 = sr(io.in)     // reg0 : original signed x (silu final multiply)
  val sgn0 = sr(io.in(31)) // reg0 : sign of x (silu g/1-g reflection)
  // rsqrt's exponent is settled at S0 and only needed at S4, so it rides the pipe beside xin.
  val eR0  = sr(eFieldR)
  val fl0  = sr(flushR)

  // ---- S1: round to nearest integer/node via the magic add ----
  val rM   = sr(fadd(s0, MAGIC)) // reg1
  val s0d  = sr(s0)              // reg1 (carry the scaled value for frac)
  val xin1 = sr(xin0)
  val sgn1 = sr(sgn0)
  val eR1  = sr(eR0)
  val fl1  = sr(fl0)

  // ---- S2: integer index iM and fractional part frac in [-0.5,0.5] ----
  val roundF = fadd(rM, NMAGIC)             // round(s0) as float (exact)
  val frac   = sr(fadd(s0d, fneg(roundF)))  // reg2 : s0 - round(s0)
  val iM     = sr(rM.asSInt - 0x4b400000.S) // reg2 : round(s0) as int
  val xin2   = sr(xin1)
  val sgn2   = sr(sgn1)
  val eR2    = sr(eR1)
  val fl2    = sr(fl1)

  // ---- S3: LUT lookup + interpolation FMA, then post: exp -> lut*corr ; silu -> reflect (1-g on x>0) ----
  val idxE      = if (hasExp) iM.asUInt(LOGN_E - 1, 0) else 0.U
  val idxS      = if (hasSilu) Mux(iM > (siluN - 1).S, (siluN - 1).U, iM.asUInt(LOGN_S - 1, 0)) else 0.U
  // m' in [1,4) maps to s0 in [0, rsqN), so the round can reach rsqN at the very top -- clamp like silu.
  val idxR      = if (hasRsqrt) Mux(iM > (rsqN - 1).S, (rsqN - 1).U, iM.asUInt(LOGN_R - 1, 0)) else 0.U
  val lutV      = lut(idxE)
  // silu and rsqrt are both base+slope tables read at the same point in the pipe, so the ffma below is
  // shared and only the ROM outputs are muxed -- no extra FP unit for the third function.
  val baseV     = if (hasRsqrt) Mux(isRsq, baseR(idxR), base(idxS)) else base(idxS)
  val slopeV    = if (hasRsqrt) Mux(isRsq, slopeR(idxR), slope(idxS)) else slope(idxS)
  // interp: exp corr = frac*LN2_N + 1 ; silu gpos / rsqrt significand = frac*slope + base  (one shared
  // ffma; same as the old FpExp/FpSilu first S3 op)
  val interp    = ffma(frac, Mux(isExp, LN2_N, slopeV), Mux(isExp, ONE, baseV))
  // post — KEEP the original per-func op (fmul for exp, fadd for silu), NOT a shared ffma. The S3 critical
  // path stays ffma->{fmul|fadd} (2 FP ops/stage) exactly like the old FpExp/FpSilu; folding both into one
  // ffma would make S3 ffma->ffma (a deeper combinational path) and risk timing. Bit-identical either way
  // (fmul(a,b)==ffma(a,b,+0); fadd(1,-g)==ffma(g,-1,+1)).
  val expTwoF   = if (hasExp) fmul(lutV, interp) else ZERO              // exp: lut * corr
  val siluOneMg = if (hasSilu) fadd(ONE, fneg(interp)) else ZERO        // silu: 1 - g
  // exp keeps twoF; silu picks g (x<=0) vs 1-g (x>0); rsqrt's interpolation IS the significand, so it
  // needs no post-transform at all and skips straight to the S4 scale.
  val siluPick  = Mux(sgn2, interp, siluOneMg)
  val nonExp    = if (hasRsqrt) Mux(isRsq, interp, siluPick) else siluPick
  val r3        = sr(Mux(isExp, expTwoF, nonExp))                       // reg3
  val nE        = if (hasExp) sr(iM >> LOGN_E) else sr(0.S)             // reg3 : exp integer part of m
  val xin3      = sr(xin2)
  val eR3       = sr(eR2)
  val fl3       = sr(fl2)

  // ---- S4: exp scales by 2^n (exp-field construct); silu multiplies by the original x ----
  // `nE + 127` is written straight into the 8-bit exponent field, so an `nE` below -127 WRAPS: at the input
  // clamp edge (`in <= -88.035`, i.e. `iM <= -16257`) `nE = -128` gives `(nE+127) & 0xFF = 0xFF` = +Inf --
  // the LARGEST representable value where the true result is ~0. Flush to zero instead. That is the correct
  // limit, and it is what makes exp TOTAL (`exp(-inf) = 0`), which the monoid fold's identity padding depends
  // on: two shards whose maxima differ by more than ~88 otherwise rescale by +Inf. `nE` is registered at S3,
  // so this mux lands in S4 in front of a single fmul, off the deep S0/S3 cones.
  val pow2n = if (hasExp) Mux(nE < (-127).S, ZERO, Cat(0.U(1.W), (nE + 127.S).asUInt(7, 0), 0.U(23.W))) else ZERO
  // rsqrt scales by the exact 2^(-(e-p)/2) computed at S0; the flush cases multiply by zero, which is how
  // this function stays total without a separate output mux (r3 is always finite -- see mPrime).
  val pow2r = if (hasRsqrt) Mux(fl3, ZERO, Cat(0.U(1.W), eR3, 0.U(23.W))) else ZERO
  val scale = if (hasRsqrt) Mux(isRsq, pow2r, xin3) else xin3
  io.out := fmul(r3, Mux(isExp, pow2n, scale))
}

object FpActivation {

  /** Register-stage latency of the pipelined datapath (matches the former FpExp/FpSilu PipeLatency so StreamMap's
    * LINEAR/EXP/SILU/RSQRT func-mux branches stay aligned).
    */
  val PipeLatency: Int = 4

  /** `io.func` encoding. Deliberately the SAME numbering as StreamMap's func CSR field (0 = LINEAR, which never
    * reaches this module), so a kernel's CSR word and this mux cannot drift apart.
    */
  val EXP:   Int = 1
  val SILU:  Int = 2
  val RSQRT: Int = 3
}
