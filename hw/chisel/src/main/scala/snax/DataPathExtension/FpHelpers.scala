package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._
import fp_native._

/** Shared FP primitives for the SIMD xDMA extensions. FP16 transport, FP32 internal.
  *
  * The native-Chisel FP units (fp_native.FpAdd/FpMul/FpFma, 1:1 ports of fpnew fp_add/fp_mul/fp_fma) support mixed
  * in/out formats. `widen` (transport->FP32) is an EXACT bit-manipulation convert (no FP unit — the value is exact in
  * FP32); `narrow` (FP32->transport, rounds) is still an add-with-zero. The mixed `ffmaT`/ `fmulT` multiply an FP32
  * operand by a raw transport value (a smaller multiplier than widening first). fp32max/min are pure Chisel (no NaN
  * handling — the host softmax path produces only finite values). Each arithmetic / convert call instantiates one
  * unit/block at the call site (legal inside a Module body).
  */
object FpHelpers {
  // FP32 / FP16 bit-pattern literals computed at elaboration
  def f32lit(v:    Float): UInt = BigInt(java.lang.Float.floatToIntBits(v).toLong & 0xffffffffL).U(32.W)
  def f16lit(bits: Int):   UInt = (bits & 0xffff).U(16.W)

  val FP32_ZERO = 0.U(32.W)
  val FP16_ZERO = 0.U(16.W)

  // Combinational FP32 max / min for finite operands.
  def fp32max(a: UInt, b: UInt): UInt = {
    val sa = a(31); val sb = b(31)
    Mux(Mux(sa =/= sb, !sa, Mux(sa, b >= a, a >= b)), a, b)
  }
  def fp32min(a: UInt, b: UInt): UInt = {
    val sa = a(31); val sb = b(31)
    Mux(Mux(sa =/= sb, sa, Mux(sa, a >= b, b >= a)), a, b)
  }
  def fp32aWins(a: UInt, b: UInt): Bool = { // a >= b (finite FP32)
    val sa = a(31); val sb = b(31)
    Mux(sa =/= sb, !sa, Mux(sa, b >= a, a >= b))
  }
  def fneg32(u: UInt): UInt = Cat(~u(31), u(30, 0))

  // ---- ONLINE-SOFTMAX MOMENT-MERGE (F2 / the in-transit nonlinear collective) ------------------------
  // Combine two flash statistics (m, l) = (running max, Sexp) under the max-rescaled monoid:
  //   (m,l) = ( max(ma,mb),  l_winner + l_loser * exp(m_loser - m_winner) )   [winner factor exp(0)=1, free]
  // In (m, S=l*exp(m)) coordinates this is just (max, +) => associative + commutative, identity (-inf, 0)
  // (doc 13 §2.2). One MAX, one subtract (loser-winner <= 0 so exp in (0,1], bounded), one EXP (reuse the
  // StreamMap/FpActivation FP32 exp LUT), one FMA. FP32-internal so it is exact regardless of transport fmt
  // (the stats travel FP32 via fp32out). `expLutN` = the exp LUT depth; numPipe threads the exp latency out
  // via the returned `lat` so a fold can align the value path.
  def momentMerge(ma: UInt, la: UInt, mb: UInt, lb: UInt, pipelined: Boolean = false,
                  expLutN: Int = 128): (UInt, UInt, Int) = {
    val aWins  = fp32aWins(ma, mb)
    val m      = Mux(aWins, ma, mb)
    val loserM = Mux(aWins, mb, ma)
    val winL   = Mux(aWins, la, lb)
    val losL   = Mux(aWins, lb, la)
    val delta  = fadd(loserM, fneg32(m))                     // loser - winner <= 0
    val exp    = Module(new FpActivation(pipelined, true, false, false, expLutN, 256))
    exp.io.in := delta; exp.io.func := false.B; exp.io.gelu := false.B
    val lat    = if (pipelined) FpActivation.PipeLatency else 0
    val losLA  = ShiftRegister(losL, lat); val winLA = ShiftRegister(winL, lat)
    // gate the loser's contribution: when its l is 0 (the monoid identity or an empty shard) it adds exactly
    // the winner -- avoids 0*exp(-huge) = 0*inf-garbage = NaN at the exp LUT's underflow edge. A real shard's
    // l = Sexp >= 1, so l==0 uniquely tags the identity.
    val l      = Mux(losLA === FP32_ZERO, winLA, ffma(losLA, exp.io.out, winLA)) // losL*exp(delta)+winL
    (ShiftRegister(m, lat), l, lat)
  }

  // FP32 arithmetic (a*b+c etc.) via the native-Chisel FP units. `numPipe` = internal pipeline depth of
  // the FP unit (the per-op "cutting" knob for timing; 0 = combinational). The host FSM accounts for it.
  def fadd(a: UInt, b: UInt, numPipe: Int = 0):                        UInt = {
    val m = Module(new FpAdd(FP32, FP32, FP32, numPipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }
  def fmul(a: UInt, b: UInt, numPipe: Int = 0):                        UInt = {
    val m = Module(new FpMul(FP32, FP32, FP32, numPipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }
  def ffma(a: UInt, b: UInt, c: UInt, numPipe: Int = 0):               UInt = {
    val m = Module(new FpFma(FP32, FP32, FP32, numPipe)); m.io.in_a := a; m.io.in_b := b; m.io.in_c := c; m.io.out
  }
  // Mixed-precision FMA: a (FP32) * xT (transport, e.g. FP16) + c (FP32) -> FP32. Since a transport value is
  // EXACT in FP32, feeding it raw (instead of widen()->FP32) is bit-identical but shrinks the multiplier from
  // PREC_A x PREC_C (24x24) to PREC_A x PREC_T (24x11 @FP16) AND drops the widen. Use wherever an FP32 operand
  // multiplies a transport-precision value.
  def ffmaT(a: UInt, xT: UInt, c: UInt, tX: FpType, numPipe: Int = 0): UInt = {
    val m = Module(new FpFma(FP32, tX, FP32, numPipe)); m.io.in_a := a; m.io.in_b := xT; m.io.in_c := c; m.io.out
  }
  def fmulT(a: UInt, xT: UInt, tX: FpType, numPipe: Int = 0): UInt = { // FP32 * transport -> FP32 (24xPREC_T mult)
    val m = Module(new FpMul(FP32, tX, FP32, numPipe)); m.io.in_a := a; m.io.in_b := xT; m.io.out
  }

  // EXACT transport->FP32 widen. Every sub-FP32 format (FP16/BF16/FP8) has <=23 mantissa & <=8 exp bits, so
  // it is representable in FP32 with NO rounding. This is a pure bit-manipulation convert (exp rebias +
  // mantissa left-align, with the one non-trivial case being a subnormal source, normalized via a small lzc
  // + shift) — it replaces `FpAdd(t,t,FP32, in_b=0)`, dropping the adder's operand-align barrel shifter and
  // rounding path while producing the identical FP32 value. `numPipe` is preserved as output delay so the
  // host FSM's latency accounting is unchanged.
  def widen(h: UInt, t: FpType, numPipe: Int = 0):  UInt = {
    if (t.width >= FP32.width) {
      ShiftRegister(h, numPipe) // transport already >= FP32: widen is identity
    } else {
      val expW       = t.expWidth
      val sigW       = t.sigWidth
      val biasT      = FpCommon.bias(t)
      val sign       = h(t.width - 1)
      val expT       = h(t.width - 2, sigW)
      val manT       = h(sigW - 1, 0)
      val expAllOnes = expT.andR
      val expZero    = expT === 0.U
      val manZero    = manT === 0.U
      val SIG32      = FP32.sigWidth                                             // 23
      // normal:   exp32 = expT - biasT + 127 ; man32 = manT << (23 - sigW)
      val expNorm    = (expT +& (127 - biasT).U(8.W))(7, 0)
      val manNorm    = (manT << (SIG32 - sigW))(SIG32 - 1, 0)
      // subnormal (exp==0, man!=0): leading one at bit (sigW-1 - lz) => exp32 = 127 - biasT - lz,
      //   shift the fraction up by (24 - sigW + lz) (drops the implicit 1 at bit 23).
      val (lz, _)    = FpCommon.lzc(manT, sigW)
      val expSub     = ((127 - biasT).U(9.W) - lz)(7, 0)
      val manSub     = (manT << ((SIG32 - sigW + 1).U +& lz))(SIG32 - 1, 0)
      val exp32      = Mux(
        expAllOnes,
        ((1 << 8) - 1).U(8.W), // inf/nan
        Mux(expZero, Mux(manZero, 0.U(8.W), expSub), expNorm)
      ) // zero / subnormal / normal
      val man32      = Mux(expZero, Mux(manZero, 0.U(SIG32.W), manSub), manNorm) // subnormal normalizes; else <<
      ShiftRegister(Cat(sign, exp32, man32), numPipe)
    }
  }
  def narrow(f: UInt, t: FpType, numPipe: Int = 0): UInt = {
    val m = Module(new FpAdd(FP32, FP32, t, numPipe)); m.io.in_a := f; m.io.in_b := FP32_ZERO; m.io.out
  }
  def square(h: UInt, t: FpType, numPipe: Int = 0): UInt = { // t*t -> FP32
    val m = Module(new FpMul(t, t, FP32, numPipe)); m.io.in_a := h; m.io.in_b := h; m.io.out
  }

  // FP16 <-> FP32 (the legacy fixed-precision aliases)
  def widenF16(h:  UInt): UInt = widen(h, FP16)
  def narrowF32(f: UInt): UInt = narrow(f, FP16)

  // ---- RUNTIME-selectable-precision edge converters -------------------------------------------------
  // The whole mechanism of runtime precision: internal compute is FP32 (format-agnostic), so a single
  // netlist serves FP16/BF16/FP8 by muxing the (cheap, compile-time) per-format widen/narrow with a 2-bit
  // `fmt` field (0=FP16, 1=BF16, 2=FP8). `fmtOf` maps an FpType -> its runtime code so goldens line up.
  // widenRt: raw transport bits (16-wide carrier; FP8 in the low 8) -> FP32. narrowRt: FP32 -> 16-wide
  // carrier (FP8 zero-extended in the low 8) so a common width muxes; the beat-packer places the field.
  val FMT_FP16 = 0
  val FMT_BF16 = 1
  val FMT_FP8  = 2
  def fmtOf(t: FpType): Int =
    if (t == FP16) FMT_FP16 else if (t == BF16) FMT_BF16 else if (t == FP8) FMT_FP8
    else throw new IllegalArgumentException(s"fmtOf: no runtime code for $t")

  def widenRt(h16: UInt, fmt: UInt, numPipe: Int = 0): UInt = {
    val wFP16 = widen(h16(15, 0), FP16, numPipe)
    val wBF16 = widen(h16(15, 0), BF16, numPipe)
    val wFP8  = widen(h16(7, 0), FP8, numPipe)
    MuxLookup(fmt, wFP16)(Seq(FMT_FP16.U -> wFP16, FMT_BF16.U -> wBF16, FMT_FP8.U -> wFP8))
  }
  def narrowRt(f: UInt, fmt: UInt, numPipe: Int = 0): UInt = {
    val nFP16 = narrow(f, FP16, numPipe)                 // 16b
    val nBF16 = narrow(f, BF16, numPipe)                 // 16b
    val nFP8  = Cat(0.U(8.W), narrow(f, FP8, numPipe))   // 8b -> 16b carrier
    MuxLookup(fmt, nFP16)(Seq(FMT_FP16.U -> nFP16, FMT_BF16.U -> nBF16, FMT_FP8.U -> nFP8))
  }

  // ---- MX (OCP Microscaling) INPUT widen ------------------------------------------------------------
  // MX = a block of 32 elements sharing one 8-bit E8M0 power-of-2 scale (value 2^(X-127)); each element is a
  // narrow FP type. widening an MX element is STILL an edge problem: decode the element to FP32 (the same
  // exact bit-manip `widen`, but the top exponent is a NORMAL value, not inf/nan, for the MX element types)
  // then ADD the block-scale exponent delta to the FP32 exponent — one integer add + a couple compares. No
  // FP unit, same ShiftRegister(numPipe) latency as widenRt, so the streaming FSM / roofline are unchanged.
  // The precision ladder widens to ~4x: MXFP4 = 4.25 bits/elem -> ~4x fewer beats/row than FP16 at the same
  // 512 b/cyc roofline. The per-block scale arrives as a small AGU side-stream (1 scale / 32 elements).
  //
  // MX element FpTypes (finite-focused; NaN codepoints not expected on the activation path). Defined locally
  // so the shared fp_unit subproject is untouched; only expWidth/sigWidth (used by the decode) matter here.
  val FP8E4M3  = new FpType { val expWidth = 4; val sigWidth = 3; val fpnewFormatEnum = "fpnew_pkg_snax::FP8ALT" }
  val FP6E3M2  = new FpType { val expWidth = 3; val sigWidth = 2; val fpnewFormatEnum = "" }
  val FP6E2M3  = new FpType { val expWidth = 2; val sigWidth = 3; val fpnewFormatEnum = "" }
  val FP4E2M1  = new FpType { val expWidth = 2; val sigWidth = 1; val fpnewFormatEnum = "" }

  // FINITE-only transport->FP32 decode: like `widen` but the all-ones exponent is a NORMAL value (MX element
  // types have no inf; E4M3's only NaN codepoint S.1111.111 decodes as a large normal, harmless on the
  // activation path). Exact (no rounding); numPipe delay kept for latency accounting.
  def widenFin(h: UInt, t: FpType, numPipe: Int = 0): UInt = {
    val sigW    = t.sigWidth
    val biasT   = FpCommon.bias(t)
    val sign    = h(t.width - 1)
    val expT    = h(t.width - 2, sigW)
    val manT    = h(sigW - 1, 0)
    val expZero = expT === 0.U
    val manZero = manT === 0.U
    val SIG32   = FP32.sigWidth
    val expNorm = (expT +& (127 - biasT).U(9.W))(7, 0)
    val manNorm = (manT << (SIG32 - sigW))(SIG32 - 1, 0)
    val (lz, _) = FpCommon.lzc(manT, sigW)
    val expSub  = ((127 - biasT).U(9.W) - lz)(7, 0)
    val manSub  = (manT << ((SIG32 - sigW + 1).U +& lz))(SIG32 - 1, 0)
    val exp32   = Mux(expZero, Mux(manZero, 0.U(8.W), expSub), expNorm)
    val man32   = Mux(expZero, Mux(manZero, 0.U(SIG32.W), manSub), manNorm)
    ShiftRegister(Cat(sign, exp32, man32), numPipe)
  }

  // widen an MX element by its block's E8M0 scale: FP32(element) then exponent += (scale - 127).
  def widenMX(elem: UInt, scaleE8M0: UInt, t: FpType, numPipe: Int = 0): UInt = {
    val w        = widenFin(elem, t, 0) // exact FP32 element (no scale yet), combinational
    val sign     = w(31); val exp = w(30, 23); val man = w(22, 0)
    val elemZero = (exp === 0.U) && (man === 0.U)
    val adj      = scaleE8M0.zext - 127.S           // E8M0 bias 127 -> signed exponent delta
    val expExt   = exp.zext + adj                   // ~10-bit signed
    val ovf      = expExt > 254.S
    val udf      = expExt < 1.S                      // <= 0 -> flush to zero (documented MX-subnormal approx)
    val expOut   = Mux(ovf, 255.U(8.W), Mux(udf, 0.U(8.W), expExt(7, 0)))
    val manOut   = Mux(ovf || udf, 0.U(23.W), man)
    val out      = Mux(elemZero || udf, Cat(sign, 0.U(31.W)),
                       Mux(ovf, Cat(sign, "h7F800000".U(31.W)), Cat(sign, expOut, manOut)))
    ShiftRegister(out, numPipe)
  }

  // 3-bit runtime fmt: 0/1/2 = plain FP16/BF16/FP8 (scale ignored -> widenRt); 3..7 = MX element types.
  val FMT_MXFP8_E5M2 = 3
  val FMT_MXFP8_E4M3 = 4
  val FMT_MXFP6_E3M2 = 5
  val FMT_MXFP6_E2M3 = 6
  val FMT_MXFP4_E2M1 = 7
  def isMX(fmt: UInt): Bool = fmt >= FMT_MXFP8_E5M2.U

  // ---- MX (OCP Microscaling) OUTPUT: derive the shared block scale + quantize each element -----------
  // narrowMX for a 32-element block = (1) a block-max-exponent reduction over the 32 FP32 values -> the shared
  // E8M0 scale = maxExp - emaxElem (OCP: the scale aligns the block's largest magnitude to the element's max
  // normal), (2) per element divide by 2^(scale-127) (exponent subtract) then RNE-narrow to the element type.
  // The block-max is a small fixed combinational/pipelined tree over the beat's lanes, so the writer still
  // drains 1 beat/cycle (roofline preserved). emaxElem = the element format's max NORMAL unbiased exponent
  // (E5M2=15, E4M3=8, E3M2=4, E2M3=2, E2M1=2), passed by the caller.
  // block-scale over the FP32 lanes of one MX block (max biased-exponent - emaxElem, clamped >= 0)
  def blockScaleE8M0(block: Seq[UInt], emaxElem: Int): UInt = {
    val maxExp = block.map(_(30, 23)).reduceLeft((a, b) => Mux(a >= b, a, b))
    Mux(maxExp > emaxElem.U, maxExp - emaxElem.U, 0.U(8.W))
  }
  // Custom bit-manip FP32 -> narrow-FP (RNE, saturate-to-max-normal, FTZ) for the FINITE MX element types
  // (E3M2/E2M3/E2M1) that fpnew's `narrow` can't emit. The reverse of widenFin: rebias, RNE-round the mantissa
  // to sigW bits (carry may bump the exponent), saturate (finite MX has no inf/nan so the top exp is a normal)
  // and flush-to-zero on underflow. Exact for the common case; matches a Scala RNE reference within 1 code.
  def narrowFin(f: UInt, t: FpType, numPipe: Int = 0): UInt = {
    val expW = t.expWidth; val sigW = t.sigWidth; val biasT = FpCommon.bias(t)
    val sign  = f(31); val eX = f(30, 23); val man23 = f(22, 0)
    val fZero = (eX === 0.U) && (man23 === 0.U)
    val eT0   = eX.zext -& (127 - biasT).S        // target biased exp (signed), assuming the implicit 1
    val maxExp = (1 << expW) - 1                   // finite MX top exp is a normal (no inf/nan)

    // ---- NORMAL path (eT0 >= 1): drop the low mantissa bits, RNE; a round carry may bump the exponent ----
    val drop   = 23 - sigW
    val kept   = man23(22, drop)                  // top sigW bits
    val rnd    = man23(drop - 1)                  // round bit
    val sticky = if (drop >= 2) man23(drop - 2, 0).orR else false.B
    val roundUp = rnd && (sticky || kept(0))      // RNE
    val manSum = Cat(0.U(1.W), kept) +& roundUp   // sigW+1 bits
    val carry  = manSum(sigW)
    val manF   = manSum(sigW - 1, 0)
    val eTn    = Mux(carry, eT0 + 1.S, eT0)
    val ovfN   = eTn > maxExp.S
    val normBody = Cat(sign, Mux(ovfN, maxExp.U(expW.W), eTn(expW - 1, 0)),
                             Mux(ovfN, ((1 << sigW) - 1).U(sigW.W), manF))

    // ---- SUBNORMAL path (eT0 <= 0): the finite MX grids HAVE subnormals, and E2M1's 0.5 is the code a
    // wide-dynamic-range MXFP4 block lands its small elements on (e.g. 2.0 in a max-24 block -> 2.0/4 = 0.5).
    // Encode subMant = RNE(fullSig >> sh), fullSig = 1.man23, sh = 24 - sigW - eT0; a round-up to 2^sigW promotes
    // the subnormal to the min normal (exp=1). WITHOUT this the whole subnormal binade FTZ'd to 0 -- the vsim
    // loopback showed 15% of elements (every value that scaled to 0.5) silently zeroed. ----
    val fullSig = Cat(1.U(1.W), man23)            // 24 bits: 1.man23 (bit23 = implicit 1)
    val shW     = (24 - sigW).S -& eT0            // >= 24-sigW since eT0 <= 0 here
    val sh      = Mux(shW >= 32.S, 31.U(5.W), shW(4, 0)) // clamp; >= 32 fully underflows to 0
    val one     = 1.U(48.W)
    val half    = (one << sh) >> 1                // 2^(sh-1) (sh >= 21 here, so >>1 is exact)
    val remMask = (one << sh) - one               // 2^sh - 1
    val remW    = Cat(0.U(24.W), fullSig) & remMask
    val qSub    = fullSig >> sh                    // integer part of fullSig/2^sh (0..1 for the top binade)
    val keptSub = qSub(sigW - 1, 0)
    val subRU   = (remW > half) || ((remW === half) && keptSub(0)) // RNE (tie to even)
    val subSum  = Cat(0.U(1.W), keptSub) +& subRU // sigW+1 bits; == 2^sigW -> promote to the min normal
    val subOvf  = subSum(sigW)
    val subBody = Cat(sign, Mux(subOvf, 1.U(expW.W), 0.U(expW.W)),
                            Mux(subOvf, 0.U(sigW.W), subSum(sigW - 1, 0)))

    val isSub = eT0 < 1.S
    ShiftRegister(Mux(fZero, Cat(sign, 0.U((expW + sigW).W)), Mux(isSub, subBody, normBody)), numPipe)
  }
  // narrow one FP32 element to an MX element type: fpnew for the 8-bit grids (E5M2/E4M3), narrowFin for the
  // finite sub-8-bit MX grids (E3M2/E2M3/E2M1).
  def narrowElem(f: UInt, t: FpType, numPipe: Int = 0): UInt =
    if (t == FP8 || t == FP8E4M3) narrow(f, t, numPipe) else narrowFin(f, t, numPipe)

  // quantize one FP32 value by the block's E8M0 scale into the narrow element type (FTZ on underflow after
  // the scale subtract; RNE via narrowElem).
  def narrowMX(f: UInt, scaleE8M0: UInt, t: FpType, numPipe: Int = 0): UInt = {
    val sign   = f(31); val eX = f(30, 23); val man = f(22, 0)
    // zero (eX==0; a widened FP16/BF16 value is never FP32-subnormal, so eX==0 means exactly 0) -> zero out.
    // WITHOUT this, an all-zero block gives blockScaleE8M0=0, and eShift = 0-(0-127) = 127 mis-scales 0.0 to a
    // normal (1.0) -> garbage codes. The guard makes a zero/all-zero block encode as zero (verified).
    val fZero  = eX === 0.U
    val adj    = scaleE8M0.zext - 127.S        // scale-127
    val eShift = eX.zext - adj                 // divide by 2^(scale-127): exponent -= (scale-127)
    val scaled = Mux(fZero || eShift < 1.S, Cat(sign, 0.U(31.W)),
                     Mux(eShift > 254.S, Cat(sign, 254.U(8.W), man), Cat(sign, eShift(7, 0), man)))
    val raw    = narrowElem(scaled, t, numPipe) // RNE to the narrow element grid
    // SATURATE overflow to max-normal. The OCP block scale aligns the block MAX to the element's top normal
    // binade, so a max element whose mantissa rounds up past the max normal overflows. For the fpnew 8-bit grids
    // (FP8=E5M2, FP8E4M3) the top exponent is the RESERVED inf/nan codepoint and fpnew's `narrow` EMITS inf --
    // MX quantization must never produce inf from a finite value, so clamp exp-all-ones to max-normal. (The
    // sub-8-bit finite MX grids use narrowFin, which already saturates and whose top exponent is a valid normal.)
    if (t == FP8 || t == FP8E4M3) {
      val eW = t.expWidth; val sW = t.sigWidth
      val maxNorm = Cat(raw(eW + sW), ((1 << eW) - 2).U(eW.W), ((1 << sW) - 1).U(sW.W))
      Mux(raw(eW + sW - 1, sW).andR, maxNorm, raw)
    } else raw
  }
  // runtime MX narrow: FP32 -> narrow MX element (16b carrier, low bits), muxed by fmt; non-MX -> narrowRt.
  def narrowMXRt(f: UInt, scaleE8M0: UInt, fmt: UInt, numPipe: Int = 0): UInt =
    MuxLookup(fmt, narrowRt(f, fmt, numPipe))(Seq(
      FMT_MXFP8_E5M2.U -> Cat(0.U(8.W), narrowMX(f, scaleE8M0, FP8,     numPipe)),
      FMT_MXFP8_E4M3.U -> Cat(0.U(8.W), narrowMX(f, scaleE8M0, FP8E4M3, numPipe)),
      FMT_MXFP6_E3M2.U -> Cat(0.U(10.W), narrowMX(f, scaleE8M0, FP6E3M2, numPipe)),
      FMT_MXFP6_E2M3.U -> Cat(0.U(10.W), narrowMX(f, scaleE8M0, FP6E2M3, numPipe)),
      FMT_MXFP4_E2M1.U -> Cat(0.U(12.W), narrowMX(f, scaleE8M0, FP4E2M1, numPipe))
    ))

  def widenMXRt(carrier: UInt, scaleE8M0: UInt, fmt: UInt, numPipe: Int = 0): UInt =
    MuxLookup(fmt, widenRt(carrier, fmt, numPipe))(Seq(
      FMT_MXFP8_E5M2.U -> widenMX(carrier(7, 0), scaleE8M0, FP8,     numPipe),
      FMT_MXFP8_E4M3.U -> widenMX(carrier(7, 0), scaleE8M0, FP8E4M3, numPipe),
      FMT_MXFP6_E3M2.U -> widenMX(carrier(5, 0), scaleE8M0, FP6E3M2, numPipe),
      FMT_MXFP6_E2M3.U -> widenMX(carrier(5, 0), scaleE8M0, FP6E2M3, numPipe),
      FMT_MXFP4_E2M1.U -> widenMX(carrier(3, 0), scaleE8M0, FP4E2M1, numPipe)
    ))
}

/** Parses the op-set list shared by the SIMD extensions. Each entry is "<OP>_<PRECISION>" where PRECISION is the
  * TRANSPORT (element) type the op processes — FP16 | BF16 | FP8 | FP32 (internal compute is always FP32). All entries
  * in one extension must share a single precision (one element width per stream). The precision is configurable from
  * the hjson and sets the datapath element width; only FP16 is currently exercised by the apps, the others elaborate
  * via the fpnew mixed-format blackboxes.
  */
object OpSpec {
  val precision: Map[String, FpType] = Map("FP16" -> FP16, "BF16" -> BF16, "FP8" -> FP8, "FP32" -> FP32)

  /** -> (bare op names, shared transport FpType). `allowedOps` validates the op names; `who` names the caller for error
    * messages.
    */
  def parse(entries: Seq[String], allowedOps: Set[String], who: String): (Seq[String], FpType) = {
    require(entries.nonEmpty, s"$who: op list must be non-empty")
    val split = entries.map { e =>
      val i = e.lastIndexOf('_')
      require(i > 0 && i < e.length - 1, s"$who: '$e' must be of the form <OP>_<PRECISION>")
      (e.substring(0, i), e.substring(i + 1))
    }
    val ops   = split.map(_._1)
    val precs = split.map(_._2).distinct
    require(ops.toSet.subsetOf(allowedOps), s"$who: ops must be a subset of $allowedOps, got $ops")
    require(precs.size == 1, s"$who: all ops must share one transport precision, got $precs")
    require(
      precision.contains(precs.head),
      s"$who: unknown precision '${precs.head}', known: ${precision.keys.mkString(", ")}"
    )
    (ops, precision(precs.head))
  }

  /** Validates an explicit config `elementWidth` against the width implied by the op-set precision (the precision tag
    * is the source of truth — FP16/BF16⇒16, FP8⇒8, FP32⇒32). The config carries the width only to make it visible; a
    * mismatch (e.g. elementWidth:8 with an FP16 op) is a config error.
    */
  def checkWidth(elementWidth: Int, transport: FpType, who: String): Unit =
    require(
      elementWidth == transport.width,
      s"$who: elementWidth=$elementWidth must equal the op precision width ${transport.width} ($transport)"
    )
}
