// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Native-Chisel FP multiply: a 1:1 port of fp_mul.sv (a*b -> out, mixed in/out formats), with
// `numPipe` (0..2) feed-forward register stages at the product / normalize boundaries. Replaces the
// FpMulFp / FpMulFpPipe SystemVerilog blackboxes. Bit-exact to fp_mul.sv (proven by FpMulDiffTest).

package fp_native

import chisel3._
import chisel3.util._

import fp_unit._

class FpMul(val typeA: FpType, val typeB: FpType, val typeC: FpType, val numPipe: Int = 0)
    extends Module
    with RequireAsyncReset {
  require(numPipe >= 0 && numPipe <= 2, "FpMul: numPipe must be 0..2")
  val latency: Int = numPipe

  val io = IO(new Bundle {
    val in_a = Input(UInt(typeA.W))
    val in_b = Input(UInt(typeB.W))
    val out  = Output(UInt(typeC.W))
  })

  import FpCommon._

  private def reg1[T <: Data](u: T): T = if (numPipe >= 1) RegNext(u) else u
  private def reg2[T <: Data](u: T): T = if (numPipe >= 2) RegNext(u) else u
  // truncate/extend to w bits via an explicit-width Wire (avoids .getWidth, which throws on deferred-width
  // RegNext outputs); := truncates a wider source (mod 2^w, matching SV) and sign-extends a narrower one.
  private def truncS(x: SInt, w: Int): SInt = { val s = Wire(SInt(w.W)); s := x; s }

  val EXP_A = typeA.expWidth; val MAN_A = typeA.sigWidth; val BIAS_A = bias(typeA); val PREC_A = MAN_A + 1
  val EXP_B = typeB.expWidth; val MAN_B = typeB.sigWidth; val BIAS_B = bias(typeB); val PREC_B = MAN_B + 1
  val EXP_C = typeC.expWidth; val MAN_C = typeC.sigWidth; val BIAS_C = bias(typeC); val PREC_C = MAN_C + 1
  val MUL_WIDTH = PREC_A + PREC_B
  val LZC_W     = log2Ceil(MUL_WIDTH)
  val EXP_WIDTH = math.max(EXP_C + 2, LZC_W)

  // ============================ SEG1: classify + exp-product + multiply ============================
  val sign_a = io.in_a(typeA.width - 1); val exp_a = io.in_a(typeA.width - 2, MAN_A); val man_a = io.in_a(MAN_A - 1, 0)
  val sign_b = io.in_b(typeB.width - 1); val exp_b = io.in_b(typeB.width - 2, MAN_B); val man_b = io.in_b(MAN_B - 1, 0)
  val ia = classify(typeA, io.in_a)
  val ib = classify(typeB, io.in_b)

  val resultSign = sign_a ^ sign_b

  // special cases (priority matches fp_mul.sv)
  val anyNan   = ia.isNan || ib.isNan
  val anyInf   = ia.isInf || ib.isInf
  val expAll1C = ((BigInt(1) << EXP_C) - 1).U(EXP_C.W)
  val qnanManC = (BigInt(1) << (MAN_C - 1)).U(MAN_C.W)
  val resultIsSpecial = WireDefault(false.B)
  val specialResult   = WireDefault(Cat(0.U(1.W), expAll1C, qnanManC)) // canonical qNaN
  when((ia.isInf && ib.isZero) || (ia.isZero && ib.isInf)) {
    resultIsSpecial := true.B
    specialResult   := Cat(resultSign, expAll1C, qnanManC)
  }.elsewhen(anyNan) {
    resultIsSpecial := true.B
  }.elsewhen(anyInf) {
    resultIsSpecial := true.B
    specialResult   := Cat(resultSign, expAll1C, 0.U(MAN_C.W))
  }

  // exponent product: exponents enter as their unsigned value (SV evaluates the expression unsigned
  // because is_subnormal is unsigned), then the result is reinterpreted signed and truncated.
  val expProdCalc =
    exp_a.zext +& ia.isSubnormal.asUInt.zext +& exp_b.zext +& ib.isSubnormal.asUInt.zext +& (BIAS_C - BIAS_A - BIAS_B).S
  val exponent_product =
    Mux(ia.isZero || ib.isZero, (2 - BIAS_C).S(EXP_WIDTH.W), truncS(expProdCalc, EXP_WIDTH))

  val mantissa_a = Cat(ia.isNormal, man_a) // PREC_A
  val mantissa_b = Cat(ib.isNormal, man_b) // PREC_B
  // explicit-width Wire: RegNext defers width inference, which would leave product_q widthless for Log2/lzc
  val product = Wire(UInt(MUL_WIDTH.W))
  product := mantissa_a * mantissa_b

  // ---- boundary B1 ----
  val product_q          = reg1(product)
  val exponent_product_q = reg1(exponent_product)
  val resultSign_q       = reg1(resultSign)
  val resultIsSpecial_q  = reg1(resultIsSpecial)
  val specialResult_q    = reg1(specialResult)

  // ============================ SEG2: lzc + normalization ============================
  val PRODUCT_SHIFTED_WIDTH = MUL_WIDTH + 1
  val STICKY_BIT_WIDTH      = PRODUCT_SHIFTED_WIDTH - (PREC_C + 1)
  val (lzCount, lzcZeroes)  = lzc(product_q, MUL_WIDTH)
  val lzcSgn                = lzCount.zext

  val normCond = ((exponent_product_q -& lzcSgn + 1.S) > 0.S) && !lzcZeroes
  val normalized_exponent =
    Mux(normCond, truncS(exponent_product_q -& lzcSgn + 1.S, EXP_WIDTH), 0.S(EXP_WIDTH.W))

  val ps_norm = (product_q << (lzCount +& 1.U))(PRODUCT_SHIFTED_WIDTH - 1, 0)
  val ps_sub  = (((product_q << 1)(PRODUCT_SHIFTED_WIDTH - 1, 0)) >> (-exponent_product_q).asUInt)(PRODUCT_SHIFTED_WIDTH - 1, 0)
  val product_shifted = Mux(normCond, ps_norm, ps_sub)

  val final_mantissa    = Wire(UInt((PREC_C + 1).W))
  val sticky_after_norm = Wire(Bool())
  if (STICKY_BIT_WIDTH > 0) {
    final_mantissa    := product_shifted(PRODUCT_SHIFTED_WIDTH - 1, STICKY_BIT_WIDTH)
    sticky_after_norm := product_shifted(STICKY_BIT_WIDTH - 1, 0).orR
  } else {
    final_mantissa    := Cat(product_shifted, 0.U((-STICKY_BIT_WIDTH).W))
    sticky_after_norm := false.B
  }
  val final_exponent = normalized_exponent

  // ---- boundary B2 ----
  val final_mantissa_q    = reg2(final_mantissa)
  val final_exponent_q    = reg2(final_exponent)
  val sticky_after_norm_q = reg2(sticky_after_norm)
  val resultSign_q2       = reg2(resultSign_q)
  val resultIsSpecial_q2  = reg2(resultIsSpecial_q)
  val specialResult_q2    = reg2(specialResult_q)

  // ============================ SEG3: round + select ============================
  val ofBeforeRound = final_exponent_q >= ((BigInt(1) << EXP_C) - 1).S
  val preRoundExp   = Mux(ofBeforeRound, ((BigInt(1) << EXP_C) - 2).U(EXP_C.W), final_exponent_q.asUInt(EXP_C - 1, 0))
  val preRoundMan   = Mux(ofBeforeRound, ((BigInt(1) << MAN_C) - 1).U(MAN_C.W), final_mantissa_q(MAN_C, 1))
  val preRoundAbs   = Cat(preRoundExp, preRoundMan)
  val roundSticky   = Mux(ofBeforeRound, "b11".U(2.W), Cat(final_mantissa_q(0), sticky_after_norm_q))
  val (roundedAbs, roundedSign) = roundRNE(preRoundAbs, resultSign_q2, roundSticky, false.B)

  io.out := Mux(resultIsSpecial_q2, specialResult_q2, Cat(roundedSign, roundedAbs))
}
