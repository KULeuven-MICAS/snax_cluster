// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Native-Chisel FP add: a 1:1 port of fp_add.sv (a+b -> out, mixed in/out formats), with `numPipe`
// (0..2) feed-forward register stages at the adder / normalize boundaries. Replaces the FpAddFp /
// FpAddFpPipe SystemVerilog blackboxes. Bit-exact to fp_add.sv (proven by FpAddDiffTest).

package fp_native

import chisel3._
import chisel3.util._

import fp_unit._

class FpAdd(val typeA: FpType, val typeB: FpType, val typeC: FpType, val numPipe: Int = 0)
    extends Module
    with RequireAsyncReset {
  require(numPipe >= 0 && numPipe <= 2, "FpAdd: numPipe must be 0..2")
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
  // RegNext outputs); := truncates a wider source (mod 2^w, matching SV) and zero/sign-extends a narrower one.
  private def truncU(x: UInt, w: Int): UInt = { val u = Wire(UInt(w.W)); u := x; u }
  private def truncS(x: SInt, w: Int): SInt = { val s = Wire(SInt(w.W)); s := x; s }

  val EXP_A = typeA.expWidth; val MAN_A = typeA.sigWidth; val BIAS_A = bias(typeA); val PREC_A = MAN_A + 1
  val EXP_B = typeB.expWidth; val MAN_B = typeB.sigWidth; val BIAS_B = bias(typeB); val PREC_B = MAN_B + 1
  val EXP_C = typeC.expWidth; val MAN_C = typeC.sigWidth; val BIAS_C = bias(typeC); val PREC_C = MAN_C + 1
  val EXP_MAX            = math.max(EXP_A, math.max(EXP_B, EXP_C))
  val SUM_W             = 2 * PREC_C + 2   // width of addend/sum vectors ([2*PREC_C+1:0])
  val LZC_W            = log2Ceil(SUM_W)
  val PREC_IN          = math.max(PREC_A, PREC_B)

  // ============================ SEG1: classify + align + add ============================
  val sign_a = io.in_a(typeA.width - 1); val exp_a = io.in_a(typeA.width - 2, MAN_A); val man_a = io.in_a(MAN_A - 1, 0)
  val sign_b = io.in_b(typeB.width - 1); val exp_b = io.in_b(typeB.width - 2, MAN_B); val man_b = io.in_b(MAN_B - 1, 0)
  val ia = classify(typeA, io.in_a)
  val ib = classify(typeB, io.in_b)

  val effSub        = sign_a ^ sign_b
  val tentativeSign = sign_a

  // special cases (priority matches fp_add.sv)
  val expAll1C = ((BigInt(1) << EXP_C) - 1).U(EXP_C.W)
  val qnanManC = (BigInt(1) << (MAN_C - 1)).U(MAN_C.W)
  val resultIsSpecial = WireDefault(false.B)
  val specialResult   = WireDefault(Cat(0.U(1.W), expAll1C, qnanManC)) // canonical qNaN
  when(ia.isNan || ib.isNan) {
    resultIsSpecial := true.B
  }.elsewhen(ia.isZero && ib.isZero) {
    resultIsSpecial := true.B
    specialResult   := 0.U(typeC.width.W)
  }.elsewhen(ia.isInf && ib.isInf && effSub) {
    resultIsSpecial := true.B
  }.elsewhen(ia.isInf) {
    resultIsSpecial := true.B
    specialResult   := Cat(sign_a, expAll1C, 0.U(MAN_C.W))
  }.elsewhen(ib.isInf) {
    resultIsSpecial := true.B
    specialResult   := Cat(sign_b, expAll1C, 0.U(MAN_C.W))
  }

  // exponent datapath (exponents enter as unsigned value; is_subnormal makes SV eval unsigned)
  val ediffCalc =
    (exp_a.zext +& (BIAS_C - BIAS_A).S +& ia.isSubnormal.asUInt.zext) -& (exp_b.zext +& (BIAS_C - BIAS_B).S +& ib.isSubnormal.asUInt.zext)
  val exponent_difference = Mux(ia.isZero || ib.isZero, 0.S((EXP_MAX + 1).W), truncS(ediffCalc, EXP_MAX + 1))
  val tentExpA = truncS(exp_a.zext +& (BIAS_C - BIAS_A).S, EXP_MAX + 1)
  val tentExpB = truncS(exp_b.zext +& (BIAS_C - BIAS_B).S, EXP_MAX + 1)
  val tentative_exponent = Mux(exponent_difference < 0.S || ia.isZero, tentExpB, tentExpA)

  // alignment shifts
  val SAW = log2Ceil(SUM_W + PREC_C) // SHIFT_AMOUNT_WIDTH (matches $clog2(LOWER_SUM_WIDTH+PREC_C) closely enough; only magnitude matters)
  val shamt_a = Wire(UInt(SAW.W)); val shamt_b = Wire(UInt(SAW.W))
  when(exponent_difference <= (-(PREC_C + 1)).S) {
    shamt_a := (PREC_C + 1).U; shamt_b := 0.U
  }.elsewhen(exponent_difference < 0.S) {
    shamt_a := truncU((-exponent_difference).asUInt, SAW); shamt_b := 0.U
  }.elsewhen(exponent_difference < (PREC_C + 1).S) {
    shamt_a := 0.U; shamt_b := truncU(exponent_difference.asUInt, SAW)
  }.otherwise {
    shamt_a := 0.U; shamt_b := (PREC_C + 1).U
  }

  val mantissa_a = Cat(ia.isNormal, man_a) // PREC_A
  val mantissa_b = Cat(ib.isNormal, man_b) // PREC_B

  val SHIFT_A = 2 * PREC_C - (PREC_A - 1)
  val LEFT_A  = if (SHIFT_A > 0) SHIFT_A else 0
  val RIGHT_A = if (SHIFT_A < 0) -SHIFT_A else 0
  val SHIFT_B = 2 * PREC_C - (PREC_B - 1)
  val LEFT_B  = if (SHIFT_B > 0) SHIFT_B else 0
  val RIGHT_B = if (SHIFT_B < 0) -SHIFT_B else 0

  val addend_a = truncU(((mantissa_a << LEFT_A) >> (shamt_a +& RIGHT_A.U)), SUM_W)
  val addend_b = truncU(((mantissa_b << LEFT_B) >> (shamt_b +& RIGHT_B.U)), SUM_W)
  val shifted_b = Mux(effSub, ~addend_b, addend_b)
  val sum_raw   = truncU(addend_a +& shifted_b +& effSub.asUInt, SUM_W)
  val resultNeg = effSub && sum_raw(SUM_W - 1)
  // explicit-width Wire: RegNext defers width inference, which would leave sum_q widthless for Log2/lzc
  val sum = Wire(UInt(SUM_W.W))
  sum := Mux(resultNeg, truncU(~(sum_raw - 1.U), SUM_W), sum_raw)

  // operand_a_larger / final_sign
  val mantissa_a_ext = mantissa_a << (PREC_IN - PREC_A)
  val mantissa_b_ext = mantissa_b << (PREC_IN - PREC_B)
  val operandALarger = Mux(exponent_difference > 0.S, true.B, Mux(exponent_difference < 0.S, false.B, mantissa_a_ext > mantissa_b_ext))
  val final_sign = Mux(effSub, Mux(operandALarger, sign_a, sign_b), tentativeSign)

  // ---- boundary B1 ----
  val sum_q                = reg1(sum)
  val tentExp_q            = reg1(tentative_exponent)
  val finalSign_q          = reg1(final_sign)
  val resultIsSpecial_q    = reg1(resultIsSpecial)
  val specialResult_q      = reg1(specialResult)

  // ============================ SEG2: lzc + normalization ============================
  val (lzCount, lzcZeroes) = lzc(sum_q, SUM_W)
  val lzcSgn = lzCount.zext

  val norm_shamt = Wire(UInt(LZC_W.W))
  val final_exponent = Wire(SInt((EXP_C + 1).W))
  when((tentExp_q -& lzcSgn + 1.S) > 0.S && !lzcZeroes) {
    final_exponent := truncS(tentExp_q -& lzcSgn + 1.S, EXP_C + 1)
    norm_shamt := truncU(lzCount, LZC_W)
  }.elsewhen(tentExp_q === 0.S) {
    final_exponent := 0.S
    norm_shamt := 1.U
  }.otherwise {
    final_exponent := 0.S
    norm_shamt := truncU(tentExp_q.asUInt, LZC_W)
  }

  val shifted = truncU(sum_q << norm_shamt, SUM_W) // {final_mantissa, sticky_bits}
  val final_mantissa = shifted(SUM_W - 1, PREC_C + 1) // high PREC_C+1 bits
  val sticky_bits    = shifted(PREC_C, 0)             // low PREC_C+1 bits
  val sticky_after_norm = sticky_bits.orR

  // ---- boundary B2 ----
  val finalMan_q   = reg2(final_mantissa)
  val finalExp_q   = reg2(final_exponent)
  val sticky_q     = reg2(sticky_after_norm)
  val finalSign_q2 = reg2(finalSign_q)
  val resultIsSpecial_q2 = reg2(resultIsSpecial_q)
  val specialResult_q2   = reg2(specialResult_q)

  // ============================ SEG3: round + select ============================
  val ofBeforeRound = finalExp_q >= ((BigInt(1) << EXP_C) - 1).S
  val preRoundExp   = Mux(ofBeforeRound, ((BigInt(1) << EXP_C) - 2).U(EXP_C.W), finalExp_q.asUInt(EXP_C - 1, 0))
  val preRoundMan   = Mux(ofBeforeRound, ((BigInt(1) << MAN_C) - 1).U(MAN_C.W), finalMan_q(MAN_C, 1))
  val preRoundAbs   = Cat(preRoundExp, preRoundMan)
  val roundSticky   = Mux(ofBeforeRound, "b11".U(2.W), Cat(finalMan_q(0), sticky_q))
  // fp_add.sv hardcodes effective_subtraction_i=1'b0 into the rounder (unlike fp_fma.sv), so an exact-zero
  // effective subtraction keeps final_sign (can yield -0) instead of being forced to +0. Match that.
  val (roundedAbs, roundedSign) = roundRNE(preRoundAbs, finalSign_q2, roundSticky, false.B)

  io.out := Mux(resultIsSpecial_q2, specialResult_q2, Cat(roundedSign, roundedAbs))
}
