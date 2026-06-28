// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Native-Chisel fused multiply-add: a 1:1 port of fp_fma.sv (a*b + c -> out, out format = c format),
// with `numPipe` (0..2) feed-forward register stages at the adder / normalize boundaries. Replaces the
// FpFmaFp SystemVerilog blackbox. Bit-exact to fp_fma.sv (proven by FpFmaDiffTest). Production uses
// FP32^3 only (ffma), but the port stays format-parametric.

package fp_native

import chisel3._
import chisel3.util._

import fp_unit._

class FpFma(val typeA: FpType, val typeB: FpType, val typeC: FpType, val numPipe: Int = 0)
    extends Module
    with RequireAsyncReset {
  require(numPipe >= 0 && numPipe <= 2, "FpFma: numPipe must be 0..2")
  val latency: Int = numPipe

  val io = IO(new Bundle {
    val in_a = Input(UInt(typeA.W))
    val in_b = Input(UInt(typeB.W))
    val in_c = Input(UInt(typeC.W))
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

  val MUL_WIDTH          = math.max(PREC_A + PREC_B, PREC_C)
  val PRODUCT_SHIFT      = MUL_WIDTH - (PREC_A + PREC_B)
  val LOWER_SUM_WIDTH    = MUL_WIDTH + 3
  val LZC_RESULT_WIDTH   = log2Ceil(LOWER_SUM_WIDTH)
  val EXP_WIDTH          = math.max(math.max(EXP_C, math.max(EXP_A, EXP_B)) + 2, LZC_RESULT_WIDTH)
  val INTERMEDIATE_WIDTH = LOWER_SUM_WIDTH + PREC_C

  // ============================ SEG1: classify + product + addend align + add ============================
  val sign_a = io.in_a(typeA.width - 1); val exp_a = io.in_a(typeA.width - 2, MAN_A); val man_a = io.in_a(MAN_A - 1, 0)
  val sign_b = io.in_b(typeB.width - 1); val exp_b = io.in_b(typeB.width - 2, MAN_B); val man_b = io.in_b(MAN_B - 1, 0)
  val sign_c = io.in_c(typeC.width - 1); val exp_c = io.in_c(typeC.width - 2, MAN_C); val man_c = io.in_c(MAN_C - 1, 0)
  val ia = classify(typeA, io.in_a)
  val ib = classify(typeB, io.in_b)
  val ic = classify(typeC, io.in_c)

  val effSub        = sign_a ^ sign_b ^ sign_c
  val tentativeSign = sign_a ^ sign_b

  // special cases (priority matches fp_fma.sv)
  val expAll1C = ((BigInt(1) << EXP_C) - 1).U(EXP_C.W)
  val qnanManC = (BigInt(1) << (MAN_C - 1)).U(MAN_C.W)
  val anyNan = ia.isNan || ib.isNan || ic.isNan
  val anyInf = ia.isInf || ib.isInf || ic.isInf
  val resultIsSpecial = WireDefault(false.B)
  val specialResult   = WireDefault(Cat(0.U(1.W), expAll1C, qnanManC))
  when((ia.isInf && ib.isZero) || (ia.isZero && ib.isInf)) {
    resultIsSpecial := true.B
    specialResult   := Cat(sign_a ^ sign_b, expAll1C, qnanManC)
  }.elsewhen(anyNan) {
    resultIsSpecial := true.B
  }.elsewhen(anyInf) {
    resultIsSpecial := true.B
    when((ia.isInf || ib.isInf) && ic.isInf && effSub) {
      specialResult := Cat(0.U(1.W), expAll1C, qnanManC)
    }.elsewhen(ia.isInf || ib.isInf) {
      specialResult := Cat(sign_a ^ sign_b, expAll1C, 0.U(MAN_C.W))
    }.elsewhen(ic.isInf) {
      specialResult := Cat(sign_c, expAll1C, 0.U(MAN_C.W))
    }
  }

  // exponent datapath (exponents enter as unsigned value)
  val exponent_addend = truncS(exp_c.zext +& (!ic.isNormal).asUInt.zext, EXP_WIDTH)
  val expProdCalc =
    exp_a.zext +& ia.isSubnormal.asUInt.zext +& exp_b.zext +& ib.isSubnormal.asUInt.zext +& (BIAS_C - BIAS_A - BIAS_B).S
  val exponent_product =
    Mux(ia.isZero || ib.isZero, (2 - BIAS_A - BIAS_B + BIAS_C).S(EXP_WIDTH.W), truncS(expProdCalc, EXP_WIDTH))
  val exponent_difference = truncS(exponent_addend -& exponent_product, EXP_WIDTH)
  val tentative_exponent  = Mux(exponent_difference > 0.S, exponent_addend, exponent_product)

  val SAW = log2Ceil(INTERMEDIATE_WIDTH + 2)
  val addend_shamt = Wire(UInt(SAW.W))
  when(exponent_difference <= (-MUL_WIDTH - 1).S) {
    addend_shamt := (INTERMEDIATE_WIDTH + 1).U
  }.elsewhen(exponent_difference <= (PREC_C + 2).S) {
    addend_shamt := truncU((((PREC_C + 3).S) -& exponent_difference).asUInt, SAW)
  }.otherwise {
    addend_shamt := 0.U
  }

  val mantissa_a = Cat(ia.isNormal, man_a) // PREC_A
  val mantissa_b = Cat(ib.isNormal, man_b) // PREC_B
  val mantissa_c = Cat(ic.isNormal, man_c) // PREC_C
  val product    = truncU(mantissa_a * mantissa_b, MUL_WIDTH)
  val product_shifted = truncU(product << (PRODUCT_SHIFT + 2), INTERMEDIATE_WIDTH + 1) // [INTERMEDIATE_WIDTH:0]

  // addend align: {addend_after_shift[IW:0], addend_sticky[PREC_C-1:0]} = (mantissa_c << (IW+1)) >> addend_shamt
  val addendFull = truncU((mantissa_c << (INTERMEDIATE_WIDTH + 1)) >> addend_shamt, INTERMEDIATE_WIDTH + 1 + PREC_C)
  val addend_after_shift = addendFull(INTERMEDIATE_WIDTH + PREC_C, PREC_C) // high IW+1 bits
  val addend_sticky_bits = addendFull(PREC_C - 1, 0)
  val sticky_before_add  = addend_sticky_bits.orR
  val addend_shifted     = Mux(effSub, ~addend_after_shift, addend_after_shift)
  val inject_carry_in    = effSub && !sticky_before_add

  // adder
  val sum_raw   = truncU(product_shifted +& addend_shifted +& inject_carry_in.asUInt, INTERMEDIATE_WIDTH + 2)
  val sum_carry = sum_raw(INTERMEDIATE_WIDTH + 1)
  val sum = Wire(UInt((INTERMEDIATE_WIDTH + 1).W)) // explicit width for lzc
  sum := Mux(effSub && !sum_carry, truncU((-sum_raw), INTERMEDIATE_WIDTH + 1), truncU(sum_raw, INTERMEDIATE_WIDTH + 1))
  val final_sign = Mux(effSub && (sum_carry === tentativeSign), true.B, Mux(effSub, false.B, tentativeSign))

  // ---- boundary B1 ----
  val sum_q                = reg1(sum)
  val expProd_q            = reg1(exponent_product)
  val expDiff_q            = reg1(exponent_difference)
  val tentExp_q            = reg1(tentative_exponent)
  val addendShamt_q        = reg1(addend_shamt)
  val stickyBefore_q       = reg1(sticky_before_add)
  val finalSign_q          = reg1(final_sign)
  val effSub_q             = reg1(effSub)
  val resultIsSpecial_q    = reg1(resultIsSpecial)
  val specialResult_q      = reg1(specialResult)

  // ============================ SEG2: lzc + normalization ============================
  val sum_lower = sum_q(LOWER_SUM_WIDTH - 1, 0)
  val (lzCount, lzcZeroes) = lzc(sum_lower, LOWER_SUM_WIDTH)
  val lzcSgn = lzCount.zext

  val norm_shamt          = Wire(UInt(SAW.W))
  val normalized_exponent = Wire(SInt(EXP_WIDTH.W))
  when((expDiff_q <= 0.S) || (effSub_q && (expDiff_q <= 2.S))) {
    when(((expProd_q -& lzcSgn + 1.S) >= 0.S) && !lzcZeroes) {
      norm_shamt          := truncU(((PREC_C + 2).U +& lzCount), SAW)
      normalized_exponent := truncS(expProd_q -& lzcSgn + 1.S, EXP_WIDTH)
    }.otherwise {
      norm_shamt          := truncU(((PREC_C + 2).S +& expProd_q).asUInt, SAW)
      normalized_exponent := 0.S
    }
  }.otherwise {
    norm_shamt          := addendShamt_q
    normalized_exponent := tentExp_q
  }

  val sum_shifted = truncU(sum_q << norm_shamt, INTERMEDIATE_WIDTH + 2) // [IW+1:0]

  // small_norm: default split, then carry/msb/denormal fixups
  val final_mantissa = Wire(UInt((PREC_C + 1).W))
  val sum_sticky     = Wire(UInt(LOWER_SUM_WIDTH.W))
  val final_exponent = Wire(SInt(EXP_WIDTH.W))
  // default {final_mantissa, sum_sticky} = sum_shifted[IW:0]
  val splitDefault = sum_shifted(INTERMEDIATE_WIDTH, 0) // IW+1 bits = (PREC_C+1)+LOWER_SUM_WIDTH
  when(sum_shifted(INTERMEDIATE_WIDTH + 1)) { // carry
    val s = (sum_shifted >> 1)(INTERMEDIATE_WIDTH, 0)
    final_mantissa := s(INTERMEDIATE_WIDTH, LOWER_SUM_WIDTH)
    sum_sticky     := s(LOWER_SUM_WIDTH - 1, 0)
    final_exponent := truncS(normalized_exponent + 1.S, EXP_WIDTH)
  }.elsewhen(sum_shifted(INTERMEDIATE_WIDTH)) { // normal
    final_mantissa := splitDefault(INTERMEDIATE_WIDTH, LOWER_SUM_WIDTH)
    sum_sticky     := splitDefault(LOWER_SUM_WIDTH - 1, 0)
    final_exponent := normalized_exponent
  }.elsewhen(normalized_exponent > 1.S) { // denormal, align left
    val s = truncU(sum_shifted << 1, INTERMEDIATE_WIDTH + 1)
    final_mantissa := s(INTERMEDIATE_WIDTH, LOWER_SUM_WIDTH)
    sum_sticky     := s(LOWER_SUM_WIDTH - 1, 0)
    final_exponent := truncS(normalized_exponent - 1.S, EXP_WIDTH)
  }.otherwise {
    final_mantissa := splitDefault(INTERMEDIATE_WIDTH, LOWER_SUM_WIDTH)
    sum_sticky     := splitDefault(LOWER_SUM_WIDTH - 1, 0)
    final_exponent := 0.S
  }
  val sticky_after_norm = sum_sticky.orR || stickyBefore_q

  // ---- boundary B2 ----
  val finalMan_q   = reg2(final_mantissa)
  val finalExp_q   = reg2(final_exponent)
  val sticky_q     = reg2(sticky_after_norm)
  val finalSign_q2 = reg2(finalSign_q)
  val effSub_q2    = reg2(effSub_q)
  val resultIsSpecial_q2 = reg2(resultIsSpecial_q)
  val specialResult_q2   = reg2(specialResult_q)

  // ============================ SEG3: round + select ============================
  val ofBeforeRound = finalExp_q >= ((BigInt(1) << EXP_C) - 1).S
  val preRoundExp   = Mux(ofBeforeRound, ((BigInt(1) << EXP_C) - 2).U(EXP_C.W), finalExp_q.asUInt(EXP_C - 1, 0))
  val preRoundMan   = Mux(ofBeforeRound, ((BigInt(1) << MAN_C) - 1).U(MAN_C.W), finalMan_q(MAN_C, 1))
  val preRoundAbs   = Cat(preRoundExp, preRoundMan)
  val roundSticky   = Mux(ofBeforeRound, "b11".U(2.W), Cat(finalMan_q(0), sticky_q))
  val (roundedAbs, roundedSign) = roundRNE(preRoundAbs, finalSign_q2, roundSticky, effSub_q2)

  io.out := Mux(resultIsSpecial_q2, specialResult_q2, Cat(roundedSign, roundedAbs))
}
