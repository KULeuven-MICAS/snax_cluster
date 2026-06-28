// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Native-Chisel ports of the fpnew shared sub-blocks used by FpAdd/FpMul/FpFma:
//   - classify   (port of fpnew_classifier_snax, is_boxed always 1)
//   - roundRNE   (port of fpnew_rounding_snax, RNE only — the only mode used here)
//   - lzc        (leading-zero count; matches lzc_snax MODE=1 for non-zero inputs, which is all the
//                 datapaths rely on — the all-zero case is gated by `empty`)
// Format metadata reuses fp_unit.FpType (expWidth/sigWidth/width).

package fp_native

import chisel3._
import chisel3.util._

import fp_unit._

/** Per-operand classification flags (subset of fpnew_pkg_snax::fp_info_t actually consumed). */
class FpInfo extends Bundle {
  val isNormal    = Bool()
  val isSubnormal = Bool()
  val isZero      = Bool()
  val isInf       = Bool()
  val isNan       = Bool()
}

object FpCommon {

  /** IEEE-754 symmetric bias for a format: 2^(expWidth-1) - 1. */
  def bias(t: FpType): Int = (1 << (t.expWidth - 1)) - 1

  /** Classify a raw operand of format `t` (is_boxed assumed 1, as the fpnew blackboxes wire it). */
  def classify(t: FpType, op: UInt): FpInfo = {
    val exp = op(t.width - 2, t.sigWidth) // expWidth bits
    val man = op(t.sigWidth - 1, 0)       // sigWidth bits
    val expAllOnes = exp.andR
    val expZero    = exp === 0.U
    val manZero    = man === 0.U
    val info = Wire(new FpInfo)
    info.isNormal    := !expZero && !expAllOnes
    info.isZero      := expZero && manZero
    info.isSubnormal := expZero && !manZero
    info.isInf       := expAllOnes && manZero
    info.isNan       := expAllOnes && !manZero
    info
  }

  /** RNE rounding (port of fpnew_rounding_snax with rnd_mode = RNE). Returns the rounded absolute
    * value (truncated to absWidth — the carry ripples exp→inf automagically) and the result sign. */
  def roundRNE(absValue: UInt, sign: Bool, roundSticky: UInt, effSub: Bool): (UInt, Bool) = {
    val absWidth = absValue.getWidth
    // RNE: 2'b00/2'b01 -> down ; 2'b10 -> to even (LSB) ; 2'b11 -> up
    val roundUp    = (roundSticky === "b11".U) || ((roundSticky === "b10".U) && absValue(0))
    val absRounded = (absValue +& roundUp.asUInt)(absWidth - 1, 0)
    val exactZero  = (absValue === 0.U) && (roundSticky === 0.U)
    // rnd_mode == RNE (never RDN), so the zero-tie-break sign is always 0 (+0)
    val signOut    = Mux(exactZero && effSub, false.B, sign)
    (absRounded, signOut)
  }

  /** Leading-zero count over `width` bits (MODE=1). `count` = #leading zeros from the MSB for a
    * non-zero input; `empty` asserted when all-zero. Consumers branch on `empty` and ignore `count`
    * when empty, so only the non-zero count must match lzc_snax. */
  def lzc(in: UInt, width: Int): (UInt, Bool) = {
    val w = Wire(UInt(width.W)) // explicit width: RegNext inputs defer width inference, which breaks Log2
    w := in
    val cntW  = log2Ceil(width)
    val empty = w === 0.U
    val count = ((width - 1).U(cntW.W) - Log2(w))(cntW - 1, 0)
    (count, empty)
  }
}
