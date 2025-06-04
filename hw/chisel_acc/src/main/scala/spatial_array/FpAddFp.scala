// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi (xiaoling.yi@kuleuven.be)

package snax_acc.spatial_array

import chisel3._
import chisel3.util._

class FPAddFPBlackBox(
  topmodule: String,
  widthA:    Int,
  widthB:    Int,
  widthC:    Int
) extends BlackBox
    // no chisel parameters, only works for FP32 for a, FP32 for b, FP32 for result as it is hardcoded in the blackbox
    with HasBlackBoxResource {

  val io = IO(new Bundle {
    val operand_a_i = Input(UInt(widthA.W))
    val operand_b_i = Input(UInt(widthB.W))
    val result_o    = Output(UInt(widthC.W))
  })
  override def desiredName: String = topmodule

  addResource("common_block/fpnew_pkg.sv")
  addResource("common_block/fpnew_classifier.sv")
  addResource("common_block/fpnew_rounding.sv")
  addResource("common_block/lzc.sv")
  addResource("src_fp_add/fp_add.sv")

}

class FPAddFP(
  topmodule:  String,
  val widthA: Int,
  val widthB: Int,
  val widthC: Int
) extends Module
    with RequireAsyncReset {

  val io = IO(new Bundle {
    val operand_a_i = Input(UInt(widthA.W))
    val operand_b_i = Input(UInt(widthB.W))
    val result_o    = Output(UInt(widthC.W))
  })

  val sv_module = Module(
    new FPAddFPBlackBox(topmodule, widthA, widthB, widthC)
  )

  io.result_o              := sv_module.io.result_o
  sv_module.io.operand_a_i := io.operand_a_i
  sv_module.io.operand_b_i := io.operand_b_i

}

object FPAddFPEmitter extends App {
  emitVerilog(
    new FPAddFP(
      topmodule = "fp_add",
      widthA    = 16,
      widthB    = 16,
      widthC    = 32
    ),
    Array("--target-dir", "generated/SpatialArray")
  )
}
