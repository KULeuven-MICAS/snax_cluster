package snax_acc.spatial_array

import chisel3._
import chisel3.util._

class FPNewFMAFPBlackBox(
  topmodule: String,
  widthA:    Int,
  widthB:    Int,
  widthC:    Int
) extends BlackBox
    with HasBlackBoxResource {

  val io = IO(new Bundle {
    val operand_a_i = Input(UInt(widthA.W))
    val operand_b_i = Input(UInt(widthB.W))
    val operand_c_i = Input(UInt(widthC.W))
    val result_o    = Output(UInt(widthC.W))
  })
  override def desiredName: String = topmodule

  addResource("baseline_fp16_fp16_fp32_fma/fpnew_pkg.sv")
  addResource("baseline_fp16_fp16_fp32_fma/cf_math_pkg.sv")
  addResource("baseline_fp16_fp16_fp32_fma/fpnew_classifier.sv")
  addResource("baseline_fp16_fp16_fp32_fma/fpnew_rounding.sv")
  addResource("baseline_fp16_fp16_fp32_fma/lzc.sv")
  addResource("baseline_fp16_fp16_fp32_fma/fpnew_fma_mixed.sv")

}

class FPNewFMAFP(
  topmodule:  String,
  val widthA: Int,
  val widthB: Int,
  val widthC: Int
) extends Module
    with RequireAsyncReset {

  val io = IO(new Bundle {
    val operand_a_i = Input(UInt(widthA.W))
    val operand_b_i = Input(UInt(widthB.W))
    val operand_c_i = Input(UInt(widthC.W))
    val result_o    = Output(UInt(widthC.W))
  })

  val sv_module = Module(
    new FPNewFMAFPBlackBox(topmodule, widthA, widthB, widthC)
  )

  io.result_o              := sv_module.io.result_o
  sv_module.io.operand_a_i := io.operand_a_i
  sv_module.io.operand_b_i := io.operand_b_i
  sv_module.io.operand_c_i := io.operand_c_i

}

object FPNewFMAFPEmitter extends App {
  emitVerilog(
    new FPNewFMAFP(
      topmodule = "fpnew_fma_mixed",
      widthA    = 16,
      widthB    = 16,
      widthC    = 32
    ),
    Array("--target-dir", "generated/SpatialArray")
  )
}
