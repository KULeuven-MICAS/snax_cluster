package snax_acc.spatial_array

import chisel3._
import chisel3.util._

class FPMULFPBlackBox(
  topmodule: String,
  widthA:    Int,
  widthB:    Int,
  widthC:    Int
) extends BlackBox
    with HasBlackBoxResource {

  val io = IO(new Bundle {
    val operand_a_i = Input(UInt(widthA.W))
    val operand_b_i = Input(UInt(widthB.W))
    val result_o    = Output(UInt(widthC.W))
  })
  override def desiredName: String = topmodule

  addResource("src_fp_mul/fpnew_pkg.sv")
  addResource("src_fp_mul/fpnew_classifier.sv")
  addResource("src_fp_mul/fpnew_rounding.sv")
  addResource("src_fp_mul/lzc.sv")
  addResource("src_fp_mul/fp_mul.sv")

}

class FPMULFP(
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
    new FPMULFPBlackBox(topmodule, widthA, widthB, widthC)
  )

  io.result_o              := sv_module.io.result_o
  sv_module.io.operand_a_i := io.operand_a_i
  sv_module.io.operand_b_i := io.operand_b_i

}

object FPMULFPEmitter extends App {
  emitVerilog(
    new FPMULFP(
      topmodule = "fp_mul",
      widthA    = 16,
      widthB    = 16,
      widthC    = 32
    ),
    Array("--target-dir", "generated/SpatialArray")
  )
}
