package snax_acc.spatial_array

import chisel3._

class AdderIO(
  inputAElemWidth:  Int,
  inputBElemWidth:  Int,
  outputCElemWidth: Int
) extends Bundle {
  val in_a  = Input(UInt(inputAElemWidth.W))
  val in_b  = Input(UInt(inputBElemWidth.W))
  val out_c = Output(UInt(outputCElemWidth.W))
}

class Adder(
  opType:           Int,
  inputAElemWidth:  Int,
  inputBElemWidth:  Int,
  outputCElemWidth: Int
) extends Module
    with RequireAsyncReset {
  val io = IO(new AdderIO(inputAElemWidth, inputBElemWidth, outputCElemWidth))
  require(
    opType == OpType.UIntUIntOp || opType == OpType.SIntSIntOp ||
      opType == OpType.Float16Int4Op || opType == OpType.Float16Float16Op
  )
  require(
    inputAElemWidth > 0 && inputBElemWidth > 0 && outputCElemWidth > 0,
    "Element widths must be greater than 0"
  )
  if (opType == OpType.UIntUIntOp) {
    io.out_c := io.in_a + io.in_b
  } else if (opType == OpType.SIntSIntOp) {
    io.out_c := (io.in_a.asTypeOf(SInt(outputCElemWidth.W)) + io.in_b.asTypeOf(
      SInt(outputCElemWidth.W)
    )).asUInt
  } else if (opType == OpType.Float16Int4Op || opType == OpType.Float16Float16Op) {
    // For Float16Int4Op and Float16Float16Op, we use a black box for floating-point addition
    // now only support fp32+fp32=fp32, as the system verilog module's parameter is fixed
    val fpAddfp = Module(
      new FPAddFPBlackBox("fp_add_fp", inputAElemWidth, inputBElemWidth, outputCElemWidth)
    )
    fpAddfp.io.operand_a_i  := io.in_a
    fpAddfp.io.operand_b_i  := io.in_b
    io.out_c                := fpAddfp.io.result_o
    assert(
      inputAElemWidth == 32 && inputBElemWidth == 32 && outputCElemWidth == 32,
      "For Float16Int4Op or Float16Float16Op, input widths must be 32, 32 and output width must be 32 for the adder module"
    )

  } else {
    // TODO: add support for other types
    // For now, just set the output to 0
    io.out_c := 0.U
  }
}

object AdderEmitterUInt extends App {
  emitVerilog(
    new Adder(OpType.UIntUIntOp, 8, 4, 16),
    Array("--target-dir", "generated/SpatialArray")
  )
}

object AdderEmitterSInt extends App {
  emitVerilog(
    new Adder(OpType.SIntSIntOp, 8, 4, 16),
    Array("--target-dir", "generated/SpatialArray")
  )
}

object AdderEmitterFloat16Float16 extends App {
  emitVerilog(
    new Adder(OpType.Float16Float16Op, 32, 32, 32),
    Array("--target-dir", "generated/SpatialArray")
  )
}
