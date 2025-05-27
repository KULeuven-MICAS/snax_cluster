package snax_acc.spatial_array

import chisel3._

class MultiplierIO(
  inputAElemWidth:  Int,
  inputBElemWidth:  Int,
  outputCElemWidth: Int
) extends Bundle {
  val in_a  = Input(UInt(inputAElemWidth.W))
  val in_b  = Input(UInt(inputBElemWidth.W))
  val out_c = Output(UInt(outputCElemWidth.W))
}

class Multiplier(
  opType:           Int,
  inputAElemWidth:  Int,
  inputBElemWidth:  Int,
  outputCElemWidth: Int
) extends Module
    with RequireAsyncReset {
  val io = IO(new MultiplierIO(inputAElemWidth, inputBElemWidth, outputCElemWidth))
  require(
    opType == OpType.UIntUIntOp || opType == OpType.SIntSIntOp ||
      opType == OpType.Float16Int4Op || opType == OpType.Float16Float16Op
  )
  require(
    inputAElemWidth > 0 && inputBElemWidth > 0 && outputCElemWidth > 0,
    "Element widths must be greater than 0"
  )
  if (opType == OpType.UIntUIntOp) {
    io.out_c := io.in_a * io.in_b
  } else if (opType == OpType.SIntSIntOp) {
    io.out_c := (io.in_a.asTypeOf(SInt(outputCElemWidth.W)) * io.in_b.asTypeOf(
      SInt(outputCElemWidth.W)
    )).asUInt
  } else if (opType == OpType.Float16Int4Op) {
    val fpMulInt = Module(
      new FPMULIntBlackBox("fp_mul_int", inputAElemWidth, inputBElemWidth, outputCElemWidth)
    )
    fpMulInt.io.operand_a_i := io.in_a
    fpMulInt.io.operand_b_i := io.in_b
    fpMulInt.io.operand_c_i := 0.U // Assuming no third operand for now
    io.out_c                := fpMulInt.io.result_o
    assert(
      inputAElemWidth == 16 && inputBElemWidth == 4 && outputCElemWidth == 32,
      "For Float16Int4Op, input widths must be 16, 4 and output width must be 32"
    )
  } else if (opType == OpType.Float16Float16Op) {
    val fpMulfp = Module(
      new FPMULFP("fp_mul", inputAElemWidth, inputBElemWidth, outputCElemWidth)
    )
    fpMulfp.io.operand_a_i  := io.in_a
    fpMulfp.io.operand_b_i  := io.in_b
    fpMulfp.io.operand_c_i  := 0.U // Assuming no third operand for now
    io.out_c                := fpMulfp.io.result_o
    assert(
      inputAElemWidth == 16 && inputBElemWidth == 16 && outputCElemWidth == 32,
      "For Float16Float16Op, input widths must be 16, 16 and output width must be 32"
    )
  } else {
    // TODO: add support for other types
    // For now, just set the output to 0
    io.out_c := 0.U
  }
}

object MultiplierEmitterUInt extends App {
  emitVerilog(
    new Multiplier(OpType.UIntUIntOp, 8, 4, 16),
    Array("--target-dir", "generated/SpatialArray")
  )
}

object MultiplierEmitterSInt extends App {
  emitVerilog(
    new Multiplier(OpType.SIntSIntOp, 8, 4, 16),
    Array("--target-dir", "generated/SpatialArray")
  )
}

object MultiplierEmitterFloat16Int4 extends App {
  emitVerilog(
    new Multiplier(OpType.Float16Int4Op, 16, 4, 32),
    Array("--target-dir", "generated/SpatialArray")
  )
}

object MultiplierEmitterFloat16Float16 extends App {
  emitVerilog(
    new Multiplier(OpType.Float16Float16Op, 16, 16, 32),
    Array("--target-dir", "generated/SpatialArray")
  )
}
