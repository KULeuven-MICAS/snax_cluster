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
  require(opType == OpType.UIntUIntOp || opType == OpType.SIntSIntOp)
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
