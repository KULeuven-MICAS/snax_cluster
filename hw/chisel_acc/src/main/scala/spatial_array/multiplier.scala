package snax_acc.spatial_array

import chisel3._

class MultiplierIO(
    inputAElemWidth: Int,
    inputBElemWidth: Int,
    mulElemWidth: Int
) extends Bundle {
  val in_a = Input(UInt(inputAElemWidth.W))
  val in_b = Input(UInt(inputBElemWidth.W))
  val out_c = Output(UInt(mulElemWidth.W))
}

class Multiplier(
    opType: Int,
    inputAElemWidth: Int,
    inputBElemWidth: Int,
    mulElemWidth: Int
) extends Module
    with RequireAsyncReset {
  val io = IO(new MultiplierIO(inputAElemWidth, inputBElemWidth, mulElemWidth))
  require(opType == OpType.UIntUIntOp || opType == OpType.SIntSIntOp)
  require(
    inputAElemWidth > 0 && inputBElemWidth > 0 && mulElemWidth > 0,
    "Element widths must be greater than 0"
  )
  if (opType == OpType.UIntUIntOp) {
    io.out_c := io.in_a * io.in_b
  } else if (opType == OpType.SIntSIntOp) {
    io.out_c := (io.in_a.asTypeOf(SInt(inputAElemWidth.W)) * io.in_b.asTypeOf(
      SInt(inputBElemWidth.W)
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
