package snax_acc.spatial_array

import chisel3._

class MultiplierIO[T <: Data with Num[T]](
    inputAType: T,
    inputBType: T,
    mulType: T
) extends Bundle {
  val in_a = Input(inputAType)
  val in_b = Input(inputBType)
  val out_c = Output(mulType)
}

class Multiplier[T <: Data with Num[T]](
    inputAType: T,
    inputBType: T,
    mulType: T
) extends Module {
  val io = IO(new MultiplierIO(inputAType, inputBType, mulType))
  io.out_c := io.in_a * io.in_b
}

object MultiplierEmitterUInt extends App {
  emitVerilog(
    new Multiplier(UInt(8.W), UInt(4.W), UInt(16.W)),
    Array("--target-dir", "generated/SpatialArray")
  )
}

object MultiplierEmitterSInt extends App {
  emitVerilog(
    new Multiplier(SInt(8.W), SInt(4.W), SInt(16.W)),
    Array("--target-dir", "generated/SpatialArray")
  )
}
