package snax_acc.spatialarray

import chisel3._
import chisel3.util._

// accumulator
class Accumulator[T <: Data with Num[T]](
    val inputType: T,
    val outputType: T,
    val numElements: Int
) extends Module {
  val io = IO(new Bundle {
    val in1 = Flipped(DecoupledIO(Vec(numElements, inputType)))
    val in2 = Flipped(DecoupledIO(Vec(numElements, inputType)))
    val add_ext_in = Input(Bool())
    val out = Output(Vec(numElements, outputType))
  })

  val accumulator_reg = RegInit(
    VecInit(Seq.fill(numElements)(0.U.asTypeOf(outputType)))
  )

  when(io.in1.fire && io.in2.fire && io.add_ext_in) {
    accumulator_reg := io.in1.bits.zip(io.in2.bits).map { case (a, b) => a + b }
  }.elsewhen(io.in1.fire && io.in2.fire && !io.add_ext_in) {
    accumulator_reg := io.in1.bits.zipWithIndex.map { case (a, i) =>
      a + accumulator_reg(i)
    }
  }

  io.in1.ready := true.B
  io.in2.ready := true.B
  io.out := accumulator_reg
}

object AccumulatorEmitterUInt extends App {
  emitVerilog(
    new Accumulator(UInt(8.W), UInt(16.W), 8),
    Array("--target-dir", "generated/SpatialArray")
  )
}
