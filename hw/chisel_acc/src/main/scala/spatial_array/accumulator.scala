package snax_acc.spatial_array

import chisel3._
import chisel3.util._

// accumulator
class Accumulator(
  val opType:          Int,
  val inputElemWidth:  Int,
  val outputElemWidth: Int,
  val numElements:     Int
) extends Module
    with RequireAsyncReset {
  val io = IO(new Bundle {
    val in1        = Flipped(DecoupledIO(Vec(numElements, UInt(inputElemWidth.W))))
    val in2        = Flipped(DecoupledIO(Vec(numElements, UInt(inputElemWidth.W))))
    val add_ext_in = Input(Bool())
    val out        = Output(Vec(numElements, UInt(outputElemWidth.W)))
  })

  require(opType == OpType.UIntUIntOp || opType == OpType.SIntSIntOp)
  require(
    inputElemWidth > 0 && outputElemWidth > 0 && numElements > 0,
    "Element widths and number of elements must be greater than 0"
  )

  val accumulator_reg = RegInit(
    VecInit(Seq.fill(numElements)(0.U(outputElemWidth.W)))
  )

  // Create the adders once outside the when block
  val adders = Seq.fill(numElements)(
    Module(
      new Adder(opType, inputElemWidth, inputElemWidth, outputElemWidth)
    ).io
  )

  when(io.in1.fire && io.in2.fire) {
    when(io.add_ext_in) {
      accumulator_reg := adders.zip(io.in1.bits.zip(io.in2.bits)).map { case (adder, (a, b)) =>
        adder.in_a := a
        adder.in_b := b
        adder.out_c
      }
    }.otherwise {
      accumulator_reg := adders.zip(io.in1.bits.zip(accumulator_reg)).map { case (adder, (a, acc)) =>
        adder.in_a := a
        adder.in_b := acc
        adder.out_c
      }
    }
  }.otherwise {
    // If not firing, keep the accumulator value
    accumulator_reg := accumulator_reg
    // adder inputs are not used, so set them to 0
    adders.zipWithIndex.foreach { case (adder, i) =>
      adder.in_a := io.in1.bits(i)
      adder.in_b := io.in2.bits(i)
    }
  }

  io.in1.ready := true.B
  io.in2.ready := true.B
  io.out       := accumulator_reg
}

object AccumulatorEmitterUInt extends App {
  emitVerilog(
    new Accumulator(OpType.UIntUIntOp, 8, 16, 8),
    Array("--target-dir", "generated/SpatialArray")
  )
}
