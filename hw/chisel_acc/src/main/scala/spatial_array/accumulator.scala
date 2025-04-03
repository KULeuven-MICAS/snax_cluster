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
    val in1         = Flipped(DecoupledIO(Vec(numElements, UInt(inputElemWidth.W))))
    val in2         = Flipped(DecoupledIO(Vec(numElements, UInt(inputElemWidth.W))))
    val accAddExtIn = Input(Bool())
    val enable      = Input(Bool())
    val accClear    = Input(Bool())
    val out         = DecoupledIO(Vec(numElements, UInt(outputElemWidth.W)))
  })

  require(opType == OpType.UIntUIntOp || opType == OpType.SIntSIntOp)
  require(
    inputElemWidth > 0 && outputElemWidth > 0 && numElements > 0,
    "Element widths and number of elements must be greater than 0"
  )

  val accumulatorReg = RegInit(
    VecInit(Seq.fill(numElements)(0.U(outputElemWidth.W)))
  )

  // Create the adders once outside the when block
  val adders = Seq.fill(numElements)(
    Module(
      new Adder(opType, inputElemWidth, inputElemWidth, outputElemWidth)
    ).io
  )

  // if not enabled, data gated
  when(io.accClear) {
    accumulatorReg := VecInit(Seq.fill(numElements)(0.U(outputElemWidth.W)))
    adders.zipWithIndex.foreach { case (adder, i) =>
      adder.in_a := io.in1.bits(i)
      adder.in_b := io.in2.bits(i)
    }
  }
    .elsewhen(io.enable) {
      when(io.in1.fire && io.in2.fire && io.accAddExtIn) {
        accumulatorReg := adders.zip(io.in1.bits.zip(io.in2.bits)).map { case (adder, (a, b)) =>
          adder.in_a := a
          adder.in_b := b
          adder.out_c
        }
      }.elsewhen(io.in1.fire && !io.accAddExtIn) {
        accumulatorReg := adders.zip(io.in1.bits.zip(accumulatorReg)).map { case (adder, (a, acc)) =>
          adder.in_a := a
          adder.in_b := acc
          adder.out_c
        }
      }.otherwise {
        // If not firing, keep the accumulator value
        accumulatorReg := accumulatorReg
        // adder inputs are not used, so set them to 0
        adders.zipWithIndex.foreach { case (adder, i) =>
          adder.in_a := io.in1.bits(i)
          adder.in_b := io.in2.bits(i)
        }
      }
    }
    .otherwise {
      // If not firing, keep the accumulator value
      accumulatorReg := accumulatorReg
      // adder inputs are not used, so set them to 0
      adders.zipWithIndex.foreach { case (adder, i) =>
        adder.in_a := io.in1.bits(i)
        adder.in_b := io.in2.bits(i)
      }
    }

  val inputDataFire  = RegNext(io.in1.fire && io.in2.fire)
  val keepOutput     = RegInit(false.B)
  val keepOutputNext = io.out.valid && !io.out.ready
  keepOutput := keepOutputNext

  // TODO: can the keepOutputNext be ommited?
  io.in1.ready := (!keepOutput) && (!keepOutputNext)
  io.in2.ready := (!keepOutput) && (!keepOutputNext)

  io.out.bits  := accumulatorReg
  io.out.valid := inputDataFire || keepOutput
}

object AccumulatorEmitterUInt extends App {
  emitVerilog(
    new Accumulator(OpType.UIntUIntOp, 8, 16, 8),
    Array("--target-dir", "generated/SpatialArray")
  )
}
