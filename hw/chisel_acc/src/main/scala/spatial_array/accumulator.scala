package snax_acc.spatial_array

import chisel3._
import chisel3.util._

class AccumulatorBlock(
  val opType:          Int,
  val inputElemWidth:  Int,
  val outputElemWidth: Int
) extends Module
    with RequireAsyncReset {
  val io = IO(new Bundle {
    val in1         = Input(UInt(inputElemWidth.W))
    val in2         = Input(UInt(inputElemWidth.W))
    val accAddExtIn = Input(Bool())
    val enable      = Input(Bool())
    val accClear    = Input(Bool())
    val out         = Output(UInt(outputElemWidth.W))
  })

  require(opType == OpType.UIntUIntOp || opType == OpType.SIntSIntOp)
  require(
    inputElemWidth > 0 && outputElemWidth > 0,
    "Element widths must be greater than 0"
  )

  val accumulatorReg = RegInit(0.U(outputElemWidth.W))
  val adder          = Module(
    new Adder(opType, inputElemWidth, inputElemWidth, outputElemWidth)
  ).io

  adder.in_a := io.in1
  adder.in_b := Mux(io.accAddExtIn, io.in2, accumulatorReg)

  val nextAcc = Wire(UInt(outputElemWidth.W))
  nextAcc := Mux(io.accClear, 0.U, Mux(io.enable, adder.out_c, accumulatorReg))

  val accUpdate = io.enable || io.accClear
  accumulatorReg := Mux(accUpdate, nextAcc, accumulatorReg)

  io.out := accumulatorReg
}

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

  val accumulater_blocks = Seq.fill(numElements) {
    Module(new AccumulatorBlock(opType, inputElemWidth, outputElemWidth))
  }

  val accUpdate = (io.in1.fire && io.enable && (!io.accAddExtIn || (io.in2.fire && io.accAddExtIn)))

  for (i <- 0 until numElements) {
    accumulater_blocks(i).io.in1         := io.in1.bits(i)
    accumulater_blocks(i).io.in2         := io.in2.bits(i)
    accumulater_blocks(i).io.accAddExtIn := io.accAddExtIn
    accumulater_blocks(i).io.enable      := accUpdate
    accumulater_blocks(i).io.accClear    := io.accClear
  }

  val inputDataFire  = RegNext(accUpdate)
  val keepOutput     = RegInit(false.B)
  val keepOutputNext = io.out.valid && !io.out.ready
  keepOutput := keepOutputNext

  // TODO: can the keepOutputNext be ommited?
  io.in1.ready := (!keepOutput) && (!keepOutputNext)
  io.in2.ready := (!keepOutput) && (!keepOutputNext)

  io.out.bits  := accumulater_blocks.map(_.io.out)
  io.out.valid := inputDataFire || keepOutput
}

object AccumulatorEmitterUInt extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Accumulator(OpType.UIntUIntOp, 8, 16, 4096),
    Array("--target-dir", "generated/SpatialArray"),
    Array(
      "--split-verilog",
      s"-o=generated/SpatialArray/Accumulator"
    )
  )
}
