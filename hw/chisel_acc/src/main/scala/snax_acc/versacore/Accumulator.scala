// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi (xiaoling.yi@kuleuven.be)

package snax_acc.versacore

import chisel3._
import chisel3.util._

/** AccumulatorBlock is a single accumulator block that performs accumulation on two input values.
  * @param opType
  * @param inputElemWidth
  * @param outputElemWidth
  */
class AccumulatorBlock(
  val opType:          Int,
  val inputElemWidth:  Int,
  val outputElemWidth: Int
) extends Module
    with RequireAsyncReset {
  val io = IO(new Bundle {
    // two inputs for accumulation
    val in1         = Input(UInt(inputElemWidth.W))
    val in2         = Input(UInt(inputElemWidth.W))
    // whether to add the external input to the accumulator or accumulate the internal reg value
    val accAddExtIn = Input(Bool())
    // enable signal
    val enable      = Input(Bool())
    // clear signal to reset the accumulator
    val accClear    = Input(Bool())
    // output of the accumulator
    val out         = Output(UInt(outputElemWidth.W))
  })

  require(
    opType == OpType.UIntUIntOp || opType == OpType.SIntSIntOp ||
      opType == OpType.Float16IntOp || opType == OpType.Float16Float16Op,
    "Unsupported operation type for AccumulatorBlock"
  )
  require(
    inputElemWidth > 0 && outputElemWidth > 0,
    "Element widths must be greater than 0"
  )

  // Internal register to hold the accumulated value
  val accumulatorReg = RegInit(0.U(outputElemWidth.W))
  // Adder module to perform the accumulation
  val adder          = Module(
    new Adder(opType, inputElemWidth, inputElemWidth, outputElemWidth)
  ).io

  // connection description
  adder.in_a := io.in1
  adder.in_b := Mux(io.accAddExtIn, io.in2, accumulatorReg)

  val nextAcc = Wire(UInt(outputElemWidth.W))
  nextAcc := Mux(io.accClear, 0.U, Mux(io.enable, adder.out_c, accumulatorReg))

  val accUpdate = io.enable || io.accClear
  accumulatorReg := Mux(accUpdate, nextAcc, accumulatorReg)

  // output of accumulator register
  io.out := accumulatorReg
}

/** Accumulator is a module that contains multiple AccumulatorBlock instances. It manages the accumulation of multiple
  * elements and provides a ready/valid interface.
  * @param opType
  * @param inputElemWidth
  * @param outputElemWidth
  * @param numElements
  */
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

  require(
    opType == OpType.UIntUIntOp || opType == OpType.SIntSIntOp ||
      opType == OpType.Float16IntOp || opType == OpType.Float16Float16Op,
    "Unsupported operation type for Accumulator"
  )
  require(
    inputElemWidth > 0 && outputElemWidth > 0 && numElements > 0,
    "Element widths and number of elements must be greater than 0"
  )

  // Create an array of AccumulatorBlock instances
  // Each block will handle one element of the input vectors
  // and produce one element of the output vector
  val accumulater_blocks = Seq.fill(numElements) {
    Module(new AccumulatorBlock(opType, inputElemWidth, outputElemWidth))
  }

  // accumulation update logic, considering the handshake
  val accUpdate = (io.in1.fire && io.enable && (!io.accAddExtIn || (io.in2.fire && io.accAddExtIn)))

  // Connect the inputs of each AccumulatorBlock
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

  // handshake
  io.in1.ready := (!keepOutput) && (!keepOutputNext)
  io.in2.ready := (!keepOutput) && (!keepOutputNext)

  // Connect the outputs of each AccumulatorBlock to the output interface
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
