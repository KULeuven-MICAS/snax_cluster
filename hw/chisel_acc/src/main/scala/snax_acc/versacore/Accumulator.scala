// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi <xiaoling.yi@kuleuven.be>

package snax_acc.versacore

import chisel3._
import chisel3.util._

import fp_unit.DataType
import fp_unit.Int16
import fp_unit.Int8

/** AccumulatorBlock is a single accumulator block that performs accumulation on two input values. */
class AccumulatorBlock(
  val inputType:  DataType,
  val outputType: DataType
) extends Module
    with RequireAsyncReset {
  val io = IO(new Bundle {
    // two inputs for accumulation
    val in1         = Input(UInt(inputType.width.W))
    val in2         = Input(UInt(inputType.width.W))
    // whether to add the external input to the accumulator or accumulate the internal reg value
    val accAddExtIn = Input(Bool())
    // enable signal
    val enable      = Input(Bool())
    // output of the accumulator
    val out         = Output(UInt(outputType.width.W))
  })

  // Internal register to hold the accumulated value
  val accumulatorReg = RegInit(0.U(outputType.width.W))
  // Adder module to perform the accumulation
  val adder          = Module(
    new Adder(inputType, inputType, outputType)
  ).io

  // connection description
  adder.in.bits.in_a := io.in1
  adder.in.bits.in_b := Mux(io.accAddExtIn, io.in2, accumulatorReg)
  adder.in.valid     := io.enable
  // the register will always accept the adder's output, so the ready signal is always true
  adder.out.ready    := true.B

  // update accumulator register based on the adder's output handshake
  // This is robust to pipeline depth within the adder
  when(adder.out.fire) {
    accumulatorReg := adder.out.bits
  }

  // output of the accumulator register
  io.out := accumulatorReg
}

/** Accumulator is a module that contains multiple AccumulatorBlock instances. It manages the accumulation of multiple
  * elements and provides a ready/valid interface.
  */
class Accumulator(
  val inputType:   DataType,
  val outputType:  DataType,
  val numElements: Int
) extends Module
    with RequireAsyncReset {
  val io = IO(new Bundle {
    val in1         = Flipped(DecoupledIO(Vec(numElements, UInt(inputType.width.W))))
    val in2         = Flipped(DecoupledIO(Vec(numElements, UInt(inputType.width.W))))
    val accAddExtIn = Input(Bool())
    val enable      = Input(Vec(numElements, Bool()))
    val out         = DecoupledIO(Vec(numElements, UInt(outputType.width.W)))
    val inputReady  = Output(Bool())
    val accUpdate   = Output(Bool())
  })

  // Create an array of AccumulatorBlock instances
  // Each block will handle one element of the input vectors
  // and produce one element of the output vector
  val accumulator_blocks = Seq.fill(numElements) {
    Module(new AccumulatorBlock(inputType, outputType))
  }

  // Join the dot product with C only when starting an accumulation. Neither
  // operand may be consumed on its own, and unused C must remain upstream.
  val outputValid = RegInit(false.B)
  val canAccept   = !outputValid || io.out.ready
  val enabled     = io.enable.asUInt.orR
  io.in1.ready  := canAccept && enabled && (!io.accAddExtIn || io.in2.valid)
  io.in2.ready  := canAccept && enabled && io.accAddExtIn && io.in1.valid
  io.inputReady := io.in1.ready

  // Every accepted dot product updates the enabled accumulator lanes.
  val accUpdate = VecInit(
    (0 until numElements).map(i => io.in1.fire && io.enable(i))
  )
  io.accUpdate := accUpdate.reduce(_ || _)

  // Connect the inputs of each AccumulatorBlock
  for (i <- 0 until numElements) {
    accumulator_blocks(i).io.in1         := io.in1.bits(i)
    accumulator_blocks(i).io.in2         := io.in2.bits(i)
    accumulator_blocks(i).io.accAddExtIn := io.accAddExtIn
    accumulator_blocks(i).io.enable      := accUpdate(i)
  }

  // Hold both data and valid while stalled. An output can be consumed and
  // replaced by the next accumulation on the same edge.
  when(canAccept) {
    outputValid := io.in1.fire
  }

  // Connect the outputs of each AccumulatorBlock to the output interface
  io.out.bits  := VecInit(accumulator_blocks.map(_.io.out))
  io.out.valid := outputValid
}

object AccumulatorEmitterUInt extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Accumulator(Int8, Int16, 4096),
    Array("--target-dir", "generated/versacore"),
    Array(
      "--split-verilog",
      s"-o=generated/versacore/Accumulator"
    )
  )
}
