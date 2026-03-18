// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi <xiaoling.yi@kuleuven.be>
// Modified by: Robin Geens <robin.geens@kuleuven.be>

package snax_acc.versacore

import chisel3._

import chisel3.util._

import fp_unit._

class AdderReq(inputTypeA: DataType, inputTypeB: DataType) extends Bundle {
  val in_a = UInt(inputTypeA.width.W)
  val in_b = UInt(inputTypeB.width.W)
}

/** AdderIO defines the input and output interfaces for the Adder module. */
class AdderIO(
  inputTypeA: DataType,
  inputTypeB: DataType,
  inputTypeC: DataType
) extends Bundle {
  val in  = Flipped(Decoupled(new AdderReq(inputTypeA, inputTypeB)))
  val out = Decoupled(UInt(inputTypeC.width.W))
}

/** Adder is a module that performs addition on two inputs based on the specified operation type. */
class Adder(
  inputTypeA: DataType,
  inputTypeB: DataType,
  inputTypeC: DataType
) extends Module
    with RequireAsyncReset {

  val io = IO(new AdderIO(inputTypeA, inputTypeB, inputTypeC))

  // Combinational handshake
  io.in.ready  := io.out.ready
  io.out.valid := io.in.valid

  val out_c = Wire(UInt(inputTypeC.width.W))
  io.out.bits := out_c

  (inputTypeA, inputTypeB, inputTypeC) match {
    case (_: IntType, _: IntType, _: IntType) =>
      out_c := (io.in.bits.in_a.asTypeOf(SInt(inputTypeC.width.W)) + io.in.bits.in_b.asTypeOf(
        SInt(inputTypeC.width.W)
      )).asUInt

    case (_: FpType, _: IntType, _: FpType) => throw new NotImplementedError()

    case (a: FpType, b: FpType, c: FpType) => {
      val fpAddFp = Module(new FpAddFpBlackBox("fp_add", a, b, c))
      fpAddFp.io.operand_a_i := io.in.bits.in_a
      fpAddFp.io.operand_b_i := io.in.bits.in_b
      out_c                  := fpAddFp.io.result_o
    }

    case (_, _, _) => throw new NotImplementedError()

  }

}

// Below are the emitters for different adder configurations for testing and evaluation purposes.
object AdderEmitterUInt extends App {
  emitVerilog(
    new Adder(Int8, Int4, Int16),
    Array("--target-dir", "generated/versacore")
  )
}

object AdderEmitterSInt extends App {
  emitVerilog(
    new Adder(Int8, Int4, Int16),
    Array("--target-dir", "generated/versacore/adder")
  )
}

object AdderEmitterFloat16Float16 extends App {
  emitVerilog(
    new Adder(FP32, FP32, FP32),
    Array("--target-dir", "generated/versacore/adder")
  )
}
