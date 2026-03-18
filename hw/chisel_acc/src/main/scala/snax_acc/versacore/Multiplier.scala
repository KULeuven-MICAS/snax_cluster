// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi <xiaoling.yi@kuleuven.be>
// Modified by: Robin Geens <robin.geens@kuleuven.be>

package snax_acc.versacore

import chisel3._

import chisel3.util._

import fp_unit._

class MultiplierReq(inputTypeA: DataType, inputTypeB: DataType) extends Bundle {
  val in_a = UInt(inputTypeA.width.W)
  val in_b = UInt(inputTypeB.width.W)
}

class MultiplierIO(inputTypeA: DataType, inputTypeB: DataType, inputTypeC: DataType) extends Bundle {
  val in  = Flipped(Decoupled(new MultiplierReq(inputTypeA, inputTypeB)))
  val out = Decoupled(UInt(inputTypeC.width.W))
}

/** Multiplier module that supports different operation types */
class Multiplier(inputTypeA: DataType, inputTypeB: DataType, inputTypeC: DataType)
    extends Module
    with RequireAsyncReset {

  val io = IO(new MultiplierIO(inputTypeA, inputTypeB, inputTypeC))

  // Combinational handshake
  io.in.ready  := io.out.ready
  io.out.valid := io.in.valid

  val out_c = Wire(UInt(inputTypeC.width.W))
  io.out.bits := out_c

  (inputTypeA, inputTypeB, inputTypeC) match {

    case (_: IntType, _: IntType, _: IntType) =>
      out_c := (io.in.bits.in_a.asTypeOf(SInt(inputTypeC.width.W)) * io.in.bits.in_b.asTypeOf(
        SInt(inputTypeC.width.W)
      )).asUInt

    case (a: FpType, b: IntType, c: FpType) => {
      val fpMulInt = Module(new FpMulIntBlackBox("fp_mul_int", a, b, c))
      fpMulInt.io.operand_a_i := io.in.bits.in_a
      fpMulInt.io.operand_b_i := io.in.bits.in_b
      out_c                   := fpMulInt.io.result_o
    }

    case (a: FpType, b: FpType, c: FpType) => {
      val fpMulfp = Module(new FpMulFp(a, b, c))
      fpMulfp.io.in_a := io.in.bits.in_a
      fpMulfp.io.in_b := io.in.bits.in_b
      out_c           := fpMulfp.io.out
    }

    case (_, _, _) => throw new NotImplementedError()

  }

}

object MultiplierEmitterUInt extends App {
  emitVerilog(
    new Multiplier(Int8, Int4, Int16),
    Array("--target-dir", "generated/versacore")
  )
}

object MultiplierEmitterSInt extends App {
  emitVerilog(
    new Multiplier(Int8, Int4, Int16),
    Array("--target-dir", "generated/versacore")
  )
}

object MultiplierEmitterFloat16Int4 extends App {
  emitVerilog(
    new Multiplier(FP16, Int4, FP32),
    Array("--target-dir", "generated/versacore")
  )
}

object MultiplierEmitterFloat16Float16 extends App {
  emitVerilog(
    new Multiplier(FP16, FP16, FP32),
    Array("--target-dir", "generated/versacore")
  )
}

object MultiplierEmitterFloat32Float32 extends App {
  emitVerilog(
    new Multiplier(FP32, FP32, FP32),
    Array("--target-dir", "generated/versacore")
  )
}
