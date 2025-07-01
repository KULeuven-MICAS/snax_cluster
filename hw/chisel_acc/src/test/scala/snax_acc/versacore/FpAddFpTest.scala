// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi <xiaoling.yi@kuleuven.be>
// Modified by: Robin Geens <robin.geens@kuleuven.be>

package snax_acc.versacore
import chisel3._

import chiseltest._
import chiseltest.simulator.VerilatorBackendAnnotation
import org.scalatest.flatspec.AnyFlatSpec
import snax_acc.utils.fpUtils

class FpAddFpTest extends AnyFlatSpec with ChiselScalatestTester with fpUtils {
  behavior of "FPAddFP"

  val test_num = 1000

  def test_fp_add_fp(dut: FPAddFP, test_id: Int, A: Float, B: Float) = {

    // Expected result
    val expected_fp = (A, dut.typeA) + (B, dut.typeB)

    // Quantize the float
    val A_uint = floatToUInt(dut.typeA, A)
    val B_uint = floatToUInt(dut.typeB, B)

    dut.io.operand_a_i.poke(A_uint.U)
    dut.io.operand_b_i.poke(B_uint.U)

    dut.clock.step(2)
    val result = dut.io.result_o.peek()

    val errorCount =
      try {
        assert((expected_fp, dut.typeC) === result)
        0 // No error
      } catch {
        case _: Throwable => {
          val A_fp          = quantize(dut.typeA, A)
          val B_fp          = quantize(dut.typeB, B)
          val result_fp     = uintToFloat(dut.typeC, result)
          val expected_uint = floatToUInt(dut.typeC, expected_fp)
          println(f"----Error in test id: $test_id----")
          println(
            f"A_fp: ${A_fp} , B_fp: ${B_fp},  expected_fp: ${expected_fp}"
          )
          println(
            f"(expected) ${expected_uint.toString(2).grouped(4).mkString("_")} (got) ${result.litValue.toString(2).grouped(4).mkString("_")}"
          )
          println(f"(expected) ${expected_fp} (got) ${result_fp}")
          1 // Error occurred
        }
      }

    dut.clock.step(2)
    errorCount
  }

  def test_all_fp_add_fp(dut: FPAddFP) = {
    var error       = 0
    val testCases   =
      Seq.fill(test_num)((genRandomValue(dut.typeA), genRandomValue(dut.typeB)))
    testCases.zipWithIndex.foreach { case ((a, b), index) =>
      error += test_fp_add_fp(dut, index + 1, a, b)
    }
    val successRate = 1 - error.toFloat / test_num
    println(s"Success rate: ${successRate * 100}%")
    assert(successRate >= 0.998)
  }

  it should "perform FP32 + FP32 -> FP32 correctly" in {
    test(
      new FPAddFP(
        topmodule = "fp_add",
        typeA     = FP32,
        typeB     = FP32,
        typeC     = FP32
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>
      test_all_fp_add_fp(dut)
    }
  }

  it should "perform FP16 + FP16 -> FP16 correctly" in {
    test(
      new FPAddFP(
        topmodule = "fp_add",
        typeA     = FP16,
        typeB     = FP16,
        typeC     = FP16
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>
      test_all_fp_add_fp(dut)
    }
  }

  it should "perform FP16 + FP16 -> FP32 correctly" in {
    test(
      new FPAddFP(
        topmodule = "fp_add",
        typeA     = FP16,
        typeB     = FP16,
        typeC     = FP32
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>
      test_all_fp_add_fp(dut)
    }
  }

  it should "perform FP16 + FP32 -> FP32 correctly" in {
    test(
      new FPAddFP(
        topmodule = "fp_add",
        typeA     = FP16,
        typeB     = FP32,
        typeC     = FP32
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>
      test_all_fp_add_fp(dut)
    }
  }

  it should "perform BF16 + BF16 -> FP32 correctly" in {
    test(
      new FPAddFP(
        topmodule = "fp_add",
        typeA     = BF16,
        typeB     = BF16,
        typeC     = FP32
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>
      test_all_fp_add_fp(dut)
    }
  }

  it should "perform BF16 + BF16 -> BF16 correctly" in {
    test(
      new FPAddFP(
        topmodule = "fp_add",
        typeA     = BF16,
        typeB     = BF16,
        typeC     = BF16
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>
      test_all_fp_add_fp(dut)
    }
  }

  it should "perform FP32 + BF16 -> FP32 correctly" in {
    test(
      new FPAddFP(
        topmodule = "fp_add",
        typeA     = FP32,
        typeB     = BF16,
        typeC     = FP32
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>
      test_all_fp_add_fp(dut)
    }
  }
}
