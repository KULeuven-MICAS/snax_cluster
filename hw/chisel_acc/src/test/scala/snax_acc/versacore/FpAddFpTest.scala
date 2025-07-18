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

  def testSingle(dut: FPAddFP, test_id: Int, A: Float, B: Float) = {

    val expected_fp = (A, dut.typeA) + (B, dut.typeB)

    // Quantize the inputs
    val A_uint = floatToUInt(dut.typeA, A)
    val B_uint = floatToUInt(dut.typeB, B)

    dut.io.operand_a_i.poke(A_uint.U)
    dut.io.operand_b_i.poke(B_uint.U)
    dut.clock.step(1)
    val result = dut.io.result_o.peek()

    try {
      assert((expected_fp, dut.typeC) === result)
    } catch {
      case e: Throwable => {
        val A_fp          = quantize(dut.typeA, A)
        val B_fp          = quantize(dut.typeB, B)
        val result_fp     = uintToFloat(dut.typeC, result)
        val expected_uint = floatToUInt(dut.typeC, expected_fp)
        println(f"----Error in test id: $test_id----")
        println(
          f"A_fp: ${A} -> ${A_fp} , B_fp: ${B} -> ${B_fp},  expected_fp: ${expected_fp} -> ${quantize(dut.typeC, expected_fp)}"
        )
        println(
          f"(expected) ${uintToStr(expected_uint, dut.typeC)} (got) ${uintToStr(result.litValue, dut.typeC)}"
        )
        println(f"(expected) ${quantize(dut.typeC, expected_fp)} (got) ${result_fp}")
        throw e
      }
    }

  }

  def test_all_fp_add_fp(dut: FPAddFP) = {
    val testCases = Seq.fill(test_num)((genRandomValue(dut.typeA), genRandomValue(dut.typeB)))
    testCases.zipWithIndex.foreach { case ((a, b), index) => testSingle(dut, index + 1, a, b) }
  }

  def testSpecialCases(dut: FPAddFP) = {
    val subnormal = uintToFloat(FP32, BigInt(1 << 22))

    val specialCases = Seq(
      (0.0f, 0.0f),
      (0.0f, 1.0f),
      (1.0f, 0.0f),                                     // Zero cases
      (subnormal, 0.0f),
      (0.0f, subnormal),
      (subnormal, subnormal),
      (Float.MinPositiveValue, Float.MinPositiveValue), // Underflow case
      (0.0f, Float.MinPositiveValue),                   // Small number cases
      //
      (Float.NaN, 1.0f),
      (1.0f, Float.NaN),
      (Float.NaN, Float.NaN),                           // NaN cases
      (Float.PositiveInfinity, 1.0f),
      (1.0f, Float.PositiveInfinity),                   // +inf cases
      (Float.NegativeInfinity, 1.0f),
      (1.0f, Float.NegativeInfinity),                   // -inf cases
      (Float.PositiveInfinity, Float.NegativeInfinity), // inf + -inf = NaN
      (Float.NegativeInfinity, Float.PositiveInfinity), // -inf + inf = NaN

      // TODO These does currently NOT work
      // (Float.MinPositiveValue, Float.MinPositiveValue), // Underflow case
      (Float.MinPositiveValue, 0.0f)
      // (0.0f, Float.MinPositiveValue)                    // Small number cases
    )

    specialCases.zipWithIndex.foreach { case ((a, b), index) => testSingle(dut, index + 1, a, b) }
  }

  it should "perform FP32 + FP32 -> FP32 correctly" in {
    test(
      new FPAddFP(topmodule = "fp_add", typeA = FP32, typeB = FP32, typeC = FP32)
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut => test_all_fp_add_fp(dut) }
  }

  it should "perform FP16 + FP16 -> FP16 correctly" in {
    test(
      new FPAddFP(topmodule = "fp_add", typeA = FP16, typeB = FP16, typeC = FP16)
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut => test_all_fp_add_fp(dut) }
  }

  it should "perform FP16 + FP16 -> FP32 correctly" in {
    test(
      new FPAddFP(topmodule = "fp_add", typeA = FP16, typeB = FP16, typeC = FP32)
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut => test_all_fp_add_fp(dut) }
  }

  it should "perform FP16 + FP32 -> FP32 correctly" in {
    test(
      new FPAddFP(topmodule = "fp_add", typeA = FP16, typeB = FP32, typeC = FP32)
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut => test_all_fp_add_fp(dut) }
  }

  it should "perform FP16 + FP32 -> FP16 correctly" in {
    test(
      new FPAddFP(topmodule = "fp_add", typeA = FP16, typeB = FP32, typeC = FP16)
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut => test_all_fp_add_fp(dut) }
  }

  it should "perform BF16 + BF16 -> FP32 correctly" in {
    test(
      new FPAddFP(topmodule = "fp_add", typeA = BF16, typeB = BF16, typeC = FP32)
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut => test_all_fp_add_fp(dut) }
  }

  it should "perform BF16 + BF16 -> BF16 correctly" in {
    test(
      new FPAddFP(topmodule = "fp_add", typeA = BF16, typeB = BF16, typeC = BF16)
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut => test_all_fp_add_fp(dut) }
  }

  it should "perform FP32 + BF16 -> FP32 correctly" in {
    test(
      new FPAddFP(topmodule = "fp_add", typeA = FP32, typeB = BF16, typeC = FP32)
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut => test_all_fp_add_fp(dut) }
  }

  it should "handle special cases in FP16 + FP32 -> FP16" in {
    test(
      new FPAddFP(topmodule = "fp_add", typeA = FP16, typeB = FP32, typeC = FP16)
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut => testSpecialCases(dut) }
  }

  it should "handle special cases in FP16 + FP32 -> FP32" in {
    test(
      new FPAddFP(topmodule = "fp_add", typeA = FP16, typeB = FP32, typeC = FP32)
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut => testSpecialCases(dut) }
  }

  it should "handle special cases in FP32 + FP16 -> BF16" in {
    test(
      new FPAddFP(topmodule = "fp_add", typeA = FP32, typeB = FP16, typeC = BF16)
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut => testSpecialCases(dut) }
  }
}
