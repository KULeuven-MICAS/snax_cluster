package snax_acc.spatial_array

import scala.util.Random

import chisel3._

import chiseltest._
import chiseltest.simulator.VerilatorBackendAnnotation
import org.scalatest.flatspec.AnyFlatSpec

// fake test as the system verilog module is for fp32-fp32 addition
class FP16AddFP16Test extends AnyFlatSpec with ChiselScalatestTester with fpUtils {
  behavior of "FPAddFP"

  it should "perform fp16 add fp16 correctly" in {
    test(
      new FPAddFP(
        topmodule = "fp_add",
        widthA    = 16,
        widthB    = 16,
        widthC    = 32
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>

      def test_fp_add_fp(test_id: Int, A: Float, B: Float, C: Float) = {
        val A_fp16 = uintToFloat(
          fp16.expWidth,
          fp16.sigWidth,
          floatToUInt(fp16.expWidth, fp16.sigWidth, A)
        ) // Convert to 16-bit representation
        val B_fp16 = uintToFloat(
          fp16.expWidth,
          fp16.sigWidth,
          floatToUInt(fp16.expWidth, fp16.sigWidth, B)
        ) // Convert to 16-bit representation
        uintToFloat(
          fp32.expWidth,
          fp32.sigWidth,
          floatToUInt(fp32.expWidth, fp32.sigWidth, C)
        ) // Convert to 32-bit representation
        val gold_O = A_fp16 + B_fp16 // Expected result

        val stimulus_a_i = floatToUInt(fp16.expWidth, fp16.sigWidth, A) // Convert to 16-bit representation
        val stimulus_b_i = floatToUInt(fp16.expWidth, fp16.sigWidth, B) // Direct 4-bit integer
        floatToUInt(fp32.expWidth, fp32.sigWidth, C) // Convert to 32-bit representation
        val expected_o = floatToUInt(fp32.expWidth, fp32.sigWidth, gold_O) // Expected output as UInt

        dut.io.operand_a_i.poke(stimulus_a_i.U)
        dut.io.operand_b_i.poke(stimulus_b_i.U)

        dut.clock.step()
        dut.clock.step()

        val reseult = dut.io.result_o.peek().litValue
        println(
          f"-----------${expected_o.toFloat}, ${reseult.toFloat}-----------"
        )
        println(
          f"-----------Test id: $test_id Expected: 0x${expected_o.toString(16)}, Got: 0x${reseult.toString(16)}-----------"
        )
        try {
          assert(reseult == expected_o)
        } catch {
          case _: java.lang.AssertionError => {
            println(
              f"-----------Test id: $test_id Expected: 0x${expected_o.toString(16)}, Got: 0x${reseult.toString(16)}-----------"
            )
            println(
              f"-----------${expected_o}, ${reseult}-----------"
            )
          }
        }

        dut.clock.step()
        dut.clock.step()
      }

      var A = -1f  // float16
      var B = 4.5f // float16
      var C = 0.0f // float32

      test_fp_add_fp(1, A, B, C)

      new Random()
      // Generate 10 random test cases
      val test_num  = 10
      val testCases = Seq.fill(test_num)(
        (
          Random.nextFloat() * 20 - 10, // Random float between -10 and 10
          Random.nextFloat() * 20 - 10,
          Random.nextFloat() * 20 - 10
        )
      )

      testCases.zipWithIndex.foreach { case ((a, b, c), index) =>
        test_fp_add_fp(index + 1, a, b, c)
      }

    }
  }
}

class FP32AddFP32Test extends AnyFlatSpec with ChiselScalatestTester with fpUtils {
  behavior of "FPAddFP"

  it should "perform fp32 add fp32 correctly" in {
    test(
      new FPAddFP(
        topmodule = "fp_add",
        widthA    = 32, // Changed to 32-bit
        widthB    = 32, // Changed to 32-bit
        widthC    = 32  // Output remains 32-bit
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>

      def test_fp_add_fp(test_id: Int, A: Float, B: Float, C: Float) = {
        val A_fp32 = uintToFloat(
          fp32.expWidth,
          fp32.sigWidth,
          floatToUInt(fp32.expWidth, fp32.sigWidth, A)
        )
        val B_fp32 = uintToFloat(
          fp32.expWidth,
          fp32.sigWidth,
          floatToUInt(fp32.expWidth, fp32.sigWidth, B)
        )
        uintToFloat(
          fp32.expWidth,
          fp32.sigWidth,
          floatToUInt(fp32.expWidth, fp32.sigWidth, C)
        )
        val gold_O = A_fp32 + B_fp32 // Expected result

        val stimulus_a_i = floatToUInt(fp32.expWidth, fp32.sigWidth, A)
        val stimulus_b_i = floatToUInt(fp32.expWidth, fp32.sigWidth, B)
        floatToUInt(fp32.expWidth, fp32.sigWidth, C)
        val expected_o   = floatToUInt(fp32.expWidth, fp32.sigWidth, gold_O)

        dut.io.operand_a_i.poke(stimulus_a_i.U)
        dut.io.operand_b_i.poke(stimulus_b_i.U)

        dut.clock.step()
        dut.clock.step()

        val result = dut.io.result_o.peek().litValue
        println(f"-----------Test id: $test_id-----------")
        println(f"Expected: 0x${expected_o.toString(16)}, Got: 0x${result.toString(16)}")
        println(f"Float expected: ${gold_O}, Result float: ${uintToFloat(fp32.expWidth, fp32.sigWidth, result)}")

        try {
          assert(result == expected_o)
        } catch {
          case _: java.lang.AssertionError => {
            println(f"Assertion failed on test $test_id")
          }
        }

        dut.clock.step()
        dut.clock.step()
      }

      var A = -1f
      var B = 4.5f
      var C = 0.0f

      test_fp_add_fp(1, A, B, C)

      // Generate random test cases
      val test_num  = 10
      val testCases = Seq.fill(test_num)(
        (
          Random.nextFloat() * 1000 - 10,
          Random.nextFloat() * 1000 - 10,
          Random.nextFloat() * 1000 - 10
        )
      )

      testCases.zipWithIndex.foreach { case ((a, b, c), index) =>
        test_fp_add_fp(index + 2, a, b, c)
      }

    }
  }
}
