package snax_acc.spatial_array

import scala.util.Random

import chisel3._

import chiseltest._
import chiseltest.simulator.VerilatorBackendAnnotation
import org.scalatest.flatspec.AnyFlatSpec

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

        // try {
        //   assert(result == expected_o)
        // } catch {
        //   case _: java.lang.AssertionError => {
        //     println(f"----Error!!!!------- Assertion failed on test $test_id")
        //   }
        // }
          assert(result == expected_o)

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
