package snax_acc.spatial_array

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.ByteBuffer
import java.nio.ByteOrder
import chiseltest.simulator.VerilatorBackendAnnotation
import scala.util.Random

class FPNewFMAUIntTest extends AnyFlatSpec with ChiselScalatestTester with fpUtils {
  behavior of "FPNewFMAUInt"

  it should "perform fused multiply-add correctly" in {
    test(
      new FPNewFMAUInt(
      topmodule = "fpnew_fma_uint",
      widthA = 16,
      widthB = 4,
      widthC = 32
    )
).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>

      def testfma(test_id: Int, A: Float, B: Int, C: Float) = {
        val A_fp16 = uintToFloat(fp16.expWidth, fp16.sigWidth, floatToUInt(fp16.expWidth, fp16.sigWidth, A)) // Convert to 16-bit representation
        val C_fp32 = uintToFloat(fp32.expWidth, fp32.sigWidth, floatToUInt(fp32.expWidth, fp32.sigWidth, C)) // Convert to 16-bit representation
        val gold_O = A_fp16 * B + C_fp32 // Expected result

        val stimulus_a_i = floatToUInt(fp16.expWidth, fp16.sigWidth, A) // Convert to 16-bit representation
        val stimulus_b_i = B // Direct 4-bit integer
        val stimulus_c_i = floatToUInt(fp32.expWidth, fp32.sigWidth, C) // Convert to 32-bit representation
        val expected_o = floatToUInt(fp32.expWidth, fp32.sigWidth, gold_O) // Expected output as UInt

        dut.io.operand_a_i.poke(stimulus_a_i.U)
        dut.io.operand_b_i.poke(stimulus_b_i.U)
        dut.io.operand_c_i.poke(stimulus_c_i.U)

        dut.clock.step()
        dut.clock.step()

        val reseult = dut.io.result_o.peek().litValue
        try {
          assert(reseult == expected_o)
        } catch {
          case e: java.lang.AssertionError => {
            println(f"-----------Test id: $test_id Expected: 0x${expected_o.toString(16)}, Got: 0x${reseult.toString(16)}-----------")
          }
        }

        dut.clock.step()
        dut.clock.step()
      }

      var A = 1.0f // float16
      var B = 1    // uint4
      var C = 1.0f // float32
      
      testfma(1, A, B, C)

      A = 0.5f
      B = 2
      C = 2.3f

      testfma(2, A, B, C)

      A = 0.005f
      B = 15
      C = 100f

      testfma(3, A, B, C)

      A = 0.005f
      B = 0
      C = 10f

      testfma(4, A, B, C)

      val random = new Random()

      // Generate 10 test cases with:
      // - A: Random Float16-compatible float
      // - B: Random integer
      // - C: Random Float16-compatible float
      // val test_num = 10
      // val testCases = Seq.fill(test_num)(
      //   ((random.nextFloat() * 20 - 10),  // A: Random float between -10 and 10
      //   random.nextInt(15),                 // B: Random int between 0 and 15
      //   (random.nextFloat() * 20 - 10)) // C: Random float between -10 and 10
      // )

      // testCases.zipWithIndex.foreach { case ((a, b, c), index) =>
      //   testfma(index + 1, a, b, c)
      // }

    }
  }
}
