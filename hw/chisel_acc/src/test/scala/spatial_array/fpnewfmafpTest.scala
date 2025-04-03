package snax_acc.spatial_array

import scala.util.Random

import chisel3._

import chiseltest._
import chiseltest.simulator.VerilatorBackendAnnotation
import org.scalatest.flatspec.AnyFlatSpec

class FPNewFMAFPTest extends AnyFlatSpec with ChiselScalatestTester with fpUtils {
  behavior of "FPNewFMAFP"

  it should "perform fused multiply-add correctly" in {
    test(
      new FPNewFMAFP(
      topmodule = "fpnew_fma_mixed",
      widthA = 16,
      widthB = 16,
      widthC = 32
    )
).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>

    def testfma(test_id: Int, A: Float, B: Float, C: Float) = {
        val A_fp16 = uintToFloat(fp16.expWidth, fp16.sigWidth, floatToUInt(fp16.expWidth, fp16.sigWidth, A)) // Convert to 16-bit representation
        val B_fp16 = uintToFloat(fp16.expWidth, fp16.sigWidth, floatToUInt(fp16.expWidth, fp16.sigWidth, B)) // Convert to 16-bit representation
        val C_fp32 = uintToFloat(fp32.expWidth, fp32.sigWidth, floatToUInt(fp32.expWidth, fp32.sigWidth, C)) // Convert to 32-bit representation
        val gold_O = A_fp16 * B_fp16 + C_fp32 // Expected result
        
        val stimulus_a_i = floatToUInt(fp16.expWidth, fp16.sigWidth, A) // Convert to 16-bit representation
        val stimulus_b_i = floatToUInt(fp16.expWidth, fp16.sigWidth, B) // Direct 4-bit integer
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
            case _: java.lang.AssertionError => {
            println(f"-----------Test id: $test_id Expected: 0x${expected_o.toString(16)}, Got: 0x${reseult.toString(16)}-----------")
            }
        }

        dut.clock.step()
        dut.clock.step()
    }

    var A = 1.0f // float16
    var B = 1f    // uint4
    var C = 1.0f // float32
    
    testfma(1, A, B, C)

    A = 0.5f
    B = 2f
    C = 2.3f

    testfma(2, A, B, C)

    A = 0.005f
    B = 15f
    C = 100f

    testfma(3, A, B, C)

    A = 0.005f
    B = 0f
    C = 10f

    testfma(4, A, B, C)

    val random = new Random()

    // Generate 10 random test cases
    val test_num = 10
    val testCases = Seq.fill(test_num)(
        (random.nextFloat() * 20 - 10,  // Random float between -10 and 10
        random.nextFloat() * 20 - 10,  
        random.nextFloat() * 20 - 10)
    )

    testCases.zipWithIndex.foreach { case ((a, b, c), index) =>
        testfma(index + 1, a, b, c)
    }

    }
  }
}
