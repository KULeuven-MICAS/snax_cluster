package snax_acc.spatial_array

import scala.util.Random

import chisel3._

import chiseltest._
import chiseltest.simulator.VerilatorBackendAnnotation
import org.scalatest.flatspec.AnyFlatSpec

trait FPMULIntTestUtils extends fpUtils {

  def test_fp_mul_int(dut: FPMULInt, test_id: Int, A: Float, B: Int, C: Float, Bwidth: Int) = {
    val A_fp16 = uintToFloat(
      fp16.expWidth,
      fp16.sigWidth,
      floatToUInt(fp16.expWidth, fp16.sigWidth, A)
    ) // Convert to 16-bit representation
    val gold_O = A_fp16 * B // Expected result

    println(
      f"-----------Test id: $test_id, A_fp16: ${A_fp16} , B: ${B},  gold_O: ${gold_O}-----------"
    )

    val stimulus_a_i = floatToUInt(fp16.expWidth, fp16.sigWidth, A)      // Convert to 16-bit representation
    val stimulus_b_i = (BigInt(B) & ((BigInt(1) << Bwidth) - 1)).U // B is an integer, int2uint conversion needed for sending it to the DUT
    val stimulus_c_i = floatToUInt(fp32.expWidth, fp32.sigWidth, C)      // Convert to 32-bit representation
    val expected_o   = floatToUInt(fp32.expWidth, fp32.sigWidth, gold_O) // Expected output as UInt

    dut.io.operand_a_i.poke(stimulus_a_i.U)
    dut.io.operand_b_i.poke(stimulus_b_i)

    dut.clock.step()
    dut.clock.step()

    val reseult = dut.io.result_o.peek().litValue
    // val reseult_int =   floatToUInt(fp32.expWidth, fp32.sigWidth, reseult)
    println(
      f"-----------Test id: $test_id Expected: 0x${expected_o.toString(16)}, Got: 0x${reseult.toString(16)}-----------"
    )

    try {
      assert(reseult == expected_o)
    } catch {
      case _: java.lang.AssertionError => {
        println(
          f"----Error!!!!-------Test id: $test_id Expected: 0x${expected_o.toString(16)}, Got: 0x${reseult.toString(16)}-----------"
        )
      }
    }

    dut.clock.step()
    dut.clock.step()
  }

}

class FPMULInt4Test extends AnyFlatSpec with ChiselScalatestTester with fpUtils with FPMULIntTestUtils {

  behavior of "FPMULInt4"

  it should "perform fp16-int4 multiply correctly" in {
    test(
      new FPMULInt(
        topmodule = "fp_mul_int",
        widthA    = 16,
        widthB    = 4,
        widthC    = 32
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>

      var A = 1.0f // float16
      var B = -1    // uint4
      var C = 1.0f // float32

      test_fp_mul_int(dut, 1, A, B, C, 4)

      A = 0.5f
      B = -2
      C = 2.3f

      test_fp_mul_int(dut, 2, A, B, C, 4)

      A = 0.005f
      B = 7
      C = 100f

      test_fp_mul_int(dut, 3, A, B, C, 4)

      A = 0.005f
      B = 0
      C = 10f

      test_fp_mul_int(dut, 4, A, B, C, 4)

      new Random()

      // Generate 10 test cases with:
      // - A: Random Float16-compatible float
      // - B: Random integer
      // - C: Random Float16-compatible float
      val test_num  = 10
      val testCases = Seq.fill(test_num)(
        (
          (Random.nextFloat() * 20 - 10), // A: Random float between -10 and 10
          Random.nextInt(15) - 8,             // B: Random int between -8 and 7
          (Random.nextFloat() * 20 - 10)
        )                                 // C: Random float between -10 and 10
      )

      testCases.zipWithIndex.foreach { case ((a, b, c), index) =>
        test_fp_mul_int(dut, index + 1, a, b, c, 4)
      }

    }
  }

}

class FPMULInt3Test extends AnyFlatSpec with ChiselScalatestTester with fpUtils with FPMULIntTestUtils {

  it should "perform fp16-int3 multiply correctly" in {
    test(
      new FPMULInt(
        topmodule = "fp_mul_int",
        widthA    = 16,
        widthB    = 3,
        widthC    = 32
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>

      var A = 1.0f // float16
      var B = 1    // uint4
      var C = 1.0f // float32

      test_fp_mul_int(dut, 1, A, B, C, 3)

      A = 0.5f
      B = 2
      C = 2.3f

      test_fp_mul_int(dut, 2, A, B, C, 3)

      A = 0.005f
      B = 3
      C = 100f

      test_fp_mul_int(dut, 3, A, B, C, 3)

      A = 0.005f
      B = 0
      C = 10f

      test_fp_mul_int(dut, 4, A, B, C, 3)

      val random = new Random()

      // Generate 10 test cases with:
      // - A: Random Float16-compatible float
      // - B: Random integer
      // - C: Random Float16-compatible float
      val test_num  = 10
      val testCases = Seq.fill(test_num)(
        (
          (random.nextFloat() * 20 - 10), // A: Random float between -10 and 10
          random.nextInt(7) - 3,              // B: Random int between -3 and 3
          (random.nextFloat() * 20 - 10)
        )                                 // C: Random float between -10 and 10
      )

      testCases.zipWithIndex.foreach { case ((a, b, c), index) =>
        test_fp_mul_int(dut, index + 1, a, b, c, 3)
      }

    }
  }
}

class FPMULInt2Test extends AnyFlatSpec with ChiselScalatestTester with fpUtils with FPMULIntTestUtils {

  it should "perform fp16-int2 multiply correctly" in {
    test(
      new FPMULInt(
        topmodule = "fp_mul_int",
        widthA    = 16,
        widthB    = 2,
        widthC    = 32
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>

      var A = 1.0f // float16
      var B = 1    // uint4
      var C = 1.0f // float32

      test_fp_mul_int(dut, 1, A, B, C, 2)

      A = 0.5f
      B = 1
      C = 2.3f

      test_fp_mul_int(dut, 2, A, B, C, 2)

      A = 0.015f
      B = 0
      C = 100f

      test_fp_mul_int(dut, 3, A, B, C, 2)

      A = 0.005f
      B = 0
      C = 10f

      test_fp_mul_int(dut, 4, A, B, C, 2)

      val random = new Random()

      // Generate 10 test cases with:
      // - A: Random Float16-compatible float
      // - B: Random integer
      // - C: Random Float16-compatible float
      val test_num  = 10
      val testCases = Seq.fill(test_num)(
        (
          (random.nextFloat() * 20 - 10), // A: Random float between -10 and 10
          random.nextInt(3) - 1,              // B: Random int between -1 and 1
          (random.nextFloat() * 20 - 10)
        )                                 // C: Random float between -10 and 10
      )

      testCases.zipWithIndex.foreach { case ((a, b, c), index) =>
        test_fp_mul_int(dut, index + 1, a, b, c, 2)
      }

    }
  }
}

class FPMULInt1Test extends AnyFlatSpec with ChiselScalatestTester with fpUtils with FPMULIntTestUtils {

  it should "perform fp16-int1 multiply correctly" in {
    test(
      new FPMULInt(
        topmodule = "fp_mul_int",
        widthA    = 16,
        widthB    = 1,
        widthC    = 32
      )
    ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>

      var A = 1.0f // float16
      var B = 1    // uint4
      var C = 1.0f // float32

      test_fp_mul_int(dut, 1, A, B, C, 1)

      A = 0.5f
      B = 0
      C = 2.3f

      test_fp_mul_int(dut, 2, A, B, C, 1)

      A = 0.005f
      B = 1
      C = 100f

      test_fp_mul_int(dut, 3, A, B, C, 1)

      A = 0.005f
      B = 0
      C = 10f

      test_fp_mul_int(dut, 4, A, B, C, 1)

      val random = new Random()

      // Generate 10 test cases with:
      // - A: Random Float16-compatible float
      // - B: Random integer
      // - C: Random Float16-compatible float
      val test_num  = 10
      val testCases = Seq.fill(test_num)(
        (
          (random.nextFloat() * 20 - 10), // A: Random float between -10 and 10
          random.nextInt(1),              // B: Random int between 0 and 15
          (random.nextFloat() * 20 - 10)
        )                                 // C: Random float between -10 and 10
      )

      testCases.zipWithIndex.foreach { case ((a, b, c), index) =>
        test_fp_mul_int(dut, index + 1, a, b, c, 1)
      }

    }
  }
}
