
package snax.DataPathExtension
import scala.util.Random

import chiseltest._
import snax.DataPathExtension.HasOldRescaleDown

class OldRescaleDownTester extends DataPathExtensionTester(TreadleBackendAnnotation) {

  def OldGoldenModel(
    input:      Int,
    input_zp:   Int,
    multiplier: Int,
    output_zp:  Int,
    shift:      Int
  ): Byte = {

    // Default values for parameters not passed in (matching the test setup)
    val max_int_i      = 127  // Max value for signed byte
    val min_int_i      = -128 // Min value for signed byte
    val double_round_i = true // Assuming double rounding is enabled

    // Step 1: Subtract input zero point
    val var_1 = input - input_zp

    // Step 2: Multiply with the multiplier avoiding overflow
    val var_2 = var_1.toLong * multiplier.toLong

    // Step 3: Right shift
    val var_3 = (var_2 >> (shift - 1)).toInt

    // Step 4: Apply double rounding if necessary
    val var_4 = if (double_round_i) {
      if (var_3 >= 0) var_3 + 1 else var_3 - 1
    } else {
      var_3
    }

    // Step 5: Final right shift
    val var_5 = var_4 >> 1

    // Step 6: Add output zero point
    val var_6 = var_5 + output_zp

    // Step 7: Clip the values to be within min and max integer range
    val var_7 = math.max(min_int_i, math.min(max_int_i, var_6))

    var_7.toByte
  }

  def hasExtension =
    new HasOldRescaleDown(
      in_elementWidth  = 32,
      out_elementWidth = 8
    )
  val input_zp     = 0
  val multiplier   = 1140768824
  val output_zp    = 0
  val shift        = 47
  val csr_vec      = Seq(input_zp, multiplier, output_zp, shift)

  val inputData  = collection.mutable.Buffer[BigInt]()
  val outputData = collection.mutable.Buffer[BigInt]()

  Random.setSeed(1) // For reproducibility

  for (_ <- 0 until 128) {
    // val inputMatrix: Array[Int] = Array.fill(64)(-5956158)
    val inputMatrix: Array[Int] = Array.fill(64)(Random.between(-2 << 22, 2 << 22))
    val inputMatrix1 = inputMatrix.slice(0, 16)
    val inputMatrix2 = inputMatrix.slice(16, 32)
    val inputMatrix3 = inputMatrix.slice(32, 48)
    val inputMatrix4 = inputMatrix.slice(48, 64)
    inputData.append(BigInt(inputMatrix1.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(inputMatrix2.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(inputMatrix3.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(inputMatrix4.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))

    val outputMatrix = inputMatrix.map { i => OldGoldenModel(i, input_zp, multiplier, output_zp, shift) }
    outputData.append(BigInt(outputMatrix.map { i => f"$i%02X" }.reverse.reduce(_ + _), 16))
  }
  val input_data_vec = inputData.toSeq

  val output_data_vec = outputData.toSeq

}
