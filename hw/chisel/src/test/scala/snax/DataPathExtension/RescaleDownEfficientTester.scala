package snax.DataPathExtension
import scala.util.Random

import snax.DataPathExtension.HasRescaleDownEfficient

class RescaleDownEfficientTester extends DataPathExtensionTester {

  def hasExtension = new HasRescaleDownEfficient(
    in_elementWidth = 32,
    out_elementWidth = 8
  )
  val input_zp = 0
  val multiplier = 1140768824
  val output_zp = 0
  val shift = 47
  val csr_vec           = Seq(input_zp, multiplier, output_zp, shift)

  val inputData  = collection.mutable.Buffer[BigInt]()
  val outputData = collection.mutable.Buffer[BigInt]()

  for (_ <- 0 until 128) {
    val inputMatrix: Array[Int] = Array.fill(64)(1360653)
    //val inputMatrix: Array[Int] = Array.fill(64)(Random.between(-2 << 22, 2 << 22))
    val inputMatrix1 = inputMatrix.slice(0,16)
    val inputMatrix2 = inputMatrix.slice(16,32)
    val inputMatrix3 = inputMatrix.slice(32,48)
    val inputMatrix4 = inputMatrix.slice(48,64)
    inputData.append(BigInt(inputMatrix1.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(inputMatrix2.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(inputMatrix3.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(inputMatrix4.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))


    //val outputMatrix = inputMatrix.map( i => ((i - input_zp) * multiplier) >> shift + output_zp)
    val outputMatrix = inputMatrix.map { i =>
      (((i.toLong - input_zp) * multiplier.toLong) >> shift) + output_zp
    }
    outputData.append(BigInt(outputMatrix.map { i => f"${i & 0xFF}%02X" }.reverse.reduce(_ + _), 16))
  }
  val input_data_vec = inputData.toSeq

  val output_data_vec = outputData.toSeq
}

