package snax.DataPathExtension
import scala.util.Random

import chiseltest._
import snax.DataPathExtension.HasElementwiseAdd

class ElementwiseAddToLong3Tester extends DataPathExtensionTester(TreadleBackendAnnotation) {

  def hasExtension = new HasElementwiseAddToLong(in_elementWidth = 8, out_elementWidth = 32, dataWidth = 512)

  val amount_of_vectors = 3
  val csr_vec           = Seq(amount_of_vectors)

  val inputData  = collection.mutable.Buffer[BigInt]()
  val outputData = collection.mutable.Buffer[BigInt]()

  for (_ <- 0 until 128) {
    val InputMatrix1: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix2: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix3: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    inputData.append(BigInt(InputMatrix1.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix2.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix3.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))

    val OutputMatrix  = InputMatrix1.zip(InputMatrix2.zip(InputMatrix3)).map { case (a, (b, c)) =>
      a.zip(b.zip(c)).map { case (x, (y, z)) => x + y + z }
    }
    val OutputMatrix1 = OutputMatrix.slice(0,2)
    val OutputMatrix2 = OutputMatrix.slice(2,4)
    val OutputMatrix3 = OutputMatrix.slice(4,6)
    val OutputMatrix4 = OutputMatrix.slice(6,8)
    outputData.append(BigInt(OutputMatrix1.flatten.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
    outputData.append(BigInt(OutputMatrix2.flatten.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
    outputData.append(BigInt(OutputMatrix3.flatten.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
    outputData.append(BigInt(OutputMatrix4.flatten.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
  }

  val input_data_vec = inputData.toSeq

  val output_data_vec = outputData.toSeq
}


class ElementwiseAddToLong15Tester extends DataPathExtensionTester(TreadleBackendAnnotation) {

  def hasExtension = new HasElementwiseAddToLong(in_elementWidth = 8, out_elementWidth = 32, dataWidth = 512)

  val amount_of_vectors = 15
  val csr_vec           = Seq(amount_of_vectors)

  val inputData  = collection.mutable.Buffer[BigInt]()
  val outputData = collection.mutable.Buffer[BigInt]()

  for (_ <- 0 until 128) {
    val InputMatrix1: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix2: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix3: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix4: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix5: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix6: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix7: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix8: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix9: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix10: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))  
    val InputMatrix11: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix12: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix13: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix14: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    val InputMatrix15: Array[Array[Int]] = Array.fill(8, 8)(Random.between(-128, 127))
    inputData.append(BigInt(InputMatrix1.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix2.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix3.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix4.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix5.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix6.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix7.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix8.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix9.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix10.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix11.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix12.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix13.flatten.map { i => f"${i & 0xff}%02X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(InputMatrix14.flatten.map { i => f"${i & 0xff}%

    val OutputMatrix  = InputMatrix1.zip(InputMatrix2.zip(InputMatrix3)).map { case (a, (b, c)) =>
      a.zip(b.zip(c)).map { case (x, (y, z)) => x + y + z }
    }
    val OutputMatrix1 = OutputMatrix.slice(0,2)
    val OutputMatrix2 = OutputMatrix.slice(2,4)
    val OutputMatrix3 = OutputMatrix.slice(4,6)
    val OutputMatrix4 = OutputMatrix.slice(6,8)
    outputData.append(BigInt(OutputMatrix1.flatten.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
    outputData.append(BigInt(OutputMatrix2.flatten.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
    outputData.append(BigInt(OutputMatrix3.flatten.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
    outputData.append(BigInt(OutputMatrix4.flatten.map { i => f"$i%08X" }.reverse.reduce(_ + _), 16))
  }

  val input_data_vec = inputData.toSeq

  val output_data_vec = outputData.toSeq
}
