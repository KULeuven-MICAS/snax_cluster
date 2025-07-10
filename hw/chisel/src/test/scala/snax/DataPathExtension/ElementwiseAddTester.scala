package snax.DataPathExtension
import scala.util.Random

import snax.DataPathExtension.hasElementwiseAdd

class ElementwiseAdd2Tester extends DataPathExtensionTester {

  def hasExtension = new hasElementwiseAdd(elementWidth = 16, dataWidth = 512)

  val amount_of_vectors = 2
  val csr_vec = Seq(amount_of_vectors)

  val inputData  = collection.mutable.Buffer[BigInt]()
  val outputData = collection.mutable.Buffer[BigInt]()

  for (_ <- 0 until 128) {
    val inputMatrix1: Array[Array[Int]] = Array.fill(8, 8)(Random.nextInt(1 << 15))
    val inputMatrix2: Array[Array[Int]] = Array.fill(8, 8)(Random.nextInt(1 << 15))
    val leftInputMatrix1  = inputMatrix1.map(row => row.slice(0, 4))
    val rightInputMatrix1 = inputMatrix1.map(row => row.slice(4, 8))
    val leftInputMatrix2  = inputMatrix2.map(row => row.slice(0, 4))
    val rightInputMatrix2 = inputMatrix2.map(row => row.slice(4, 8))
    inputData.append(BigInt(leftInputMatrix1.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(leftInputMatrix2.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(rightInputMatrix1.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(rightInputMatrix2.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))

    val leftOutputMatrix = leftInputMatrix1.zip(leftInputMatrix2).map { case (a, b) => a.zip(b).map { case (x, y) => x + y } }
    val rightOutputMatrix = rightInputMatrix1.zip(rightInputMatrix2).map { case (a, b) => a.zip(b).map { case (x, y) => x + y } }
    outputData.append(BigInt(leftOutputMatrix.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
    outputData.append(BigInt(rightOutputMatrix.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
  }

  val input_data_vec = inputData.toSeq

  val output_data_vec = outputData.toSeq
}

class ElementwiseAdd3Tester extends DataPathExtensionTester {

  def hasExtension = new hasElementwiseAdd(elementWidth = 16, dataWidth = 512)

  val amount_of_vectors = 3
  val csr_vec = Seq(amount_of_vectors)

  val inputData  = collection.mutable.Buffer[BigInt]()
  val outputData = collection.mutable.Buffer[BigInt]()

  for (_ <- 0 until 128) {
    val inputMatrix1: Array[Array[Int]] = Array.fill(8, 8)(Random.nextInt(1 << 14))
    val inputMatrix2: Array[Array[Int]] = Array.fill(8, 8)(Random.nextInt(1 << 14))
    val inputMatrix3: Array[Array[Int]] = Array.fill(8, 8)(Random.nextInt(1 << 14))
    val leftInputMatrix1  = inputMatrix1.map(row => row.slice(0, 4))
    val rightInputMatrix1 = inputMatrix1.map(row => row.slice(4, 8))
    val leftInputMatrix2  = inputMatrix2.map(row => row.slice(0, 4))
    val rightInputMatrix2 = inputMatrix2.map(row => row.slice(4, 8))
    val leftInputMatrix3  = inputMatrix3.map(row => row.slice(0, 4))
    val rightInputMatrix3 = inputMatrix3.map(row => row.slice(4, 8))
    inputData.append(BigInt(leftInputMatrix1.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(leftInputMatrix2.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(leftInputMatrix3.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(rightInputMatrix1.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(rightInputMatrix2.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(rightInputMatrix3.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))

    val leftOutputMatrix = leftInputMatrix1.zip(leftInputMatrix2.zip(leftInputMatrix3)).map { case (a, (b,c)) => a.zip(b.zip(c)).map { case (x, (y, z)) => x + y + z } }
    val rightOutputMatrix = rightInputMatrix1.zip(rightInputMatrix2.zip(rightInputMatrix3)).map { case (a, (b,c)) => a.zip(b.zip(c)).map { case (x, (y, z)) => x + y + z } }
    outputData.append(BigInt(leftOutputMatrix.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
    outputData.append(BigInt(rightOutputMatrix.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
  }

  val input_data_vec = inputData.toSeq

  val output_data_vec = outputData.toSeq
}