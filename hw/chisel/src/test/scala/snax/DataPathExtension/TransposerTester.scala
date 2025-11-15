package snax.DataPathExtension
import scala.util.Random

import snax.DataPathExtension.HasTransposer

class TransposerTester extends DataPathExtensionTester {

  def hasExtension = new HasTransposer(row = Seq(8), col = Seq(8), elementWidth = Seq(16), dataWidth = 512)

  val csr_vec = Seq()

  val inputData  = collection.mutable.Buffer[BigInt]()
  val outputData = collection.mutable.Buffer[BigInt]()

  for (_ <- 0 until 128) {
    val inputMatrix: Array[Array[Int]] = Array.fill(8, 8)(Random.nextInt(1 << 16))
    val leftInputMatrix  = inputMatrix.map(row => row.slice(0, 4))
    val rightInputMatrix = inputMatrix.map(row => row.slice(4, 8))
    inputData.append(BigInt(leftInputMatrix.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
    inputData.append(BigInt(rightInputMatrix.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))

    val outputMatrix: Array[Array[Int]] = inputMatrix.transpose
    val leftOutputMatrix  = outputMatrix.map(row => row.slice(0, 4))
    val rightOutputMatrix = outputMatrix.map(row => row.slice(4, 8))
    outputData.append(BigInt(leftOutputMatrix.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
    outputData.append(BigInt(rightOutputMatrix.flatten.map { i => f"$i%04X" }.reverse.reduce(_ + _), 16))
  }

  val input_data_vec = inputData.toSeq

  val output_data_vec = outputData.toSeq
}

class TransposerRow32Col4Ewidth8Dwidth2048Tester extends DataPathExtensionTester {
  
  override lazy val hasExtension = new HasTransposer(row = Seq(32), col = Seq(4), elementWidth = Seq(8), dataWidth = 2048)

  val csr_vec = Seq()

  val inputData  = collection.mutable.Buffer[BigInt]()
  val outputData = collection.mutable.Buffer[BigInt]()

  for (_ <- 0 until 128) {
    val inputMatrix: Array[Array[Int]] = Array.fill(32, 4)(Random.nextInt(1 << 8))
    inputData.append(BigInt(inputMatrix.flatten.map { i => f"$i%02X" }.reverse.reduce(_ + _), 16))

    val outputMatrix: Array[Array[Int]] = inputMatrix.transpose
    outputData.append(BigInt(outputMatrix.flatten.map { i => f"$i%02X" }.reverse.reduce(_ + _), 16))
  }

  val input_data_vec = inputData.toSeq

  val output_data_vec = outputData.toSeq
}

class TransposerRow1Col8Ewidth8Dwidth2048Tester extends DataPathExtensionTester(debugMode = true){
  
  override lazy val hasExtension = new HasTransposer(row = Seq(1), col = Seq(8), elementWidth = Seq(8), dataWidth = 2048)

  val csr_vec = Seq()

  val inputData  = collection.mutable.Buffer[BigInt]()
  val outputData = collection.mutable.Buffer[BigInt]()

  for (_ <- 0 until 1) {
    val inputMatrix: Array[Array[Int]] = Array.fill(1, 8)(Random.nextInt(1 << 8))
    inputData.append(BigInt(inputMatrix.flatten.map { i => f"$i%02X" }.reverse.reduce(_ + _), 16))

    val outputMatrix: Array[Array[Int]] = inputMatrix.transpose
    outputData.append(BigInt(outputMatrix.flatten.map { i => f"$i%02X" }.reverse.reduce(_ + _), 16))
  }

  val input_data_vec = inputData.toSeq

  val output_data_vec = outputData.toSeq
}

class TransposerRow4Col8Ewidth8Dwidth2048Tester extends DataPathExtensionTester {
  
  override lazy val hasExtension = new HasTransposer(row = Seq(4), col = Seq(8), elementWidth = Seq(8), dataWidth = 2048)

  val csr_vec = Seq()

  val inputData  = collection.mutable.Buffer[BigInt]()
  val outputData = collection.mutable.Buffer[BigInt]()

  for (_ <- 0 until 128) {
    val inputMatrix: Array[Array[Int]] = Array.fill(4, 8)(Random.nextInt(1 << 8))
    inputData.append(BigInt(inputMatrix.flatten.map { i => f"$i%02X" }.reverse.reduce(_ + _), 16))

    val outputMatrix: Array[Array[Int]] = inputMatrix.transpose
    outputData.append(BigInt(outputMatrix.flatten.map { i => f"$i%02X" }.reverse.reduce(_ + _), 16))
  }

  val input_data_vec = inputData.toSeq

  val output_data_vec = outputData.toSeq
}

class TransposerRow8Col8Ewidth8Dwidth2048Tester extends DataPathExtensionTester {
  
  override lazy val hasExtension = new HasTransposer(row = Seq(8), col = Seq(8), elementWidth = Seq(8), dataWidth = 2048)

  val csr_vec = Seq()

  val inputData  = collection.mutable.Buffer[BigInt]()
  val outputData = collection.mutable.Buffer[BigInt]()

  for (_ <- 0 until 128) {
    val inputMatrix: Array[Array[Int]] = Array.fill(8, 8)(Random.nextInt(1 << 8))
    inputData.append(BigInt(inputMatrix.flatten.map { i => f"$i%02X" }.reverse.reduce(_ + _), 16))

    val outputMatrix: Array[Array[Int]] = inputMatrix.transpose
    outputData.append(BigInt(outputMatrix.flatten.map { i => f"$i%02X" }.reverse.reduce(_ + _), 16))
  }

  val input_data_vec = inputData.toSeq

  val output_data_vec = outputData.toSeq
}

class TransposerRow8Col64Ewidth8Dwidth4096Tester extends DataPathExtensionTester {
  
  override lazy val hasExtension = new HasTransposer(row = Seq(8), col = Seq(64), elementWidth = Seq(8), dataWidth = 4096)

  val csr_vec = Seq()

  val inputData  = collection.mutable.Buffer[BigInt]()
  val outputData = collection.mutable.Buffer[BigInt]()

  for (_ <- 0 until 128) {
    val inputMatrix: Array[Array[Int]] = Array.fill(8, 64)(Random.nextInt(1 << 8))
    inputData.append(BigInt(inputMatrix.flatten.map { i => f"$i%02X" }.reverse.reduce(_ + _), 16))

    val outputMatrix: Array[Array[Int]] = inputMatrix.transpose
    outputData.append(BigInt(outputMatrix.flatten.map { i => f"$i%02X" }.reverse.reduce(_ + _), 16))
  }

  val input_data_vec = inputData.toSeq

  val output_data_vec = outputData.toSeq
}
