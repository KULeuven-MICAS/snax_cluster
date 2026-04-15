package snax.DataPathExtension
import scala.util.Random

import chiseltest._
import snax.DataPathExtension.HasTransposer

private object XDMATransposerTesterHelper {
  private val sharedRow          = Seq(8, 8, 8)
  private val sharedCol          = Seq(8, 8, 8)
  private val sharedElementWidth = Seq(8, 16, 32)
  private val sharedDataWidth    = 512
  private val numTestMatrices    = 128

  def newHasExtension: HasTransposer =
    new HasTransposer(
      row = sharedRow,
      col = sharedCol,
      elementWidth = sharedElementWidth,
      dataWidth = sharedDataWidth
    )

  private def randomValue(random: Random, elementWidth: Int): Long =
    elementWidth match {
      case 8 | 16 => random.nextInt(1 << elementWidth).toLong
      case 32     => random.nextInt().toLong & 0xffffffffL
      case _ =>
        throw new IllegalArgumentException(
          s"Unsupported element width $elementWidth for TransposerTester"
        )
    }

  private def transferPerTranspose(row: Int, col: Int, elementWidth: Int): Int = {
    val elementPerTransfer = math.min(sharedDataWidth, row * col * elementWidth) / elementWidth
    (row * col + elementPerTransfer - 1) / elementPerTransfer
  }

  private def packMatrix(matrix: Array[Array[Long]], elementWidth: Int): BigInt =
    matrix.flatten.reverse.foldLeft(BigInt(0)) { case (acc, value) =>
      (acc << elementWidth) + BigInt(value)
    }

  private def buildTransfers(
    matrix:               Array[Array[Long]],
    transferPerTranspose: Int,
    columnsPerTransfer:   Int,
    elementWidth:         Int
  ): Seq[BigInt] =
    (0 until transferPerTranspose).map { transferIdx =>
      val transferStart = transferIdx * columnsPerTransfer
      val transferEnd   = transferStart + columnsPerTransfer
      packMatrix(matrix.map(row => row.slice(transferStart, transferEnd)), elementWidth)
    }

  def generateVectors(
    row:          Int,
    col:          Int,
    elementWidth: Int,
    seed:         Int
  ): (Seq[BigInt], Seq[BigInt]) = {
    val transferCount = transferPerTranspose(row, col, elementWidth)
    require(
      col % transferCount == 0 && row % transferCount == 0,
      s"TransposerTester expects row and col to be divisible by transfer count $transferCount"
    )

    val random     = new Random(seed)
    val inputData  = collection.mutable.Buffer[BigInt]()
    val outputData = collection.mutable.Buffer[BigInt]()

    for (_ <- 0 until numTestMatrices) {
      val inputMatrix: Array[Array[Long]] = Array.fill(row, col)(randomValue(random, elementWidth))
      val outputMatrix                    = inputMatrix.transpose

      inputData ++= buildTransfers(inputMatrix, transferCount, col / transferCount, elementWidth)
      outputData ++= buildTransfers(outputMatrix, transferCount, row / transferCount, elementWidth)
    }

    (inputData.toSeq, outputData.toSeq)
  }
}

abstract class XDMATransposerTester(modeIdx: Int, row: Int, col: Int, elementWidth: Int, seed: Int)
    extends DataPathExtensionTester {

  override lazy val hasExtension = XDMATransposerTesterHelper.newHasExtension

  override val csr_vec = Seq(modeIdx)

  private val (generatedInputData, generatedOutputData) =
    XDMATransposerTesterHelper.generateVectors(row, col, elementWidth, seed)

  override val input_data_vec  = generatedInputData
  override val output_data_vec = generatedOutputData
}

class XDMATransposerMode1Tester extends XDMATransposerTester(modeIdx = 0, row = 8, col = 8, elementWidth = 8, seed = 8)

class XDMATransposerMode2Tester extends XDMATransposerTester(modeIdx = 1, row = 8, col = 8, elementWidth = 16, seed = 16)

class XDMATransposerMode3Tester extends XDMATransposerTester(modeIdx = 2, row = 8, col = 8, elementWidth = 32, seed = 32)

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

  for (_ <- 0 until 128) {
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

class TransposerRow8Col64Ewidth8Dwidth4096Tester extends DataPathExtensionTester(TreadleBackendAnnotation) {

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
