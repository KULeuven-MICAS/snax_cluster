package snax.DataPathExtension

import scala.util.Random

import chiseltest._

abstract class ElementWiseAddTester(numOperandPerAdd: Int, numOperand: Int, seed: Int = 0, debugMode: Boolean = false)
    extends DataPathExtensionTester(TreadleBackendAnnotation, debugMode) {

  private def elementWidth     = 32
  private def dataWidth        = 512
  private def elementPerVector = dataWidth / elementWidth

  require(numOperandPerAdd > 0, "Number of operands per addition must be greater than 0")
  require(numOperand % elementPerVector == 0, "Total number of operands must be a multiple of one data vector")
  require(
    numOperand       % (numOperandPerAdd * elementPerVector) == 0,
    "Total number of operands must be a multiple of operands per addition"
  )

  def hasExtension = new HasElementwiseAdd(elementWidth = elementWidth, dataWidth = dataWidth)

  val csr_vec = Seq(numOperandPerAdd)

  private val random = new Random(seed)

  private val inputData: Array[Array[Int]] =
    Array.fill(numOperand / elementPerVector, elementPerVector)(random.nextInt())

  private val outputData: Array[Array[Int]] =
    inputData
      .grouped(numOperandPerAdd)
      .map { group =>
        group.reduce { (left, right) =>
          left.zip(right).map { case (x, y) => x + y }
        }
      }
      .toArray

  private def packVector(data: Array[Int]): BigInt = {
    require(data.length == elementPerVector, s"Expected $elementPerVector elements per data vector")

    data.zipWithIndex.foldLeft(BigInt(0)) { case (packed, (value, laneIdx)) =>
      packed | (BigInt(value.toLong & 0xffffffffL) << (laneIdx * elementWidth))
    }
  }

  val input_data_vec = inputData.map(packVector).toSeq

  val output_data_vec = outputData.map(packVector).toSeq
}

class ElementwiseAdd1Tester extends ElementWiseAddTester(numOperandPerAdd = 1, numOperand = 1 * 16 * 256, seed = 1, debugMode = true)

class ElementwiseAdd2Tester extends ElementWiseAddTester(numOperandPerAdd = 2, numOperand = 2 * 16 * 256, seed = 2, debugMode = true)

class ElementwiseAdd3Tester extends ElementWiseAddTester(numOperandPerAdd = 3, numOperand = 3 * 16 * 256, seed = 3, debugMode = true)
