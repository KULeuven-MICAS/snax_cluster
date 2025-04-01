package snax_acc.spatial_array

import chisel3._

object OpType {
  def UIntUIntOp = 1
  def SIntSIntOp = 2
  def Float16IntOp = 3
  def Float16Float16Op = 4
}

class SpatialArrayParam(
    val opType: Seq[Int],
    val macNum: Seq[Int],
    val inputAElemWidth: Seq[Int],
    val inputBElemWidth: Seq[Int],
    val inputCElemWidth: Seq[Int],
    val mulElemWidth: Seq[Int],
    val outElemWidth: Seq[Int],
    val inputAWidth: Int,
    val inputBWidth: Int,
    val inputCWidth: Int,
    val outputDWidth: Int,
    val arrayDim: Seq[Seq[Seq[Int]]],
    val configWidth: Int = 32
)

object SpatialArrayParam {
  // test config
  def apply(): SpatialArrayParam = apply(
    opType = Seq(OpType.UIntUIntOp, OpType.SIntSIntOp),
    macNum = Seq(1024, 2048),
    inputAElemWidth = Seq(8, 4),
    inputBElemWidth = Seq(8, 4),
    inputCElemWidth = Seq(8, 4),
    mulElemWidth = Seq(16, 8),
    outElemWidth = Seq(32, 16),
    inputAWidth = 512,
    inputBWidth = 512,
    inputCWidth = 16384,
    outputDWidth = 16384,
    // Seq(Mu, Ku, Nu)
    arrayDim = Seq(
      Seq(Seq(8, 8, 8), Seq(32, 1, 16), Seq(32, 2, 16)),
      Seq(Seq(8, 16, 8), Seq(32, 1, 32), Seq(32, 1, 16))
    )
  )

  def apply(
      opType: Seq[Int],
      macNum: Seq[Int],
      inputAElemWidth: Seq[Int],
      inputBElemWidth: Seq[Int],
      inputCElemWidth: Seq[Int],
      mulElemWidth: Seq[Int],
      outElemWidth: Seq[Int],
      inputAWidth: Int,
      inputBWidth: Int,
      inputCWidth: Int,
      outputDWidth: Int,
      arrayDim: Seq[Seq[Seq[Int]]]
  ): SpatialArrayParam = new SpatialArrayParam(
    opType = opType,
    macNum = macNum,
    inputAElemWidth = inputAElemWidth,
    inputBElemWidth = inputBElemWidth,
    inputCElemWidth = inputCElemWidth,
    mulElemWidth = mulElemWidth,
    outElemWidth = outElemWidth,
    inputAWidth = inputAWidth,
    inputBWidth = inputBWidth,
    inputCWidth = inputCWidth,
    outputDWidth = outputDWidth,
    arrayDim = arrayDim
  )
}
