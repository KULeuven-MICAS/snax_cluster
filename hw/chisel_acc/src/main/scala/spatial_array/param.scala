package snax_acc.spatial_array

import chisel3._

object OpType {
  def UIntUIntOp = 1
  def SIntSIntOp = 2
  def Float16IntOp = 3
  def Float16Float16Op = 4
}

class SpatialArrayParam(
    val opType: Int,
    val macNum: Int,
    val inputAElemWidth: Int,
    val inputBElemWidth: Int,
    val inputCElemWidth: Int,
    val mulElemWidth: Int,
    val outElemWidth: Int,
    val inputAWidth: Int,
    val inputBWidth: Int,
    val inputCWidth: Int,
    val outputWidth: Int,
    val arrayDim: Seq[Seq[Int]]
)

object SpatialArrayParam {
  // test config
  def apply(): SpatialArrayParam = apply(
    opType = OpType.UIntUIntOp,
    macNum = 1024,
    inputAElemWidth = 8,
    inputBElemWidth = 8,
    inputCElemWidth = 8,
    mulElemWidth = 16,
    outElemWidth = 32,
    inputAWidth = 512,
    inputBWidth = 512,
    inputCWidth = 16384,
    outputWidth = 16384,
    // Seq(Mu, Ku, Nu)
    arrayDim = Seq(Seq(8, 8, 8), Seq(32, 1, 16), Seq(32, 2, 16))
  )

  def apply(
      opType: Int,
      macNum: Int,
      inputAElemWidth: Int,
      inputBElemWidth: Int,
      inputCElemWidth: Int,
      mulElemWidth: Int,
      outElemWidth: Int,
      inputAWidth: Int,
      inputBWidth: Int,
      inputCWidth: Int,
      outputWidth: Int,
      arrayDim: Seq[Seq[Int]]
  ): SpatialArrayParam = new SpatialArrayParam(
    opType = OpType.UIntUIntOp,
    macNum = macNum,
    inputAElemWidth = inputAElemWidth,
    inputBElemWidth = inputBElemWidth,
    inputCElemWidth = inputCElemWidth,
    mulElemWidth = mulElemWidth,
    outElemWidth = outElemWidth,
    inputAWidth = inputAWidth,
    inputBWidth = inputBWidth,
    inputCWidth = inputCWidth,
    outputWidth = outputWidth,
    arrayDim = arrayDim
  )
}
