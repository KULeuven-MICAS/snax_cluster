package snax_acc.spatial_array

import chisel3._

class SpatialArrayParam(
    val macNum: Int,
    val inputAType: Data,
    val inputBType: Data,
    val inputCType: Data,
    val mulType: Data,
    val outType: Data,
    val inputAWidth: Int,
    val inputBWidth: Int,
    val inputCWidth: Int,
    val outputWidth: Int,
    val arrayDim: Seq[Seq[Int]]
)

object SpatialArrayParam {
  // test config
  def apply(): SpatialArrayParam = apply(
    macNum = 1024,
    inputAType = UInt(8.W),
    inputBType = UInt(8.W),
    inputCType = UInt(8.W),
    mulType = UInt(16.W),
    outType = UInt(32.W),
    inputAWidth = 512,
    inputBWidth = 512,
    inputCWidth = 16384,
    outputWidth = 16384,
    // Seq(Mu, Ku, Nu)
    arrayDim = Seq(Seq(8, 8, 8), Seq(32, 1, 16), Seq(32, 2, 16))
  )

  def apply(
      macNum: Int,
      inputAType: Data,
      inputBType: Data,
      inputCType: Data,
      mulType: Data,
      outType: Data,
      inputAWidth: Int,
      inputBWidth: Int,
      inputCWidth: Int,
      outputWidth: Int,
      arrayDim: Seq[Seq[Int]]
  ): SpatialArrayParam = new SpatialArrayParam(
    macNum = macNum,
    inputAType = inputAType,
    inputBType = inputBType,
    inputCType = inputCType,
    mulType = mulType,
    outType = outType,
    inputAWidth = inputAWidth,
    inputBWidth = inputBWidth,
    inputCWidth = inputCWidth,
    outputWidth = outputWidth,
    arrayDim = arrayDim
  )
}
