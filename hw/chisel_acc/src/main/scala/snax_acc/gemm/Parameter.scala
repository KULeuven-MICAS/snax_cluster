package snax_acc.gemm

case class GemmParams(
  dataWidthA:          Int,
  dataWidthB:          Int,
  dataWidthMul:        Int,
  dataWidthC:          Int,
  dataWidthAccum:      Int,
  subtractionCfgWidth: Int,
  meshRow:             Int,
  tileSize:            Int,
  meshCol:             Int,
  addrWidth:           Int,
  sizeConfigWidth:     Int
)

trait HasGemmParams {
  val params: GemmParams
  // Unpack
  lazy val dataWidthA          = params.dataWidthA
  lazy val dataWidthB          = params.dataWidthB
  lazy val dataWidthMul        = params.dataWidthMul
  lazy val dataWidthC          = params.dataWidthC
  lazy val dataWidthAccum      = params.dataWidthAccum
  lazy val subtractionCfgWidth = params.subtractionCfgWidth
  lazy val meshRow             = params.meshRow
  lazy val tileSize            = params.tileSize
  lazy val meshCol             = params.meshCol
  lazy val addrWidth           = params.addrWidth
  lazy val sizeConfigWidth     = params.sizeConfigWidth

  // Compute
  val a_bits_len  = meshRow * tileSize * dataWidthA
  val b_bits_len  = tileSize * meshCol * dataWidthB
  val sa_bits_len = dataWidthA
  val sb_bits_len = dataWidthB
}

object GemmConstant {

  lazy val dataWidthA     = 8
  lazy val dataWidthB     = 2
  lazy val dataWidthMul   = 18
  lazy val dataWidthC     = 18
  lazy val dataWidthAccum = 18

  lazy val subtractionCfgWidth = 32

  lazy val meshRow  = 1
  lazy val tileSize = 32
  lazy val meshCol  = 32

  lazy val addrWidth       = 32
  lazy val sizeConfigWidth = 32

}

object DefaultConfig {
  val gemmConfig = GemmParams(
    GemmConstant.dataWidthA,
    GemmConstant.dataWidthB,
    GemmConstant.dataWidthMul,
    GemmConstant.dataWidthC,
    GemmConstant.dataWidthAccum,
    GemmConstant.subtractionCfgWidth,
    GemmConstant.meshRow,
    GemmConstant.tileSize,
    GemmConstant.meshCol,
    GemmConstant.addrWidth,
    GemmConstant.sizeConfigWidth
  )
}
