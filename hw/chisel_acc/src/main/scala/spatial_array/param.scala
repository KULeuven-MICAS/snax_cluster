package snax_acc.spatial_array

object OpType {
  def UIntUIntOp       = 1
  def SIntSIntOp       = 2
  def Float16Int4Op    = 3
  def Float16Float16Op = 4

  def fromString(str: String): Int =
    str match {
      case "SIntSInt" => SIntSIntOp
      case "UIntUInt" => UIntUIntOp
      case _          => throw new IllegalArgumentException(s"Unsupported OpType: $str")
    }

}

class SpatialArrayParam(
  val opType:                 Seq[Int],
  val macNum:                 Seq[Int],
  val inputAElemWidth:        Seq[Int],
  val inputBElemWidth:        Seq[Int],
  val inputCElemWidth:        Seq[Int],
  val mulElemWidth:           Seq[Int],
  val outputDElemWidth:       Seq[Int],
  val arrayInputAWidth:       Int,
  val arrayInputBWidth:       Int,
  val arrayInputCWidth:       Int,
  val arrayOutputDWidth:      Int,
  val arrayDim:               Seq[Seq[Seq[Int]]],
  val serialInputCDataWidth:  Int,
  val serialOutputDDataWidth: Int,
  val adderTreeDelay:         Int = 0,
  val configWidth:            Int = 32,
  val csrNum:                 Int = 7
)

object SpatialArrayParam {
  // test config
  def apply(): SpatialArrayParam =
    apply(
      opType                 = Seq(OpType.UIntUIntOp, OpType.SIntSIntOp),
      macNum                 = Seq(1024, 2048),
      inputAElemWidth        = Seq(8, 4),
      inputBElemWidth        = Seq(8, 4),
      inputCElemWidth        = Seq(32, 16),
      mulElemWidth           = Seq(16, 8),
      outputDElemWidth       = Seq(32, 16),
      arrayInputAWidth       = 512,
      arrayInputBWidth       = 512,
      arrayInputCWidth       = 16384,
      arrayOutputDWidth      = 16384,
      serialInputCDataWidth  = 512,
      serialOutputDDataWidth = 512,
      // Seq(Mu, Ku, Nu)
      arrayDim               = Seq(
        Seq(Seq(8, 8, 8), Seq(32, 1, 16), Seq(32, 2, 16)),
        Seq(Seq(8, 16, 8), Seq(32, 1, 32), Seq(32, 1, 16))
      )
    )

  def apply(
    opType:                 Seq[Int],
    macNum:                 Seq[Int],
    inputAElemWidth:        Seq[Int],
    inputBElemWidth:        Seq[Int],
    inputCElemWidth:        Seq[Int],
    mulElemWidth:           Seq[Int],
    outputDElemWidth:       Seq[Int],
    arrayInputAWidth:       Int,
    arrayInputBWidth:       Int,
    arrayInputCWidth:       Int,
    arrayOutputDWidth:      Int,
    arrayDim:               Seq[Seq[Seq[Int]]],
    serialInputCDataWidth:  Int,
    serialOutputDDataWidth: Int,
    adderTreeDelay:         Int = 0
  ): SpatialArrayParam =
    new SpatialArrayParam(
      opType                 = opType,
      macNum                 = macNum,
      inputAElemWidth        = inputAElemWidth,
      inputBElemWidth        = inputBElemWidth,
      inputCElemWidth        = inputCElemWidth,
      mulElemWidth           = mulElemWidth,
      outputDElemWidth       = outputDElemWidth,
      arrayInputAWidth       = arrayInputAWidth,
      arrayInputBWidth       = arrayInputBWidth,
      arrayInputCWidth       = arrayInputCWidth,
      arrayOutputDWidth      = arrayOutputDWidth,
      arrayDim               = arrayDim,
      serialInputCDataWidth  = serialInputCDataWidth,
      serialOutputDDataWidth = serialOutputDDataWidth,
      adderTreeDelay         = adderTreeDelay
    )
}
