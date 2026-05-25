package snax.DataPathExtension

import chisel3._
import chisel3.util._

class HasElementwiseAdd(
  elementWidth: Int = 32,
  dataWidth:    Int = 512
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = s"ElementwiseAddBit${elementWidth}",
      userCsrNum = 1,
      dataWidth  = dataWidth match {
        case 0 => elementWidth
        case _ => dataWidth
      }
    )

  def instantiate(clusterName: String): ElementwiseAdd =
    Module(
      new ElementwiseAdd(elementWidth) {
        override def desiredName = clusterName + namePostfix
      }
    )
}

class ElementwiseAdd(
  elementWidth: Int
)(implicit extensionParam: DataPathExtensionParam)
    extends DataPathExtension {
  require(
    extensionParam.dataWidth % elementWidth == 0,
    s"Data width ${extensionParam.dataWidth} must be a multiple of element width $elementWidth"
  )

  private val elementPerVector = extensionParam.dataWidth / elementWidth

  // Counter to record the steps
  val counter = Module(new snax.utils.BasicCounter(8) {
    override val desiredName = "ElementwiseAddCounter"
  })

  val csrOperandCount = ext_csr_i(0)(7, 0)
  val operandCount    = Mux(csrOperandCount === 0.U, 1.U(8.W), csrOperandCount)
  counter.io.ceil  := operandCount
  counter.io.reset := ext_start_i
  counter.io.tick  := ext_data_i.fire

  // The register holds the partial sum and then the completed output.
  val regs        = RegInit(
    VecInit(
      Seq.fill(elementPerVector)(0.U(elementWidth.W))
    )
  )
  val outputValid = RegInit(false.B)

  val input_data = ext_data_i.bits.asTypeOf(
    Vec(elementPerVector, UInt(elementWidth.W))
  )

  val nextSum = Wire(Vec(elementPerVector, UInt(elementWidth.W)))
  for (i <- 0 until elementPerVector) {
    nextSum(i) := Mux(counter.io.value === 0.U, input_data(i), regs(i) + input_data(i))
  }

  val lastInput  = counter.io.value === (operandCount - 1.U)
  val outputFire = ext_data_o.fire
  val inputFire  = ext_data_i.fire

  when(ext_start_i) {
    regs        := VecInit(Seq.fill(elementPerVector)(0.U(elementWidth.W)))
    outputValid := false.B
  }.elsewhen(inputFire) {
    regs        := nextSum
    outputValid := lastInput
  }.elsewhen(outputFire) {
    regs        := VecInit(Seq.fill(elementPerVector)(0.U(elementWidth.W)))
    outputValid := false.B
  }

  ext_data_o.bits  := Cat(regs.reverse)
  ext_data_o.valid := outputValid
  ext_data_i.ready := !outputValid || ext_data_o.ready
  ext_busy_o       := (counter.io.value =/= 0.U) || outputValid
}
