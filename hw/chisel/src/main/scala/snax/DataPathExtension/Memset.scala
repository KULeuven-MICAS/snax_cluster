package snax.DataPathExtension

import chisel3._

class HasMemset extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "Memset",
      userCsrNum = 1,
      dataWidth  = 512
    )

  def instantiate(clusterName: String): Memset =
    Module(new Memset {
      override def desiredName = clusterName + namePostfix
    })
}

class Memset()(implicit extensionParam: DataPathExtensionParam) extends DataPathExtension {
  // The CSR is a 32-bit PATTERN tiled across the beat, not a byte, so one
  // mechanism covers INT8, FP16, BF16, FP32 and INT32 constants without the
  // extension knowing any of those formats. A byte fill is the pattern with
  // all four lanes equal.
  val out = WireInit(
    VecInit(Seq.fill(extensionParam.dataWidth / 32)(ext_csr_i(0)))
  )
  ext_data_i.ready := ext_data_o.ready
  ext_data_o.valid := ext_data_i.valid
  ext_data_o.bits  := out.asUInt
  ext_busy_o       := false.B
}

object MemsetEmitter extends App {
  println(
    getVerilogString(
      new Memset()(
        new DataPathExtensionParam(
          moduleName = "Memset",
          userCsrNum = 1,
          dataWidth  = 512
        )
      )
    )
  )
}
