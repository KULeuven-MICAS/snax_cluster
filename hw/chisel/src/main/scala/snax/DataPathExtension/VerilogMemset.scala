package snax.DataPathExtension

import chisel3._
import chisel3.util._

object HasVerilogMemset extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "VerilogMemset",
      userCsrNum = 1,
      dataWidth = 512
    )
  def instantiate(clusterName: String): SystemVerilogDataPathExtension = Module(
    new SystemVerilogDataPathExtension(
      topmodule = "VerilogMemset",
      filelist = Seq("src/main/systemverilog/VerilogMemset/VerilogMemset.sv")
    ) {
      override def desiredName = clusterName + namePostfix
    }
  )
}
