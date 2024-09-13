package snax.xdma.xdmaExtension

import chisel3._
import chisel3.util._
import snax.xdma.DesignParams.DatapathExtensionParam

object HasVerilogMemset extends HasDatapathExtension {
  implicit val extensionParam: DatapathExtensionParam = new DatapathExtensionParam(
    moduleName = "VerilogMemset",
    userCsrNum = 1,
    dataWidth = 512
  )
  def instantiate(clusterName: String): SystemVerilogDatapathExtension = Module(
    new SystemVerilogDatapathExtension(
      topmodule = "VerilogMemset",
      filelist = Seq("src/main/systemverilog/VerilogMemset/VerilogMemset.sv")
    ) {
      override def desiredName = clusterName + namePostfix
    }
  )
}
