// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi <xiaoling.yi@kuleuven.be>

package snax.DataPathExtension

import chisel3._
import chisel3.util._

class IntlowToInthighPE(in_elementWidth: Int = 4, out_elementWidth: Int = 8) extends Module {
  val io = IO(new Bundle {
    val in  = Input(SInt(in_elementWidth.W))
    val out = Output(SInt(out_elementWidth.W))
  })

  // Sign-extend the low-bit input to high bits
  io.out := io.in.asSInt.pad(out_elementWidth)
}

class IntlowToInthighConverter(dataWidth: Int = 512, in_elementWidth: Int = 4, out_elementWidth: Int = 8)(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {
  require(
    dataWidth % in_elementWidth == 0,
    s"dataWidth ($dataWidth) must be multiple of in_elementWidth ($in_elementWidth)"
  )
  require(
    dataWidth % out_elementWidth == 0,
    s"dataWidth ($dataWidth) must be multiple of out_elementWidth ($out_elementWidth)"
  )
  require(
    in_elementWidth < out_elementWidth,
    s"in_elementWidth ($in_elementWidth) must be less than out_elementWidth ($out_elementWidth)"
  )

  val numPEs = dataWidth / out_elementWidth

  val peArray = Seq.fill(numPEs) {
    Module(new IntlowToInthighPE(in_elementWidth, out_elementWidth) {
      override def desiredName = extensionParam.moduleName + "_intlow_to_inthigh_pe"
    })
  }

  // Correct wire declarations
  val pe_inputs  = Wire(Vec(numPEs, SInt(in_elementWidth.W)))
  val pe_outputs = Wire(Vec(numPEs, SInt(out_elementWidth.W)))

  // Extract each in_elementWidth-bit block from ext_data_i
  for (i <- 0 until numPEs) {
    pe_inputs(i) := ext_data_i.bits(in_elementWidth * (i + 1) - 1, in_elementWidth * i).asSInt
  }

  // Connect PEs
  for (i <- 0 until numPEs) {
    peArray(i).io.in := pe_inputs(i)
    pe_outputs(i)    := peArray(i).io.out
  }

  // Combine outputs into ext_data_o
  ext_data_o.bits := Cat(pe_outputs.reverse).asTypeOf(ext_data_o.bits)

  // Pass through valid/ready signals
  ext_data_o.valid := ext_data_i.valid
  ext_data_i.ready := ext_data_o.ready
  ext_busy_o       := 0.U(1.W)
}

class HasIntlowToInthighConverter(dataWidth: Int = 512, in_elementWidth: Int = 4, out_elementWidth: Int = 8)
    extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = s"IntlowToInthighConverter_${dataWidth}",
      userCsrNum = 0,
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): IntlowToInthighConverter =
    Module(
      new IntlowToInthighConverter(dataWidth, in_elementWidth, out_elementWidth) {
        override def desiredName = clusterName + namePostfix
      }
    )

}
