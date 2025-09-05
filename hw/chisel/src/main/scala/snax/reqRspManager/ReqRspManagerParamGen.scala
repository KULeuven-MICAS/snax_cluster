// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.reqRspManager

import snax.utils._

import chisel3._
import chisel3.util._

// Scala main function for generating ReqRspManager system verilog file
object ReqRspManagerGen {
  def main(args: Array[String]) : Unit = {
    val outPath = args.headOption.getOrElse("../../target/snitch_cluster/generated/.")
    emitVerilog(
      new ReqRspManager(
        numReadWriteReg = 7,
        numReadOnlyReg = 2,
        addrWidth = 32,
        moduleTagName = "snax_versacore_reqrspman_"
      ),
      Array("--target-dir", outPath)
    )
  }
}
