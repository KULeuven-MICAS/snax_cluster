// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.csr_manager

import snax.utils._

import chisel3._
import chisel3.util._

// Scala main function for generating CsrManager system verilog file
object CsrManagerGen {
  def main(args: Array[String]) : Unit = {
    val outPath = args.headOption.getOrElse("../../target/snitch_cluster/generated/.")
    emitVerilog(
      new CsrManager(
        csrNumReadWrite = 19,
        csrNumReadOnly = 2,
        csrAddrWidth = 32,
        csrModuleTagName = "snax_streamer_gemmX_csrman_"
      ),
      Array("--target-dir", outPath)
    )
  }
}
