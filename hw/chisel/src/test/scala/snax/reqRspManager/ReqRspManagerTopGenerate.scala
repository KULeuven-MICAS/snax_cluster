package snax.reqRspManager

import chisel3._

import org.scalatest.flatspec.AnyFlatSpec
import snax.reqRspManager.ReqRspManager

class ReqRspManagerTopGenerate extends AnyFlatSpec {

  emitVerilog(
    new ReqRspManager(
      W32ReqRspManagerTestParameters.numReadWriteReg,
      W32ReqRspManagerTestParameters.numReadOnlyReg,
      W32ReqRspManagerTestParameters.addrWidth
    ),
    Array("--target-dir", "generated/reqRspManager")
  )
}
