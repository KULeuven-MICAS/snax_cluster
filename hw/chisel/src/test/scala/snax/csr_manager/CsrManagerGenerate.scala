package snax.reqRspManager

import chisel3._

import org.scalatest.flatspec.AnyFlatSpec
import snax.reqRspManager.ReqRspManager

class CsrManagerTopGenerate extends AnyFlatSpec {

  emitVerilog(
    new ReqRspManager(
      CsrManagerTestParameters.csrNumReadWrite,
      CsrManagerTestParameters.csrNumReadOnly,
      CsrManagerTestParameters.csrAddrWidth
    ),
    Array("--target-dir", "generated/reqRspManager")
  )

}
