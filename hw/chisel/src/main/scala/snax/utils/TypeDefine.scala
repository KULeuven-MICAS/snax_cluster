package snax.utils

import chisel3._

// simplified tcdm interface
class RegReq(addrWidth: Int, dataWidth: Int) extends Bundle {
  val addr     = UInt(addrWidth.W)
  val write    = Bool()
  val data     = UInt(dataWidth.W)
  val strb     = UInt((dataWidth / 8).W)
}

class RegRsp(dataWidth: Int) extends Bundle {
  val data = UInt(dataWidth.W)
}

class SparseTCDMReq(addrWidth: Int, dataWidth: Int) extends RegReq(addrWidth, dataWidth) {
  val priority = Bool()
}

class SparseTCDMResp(dataWidth: Int) extends RegRsp(dataWidth)
