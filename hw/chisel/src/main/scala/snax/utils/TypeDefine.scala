package snax.utils

import chisel3._

// simplified tcdm interface
class RegReq(addrWidth: Int, dataWidth: Int) extends Bundle {
  val addr  = UInt(addrWidth.W)
  val write = Bool()
  val data  = UInt(dataWidth.W)
  val strb  = UInt((dataWidth / 8).W)
}

class RegRsp(dataWidth: Int) extends Bundle {
  val data = UInt(dataWidth.W)
}

// How close a TCDM request's requester is to stalling the engine it serves, carried in the
// request's `tcdm_priority` user field: 0 has slack, 3 stalls it now. The interconnect serves the
// highest grade a bank sees first.
object TcdmUrgency {
  val width      = 2
  // The grade of a requester built with a static high priority (`higherStaticPriority`): above a
  // channel with slack, equal to one whose FIFO is a beat from stalling, below one stalled now.
  val staticHigh = 2
}

class SparseTCDMReq(addrWidth: Int, dataWidth: Int) extends RegReq(addrWidth, dataWidth) {
  val priority = UInt(TcdmUrgency.width.W)
}

class SparseTCDMRsp(dataWidth: Int) extends RegRsp(dataWidth)
