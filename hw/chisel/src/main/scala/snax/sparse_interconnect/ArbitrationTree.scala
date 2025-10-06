package snax.sparse_interconnect

import chisel3._
import chisel3.util._

/** Performs forward arbitration for a single output memory bank. */
class ArbitrationTree(NumInp: Int, addrWidth: Int, dataWidth: Int, strbWidth: Int, userWidth: Int) extends Module {
  val io = IO(new Bundle {
    val tcdmReqs = Vec(NumInp, Flipped(Decoupled(new TcdmReq(addrWidth, dataWidth, strbWidth, userWidth))))
    val tcdmRsp  = Decoupled(new TcdmRsp(dataWidth))
    val memReq   = Decoupled(new TcdmReq(addrWidth, dataWidth, strbWidth, userWidth))
    val memRsp   = Flipped(Decoupled(new TcdmRsp(dataWidth)))
  })

  // Default values for tcdm requests
  for (i <- 0 until NumInp) {
    io.tcdmReqs(i).ready := false.B
  }

  // Default values for memory request
  io.memReq.valid := false.B
  io.memReq.bits  := DontCare

  // Arbitration and request routing
  val arbiter = Module(new RoundRobinArbiter(NumInp))
  arbiter.io.requests        := io.tcdmReqs.map { req => req.valid }
  arbiter.io.selection.ready := io.memReq.ready

  // Propagate the request to the memory bank
  when(arbiter.io.selection.valid) {
    io.memReq.bits                               := io.tcdmReqs(arbiter.io.selection.bits).bits
    io.memReq.valid                              := true.B
    // Return ready signal to the original requestor
    io.tcdmReqs(arbiter.io.selection.bits).ready := io.memReq.ready
  }

  // Simply Return Response:
  io.tcdmRsp <> io.memRsp

}
