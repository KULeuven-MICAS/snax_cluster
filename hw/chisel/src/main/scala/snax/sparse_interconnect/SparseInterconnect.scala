package snax.sparse_interconnect

import chisel3._
import chisel3.util._

class SparseInterconnect(NumInp: Int, NumOut: Int, addrWidth: Int, dataWidth: Int, strbWidth: Int, userWidth: Int)
    extends Module {
  val io = IO(new Bundle {
    val tcdmReqs = Vec(NumInp, Flipped(Decoupled(new TcdmReq(addrWidth, dataWidth, strbWidth, userWidth))))
    val tcdmRsps = Vec(NumInp, Decoupled(new TcdmRsp(dataWidth)))
    val memReqs  = Vec(NumOut, Decoupled(new MemReq(addrWidth, dataWidth, strbWidth, userWidth)))
    val memRsps  = Vec(NumOut, Flipped(Decoupled(new MemRsp(dataWidth))))
  })

  // Address construction:
  // bank addr - bank offset - byte offset
  val byteOffsetWidth = log2Ceil(addrWidth / 8)
  val bankSelectWidth = log2Ceil(NumOut)

  // Determine bank selection of the requests
  val bankSelect = Wire(Vec(NumInp, UInt(log2Ceil(NumOut).W)))
  for (i <- 0 until NumInp) {
    bankSelect(i) := io.tcdmReqs(i).bits.addr(bankSelectWidth + byteOffsetWidth - 1, byteOffsetWidth)
  }

  // Registers to track the source of each memory bank's response
  val lastReqSource = RegInit(VecInit(Seq.fill(NumOut)(0.U(log2Ceil(NumInp).W))))
  val lastReqValid  = RegInit(VecInit(Seq.fill(NumOut)(false.B)))

  // Default values for tcdm requests
  for (i <- 0 until NumInp) {
    io.tcdmReqs(i).ready := false.B
  }

  // Default values for memory requests
  for (out <- 0 until NumOut) {
    io.memReqs(out).valid := false.B
    io.memReqs(out).bits  := DontCare
  }

  // Arbitration and request routing
  for (out <- 0 until NumOut) {

    // Collect all valid requests to this bank:
    val validRequests = io.tcdmReqs.zip(bankSelect).map { case (req, bank) =>
      req.valid && (bank === out.U)
    }

    val anyValid: Bool = validRequests.reduce(_ || _)
    // just take the first valid request for now
    val selectedRequest = PriorityEncoder(validRequests)

    // Propagate the request to the memory bank
    when(anyValid) {
      io.memReqs(out).bits      := io.tcdmReqs(selectedRequest).bits
      // The bank and byte offset should be subtracted from the address for memory requests
      io.memReqs(out).bits.addr :=
        io.tcdmReqs(selectedRequest).bits.addr(addrWidth - 1, bankSelectWidth + byteOffsetWidth)
      io.memReqs(out).valid     := true.B

      // Return ready signal to the original requestor
      io.tcdmReqs(selectedRequest).ready := io.memReqs(out).ready
    }

    // Track the source of the request
    when(io.memReqs(out).fire) {
      lastReqSource(out) := selectedRequest
      lastReqValid(out)  := true.B
    }.otherwise(
      lastReqValid(out) := false.B
    )
  }

  // Default Response Routing
  for (inp <- 0 until NumInp) {
    io.tcdmRsps(inp).valid := false.B
    io.tcdmRsps(inp).bits  := DontCare
  }

  for (out <- 0 until NumOut) {
    when(io.memRsps(out).valid && lastReqValid(out)) {
      val source = lastReqSource(out)
      io.tcdmRsps(source).valid := true.B
      io.tcdmRsps(source).bits  := io.memRsps(out).bits
      io.memRsps(out).ready     := io.tcdmRsps(source).ready
    }.otherwise {
      io.memRsps(out).ready := false.B
    }
  }

}

object SparseInterconnectGen {
  def main(args: Array[String]): Unit = {

    val parsedArgs = snax.utils.ArgParser.parse(args)

    val outPath = parsedArgs.getOrElse(
      "hw-target-dir",
      "generated"
    )

    val NumInp    = parsedArgs.get("NumInp").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("NumInp argument is required")
    }
    val NumOut    = parsedArgs.get("NumOut").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("NumOut argument is required")
    }
    val addrWidth = parsedArgs.get("addrWidth").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("addrWidth argument is required")
    }
    val dataWidth = parsedArgs.get("dataWidth").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("dataWidth argument is required")
    }
    val strbWidth = parsedArgs.get("strbWidth").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("strbWidth argument is required")
    }
    val userWidth = parsedArgs.get("userWidth").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("userWidth argument is required")
    }

    emitVerilog(
      new SparseInterconnect(NumInp, NumOut, addrWidth, dataWidth, strbWidth, userWidth),
      Array("--target-dir", outPath)
    )
  }
}
