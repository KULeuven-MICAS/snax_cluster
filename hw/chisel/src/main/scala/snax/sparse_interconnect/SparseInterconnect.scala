package snax.sparse_interconnect

import chisel3._
import chisel3.util._

class SparseInterconnect(NumInp: Int, NumOut: Int, addrWidth: Int, dataWidth: Int, strbWidth: Int, userWidth: Int)
    extends Module {
  val io = IO(new Bundle {
    val tcdmReqs = Vec(NumInp, Flipped(Decoupled(new TcdmReq(addrWidth, dataWidth, strbWidth, userWidth))))
    val tcdmRsps = Vec(NumInp, Decoupled(new TcdmRsp(dataWidth)))
    val memReqs  = Vec(NumOut, Decoupled(new TcdmReq(addrWidth, dataWidth, strbWidth, userWidth)))
    val memRsps  = Vec(NumOut, Flipped(Decoupled(new TcdmRsp(dataWidth))))
  })

  // === Bank Selection ===

  // Address construction:
  // [bank addr | bank offset | byte offset]
  val byteOffsetWidth = log2Ceil(addrWidth / 8)
  val bankSelectWidth = log2Ceil(NumOut)

  // Determines the bank selection of the requests
  val bankSelect = Wire(Vec(NumInp, UInt(log2Ceil(NumOut).W)))
  for (i <- 0 until NumInp) {
    bankSelect(i) := io.tcdmReqs(i).bits.addr(bankSelectWidth + byteOffsetWidth - 1, byteOffsetWidth)
  }

  // Determines the success of each request
  val reqFire = Wire(Vec(NumInp, Bool()))
  reqFire := io.tcdmReqs.map(req => req.fire)

  // === Forward Request Routing (tcdm -> bank) ===

  // one arbitration module per output memory bank
  val arbiters = Seq.fill(NumOut)(
    Module(
      new ArbitrationTree(NumInp, addrWidth, dataWidth, strbWidth, userWidth)
    )
  )

  // Default ready signals to false
  io.tcdmReqs.foreach(_.ready := false.B)

  for (out <- 0 until NumOut) {

    // Connect the inputs
    for (in <- 0 until NumInp) {
      // Connect the request to the arbiter
      arbiters(out).io.tcdmReqs(in).bits <> io.tcdmReqs(in).bits
      // Only send the relevant part of the address
      arbiters(out).io.tcdmReqs(in).bits.addr :=
        io.tcdmReqs(in).bits.addr(addrWidth - 1, bankSelectWidth + byteOffsetWidth)
      // Valid only on correct arbiter
      arbiters(out).io.tcdmReqs(in).valid     := io.tcdmReqs(in).valid && (bankSelect(in) === out.U)
      // Reverse routing of the ready signal
      when(bankSelect(in) === out.U) {
        io.tcdmReqs(in).ready := arbiters(out).io.tcdmReqs(in).ready
      }
    }

    // Connect to the memory output ports.
    arbiters(out).io.memReq <> io.memReqs(out)
    arbiters(out).io.memRsp <> io.memRsps(out)

    // default value for tcdmrsp ready
    arbiters(out).io.tcdmRsp.ready := false.B

  }

  // === Response Routing ===

  // response arbitration is a simple mux based on bank selection in previous cycle
  // this assumes that the memory banks have a 1-cycle latency
  val prevBankRequest = RegNext(bankSelect)
  val prevReqFire     = RegNext(reqFire)

  for (in <- 0 until NumInp) {
    io.tcdmRsps(in).valid := false.B
    io.tcdmRsps(in).bits  := DontCare

    // Mux the response based on the previous bank selection
    for (out <- 0 until NumOut) {
      when(prevBankRequest(in) === out.U && prevReqFire(in)) {
        io.tcdmRsps(in) <> arbiters(out).io.tcdmRsp
      }
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
