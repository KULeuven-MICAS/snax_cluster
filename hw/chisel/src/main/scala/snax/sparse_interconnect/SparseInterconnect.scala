package snax.sparse_interconnect

import chisel3._
import chisel3.util._
import snax.sparse_interconnect.SparseConfig

class SparseInterconnect(
  NumInp:        Int,
  NumOut:        Int,
  memAddrWidth:  Int,
  tcdmAddrWidth: Int,
  dataWidth:     Int,
  strbWidth:     Int,
  priorityWidth: Int,
  sparse_config: SparseConfig
) extends Module {
  val io = IO(new Bundle {
    val tcdmReqs = Vec(NumInp, Flipped(Decoupled(new TcdmReq(tcdmAddrWidth, dataWidth, strbWidth, priorityWidth))))
    val tcdmRsps = Vec(NumInp, Decoupled(new TcdmRsp(dataWidth)))
    val memReqs  = Vec(NumOut, Decoupled(new TcdmReq(memAddrWidth, dataWidth, strbWidth, priorityWidth)))
    val memRsps  = Vec(NumOut, Flipped(Decoupled(new TcdmRsp(dataWidth))))
  })

  // === Bank Selection ===

  // Address construction:
  // [bank addr | bank offset | byte offset]
  val byteOffsetWidth = log2Ceil(dataWidth / 8)
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
      new ArbitrationTree(sparse_config.inputsPerBank, memAddrWidth, dataWidth, strbWidth, priorityWidth)
    )
  )

  // Default ready signals to false
  io.tcdmReqs.foreach(_.ready := false.B)

  for (out <- 0 until NumOut) {

    // Connect the inputs
    for (in <- 0 until sparse_config.inputsPerBank) {
      val global_in = sparse_config.get_global_idx(in, out)
      // Connect the request to the arbiter
      arbiters(out).io.tcdmReqs(in).bits <> io.tcdmReqs(global_in).bits
      // Only send the relevant part of the address
      arbiters(out).io.tcdmReqs(in).bits.addr :=
        io.tcdmReqs(global_in).bits.addr(tcdmAddrWidth - 1, bankSelectWidth + byteOffsetWidth)
      // Valid only on correct arbiter
      arbiters(out).io.tcdmReqs(in).valid     := io.tcdmReqs(global_in).valid && (bankSelect(global_in) === out.U)
      // Reverse routing of the ready signal
      when(bankSelect(global_in) === out.U) {
        io.tcdmReqs(global_in).ready := arbiters(out).io.tcdmReqs(in).ready
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

    val (port_idx, index) = sparse_config.getPortAndIndex(in)
    val port              = sparse_config.ports(port_idx)
    // Print for debug
    println(
      s"Routing outputs for input $in: port $port_idx, index $index, sparse_port_access_granularity ${port.access_granularity}"
    )

    // Mux the response based on the previous bank selection
    for (sparse_out <- 0 until NumOut / port.access_granularity) {
      val out = sparse_out * port.access_granularity + (index % port.access_granularity)
      println(s"Routing outputs for input $in: port $port_idx, index $index, sparse_out $sparse_out, out $out")
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

    val NumInp        = parsedArgs.get("NumInp").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("NumInp argument is required")
    }
    val NumOut        = parsedArgs.get("NumOut").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("NumOut argument is required")
    }
    val memAddrWidth  = parsedArgs.get("memAddrWidth").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("memAddrWidth argument is required")
    }
    val tcdmAddrWidth = parsedArgs.get("tcdmAddrWidth").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("tcdmAddrWidth argument is required")
    }
    val dataWidth     = parsedArgs.get("dataWidth").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("dataWidth argument is required")
    }
    val strbWidth     = parsedArgs.get("strbWidth").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("strbWidth argument is required")
    }
    val priorityWidth = parsedArgs.get("priorityWidth").map(_.toInt).getOrElse {
      throw new IllegalArgumentException("priorityWidth argument is required")
    }

    // Parse the sparse configuration string
    val sparseConfigString = parsedArgs.getOrElse(
      "sparseConfig", {
        throw new IllegalArgumentException("sparseConfig argument is required")
      }
    )

    println(s"SparseConfig String: $sparseConfigString")
    val sparseConfig = SparseConfig.parseSparseConfig(sparseConfigString)
    println(s"Parsed SparseConfig: $sparseConfig")

    emitVerilog(
      new SparseInterconnect(
        NumInp,
        NumOut,
        memAddrWidth,
        tcdmAddrWidth,
        dataWidth,
        strbWidth,
        priorityWidth,
        sparseConfig
      ),
      Array("--target-dir", outPath)
    )

  }
}
