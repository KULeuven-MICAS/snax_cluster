package snax.xdma.xdmaStreamer

import chisel3._
import chisel3.util._

import snax.utils._
import snax.xdma.CommonCells._
import snax.xdma.DesignParams._

// ReaderWriter is the module that has a reader port and writer port, but they share one TCDM interface.
// This is suitable for the case that the throughput is not high.

class ReaderWriter(
    readerparam: ReaderWriterParam,
    writerparam: ReaderWriterParam,
    clusterName: String = "unnamed_cluster"
) extends Module
    with RequireAsyncReset {

  override val desiredName = s"${clusterName}_xdma_ReaderWriter"

  // As they share the same TCDM interface, different number of channel is meaningless
  require(
    readerparam.tcdm_param.numChannel == writerparam.tcdm_param.numChannel
  )

  val io = IO(new Bundle {
    val readercfg = Input(new AddressGenUnitCfgIO(readerparam.agu_param))
    val writercfg = Input(new AddressGenUnitCfgIO(writerparam.agu_param))
    val tcdm_req = Vec(
      readerparam.tcdm_param.numChannel,
      Decoupled(
        new TcdmReq(
          addrWidth = readerparam.tcdm_param.addrWidth,
          tcdmDataWidth = readerparam.tcdm_param.dataWidth
        )
      )
    )
    val tcdm_rsp = Vec(
      readerparam.tcdm_param.numChannel,
      Flipped(
        Valid(new TcdmRsp(tcdmDataWidth = readerparam.tcdm_param.dataWidth))
      )
    )
    val readerdata = Decoupled(
      UInt(
        (readerparam.tcdm_param.dataWidth * readerparam.tcdm_param.numChannel).W
      )
    )
    val writerdata = Flipped(
      Decoupled(
        UInt(
          (writerparam.tcdm_param.dataWidth * writerparam.tcdm_param.numChannel).W
        )
      )
    )
    // The signal to control which byte is read from or written to TCDM
    val readerStrb = Input(UInt((readerparam.tcdm_param.dataWidth / 8).W))
    val writerStrb = Input(UInt((writerparam.tcdm_param.dataWidth / 8).W))
    // The signal trigger the start of Address Generator. The non-empty of address generator will cause data requestor to read the data
    val readerStart = Input(Bool())
    val writerStart = Input(Bool())
    // The module is busy if addressgen is busy or fifo in addressgen is not empty
    val readerBusy = Output(Bool())
    val writerBusy = Output(Bool())
    // The data buffer is empty
    val readerBufferEmpty = Output(Bool())
    val writerBufferEmpty = Output(Bool())
  })

  // Reader
  val reader = Module(
    new Reader(
      readerparam,
      clusterName = s"${clusterName}_RWReader"
    )
  )

  reader.io.cfg := io.readercfg
  reader.io.data <> io.readerdata
  reader.io.strb := io.readerStrb
  reader.io.start := io.readerStart
  io.readerBusy := reader.io.busy
  io.readerBufferEmpty := reader.io.bufferEmpty

  // Writer
  val writer = Module(
    new Writer(
      writerparam,
      clusterName = s"${clusterName}_RWWriter"
    )
  )

  writer.io.cfg := io.writercfg
  writer.io.data <> io.writerdata
  writer.io.strb := io.writerStrb
  writer.io.start := io.writerStart
  io.writerBusy := writer.io.busy
  io.writerBufferEmpty := writer.io.bufferEmpty

  // Both reader and writer share the same Request interface
  val readerwriterArbiter = Seq.fill(readerparam.tcdm_param.numChannel)(
    Module(
      new Arbiter(
        new TcdmReq(
          readerparam.tcdm_param.addrWidth,
          readerparam.tcdm_param.dataWidth
        ),
        2
      )
    )
  )

  // Writer has the higher priority, and reversely connected to avoid contention when some channels are turned off
  readerwriterArbiter.reverse.zip(writer.io.tcdm_req).foreach {
    case (arbiter, writerReq) => arbiter.io.in(0) <> writerReq
  }

  // Reader has the lower priority
  readerwriterArbiter.zip(reader.io.tcdm_req).foreach {
    case (arbiter, readerReq) => arbiter.io.in(1) <> readerReq
  }

  // Connect the arbiter to the TCDM interface
  readerwriterArbiter.zip(io.tcdm_req).foreach { case (arbiter, tcdmReq) =>
    tcdmReq <> arbiter.io.out
  }

  // Connect the response from TCDM to the reader
  io.tcdm_rsp <> reader.io.tcdm_rsp
}

object ReaderWriterEmitter extends App {
  println(getVerilogString(new ReaderWriter(new ReaderWriterParam, new ReaderWriterParam)))
}
