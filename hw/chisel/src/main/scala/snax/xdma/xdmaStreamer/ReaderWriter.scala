package snax.xdma.xdmaStreamer

import chisel3._
import chisel3.util._

import snax.utils._
import snax.xdma.CommonCells._
import snax.xdma.DesignParams._

// ReaderWriter is the module that has a reader port and writer port, but they share one TCDM interface.
// This is suitable for the case that the throughput is not high.

class ReaderWriter(
    readerParam: ReaderWriterParam,
    writerParam: ReaderWriterParam,
    clusterName: String = "unnamed_cluster"
) extends Module
    with RequireAsyncReset {

  override val desiredName = s"${clusterName}_xdma_ReaderWriter"

  val io = IO(new ReaderWriterIO(readerParam, writerParam))

  // Reader
  val reader = Module(
    new Reader(
      readerParam,
      clusterName = s"${clusterName}_RWReader"
    )
  )

  reader.io.aguCfg := io.readerInterface.aguCfg
  reader.io.readerwriterCfg := io.readerInterface.readerwriterCfg
  reader.io.data <> io.readerInterface.data
  reader.io.start := io.readerInterface.start
  io.readerInterface.busy := reader.io.busy
  io.readerInterface.bufferEmpty := reader.io.bufferEmpty

  // Writer
  val writer = Module(
    new Writer(
      writerParam,
      clusterName = s"${clusterName}_RWWriter"
    )
  )

  writer.io.aguCfg := io.writerInterface.aguCfg
  writer.io.readerwriterCfg := io.writerInterface.readerwriterCfg
  writer.io.data <> io.writerInterface.data
  writer.io.start := io.writerInterface.start
  io.writerInterface.busy := writer.io.busy
  io.writerInterface.bufferEmpty := writer.io.bufferEmpty

  // Both reader and writer share the same Request interface
  val readerwriterMux = Seq.fill(readerParam.tcdmParam.numChannel)(
    Module(
      new MuxDecoupled(
        new TcdmReq(
          readerParam.tcdmParam.addrWidth,
          readerParam.tcdmParam.dataWidth
        ),
        2
      )
    )
  )

  // Writer is put at 0th input
  readerwriterMux.zip(writer.io.tcdmReq).foreach { case (mux, writerReq) =>
    mux.io.in(0) <> writerReq
  }

  // Reader is put at 1st input
  readerwriterMux.zip(reader.io.tcdmReq).foreach { case (mux, readerReq) =>
    mux.io.in(1) <> readerReq
  }

  // Connect the DecoupledMux to the TCDM interface
  readerwriterMux.zip(io.readerInterface.tcdmReq).foreach {
    case (mux, tcdmReq) => tcdmReq <> mux.io.out
  }

  // Channel Selection Logic
  val sel = Mux(writer.io.tcdmReq.map(_.valid).reduce(_ || _), 0.U, 1.U)
  readerwriterMux.foreach(_.io.sel := sel)

  // Connect the response from TCDM to the reader
  io.readerInterface.tcdmRsp <> reader.io.tcdmRsp
}

object ReaderWriterEmitter extends App {
  println(
    getVerilogString(
      new ReaderWriter(new ReaderWriterParam, new ReaderWriterParam)
    )
  )
}
