package snax.xdma.xdmaFrontend

import chisel3._
import chisel3.util._

import snax.utils._

import snax.utils.DecoupledCut._
import snax.utils.BitsConcat._

import snax.readerWriter.{
  AddressGenUnitCfgIO,
  ReaderWriterCfgIO,
  Reader,
  Writer,
  ReaderWriterParam
}
import snax.xdma.DesignParams._
import snax.DataPathExtension._
import snax.xdma.io._

class XDMADataPath(
    readerParam: XDMAParam,
    writerParam: XDMAParam,
    clusterName: String = "unnamed_cluster"
) extends Module
    with RequireAsyncReset {

  override val desiredName = s"${clusterName}_xdma_datapath"

  val io = IO(new Bundle {
    // All config signal for reader and writer
    val readerCfg = Input(new XDMACfgIO(readerParam))
    val writerCfg = Input(new XDMACfgIO(writerParam))

    // Two start signal will inform the new cfg is available, trigger agu, and inform all extension that a stream is coming
    val readerStart = Input(Bool())
    val writerStart = Input(Bool())
    // Two busy signal only go down if a stream fully passthrough the reader / writter.
    // reader_busy_o signal == 0 indicates that the reader side is available for next task
    val readerBusy = Output(Bool())
    // writer_busy_o signal == 0 indicates that the writer side is available for next task
    val writerBusy = Output(Bool())

    // TCDM request and response signal
    val tcdmReader = new Bundle {
      val req = Vec(
        readerParam.rwParam.tcdmParam.numChannel,
        Decoupled(
          new TcdmReq(
            readerParam.rwParam.tcdmParam.addrWidth,
            readerParam.rwParam.tcdmParam.dataWidth
          )
        )
      )
      val rsp = Vec(
        readerParam.rwParam.tcdmParam.numChannel,
        Flipped(
          Valid(
            new TcdmRsp(tcdmDataWidth = readerParam.rwParam.tcdmParam.dataWidth)
          )
        )
      )
    }
    val tcdmWriter = new Bundle {
      val req = Vec(
        writerParam.rwParam.tcdmParam.numChannel,
        Decoupled(
          new TcdmReq(
            writerParam.rwParam.tcdmParam.addrWidth,
            writerParam.rwParam.tcdmParam.dataWidth
          )
        )
      )
    }

    // The data for the cluster-level in/out
    // Cluster-level input -> fromRemote <> Writer
    // Cluster-level output -> toRemote <> Reader
    val remoteXDMAData = new Bundle {
      val fromRemote = Flipped(
        Decoupled(
          UInt(
            (writerParam.rwParam.tcdmParam.dataWidth * writerParam.rwParam.tcdmParam.numChannel).W
          )
        )
      )
      val fromRemoteAccompaniedCfg = Output(
        new XDMADataPathCfgIO(
          axiParam = writerParam.axiParam,
          crossClusterParam = writerParam.crossClusterParam
        )
      )

      val toRemote =
        Decoupled(
          UInt(
            (readerParam.rwParam.tcdmParam.dataWidth * readerParam.rwParam.tcdmParam.numChannel).W
          )
        )
      val toRemoteAccompaniedCfg = Output(
        new XDMADataPathCfgIO(
          axiParam = readerParam.axiParam,
          crossClusterParam = readerParam.crossClusterParam
        )
      )
    }
  })

  val reader = Module(
    new Reader(readerParam.rwParam, moduleNamePrefix = clusterName)
  )
  val writer = Module(
    new Writer(writerParam.rwParam, moduleNamePrefix = clusterName)
  )

  // Connect TCDM memory to reader and writer
  reader.io.tcdmReq <> io.tcdmReader.req
  reader.io.tcdmRsp <> io.tcdmReader.rsp
  writer.io.tcdmReq <> io.tcdmWriter.req

  // Connect the wire (ctrl plane)
  reader.io.aguCfg := io.readerCfg.aguCfg
  reader.io.readerwriterCfg := io.readerCfg.readerwriterCfg
  reader.io.start := io.readerStart
  // reader_busy_o is connected later as the busy signal from the signal is needed

  writer.io.aguCfg := io.writerCfg.aguCfg
  writer.io.readerwriterCfg := io.writerCfg.readerwriterCfg
  writer.io.start := io.writerStart
  // writer_busy_o is connected later as the busy signal from the signal is needed

  // Connect the extension
  // Reader Side
  val readerDataAfterExtension = Wire(chiselTypeOf(reader.io.data))

  val readerExtensions = Module(
    new DataPathExtensionHost(
      readerParam.extParam,
      dataWidth =
        readerParam.rwParam.tcdmParam.dataWidth * readerParam.rwParam.tcdmParam.numChannel,
      headCut = false,
      tailCut = false,
      halfCut = false,
      moduleNamePrefix = clusterName
    )
  )
  readerExtensions.io.data.in <> reader.io.data
  readerExtensions.io.data.out <> readerDataAfterExtension
  readerExtensions.io.connectCfgWithList(io.readerCfg.extCfg)
  readerExtensions.io.start := io.readerStart
  io.readerBusy := reader.io.busy | (~reader.io.bufferEmpty) | readerExtensions.io.busy

  // Writer side
  val writerDataBeforeExtension = Wire(chiselTypeOf(writer.io.data))

  val writerExtensions = Module(
    new DataPathExtensionHost(
      writerParam.extParam,
      dataWidth =
        writerParam.rwParam.tcdmParam.dataWidth * writerParam.rwParam.tcdmParam.numChannel,
      headCut = false,
      tailCut = false,
      halfCut = false,
      moduleNamePrefix = clusterName
    )
  )

  writerExtensions.io.data.in <> writerDataBeforeExtension
  writerExtensions.io.data.out <> writer.io.data
  writerExtensions.io.connectCfgWithList(io.writerCfg.extCfg)
  writerExtensions.io.start := io.writerStart
  io.writerBusy := writer.io.busy | (~writer.io.bufferEmpty) | writerExtensions.io.busy

  // The following code only tackle with readerDataAfterExtension and writerDataBeforeExtension: they should be either loopbacked or forwarded to the external interface (cluster_data_i / cluster_data_o)
  val readerDemux = Module(
    new DemuxDecoupled(
      chiselTypeOf(readerDataAfterExtension.bits),
      numOutput = 2
    ) {
      override def desiredName = clusterName + "_xdma_datapath_demux"
    }
  )
  val writerMux = Module(
    new MuxDecoupled(
      chiselTypeOf(writerDataBeforeExtension.bits),
      numInput = 2
    ) {
      override def desiredName = clusterName + "_xdma_datapath_mux"
    }
  )

  readerDemux.io.sel := io.readerCfg.loopBack
  writerMux.io.sel := io.writerCfg.loopBack
  readerDataAfterExtension <> readerDemux.io.in
  writerMux.io.out <> writerDataBeforeExtension

  readerDemux.io.out(1) <> writerMux.io.in(1)
  readerDemux.io.out(0) <> io.remoteXDMAData.toRemote
  writerMux.io.in(0) <> io.remoteXDMAData.fromRemote

  // Connect the AccompaniedCfg signal
  io.remoteXDMAData.fromRemoteAccompaniedCfg.convertFromXDMACfgIO(
    loopBack = io.writerCfg.loopBack,
    cfg = io.writerCfg,
    isReaderSide = false
  )
  io.remoteXDMAData.toRemoteAccompaniedCfg.convertFromXDMACfgIO(
    loopBack = io.readerCfg.loopBack,
    cfg = io.readerCfg,
    isReaderSide = true
  )

  io.remoteXDMAData.fromRemoteAccompaniedCfg.readyToTransmit := Mux(
    io.writerCfg.loopBack,
    false.B,
    writer.io.busy
  )

  io.remoteXDMAData.toRemoteAccompaniedCfg.readyToTransmit := Mux(
    io.readerCfg.loopBack,
    false.B,
    reader.io.busy
  )
}

// Below is the class to determine if chisel generate Verilog correctly

object XDMADataPathEmitter extends App {
  emitVerilog(
    new XDMADataPath(
      readerParam = new XDMAParam(
        axiParam = new AXIParam,
        crossClusterParam = new CrossClusterParam,
        rwParam = new ReaderWriterParam,
        extParam = Seq()
      ),
      writerParam = new XDMAParam(
        axiParam = new AXIParam,
        crossClusterParam = new CrossClusterParam,
        rwParam = new ReaderWriterParam,
        extParam = Seq()
      )
    ),
    Array("--target-dir", "generated")
  )

}
