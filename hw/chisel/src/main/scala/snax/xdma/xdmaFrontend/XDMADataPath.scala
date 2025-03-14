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

// The ReaderWriterCfg Class that used for interface between local Datapath and DMA Ctrl
// The length of addresses is the short version, which is just enough to reside TCDM
class XDMADataPathCfgIO(param: XDMADataPathParam) extends Bundle {
  val aguCfg =
    new AddressGenUnitCfgIO(param =
      param.rwParam.aguParam
    ) // Buffered within AGU
  val readerwriterCfg = new ReaderWriterCfgIO(param.rwParam)
  val loopBack = Bool()

  val extCfg = if (param.extParam.length != 0) {
    Vec(
      param.extParam.map { i => i.extensionParam.userCsrNum }.reduce(_ + _) + 1,
      UInt(32.W)
    ) // The total csr required by all extension + 1 for the bypass signal
  } else Vec(0, UInt(32.W))

  def connectWithList(
      csrList: IndexedSeq[UInt]
  ): IndexedSeq[UInt] = {
    var remaincsrList = csrList
    remaincsrList = aguCfg.connectWithList(remaincsrList)
    remaincsrList = readerwriterCfg.connectWithList(remaincsrList)
    extCfg := remaincsrList.take(extCfg.length)
    remaincsrList = remaincsrList.drop(extCfg.length)
    remaincsrList
  }
}

class XDMADataPath(
    readerparam: XDMADataPathParam,
    writerparam: XDMADataPathParam,
    clusterName: String = "unnamed_cluster"
) extends Module
    with RequireAsyncReset {

  override val desiredName = s"${clusterName}_xdma_datapath"

  val io = IO(new Bundle {
    // All config signal for reader and writer
    val readerCfg = Input(new XDMADataPathCfgIO(readerparam))
    val writerCfg = Input(new XDMADataPathCfgIO(writerparam))

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
        readerparam.rwParam.tcdmParam.numChannel,
        Decoupled(
          new TcdmReq(
            readerparam.rwParam.tcdmParam.addrWidth,
            readerparam.rwParam.tcdmParam.dataWidth
          )
        )
      )
      val rsp = Vec(
        readerparam.rwParam.tcdmParam.numChannel,
        Flipped(
          Valid(
            new TcdmRsp(tcdmDataWidth = readerparam.rwParam.tcdmParam.dataWidth)
          )
        )
      )
    }
    val tcdmWriter = new Bundle {
      val req = Vec(
        writerparam.rwParam.tcdmParam.numChannel,
        Decoupled(
          new TcdmReq(
            writerparam.rwParam.tcdmParam.addrWidth,
            writerparam.rwParam.tcdmParam.dataWidth
          )
        )
      )
    }

    // The data for the cluster-level in/out
    // Cluster-level input <> Writer
    // Cluster-level output <> Reader
    val remoteXDMAData = new Bundle {
      val fromRemote = Flipped(
        Decoupled(
          UInt(
            (writerparam.rwParam.tcdmParam.dataWidth * writerparam.rwParam.tcdmParam.numChannel).W
          )
        )
      )
      val toRemote =
        Decoupled(
          UInt(
            (writerparam.rwParam.tcdmParam.dataWidth * writerparam.rwParam.tcdmParam.numChannel).W
          )
        )
    }
  })

  val reader = Module(
    new Reader(readerparam.rwParam, moduleNamePrefix = clusterName)
  )
  val writer = Module(
    new Writer(writerparam.rwParam, moduleNamePrefix = clusterName)
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
      readerparam.extParam,
      dataWidth =
        readerparam.rwParam.tcdmParam.dataWidth * readerparam.rwParam.tcdmParam.numChannel,
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
      writerparam.extParam,
      dataWidth =
        writerparam.rwParam.tcdmParam.dataWidth * writerparam.rwParam.tcdmParam.numChannel,
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
}

// Below is the class to determine if chisel generate Verilog correctly

object XDMADataPathEmitter extends App {
  println(
    getVerilogString(
      new XDMADataPath(
        readerparam = new XDMADataPathParam(
          rwParam = new ReaderWriterParam,
          extParam = Seq()
        ),
        writerparam = new XDMADataPathParam(
          rwParam = new ReaderWriterParam,
          extParam = Seq()
        )
      )
    )
  )
}
