package snax.xdma.xdmaFrontend

import chisel3._
import chisel3.util._

import snax.DataPathExtension._
import snax.readerWriter.Reader
import snax.readerWriter.ReaderWriterParam
import snax.readerWriter.Writer
import snax.utils._
import snax.xdma.DesignParams._
import snax.xdma.xdmaIO.XDMADataPathCfgIO
import snax.xdma.xdmaIO.XDMAIntraClusterCfgIO
import snax.xdma.xdmaTop.XDMATopGen.cfgParam

class XDMADataPath(readerParam: XDMAParam, writerParam: XDMAParam, clusterName: String = "unnamed_cluster")
    extends Module
    with RequireAsyncReset {

  override val desiredName = s"${clusterName}_xdma_datapath"

  val io = IO(new Bundle {
    // All config signal for reader and writer
    val readerCfg = Input(new XDMAIntraClusterCfgIO(readerParam))
    val writerCfg = Input(new XDMAIntraClusterCfgIO(writerParam))

    // Two start signal will inform the new cfg is available, trigger agu, and inform all extension that a stream is coming
    val readerStart = Input(Bool())
    val writerStart = Input(Bool())
    // Two busy signal only go down if a stream fully passthrough the reader / writter.
    // reader_busy_o signal == 0 indicates that the reader side is available for next task
    val readerBusy  = Output(Bool())
    // writer_busy_o signal == 0 indicates that the writer side is available for next task
    val writerBusy  = Output(Bool())

    /** O5 and the starvation watchdog, on their way to a software-readable register. Both were dead-ended at the
      * data switch: the hardware knew, and nothing above it could ask.
      */
    val junctionStarved = Output(Bool())
    val junctionCfgErr  = Output(Bool())

    // TCDM request and response signal
    val tcdmReader = new Bundle {
      val req = Vec(
        readerParam.rwParam.tcdmParam.numChannel,
        Decoupled(
          new SparseTCDMReq(
            readerParam.rwParam.tcdmParam.addrWidth,
            readerParam.rwParam.tcdmParam.dataWidth
          )
        )
      )
      val rsp = Vec(
        readerParam.rwParam.tcdmParam.numChannel,
        Flipped(
          Valid(
            new SparseTCDMRsp(dataWidth = readerParam.rwParam.tcdmParam.dataWidth)
          )
        )
      )
    }
    val tcdmWriter = new Bundle {
      val req = Vec(
        writerParam.rwParam.tcdmParam.numChannel,
        Decoupled(
          new SparseTCDMReq(
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
      val fromRemote               = Flipped(
        Decoupled(
          UInt(
            (writerParam.rwParam.tcdmParam.dataWidth * writerParam.rwParam.tcdmParam.numChannel).W
          )
        )
      )
      val fromRemoteAccompaniedCfg = Output(
        new XDMADataPathCfgIO(
          axiParam          = writerParam.axiParam,
          crossClusterParam = writerParam.crossClusterParam
        )
      )

      val toRemote               =
        Decoupled(
          UInt(
            (readerParam.rwParam.tcdmParam.dataWidth * readerParam.rwParam.tcdmParam.numChannel).W
          )
        )
      val toRemoteAccompaniedCfg = Output(
        new XDMADataPathCfgIO(
          axiParam          = readerParam.axiParam,
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
  reader.io.aguCfg          := io.readerCfg.aguCfg
  reader.io.readerwriterCfg := io.readerCfg.readerwriterCfg
  reader.io.start           := io.readerStart
  // reader_busy_o is connected later as the busy signal from the signal is needed

  writer.io.aguCfg          := io.writerCfg.aguCfg
  writer.io.readerwriterCfg := io.writerCfg.readerwriterCfg
  writer.io.start           := io.writerStart
  // writer_busy_o is connected later as the busy signal from the signal is needed

  // Connect the extension
  // Reader Side
  val readerDataAfterExtension = Wire(chiselTypeOf(reader.io.data))

  val readerExtensions = Module(
    new DataPathExtensionHost(
      readerParam.extParam,
      dataWidth        = readerParam.rwParam.tcdmParam.dataWidth * readerParam.rwParam.tcdmParam.numChannel,
      headCut          = false,
      tailCut          = false,
      halfCut          = false,
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
      dataWidth        = writerParam.rwParam.tcdmParam.dataWidth * writerParam.rwParam.tcdmParam.numChannel,
      headCut          = false,
      tailCut          = false,
      halfCut          = false,
      moduleNamePrefix = clusterName
    )
  )

  writerExtensions.io.data.in <> writerDataBeforeExtension
  writerExtensions.io.data.out <> writer.io.data
  // The writer-side plugin CSR region is laid out extensions-first, junctions-second: the extension host consumes
  // its share and hands the remainder to the switch's junction bank.
  val writerJunctionCfg = writerExtensions.io.connectCfgWithList(io.writerCfg.extCfg.toIndexedSeq)
  writerExtensions.io.start := io.writerStart

  // ============================ the crossing ============================
  // All four crossing ports and every mode that routes between them live in XDMADataSwitch. Its CHAINGATHER and
  // GATHERROOT modes fold the arriving remote partial with this node's local operand at a DataPathJunction instead
  // of copying it onward.
  val dataSwitch = Module(
    new XDMADataSwitch(
      param       = writerParam,
      dataWidth   = writerParam.rwParam.tcdmParam.dataWidth * writerParam.rwParam.tcdmParam.numChannel,
      clusterName = clusterName
    )
  )

  readerDataAfterExtension <> dataSwitch.io.localIn
  dataSwitch.io.localOut <> writerDataBeforeExtension
  io.remoteXDMAData.toRemote <> dataSwitch.io.toRemote
  dataSwitch.io.fromRemote <> io.remoteXDMAData.fromRemote

  io.junctionStarved := dataSwitch.io.junctionStarved
  io.junctionCfgErr  := dataSwitch.io.junctionCfgErr

  dataSwitch.io.readerLocalLoopback  := io.readerCfg.localLoopback
  dataSwitch.io.writerLocalLoopback  := io.writerCfg.localLoopback
  dataSwitch.io.writerRemoteLoopback := io.writerCfg.remoteLoopback
  dataSwitch.io.writerStart          := io.writerStart
  dataSwitch.io.writerBusyRaw        := writer.io.busy
  dataSwitch.io.readerBusy           := io.readerBusy
  if (writerParam.junctionParam.nonEmpty) {
    dataSwitch.io.junctionCfg.enable  := writerJunctionCfg.head
    dataSwitch.io.junctionCfg.userCsr := writerJunctionCfg.tail.take(dataSwitch.io.junctionCfg.userCsr.length)
  } else {
    dataSwitch.io.junctionCfg := DontCare
  }

  val isChainedWrite = dataSwitch.io.isChainedWrite
  io.writerBusy := dataSwitch.io.writerBusy

  // Connect the AccompaniedCfg signal
  // Create three intermediate wires to convert from XDMAIntraClusterCfgIO to XDMADataPathCfgIO
  // Normal Read: toRemoteAccompaniedCfg <- Coming from reader side
  // Normal Write: fromRemoteAccompaniedCfg <- Coming from writer side
  // Chained Write (Reader side's cfg): toRemoteChainedWriteAccompaniedCfg <- Coming from writer side
  val fromRemoteAccompaniedCfg = Wire(
    chiselTypeOf(io.remoteXDMAData.fromRemoteAccompaniedCfg)
  )
  fromRemoteAccompaniedCfg.convertFromXDMAIntraClusterCfgIO(
    cfg            = io.writerCfg,
    isChainedWrite = false
  )

  val toRemoteAccompaniedCfg = Wire(
    chiselTypeOf(io.remoteXDMAData.toRemoteAccompaniedCfg)
  )
  toRemoteAccompaniedCfg.convertFromXDMAIntraClusterCfgIO(
    cfg            = io.readerCfg,
    isChainedWrite = false
  )

  val toRemoteChainedWriteAccompaniedCfg = Wire(
    chiselTypeOf(io.remoteXDMAData.toRemoteAccompaniedCfg)
  )
  toRemoteChainedWriteAccompaniedCfg.convertFromXDMAIntraClusterCfgIO(
    cfg            = io.writerCfg,
    isChainedWrite = true
  )

  // The readyToSubmit signal should only be high when the localLoopback is false
  // (The data needs to come from / to the remote side)
  fromRemoteAccompaniedCfg.readyToTransfer := Mux(
    io.writerCfg.localLoopback,
    false.B,
    io.writerBusy
  )

  // Direction tag for the ARRIVING stream. `origination` alone cannot decide it: a ChainGather's collector
  // originates the cfg AND receives the chain's payload, so it would tag its own inbound stream a remote-read and
  // the chain's tail bit would never be set. `collectiveMode` says the payload is chain traffic regardless of who
  // issued the cfg; with it clear this is exactly the previous expression.
  fromRemoteAccompaniedCfg.taskType := Mux(
    io.writerCfg.origination === io.writerCfg.originationIsFromLocal.B && !io.writerCfg.collectiveMode,
    fromRemoteAccompaniedCfg.taskTypeIsRemoteRead.B,
    fromRemoteAccompaniedCfg.taskTypeIsRemoteWrite.B
  )

  // At a gather node the local reader runs CONCURRENTLY with the chained transfer, supplying the junction's local
  // operand. Without the `isGather` term, `readerBusy` rising before the gather state machine would present a
  // to-remote cfg that looks like a fresh chain HEAD: the adapter would latch a first-chained-write and emit a
  // finish to the local core for a task that has not happened. A gather node's outgoing cfg is the CHAINED one
  // (middle) or nothing at all (root), never this one.
  toRemoteAccompaniedCfg.readyToTransfer := Mux(
    io.readerCfg.localLoopback || dataSwitch.io.isGather,
    false.B,
    io.readerBusy
  )

  // Direction tag for the OUTGOING stream, and the mirror of the same problem: a ChainGather HEAD is configured
  // remotely but pushes the chain's first payload, so `origination` would tag it a remote-read and the adapter's
  // chain FSMs -- which switch on this bit -- would not see a chain at all. With `collectiveMode` clear this is
  // exactly the previous expression.
  toRemoteAccompaniedCfg.taskType := Mux(
    io.readerCfg.origination === io.readerCfg.originationIsFromLocal.B || io.readerCfg.collectiveMode,
    toRemoteAccompaniedCfg.taskTypeIsRemoteWrite.B,
    toRemoteAccompaniedCfg.taskTypeIsRemoteRead.B
  )

  // A CHAINGATHER middle hop forwards the FOLD to the next hop, so it carries the same shifted (chained) cfg a
  // CHAINWRITE middle hop does, which keeps data head-to-tail with Grant/Finish tail-to-head. A GATHERROOT is the
  // tail and forwards nothing.
  val isChainedForward = isChainedWrite || dataSwitch.io.isGatherMid
  toRemoteChainedWriteAccompaniedCfg.readyToTransfer := isChainedForward
  toRemoteChainedWriteAccompaniedCfg.taskType        := toRemoteChainedWriteAccompaniedCfg.taskTypeIsRemoteWrite.B

  // The actual output of AccompaniedCfg is determined by the remoteLoopback signal:
  // If the remoteLoopback signal is high, toRemoteAccompaniedCfg needs to be shifted
  io.remoteXDMAData.fromRemoteAccompaniedCfg := fromRemoteAccompaniedCfg
  io.remoteXDMAData.toRemoteAccompaniedCfg   := Mux(
    isChainedForward,
    toRemoteChainedWriteAccompaniedCfg,
    toRemoteAccompaniedCfg
  )
}

// Below is the class to determine if chisel generate Verilog correctly
object XDMADataPathEmitter extends App {
  emitVerilog(
    new XDMADataPath(
      readerParam = new XDMAParam(
        cfgParam          = new XDMAConfigParam(addrWidth = 32, dataWidth = 32),
        axiParam          = new XDMAAXIParam,
        crossClusterParam = new XDMACrossClusterParam,
        rwParam           = new ReaderWriterParam,
        extParam          = Seq()
      ),
      writerParam = new XDMAParam(
        cfgParam          = new XDMAConfigParam(addrWidth = 32, dataWidth = 32),
        axiParam          = new XDMAAXIParam,
        crossClusterParam = new XDMACrossClusterParam,
        rwParam           = new ReaderWriterParam,
        extParam          = Seq()
      )
    ),
    Array("--target-dir", "generated")
  )

}
