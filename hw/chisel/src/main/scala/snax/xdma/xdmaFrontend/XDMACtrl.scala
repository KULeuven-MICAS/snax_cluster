package snax.xdma.xdmaFrontend

import chisel3._
import chisel3.util._

import snax.utils.DecoupledCut._
import snax.utils.DemuxDecoupled
import snax.utils._
import snax.xdma.DesignParams.XDMAParam
import snax.xdma.xdmaIO.XDMACfgIO
import snax.xdma.xdmaIO.XDMAChainRole
import snax.xdma.xdmaIO.XDMAInterClusterCfgIO
import snax.xdma.xdmaIO.XDMAInterClusterCfgIODeserializer
import snax.xdma.xdmaIO.XDMAInterClusterCfgIOSerializer
import snax.xdma.xdmaIO.XDMAIntraClusterCfgIO
import snax.reqRspManager.{ReqRspManager, SnaxReqRspIO}

class XDMACtrlIO(readerParam: XDMAParam, writerParam: XDMAParam) extends Bundle {
  // clusterBaseAddress to determine if it is the local command or remote command
  val clusterBaseAddress = Input(
    UInt(writerParam.axiParam.addrWidth.W)
  )
  // O5 and the starvation watchdog, from the data switch. Both were dead-ended there: the hardware knew and
  // nothing above it could ask. They terminate in the read-only CSR bank.
  val junctionStarved    = Input(Bool())
  val junctionCfgErr     = Input(Bool())
  // Local DMADatapath control signal (Which is connected to DMADataPath)
  val localXDMACfg       = new Bundle {
    val readerCfg = Output(new XDMAIntraClusterCfgIO(readerParam))
    val writerCfg = Output(new XDMAIntraClusterCfgIO(writerParam))

    // Two start signal will inform the new cfg is available, trigger agu, and inform all extension that a stream is coming
    val readerStart = Output(Bool())
    val writerStart = Output(Bool())
    // Two busy signal only go down if a stream fully passthrough the reader / writter, which is provided by DataPath
    // These signals should be readable by the outside; these two will also be used to determine whether the next task can be executed.
    val readerBusy  = Input(Bool())
    val writerBusy  = Input(Bool())
  }
  // Remote control signal, which include the signal from other cluster or signal to other cluster. Both of them is AXI related, serialized signal
  // The remote control signal will contain only src information, in other words, the DMA system can proceed remote read or local read, but only local write
  val remoteXDMACfg      = new Bundle {
    val fromRemote = Flipped(Decoupled(UInt(readerParam.axiParam.dataWidth.W)))
    val toRemote   = Decoupled(UInt(readerParam.axiParam.dataWidth.W))
  }
  // This is the port for CSR Manager to SNAX port
  val csrIO = new SnaxReqRspIO(addrWidth = readerParam.cfgParam.addrWidth, dataWidth = readerParam.cfgParam.dataWidth)

  // The external port to indicate the finish of one task
  val remoteTaskFinished = Input(Bool())
}

class SrcConfigRouter(dataType: XDMACfgIO, clusterName: String = "unnamed_cluster")
    extends Module
    with RequireAsyncReset {

  override val desiredName = s"${clusterName}_xdma_ctrl_srcConfigRouter"

  val io = IO(new Bundle {
    val clusterBaseAddress = Input(UInt(dataType.readerPtr.getWidth.W))
    val from               = Flipped(new Bundle {
      val remote = Decoupled(dataType)
      val local  = Decoupled(dataType)
    })
    val to                 = new Bundle {
      val remote = Decoupled(dataType)
      val local  = Decoupled(dataType)
    }
  })

  val inputCfgArbiter = Module(new Arbiter(dataType, 2) {
    override val desiredName =
      s"${clusterName}_xdma_ctrl_SrcConfigRouter_Arbiter"
  })
  inputCfgArbiter.io.in(0) <> io.from.local
  inputCfgArbiter.io.in(1) <> io.from.remote

  val outputCfgDemux = Module(
    new DemuxDecoupled(dataType = dataType, numOutput = 3) {
      override val desiredName =
        s"${clusterName}_xdma_ctrl_SrcConfigRouter_Demux"
    }
  )
  inputCfgArbiter.io.out -|> outputCfgDemux.io.in

  // At the output of FIFO: Do the rule check
  val cTypeLocal :: cTypeRemote :: cTypeDiscard :: Nil = Enum(3)
  val cValue                                           = Wire(chiselTypeOf(cTypeDiscard))

  when(
    outputCfgDemux.io.in.bits.readerPtr(
      outputCfgDemux.io.in.bits.readerPtr.getWidth - 1,
      log2Ceil(outputCfgDemux.io.in.bits.param.rwParam.tcdmParam.tcdmSize) + 10
    ) === io
      .clusterBaseAddress(
        outputCfgDemux.io.in.bits.readerPtr.getWidth - 1,
        log2Ceil(
          outputCfgDemux.io.in.bits.param.rwParam.tcdmParam.tcdmSize
        ) + 10
      )
  ) {
    cValue := cTypeLocal // When cfg has the Ptr that fall within local TCDM, the data should be forwarded to the local ctrl path
  }.elsewhen(outputCfgDemux.io.in.bits.readerPtr === 0.U) {
    cValue := cTypeDiscard // When cfg has the Ptr that is zero, This means that the frame need to be thrown away. This is important as when the data is moved from DRAM to TCDM or vice versa, DRAM part is handled by iDMA, thus only one config instead of two is submitted
  }.otherwise {
    cValue := cTypeRemote // For the remaining condition, the config is forward to remote DMA
  }

  outputCfgDemux.io.sel                                    := cValue
  // Port local is connected to the outside
  outputCfgDemux.io.out(cTypeLocal.litValue.toInt) <> io.to.local
  // Port remote is connected to the outside
  outputCfgDemux.io.out(cTypeRemote.litValue.toInt) <> io.to.remote
  // Port discard is not connected and will always be discarded
  outputCfgDemux.io.out(cTypeDiscard.litValue.toInt).ready := true.B
}

class DstConfigRouter(dataType: XDMACfgIO, clusterName: String = "unnamed_cluster")
    extends Module
    with RequireAsyncReset {

  override val desiredName = s"${clusterName}_xdma_ctrl_dstConfigRouter"

  val io = IO(new Bundle {
    val clusterBaseAddress = Input(UInt(dataType.writerPtr(0).getWidth.W))
    val from               = Flipped(new Bundle {
      val remote = Decoupled(dataType)
      val local  = Decoupled(dataType)
    })
    val to                 = new Bundle {
      val remote       = Decoupled(dataType)
      val local        = Decoupled(dataType)
      // ChainGather only: the READER-side companion frame for the hop this pass dispatched. A frame crossing the
      // wire carries `isWriterSide` and the receiver demuxes on it, so one frame configures either a reader or a
      // writer -- never both. A gather hop needs both, so the unroll emits two.
      val remoteReader = Decoupled(dataType)
    }
  })

  val inputCfgArbiter = Module(new Arbiter(dataType, 3) {
    override val desiredName =
      s"${clusterName}_xdma_ctrl_SrcConfigRouter_Arbiter"
  })
  // inputCfgArbiter.io.in(0) <> shifted cfg
  inputCfgArbiter.io.in(1) <> io.from.local
  inputCfgArbiter.io.in(2) <> io.from.remote

  val bufferedCfg = Wire(chiselTypeOf(inputCfgArbiter.io.out))
  inputCfgArbiter.io.out -||> bufferedCfg

  // At the output of cut: Do the rule check
  // forwardToLocal: The cfg is for local side, so it needs to be sent to local output
  // forwardToChainedWrite: The cfg is chained write, so it needs to be shifted and looped back
  // forwardToRemote: The cfg is for remote side, so it needs to be sent to remote output
  val forwardToLocal        = {
    bufferedCfg.bits
      .writerPtr(0)
      .apply(
        bufferedCfg.bits.writerPtr(0).getWidth - 1,
        log2Ceil(bufferedCfg.bits.param.rwParam.tcdmParam.tcdmSize) + 10
      ) === io
      .clusterBaseAddress(
        bufferedCfg.bits.writerPtr(0).getWidth - 1,
        log2Ceil(bufferedCfg.bits.param.rwParam.tcdmParam.tcdmSize) + 10
      )
  }
  val forwardToChainedWrite = {
    if (bufferedCfg.bits.writerPtr.length > 1) {
      bufferedCfg.bits.writerPtr(
        1
      ) =/= 0.U && bufferedCfg.bits.origination === bufferedCfg.bits.originationIsFromLocal.B
    } else false.B
  } && bufferedCfg.bits.origination === bufferedCfg.bits.originationIsFromLocal.B

  val destIsRemote = ~forwardToLocal && bufferedCfg.bits.writerPtr(0) =/= 0.U

  // The FIRST unroll pass of a gather dispatches the chain HEAD. A head only sources -- nothing arrives at it --
  // so it must not be given a writer-side frame: that would arm its junction on a stream that never comes and
  // park the join on a starved operand. Every later hop gets both frames.
  val gatherHeadPass      = bufferedCfg.bits.collectiveMode && (~bufferedCfg.bits.chainUnrolled)
  val forwardToRemote     = destIsRemote                    && (~gatherHeadPass)
  // Every remote hop of a gather also needs its reader configured, on its own partial.
  val emitReaderCompanion = destIsRemote                    && bufferedCfg.bits.collectiveMode

  val outputCfgSplitter = Module(
    new SplitterDecoupled(
      dataType  = chiselTypeOf(bufferedCfg.bits),
      numOutput = 4
    ) {
      override val desiredName =
        s"${clusterName}_xdma_ctrl_DstConfigRouter_Splitter"
    }
  )
  outputCfgSplitter.io.in <> bufferedCfg

  // Port 0: The chainedwrite Cfg
  dontTouch(outputCfgSplitter.io.sel(0))
  outputCfgSplitter.io.sel(0) := forwardToChainedWrite
  inputCfgArbiter.io.in(0) <> outputCfgSplitter.io.out(0)

  inputCfgArbiter.io.in(0).bits.readerPtr      := outputCfgSplitter.io.out(0).bits.writerPtr(0)
  inputCfgArbiter.io
    .in(0)
    .bits
    .writerPtr
    .zip(outputCfgSplitter.io.out(0).bits.writerPtr.tail)
    .foreach { case (a, b) =>
      a := b
    }
  inputCfgArbiter.io.in(0).bits.writerPtr.last := 0.U
  // Distinguishes later passes from the first one, whose destination is the chain head.
  inputCfgArbiter.io.in(0).bits.chainUnrolled  := true.B

  // Port 1: The local Cfg
  outputCfgSplitter.io.sel(1) := forwardToLocal
  outputCfgSplitter.io.out(1) <> io.to.local
  // Same stamp as the remote port: a frame landing on this node's writer forwards onward if it still has a next
  // hop, and terminates the chain otherwise. This is the value the receiver used to derive for itself.
  io.to.local.bits.chainRole  := Mux(
    outputCfgSplitter.io.out(1).bits.writerPtr(1) =/= 0.U,
    XDMAChainRole.MIDDLE.U,
    XDMAChainRole.TAIL.U
  )
  // The port-0 loopback shift updates writerPtr but NOT aguCfg.ptr. A locally-terminated chain -- which is
  // exactly the ChainGather collector's TAIL -- would otherwise arm its writer AGU at the ORIGINAL
  // writerPtr(0) (the far source's offset) instead of this node's own dst, so the fold lands at the wrong
  // local address and the real dst is never written. Re-point the writer AGU at the current terminal, the
  // same re-derivation a remotely-received frame does in convertToXDMACfgIO. No-op for a plain local write
  // (writerPtr(0) was never shifted); gather-MIDDLE and ChainWrite terminals are unaffected.
  io.to.local.bits.aguCfg.ptr := outputCfgSplitter.io
    .out(1)
    .bits
    .writerPtr(0)(io.to.local.bits.aguCfg.ptr.getWidth - 1, 0)
  // Port 2: The remote Cfg
  outputCfgSplitter.io.sel(2) := forwardToRemote
  outputCfgSplitter.io.out(2) <> io.to.remote
  // Stamp the destination's position in the chain. The receiving node used to infer this from its own
  // `writerPtr(1) =/= 0` after the frame landed; computing it here from the same pointer produces the identical
  // value, but it also works when position cannot be inferred locally -- a gather head is remote-configured, so
  // its `origination` says "from remote" while its role is HEAD.
  io.to.remote.bits.chainRole := Mux(
    outputCfgSplitter.io.out(2).bits.writerPtr(1) =/= 0.U,
    XDMAChainRole.MIDDLE.U,
    XDMAChainRole.TAIL.U
  )

  // Port 3: the ChainGather reader-side companion. It says "read YOUR OWN partial and send it on":
  //   readerPtr    := writerPtr(0)   -- the hop's own operand, which is what routes this frame to it
  //   writerPtr(0) := writerPtr(1)   -- its next hop
  // At the head that is the whole configuration (it sources the chain). At a middle the switch steers the reader
  // into the junction instead, so writerPtr(0) only has to be non-zero, and the head is the one pass whose role
  // cannot be re-derived downstream -- hence the explicit stamp.
  outputCfgSplitter.io.sel(3)            := emitReaderCompanion
  io.to.remoteReader.valid               := outputCfgSplitter.io.out(3).valid
  outputCfgSplitter.io.out(3).ready      := io.to.remoteReader.ready
  io.to.remoteReader.bits                := outputCfgSplitter.io.out(3).bits
  io.to.remoteReader.bits.readerPtr      := outputCfgSplitter.io.out(3).bits.writerPtr(0)
  io.to.remoteReader.bits.writerPtr(0)   := outputCfgSplitter.io.out(3).bits.writerPtr(1)
  io.to.remoteReader.bits.writerPtr.drop(1).foreach(_ := 0.U)
  io.to.remoteReader.bits.chainRole      := Mux(gatherHeadPass, XDMAChainRole.HEAD.U, XDMAChainRole.MIDDLE.U)
  io.to.remoteReader.bits.collectiveMode := true.B
}

class XDMACtrl(readerparam: XDMAParam, writerparam: XDMAParam, clusterName: String = "unnamed_cluster")
    extends Module
    with RequireAsyncReset {
  val io = IO(
    new XDMACtrlIO(
      readerParam = readerparam,
      writerParam = writerparam
    )
  )

  override val desiredName = s"${clusterName}_xdma_ctrl"

  val numCSRPerPtr = (writerparam.axiParam.addrWidth + 31) / 32

  val csrManager = Module(
    new ReqRspManager(
      numReadWriteReg = numCSRPerPtr + // Reader Pointer needs numCSRPerPtr CSRs
        readerparam.crossClusterParam.maxSpatialDimension +      // Spatial Strides for reader
        readerparam.crossClusterParam.maxTemporalDimension * 2 + // Temporal Strides + Bounds for reader
        {
          if (readerparam.rwParam.configurableChannel) 1 else 0
        } +                                                      // Enabled Channel for reader
        0 +                        // Enabled Byte for reader: non-effective, so donot assign CSR
        readerparam.pluginCsrNum + // reader extensions (custom CSR + bypass CSR) + reader junctions, if any
        numCSRPerPtr * writerparam.crossClusterParam.maxMulticastDest + // Writer Pointer needs numCSRPerPtr * maxMulticastDest CSRs
        writerparam.crossClusterParam.maxSpatialDimension +      // Spatial Strides for writer
        writerparam.crossClusterParam.maxTemporalDimension * 2 + // Temporal Strides + Bounds for writer
        {
          if (writerparam.rwParam.configurableChannel) 1 else 0
        } +                                                      // Enabled Channel for writer
        {
          if (writerparam.rwParam.configurableByteMask) 1 else 0
        } +                                                      // Enabled Byte for writer
        writerparam.pluginCsrNum + // writer extensions (custom CSR + bypass CSR) + the switch's junction bank
        1, // The start CSR
      numReadOnlyReg  = 8,
      // 1) submitted local requests 2) submitted remote requests 3) finished local requests 4) finished remote
      // requests 5) XDMA task performance counter 6) reader performance counter 7) writer performance counter
      // 8) junction status -- O5 configuration error and the starvation watchdog, sticky since the last start.
      // Appended at the END of the read-only bank so no read-write pointer moves.
      addrWidth       = readerparam.cfgParam.addrWidth,
      ioDataWidth     = readerparam.cfgParam.dataWidth,
      regDataWidth    = 32,
      // Set a name for the module class so that it will not overlapped with other csrManagers in user-defined accelerators
      moduleTagName   = s"${clusterName}_xdma_"
    )
  )

  csrManager.io.reqRspIO <> io.csrIO

  // Cfg from local side
  val preRoute_src_local = Wire(
    Decoupled(new XDMACfgIO(readerparam))
  )
  val preRoute_dst_local = Wire(
    Decoupled(new XDMACfgIO(writerparam))
  )
  var remainingCSR       = csrManager.io.readWriteRegIO.bits.toIndexedSeq

  // Connect readerPtr + writerPtr with the CSR list
  preRoute_src_local.bits.connectPtrWithList(remainingCSR)
  preRoute_dst_local.bits.connectPtrWithList(remainingCSR)
  val nextSrcPtrCSR = remainingCSR.take(numCSRPerPtr)
  val nextDstPtrCSR = remainingCSR.drop(numCSRPerPtr).take(numCSRPerPtr)

  // Drop readerPtr + writerPtr from the CSR list
  remainingCSR = preRoute_src_local.bits.dropPtrFromList(remainingCSR)

  // Connect reader + writer + ext to the structured signal: Src side
  remainingCSR = preRoute_src_local.bits.connectWithList(nextSrcPtrCSR ++ remainingCSR)

  // Connect reader + writer + ext to the structured signal: Dst side
  remainingCSR = preRoute_dst_local.bits.connectWithList(nextDstPtrCSR ++ remainingCSR)

  if (remainingCSR.length > 1)
    println("There is some error in CSR -> Structured CFG assigning")

  // LocalLoopback: The loopback indicator to enable the reader's data directly sending back to the writer
  // Connect the loopBack signal: The loopBack signal is generated by comparing the Ptr of two side

  val localLoopback =
    preRoute_dst_local.bits.readerPtr(
      preRoute_dst_local.bits.readerPtr.getWidth - 1,
      log2Ceil(preRoute_dst_local.bits.param.rwParam.tcdmParam.tcdmSize) + 10
    ) === preRoute_dst_local.bits
      .writerPtr(0)
      .apply(
        preRoute_dst_local.bits.writerPtr(0).getWidth - 1,
        log2Ceil(preRoute_dst_local.bits.param.rwParam.tcdmParam.tcdmSize) + 10
      )
  preRoute_src_local.bits.localLoopback := localLoopback
  preRoute_dst_local.bits.localLoopback := localLoopback

  // RemoteLoopback: The loopback indicator to enable fromRemoteData directly sending back to toRemoteData
  // Connect the loopBack signal: currently, the remoteLoopback signal is always set to false; it will be judgen after the cfgRouter
  preRoute_src_local.bits.remoteLoopback := false.B
  preRoute_dst_local.bits.remoteLoopback := false.B

  // A locally-originated cfg is the head of whatever chain it starts -- which is what `chainRoleDefault`
  // evaluates to for origination = fromLocal, so this is the existing inference written down rather than a
  // change. Frames leaving for a remote node are stamped in DstConfigRouter instead.
  preRoute_src_local.bits.chainRole := XDMAChainRole.HEAD.U
  preRoute_dst_local.bits.chainRole := XDMAChainRole.HEAD.U

  // A transfer is a ChainGather exactly when software armed a junction on it -- the junction bank's enable bit
  // already rides the writer-side cfg region, so no separate "which collective" CSR is needed. The cfg the core
  // submits has not been unrolled yet.
  preRoute_src_local.bits.collectiveMode := preRoute_dst_local.bits.junctionEnabled
  preRoute_dst_local.bits.collectiveMode := preRoute_dst_local.bits.junctionEnabled
  preRoute_src_local.bits.chainUnrolled  := false.B
  preRoute_dst_local.bits.chainUnrolled  := false.B

  // axiTransferBeatSize: The cycle of transfer for this cfg
  // Since the writer side connects the Datapath Extensions with the number of data beats unchanged, both reader and writer side take the value from the writer side
  val axiTransferBeatSize = preRoute_dst_local.bits.aguCfg.temporalBounds.reduceTree { (a, b) =>
    (a * b).apply(preRoute_dst_local.bits.axiTransferBeatSize.getWidth - 1, 0)
  }
  preRoute_src_local.bits.axiTransferBeatSize := axiTransferBeatSize
  preRoute_dst_local.bits.axiTransferBeatSize := axiTransferBeatSize

  // Connect Valid and bits: Only when both preRoutes are ready, postRulecheck is ready
  csrManager.io.readWriteRegIO.ready := preRoute_src_local.ready & preRoute_dst_local.ready
  preRoute_src_local.valid           := csrManager.io.readWriteRegIO.fire
  preRoute_dst_local.valid           := csrManager.io.readWriteRegIO.fire

  // Task ID Counter to assign the ID for each transaction
  val localSubmittedTaskIDCounter = Module(
    new BasicCounter(
      width   = 8,
      hasCeil = false
    ) {
      override val desiredName = s"${clusterName}_xdmaCtrl_localTaskIDCounter"
    }
  )
  localSubmittedTaskIDCounter.io.ceil := DontCare
  localSubmittedTaskIDCounter.io.tick  := localLoopback && csrManager.io.readWriteRegIO.fire
  localSubmittedTaskIDCounter.io.reset := false.B

  val remoteSubmittedTaskIDCounter = Module(
    new BasicCounter(
      width   = 8,
      hasCeil = false
    ) {
      override val desiredName = s"${clusterName}_xdmaCtrl_remoteTaskIDCounter"
    }
  )
  remoteSubmittedTaskIDCounter.io.ceil := DontCare
  remoteSubmittedTaskIDCounter.io.tick  := (~localLoopback) && csrManager.io.readWriteRegIO.fire
  remoteSubmittedTaskIDCounter.io.reset := false.B

  csrManager.io.readOnlyReg(0) := localSubmittedTaskIDCounter.io.value
  csrManager.io.readOnlyReg(1) := remoteSubmittedTaskIDCounter.io.value

  // Connect the task ID to the structured signal
  val taskID = Mux(
    localLoopback,
    localSubmittedTaskIDCounter.io.value,
    remoteSubmittedTaskIDCounter.io.value
  ) + 1.U
  preRoute_src_local.bits.taskID := taskID
  preRoute_dst_local.bits.taskID := taskID

  // Demux the cfg from the cfg_in port
  val cfgFromRemoteDemux = Module(
    new DemuxDecoupled(
      dataType  = new XDMAInterClusterCfgIO(readerparam, writerparam),
      numOutput = 2
    ) {
      override val desiredName = s"${clusterName}_xdma_ctrl_remoteCfgDemux"
    }
  )

  val interClusterCfgDeserializer = Module(
    new XDMAInterClusterCfgIODeserializer(writerparam)
  )
  interClusterCfgDeserializer.io.cfgIn <> io.remoteXDMACfg.fromRemote
  cfgFromRemoteDemux.io.in <> interClusterCfgDeserializer.io.cfgOut
  cfgFromRemoteDemux.io.sel := interClusterCfgDeserializer.io.cfgOut.bits.isWriterSide

  // Mux+Arbitrator the Cfg to cfg_out port
  val cfgToRemoteMux = Module(
    new Arbiter(
      gen = new XDMAInterClusterCfgIO(readerparam, writerparam),
      n   = 3
    ) {
      override val desiredName = s"${clusterName}_xdma_ctrl_remoteCfgMux"
    }
  )

  val interClusterCfgSerializer = Module(
    new XDMAInterClusterCfgIOSerializer(writerparam)
  )
  cfgToRemoteMux.io.out <> interClusterCfgSerializer.io.cfgIn

  interClusterCfgSerializer.io.cfgOut <> io.remoteXDMACfg.toRemote
  // Arbitration is done by Arbiter, no sel signal is needed

  // Command Router
  val srcCfgRouter = Module(
    new SrcConfigRouter(
      dataType    = chiselTypeOf(preRoute_src_local.bits),
      clusterName = clusterName
    )
  )
  srcCfgRouter.io.clusterBaseAddress := io.clusterBaseAddress

  // io.from
  // The local side is connected to the signal from the CSR manager
  srcCfgRouter.io.from.local <> preRoute_src_local
  // The remote side is connected to Demux
  srcCfgRouter.io.from.remote.valid  := cfgFromRemoteDemux.io.out(0).valid
  cfgFromRemoteDemux.io.out(0).ready := srcCfgRouter.io.from.remote.ready
  srcCfgRouter.io.from.remote.bits   := cfgFromRemoteDemux.io.out(0).bits.convertToXDMACfgIO(readerSide = true)

  // io.to
  // The local side is connected to the later dispatch unit
  val postRoute_src_local = Wire(Decoupled(new XDMACfgIO(readerparam)))
  srcCfgRouter.io.to.local <> postRoute_src_local
  // The remote side is connected to the arbitrator
  cfgToRemoteMux.io.in(0).valid   := srcCfgRouter.io.to.remote.valid
  srcCfgRouter.io.to.remote.ready := cfgToRemoteMux.io.in(0).ready
  cfgToRemoteMux.io.in(0).bits.convertFromXDMACfgIO(writerSide = false, cfg = srcCfgRouter.io.to.remote.bits)

  // Command Router
  val dstCfgRouter = Module(
    new DstConfigRouter(
      dataType    = chiselTypeOf(preRoute_dst_local.bits),
      clusterName = clusterName
    )
  )
  dstCfgRouter.io.clusterBaseAddress := io.clusterBaseAddress

  // io.from
  // The local side is connected to the signal from the CSR manager
  dstCfgRouter.io.from.local <> preRoute_dst_local
  // The remote side is connected to the Demux
  dstCfgRouter.io.from.remote.valid  := cfgFromRemoteDemux.io.out(1).valid
  cfgFromRemoteDemux.io.out(1).ready := dstCfgRouter.io.from.remote.ready
  dstCfgRouter.io.from.remote.bits   := cfgFromRemoteDemux.io.out(1).bits.convertToXDMACfgIO(readerSide = false)

  // io.to
  // The local side is connected to the later dispatch unit
  val postRoute_dst_local = Wire(
    Decoupled(new XDMACfgIO(writerparam))
  )
  postRoute_dst_local <> dstCfgRouter.io.to.local
  // The remote side is connected to the Serializer
  // The ChainGather reader-side companion, emitted as writerSide=false so the receiver's demux lands it on its
  // READER path. It takes the lower Arbiter index (= higher priority) than the writer-side frame below, so a hop
  // is told what to read before it is told to fold. Liveness does not depend on that order -- a gather node's
  // switch only leaves its idle state once its reader runs, and the junction's skid FIFOs absorb the skew -- but
  // arming the operand source first keeps the arriving stream from waiting on a reader that has no cfg yet.
  cfgToRemoteMux.io.in(1).valid         := dstCfgRouter.io.to.remoteReader.valid
  dstCfgRouter.io.to.remoteReader.ready := cfgToRemoteMux.io.in(1).ready
  cfgToRemoteMux.io.in(1).bits.convertFromXDMACfgIO(writerSide = false, cfg = dstCfgRouter.io.to.remoteReader.bits)

  cfgToRemoteMux.io.in(2).valid   := dstCfgRouter.io.to.remote.valid
  dstCfgRouter.io.to.remote.ready := cfgToRemoteMux.io.in(2).ready
  cfgToRemoteMux.io.in(2).bits.convertFromXDMACfgIO(writerSide = true, cfg = dstCfgRouter.io.to.remote.bits)

  // Judge remoteLoopback signal
  postRoute_src_local.bits.remoteLoopback := false.B
  postRoute_dst_local.bits.remoteLoopback := {
    if (writerparam.crossClusterParam.maxMulticastDest == 1)
      false.B
    else
      postRoute_dst_local.bits.writerPtr(
        1
      ) =/= 0.U
  }

  // Connect these two cfg to the actual input: Need two small (Mealy) FSMs to manage the start signal and pop out the consumed cfg
  val sIdle :: sWaitBusy :: sBusy :: Nil = Enum(3)

  sIdle.suggestName("sIdle")
  sWaitBusy.suggestName("sWaitBusy")
  sBusy.suggestName("sBusy")

  // Two registers to store the current state
  val nextStateSrc    = Wire(chiselTypeOf(sIdle))
  val nextStateDst    = Wire(chiselTypeOf(sIdle))
  dontTouch(nextStateSrc)
  dontTouch(nextStateDst)
  val currentStateSrc = RegNext(nextStateSrc, sIdle)
  val currentStateDst = RegNext(nextStateDst, sIdle)

  // Two Data Cut to store the buffer the current cfg
  val currentCfgSrc = Wire(chiselTypeOf(postRoute_src_local))
  val currentCfgDst = Wire(chiselTypeOf(postRoute_dst_local))
  postRoute_src_local -|> currentCfgSrc
  postRoute_dst_local -|> currentCfgDst

  // Default value: Not pop out config, not start reader/writer, not change state
  io.localXDMACfg.readerStart := false.B
  io.localXDMACfg.writerStart := false.B
  currentCfgSrc.ready         := false.B
  currentCfgDst.ready         := false.B
  nextStateSrc                := currentStateSrc
  nextStateDst                := currentStateDst

  // Control signals in Src Path
  switch(currentStateSrc) {
    is(sIdle) {
      when {
        // Cfg at source side is valid, Reader is not busy, the Writer's current / next cfg is not Chained Write.
        // INTERLOCK HALF 1 of 2. A chained WRITE locks the reader out: it has nothing to contribute to a broadcast.
        // A chained GATHER does not, because the reader is what supplies the junction's local operand and must run
        // concurrently with the chained transfer. The matching half is in the Dst path below; both must agree.
        currentCfgSrc.valid                       && (~(currentCfgDst.valid && currentCfgDst.bits.remoteLoopback &&
          (~currentCfgDst.bits.junctionEnabled))) && (
          // The local loopback condition: The next cfg at the writer side is its counterpart
          (currentCfgSrc.bits.localLoopback && currentCfgSrc.bits.readerPtr === currentCfgDst.bits.readerPtr && currentCfgSrc.bits
            .writerPtr(0) === currentCfgDst.bits.writerPtr(0)) ||
            // The remote read condition: All loopback is false
            (currentCfgSrc.bits.localLoopback === false.B && currentCfgSrc.bits.remoteLoopback === false.B)
        )
      } {
        // Start the reader side
        nextStateSrc                := sWaitBusy
        io.localXDMACfg.readerStart := true.B
      }
    }
    is(sWaitBusy) {
      when(currentCfgSrc.bits.axiTransferBeatSize === 0.U) {
        // If the transfer size is zero, do not wait for busy signal
        nextStateSrc        := sIdle
        currentCfgSrc.ready := true.B
      }.elsewhen(io.localXDMACfg.readerBusy) {
        nextStateSrc := sBusy
      }
    }
    is(sBusy) {
      when(~io.localXDMACfg.readerBusy) {
        nextStateSrc        := sIdle
        currentCfgSrc.ready := true.B
      }
    }
  }

  // Control signals in Dst Path
  switch(currentStateDst) {
    is(sIdle) {
      when {
        // Cfg at destination side is valid, Writer is not busy
        currentCfgDst.valid && (
          // The local loopback condition: The next cfg at the writer side is its counterpart
          (currentCfgDst.bits.localLoopback && currentCfgSrc.bits.readerPtr === currentCfgDst.bits.readerPtr && currentCfgSrc.bits
            .writerPtr(0) === currentCfgDst.bits.writerPtr(0)) ||
            // The remote write condition: All loopback is false
            (currentCfgDst.bits.localLoopback === false.B && currentCfgDst.bits.remoteLoopback === false.B) ||
            // The remote chained write condition: The remote loopback is true and the next state of the counterpart is sIdle
            // INTERLOCK HALF 2 of 2. Symmetric to the Src path above: a chained WRITE waits for an idle reader, a
            // chained GATHER requires a concurrently running reader. Both halves must agree.
            (currentCfgDst.bits.remoteLoopback === true.B &&
              (currentStateSrc === sIdle || currentCfgDst.bits.junctionEnabled))
        )
      } {
        // Start the reader side
        nextStateDst                := sWaitBusy
        io.localXDMACfg.writerStart := true.B
      }
    }
    is(sWaitBusy) {
      when(currentCfgDst.bits.axiTransferBeatSize === 0.U) {
        // If the transfer size is zero, do not wait for busy signal
        nextStateDst        := sIdle
        currentCfgDst.ready := true.B
      }.elsewhen(io.localXDMACfg.writerBusy) {
        nextStateDst := sBusy
      }
    }
    is(sBusy) {
      when(~io.localXDMACfg.writerBusy) {
        nextStateDst        := sIdle
        currentCfgDst.ready := true.B
      }
    }
  }

  // Data Signals in Src Path
  io.localXDMACfg.readerCfg.convertFromXDMACfgIO(currentCfgSrc.bits)
  // Data Signals in Dst Path
  io.localXDMACfg.writerCfg.convertFromXDMACfgIO(currentCfgDst.bits)

  // Counter for finished task
  val localFinishedTaskIDCounter = Module(new BasicCounter(8, hasCeil = false) {
    override val desiredName =
      s"${clusterName}_xdma_ctrl_localFinishedTaskCounter"
  })
  localFinishedTaskIDCounter.io.ceil := DontCare
  localFinishedTaskIDCounter.io.reset := false.B
  localFinishedTaskIDCounter.io.tick  := currentCfgDst.fire && currentCfgDst.bits.localLoopback

  val remoteFinishedTaskIDCounter = Module(
    new BasicCounter(8, hasCeil = false) {
      override val desiredName =
        s"${clusterName}_xdma_ctrl_remoteFinishedTaskCounter"
    }
  )
  remoteFinishedTaskIDCounter.io.ceil := DontCare
  remoteFinishedTaskIDCounter.io.reset := false.B
  remoteFinishedTaskIDCounter.io.tick  := io.remoteTaskFinished

  // Connect the finished task counter to the read-only CSR
  // ---- junction status ---------------------------------------------------------------------------------
  // STICKY since the last start. A configuration error and a starved join are both transient by nature -- the
  // error is asserted only while the offending word is armed, and the watchdog clears the moment a pair fires --
  // so a register that merely sampled them would report a clean transfer for a broken one.
  val jctCfgErrSticky  = RegInit(false.B)
  val jctStarvedSticky = RegInit(false.B)
  when(io.localXDMACfg.writerStart) {
    jctCfgErrSticky  := false.B
    jctStarvedSticky := false.B
  }.otherwise {
    when(io.junctionCfgErr) { jctCfgErrSticky := true.B }
    when(io.junctionStarved) { jctStarvedSticky := true.B }
  }
  csrManager.io
    .readOnlyReg(7) := Cat(0.U(29.W), io.junctionStarved || io.junctionCfgErr, jctStarvedSticky, jctCfgErrSticky)

  csrManager.io.readOnlyReg(2) := localFinishedTaskIDCounter.io.value
  csrManager.io.readOnlyReg(3) := remoteFinishedTaskIDCounter.io.value

  // Performance Counter for the last XDMA task's time
  val pcIdle :: pcRunning :: Nil = Enum(2)

  val perfCounterTask = Module(new BasicCounter(width = 32, hasCeil = false) {
    override val desiredName = s"${clusterName}_xdma_ctrl_perfCounterTask"
  })
  csrManager.io.readOnlyReg(4) := perfCounterTask.io.value

  val pctCurrentState = RegInit(pcIdle)
  val pctNextState    = pctCurrentState
  pctCurrentState := pctNextState
  dontTouch(pctCurrentState)
  dontTouch(pctNextState)

  perfCounterTask.io.reset := false.B
  perfCounterTask.io.tick  := false.B
  perfCounterTask.io.ceil  := DontCare

  switch(pctCurrentState) {
    is(pcIdle) {
      when(csrManager.io.readWriteRegIO.fire) {
        perfCounterTask.io.reset := true.B
        pctNextState             := pcRunning
      }
    }
    is(pcRunning) {
      perfCounterTask.io.tick := true.B
      when((currentCfgDst.fire && currentCfgDst.bits.localLoopback) || io.remoteTaskFinished) {
        perfCounterTask.io.tick := false.B
        pctNextState            := pcIdle
      }
    }
  }

  // Performance Counter for the last reader's busy time
  val perfCounterReader = Module(new BasicCounter(width = 32, hasCeil = false) {
    override val desiredName = s"${clusterName}_xdma_ctrl_perfCounterReader"
  })
  csrManager.io.readOnlyReg(5) := perfCounterReader.io.value

  val pcrCurrentState = RegInit(pcIdle)
  val pcrNextState    = pcrCurrentState
  pcrCurrentState := pcrNextState
  dontTouch(pcrCurrentState)
  dontTouch(pcrNextState)

  perfCounterReader.io.reset := false.B
  perfCounterReader.io.tick  := false.B
  perfCounterReader.io.ceil  := DontCare

  switch(pcrCurrentState) {
    is(pcIdle) {
      when(io.localXDMACfg.readerBusy) {
        perfCounterReader.io.reset := true.B
        pcrNextState               := pcRunning
      }
    }
    is(pcRunning) {
      perfCounterReader.io.tick := true.B
      when(~io.localXDMACfg.readerBusy) {
        perfCounterReader.io.tick := false.B
        pcrNextState              := pcIdle
      }
    }
  }

  // Performance Counter for the last writer's busy time
  val perfCounterWriter = Module(new BasicCounter(width = 32, hasCeil = false) {
    override val desiredName = s"${clusterName}_xdma_ctrl_perfCounterWriter"
  })
  csrManager.io.readOnlyReg(6) := perfCounterWriter.io.value

  val pcwCurrentState = RegInit(pcIdle)
  val pcwNextState    = pcwCurrentState
  pcwCurrentState := pcwNextState
  dontTouch(pcwCurrentState)
  dontTouch(pcwNextState)

  perfCounterWriter.io.reset := false.B
  perfCounterWriter.io.tick  := false.B
  perfCounterWriter.io.ceil  := DontCare

  switch(pcwCurrentState) {
    is(pcIdle) {
      when(io.localXDMACfg.writerBusy) {
        perfCounterWriter.io.reset := true.B
        pcwNextState               := pcRunning
      }
    }
    is(pcRunning) {
      perfCounterWriter.io.tick := true.B
      when(~io.localXDMACfg.writerBusy) {
        perfCounterWriter.io.tick := false.B
        pcwNextState              := pcIdle
      }
    }
  }
}
