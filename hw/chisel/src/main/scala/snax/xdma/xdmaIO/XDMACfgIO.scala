package snax.xdma.xdmaIO

import chisel3._
import chisel3.util._

import snax.readerWriter.AddressGenUnitCfgIO
import snax.readerWriter.AddressGenUnitParam
import snax.readerWriter.ReaderWriterCfgIO
import snax.utils._
import snax.xdma.DesignParams._

/** Where a node sits in a CHAIN, stamped explicitly per frame rather than inferred.
  *
  * ChainWrite can infer position from `origination` because the initiator IS the data head. ChainGather separates the
  * two: the initiator is the COLLECTOR, which sits at the tail, and the head is a node the initiator configured
  * remotely. So position has to be carried.
  *
  * `chainRoleDefault` reproduces the origination-based inference exactly, so a ChainWrite frame that does not override
  * it is bit-identical to before.
  */
object XDMAChainRole {
  val HEAD   = 0 // sources the chain: its to-remote cfg carries is_first_cw
  val MIDDLE = 1 // forwards onward: neither bit
  val TAIL   = 2 // terminates the chain: its cfg carries is_last_cw
  val width  = 2
}

// The sturctured class that used to store the CFG of reader and writer, connected with the CSR
// The full address (readerPtr, writerPtr) is included in this class, for the purpose of cross-cluster communication
// The truncated version of address is assigned to aguCfg for the generation of local address which will be consumed by TCDM
// Loopback signal is also included in this class, for the purpose of early judgement
// Also the extension cfg is included in this class
class XDMACfgIO(val param: XDMAParam) extends Bundle {
  val taskID = UInt(4.W)

  // Definition origination = 0 means the data is from local, origination = 1 means the data is from remote
  val originationIsFromLocal  = false
  val originationIsFromRemote = true
  val origination             = Bool()
  val readerPtr               = UInt(param.axiParam.addrWidth.W)
  // val writerPtr = UInt(param.axiParam.addrWidth.W)
  val writerPtr               =
    Vec(
      param.crossClusterParam.maxMulticastDest,
      UInt(param.axiParam.addrWidth.W)
    )

  val axiTransferBeatSize = UInt(param.crossClusterParam.tcdmAddressWidth.W)

  val aguCfg          =
    new AddressGenUnitCfgIO(param =
      AddressGenUnitParam(
        temporalDimension = param.crossClusterParam.maxTemporalDimension,
        numChannel        = param.axiParam.dataWidth / param.crossClusterParam.wordlineWidth,
        outputBufferDepth = param.rwParam.aguParam.outputBufferDepth,
        tcdmSize          = param.crossClusterParam.tcdmSize
      )
    ) // Buffered within AGU
  val readerwriterCfg = new ReaderWriterCfgIO(param.rwParam)
  // The LocalLoopback signal to control the data in reader directly sending back to writer
  val localLoopback   = Bool()
  // The RemoteLoopback signal to control the data in fromRemoteData directly seending back to toRemoteData
  val remoteLoopback  = Bool()
  // Position in a chain (XDMAChainRole). Carried rather than inferred -- see XDMAChainRole.
  val chainRole       = UInt(XDMAChainRole.width.W)
  // 1 = this cfg belongs to a ChainGather. It fixes the TRANSPORT DIRECTION tag: a gather's payload travels
  // head -> tail exactly like a chained write, but the nodes carrying it did not originate the cfg (the head is
  // remote-configured) and the node that did originate it sits at the tail. `origination` therefore cannot decide
  // the direction, which is the third thing it was being asked to decide -- alongside ownership and position.
  val collectiveMode  = Bool()
  // LOCAL ONLY (never crosses the wire): cleared on the cfg the core submits, set on the loopback copy the unroll
  // shifts. It distinguishes the FIRST unroll pass, whose destination is the chain head, from later passes.
  val chainUnrolled   = Bool()

  // The datapath-plugin CSR region: the extension chain (user CSRs + 1 bypass bitmask) followed by the junction
  // bank (user CSRs + 1 enable bitmask). Holding the junction in this vector means a gather chain needs no
  // separate serialized cfg field, since the region crosses the inter-cluster serdes on the writer-side frame.
  val extCfg = Vec(param.pluginCsrNum, UInt(32.W))

  /** The GATHER predicate.
    *
    * A collective gather is a chained transfer whose DataPathJunction is enabled, and the junction bank's enable
    * bitmask is the first word of the junction region of `extCfg` -- a region that crosses the inter-cluster serdes on
    * the writer-side frame, so every hop of a chain sees it. The control plane needs this because a gather node must
    * run its READER concurrently with the chained transfer (it is the source of the junction's local operand), whereas
    * a chained WRITE locks the reader out.
    */
  def junctionEnabled: Bool = if (param.junctionCsrNum == 0) false.B else extCfg(param.extCsrNum).orR

  /** The chain position implied by `origination` + `remoteLoopback` -- what ChainWrite has always used. A frame that is
    * not explicitly stamped takes this, so ChainWrite behaviour is unchanged.
    */
  def chainRoleDefault: UInt =
    Mux(
      origination === originationIsFromLocal.B,
      XDMAChainRole.HEAD.U,
      Mux(remoteLoopback, XDMAChainRole.MIDDLE.U, XDMAChainRole.TAIL.U)
    )

  /** "This node issued the task, so raise its core's finish when the chain retires." Task OWNERSHIP, which is
    * independent of data position: for ChainWrite the initiator is the head, for ChainGather it is the collector at the
    * tail. In both cases the owning node is the one whose cfg started locally.
    */
  def isInitiator: Bool = origination === originationIsFromLocal.B

  // Connect the readerPtr + writerPtr with the CSR list (not removed from the list)
  def connectPtrWithList(csrList: IndexedSeq[UInt]): Unit = {
    var remainingCSR = csrList
    val numCSRPerPtr = (param.axiParam.addrWidth + 31) / 32
    readerPtr := Cat(remainingCSR.take(numCSRPerPtr).reverse)
    remainingCSR = remainingCSR.drop(numCSRPerPtr)
    writerPtr.foreach { i =>
      i := Cat(remainingCSR.take(numCSRPerPtr).reverse)
      remainingCSR = remainingCSR.drop(numCSRPerPtr)
    }
  }

  // Drop the readerPtr + writerPtr from the CSR list
  def dropPtrFromList(csrList: IndexedSeq[UInt]): IndexedSeq[UInt] = {
    var remainingCSR = csrList
    val numCSRPerPtr = (param.axiParam.addrWidth + 31) / 32
    remainingCSR = remainingCSR.drop(numCSRPerPtr)
    writerPtr.foreach { _ =>
      remainingCSR = remainingCSR.drop(numCSRPerPtr)
    }
    remainingCSR
  }

  // Connect the remaining CFG with the CSR list
  def connectWithList(
    csrList: IndexedSeq[UInt]
  ): IndexedSeq[UInt] = {
    origination := originationIsFromLocal.B
    var remaincsrList = csrList
    remaincsrList = aguCfg.connectWithList(remaincsrList)
    remaincsrList = readerwriterCfg.connectWithList(remaincsrList)
    extCfg := remaincsrList.take(extCfg.length)
    remaincsrList = remaincsrList.drop(extCfg.length)
    remaincsrList
  }
}

// The structured class that used by reader / writer hardware
class XDMAIntraClusterCfgIO(param: XDMAParam) extends Bundle {
  val taskID = UInt(4.W)

  // Definition origination = 0 means the data is from local, origination = 1 means the data is from remote
  val originationIsFromLocal  = false
  val originationIsFromRemote = true
  val origination             = Bool()
  val readerPtr               = UInt(param.axiParam.addrWidth.W)
  // val writerPtr = UInt(param.axiParam.addrWidth.W)
  val writerPtr               =
    Vec(
      param.crossClusterParam.maxMulticastDest,
      UInt(param.axiParam.addrWidth.W)
    )

  val axiTransferBeatSize = UInt(param.crossClusterParam.tcdmAddressWidth.W)

  val aguCfg          =
    new AddressGenUnitCfgIO(param = param.rwParam.aguParam) // Buffered within AGU
  val readerwriterCfg = new ReaderWriterCfgIO(param.rwParam)
  // The LocalLoopback signal to control the data in reader directly sending back to writer
  val localLoopback   = Bool()
  // The RemoteLoopback signal to control the data in fromRemoteData directly seending back to toRemoteData
  val remoteLoopback  = Bool()
  // Position in a chain (XDMAChainRole), task ownership, and the ChainGather direction tag; all three are
  // consumed by the accompany cfg.
  val chainRole       = UInt(XDMAChainRole.width.W)
  val isInitiator     = Bool()
  val collectiveMode  = Bool()

  // The datapath-plugin CSR region: the extension chain (user CSRs + 1 bypass bitmask) followed by the junction
  // bank (user CSRs + 1 enable bitmask). Holding the junction in this vector means a gather chain needs no
  // separate serialized cfg field, since the region crosses the inter-cluster serdes on the writer-side frame.
  val extCfg = Vec(param.pluginCsrNum, UInt(32.W))

  // Convert the XDMACfgIO to XDMAIntraClusterCfgIO
  def convertFromXDMACfgIO(cfg: XDMACfgIO): Unit = {
    taskID                   := cfg.taskID
    origination              := cfg.origination
    readerPtr                := cfg.readerPtr
    writerPtr                := cfg.writerPtr
    axiTransferBeatSize      := cfg.axiTransferBeatSize
    aguCfg.addressRemapIndex := cfg.aguCfg.addressRemapIndex
    aguCfg.ptr               := cfg.aguCfg.ptr
    aguCfg.spatialStrides    := cfg.aguCfg.spatialStrides.take(
      aguCfg.spatialStrides.length
    )

    // The unused fields in the temporalStrides should always be 0
    aguCfg.temporalStrides := cfg.aguCfg.temporalStrides.take(
      aguCfg.temporalStrides.length
    )

    // The unused fields in the temporalBounds should always be 1
    aguCfg.temporalBounds := cfg.aguCfg.temporalBounds.take(
      aguCfg.temporalBounds.length
    )

    readerwriterCfg := cfg.readerwriterCfg
    localLoopback   := cfg.localLoopback
    remoteLoopback  := cfg.remoteLoopback
    chainRole       := cfg.chainRole
    isInitiator     := cfg.isInitiator
    collectiveMode  := cfg.collectiveMode
    extCfg          := cfg.extCfg
  }
}

class XDMAInterClusterCfgIO(readerParam: XDMAParam, writerParam: XDMAParam) extends Bundle {
  val taskID              = UInt(4.W)
  val isWriterSide        = Bool()
  val readerPtr           = UInt(readerParam.crossClusterParam.AxiAddressWidth.W)
  // Writer pointer only needs first two elements, as now the broadcast is tackled in XDMACfgIO level
  val writerPtr           = Vec(
    2,
    UInt(readerParam.crossClusterParam.AxiAddressWidth.W)
  )
  val axiTransferBeatSize = UInt(readerParam.crossClusterParam.tcdmAddressWidth.W)
  val spatialStride       = UInt(readerParam.crossClusterParam.tcdmAddressWidth.W)

  val temporalBounds  = Vec(
    readerParam.crossClusterParam.maxTemporalDimension,
    UInt(readerParam.crossClusterParam.tcdmAddressWidth.W)
  )
  val temporalStrides = Vec(
    readerParam.crossClusterParam.maxTemporalDimension,
    UInt(readerParam.crossClusterParam.tcdmAddressWidth.W)
  )
  val enabledChannel  = UInt(readerParam.crossClusterParam.channelNum.W)
  val enabledByte     = UInt((readerParam.crossClusterParam.wordlineWidth / 8).W)

  // Cross-cluster WRITER-side extension config. The DataPathExtension CSRs must cross the die-to-die link so the
  // RECEIVER's writer-side extension is configured to process the incoming remote stream (fromRemote ->
  // writerExtensions -> writer). Cross-cluster extension processing is WRITER-side ONLY (that is where remote data
  // lands), so only the writer extensions' config is carried; the reader side stays bypassed cross-cluster. Sized
  // to the writer's extensions (total userCsrNum + 1 bypass). Meaningful only on the isWriterSide frame (zeroed on
  // the reader/src frame). This is the enabler for the in-fabric cross-cluster collective (F3) + CROSS's
  // cross-chiplet per-channel statistic.
  val writerExtCfg = Vec(writerParam.pluginCsrNum, UInt(32.W))

  // Chain position, stamped by the initiator during the unroll. `isInitiator` is deliberately NOT carried: it is
  // "did this cfg start here", which each node derives from its own `origination`.
  val chainRole      = UInt(XDMAChainRole.width.W)
  val collectiveMode = Bool()

  def convertFromXDMACfgIO(
    writerSide: Boolean,
    cfg:        XDMACfgIO
  ): Unit = {
    taskID                   := cfg.taskID
    isWriterSide             := writerSide.B
    readerPtr                := cfg.readerPtr
    writerPtr(0)             := cfg.writerPtr(0)
    if (writerPtr.length > 1 && cfg.writerPtr.length > 1) writerPtr(1) := cfg.writerPtr(1)
    else writerPtr(1)        := 0.U(0.W)
    axiTransferBeatSize      := cfg.axiTransferBeatSize
    spatialStride            := cfg.aguCfg
      .spatialStrides(0)
      .apply(
        cfg.aguCfg.spatialStrides(0).getWidth - 1,
        log2Ceil(readerParam.crossClusterParam.wordlineWidth / 8)
      )
    temporalStrides          := cfg.aguCfg.temporalStrides.map(
      _.apply(
        cfg.aguCfg.temporalStrides(0).getWidth - 1,
        log2Ceil(readerParam.crossClusterParam.wordlineWidth / 8)
      )
    )
    temporalBounds           := cfg.aguCfg.temporalBounds
    enabledChannel           := cfg.readerwriterCfg.enabledChannel
    enabledByte              := cfg.readerwriterCfg.enabledByte
    chainRole                := cfg.chainRole
    collectiveMode           := cfg.collectiveMode
    // Carry the writer extension config only on the writer-side (dst) frame; on the reader-side (src) frame the
    // cfg is the reader XDMACfgIO whose extCfg is the READER extensions, which are not carried cross-cluster.
    if (writerSide) {
      if (writerExtCfg.length != 0) writerExtCfg := cfg.extCfg
    } else writerExtCfg.foreach(_ := 0.U)
  }

  def convertToXDMACfgIO(readerSide: Boolean): XDMACfgIO = {
    val xdmaCfg = if (readerSide) { Wire(new XDMACfgIO(readerParam)) }
    else { Wire(new XDMACfgIO(writerParam)) }

    xdmaCfg                                                := 0.U.asTypeOf(xdmaCfg)
    xdmaCfg.taskID                                         := taskID
    xdmaCfg.readerPtr                                      := readerPtr
    xdmaCfg.writerPtr(0)                                   := writerPtr(0)
    if (xdmaCfg.writerPtr.length > 1) xdmaCfg.writerPtr(1) := writerPtr(1)
    if (xdmaCfg.writerPtr.length > 2) xdmaCfg.writerPtr.tail.tail.foreach(_ := 0.U(0.W))
    xdmaCfg.axiTransferBeatSize                            := axiTransferBeatSize
    xdmaCfg.aguCfg.ptr                                     := { if (readerSide) readerPtr else writerPtr(0) }
    xdmaCfg.aguCfg.spatialStrides(0)                       := spatialStride ## 0.U(
      log2Ceil(readerParam.crossClusterParam.wordlineWidth / 8).W
    )

    xdmaCfg.aguCfg.temporalStrides := temporalStrides
      .map(
        _ ## 0.U(log2Ceil(readerParam.crossClusterParam.wordlineWidth / 8).W)
      )
      .take(
        xdmaCfg.aguCfg.temporalStrides.length
      )

    xdmaCfg.aguCfg.temporalBounds := temporalBounds.take(
      xdmaCfg.aguCfg.temporalStrides.length
    )

    if (xdmaCfg.extCfg.length != 0) {
      if (readerSide) {
        xdmaCfg.extCfg(0) := 0.U // reader/src side: extensions stay bypassed cross-cluster (bypass bit = 0)
      } else {
        // writer/dst side: drive the receiver's writer extension from the carried writerExtCfg so it is ACTIVE on
        // the incoming remote stream (this is what makes the cross-cluster in-fabric collective / CROSS work).
        if (writerExtCfg.length != 0) xdmaCfg.extCfg := writerExtCfg
        else xdmaCfg.extCfg(0)                       := 0.U
      }
    }

    xdmaCfg.chainRole                      := chainRole
    xdmaCfg.collectiveMode                 := collectiveMode
    xdmaCfg.chainUnrolled                  := false.B // local-only; a received frame is never re-unrolled
    xdmaCfg.readerwriterCfg.enabledChannel := enabledChannel
    xdmaCfg.readerwriterCfg.enabledByte    := enabledByte
    xdmaCfg.origination                    := xdmaCfg.originationIsFromRemote.B
    xdmaCfg
  }
}

// Frame head 1:
//  isWriterSide: 1b
//  TotalFrames: 4b

// Frame head n (n > 1):
//  isWriterSide: 1b
//  Frame Index: 4b

//  taskID: 4b
//  readerPtr: 48b
//  writerPtr: 48b
//  Broadcast writerPtr: 48b
//  axiTransferBeatSize: 16b
//  spatialStride: 16b
//  temporalBounds: 16b * Dim
//  temporalStrides: 16b * Dim
//  enabledChannel: 8b
//  enabledByte: 8b
//  writerExtCfg: 32b * (writer-ext total userCsrNum + 1 bypass)   [MSB; writer-side frame only]

class XDMAInterClusterCfgIOSerializer(readerwriterParam: XDMAParam) extends Module {
  // Disambiguate the generated Verilog module name by the writer-ext CSR total (the quantity that
  // sizes writerExtCfg's Vec, hence this module's port list). Without this, two SEPARATE Chisel
  // elaboration runs with DIFFERENT writer-extension sets (e.g. the chip-level hemaia_xdma_cfg,
  // whose writer_extensions is empty, vs. a per-cluster xdma cfg with real writer extensions) both
  // emit a module literally named "XDMAInterClusterCfgIOSerializer" -- when both land in the SAME
  // Verilog compilation (any HeMAiA SoC-level build), the simulator's work library keeps only ONE
  // definition and every instantiation site expecting the OTHER one's ports fails to elaborate
  // (vopt-2912 "Port ... not found"). This does not affect a standalone single-elaboration build
  // (snax_cluster's own sim flow, or a chiseltest), which is why neither surfaced it.
  override def desiredName: String = "XDMAInterClusterCfgIOSerializer_wext" + readerwriterParam.pluginCsrNum
  val io = IO(new Bundle {
    val cfgIn  = Flipped(Decoupled(new XDMAInterClusterCfgIO(readerwriterParam, readerwriterParam)))
    val cfgOut = Decoupled(UInt(readerwriterParam.axiParam.dataWidth.W))
  })

  // Serialize the entire cfg to one vector
  var cfgSerialized =
    io.cfgIn.bits.collectiveMode ## io.cfgIn.bits.chainRole ## io.cfgIn.bits.enabledByte ## io.cfgIn.bits.enabledChannel ## io.cfgIn.bits.temporalStrides.reverse
      .reduce(
        _ ## _
      ) ## io.cfgIn.bits.temporalBounds.reverse.reduce(
      _ ## _
    ) ## io.cfgIn.bits.spatialStride ## io.cfgIn.bits.axiTransferBeatSize ## io.cfgIn.bits.writerPtr(1) ## io.cfgIn.bits
      .writerPtr(0) ## io.cfgIn.bits.readerPtr ## io.cfgIn.bits.taskID

  // Append the writer-side extension config at the MSB (extracted last on the deserializer, symmetric order).
  if (io.cfgIn.bits.writerExtCfg.length != 0)
    cfgSerialized = io.cfgIn.bits.writerExtCfg.reverse.reduce(_ ## _) ## cfgSerialized

  val frameBodyLength = readerwriterParam.axiParam.dataWidth - 5
  val frameNum        = (cfgSerialized.getWidth + frameBodyLength - 1) / frameBodyLength
  // The frame-count / frame-index head is 4 bits (see below), so at most 15 frames can be addressed. Adding
  // writer extensions widens the serialized cfg; guard the hard ceiling at elaboration (free). Reached only at
  // absurd writer-ext CSR counts today.
  require(frameNum <= 15, s"XDMAInterClusterCfgIOSerializer: $frameNum frames exceeds the 4-bit frame head (max 15)")

  // Pad the zero at the MSB of the CFG so that the data can be aligned to the AXI bus
  cfgSerialized = 0.U((frameBodyLength * frameNum - cfgSerialized.getWidth).W) ## cfgSerialized

  val frames = collection.mutable.Map[Int, UInt]()
  for (i <- 0 until frameNum) {
    frames(i) = cfgSerialized(
      i * frameBodyLength + frameBodyLength - 1,
      i * frameBodyLength
    )
  }

  // Append the frame head to each frame
  for (i <- 0 until frameNum) {
    val frameHead = Wire(UInt(5.W))
    if (i == 0) {
      frameHead := Cat(frameNum.U(4.W), io.cfgIn.bits.isWriterSide)
    } else {
      frameHead := Cat(i.U(4.W), io.cfgIn.bits.isWriterSide)
    }
    frames(i) = Cat(frames(i), frameHead)
  }

  // Use WidthConverter to convert the frames to one port
  val widthConverter = Module(
    new WidthDownConverter(chiselTypeOf(frames(0)), frameNum) {
      override def desiredName: String = "width_down_converter_W_" + frames(0).getWidth + "_D_" + frameNum
    }
  )
  widthConverter.io.in.valid := io.cfgIn.valid
  io.cfgIn.ready          := widthConverter.io.in.ready
  widthConverter.io.in.bits.zipWithIndex.foreach({ case (i, j) =>
    i := frames(j)
  })
  widthConverter.io.start := false.B
  io.cfgOut <> widthConverter.io.out
}

class XDMAInterClusterCfgIODeserializer(readerwriterParam: XDMAParam) extends Module {
  // See XDMAInterClusterCfgIOSerializer's desiredName override for why this is needed.
  override def desiredName: String = "XDMAInterClusterCfgIODeserializer_wext" + readerwriterParam.pluginCsrNum
  val io = IO(new Bundle {
    val cfgIn  = Flipped(Decoupled(UInt(readerwriterParam.axiParam.dataWidth.W)))
    val cfgOut = Decoupled(new XDMAInterClusterCfgIO(readerwriterParam, readerwriterParam))
  })

  val frameBodyLength = readerwriterParam.axiParam.dataWidth - 5
  val frameNum        = (io.cfgOut.bits.getWidth + frameBodyLength - 1) / frameBodyLength
  require(frameNum <= 15, s"XDMAInterClusterCfgIODeserializer: $frameNum frames exceeds the 4-bit frame head (max 15)")
  val frameBody = RegInit(VecInit(Seq.fill(frameNum)(0.U(frameBodyLength.W))))

  val frameIndex   = RegInit(2.U(4.W))
  val frameCounter = Module(new BasicCounter(width = 4, hasCeil = true))

  val isWriterSide = RegInit(false.B)

  // The FSM to control multi-frame cfg transfer
  // States
  val sIdle :: sReceiveMoreFrames :: sSendCfg :: Nil = Enum(3)
  val nextState                                      = Wire(chiselTypeOf(sIdle))
  val currentState                                   = RegNext(nextState, sIdle)
  nextState := currentState

  // Default values
  // The counter's ceiling is the number of frames (only in the first frame)
  frameCounter.io.ceil  := frameIndex
  // The counter's synchronized reset signal should be triggered immediately after all the frames are received
  frameCounter.io.reset := currentState === sSendCfg
  frameCounter.io.tick  := false.B
  io.cfgIn.ready        := false.B
  io.cfgOut.valid       := false.B

  switch(currentState) {
    is(sIdle) {
      when(io.cfgIn.valid) {
        io.cfgIn.ready := true.B
        isWriterSide   := io.cfgIn.bits(0)
        frameBody(0)   := io.cfgIn.bits(readerwriterParam.axiParam.dataWidth - 1, 5)
        when(io.cfgIn.bits(4, 1) === 1.U) {
          // There is only one frame
          nextState := sSendCfg
        }.elsewhen(io.cfgIn.bits(4, 1) > 1.U) {
          frameIndex           := io.cfgIn.bits(4, 1)
          frameCounter.io.tick := true.B
          nextState            := sReceiveMoreFrames
        }
      }
    }
    is(sReceiveMoreFrames) {
      when(io.cfgIn.valid) {
        io.cfgIn.ready       := true.B
        frameCounter.io.tick := true.B
        // Tackle the case when the frame is more than cfg's maximum value
        when(frameCounter.io.value < frameNum.U) {
          frameBody(frameCounter.io.value) := io.cfgIn.bits(readerwriterParam.axiParam.dataWidth - 1, 5)
        }
        // Tackle the case when the frame is less than cfg's maximum value: the remaining frames are all 0 (Init in sSendCfg)
        when(frameCounter.io.lastVal) {
          nextState := sSendCfg
        }
      }
    }
    is(sSendCfg) {
      io.cfgOut.valid := true.B
      when(io.cfgOut.ready) {
        nextState := sIdle
        frameBody.foreach(_ := 0.U)
      }
    }
  }

  // The deserializer to convert the buffered frames to the output cfg
  var cfgSerialized = frameBody.reverse.reduce(_ ## _)
  io.cfgOut.bits.isWriterSide := isWriterSide

  // Assign task ID
  io.cfgOut.bits.taskID := cfgSerialized(3, 0)
  cfgSerialized = cfgSerialized(cfgSerialized.getWidth - 1, 4)
  // Assign the readerPtr
  io.cfgOut.bits.readerPtr := cfgSerialized(
    readerwriterParam.crossClusterParam.AxiAddressWidth - 1,
    0
  )
  cfgSerialized = cfgSerialized(
    cfgSerialized.getWidth - 1,
    readerwriterParam.crossClusterParam.AxiAddressWidth
  )

  // Assign the writerPtr
  io.cfgOut.bits.writerPtr.foreach { i =>
    i := cfgSerialized(
      readerwriterParam.crossClusterParam.AxiAddressWidth - 1,
      0
    )
    cfgSerialized = cfgSerialized(
      cfgSerialized.getWidth - 1,
      readerwriterParam.crossClusterParam.AxiAddressWidth
    )
  }

  // Assign axiTransferBeatSize
  io.cfgOut.bits.axiTransferBeatSize := cfgSerialized(
    readerwriterParam.crossClusterParam.tcdmAddressWidth - 1,
    0
  )
  cfgSerialized = cfgSerialized(
    cfgSerialized.getWidth - 1,
    readerwriterParam.crossClusterParam.tcdmAddressWidth
  )
  // Assign spatialStride
  io.cfgOut.bits.spatialStride := cfgSerialized(
    readerwriterParam.crossClusterParam.tcdmAddressWidth - 1,
    0
  )
  cfgSerialized = cfgSerialized(
    cfgSerialized.getWidth - 1,
    readerwriterParam.crossClusterParam.tcdmAddressWidth
  )
  // Assign temporalBounds
  io.cfgOut.bits.temporalBounds.foreach { i =>
    i := cfgSerialized(
      readerwriterParam.crossClusterParam.tcdmAddressWidth - 1,
      0
    )
    cfgSerialized = cfgSerialized(
      cfgSerialized.getWidth - 1,
      readerwriterParam.crossClusterParam.tcdmAddressWidth
    )
  }
  // Assign temporalStrides
  io.cfgOut.bits.temporalStrides.foreach { i =>
    i := cfgSerialized(
      readerwriterParam.crossClusterParam.tcdmAddressWidth - 1,
      0
    )
    cfgSerialized = cfgSerialized(
      cfgSerialized.getWidth - 1,
      readerwriterParam.crossClusterParam.tcdmAddressWidth
    )
  }
  // Assign enabledChannel
  io.cfgOut.bits.enabledChannel := cfgSerialized(
    readerwriterParam.crossClusterParam.channelNum - 1,
    0
  )
  cfgSerialized = cfgSerialized(
    cfgSerialized.getWidth - 1,
    readerwriterParam.crossClusterParam.channelNum
  )
  // Assign enabledByte
  io.cfgOut.bits.enabledByte := cfgSerialized(
    readerwriterParam.crossClusterParam.wordlineWidth / 8 - 1,
    0
  )
  cfgSerialized = cfgSerialized(
    cfgSerialized.getWidth - 1,
    readerwriterParam.crossClusterParam.wordlineWidth / 8
  )
  // Assign the chain position (symmetric to the serializer, which appends it just above enabledByte)
  io.cfgOut.bits.chainRole := cfgSerialized(XDMAChainRole.width - 1, 0)
  cfgSerialized = cfgSerialized(cfgSerialized.getWidth - 1, XDMAChainRole.width)
  // Assign the ChainGather direction tag
  io.cfgOut.bits.collectiveMode := cfgSerialized(0)
  cfgSerialized = cfgSerialized(cfgSerialized.getWidth - 1, 1)
  // Assign the writer-side extension config (32b per CSR; symmetric to the serializer's MSB prepend). The remaining
  // MSBs are zero padding.
  io.cfgOut.bits.writerExtCfg.foreach { i =>
    i := cfgSerialized(31, 0)
    cfgSerialized = cfgSerialized(cfgSerialized.getWidth - 1, 32)
  }
}

class XDMADataPathCfgIO(axiParam: XDMAAXIParam, crossClusterParam: XDMACrossClusterParam) extends Bundle {
  val readyToTransfer       = Bool()
  val taskID                = UInt(4.W)
  val length                = UInt(crossClusterParam.tcdmAddressWidth.W)
  val taskType              = Bool()
  val taskTypeIsRemoteRead  = false
  val taskTypeIsRemoteWrite = true
  val isFirstChainedWrite   = Bool()
  val isLastChainedWrite    = Bool()
  // Task ownership, independent of data position: the adapter raises this node's core finish only when set.
  val isInitiator           = Bool()
  val src                   = UInt(axiParam.addrWidth.W)
  val dst                   = UInt(axiParam.addrWidth.W)

  def convertFromXDMAIntraClusterCfgIO(
    cfg:            XDMAIntraClusterCfgIO,
    isChainedWrite: Boolean
  ): Unit = {
    taskID := cfg.taskID
    length := cfg.axiTransferBeatSize
    src    := {
      if (isChainedWrite)
        cfg.writerPtr(0)
      else cfg.readerPtr
    }
    dst    := {
      if (isChainedWrite)
        cfg.writerPtr(1)
      else cfg.writerPtr(0)
    }

    // Position comes from the explicitly carried chainRole. For ChainWrite the initiator stamps the same values
    // the origination-based inference produced, so these two bits are unchanged for every existing transfer.
    isFirstChainedWrite := taskType === taskTypeIsRemoteWrite.B && cfg.chainRole === XDMAChainRole.HEAD.U
    isLastChainedWrite  := taskType === taskTypeIsRemoteWrite.B && cfg.chainRole === XDMAChainRole.TAIL.U
    isInitiator         := cfg.isInitiator
  }
}
