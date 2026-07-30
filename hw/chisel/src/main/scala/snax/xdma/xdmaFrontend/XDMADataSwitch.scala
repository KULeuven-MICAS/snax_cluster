package snax.xdma.xdmaFrontend

import chisel3._
import chisel3.util._

import snax.DataPathJunction._
import snax.utils._
import snax.xdma.DesignParams.XDMAParam

/** ============================================================================================================
  * `XDMADataSwitch` -- the crossing where the four datapath ports meet, including the gather-node modes.
  * ============================================================================================================
  *
  * {{{
  *   (1) local read    data arriving from the reader (already through the reader extension chain)
  *   (2) to next hop   data leaving toward the remote side
  *   (3) local write   data heading into the writer (before the writer extension chain)
  *   (4) from remote   data arriving from the remote side
  * }}}
  *
  * {{{
  *   mode          dataflow                              block            role
  *   LOCAL         (1) -> (3)                            --               loopback / reshuffle
  *   READ          (1) -> (2)                            --               source (also gather-chain head)
  *   WRITE         (4) -> (3)                            --               plain destination
  *   CHAINWRITE    (4) -> Splitter -> (2) + (3)          SplitterDecoupled broadcast middle hop
  *   CHAINGATHER   ((4), (1)) -> Junction -> (2)         DataPathJunction reduce middle hop
  *   GATHERROOT    ((4), (1)) -> Junction -> (3)         DataPathJunction reduce tail
  * }}}
  *
  * A multicast chain hands each hop a COPY (the splitter). A gather chain hands each hop a FOLD: the arriving
  * partial meets this node's local operand at the junction and only the combined result travels on. The two are
  * protocol inverses -- same chain unroll, same Grant/Finish direction, one different block at the crossing.
  *
  * ---- MODE DERIVATION ----
  *
  * A gather is a chained transfer whose JUNCTION is enabled, and the junction's enable bitmask rides the
  * writer-side `extCfg` region, which crosses the inter-cluster serdes to every hop. So:
  * {{{
  *   gather      = a junction is selected for this transfer  (junction host `active`)
  *   CHAINGATHER = gather && writerRemoteLoopback            (a next hop exists: writerPtr(1) != 0)
  *   GATHERROOT  = gather && !writerRemoteLoopback           (this node is the tail; the answer lands here)
  * }}}
  * Data flows head-to-tail with Grant/Finish tail-to-head, as in CHAINWRITE, so `isFirstChainedWrite` /
  * `isLastChainedWrite` keep their meaning and the AXI adapter is unaffected.
  *
  * ---- WHY A GATHER MIDDLE NODE NEEDS ITS OWN BUSY LEVEL ----
  *
  * `fromRemote...readyToTransfer` -- which arms both the grant manager and the finish FSM -- is derived from the
  * writer's busy level, and in CHAINWRITE that level is raised by `isChainedWrite`, which can only rise from
  * `writerCfg.remoteLoopback && writer.io.busy`. A gather MIDDLE node does not write locally, so `writer.io.busy`
  * never asserts: `readyToTransfer` would never assert, the grant manager would never leave `IDLE`, and the chain
  * would stall with no diagnostic. `isGather` below is therefore raised by a FSM keyed on the LOCAL READER and the
  * JUNCTION, never on `writer.io.busy`, and `io.writerBusy` ORs it in.
  */
class XDMADataSwitch(param: XDMAParam, dataWidth: Int, clusterName: String = "unnamed_cluster")
    extends Module
    with RequireAsyncReset {

  override val desiredName = s"${clusterName}_xdma_datapath_switch"

  private val nJunction   = scala.math.max(1, param.junctionParam.length)
  private val nJunctionCsr = scala.math.max(1, param.junctionParam.map(_.junctionParam.userCsrNum).sum)

  val io = IO(new Bundle {
    // the four ports of the crossing
    val localIn    = Flipped(Decoupled(UInt(dataWidth.W))) // (1) from the reader (post reader-extension)
    val localOut   = Decoupled(UInt(dataWidth.W))          // (3) to the writer (pre writer-extension)
    val toRemote   = Decoupled(UInt(dataWidth.W))          // (2) to the next hop
    val fromRemote = Flipped(Decoupled(UInt(dataWidth.W))) // (4) from the remote side

    // cfg / status
    val readerLocalLoopback  = Input(Bool())
    val writerLocalLoopback  = Input(Bool())
    val writerRemoteLoopback = Input(Bool())
    val writerStart          = Input(Bool())
    val writerBusyRaw        = Input(Bool()) // writer.io.busy -- the RAW level, before any chain/gather term
    val readerBusy           = Input(Bool())

    val junctionCfg = new Bundle {
      val enable  = Input(UInt(nJunction.W))
      val userCsr = Input(Vec(nJunctionCsr, UInt(32.W)))
    }

    val isChainedWrite  = Output(Bool())
    val isGather        = Output(Bool()) // a gather transfer is in flight at this node (either gather mode)
    val isGatherMid     = Output(Bool()) // ... and it forwards to a next hop (CHAINGATHER)
    val isGatherRoot    = Output(Bool()) // ... and it is the tail (GATHERROOT)
    val writerBusy      = Output(Bool()) // the composed level the rest of the datapath must use
    val junctionBusy    = Output(Bool())
    val junctionStarved = Output(Bool())
    val junctionCfgErr  = Output(Bool()) // O5: the armed junction cannot honour its configuration word
  })

  // ============================ the junction bank ============================
  val junctionHost = Module(new DataPathJunctionHost(param.junctionParam, dataWidth, clusterName))
  junctionHost.io.cfg.enable  := io.junctionCfg.enable
  junctionHost.io.cfg.userCsr := io.junctionCfg.userCsr
  junctionHost.io.start       := io.writerStart

  val gatherCfg = junctionHost.io.active // a junction is selected by this transfer's cfg

  // ============================ mode state machines ============================
  // CHAINWRITE. Suppressed when the transfer is a gather, because a gather middle node's writer never goes busy
  // and this FSM keys on exactly that.
  val stateIdle :: stateChainedWrite :: stateChainedWriteWait :: Nil = Enum(3)

  val isChainedWrite = WireInit(false.B)
  val nextState      = Wire(chiselTypeOf(stateIdle))
  val currentState   = RegNext(nextState, stateIdle)
  nextState := currentState

  switch(currentState) {
    is(stateIdle) {
      // Mealy FSM to pull up isChainedWrite
      when(io.writerRemoteLoopback && !gatherCfg && io.writerBusyRaw) {
        nextState      := stateChainedWrite
        isChainedWrite := true.B
      }
    }
    is(stateChainedWrite) {
      isChainedWrite := true.B
      when(~io.writerBusyRaw) { nextState := stateChainedWriteWait }
    }
    is(stateChainedWriteWait) {
      // Moore FSM to pull down isChainedWrite
      isChainedWrite := true.B
      when(~io.fromRemote.valid) { nextState := stateIdle }
    }
  }

  // GATHER. Keyed on the LOCAL READER and the JUNCTION, never on `writer.io.busy`: a gather middle node writes
  // nothing locally, so keying on the writer would stall the chain silently.
  val gIdle :: gActive :: gWait :: Nil = Enum(3)

  val isGather      = WireInit(false.B)
  val gNextState    = Wire(chiselTypeOf(gIdle))
  val gCurrentState = RegNext(gNextState, gIdle)
  gNextState := gCurrentState

  // The ENTRY condition depends only on signals outside the switch's own dataflow. `isGather` gates the splitter
  // that feeds the junction, so making entry depend on the junction's busy level would close a combinational loop
  // (isGather -> splitter -> junction.a.valid -> junction.busy -> isGather). Entry keys on the local reader /
  // writer only; the junction's busy level is consulted for the EXIT decision, which is registered, so a fold
  // still in flight cannot drop the mode early.
  val gatherTrigger = io.readerBusy || io.writerBusyRaw
  val gatherWorking = gatherTrigger || junctionHost.io.busy

  switch(gCurrentState) {
    is(gIdle) {
      when(gatherCfg && gatherTrigger) {
        gNextState := gActive
        isGather   := true.B
      }
    }
    is(gActive) {
      isGather := true.B
      when(~gatherWorking) { gNextState := gWait }
    }
    is(gWait) {
      isGather := true.B
      when(~io.fromRemote.valid) { gNextState := gIdle }
    }
  }

  val isGatherMid  = isGather && io.writerRemoteLoopback
  val isGatherRoot = isGather && !io.writerRemoteLoopback

  io.isChainedWrite := isChainedWrite
  io.isGather       := isGather
  io.isGatherMid    := isGatherMid
  io.isGatherRoot   := isGatherRoot
  // A busy level that does not depend on the local writer ever going busy, so a gather middle node still arms
  // the grant manager and the finish FSM.
  io.writerBusy      := io.writerBusyRaw | isChainedWrite | isGather
  io.junctionBusy    := junctionHost.io.busy
  io.junctionStarved := junctionHost.io.starved
  io.junctionCfgErr  := junctionHost.io.cfgerr

  // ============================ the crossing datapath ============================
  // (1) reader side: either loop back locally, or head for the crossing.
  val localLoopbackDemux = Module(
    new DemuxDecoupled(UInt(dataWidth.W), numOutput = 2) {
      override def desiredName = clusterName + "_xdma_datapath_local_demux"
    }
  )
  val localLoopbackMux = Module(
    new MuxDecoupled(UInt(dataWidth.W), numInput = 2) {
      override def desiredName = clusterName + "_xdma_datapath_local_mux"
    }
  )
  localLoopbackDemux.io.sel := io.readerLocalLoopback
  localLoopbackMux.io.sel   := io.writerLocalLoopback
  localLoopbackDemux.io.in <> io.localIn
  localLoopbackMux.io.out <> io.localOut
  localLoopbackDemux.io.out(1) <> localLoopbackMux.io.in(1)

  val readerToCrossing = Wire(Decoupled(UInt(dataWidth.W)))
  val crossingToWriter = Wire(Decoupled(UInt(dataWidth.W)))
  localLoopbackDemux.io.out(0) <> readerToCrossing
  localLoopbackMux.io.in(0) <> crossingToWriter

  // (4) from remote: to the writer in the non-gather modes, additionally forwarded in CHAINWRITE. In BOTH gather
  // modes the local write is suppressed -- the writer must receive the FOLD, not the raw partial.
  val remoteSplitter = Module(
    new SplitterDecoupled(UInt(dataWidth.W), numOutput = 2) {
      override def desiredName = clusterName + "_xdma_datapath_remote_splitter"
    }
  )
  remoteSplitter.io.sel(0) := !isGather
  remoteSplitter.io.sel(1) := isChainedWrite || isGather

  val remoteDirectToWriter = Wire(Decoupled(UInt(dataWidth.W)))
  val remoteForwarded      = Wire(Decoupled(UInt(dataWidth.W)))
  remoteSplitter.io.out(0) <> remoteDirectToWriter
  remoteSplitter.io.out(1) <> remoteForwarded

  // the forwarded remote stream is either copied onward (CHAINWRITE) or folded (gather)
  val forwardDemux = Module(
    new DemuxDecoupled(UInt(dataWidth.W), numOutput = 2) {
      override def desiredName = clusterName + "_xdma_datapath_forward_demux"
    }
  )
  forwardDemux.io.sel := isGather
  forwardDemux.io.in <> remoteForwarded

  // the local operand is either sent straight out (READ / chain head) or folded (gather)
  val localReadDemux = Module(
    new DemuxDecoupled(UInt(dataWidth.W), numOutput = 2) {
      override def desiredName = clusterName + "_xdma_datapath_localread_demux"
    }
  )
  localReadDemux.io.sel := isGather
  localReadDemux.io.in <> readerToCrossing

  // THE JUNCTION: operand a = the arriving remote partial, operand b = this node's local operand.
  junctionHost.io.data.a <> forwardDemux.io.out(1)
  junctionHost.io.data.b <> localReadDemux.io.out(1)

  // the fold continues along the chain (CHAINGATHER) or lands in the local writer (GATHERROOT)
  val gatherDemux = Module(
    new DemuxDecoupled(UInt(dataWidth.W), numOutput = 2) {
      override def desiredName = clusterName + "_xdma_datapath_gather_demux"
    }
  )
  gatherDemux.io.sel := isGatherRoot
  gatherDemux.io.in <> junctionHost.io.data.out

  // (2) to the next hop: the local read, the CHAINWRITE copy, or the CHAINGATHER fold
  val toRemoteMux = Module(
    new MuxDecoupled(UInt(dataWidth.W), numInput = 3) {
      override def desiredName = clusterName + "_xdma_datapath_remote_mux"
    }
  )
  toRemoteMux.io.sel := Mux(isGather, 2.U, Mux(isChainedWrite, 1.U, 0.U))
  toRemoteMux.io.in(0) <> localReadDemux.io.out(0)
  toRemoteMux.io.in(1) <> forwardDemux.io.out(0)
  toRemoteMux.io.in(2) <> gatherDemux.io.out(0)
  io.toRemote <> toRemoteMux.io.out

  // (3) to the writer: the raw remote stream, or the GATHERROOT fold
  val writerSrcMux = Module(
    new MuxDecoupled(UInt(dataWidth.W), numInput = 2) {
      override def desiredName = clusterName + "_xdma_datapath_writersrc_mux"
    }
  )
  writerSrcMux.io.sel := isGatherRoot
  writerSrcMux.io.in(0) <> remoteDirectToWriter
  writerSrcMux.io.in(1) <> gatherDemux.io.out(1)
  writerSrcMux.io.out <> crossingToWriter

  // The remote stream may only enter while this node is armed for it. The composed busy level lets a gather
  // middle node, whose local writer stays idle, accept beats.
  remoteSplitter.io.in.valid := io.fromRemote.valid && io.writerBusy
  remoteSplitter.io.in.bits  := io.fromRemote.bits
  io.fromRemote.ready        := remoteSplitter.io.in.ready && io.writerBusy
}
