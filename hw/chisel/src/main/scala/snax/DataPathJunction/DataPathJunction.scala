package snax.DataPathJunction

import chisel3._
import chisel3.util._

import snax.utils._

/** ============================================================================================================
  * `DataPathJunction` -- the 2-in / 1-out operator class of the data mover.
  * ============================================================================================================
  *
  * A `DataPathExtension` is a single-stream, in-line plugin that TRANSFORMS a stream -- layout, transpose, cast,
  * per-element map/reduce. Its ABI is one `data_i`, one `data_o`.
  *
  * A collective fold is not that shape. It consumes TWO streams -- the arriving remote partial and this node's local
  * operand -- and produces ONE.
  *
  * ==== Extensions transform a stream. Junctions fold two streams into one. ====
  *
  * {{{
  *                       DataPathExtension                   DataPathJunction
  *   arity               1 -> 1                              2 -> 1
  *   where               in-line in the reader/writer chain  at the crossing, in the data switch
  *   granularity         per element (map, cast, layout)     per collective (fold / combine)
  *   state               streaming or slot-accumulated       stateless in the operand pair
  *   example             StreamMapRt, StreamCastRt           UnifiedJunction
  * }}}
  *
  * The split follows the wiring: the local operand entering a junction has already traversed the reader extension
  * chain, so the datapath is a per-element stage (Extension: `map`) followed by a per-collective stage (Junction:
  * `combine`).
  *
  * TWO PROPERTIES EVERY JUNCTION INHERITS FROM THIS CLASS:
  *
  *   1. BYPASS DISCIPLINE. With `enable_i = 0` the junction is a wire from `a_i` to `out_o` -- byte-identical to a raw
  *      forward, and `b_i` is not consumed, so a disabled junction is functionally inert.
  *   2. STARVATION VISIBILITY. A two-stream join stalls indefinitely if the two operand streams do not carry the same
  *      number of beats, and beat-count equality is a software contract on two independently dispatched cfgs with no
  *      hardware backstop. Every junction therefore exposes `starved_o`, raised when one side has waited on the other
  *      for longer than `starveLimit` cycles.
  *
  * IMPLEMENTATION CONTRACT for a concrete junction:
  *   1. drive `jct_a_i` / `jct_b_i` from your datapath inputs,
  *   2. drive `jct_data_o` from your datapath output,
  *   3. read config from `jct_csr_i`, the new-stream pulse from `jct_start_i`,
  *   4. drive `jct_busy_o` low only when nothing is in flight.
  */
class JunctionParam(
  val moduleName:   String,
  val userCsrNum:   Int,
  val dataWidth:    Int,
  val starveLimit:  Int = 4096
) {
  require(userCsrNum >= 1, "JunctionParam: at least one user CSR is required")
  require(starveLimit >= 16, "JunctionParam: starveLimit must be >= 16")
}

/** Parent (abstract) class for the generation params of a junction. The `Has...` / `instantiate` split lets the xDMA
  * generator read a junction's parameters (CSR count, data width) before elaborating it.
  */
abstract class HasDataPathJunction {
  implicit val junctionParam: JunctionParam

  def namePostfix = "_DataPathJunction_" + junctionParam.moduleName
  def instantiate(clusterName: String): DataPathJunction
}

abstract class DataPathJunction(implicit junctionParam: JunctionParam) extends Module with RequireAsyncReset {

  val io = IO(new Bundle {
    val csr_i     = Input(Vec(junctionParam.userCsrNum, UInt(32.W)))
    val start_i   = Input(Bool()) // a new stream is coming: re-arm the internal flow control
    val enable_i  = Input(Bool()) // 0 => the junction is a transparent wire from a_i to out_o
    val a_i       = Flipped(Decoupled(UInt(junctionParam.dataWidth.W))) // the arriving remote stream
    val b_i       = Flipped(Decoupled(UInt(junctionParam.dataWidth.W))) // the local operand stream
    val out_o     = Decoupled(UInt(junctionParam.dataWidth.W))
    val busy_o    = Output(Bool())
    val starved_o = Output(Bool()) // watchdog: one operand stream has starved the join
  })

  private[this] val bypass_data = Wire(Decoupled(UInt(junctionParam.dataWidth.W)))

  // ---- signals under the concrete junction's namespace ----
  val jct_a_i = Wire(Decoupled(UInt(junctionParam.dataWidth.W)))
  dontTouch(jct_a_i)
  val jct_b_i = Wire(Decoupled(UInt(junctionParam.dataWidth.W)))
  dontTouch(jct_b_i)
  val jct_data_o = Wire(Decoupled(UInt(junctionParam.dataWidth.W)))
  dontTouch(jct_data_o)
  val jct_csr_i   = io.csr_i
  val jct_start_i = io.start_i
  val jct_busy_o  = Wire(Bool())
  dontTouch(jct_busy_o)

  // ---- bypass structure on the a-path: a demux/mux pair steered by enable_i ----
  private[this] val inputDemux = Module(
    new DemuxDecoupled(UInt(junctionParam.dataWidth.W), numOutput = 2) {
      override def desiredName = "DataPathJunction_Demux_W" + junctionParam.dataWidth.toString
    }
  )
  inputDemux.io.sel := io.enable_i
  inputDemux.io.in <> io.a_i
  inputDemux.io.out(1) <> jct_a_i    // enabled: a_i feeds the junction
  inputDemux.io.out(0) <> bypass_data // disabled: a_i is forwarded verbatim

  private[this] val outputMux = Module(
    new MuxDecoupled(UInt(junctionParam.dataWidth.W), numInput = 2) {
      override def desiredName = "DataPathJunction_Mux_W" + junctionParam.dataWidth.toString
    }
  )
  outputMux.io.sel := io.enable_i
  outputMux.io.out <> io.out_o
  outputMux.io.in(1) <> jct_data_o
  outputMux.io.in(0) <> bypass_data

  // The b-path has no bypass destination: when the junction is disabled the local operand is simply not consumed
  // (the switch does not start the local reader in a non-collective mode).
  jct_b_i.valid := io.b_i.valid && io.enable_i
  jct_b_i.bits  := io.b_i.bits
  io.b_i.ready  := jct_b_i.ready && io.enable_i

  io.busy_o := jct_busy_o || (io.enable_i && (io.a_i.valid || io.b_i.valid))

  // ---- starvation watchdog ----
  // Counts consecutive cycles in which the join is enabled and exactly one operand is offered. A legitimate skew
  // (the remote hop running ahead of the local reader) clears the counter the moment a pair fires.
  private[this] val halfOffered = io.enable_i && (jct_a_i.valid ^ jct_b_i.valid)
  private[this] val paired      = jct_a_i.valid && jct_b_i.valid
  private[this] val starveCnt   = RegInit(0.U(log2Ceil(junctionParam.starveLimit + 1).W))
  when(io.start_i || paired || !halfOffered) {
    starveCnt := 0.U
  }.elsewhen(starveCnt =/= junctionParam.starveLimit.U) {
    starveCnt := starveCnt + 1.U
  }
  io.starved_o := starveCnt === junctionParam.starveLimit.U
}

/** ============================================================================================================
  * `DataPathJunctionHost` -- instantiates the configured junctions at ONE crossing and selects between them.
  * ============================================================================================================
  *
  * Junctions at a crossing are ALTERNATIVES, not a chain: extensions compose in series, but a second 2->1 stage would
  * need a third stream. The host routes the operand pair to the one junction whose enable bit is set, and falls back
  * to a transparent a -> out forward when none is.
  *
  * The `enable` bitmask and the concatenated user CSRs share the layout `DataPathExtensionHost` uses, so a junction's
  * config rides the xDMA ext-cfg CSR region and crosses the inter-cluster serdes with it.
  */
class DataPathJunctionHostIO(junctionList: Seq[HasDataPathJunction], dataWidth: Int = 512) extends Bundle {
  val data = new Bundle {
    val a   = Flipped(Decoupled(UInt(dataWidth.W)))
    val b   = Flipped(Decoupled(UInt(dataWidth.W)))
    val out = Decoupled(UInt(dataWidth.W))
  }
  val cfg = new Bundle {
    val enable  = Input(UInt(scala.math.max(1, junctionList.length).W))
    val userCsr = Input(Vec(scala.math.max(1, junctionList.map(_.junctionParam.userCsrNum).sum), UInt(32.W)))
  }
  val start   = Input(Bool())
  val busy    = Output(Bool())
  val active  = Output(Bool()) // any junction selected: the switch uses this to arm the collective dataflow
  val starved = Output(Bool())

  /** Consume the leading `1 + sum(userCsrNum)` CSRs of `csrList` -- enable bitmask first, then the per-junction user
    * CSRs -- and return the remainder.
    */
  def connectCfgWithList(csrList: IndexedSeq[UInt]): IndexedSeq[UInt] = {
    var remaincsrList = csrList
    if (junctionList.isEmpty) {
      cfg := DontCare
    } else {
      cfg.enable   := remaincsrList.head
      remaincsrList = remaincsrList.tail
      val n = junctionList.map(_.junctionParam.userCsrNum).sum
      cfg.userCsr  := remaincsrList.take(n)
      remaincsrList = remaincsrList.drop(n)
    }
    remaincsrList
  }
}

class DataPathJunctionHost(
  junctionList:     Seq[HasDataPathJunction],
  dataWidth:        Int    = 512,
  moduleNamePrefix: String = "unnamed_cluster"
) extends Module {
  override def desiredName = s"${moduleNamePrefix}_DataPathJunctionHost"
  val io                   = IO(new DataPathJunctionHostIO(junctionList, dataWidth = dataWidth))

  if (junctionList.isEmpty) {
    // No junction configured: the host is a wire, and the collective modes are unreachable.
    io.data.out <> io.data.a
    io.data.b.ready := false.B
    io.busy         := false.B
    io.active       := false.B
    io.starved      := false.B
  } else {
    var remainingCSR = io.cfg.userCsr.toIndexedSeq
    val enables      = io.cfg.enable.asBools.take(junctionList.length)

    val junctions = junctionList.zipWithIndex.map { case (item, index) =>
      require(
        item.junctionParam.dataWidth == dataWidth,
        s"Data width of the junction (${item.junctionParam.dataWidth}) does not match the host ($dataWidth)"
      )
      val junction = item.instantiate(moduleNamePrefix)
      junction.io.start_i  := io.start
      junction.io.enable_i := enables(index)
      for (i <- junction.io.csr_i) {
        i := remainingCSR.head
        remainingCSR = remainingCSR.tail
      }
      junction.suggestName(s"${moduleNamePrefix}_Junction_${index}_${item.junctionParam.moduleName}")
      junction
    }

    if (remainingCSR.nonEmpty)
      println("Debug: Some remaining CSRs are unconnected at the junction host. Check the code.")

    val anyEnable = enables.reduce(_ || _)
    // The operand pair goes to exactly one junction. Two enable bits set at once would OR two folds together
    // through the Mux1H below, so flag it.
    assert(PopCount(VecInit(enables)) <= 1.U,
           "DataPathJunctionHost: at most one junction may be enabled per transfer")

    // Operand routing: only the selected junction sees `valid`, so the others' internal bypass stays quiet.
    junctions.zip(enables).foreach { case (j, en) =>
      j.io.a_i.valid    := io.data.a.valid && en
      j.io.a_i.bits     := io.data.a.bits
      j.io.b_i.valid    := io.data.b.valid && en
      j.io.b_i.bits     := io.data.b.bits
      j.io.out_o.ready  := io.data.out.ready && en
    }

    def sel(f: DataPathJunction => Bool): Bool =
      junctions.zip(enables).map { case (j, en) => f(j) && en }.reduce(_ || _)
    def selBits(f: DataPathJunction => UInt): UInt =
      Mux1H(enables, junctions.map(f))

    io.data.a.ready   := Mux(anyEnable, sel(_.io.a_i.ready), io.data.out.ready)
    io.data.b.ready   := sel(_.io.b_i.ready)
    io.data.out.valid := Mux(anyEnable, sel(_.io.out_o.valid), io.data.a.valid)
    io.data.out.bits  := Mux(anyEnable, selBits(_.io.out_o.bits), io.data.a.bits)

    io.busy    := junctions.map(_.io.busy_o).reduce(_ || _)
    io.active  := anyEnable
    io.starved := sel(_.io.starved_o)
  }
}
