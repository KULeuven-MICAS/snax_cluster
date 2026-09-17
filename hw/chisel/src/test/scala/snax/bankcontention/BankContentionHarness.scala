package snax.bankcontention

import chisel3._
import chisel3.util._

import snax.readerWriter._

/** A stream: where it starts, how its address generator walks, and which direction it moves.
  * Mirrors the streamer CSRs a kernel writes (`ptr`, `Xtlbound*`, `Xtlstride*`).
  */
case class StreamSpec(ptr: Int, temporalBounds: Seq[Int], temporalStrides: Seq[Int],
                      isWriter: Boolean = false, fifoDepth: Option[Int] = None,
                      rateShift: Int = 0) {
  def nSteps: Int = temporalBounds.product
  /** Engine steps this stream spans: it moves once every `2^rateShift` of them.
    *
    * Without this every stream runs at one step per cycle, which is the wrong shape for a real
    * kernel and, worse, saturates the banks: with five full-rate streams the engine sits idle
    * 70% of the time and FIFO depth stops mattering for anyone. FlashAttention's operand
    * readers run at rate 1 and its C/D pair at 1/8, and its array is BUSY 91% of the time --
    * a different operating point entirely, and the one where depth earns its keep. */
  def engineSteps: Int = nSteps << rateShift
}

/** Several real streamers competing for one bank array, with an engine driving them in step.
  *
  * WHAT IS RTL HERE AND WHAT IS NOT. The streams are the shipped `Reader` and `Writer` modules
  * -- address generator, the `DataRequestor` that HOLDS an ungranted address until it fires, the
  * `DataResponser`'s in-flight counter, and the `ComplexQueue` whose fill level drives the TCDM
  * priority bit. None of it is re-implemented here.
  *
  * The bank array below IS testbench code, deliberately. The split cluster sets
  * `tcdm.sparse_interconnect: false`, so `SparseInterconnect` is not in its datapath. The policy
  * is small and fully specified -- one grant per bank per cycle, highest asserted priority
  * first, round-robin among equals -- so it is written out here, and it is the same policy the
  * Python arbiter implements. The thing under test is everything upstream of it.
  *
  * WHY A WRITER MATTERS. The priority bit is `count <= 1` for a reader and `count >= depth - 1`
  * for a writer (`ComplexQueue.scala`), so a DEPTH-1 WRITER ASSERTS IT PERMANENTLY. How much
  * that costs a concurrent reader is the question a FIFO budget turns on, and it is the one the
  * reader-only cases could not ask. A cost model that gets it wrong will happily recommend
  * moving depth from the readers to the drain, which is what happened.
  *
  * The engine consumes one step from every reader and produces one for every writer per cycle,
  * which is what an output-stationary array does with its operands and its drain: it advances
  * only when every reader has data and every writer has room.
  */
class BankContentionHarness(
  specs:       Seq[StreamSpec],
  numBanks:    Int,
  numChannel:  Int = 8,
  bufferDepth: Int = 8,
  tcdmSizeKiB: Int = 1024,
  dataWidth:   Int = 64
) extends Module
    with RequireAsyncReset {

  require(specs.nonEmpty, "need at least one stream")
  require(specs.map(_.temporalBounds.length).distinct.length == 1,
          "all streams must declare the same temporal dimension")
  require(specs.map(_.engineSteps).distinct.length == 1,
          "every stream must span the same number of ENGINE steps (nSteps << rateShift)")

  val nStreams = specs.length
  val nSteps   = specs.head.engineSteps
  val byteOffW = log2Ceil(dataWidth / 8)
  val bankSelW = log2Ceil(numBanks)
  val nReq     = nStreams * numChannel

  val io = IO(new Bundle {
    val start   = Input(Bool())
    val done    = Output(Bool())
    val cycles  = Output(UInt(32.W))
    val steps   = Output(UInt(32.W))
    /** Cycles the engine was blocked AND this stream was the (or a) reason -- the "all"
      * attribution, so the figures need no tie-break to be well defined. */
    val stall   = Output(Vec(nStreams, UInt(32.W)))
    val blocked = Output(UInt(32.W))
    /** Requests that asked for a bank and did not get it. */
    val retries = Output(Vec(nStreams, UInt(32.W)))
    /** Cycles this stream asserted the TCDM priority bit. */
    val urgent  = Output(Vec(nStreams, UInt(32.W)))
  })

  def paramFor(spec: StreamSpec) = new ReaderWriterParam(
    spatialBounds     = List(numChannel),
    temporalDimension = specs.head.temporalBounds.length,
    tcdmDataWidth     = dataWidth,
    tcdmSize          = tcdmSizeKiB,
    numChannel        = numChannel,
    dataBufferDepth   = spec.fifoDepth.getOrElse(bufferDepth),
    dynamicPriority   = true
  )

  val readers = specs.zipWithIndex.collect {
    case (sp, i) if !sp.isWriter => i -> Module(new Reader(paramFor(sp), moduleNamePrefix = "bc"))
  }.toMap
  val writers = specs.zipWithIndex.collect {
    case (sp, i) if sp.isWriter => i -> Module(new Writer(paramFor(sp), moduleNamePrefix = "bc"))
  }.toMap

  // ---- configure every stream's AGU from its spec, and hold it ---------------------------
  def cfg(i: Int, aguCfg: AddressGenUnitCfgIO, rwCfg: ReaderWriterCfgIO, start: Bool): Unit = {
    val spec = specs(i)
    aguCfg.ptr := spec.ptr.U
    aguCfg.spatialStrides(0) := (dataWidth / 8).U          // channels are contiguous words
    spec.temporalBounds.zipWithIndex.foreach { case (b, d) => aguCfg.temporalBounds(d) := b.U }
    spec.temporalStrides.zipWithIndex.foreach { case (v, d) => aguCfg.temporalStrides(d) := v.U }
    aguCfg.addressRemapIndex := 0.U
    rwCfg.enabledChannel := ~0.U(numChannel.W)
    rwCfg.enabledByte    := ~0.U((dataWidth / 8).W)
    start                := io.start
  }
  readers.foreach { case (i, r) => cfg(i, r.io.aguCfg, r.io.readerwriterCfg, r.io.start) }
  writers.foreach { case (i, w) => cfg(i, w.io.aguCfg, w.io.readerwriterCfg, w.io.start) }

  // ---- the bank array: one grant per bank, priority first, round-robin on ties -----------
  def flat(s: Int, c: Int): Int = s * numChannel + c
  val reqValid = Wire(Vec(nReq, Bool()))
  val reqBank  = Wire(Vec(nReq, UInt(bankSelW.W)))
  val reqPrio  = Wire(Vec(nReq, Bool()))
  val granted  = Wire(Vec(nReq, Bool()))

  for (s <- 0 until nStreams; c <- 0 until numChannel) {
    val i = flat(s, c)
    val req = if (specs(s).isWriter) writers(s).io.tcdmReq(c) else readers(s).io.tcdmReq(c)
    reqValid(i) := req.valid
    reqBank(i)  := req.bits.addr(bankSelW + byteOffW - 1, byteOffW)
    reqPrio(i)  := req.bits.priority
    req.ready   := granted(i)
    if (!specs(s).isWriter) {                 // one-cycle bank latency, as the crossbar assumes
      readers(s).io.tcdmRsp(c).valid     := RegNext(granted(i) && reqValid(i), false.B)
      readers(s).io.tcdmRsp(c).bits.data := 0.U
    }
  }

  granted.foreach(_ := false.B)
  val rr = RegInit(VecInit(Seq.fill(numBanks)(0.U(log2Ceil(nReq).W))))
  for (b <- 0 until numBanks) {
    // Bitmask arbitration. The obvious way to write round-robin -- build a ROTATED copy of the
    // request vector and take its first set bit -- costs an nReq-way dynamic mux per element,
    // so O(nReq^2) muxes per bank plus a modulo (a real divider, since nReq is not a power of
    // two). At 32 banks and 40 requestors that made the testbench 96% of the circuit and the
    // interpreter crawl. Masking off the bits below the pointer is the same function in O(nReq).
    val asking   = VecInit((0 until nReq).map(i => reqValid(i) && reqBank(i) === b.U)).asUInt
    val urgent   = VecInit((0 until nReq).map(i => reqValid(i) && reqBank(i) === b.U &&
                                                   reqPrio(i))).asUInt
    val eligible = Mux(urgent.orR, urgent, asking)          // priority masks the rest
    val above    = eligible & ~((1.U(nReq.W) << rr(b)).asUInt - 1.U)   // at or after the pointer
    val winner   = Mux(above.orR, PriorityEncoder(above), PriorityEncoder(eligible))
    when(eligible.orR) {
      granted(winner) := true.B
      rr(b) := Mux(winner === (nReq - 1).U, 0.U, winner + 1.U)        // wrap without a modulo
    }
  }

  // ---- the engine: every stream DUE this step must be ready -------------------------------
  val cycles  = RegInit(0.U(32.W))
  val steps   = RegInit(0.U(32.W))

  // A stream at rateShift k moves on one engine step in every 2^k. The others are not consulted.
  def dueOf(i: Int): Bool = {
    val k = specs(i).rateShift
    if (k == 0) true.B else (steps & ((1 << k) - 1).U) === 0.U
  }
  val due = VecInit((0 until nStreams).map(dueOf))

  val readersReady = if (readers.isEmpty) true.B
                     else readers.map { case (i, r) => !due(i) || r.io.data.valid }.reduce(_ && _)
  val writersReady = if (writers.isEmpty) true.B
                     else writers.map { case (i, w) => !due(i) || w.io.data.ready }.reduce(_ && _)
  val allReady     = readersReady && writersReady
  readers.foreach { case (i, r) => r.io.data.ready := allReady && due(i) }
  writers.foreach { case (i, w) =>
    w.io.data.valid := allReady && due(i)
    w.io.data.bits  := 0.U
  }
  val blocked = RegInit(0.U(32.W))
  val stalls  = RegInit(VecInit(Seq.fill(nStreams)(0.U(32.W))))
  val retries = RegInit(VecInit(Seq.fill(nStreams)(0.U(32.W))))
  val urgents = RegInit(VecInit(Seq.fill(nStreams)(0.U(32.W))))
  val running = RegInit(false.B)

  when(io.start) { running := true.B }
  when(steps === nSteps.U) { running := false.B }

  when(running) {
    cycles := cycles + 1.U
    when(allReady) {
      steps := steps + 1.U
    }.otherwise {
      blocked := blocked + 1.U
      for (s <- 0 until nStreams) {
        val notReady = if (specs(s).isWriter) !writers(s).io.data.ready
                       else !readers(s).io.data.valid
        when(due(s) && notReady) { stalls(s) := stalls(s) + 1.U }
      }
    }
    for (s <- 0 until nStreams) {
      val missed = (0 until numChannel).map(c => reqValid(flat(s, c)) && !granted(flat(s, c)))
      retries(s) := retries(s) + PopCount(VecInit(missed))
      // the priority bit as the TCDM sees it, on any channel that is actually asking
      val urg = (0 until numChannel).map(c => reqValid(flat(s, c)) && reqPrio(flat(s, c)))
      when(VecInit(urg).reduce(_ || _)) { urgents(s) := urgents(s) + 1.U }
    }
  }

  io.done    := steps === nSteps.U
  io.cycles  := cycles
  io.steps   := steps
  io.blocked := blocked
  io.stall   := stalls
  io.retries := retries
  io.urgent  := urgents
}
