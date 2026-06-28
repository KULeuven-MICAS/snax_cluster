package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._
import fp_native._

/** StreamElementwise: per-lane FP binary combine of `operandCount` consecutive input beats, emitted as ONE
  * full output beat (NO horizontal collapse — the element-wise sibling of StreamReduce).
  *
  *   - Transport: configurable element type (FP16/BF16/FP8/FP32, set by the op-set; e.g. 32 FP16 lanes per
  *     512-bit beat). Internal math: FP32 (widen on input, narrow on output). op = MUL | ADD, runtime CSR.
  *   - The operands arrive as consecutive beats on the SINGLE input stream — the AGU interleaves them (an
  *     inner temporal dim of count=operandCount striding between the operand L1 regions), exactly as the
  *     ElementwiseAdd app does. For a binary op operandCount=2: out(i) = beat0(i) op beat1(i).
  *
  * Reuses ElementwiseAdd's operand-count accumulate-then-emit FSM + StreamReduce's FP32-internal op-set and
  * `computeLanes` time-mux, but emits the per-lane partials (Cat(regs)) instead of a reduced scalar (no
  * reduceTree, no tap).
  *
  * CSR layout: csr(0) = operandCount (#beats combined per output, 0->1); csr(1) bits[7:0] = op
  * (0=MUL, 1=ADD; present/used only when >1 op is built).
  *
  * Configurable from the hjson: `op` (the LIST of supported ops, a subset of {MUL, ADD}), `computeLanes`
  * (number of physical combine ALUs, time-muxed over `subCycles` = lanes/computeLanes cycles per beat;
  * computeLanes = lanes ⇒ the fully-parallel 1-cycle/beat path), and `elementWidth` (must match the op-set
  * precision: FP16/BF16⇒16, FP8⇒8, FP32⇒32). All REQUIRED, e.g.
  * {elementWidth:16, computeLanes:8, op:["MUL_FP16","ADD_FP16"]}.
  */
class HasStreamElementwise(
  computeLanes: Int,
  op:           Seq[String], // each entry "<OP>_<PRECISION>", e.g. "MUL_FP16"
  elementWidth: Int,         // transport element width (bits); must match the op-set precision
  dataWidth:    Int = 512,
  fpPipe:       Int = 1      // internal pipeline depth of each FP unit (timing cut knob)
) extends HasDataPathExtension {
  private val (ops, transport) = OpSpec.parse(op, Set("MUL", "ADD"), "HasStreamElementwise") // validate op names + precision
  OpSpec.checkWidth(elementWidth, transport, "HasStreamElementwise")
  require(computeLanes > 0, "HasStreamElementwise: computeLanes must be > 0")
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "StreamElementwise",
      userCsrNum = if (ops.size > 1) 2 else 1, // operandCount + (op-select only when >1 op listed)
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): StreamElementwise =
    Module(new StreamElementwise(computeLanes, op, elementWidth, fpPipe) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamElementwise(
  computeLanesParam: Int = 0,
  op:                Seq[String] = Seq("MUL_FP16", "ADD_FP16"),
  elementWidth:      Int = 16,
  fpPipeParam:       Int = 1,
  pipelined:         Boolean = true
)(
  implicit extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers.{widen => widenT, narrow => narrowT}

  // transport (element) precision comes from the op-set; internal compute stays FP32
  val (ops, transport) = OpSpec.parse(op, Set("MUL", "ADD"), "StreamElementwise")
  OpSpec.checkWidth(elementWidth, transport, "StreamElementwise")
  val accWidth = 32 // FP32 internal
  val lanes    = extensionParam.dataWidth / elementWidth
  val computeLanes = if (computeLanesParam <= 0 || computeLanesParam > lanes) lanes else computeLanesParam
  require(lanes % computeLanes == 0, "StreamElementwise: lanes must be a multiple of computeLanes")
  val subCycles    = lanes / computeLanes

  val hasMul  = ops.contains("MUL")
  val hasAdd  = ops.contains("ADD")
  val multiOp = ops.size > 1

  def OP_MUL = 0.U
  def OP_ADD = 1.U

  val FP32_ZERO = 0.U(accWidth.W)

  val csrOperandCount = ext_csr_i(0)
  val operandCount    = Mux(csrOperandCount === 0.U, 1.U(16.W), csrOperandCount(15, 0))
  // op-select CSR exists only when >1 op is built; otherwise the single op is fixed
  val opcode: UInt = if (multiOp) ext_csr_i(1)(7, 0) else if (hasAdd) OP_ADD else OP_MUL

  // ---- pipeline depth (timing) ----------------------------------------------------------------
  // Each FP unit is internally pipelined by `fpPipe` (cfg "cutting" knob); plus laneLat to register the
  // time-mux lane select. Per-lane combine latency = laneLat + fpPipe (widen) + fpPipe (mul/add).
  val fpPipe  = if (pipelined) fpPipeParam else 0
  val laneLat = if (pipelined) 1 else 0
  val accLat  = laneLat + 2 * fpPipe
  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  def fmul(a: UInt, b: UInt): UInt = {
    val m = Module(new FpMul(FP32, FP32, FP32, fpPipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }
  def fadd(a: UInt, b: UInt): UInt = {
    val m = Module(new FpAdd(FP32, FP32, FP32, fpPipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }
  def widen(h:  UInt): UInt = widenT(h, transport, fpPipe) // transport -> FP32
  def narrow(f: UInt): UInt = narrowT(f, transport)        // FP32 -> transport (output; not timing-critical)

  // one lane's new partial: combine `laneIn` into `prev` per the active op (only built ops built).
  // first beat seeds the partial with the (widened) input; later beats fold in via mul/add.
  // Pipeline: register the lane select (laneLat), widen (fpPipe), then mul/add (fpPipe); the seed and the
  // accumulator are ShiftRegister-aligned to the combine output. Total issue->result latency = accLat.
  def accLane(laneIn: UInt, prev: UInt, first: Bool): UInt = {
    val laneR   = sr(laneIn)                                  // +laneLat
    val widened = widen(laneR)                                // +fpPipe
    val prevA   = ShiftRegister(prev,  laneLat + fpPipe)
    val firstA  = ShiftRegister(first, laneLat + fpPipe)
    val mulRes  = if (hasMul) fmul(prevA, widened) else FP32_ZERO // +fpPipe
    val addRes  = if (hasAdd) fadd(prevA, widened) else FP32_ZERO
    val seedD   = ShiftRegister(widened, fpPipe)             // first-beat seed, aligned to the combine output
    val firstAA = ShiftRegister(firstA, fpPipe)
    val accMul  = if (hasMul) Mux(firstAA, seedD, mulRes) else FP32_ZERO
    val accAdd  = if (hasAdd) Mux(firstAA, seedD, addRes) else FP32_ZERO
    if (multiOp) Mux(opcode === OP_ADD, accAdd, accMul)
    else if (hasAdd) accAdd
    else accMul
  }

  // ---- per-lane FP32 partials ----
  val regs = RegInit(VecInit(Seq.fill(lanes)(0.U(accWidth.W))))

  // per-lane result beat: narrow each FP32 partial back to transport (lane 0 in the low bits)
  val resultBeat = Cat((0 until lanes).map(i => narrow(regs(i))).reverse)

  // ---- unified time-mux + pipeline FSM ----
  // `computeLanes` pipelined ALUs sweep the beat over `subCycles` sub-groups (one issued/cycle); results
  // retire `accLat` cycles later into the per-lane partials. After the last beat of a row the full beat is
  // emitted. computeLanes == lanes ⇒ subCycles == 1. Per-beat cost = subCycles + accLat cycles.
  val inBeat   = Reg(UInt((lanes * elementWidth).W))
  val inLanes  = inBeat.asTypeOf(Vec(lanes, UInt(elementWidth.W)))
  val busy     = RegInit(false.B)
  val outValid = RegInit(false.B)
  val beatCnt  = RegInit(0.U(16.W))
  val firstBeat = beatCnt === 0.U
  val lastBeat  = beatCnt === (operandCount - 1.U)
  val total     = subCycles - 1 + accLat
  val step      = RegInit(0.U(log2Ceil(total + 1).max(1).W))

  // index of lane (s*computeLanes + j) into the `lanes`-wide Vec, width-exact to silence W004
  def li(s: UInt, j: Int): UInt = (s * computeLanes.U + j.U)(log2Ceil(lanes) - 1, 0)

  val issuing  = busy && (step < subCycles.U)
  val subIssue = step
  val res = Wire(Vec(computeLanes, UInt(accWidth.W)))
  for (j <- 0 until computeLanes)
    res(j) := accLane(inLanes(li(subIssue, j)), regs(li(subIssue, j)), firstBeat)

  val subRetire   = ShiftRegister(subIssue, accLat)
  val retireValid = ShiftRegister(issuing, accLat, false.B, true.B)
  when(busy && retireValid) {
    for (j <- 0 until computeLanes) regs(li(subRetire, j)) := res(j)
  }

  ext_data_i.ready := !busy && !outValid

  when(ext_start_i) {
    busy := false.B; outValid := false.B; step := 0.U; beatCnt := 0.U
  }.elsewhen(busy) {
    when(step === total.U) {
      busy := false.B; step := 0.U
      when(lastBeat) { outValid := true.B }.otherwise { beatCnt := beatCnt + 1.U }
    }.otherwise {
      step := step + 1.U
    }
  }.elsewhen(ext_data_i.fire) {
    inBeat := ext_data_i.bits; busy := true.B; step := 0.U
  }
  when(outValid && ext_data_o.ready && !ext_start_i) { outValid := false.B; beatCnt := 0.U }

  ext_data_o.bits  := resultBeat
  ext_data_o.valid := outValid
  ext_busy_o       := busy || outValid || (beatCnt =/= 0.U)
}
