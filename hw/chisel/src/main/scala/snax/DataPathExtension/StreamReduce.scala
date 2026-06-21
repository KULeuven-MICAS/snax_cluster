package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._

/** StreamReduce: vertical (across-beat) + horizontal (across-lane) FP reduction of a streamed row to a
  * single scalar, splatted across all lanes of one output beat.
  *
  *   - Transport: configurable element type (FP16/BF16/FP8/FP32, set by the op-set; e.g. 32 FP16 lanes per
  *     512-bit beat). Internal math: FP32 (widen on input, narrow on output).
  *   - op = MAX | ADD | SUMSQ (SUMSQ squares each lane before summing). Selected at runtime via CSR.
  *
  * Reuses the ElementwiseAdd/MaxPool accumulate-then-emit FSM. The horizontal cross-lane collapse is a
  * balanced reduceTree (tree order; small numeric tolerance accepted per design decision #4).
  *
  * CSR layout: csr(0) = operandCount (#beats in the row), csr(1) = op: bits[7:0] = 0=MAX,1=ADD,2=SUMSQ;
  * bit[8] = tap. When tap=1 the row is passed through unchanged (1:1) AND the reduction is accumulated in
  * parallel; after the last input beat one extra trailing beat (the scalar, splatted) is emitted, so the
  * output is N+1 beats. tap=0 = legacy behaviour (consume N beats, emit only the scalar). The tap mode lets
  * one chained task produce both a transformed row and its reduction (e.g. softmax exp-row + Σexp).
  *
  * Configurable from the hjson: `op` (the LIST of supported reductions, a subset of {MAX, ADD, SUMSQ} —
  * only those ops' arithmetic is built; the op CSR field selects among them at runtime when >1 is listed),
  * `computeLanes` (time-mux width; pass the full lane count for the combinational path, or a smaller value
  * for the FSM that sweeps `computeLanes` ALUs across the lane partials over `subCycles` cycles per beat),
  * and `elementWidth` (the transport element width in bits — must match the op-set precision: FP16/BF16⇒16,
  * FP8⇒8, FP32⇒32). All are REQUIRED (no defaults) — the config must spell them out, e.g. {elementWidth:16,
  * computeLanes:32, op:["MAX_FP16","ADD_FP16","SUMSQ_FP16"]}. The tap bit stays runtime, so userCsrNum=2
  * is unchanged.
  *
  * HARDWARE UNIT COUNT — for a 512-bit input there are `computeLanes` physical ALUs (widen + add/max/square),
  * NOT one per lane, time-multiplexed over `subCycles` (= lanes/computeLanes) cycles per beat; but there are
  * ALWAYS `lanes` (= dataWidth/elementWidth, e.g. 32 at FP16 / 64 at FP8) FP32 partial-accumulator registers
  * regs[0..lanes-1] (one running partial per lane — cheap flops) plus one horizontal reduceTree at the end.
  * So computeLanes trades ALU count for cycles; the per-lane partials are inherent (they hold each lane's
  * running reduction across beats). computeLanes = lanes ⇒ PATH A: one ALU per lane in parallel, 1 cycle/beat.
  *
  * NUMERICAL EXAMPLE — sum (op=ADD), FP16, 32 lanes, computeLanes=8 ⇒ subCycles=4, operandCount=2 beats.
  * 32 FP32 partial accumulators regs[0..31]; 8 ALUs A0..A7 REUSED across the 4 sub-cycles AND across beats.
  * On sub-cycle s, ALU Aj folds logical lane (s·8+j) into regs[s·8+j]; the partials persist between beats:
  *
  *   CC  phase        A0..A7 read lanes → accumulate into regs    notes
  *   --  -----------  -----------------------------------------   --------------------
  *    0  accept b0    latch beat0                                 beatCnt=0 (first)
  *    1  b0 sub=0     lanes  0..7  → regs[ 0..7]  = beat0[ 0..7]  first: init (prev=0)
  *    2  b0 sub=1     lanes  8..15 → regs[ 8..15] = beat0[ 8..15]
  *    3  b0 sub=2     lanes 16..23 → regs[16..23] = beat0[16..23]
  *    4  b0 sub=3     lanes 24..31 → regs[24..31] = beat0[24..31] beat0 done, beatCnt=1
  *    5  accept b1    latch beat1
  *    6  b1 sub=0     lanes  0..7  → regs[ 0..7] += beat1[ 0..7]  accumulate (prev+new)
  *    7  b1 sub=1     lanes  8..15 → regs[ 8..15]+= beat1[ 8..15]
  *    8  b1 sub=2     lanes 16..23 → regs[16..23]+= beat1[16..23]
  *    9  b1 sub=3     lanes 24..31 → regs[24..31]+= beat1[24..31] last beat → outValid
  *   10  collapse     reduceTree(regs[0..31]) = Σ → narrow → splat: ext_data_o = [Σ, …, Σ]
  * e.g. beat0=[1,2,…], beat1=[3,4,…] ⇒ regs=[4,6,…] ⇒ Σ over all 32 ⇒ every out lane = Σ. Lane reuse:
  * ALU Aj owns partials {j, 8+j, 16+j, 24+j} on every beat. With tap=1 each input beat is also re-emitted
  * (1:1) before the trailing Σ beat. computeLanes=32 ⇒ PATH A: 1 cycle/beat, all 32 lanes in parallel.
  */
class HasStreamReduce(
  computeLanes: Int,
  op:           Seq[String], // each entry "<OP>_<PRECISION>", e.g. "ADD_FP16"
  elementWidth: Int,         // transport element width (bits); must match the op-set precision
  dataWidth:    Int = 512
) extends HasDataPathExtension {
  private val (_, transport) = OpSpec.parse(op, Set("MAX", "ADD", "SUMSQ"), "HasStreamReduce") // validate op names + precision
  OpSpec.checkWidth(elementWidth, transport, "HasStreamReduce") // explicit width must match the precision tag
  require(computeLanes > 0, "HasStreamReduce: computeLanes must be > 0")
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "StreamReduce",
      userCsrNum = 2, // operandCount + (op-select|tap); tap stays runtime so this is fixed
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): StreamReduce =
    Module(new StreamReduce(computeLanes, op, elementWidth) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamReduce(
  computeLanesParam: Int = 0,
  op:                Seq[String] = Seq("MAX_FP16", "ADD_FP16", "SUMSQ_FP16"),
  elementWidth:      Int = 16
)(
  implicit extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers.{widen => widenT, narrow => narrowT, square => squareT}

  // transport (element) precision comes from the op-set; internal compute stays FP32
  val (ops, transport) = OpSpec.parse(op, Set("MAX", "ADD", "SUMSQ"), "StreamReduce")
  OpSpec.checkWidth(elementWidth, transport, "StreamReduce") // config width must match the op-set precision
  val accWidth = 32 // FP32 internal
  val lanes    = extensionParam.dataWidth / elementWidth
  val computeLanes = if (computeLanesParam <= 0 || computeLanesParam > lanes) lanes else computeLanesParam
  require(lanes % computeLanes == 0, "StreamReduce: lanes must be a multiple of computeLanes")
  val subCycles    = lanes / computeLanes

  // which reductions are built
  val hasMax   = ops.contains("MAX")
  val hasAdd   = ops.contains("ADD")
  val hasSumsq = ops.contains("SUMSQ")
  val multiOp  = ops.size > 1

  def OP_MAX   = 0.U
  def OP_ADD   = 1.U
  def OP_SUMSQ = 2.U

  val csrOperandCount = ext_csr_i(0)
  val operandCount    = Mux(csrOperandCount === 0.U, 1.U(16.W), csrOperandCount(15, 0))
  val opField = ext_csr_i(1)
  val opcode  = opField(7, 0)     // 0=MAX,1=ADD,2=SUMSQ (used only when >1 op is built)
  val tap     = opField(8).asBool // pass the row through + emit the scalar as a trailing beat

  val FP32_ZERO = 0.U(accWidth.W)

  // FP32 max (finite inputs; NaN not expected from the host softmax path)
  def fp32max(a: UInt, b: UInt): UInt = {
    val sa = a(accWidth - 1); val sb = b(accWidth - 1)
    Mux(Mux(sa =/= sb, !sa, Mux(sa, b >= a, a >= b)), a, b)
  }
  def fp32add(a: UInt, b: UInt): UInt = {
    val m = Module(new FpAddFp(FP32, FP32, FP32)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }
  def widen(h:  UInt): UInt = widenT(h, transport)  // transport -> FP32
  def square(h: UInt): UInt = squareT(h, transport) // transport*transport -> FP32
  def narrow(f: UInt): UInt = narrowT(f, transport) // FP32 -> transport

  // one lane's new partial: accumulate `laneIn` into `prev` per the active op (only built ops instantiated)
  def accLane(laneIn: UInt, prev: UInt, first: Bool): UInt = {
    val widened = widen(laneIn)
    val addend =
      if (hasSumsq && hasAdd) Mux(opcode === OP_SUMSQ, square(laneIn), widened)
      else if (hasSumsq) square(laneIn)
      else widened
    val accAdd = if (hasAdd || hasSumsq) fp32add(addend, Mux(first, FP32_ZERO, prev)) else FP32_ZERO
    val accMax = if (hasMax) Mux(first, widened, fp32max(widened, prev)) else FP32_ZERO
    if (multiOp) Mux(opcode === OP_MAX, accMax, accAdd)
    else if (hasMax) accMax
    else accAdd
  }

  // ---- per-lane FP32 partials (shared by both paths) ----
  val regs = RegInit(VecInit(Seq.fill(lanes)(0.U(accWidth.W))))

  // ---- horizontal collapse of the lane partials -> 1 FP32 -> transport, splatted ----
  val addTree    = if (hasAdd || hasSumsq) regs.reduceTree((a, b) => fp32add(a, b)) else FP32_ZERO
  val maxTree    = if (hasMax) regs.reduceTree((a, b) => fp32max(a, b)) else FP32_ZERO
  val scalarFP32 =
    if (multiOp) Mux(opcode === OP_MAX, maxTree, addTree) else if (hasMax) maxTree else addTree
  val scalarBeat = Cat(Seq.fill(lanes)(narrow(scalarFP32)))

  if (computeLanes == lanes) {
    // PATH A — fully parallel, combinational accumulate (today's behaviour, unchanged)
    val counter = Module(new snax.utils.BasicCounter(16) {
      override val desiredName = "StreamReduceCounter"
    })
    counter.io.ceil  := operandCount
    counter.io.reset := ext_start_i
    counter.io.tick  := ext_data_i.fire
    val isFirst = counter.io.value === 0.U
    val isLast  = counter.io.value === (operandCount - 1.U)
    val outputValid = RegInit(false.B)

    val input_lanes = ext_data_i.bits.asTypeOf(Vec(lanes, UInt(elementWidth.W)))
    val nextRegs = VecInit((0 until lanes).map(i => accLane(input_lanes(i), regs(i), isFirst)))

    when(ext_start_i) {
      regs := VecInit(Seq.fill(lanes)(0.U(accWidth.W))); outputValid := false.B
    }.elsewhen(ext_data_i.fire) {
      regs := nextRegs; outputValid := isLast
    }.elsewhen(ext_data_o.fire) {
      outputValid := false.B
    }

    when(outputValid) {
      ext_data_o.bits  := scalarBeat
      ext_data_o.valid := true.B
      ext_data_i.ready := Mux(tap, false.B, ext_data_o.ready)
    }.otherwise {
      ext_data_o.bits  := Mux(tap, ext_data_i.bits, scalarBeat)
      ext_data_o.valid := Mux(tap, ext_data_i.valid, false.B)
      ext_data_i.ready := Mux(tap, ext_data_o.ready, true.B)
    }
    ext_busy_o := (counter.io.value =/= 0.U) || outputValid
  } else {
    // PATH B — time-multiplexed: computeLanes ALUs sweep the lane partials over subCycles cycles per beat
    val inBeat   = Reg(UInt((lanes * elementWidth).W)) // latched raw input beat
    val inLanes  = inBeat.asTypeOf(Vec(lanes, UInt(elementWidth.W)))
    val sub      = RegInit(0.U(log2Ceil(subCycles).W))
    val busy     = RegInit(false.B)
    val emitPass = RegInit(false.B) // tap: a passthrough beat is ready to emit
    val outValid = RegInit(false.B) // trailing scalar ready
    val beatCnt  = RegInit(0.U(16.W))
    val firstBeat = beatCnt === 0.U
    val lastBeat  = beatCnt === (operandCount - 1.U)

    // index of lane (sub*computeLanes + j) into the `lanes`-wide Vec, width-exact to silence W004
    def li(j: Int): UInt = (sub * computeLanes.U + j.U)(log2Ceil(lanes) - 1, 0)

    val res = Wire(Vec(computeLanes, UInt(accWidth.W)))
    for (j <- 0 until computeLanes)
      res(j) := accLane(inLanes(li(j)), regs(li(j)), firstBeat)

    ext_data_i.ready := !busy && !emitPass && !outValid

    when(ext_start_i) {
      busy := false.B; emitPass := false.B; outValid := false.B; sub := 0.U; beatCnt := 0.U
    }.elsewhen(busy) {
      for (j <- 0 until computeLanes) regs(li(j)) := res(j)
      when(sub === (subCycles - 1).U) {
        busy := false.B; sub := 0.U
        when(tap) { emitPass := true.B }
        when(lastBeat) { outValid := true.B }.otherwise { beatCnt := beatCnt + 1.U }
      }.otherwise {
        sub := sub + 1.U
      }
    }.elsewhen(ext_data_i.fire) {
      inBeat := ext_data_i.bits; busy := true.B; sub := 0.U
    }
    when(emitPass && ext_data_o.ready && !ext_start_i) { emitPass := false.B }
    when(outValid && !emitPass && ext_data_o.ready && !ext_start_i) { outValid := false.B; beatCnt := 0.U }

    when(emitPass) {
      ext_data_o.bits  := inBeat // tap passthrough of the latched beat
      ext_data_o.valid := true.B
    }.otherwise {
      ext_data_o.bits  := scalarBeat
      ext_data_o.valid := outValid
    }
    ext_busy_o := busy || emitPass || outValid || (beatCnt =/= 0.U)
  }
}
