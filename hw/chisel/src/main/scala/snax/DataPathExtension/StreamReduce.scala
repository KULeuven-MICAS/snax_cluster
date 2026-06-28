package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._
import fp_native._

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
  * bit[8] = tap; bit[9] = fp32out (emit the scalar in FP32 instead of narrowing to the transport grid — for
  * a reduction that overflows FP16, e.g. an unscaled-GEMM SUMSQ ~1e9, this delivers the true value to the
  * host instead of inf; the scalar is splatted as FP32 so the host reads the beat's low 32 bits as a float).
  * When tap=1 the row is passed through unchanged (1:1) AND the reduction is accumulated in
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
  * computeLanes:32, op:["MAX_FP16","ADD_FP16","SUMSQ_FP16"]}. The tap and fp32out bits stay runtime, so
  * userCsrNum=2 is unchanged.
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
  dataWidth:    Int = 512,
  fpPipe:       Int = 1,     // accumulate FP-unit internal pipeline depth, 0..2 (per-op timing cut knob)
  treePipe:     Int = 1      // reduce-tree add internal pipeline depth, 0..2 (SEPARATE cut knob)
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
    Module(new StreamReduce(computeLanes, op, elementWidth, fpPipe, treePipe) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamReduce(
  computeLanesParam: Int = 0,
  op:                Seq[String] = Seq("MAX_FP16", "ADD_FP16", "SUMSQ_FP16"),
  elementWidth:      Int = 16,
  fpPipeParam:       Int = 1,
  treePipeParam:     Int = 1,
  pipelined:         Boolean = true
)(
  implicit extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers.{narrow => narrowT}

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
  val fp32out = opField(9).asBool // emit the scalar in FP32 (no narrow) so a large reduction (e.g. an
  //                                 unscaled-GEMM SUMSQ ~1e9) reaches the host as the true value instead
  //                                 of overflowing the transport FP16 range to inf/garbage. Splatted as FP32.

  val FP32_ZERO = 0.U(accWidth.W)

  // ---- pipeline depths (timing) ----------------------------------------------------------------
  // A single FP32 add/mul overruns the clock as one combinational op, so the reduce hot path uses the
  // INTERNALLY-pipelined native FP units (fp_native.FpAdd/FpMul, numPipe register stages each) rather than
  // relying on synthesis register-retiming. Two SEPARATE cut knobs (cfg): `fpPipe` for the per-lane
  // accumulate, `treePipe` for the horizontal reduce-tree adds. Per-lane accumulate latency =
  //   laneLat (register the time-mux lane select) + fpPipe (widen/square) + fpPipe (accumulate add).
  // Horizontal reduce tree: one treePipe-pipelined add per level (add) / one registered compare per
  // level (max), aligned to a common treeLat.
  val fpPipe     = if (pipelined) fpPipeParam else 0   // accumulate FP-unit pipeline depth (cfg)
  val treePipe   = if (pipelined) treePipeParam else 0 // reduce-tree add pipeline depth (cfg, separate knob)
  val laneLat    = if (pipelined) 1 else 0      // register after the time-mux lane-select mux
  val accLat     = laneLat + 2 * fpPipe         // issue -> per-lane accumulate result
  val numLevels  = log2Ceil(lanes)
  val treeAddLat = numLevels * treePipe         // add tree: treePipe registers per level
  val treeMaxLat = if (pipelined) numLevels else 0 // max tree: 1 register per level
  val treeLat    = scala.math.max(treeAddLat, treeMaxLat)
  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  // FP32 max (finite inputs; NaN not expected from the host softmax path)
  def fp32max(a: UInt, b: UInt): UInt = {
    val sa = a(accWidth - 1); val sb = b(accWidth - 1)
    Mux(Mux(sa =/= sb, !sa, Mux(sa, b >= a, a >= b)), a, b)
  }
  def narrow(f: UInt): UInt = narrowT(f, transport) // FP32 -> transport

  // Internally-pipelined native FP units. The per-lane accumulate (widenP/squareP/addP) is cut by
  // `fpPipe`; the horizontal reduce-tree add (treeAdd) by the separate `treePipe`. Each call instantiates
  // one unit at the call site (numPipe=0 => combinational).
  def widenP(h:  UInt): UInt = {
    val m = Module(new FpAdd(transport, transport, FP32, fpPipe)); m.io.in_a := h; m.io.in_b := 0.U(transport.width.W); m.io.out
  }
  def squareP(h: UInt): UInt = {
    val m = Module(new FpMul(transport, transport, FP32, fpPipe)); m.io.in_a := h; m.io.in_b := h; m.io.out
  }
  def addP(a: UInt, b: UInt): UInt = {
    val m = Module(new FpAdd(FP32, FP32, FP32, fpPipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }
  def treeAdd(a: UInt, b: UInt): UInt = {
    val m = Module(new FpAdd(FP32, FP32, FP32, treePipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }

  // one lane's new partial: accumulate `laneIn` into `prev` per the active op (only built ops built).
  // Pipeline: register the time-mux lane select (laneLat), widen/square through an internally-pipelined
  // FP unit (fpPipe), then fold into `prev` through an internally-pipelined add (fpPipe); max stays a
  // cheap compare, latency-matched. Total issue->result latency = accLat = laneLat + 2*fpPipe.
  def accLane(laneIn: UInt, prev: UInt, first: Bool): UInt = {
    val laneR   = sr(laneIn)                     // +laneLat : break the lane-select mux off the FP op
    val widened = widenP(laneR)                  // +fpPipe
    val sq      = if (hasSumsq) squareP(laneR) else widened
    // align the (stable) accumulator and the first-beat flag with the widened/squared inputs
    val prevA  = ShiftRegister(prev,  laneLat + fpPipe)
    val firstA = ShiftRegister(first, laneLat + fpPipe)
    val addend =
      if (hasSumsq && hasAdd) Mux(opcode === OP_SUMSQ, sq, widened)
      else if (hasSumsq) sq
      else widened
    val accAdd = if (hasAdd || hasSumsq) addP(addend, Mux(firstA, FP32_ZERO, prevA)) else FP32_ZERO
    // max is a cheap combinational compare; register it by fpPipe to line up with the pipelined add
    val accMax = if (hasMax) ShiftRegister(Mux(firstA, widened, fp32max(widened, prevA)), fpPipe) else FP32_ZERO
    if (multiOp) Mux(opcode === OP_MAX, accMax, accAdd)
    else if (hasMax) accMax
    else accAdd
  }

  // ---- per-lane FP32 partials ----
  val regs = RegInit(VecInit(Seq.fill(lanes)(0.U(accWidth.W))))

  // ---- horizontal collapse of the lane partials -> 1 FP32 -> transport, splatted ----
  // Fully-registered pairwise reduce: the add tree uses one internally-pipelined add per level (so a
  // single FP add never crosses a clock period — the tree's two-serial-add segments were the worst
  // path), the max tree one registered compare per level; the two are aligned to the common treeLat.
  // Read only once `regs` are stable (the FSM tree-drain waits treeLat cycles after the last accumulate).
  def treeReduce(xs: Seq[UInt], f: (UInt, UInt) => UInt, carry: UInt => UInt): UInt = {
    var cur = xs.toIndexedSeq
    while (cur.length > 1) {
      cur = (0 until (cur.length + 1) / 2).map { k =>
        if (2 * k + 1 < cur.length) f(cur(2 * k), cur(2 * k + 1)) else carry(cur(2 * k))
      }.toIndexedSeq
    }
    cur.head
  }
  val addTreeRaw = if (hasAdd || hasSumsq) treeReduce(regs, (a, b) => treeAdd(a, b), x => ShiftRegister(x, treePipe)) else FP32_ZERO
  val maxTreeRaw = if (hasMax) treeReduce(regs, (a, b) => sr(fp32max(a, b)), x => sr(x)) else FP32_ZERO
  val addTree    = ShiftRegister(addTreeRaw, treeLat - treeAddLat) // pad to the common tree latency
  val maxTree    = ShiftRegister(maxTreeRaw, treeLat - treeMaxLat)
  val scalarFP32 =
    if (multiOp) Mux(opcode === OP_MAX, maxTree, addTree) else if (hasMax) maxTree else addTree
  // Output packing: narrow the FP32 scalar to the transport grid (default), or — when fp32out is set and
  // the transport is narrower than FP32 — splat the raw FP32 scalar across the 512-bit beat (dataWidth/32
  // copies). The beat stays 512-bit either way, so the writer AGU is unchanged; the host just reads the
  // low 32 bits as a float instead of the low `elementWidth` bits. (For an FP32 transport there is nothing
  // to narrow, so the mux folds away at elaboration.)
  val narrowedBeat = Cat(Seq.fill(lanes)(narrow(scalarFP32)))
  val scalarBeat =
    if (transport.width >= accWidth) narrowedBeat
    else Mux(fp32out, Cat(Seq.fill(extensionParam.dataWidth / accWidth)(scalarFP32)), narrowedBeat)

  // ---- unified time-mux + pipeline FSM ----
  // `computeLanes` pipelined ALUs sweep the beat over `subCycles` sub-groups (one issued/cycle); results
  // retire `accLat` cycles later into the per-lane partials. After the last beat of a row, a tree-drain
  // phase waits `treeLat` cycles for the pipelined reduceTree to settle, then the scalar is emitted.
  // computeLanes == lanes ⇒ subCycles == 1. tap ⇒ each input beat is re-emitted before the trailing scalar.
  val inBeat   = Reg(UInt((lanes * elementWidth).W)) // latched raw input beat
  val inLanes  = inBeat.asTypeOf(Vec(lanes, UInt(elementWidth.W)))
  val busy     = RegInit(false.B)
  val emitPass = RegInit(false.B) // tap: a passthrough beat is ready to emit
  val outValid = RegInit(false.B) // trailing scalar ready
  val treeBusy = RegInit(false.B) // draining the pipelined reduceTree after the last beat
  val beatCnt  = RegInit(0.U(16.W))
  val firstBeat = beatCnt === 0.U
  val lastBeat  = beatCnt === (operandCount - 1.U)
  val total     = subCycles - 1 + accLat
  val step      = RegInit(0.U(log2Ceil(total + 1).max(1).W))
  val treeStepW = if (treeLat > 0) log2Ceil(treeLat + 1) else 1
  val treeStep  = RegInit(0.U(treeStepW.W))

  // index of lane (s*computeLanes + j) into the `lanes`-wide Vec, width-exact to silence W004
  def li(s: UInt, j: Int): UInt = (s * computeLanes.U + j.U)(log2Ceil(lanes) - 1, 0)

  val issuing  = busy && (step < subCycles.U)
  val subIssue = step // low log2(lanes) bits matter; gated off during drain
  val res = Wire(Vec(computeLanes, UInt(accWidth.W)))
  for (j <- 0 until computeLanes)
    res(j) := accLane(inLanes(li(subIssue, j)), regs(li(subIssue, j)), firstBeat)

  val subRetire   = ShiftRegister(subIssue, accLat)
  val retireValid = ShiftRegister(issuing, accLat, false.B, true.B)
  when(busy && retireValid) {
    for (j <- 0 until computeLanes) regs(li(subRetire, j)) := res(j)
  }

  ext_data_i.ready := !busy && !treeBusy && !emitPass && !outValid

  when(ext_start_i) {
    busy := false.B; emitPass := false.B; outValid := false.B; treeBusy := false.B
    step := 0.U; treeStep := 0.U; beatCnt := 0.U
  }.elsewhen(busy) {
    when(step === total.U) {
      busy := false.B; step := 0.U
      when(tap) { emitPass := true.B }
      when(lastBeat) {
        if (treeLat > 0) { treeBusy := true.B; treeStep := 0.U } else { outValid := true.B }
      }.otherwise { beatCnt := beatCnt + 1.U }
    }.otherwise {
      step := step + 1.U
    }
  }.elsewhen(treeBusy) {
    when(treeStep === (treeLat - 1).U) { treeBusy := false.B; outValid := true.B }
      .otherwise { treeStep := treeStep + 1.U }
  }.elsewhen(ext_data_i.fire) {
    inBeat := ext_data_i.bits; busy := true.B; step := 0.U
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
  ext_busy_o := busy || treeBusy || emitPass || outValid || (beatCnt =/= 0.U)
}
