package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._
import fp_native._

/** StreamReduce: vertical (across-beat) + horizontal (across-lane) FP reduction of a streamed row to a single scalar,
  * splatted across all lanes of one output beat.
  *
  *   - Transport: configurable element type (FP16/BF16/FP8/FP32, set by the op-set; e.g. 32 FP16 lanes per 512-bit
  *     beat). Internal math: FP32 (widen on input, narrow on output).
  *   - op = MAX | ADD | SUMSQ (SUMSQ squares each lane before summing). Selected at runtime via CSR.
  *
  * Reuses the ElementwiseAdd/MaxPool accumulate-then-emit FSM. The horizontal cross-lane collapse is a balanced
  * reduceTree (tree order; small numeric tolerance accepted per design decision #4).
  *
  * CSR layout: csr(0) = operandCount (#beats in the row), csr(1) = op: bits[7:0] = 0=MAX,1=ADD,2=SUMSQ; bit[8] = tap;
  * bit[9] = fp32out (emit the scalar in FP32 instead of narrowing to the transport grid — for a reduction that
  * overflows FP16, e.g. an unscaled-GEMM SUMSQ ~1e9, this delivers the true value to the host instead of inf; the
  * scalar is splatted as FP32 so the host reads the beat's low 32 bits as a float). When tap=1 the row is passed
  * through unchanged (1:1) AND the reduction is accumulated in parallel; after the last input beat one extra trailing
  * beat (the scalar, splatted) is emitted, so the output is N+1 beats. tap=0 = legacy behaviour (consume N beats, emit
  * only the scalar). The tap mode lets one chained task produce both a transformed row and its reduction (e.g. softmax
  * exp-row + Σexp).
  *
  * Configurable from the hjson: `op` (the LIST of supported reductions, a subset of {MAX, ADD, SUMSQ} — only those ops'
  * arithmetic is built; the op CSR field selects among them at runtime when >1 is listed), `computeLanes` (time-mux
  * width; pass the full lane count for the combinational path, or a smaller value for the FSM that sweeps
  * `computeLanes` ALUs across the lane partials over `subCycles` cycles per beat), and `elementWidth` (the transport
  * element width in bits — must match the op-set precision: FP16/BF16⇒16, FP8⇒8, FP32⇒32). All are REQUIRED (no
  * defaults) — the config must spell them out, e.g. {elementWidth:16, computeLanes:32,
  * op:["MAX_FP16","ADD_FP16","SUMSQ_FP16"]}. The tap and fp32out bits stay runtime, so userCsrNum=2 is unchanged.
  *
  * HARDWARE UNIT COUNT — for a 512-bit input there are `computeLanes` physical ALUs (widen + add/max/square), NOT one
  * per lane, time-multiplexed over `subCycles` (= lanes/computeLanes) cycles per beat; but there are ALWAYS `lanes` (=
  * dataWidth/elementWidth, e.g. 32 at FP16 / 64 at FP8) FP32 partial-accumulator registers regs[0..lanes-1] (one
  * running partial per lane — cheap flops) plus one horizontal reduceTree at the end. So computeLanes trades ALU count
  * for cycles; the per-lane partials are inherent (they hold each lane's running reduction across beats). computeLanes
  * \= lanes ⇒ PATH A: one ALU per lane in parallel, 1 cycle/beat.
  *
  * NUMERICAL EXAMPLE — sum (op=ADD), FP16, 32 lanes, computeLanes=8 ⇒ subCycles=4, operandCount=2 beats. 32 FP32
  * partial accumulators regs[0..31]; 8 ALUs A0..A7 REUSED across the 4 sub-cycles AND across beats. On sub-cycle s, ALU
  * Aj folds logical lane (s·8+j) into regs[s·8+j]; the partials persist between beats:
  *
  * CC phase A0..A7 read lanes → accumulate into regs notes -- ----------- -----------------------------------------
  * -------------------- 0 accept b0 latch beat0 beatCnt=0 (first) 1 b0 sub=0 lanes 0..7 → regs[ 0..7] = beat0[ 0..7]
  * first: init (prev=0) 2 b0 sub=1 lanes 8..15 → regs[ 8..15] = beat0[ 8..15] 3 b0 sub=2 lanes 16..23 → regs[16..23] =
  * beat0[16..23] 4 b0 sub=3 lanes 24..31 → regs[24..31] = beat0[24..31] beat0 done, beatCnt=1 5 accept b1 latch beat1 6
  * b1 sub=0 lanes 0..7 → regs[ 0..7] += beat1[ 0..7] accumulate (prev+new) 7 b1 sub=1 lanes 8..15 → regs[ 8..15]+=
  * beat1[ 8..15] 8 b1 sub=2 lanes 16..23 → regs[16..23]+= beat1[16..23] 9 b1 sub=3 lanes 24..31 → regs[24..31]+=
  * beat1[24..31] last beat → outValid 10 collapse reduceTree(regs[0..31]) = Σ → narrow → splat: ext_data_o = [Σ, …, Σ]
  * e.g. beat0=[1,2,…], beat1=[3,4,…] ⇒ regs=[4,6,…] ⇒ Σ over all 32 ⇒ every out lane = Σ. Lane reuse: ALU Aj owns
  * partials {j, 8+j, 16+j, 24+j} on every beat. With tap=1 each input beat is also re-emitted (1:1) before the trailing
  * Σ beat. computeLanes=32 ⇒ PATH A: 1 cycle/beat, all 32 lanes in parallel.
  */
class HasStreamReduce(
  computeLanes: Int,
  op:           Seq[String], // each entry "<OP>_<PRECISION>", e.g. "ADD_FP16"
  elementWidth: Int, // transport element width (bits); must match the op-set precision
  dataWidth:    Int = 512,
  fpPipe:       Int = 1, // accumulate FP-unit internal pipeline depth, 0..2 (per-op timing cut knob)
  treePipe:     Int = 1, // reduce-fold add internal pipeline depth, 0..2 (SEPARATE cut knob)
  treeLanes:    Int = 2, // # of horizontal-fold ALUs (area knob): the once-per-row collapse is time-muxed
  accBanks:     Int = 0, // per-lane partial banks (0 = auto = nextPow2(accLat+1) to break the accumulate
  //                        recurrence and reach 1 beat/cycle at computeLanes=lanes; 1 = legacy single partial)
  foldParallel: Int = 0  // 1 = fully-pipelined parallel reduce tree (rows pipeline, multi-row hits the
  //                        roofline; costs lanes-1 FP adders); 0 = time-muxed area-saving fold (serialized)
) extends HasDataPathExtension {
  private val (_, transport) =
    OpSpec.parse(op, Set("MAX", "ADD", "SUMSQ", "FMA"), "HasStreamReduce") // validate op names + precision
  OpSpec.checkWidth(elementWidth, transport, "HasStreamReduce") // explicit width must match the precision tag
  require(computeLanes > 0, "HasStreamReduce: computeLanes must be > 0")
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "StreamReduce",
      userCsrNum = 2, // operandCount + (op-select|tap); tap stays runtime so this is fixed
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): StreamReduce =
    Module(new StreamReduce(computeLanes, op, elementWidth, fpPipe, treePipe, treeLanes, accBanks,
                            foldParallel != 0) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamReduce(
  computeLanesParam: Int         = 0,
  op:                Seq[String] = Seq("MAX_FP16", "ADD_FP16", "SUMSQ_FP16"),
  elementWidth:      Int         = 16,
  fpPipeParam:       Int         = 1,
  treePipeParam:     Int         = 1,
  treeLanesParam:    Int         = 2,
  accBanksParam:     Int         = 0,
  foldParallel:      Boolean     = false,
  pipelined:         Boolean     = true
)(implicit
  extensionParam:    DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers.{narrow => narrowT}

  // transport (element) precision comes from the op-set; internal compute stays FP32
  val (ops, transport) = OpSpec.parse(op, Set("MAX", "ADD", "SUMSQ", "FMA"), "StreamReduce")
  OpSpec.checkWidth(elementWidth, transport, "StreamReduce") // config width must match the op-set precision
  val accWidth     = 32 // FP32 internal
  val lanes        = extensionParam.dataWidth / elementWidth
  val computeLanes = if (computeLanesParam <= 0 || computeLanesParam > lanes) lanes else computeLanesParam
  require(lanes % computeLanes == 0, "StreamReduce: lanes must be a multiple of computeLanes")
  val subCycles = lanes / computeLanes
  // horizontal-fold width: how many pairwise reductions per fold cycle (area vs fold-latency). Clamp to
  // lanes/2 (max useful) and >=1.
  val treeLanes = scala.math.max(1, scala.math.min(treeLanesParam, lanes / 2))

  // which reductions are built
  // Op-set is now the single fused "FMA" op (covers the add- and square-accumulate, selected at runtime by
  // the op CSR) plus the non-FMA "MAX" compare. hasAdd/hasSumsq are the runtime CAPABILITIES the built
  // datapath must provide: the FMA op provides both, so every accLane/fold expression below stays as-is.
  // Legacy op-sets ("ADD"/"SUMSQ"/"MAX") still parse and build the same minimal subset.
  val hasFMA   = ops.contains("FMA")
  val hasMax   = ops.contains("MAX")
  val hasAdd   = hasFMA || ops.contains("ADD")   // FMA: acc' = x*1 + acc
  val hasSumsq = hasFMA || ops.contains("SUMSQ") // FMA: acc' = x*x + acc  (op CSR bit picks add vs square)
  val multiOp  = ops.size > 1

  // op CSR (ext_csr_i(1)[7:0]) values — UNCHANGED so the SW interface is source-compatible: 0=MAX (compare),
  // 1=ADD (FMA, multiplicand 1.0), 2=SUMSQ (FMA, multiplicand = the input).
  def OP_MAX   = 0.U
  def OP_ADD   = 1.U
  def OP_SUMSQ = 2.U

  val csrOperandCount = ext_csr_i(0)
  val operandCount    = Mux(csrOperandCount === 0.U, 1.U(16.W), csrOperandCount(15, 0))
  val opField         = ext_csr_i(1)
  val opcode          = opField(7, 0)     // 0=MAX,1=ADD,2=SUMSQ (used only when >1 op is built)
  val tap             = opField(8).asBool // pass the row through + emit the scalar as a trailing beat
  val fp32out         = opField(9).asBool // emit the scalar in FP32 (no narrow) so a large reduction (e.g. an
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
  val laneLat    = if (pipelined) 1 else 0             // register after the time-mux lane-select mux
  val accLat     = laneLat + 2 * fpPipe                // issue -> per-lane accumulate result
  // ---- accumulator BANKING (breaks the per-lane read-after-write recurrence) ------------------
  // Consecutive beats round-robin through `accBanks` independent per-lane partials, so the same (bank,lane)
  // partial is reused only every accBanks beats (= accBanks*subCycles cycles). When accBanks*subCycles >=
  // accLat+1 the retire always completes before the reuse READ, so the inter-beat `gap` is 0 and the issue
  // streams at 1 beat/subCycles (=> 1 beat/cycle at computeLanes=lanes). Default = smallest power of two >=
  // accLat+1 (a bank index is then a bit-slice of beatIdx, no modulo). accBanks==1 = the legacy single
  // partial with the recurrence gap. The banks are combined once per row at the fold snapshot (see collapse).
  val nBanks =
    if (accBanksParam <= 0) (1 << log2Ceil(accLat + 1)) // auto: smallest power of two >= accLat+1
    else if (accBanksParam == 1) 1                      // legacy single partial (recurrence gap kept)
    else (1 << log2Ceil(accBanksParam))                 // explicit request, rounded up to a power of two
  def bankOf(idx: UInt): UInt = if (nBanks == 1) 0.U(1.W) else idx(log2Ceil(nBanks) - 1, 0)
  // The ADD/SUMSQ accumulate fuses the (square)+(add) into ONE mixed-format FMA (see accLane). Give it the
  // SAME latency as the old widen/square + add (2*fpPipe) so accLat — and the whole streaming FSM built on
  // it — is unchanged; FpFma caps numPipe at 2, so for 2*fpPipe>2 the remainder is a plain output shift-reg.
  val fmaPipe    = scala.math.min(2, 2 * fpPipe)
  // +1.0 in the transport format (sign 0, exp = bias, mantissa 0): FMA(x, 1.0, prev) == x + prev for ADD.
  val ONE_T      = ((((1 << (transport.expWidth - 1)) - 1) << transport.sigWidth)).U(transport.width.W)
  val numLevels  = log2Ceil(lanes)
  val treeAddLat = numLevels * treePipe                // add tree: treePipe registers per level
  val treeMaxLat = if (pipelined) numLevels else 0     // max tree: 1 register per level
  val treeLat    = scala.math.max(treeAddLat, treeMaxLat)
  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  // FP32 max (finite inputs; NaN not expected from the host softmax path)
  def fp32max(a: UInt, b: UInt): UInt = {
    val sa = a(accWidth - 1); val sb = b(accWidth - 1)
    Mux(Mux(sa =/= sb, !sa, Mux(sa, b >= a, a >= b)), a, b)
  }
  def narrow(f: UInt): UInt = narrowT(f, transport) // FP32 -> transport

  // Internally-pipelined native FP units. The per-lane accumulate (widenP + fused fmaP) is cut by
  // `fpPipe`; the horizontal reduce-tree add (treeAdd) by the separate `treePipe`. Each call instantiates
  // one unit at the call site (numPipe=0 => combinational).
  // exact transport->FP32 widen (bit-manipulation convert, not an FP adder — see FpHelpers.widen)
  def widenP(h: UInt): UInt = FpHelpers.widen(h, transport, fpPipe)
  // Mixed-format fused multiply-add x*y+c (transport x/y, FP32 c/out). Replaces the separate FpMul(square)
  // + FpAdd(accumulate): FP16*FP16 is exact in the FMA's 24-bit product, so the fused single rounding is
  // BIT-IDENTICAL to square-then-add (SUMSQ: FMA(x,x,prev); ADD: FMA(x,1.0,prev) = x+prev). One FP unit/lane.
  def fmaP(x: UInt, y: UInt, c: UInt): UInt = {
    val m = Module(new FpFma(transport, transport, FP32, fmaPipe)); m.io.in_a := x; m.io.in_b := y; m.io.in_c := c;
    m.io.out
  }
  def treeAdd(a: UInt, b: UInt):       UInt = {
    val m = Module(new FpAdd(FP32, FP32, FP32, treePipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }

  // one lane's new partial: accumulate `laneIn` into `prev` per the active op (only built ops built).
  // ADD/SUMSQ share ONE fused multiply-add: FMA(x, y, prev) with y = x for SUMSQ (x^2+prev) and y = 1.0 for
  // ADD (x+prev) — bit-identical to the old square-then-add (FP16^2 is exact in the FMA product), one FP unit
  // per lane instead of two (square + add). MAX keeps its own cheap widen+compare on a separate path.
  // Pipeline (total issue->result latency = accLat = laneLat + 2*fpPipe, UNCHANGED):
  //   ADD/SUMSQ: laneR (+laneLat) -> FMA (+fmaPipe) -> output shift (+2*fpPipe-fmaPipe).
  //   MAX:       laneR (+laneLat) -> widen (+fpPipe) -> compare -> shift (+fpPipe).
  def accLane(laneIn: UInt, prev: UInt, first: Bool): UInt = {
    val laneR   = sr(laneIn)                               // +laneLat : break the lane-select mux off the FP op
    // FMA multiplicand: square (SUMSQ) or 1.0 (plain ADD). The FMA consumes `prev` at its input (laneLat),
    // so align the accumulator/first-flag to laneLat here (the MAX path aligns to laneLat+fpPipe below).
    val fmaY    =
      if (hasSumsq && hasAdd) Mux(opcode === OP_SUMSQ, laneR, ONE_T)
      else if (hasSumsq) laneR
      else ONE_T
    val prevF   = ShiftRegister(prev, laneLat)
    val firstF  = ShiftRegister(first, laneLat)
    val accAdd  =
      if (hasAdd || hasSumsq)
        ShiftRegister(fmaP(laneR, fmaY, Mux(firstF, FP32_ZERO, prevF)), 2 * fpPipe - fmaPipe)
      else FP32_ZERO
    // MAX: widen (exact FP16->FP32 bit-convert, see widenP) then a cheap compare, registered to accLat
    val widened = if (hasMax) widenP(laneR) else FP32_ZERO // +fpPipe
    val prevA   = ShiftRegister(prev, laneLat + fpPipe)
    val firstA  = ShiftRegister(first, laneLat + fpPipe)
    val accMax  = if (hasMax) ShiftRegister(Mux(firstA, widened, fp32max(widened, prevA)), fpPipe) else FP32_ZERO
    if (multiOp) Mux(opcode === OP_MAX, accMax, accAdd)
    else if (hasMax) accMax
    else accAdd
  }

  // ---- per-lane FP32 partials (nBanks independent partials per lane, round-robin by beat) ----
  val regs = RegInit(VecInit(Seq.fill(lanes * nBanks)(0.U(accWidth.W))))
  // flat index of partial (bank, lane), width-exact
  def bankIdx(bank: UInt, laneSel: UInt): UInt = (bank * lanes.U + laneSel)(log2Ceil(lanes * nBanks) - 1, 0)

  // ---- streaming time-mux + pipeline FSM (accumulate core is identical to StreamElementwise) -----------
  // Continuous-issue: the computeLanes ALUs fold one sub-group per cycle into the per-lane partials across
  // back-to-back operand beats AND back-to-back rows (was: per-beat accLat drain + per-row tree-drain
  // stall). At a row's last-sub retire the COMPLETE partials are captured (outNow) and pushed into the
  // horizontal reduce tree; the scalar lands `treeLat` later and is narrowed/splatted into one output beat.
  // Feeding the tree from the CAPTURED partials (not `regs`) lets the next row start accumulating into regs
  // immediately, so rows pipeline through the tree. Steady-state = 1 beat/subCycles cycles.
  //
  // Accumulator recurrence (operand beat k+1 reads regs[L] from beat k): the time-mux spaces same-lane
  // issues by subCycles cycles, so no stall is needed when subCycles >= accLat+1; otherwise a compile-time
  // `gap` spaces accumulating beats WITHIN a row (row boundaries seed, no gap). A credit counter caps
  // in-flight+queued OUTPUT beats to the queue depth. tap mode re-emits each input beat (passthrough) before
  // the row's scalar; to keep that order on a single stream it serializes rows (next row waits for the
  // scalar). CONTRACT: ext_start_i only asserts when idle (ext_busy_o low).
  val gap      = scala.math.max(0, accLat + 1 - nBanks * subCycles) // 0 when banking spaces reuse >= accLat+1
  val beatSpan = subCycles + gap                                    // issue cycles allotted per operand beat

  // index of lane (s*computeLanes + j) into the `lanes`-wide Vec, width-exact to silence W004
  def li(s: UInt, j: Int): UInt = (s * computeLanes.U + j.U)(log2Ceil(lanes) - 1, 0)

  // ---- issue pointer + row bookkeeping ----
  val inBeat   = Reg(UInt((lanes * elementWidth).W))
  val inLanes  = inBeat.asTypeOf(Vec(lanes, UInt(elementWidth.W)))
  val haveBeat = RegInit(false.B)
  val sub      = RegInit(0.U(log2Ceil(subCycles).max(1).W))
  val beatIdx  = RegInit(0.U(16.W))                       // operand index of the beat being issued
  val nextIdx  = RegInit(0.U(16.W))                       // operand index of the NEXT beat to accept
  val stall    = RegInit(0.U(log2Ceil(gap + 1).max(1).W)) // remaining recurrence-gap cycles (gap>0 only)

  val lastSub        = sub === (subCycles - 1).U
  val firstInBank    = beatIdx < nBanks.U    // first nBanks beats each SEED their bank; later beats accumulate
  val issueBank      = bankOf(beatIdx)       // bank of the beat currently being issued
  val lastBeatInRow  = beatIdx === (operandCount - 1.U)
  val nextIsRowStart = nextIdx === 0.U
  val overlapOK      = if (gap == 0) true.B else nextIsRowStart

  // ---- output queue + credit (output BEATS: one scalar per row + one passthrough per beat in tap) ----
  // The fold serializes the output (treeBusy holds input off), so only ~1 scalar is ever in flight; the
  // credit reserves a slot per beat, so a shallow queue just backpressures earlier (never overflows). Floor
  // of 3 keeps a tap row-start's 2-slot reserve + slack; the writer drains passthroughs faster than the
  // cl-time-muxed producer makes them, so this shallow depth costs no app throughput.
  val inFlightMax = (accLat + treeLat + beatSpan - 1) / beatSpan + 1 // beats draining after issue stops
  // Cover the STEADY-STATE credit round-trip (accept -> issue -> accLat+treeLat retire/fold -> Queue enq->deq
  // -> credit++, plus downstream ready latency), so the credit counter doesn't starve the producer below the
  // roofline once the recurrence gap is removed (see StreamMap). Area is not the constraint at cl=lanes.
  val creditRoundTrip = accLat + treeLat + 4
  val Qdepth = scala.math.max(3,
    scala.math.max(inFlightMax + 1, (creditRoundTrip + beatSpan - 1) / beatSpan + 1))
  val outQ        = Module(new Queue(UInt((lanes * elementWidth).W), entries = Qdepth))
  val credit      = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))
  val tapDraining  = RegInit(false.B) // tap: a row's beats are all in, its scalar not yet emitted
  val treeBusy     = RegInit(false.B) // the horizontal fold is running (input held off so treeBuf is safe)
  // Banking removes the inter-beat gap, so a short next row could stream in and COMPLETE while the previous
  // row's fold is still busy (treeBuf is single-buffered) -> assert. Hold the NEXT row's start off from the
  // moment a row's last beat issues (foldInFlight) until that row's scalar is out. Serializes rows through the
  // fold (the fold-bound multi-row regime is unchanged); a single big row sets this only at the very end.
  val foldInFlight = RegInit(false.B)

  // accept a new input beat: overlap the current beat's last-sub issue when the successor needs no gap,
  // else from idle once the gap drained. Reserve a scalar slot at a row start + a passthrough slot in tap.
  // +& (width-expanding add): a plain + of two 1-bit Muxes overflows 1+1 -> 0, which would drop beat-0's
  // combined scalar+passthrough reservation in a tap row and leak 2 credits (busy stuck high).
  val reserveOnAccept   = Mux(nextIsRowStart, 1.U, 0.U) +& Mux(tap, 1.U, 0.U)
  val acceptDuringIssue = haveBeat  && lastSub        && overlapOK
  val acceptWhenIdle    = !haveBeat && (stall === 0.U)
  val creditOK          = credit    >= reserveOnAccept
  val tapBlock          = tap       && nextIsRowStart && tapDraining // hold the next row until the scalar is out
  // hold the next row until the current fold completes. The `haveBeat && lastSub && lastBeatInRow` term is the
  // COMBINATIONAL rowLastIssue: a next-row beat-0 can overlap-accept the SAME cycle a row's last beat issues,
  // before the registered foldInFlight engages, so gate on it too (closes the same-cycle hole).
  // parallel tree: non-tap rows pipeline through the fold (no serialization); only tap needs scalar-before-
  // next-row ordering. Time-muxed fold: always serialize (single treeBuf).
  val foldBlock         = nextIsRowStart && (foldInFlight || (haveBeat && lastSub && lastBeatInRow)) &&
                          (if (foldParallel) tap else true.B)
  ext_data_i.ready := (acceptDuringIssue || acceptWhenIdle) && creditOK && !tapBlock && !foldBlock && !treeBusy && !ext_start_i
  val accept = ext_data_i.fire

  // ---- issue ----
  val issuing = haveBeat
  val res     = Wire(Vec(computeLanes, UInt(accWidth.W)))
  for (j <- 0 until computeLanes)
    res(j) := accLane(inLanes(li(sub, j)), regs(bankIdx(issueBank, li(sub, j))), firstInBank)

  // A valid-pulse pipeline that is CLEARED by ext_start_i, so an in-flight scalar pulse from the previous
  // task (still propagating through accLat + the treeLat-deep reduce tree) can never fire — and push an
  // unreserved beat — after ext_start_i has reset the credit counter. (The controller re-starts the next
  // task while the reduce's tree is still draining, so relying on "start only when idle" is not enough.)
  def clrPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(ext_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(ext_start_i, false.B, r(i - 1))
      r(n - 1)
    }

  // ---- retire: fold partials into regs; capture the complete row at its last-sub retire ----
  val subRetire     = ShiftRegister(sub, accLat)
  val bankRetire    = ShiftRegister(issueBank, accLat) // bank of the beat retiring this cycle
  val retireValid   = ShiftRegister(issuing, accLat, false.B, true.B)
  val rowLastIssue  = issuing && lastSub && lastBeatInRow
  val rowLastRetire = clrPipe(rowLastIssue, accLat)
  val outNowBanked  = WireInit(regs)                   // all nBanks*lanes partials, retiring group overlaid
  when(retireValid) {
    for (j <- 0 until computeLanes) {
      regs(bankIdx(bankRetire, li(subRetire, j)))         := res(j)
      outNowBanked(bankIdx(bankRetire, li(subRetire, j))) := res(j)
    }
  }

  // ---- bank collapse: combine the nBanks per-lane partials into ONE partial per lane, once per row, into
  // the fold input `collapsed`. Banks that received no beat in this row (operandCount < nBanks) are masked
  // with the op identity (+0.0 for ADD/SUMSQ, -inf for MAX) so short rows are correct. MAX is associative =>
  // bit-exact; ADD/SUMSQ reassociate the vertical sum (absorbed by the reduce tolerance). Combinational
  // (nBanks-1 FP ops/lane, ~log2(nBanks) deep) => no extra cycle; nBanks==1 is a pass-through (legacy).
  val collapsed = Wire(Vec(lanes, UInt(accWidth.W)))
  if (nBanks == 1) {
    for (L <- 0 until lanes) collapsed(L) := outNowBanked(L)
  } else {
    val NEG_INF    = "hFF800000".U(accWidth.W)
    val validBanks = Mux(operandCount > nBanks.U, nBanks.U, operandCount)
    def collapseAdd(x: UInt, y: UInt): UInt = {
      val m = Module(new FpAdd(FP32, FP32, FP32, 0)); m.io.in_a := x; m.io.in_b := y; m.io.out
    }
    def collapseCombine(x: UInt, y: UInt): UInt =
      if (multiOp) Mux(opcode === OP_MAX, fp32max(x, y), collapseAdd(x, y))
      else if (hasMax) fp32max(x, y)
      else collapseAdd(x, y)
    val identity =
      if (multiOp) Mux(opcode === OP_MAX, NEG_INF, FP32_ZERO)
      else if (hasMax) NEG_INF
      else FP32_ZERO
    for (L <- 0 until lanes) {
      // b, L are Scala Ints here -> static flat index (no dynamic-width slice); mask banks not written this row
      val bankVals = (0 until nBanks).map(b => Mux(b.U < validBanks, outNowBanked(b * lanes + L), identity))
      collapsed(L) := bankVals.reduceLeft((x, y) => collapseCombine(x, y))
    }
  }

  // ---- horizontal fold: fully-pipelined PARALLEL tree (foldParallel: rows pipeline => multi-row hits the
  // roofline; costs lanes-1 FP adders) OR the TIME-MUXED in-place log-fold (area knob: treeLanes ALUs/cycle,
  // serialized by treeBusy). Both use the SAME balanced pairwise order (bit-identical scalar). ------------
  val (scalarFP32, scalarValid) = if (foldParallel) {
    // Pipelined balanced reduce tree: feed `collapsed` every cycle; the rowLastRetire snapshot's scalar
    // emerges numLevels*foldLatP cycles later (valid tracked). No treeBusy => input never stalls for the fold,
    // so consecutive rows pipeline through the tree at the accumulate rate (banked => 1 beat/cycle).
    val foldLatP = if (pipelined) scala.math.max(treePipe, 1) else 0
    def combineP(x: UInt, y: UInt): UInt = {
      val addA = if (hasAdd || hasSumsq) ShiftRegister(treeAdd(x, y), foldLatP - treePipe) else FP32_ZERO
      val maxv = if (hasMax) ShiftRegister(fp32max(x, y), foldLatP) else FP32_ZERO
      if (multiOp) Mux(opcode === OP_MAX, maxv, addA) else if (hasMax) maxv else addA
    }
    var lvl: Seq[UInt] = (0 until lanes).map(i => collapsed(i))
    for (_ <- 0 until numLevels) lvl = (0 until (lvl.length / 2)).map(i => combineP(lvl(2 * i), lvl(2 * i + 1)))
    treeBusy := false.B // the pipelined tree never stalls the input
    (lvl.head, clrPipe(rowLastRetire, numLevels * foldLatP))
  } else {
    // TIME-MUXED in-place log-fold: snapshot collapsed into treeBuf, reduce over numLevels rounds with
    // treeLanes ALUs/cycle; treeBusy holds input off (single big row hides it, multi-row pays a bubble).
    val treeBuf   = Reg(Vec(lanes, UInt(accWidth.W)))
    val foldLat   = if (pipelined) scala.math.max(treePipe, 1) else 0
    val laneIdxW  = log2Ceil(lanes)
    val foldRound = RegInit(0.U(log2Ceil(numLevels + 1).W))
    val foldCol   = RegInit(0.U(log2Ceil(lanes / 2 / treeLanes + 1).max(1).W))
    val foldWait  = RegInit(0.U(log2Ceil(foldLat + 1).max(1).W))
    val roundPairs  = (lanes.U >> (foldRound + 1.U))(laneIdxW, 0)
    val roundChunks = (roundPairs + (treeLanes - 1).U) / treeLanes.U
    val lastChunk   = foldCol === (roundChunks - 1.U)
    val foldRes = Wire(Vec(treeLanes, UInt(accWidth.W)))
    for (k <- 0 until treeLanes) {
      val pIdx = (foldCol * treeLanes.U + k.U)(laneIdxW, 0)
      val a    = treeBuf((pIdx << 1)(laneIdxW - 1, 0))
      val b    = treeBuf(((pIdx << 1) | 1.U)(laneIdxW - 1, 0))
      val addA = if (hasAdd || hasSumsq) ShiftRegister(treeAdd(a, b), foldLat - treePipe) else FP32_ZERO
      val maxv = if (hasMax) ShiftRegister(fp32max(a, b), foldLat) else FP32_ZERO
      foldRes(k) := (if (multiOp) Mux(opcode === OP_MAX, maxv, addA) else if (hasMax) maxv else addA)
    }
    val foldWriteback = treeBusy && (foldWait === foldLat.U)
    val treeDone      = RegInit(false.B)
    when(ext_start_i) {
      treeBusy := false.B; foldRound := 0.U; foldCol := 0.U; foldWait := 0.U; treeDone := false.B
    }.otherwise {
      treeDone := false.B
      when(!treeBusy) {
        when(rowLastRetire) { treeBuf := collapsed; treeBusy := true.B; foldRound := 0.U; foldCol := 0.U; foldWait := 0.U }
      }.otherwise {
        when(foldWriteback) {
          for (k <- 0 until treeLanes) {
            val pIdx = (foldCol * treeLanes.U + k.U)(laneIdxW, 0)
            when(pIdx < roundPairs) { treeBuf(pIdx(laneIdxW - 1, 0)) := foldRes(k) }
          }
          foldWait := 0.U
          when(lastChunk) {
            when(foldRound === (numLevels - 1).U) { treeBusy := false.B; treeDone := true.B }.otherwise {
              foldRound := foldRound + 1.U; foldCol := 0.U
            }
          }.otherwise { foldCol := foldCol + 1.U }
        }.otherwise { foldWait := foldWait + 1.U }
      }
    }
    assert(!(treeBusy && rowLastRetire), "StreamReduce fold: a row completed while the fold was still busy")
    (treeBuf(0), treeDone)
  }
  // Output packing: narrow the FP32 scalar to the transport grid (default), or — when fp32out is set and
  // the transport is narrower than FP32 — splat the raw FP32 scalar across the 512-bit beat (dataWidth/32
  // copies). The beat stays 512-bit either way, so the writer AGU is unchanged. (FP32 transport => nothing
  // to narrow, the mux folds away at elaboration.)
  val narrowedBeat  = Cat(Seq.fill(lanes)(narrow(scalarFP32)))
  val scalarBeat    =
    if (transport.width >= accWidth) narrowedBeat
    else Mux(fp32out, Cat(Seq.fill(extensionParam.dataWidth / accWidth)(scalarFP32)), narrowedBeat)

  // ---- push to the output queue: passthrough beat (tap, on accept) then the row's scalar (scalarValid) ----
  val passPush = tap && accept
  outQ.io.enq.valid := (scalarValid || passPush) && !ext_start_i // no stale push on the restart cycle
  outQ.io.enq.bits  := Mux(passPush, ext_data_i.bits, scalarBeat)
  assert(!(scalarValid && passPush), "StreamReduce: passthrough/scalar output collision (tap ordering bug)")
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "StreamReduce: output queue overflow (credit bug)")
  outQ.io.deq.ready := ext_data_o.ready
  ext_data_o.valid  := outQ.io.deq.valid
  ext_data_o.bits   := outQ.io.deq.bits

  // ---- issue-pointer + credit + tap-serialization update ----
  val deq             = outQ.io.deq.fire
  val acceptLastOfRow = accept && (nextIdx === (operandCount - 1.U))
  when(ext_start_i) {
    haveBeat := false.B; sub          := 0.U; beatIdx := 0.U; nextIdx := 0.U; stall := 0.U
    credit   := Qdepth.U; tapDraining := false.B; foldInFlight := false.B
  }.otherwise {
    when(stall =/= 0.U) { stall := stall - 1.U }
    when(accept) {
      inBeat  := ext_data_i.bits; haveBeat := true.B; sub := 0.U
      beatIdx := nextIdx
      nextIdx := Mux(nextIdx === (operandCount - 1.U), 0.U, nextIdx + 1.U)
    }.elsewhen(issuing) {
      when(lastSub) {
        when(overlapOK) { haveBeat := false.B }.otherwise { haveBeat := false.B; stall := gap.U }
      }.otherwise { sub := sub + 1.U }
    }
    when(tap && acceptLastOfRow) { tapDraining := true.B }.elsewhen(scalarValid) { tapDraining := false.B }
    // a row's fold is pending from its last-beat issue until its scalar retires: gate the next row start on it
    when(scalarValid) { foldInFlight := false.B }.elsewhen(rowLastIssue) { foldInFlight := true.B }
    credit := credit - Mux(accept, reserveOnAccept, 0.U) + Mux(deq, 1.U, 0.U)
  }

  ext_busy_o := (credit =/= Qdepth.U) || haveBeat || (stall =/= 0.U) || tapDraining || treeBusy || foldInFlight
}
