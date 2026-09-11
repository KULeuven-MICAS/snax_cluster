package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._
import fp_native._

/** StreamElementwise: per-lane FP binary combine of `operandCount` consecutive input beats, emitted as ONE full output
  * beat (NO horizontal collapse — the element-wise sibling of StreamReduce).
  *
  *   - Transport: configurable element type (FP16/BF16/FP8/FP32, set by the op-set; e.g. 32 FP16 lanes per 512-bit
  *     beat). Internal math: FP32 (widen on input, narrow on output). op = MUL | ADD, runtime CSR.
  *   - The operands arrive as consecutive beats on the SINGLE input stream — the AGU interleaves them (an inner
  *     temporal dim of count=operandCount striding between the operand L1 regions), exactly as the ElementwiseAdd app
  *     does. For a binary op operandCount=2: out(i) = beat0(i) op beat1(i).
  *
  * Reuses ElementwiseAdd's operand-count accumulate-then-emit FSM + StreamReduce's FP32-internal op-set and
  * `computeLanes` time-mux, but emits the per-lane partials (Cat(regs)) instead of a reduced scalar (no reduceTree, no
  * tap).
  *
  * CSR layout: csr(0) = operandCount (#beats combined per output, 0->1); csr(1) bit[8] = sticky-B (latch the
  * first beat as operand B and combine every later beat against it -- see the decode below); csr(1) bits[7:0] = op (0=MUL, 1=ADD;
  * present/used only when >1 op is built).
  *
  * Configurable from the hjson: `op` (the LIST of supported ops, a subset of {MUL, ADD}), `computeLanes` (number of
  * physical combine ALUs, time-muxed over `subCycles` = lanes/computeLanes cycles per beat; computeLanes = lanes ⇒ the
  * fully-parallel 1-cycle/beat path), and `elementWidth` (must match the op-set precision: FP16/BF16⇒16, FP8⇒8,
  * FP32⇒32). All REQUIRED, e.g. {elementWidth:16, computeLanes:8, op:["MUL_FP16","ADD_FP16"]}.
  */
class HasStreamElementwise(
  computeLanes: Int,
  op:           Seq[String], // each entry "<OP>_<PRECISION>", e.g. "MUL_FP16"
  elementWidth: Int, // transport element width (bits); must match the op-set precision
  dataWidth:    Int = 512,
  fpPipe:       Int = 1 // internal pipeline depth of each FP unit (timing cut knob)
) extends HasDataPathExtension {
  private val (ops, transport) =
    OpSpec.parse(op, Set("MUL", "ADD", "FMA"), "HasStreamElementwise") // validate op names + precision
  OpSpec.checkWidth(elementWidth, transport, "HasStreamElementwise")
  require(computeLanes > 0, "HasStreamElementwise: computeLanes must be > 0")
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "StreamElementwise",
      // operandCount + (op-select). The op-select CSR is needed when both combines are runtime-selectable:
      // the fused FMA op provides both, and a legacy MUL+ADD set lists both. A single legacy op fixes it.
      userCsrNum = if (ops.contains("FMA") || ops.size > 1) 2 else 1,
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): StreamElementwise =
    Module(new StreamElementwise(computeLanes, op, elementWidth, fpPipe) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamElementwise(
  computeLanesParam: Int         = 0,
  op:                Seq[String] = Seq("MUL_FP16", "ADD_FP16"),
  elementWidth:      Int         = 16,
  fpPipeParam:       Int         = 1,
  pipelined:         Boolean     = true
)(implicit
  extensionParam:    DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers.{narrow => narrowT, widen => widenT}

  // transport (element) precision comes from the op-set; internal compute stays FP32
  val (ops, transport) = OpSpec.parse(op, Set("MUL", "ADD", "FMA"), "StreamElementwise")
  OpSpec.checkWidth(elementWidth, transport, "StreamElementwise")
  val accWidth         = 32 // FP32 internal
  val lanes            = extensionParam.dataWidth / elementWidth
  val computeLanes     = if (computeLanesParam <= 0 || computeLanesParam > lanes) lanes else computeLanesParam
  require(lanes % computeLanes == 0, "StreamElementwise: lanes must be a multiple of computeLanes")
  val subCycles = lanes / computeLanes

  // The single fused "FMA" op covers BOTH combines (MUL = acc*x+0, ADD = acc*1+x), selected at runtime by
  // the op CSR. hasMul/hasAdd are the runtime capabilities the built datapath must provide; the FMA op
  // provides both, so `bothOps` is true and the op CSR is honoured. Legacy single-op sets still build the
  // cheaper plain mul/add.
  val hasFMA  = ops.contains("FMA")
  val hasMul  = hasFMA || ops.contains("MUL")
  val hasAdd  = hasFMA || ops.contains("ADD")
  val bothOps = hasMul && hasAdd // both combines available => the FMA path with a runtime op-select

  // op CSR (ext_csr_i(1)[7:0]) values — UNCHANGED for source-compatibility: 0=MUL, 1=ADD.
  def OP_MUL = 0.U
  def OP_ADD = 1.U

  val FP32_ZERO = 0.U(accWidth.W)

  val csrOperandCount = ext_csr_i(0)
  val operandCount    = Mux(csrOperandCount === 0.U, 1.U(16.W), csrOperandCount(15, 0))
  // op-select CSR is read only when both combines are built; a single-op build fixes the op
  val opcode: UInt = if (bothOps) ext_csr_i(1)(7, 0) else if (hasAdd) OP_ADD else OP_MUL

  // STICKY-B (op CSR bit[8]): latch the FIRST beat of the task as operand B and combine every LATER beat
  // against it, instead of taking both operands from the stream.
  //
  // With operandCount=2 the two operands must arrive interleaved, which means a broadcast operand has to be
  // physically replicated once per data beat first -- a whole extra pass to write it and a doubled read
  // stream to consume it. The AGU cannot avoid that: a single affine address stream cannot hold one
  // operand's address fixed while the other walks, so the replication is not removable in software.
  //
  // This is worth a bit precisely because a broadcast operand is common ONCE THE DATA IS TRANSPOSED. In
  // FlashAttention the row maxima are one beat for the whole tile (one lane per query row), so `sm = S - m`
  // is a single latched beat against a 64-beat stream. In the untransposed form m varied per row and this
  // mode would not have applied at all.
  //
  // Use it with operandCount=1, so each beat is its own row: beat 0 seeds the latch and passes through
  // (its output is the broadcast value itself -- point the writer one beat early and discard it), and beats
  // 1..N emit op(B, beat). Input and output stay 1:1, so the credit accounting below is untouched.
  val stickyB: Bool = if (extensionParam.userCsrNum >= 2) ext_csr_i(1)(8).asBool else false.B

  // ---- pipeline depth (timing) ----------------------------------------------------------------
  // Each FP unit is internally pipelined by `fpPipe` (cfg "cutting" knob); plus laneLat to register the
  // time-mux lane select. Per-lane combine latency = laneLat + fpPipe (widen) + fpPipe (mul/add).
  val fpPipe  = if (pipelined) fpPipeParam else 0
  val laneLat = if (pipelined) 1 else 0
  val accLat  = laneLat + 2 * fpPipe
  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  def fmul(a: UInt, b: UInt):            UInt = {
    val m = Module(new FpMul(FP32, FP32, FP32, fpPipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }
  def fadd(a: UInt, b: UInt):            UInt = {
    val m = Module(new FpAdd(FP32, FP32, FP32, fpPipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }
  // one FMA covers BOTH ops when both are built: MUL = prev*x+0, ADD = prev*1+x. Bit-identical to a
  // separate FpMul/FpAdd (prev*1.0 is exact), so a single FP unit/lane replaces the mul+add pair.
  def ffma(a: UInt, b: UInt, c: UInt):   UInt = {
    val m = Module(new FpFma(FP32, FP32, FP32, fpPipe)); m.io.in_a := a; m.io.in_b := b; m.io.in_c := c; m.io.out
  }
  // Mixed-precision variants: multiply an FP32 operand by a RAW transport value (exact in FP32) -> a
  // 24x11 (@FP16) multiplier instead of 24x24, bit-identical to widening first. Used for the combine's
  // multiplicand (the ADD addend `c` still needs the widened FP32 value).
  def ffmaT(a: UInt, xT: UInt, c: UInt): UInt = {
    val m = Module(new FpFma(FP32, transport, FP32, fpPipe)); m.io.in_a := a; m.io.in_b := xT; m.io.in_c := c; m.io.out
  }
  def fmulT(a: UInt, xT: UInt):          UInt = {
    val m = Module(new FpMul(FP32, transport, FP32, fpPipe)); m.io.in_a := a; m.io.in_b := xT; m.io.out
  }
  def widen(h: UInt): UInt = widenT(h, transport, fpPipe) // transport -> FP32
  def narrow(f: UInt): UInt = narrowT(f, transport) // FP32 -> transport (output; not timing-critical)
  // +1.0 in the transport format (multiplicand for the ADD path through the mixed FMA)
  val ONE_T = ((((1 << (transport.expWidth - 1)) - 1) << transport.sigWidth)).U(transport.width.W)

  // one lane's new partial: combine `laneIn` into `prev` per the active op (only built ops built).
  // first beat seeds the partial with the (widened) input; later beats fold in via mul/add.
  // Pipeline: register the lane select (laneLat), widen (fpPipe), then combine (fpPipe); the seed and the
  // accumulator are ShiftRegister-aligned to the combine output. Total issue->result latency = accLat.
  // When both ops are built, the combine is ONE muxed FMA (prev*sel_b + sel_c) instead of a parallel
  // mul+add; single-op configs keep the plain (cheaper) mul or add.
  def accLane(laneIn: UInt, prev: UInt, first: Bool): UInt = {
    val laneR    = sr(laneIn)                     // +laneLat
    val widened  = widen(laneR)                   // +fpPipe (kept for the ADD addend + first-beat seed)
    val laneRd   = ShiftRegister(laneR, fpPipe)   // raw x aligned to the combine input (FP16)
    val prevA    = ShiftRegister(prev, laneLat + fpPipe)
    val firstA   = ShiftRegister(first, laneLat + fpPipe)
    val combined =                                // +fpPipe ; multiplicand fed raw (24x11 mult)
      if (bothOps) ffmaT(prevA, Mux(opcode === OP_ADD, ONE_T, laneRd), Mux(opcode === OP_ADD, widened, FP32_ZERO))
      else if (hasMul) fmulT(prevA, laneRd)
      else fadd(prevA, widened)
    val seedD    = ShiftRegister(widened, fpPipe) // first-beat seed, aligned to the combine output
    val firstAA  = ShiftRegister(firstA, fpPipe)
    Mux(firstAA, seedD, combined)
  }

  // ---- per-lane FP32 partials ----
  val regs = RegInit(VecInit(Seq.fill(lanes)(0.U(accWidth.W))))

  // ---- streaming time-mux + pipeline FSM --------------------------------------------------------------
  // Continuous-issue rewrite (was: per input beat accept -> issue subCycles -> DRAIN accLat idle -> re-accept,
  // paying the accLat drain on EVERY operand beat plus an emit stall per row). The computeLanes pipelined
  // ALUs now issue one sub-group per cycle across back-to-back operand beats AND back-to-back rows; the
  // per-lane partials retire accLat cycles later. A row's result (Cat(narrow(regs))) is captured the cycle
  // its last operand beat's last sub retires and pushed into a small output Queue (skid) for backpressure.
  //
  // Per-lane ACCUMULATOR RECURRENCE: operand beat k+1 reads regs[L] written by beat k. The time-mux spaces
  // successive same-lane issues by `subCycles` cycles, so the recurrence is met with NO stall when
  // subCycles >= accLat+1. When subCycles < accLat+1 (e.g. computeLanes==lanes => subCycles==1) a compile-
  // time `gap` = accLat+1-subCycles of idle cycles is inserted between accumulating beats WITHIN a row (row
  // boundaries need no gap: the next row's beat 0 SEEDS, it doesn't read regs). gap==0 => fully streamed.
  // A credit counter caps in-flight + queued rows to the queue depth (the FP pipeline has no stall input,
  // so it keeps draining after issue stops). CONTRACT: ext_start_i only asserts when idle (ext_busy_o low).
  val gap      = scala.math.max(0, accLat + 1 - subCycles) // inter-beat recurrence stall (compile-time)
  val beatSpan = subCycles + gap                           // issue cycles allotted per operand beat

  val inFlightMax = (accLat + beatSpan - 1) / beatSpan + 1 // rows still draining after issue stops
  // minimum credit-safe depth (see StreamMap): the credit caps outstanding, so the queue never overflows
  // below inFlightMax; slack only helps under sustained backpressure the fast writer never causes.
  val Qdepth      = scala.math.max(2, inFlightMax)
  val outQ        = Module(new Queue(UInt((lanes * elementWidth).W), entries = Qdepth))

  // index of lane (s*computeLanes + j) into the `lanes`-wide Vec, width-exact to silence W004
  def li(s: UInt, j: Int): UInt = (s * computeLanes.U + j.U)(log2Ceil(lanes) - 1, 0)

  val inBeat   = Reg(UInt((lanes * elementWidth).W))
  val inLanes  = inBeat.asTypeOf(Vec(lanes, UInt(elementWidth.W)))
  val haveBeat = RegInit(false.B)                         // a beat is latched and issuing
  val sub      = RegInit(0.U(log2Ceil(subCycles).max(1).W))
  val beatIdx  = RegInit(0.U(16.W))                       // operand index of the beat being issued
  val nextIdx  = RegInit(0.U(16.W))                       // operand index of the NEXT beat to accept
  val stall    = RegInit(0.U(log2Ceil(gap + 1).max(1).W)) // remaining recurrence-gap cycles (gap>0 only)
  val credit   = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  // The seed beat is the one that loads the sticky operand; it clears once that beat has been ISSUED, so
  // the next accepted beat combines against it rather than overwriting it.
  val seedPhase = RegInit(false.B)
  // ... and `regs` only holds the operand accLat cycles later, when the seed RETIRES. Every later beat
  // reads regs, so acceptance is held off until then.
  //
  // This is NOT the accumulate path's beat-to-beat recurrence: in sticky mode regs is held, so beat k+1
  // does not depend on beat k. The dependency is on the seed alone, which makes this a ONE-TIME wait of
  // accLat cycles per task instead of the `gap` inserted between every accumulating beat. Getting that
  // wrong is silent -- with operandCount=1 every beat looks like a row start, so the existing gap logic
  // inserts nothing and the first data beats read a stale latch.
  val seedDone = RegInit(false.B)

  val lastSub        = sub === (subCycles - 1).U
  val firstBeat      = beatIdx === 0.U
  val lastBeatInRow  = beatIdx === (operandCount - 1.U)
  val nextIsRowStart = nextIdx === 0.U // next accepted beat seeds a new row => no gap
  val overlapOK      = if (gap == 0) true.B else nextIsRowStart

  // accept a new input beat: overlap the current beat's last-sub issue when the successor needs no gap,
  // else from idle once the gap has drained. Reserve an output credit only when starting a new row.
  // Never overlap acceptance with the seed beat's own issue, or the successor would be latched before the
  // stall below can take effect.
  val acceptDuringIssue = haveBeat  && lastSub && overlapOK && !(stickyB && seedPhase)
  val acceptWhenIdle    = !haveBeat && (stall === 0.U)
  val creditOK          = !nextIsRowStart || (credit =/= 0.U)
  val stickyStall       = stickyB && !seedPhase && !seedDone // seed issued, latch not yet written
  ext_data_i.ready := (acceptDuringIssue || acceptWhenIdle) && creditOK && !ext_start_i && !stickyStall
  val accept = ext_data_i.fire

  // ---- issue ----
  val issuing = haveBeat
  val stickySeed = stickyB && seedPhase
  when(ext_start_i) {
    seedPhase := true.B
  }.elsewhen(issuing && lastSub && seedPhase) {
    seedPhase := false.B
  }
  val res     = Wire(Vec(computeLanes, UInt(accWidth.W)))
  for (j <- 0 until computeLanes)
    // In sticky mode only the seed beat seeds; every later beat reads regs (the latched operand B) as
    // `prev` and combines against it, which is exactly accLane's non-first path.
    res(j) := accLane(inLanes(li(sub, j)), regs(li(sub, j)), Mux(stickyB, stickySeed, firstBeat))

  // A valid-pulse pipeline CLEARED by ext_start_i, so a row-emit still in flight from the previous task
  // cannot fire — and push an unreserved beat past the credit reset — after the controller restarts the
  // next task while this pipeline is still draining.
  def clrPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(ext_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(ext_start_i, false.B, r(i - 1))
      r(n - 1)
    }

  // ---- retire: fold the partials into regs; assemble the row result at its last-sub retire ----
  val subRetire     = ShiftRegister(sub, accLat)
  val retireValid   = ShiftRegister(issuing, accLat, false.B, true.B)
  val rowLastIssue  = issuing && lastSub && lastBeatInRow
  val rowLastRetire = clrPipe(rowLastIssue, accLat)
  // Under the time mux a beat's result is assembled from `regs` at its LAST sub's retire, which works
  // because each earlier sub-group has already written its result there. Sticky mode holds regs (that is
  // the whole point), so the earlier sub-groups have nowhere to land and the assembled beat reads back the
  // broadcast in every lane but the last group's. `outRegs` is that landing place: always written, never
  // read as an operand. Costs lanes x 32 flops; invisible at computeLanes == lanes, which is why this only
  // showed up in the cl=8 test.
  val outRegs       = Reg(Vec(lanes, UInt(accWidth.W)))
  val seedRetiring  = ShiftRegister(stickySeed && issuing, accLat, false.B, true.B)
  // The latch is only complete when the seed's LAST sub-group has retired, not its first: under the time
  // mux the seed writes regs over `subCycles` cycles, so releasing on the leading edge lets the next beat
  // read lanes that are still stale. Same alignment as rowLastRetire.
  val seedFullyRetired = ShiftRegister(stickySeed && issuing && lastSub, accLat, false.B, true.B)
  when(ext_start_i) { seedDone := false.B }.elsewhen(seedFullyRetired) { seedDone := true.B }
  val holdSticky    = stickyB && !seedRetiring // keep operand B; only the result leaves
  val outBase       = Wire(Vec(lanes, UInt(accWidth.W)))
  for (i <- 0 until lanes) outBase(i) := Mux(stickyB, outRegs(i), regs(i))
  val outNow        = WireInit(outBase)
  when(retireValid) {
    for (j <- 0 until computeLanes) {
      when(!holdSticky) { regs(li(subRetire, j)) := res(j) }
      outRegs(li(subRetire, j)) := res(j)
      outNow(li(subRetire, j))  := res(j)
    }
  }
  // narrow each FP32 partial to transport (lane 0 low), captured combinationally so the whole beat is
  // grabbed before the next row's beat-0 seed (one cycle later) overwrites lane 0.
  outQ.io.enq.valid := rowLastRetire && !ext_start_i // no stale push on the restart cycle
  outQ.io.enq.bits  := Cat((0 until lanes).map(i => narrow(outNow(i))).reverse)
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "StreamElementwise: output queue overflow (credit bug)")
  outQ.io.deq.ready := ext_data_o.ready
  ext_data_o.valid  := outQ.io.deq.valid
  ext_data_o.bits   := outQ.io.deq.bits

  // ---- issue-pointer + credit update ----
  val deq       = outQ.io.deq.fire
  val doReserve = accept && nextIsRowStart
  when(ext_start_i) {
    haveBeat := false.B; sub := 0.U; beatIdx := 0.U; nextIdx := 0.U; stall := 0.U; credit := Qdepth.U
  }.otherwise {
    when(stall =/= 0.U) { stall := stall - 1.U }
    when(accept) {
      inBeat  := ext_data_i.bits; haveBeat := true.B; sub := 0.U
      beatIdx := nextIdx
      nextIdx := Mux(nextIdx === (operandCount - 1.U), 0.U, nextIdx + 1.U)
    }.elsewhen(issuing) {
      when(lastSub) {
        when(overlapOK) { haveBeat := false.B } // no accept this cycle: go idle
          .otherwise { haveBeat := false.B; stall := gap.U } // mid-row, gap>0: stall before next beat
      }.otherwise { sub := sub + 1.U }
    }
    when(doReserve =/= deq) { credit := Mux(doReserve, credit - 1.U, credit + 1.U) }
  }

  ext_busy_o := (credit =/= Qdepth.U) || haveBeat || (stall =/= 0.U)
}
