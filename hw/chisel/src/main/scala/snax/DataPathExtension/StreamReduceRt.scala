package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._
import fp_native._

/** StreamReduceRt: RUNTIME-precision StreamReduce. One elaborated netlist reduces a streamed row (vertical
  * across-beat + horizontal across-lane) to a single scalar, splatted across one output beat, at FP16 / BF16
  * / FP8 selected at RUNTIME by a 2-bit `fmt` CSR field — instead of StreamReduce's elaboration-time
  * precision (one element width => one netlist per precision).
  *
  *   - Internal math is FP32 (format-agnostic), so runtime precision is an EDGE problem only: `widenRt` on
  *     input, `narrowRt` + a runtime splat on output. The FP32 per-lane accumulate (FMA / max), the
  *     accumulator banking, the horizontal fold (parallel tree OR time-muxed log-fold), the tap passthrough
  *     and the credit/queue FSM are ALL unchanged from StreamReduce. See FpHelpers.widenRt/narrowRt.
  *   - Built for the MAX lane count `maxLanes = dataWidth/8 = 64` (FP8's element count). At FP16/BF16 only
  *     the low 32 lanes carry data; the slicer picks the layout by `fmt` and the HORIZONTAL FOLD masks the
  *     high 32 lanes to the op identity (-inf for MAX, +0 for ADD/SUMSQ) so a reduction over an all-negative
  *     FP16 row is not corrupted by the padding lanes. With computeLanes = maxLanes (=64) subCycles=1, so
  *     BOTH FP16 and FP8 stream at 1 beat/cycle (full 512 b/cyc).
  *   - op in {MAX, ADD, SUMSQ} selected at runtime (CSR); FMA covers ADD+SUMSQ. Precision is runtime, so the
  *     op list carries NO precision tag.
  *
  * CSR layout: csr(0) bits[15:0]=operandCount, bits[17:16]=fmt (0=FP16,1=BF16,2=FP8); csr(1) bits[7:0]=op
  * (0=MAX,1=ADD,2=SUMSQ), bit[8]=tap, bit[9]=fp32out (emit the scalar as splatted FP32, fmt-independent).
  */
class HasStreamReduceRt(
  computeLanes: Int,
  op:           Seq[String] = Seq("FMA", "MAX"), // capability list (no precision tag): subset of {MAX,ADD,SUMSQ,FMA}
  dataWidth:    Int = 512,
  fpPipe:       Int = 1,
  treePipe:     Int = 1,
  treeLanes:    Int = 2,
  accBanks:     Int = 0,
  foldParallel: Int = 0
) extends HasDataPathExtension {
  private val allowed = Set("MAX", "ADD", "SUMSQ", "FMA", "ARGMAX")
  require(op.nonEmpty && op.toSet.subsetOf(allowed),
          s"HasStreamReduceRt: op must be a non-empty subset of $allowed, got $op")
  require(computeLanes > 0, "HasStreamReduceRt: computeLanes must be > 0")
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(moduleName = "StreamReduceRt", userCsrNum = 2, dataWidth = dataWidth)

  def instantiate(clusterName: String): StreamReduceRt =
    Module(new StreamReduceRt(computeLanes, op, fpPipe, treePipe, treeLanes, accBanks, foldParallel != 0) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamReduceRt(
  computeLanesParam: Int         = 8,
  op:                Seq[String] = Seq("FMA", "MAX"),
  fpPipeParam:       Int         = 1,
  treePipeParam:     Int         = 1,
  treeLanesParam:    Int         = 2,
  accBanksParam:     Int         = 0,
  foldParallel:      Boolean     = false,
  pipelined:         Boolean     = true
)(implicit
  extensionParam:    DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers._

  val accWidth     = 32 // FP32 internal
  val maxLanes     = extensionParam.dataWidth / 8 // 64: FP8 element count = the max lanes/beat
  val lanes        = maxLanes
  val computeLanes = if (computeLanesParam <= 0 || computeLanesParam > lanes) lanes else computeLanesParam
  require(lanes % computeLanes == 0, "StreamReduceRt: maxLanes must be a multiple of computeLanes")
  val subCycles = lanes / computeLanes
  val treeLanes = scala.math.max(1, scala.math.min(treeLanesParam, lanes / 2))

  // which reductions are built (capability list; FMA provides both ADD and SUMSQ, op CSR selects at runtime)
  val hasFMA    = op.contains("FMA")
  val hasMax    = op.contains("MAX")
  val hasArgmax = op.contains("ARGMAX") // MoE gating: reduce to the INDEX of the max lane
  val hasAdd    = hasFMA || op.contains("ADD")
  val hasSumsq  = hasFMA || op.contains("SUMSQ")
  // ARGMAX reuses the MAX comparator; `hasCompare` gates the compare hardware (== hasMax when argmax absent,
  // so existing configs are byte-identical). multiOp = compare built alongside a sum => op CSR picks.
  val hasCompare = hasMax || hasArgmax
  val multiOp    = hasCompare && (hasAdd || hasSumsq)
  val idxWidth   = 32
  require(!hasArgmax || foldParallel, "StreamReduceRt: ARGMAX needs foldParallel (the index tree parallels the fold)")

  def OP_MAX    = 0.U
  def OP_ADD    = 1.U
  def OP_SUMSQ  = 2.U
  def OP_ARGMAX = 3.U

  val csr0            = ext_csr_i(0)
  val operandCount    = Mux(csr0(15, 0) === 0.U, 1.U(16.W), csr0(15, 0))
  val fmt             = csr0(17, 16)      // 0=FP16, 1=BF16, 2=FP8
  // RUNTIME-variable subCycles: issue only the ACTIVE lanes of the fmt (FP16/BF16=lanes/2, FP8=lanes). The
  // fold mask below already forces lanes >= lanes/2 to the op identity for non-FP8 (== the L<activeLanes
  // bound), so un-issued upper lanes don't corrupt the reduce. Banking meets the recurrence at runSub>=1.
  require(isPow2(computeLanes), "StreamReduceRt: computeLanes must be a power of two (runSub shift)")
  val activeLanes = Mux(fmt <= FMT_BF16.U, (lanes / 2).U, lanes.U)
  val runSub      = Mux(activeLanes < computeLanes.U, 1.U, activeLanes >> log2Ceil(computeLanes))
  val opField         = ext_csr_i(1)
  val opcode          = opField(7, 0)     // 0=MAX,1=ADD,2=SUMSQ,3=ARGMAX
  val tap             = opField(8).asBool
  val fp32out         = opField(9).asBool
  val argmaxMode = if (hasArgmax) opcode === OP_ARGMAX else false.B
  val wantMax    = argmaxMode || (if (hasMax) opcode === OP_MAX else false.B) // route the compare/max path

  val FP32_ZERO = 0.U(accWidth.W)
  val FP32_ONE  = "h3F800000".U(accWidth.W)
  val NEG_INF   = "hFF800000".U(accWidth.W)

  // ---- pipeline depths (timing) — identical accLat to StreamReduce (widenRt(fpPipe) + ffma(fpPipe)) ----
  val fpPipe     = if (pipelined) fpPipeParam else 0
  val treePipe   = if (pipelined) treePipeParam else 0
  val laneLat    = if (pipelined) 1 else 0
  val accLat     = laneLat + 2 * fpPipe
  val nBanks =
    if (accBanksParam <= 0) (1 << log2Ceil(accLat + 1))
    else if (accBanksParam == 1) 1
    else (1 << log2Ceil(accBanksParam))
  def bankOf(idx: UInt): UInt = if (nBanks == 1) 0.U(1.W) else idx(log2Ceil(nBanks) - 1, 0)
  val numLevels  = log2Ceil(lanes)
  val treeAddLat = numLevels * treePipe
  val treeMaxLat = if (pipelined) numLevels else 0
  val treeLat    = scala.math.max(treeAddLat, treeMaxLat)
  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  // FP32 max (finite inputs; NaN not expected from the host softmax path)
  def fp32max(a: UInt, b: UInt): UInt = Mux(fp32aWins(a, b), a, b)
  // true iff a >= b (finite FP32) — the exact comparator fp32max uses; reused to carry the argmax index.
  def fp32aWins(a: UInt, b: UInt): Bool = {
    val sa = a(accWidth - 1); val sb = b(accWidth - 1)
    Mux(sa =/= sb, !sa, Mux(sa, b >= a, a >= b))
  }
  // FP32^3 fused multiply-add (numPipe = fpPipe): ADD = x*1+prev, SUMSQ = x*x+prev. Runtime precision widens
  // to FP32 FIRST (the fmt-select can't be fused into the fixed-format fpnew FMA), so this is a plain FP32 FMA
  // rather than StreamReduce's mixed transport-FMA — the intended runtime-precision cost.
  def ffma(a: UInt, b: UInt, c: UInt): UInt = {
    val m = Module(new FpFma(FP32, FP32, FP32, fpPipe)); m.io.in_a := a; m.io.in_b := b; m.io.in_c := c; m.io.out
  }
  def treeAdd(a: UInt, b: UInt): UInt = {
    val m = Module(new FpAdd(FP32, FP32, FP32, treePipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }

  // one lane's new partial in FP32: widenRt(carrier,fmt) -> FP32, then accumulate into `prev`.
  //   ADD/SUMSQ: FP32 FMA  acc' = x*y + (first?0:prev),  y = x (SUMSQ) or 1.0 (ADD).
  //   MAX:       acc' = first ? x : max(x, prev), registered to accLat.
  // Total issue->result latency = accLat = laneLat + 2*fpPipe (== StreamReduce), so the FSM is reused verbatim.
  def accLane(carrier: UInt, prev: UInt, first: Bool): UInt = {
    val carR    = sr(carrier)                        // +laneLat
    val widened = widenRt(carR, fmt, fpPipe)         // +fpPipe -> FP32
    val prevA   = ShiftRegister(prev,  laneLat + fpPipe)
    val firstA  = ShiftRegister(first, laneLat + fpPipe)
    val fmaY    =
      if (hasSumsq && hasAdd) Mux(opcode === OP_SUMSQ, widened, FP32_ONE)
      else if (hasSumsq) widened
      else FP32_ONE
    val accAdd  =
      if (hasAdd || hasSumsq) ffma(widened, fmaY, Mux(firstA, FP32_ZERO, prevA)) // +fpPipe
      else FP32_ZERO
    val accMax  = if (hasCompare) ShiftRegister(Mux(firstA, widened, fp32max(widened, prevA)), fpPipe) else FP32_ZERO
    if (multiOp) Mux(wantMax, accMax, accAdd)
    else if (hasCompare) accMax
    else accAdd
  }

  // ---- per-lane FP32 partials (nBanks independent partials per lane, round-robin by beat) ----
  val regs = RegInit(VecInit(Seq.fill(lanes * nBanks)(0.U(accWidth.W))))
  def bankIdx(bank: UInt, laneSel: UInt): UInt = (bank * lanes.U + laneSel)(log2Ceil(lanes * nBanks) - 1, 0)

  // ---- streaming time-mux + pipeline FSM (identical to StreamReduce; lanes = maxLanes) ----
  val gap      = scala.math.max(0, accLat + 1 - nBanks * subCycles)
  val beatSpan = subCycles + gap

  def li(s: UInt, j: Int): UInt = (s * computeLanes.U + j.U)(log2Ceil(lanes) - 1, 0)

  val inBeat   = Reg(UInt(extensionParam.dataWidth.W)) // raw beat; sliced by fmt at issue
  val fp8View  = inBeat.asTypeOf(Vec(maxLanes, UInt(8.W)))      // 64 FP8 slots
  val f16View  = inBeat.asTypeOf(Vec(maxLanes / 2, UInt(16.W))) // 32 FP16/BF16 slots
  def carrierOf(slot: UInt): UInt = {
    val c8  = Cat(0.U(8.W), fp8View(slot))
    val c16 = Mux(slot < (maxLanes / 2).U, f16View(slot(log2Ceil(maxLanes / 2) - 1, 0)), 0.U(16.W))
    Mux(fmt === FMT_FP8.U, c8, c16)
  }

  val haveBeat = RegInit(false.B)
  val sub      = RegInit(0.U(log2Ceil(subCycles).max(1).W))
  val beatIdx  = RegInit(0.U(16.W))
  val nextIdx  = RegInit(0.U(16.W))
  val stall    = RegInit(0.U(log2Ceil(gap + 1).max(1).W))

  val lastSub        = sub === (runSub - 1.U) // runtime count: narrow fmts finish in fewer subgroups
  val firstInBank    = beatIdx < nBanks.U
  val issueBank      = bankOf(beatIdx)
  val lastBeatInRow  = beatIdx === (operandCount - 1.U)
  val nextIsRowStart = nextIdx === 0.U
  val overlapOK      = if (gap == 0) true.B else nextIsRowStart

  // size for the SMALLEST runtime beatSpan (narrowest fmt FP16 = lanes/2 active) so the faster narrow-format
  // beat rate doesn't starve the credit round-trip (see StreamMapRt).
  val minRunSub   = scala.math.max(1, (lanes / 2) / computeLanes)
  val beatSpanMin = minRunSub + gap
  val inFlightMax = (accLat + treeLat + beatSpanMin - 1) / beatSpanMin + 1
  val creditRoundTrip = accLat + treeLat + 4
  val Qdepth = scala.math.max(3,
    scala.math.max(inFlightMax + 1, (creditRoundTrip + beatSpanMin - 1) / beatSpanMin + 1))
  val outQ        = Module(new Queue(UInt(extensionParam.dataWidth.W), entries = Qdepth))
  val credit      = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))
  val tapDraining  = RegInit(false.B)
  val treeBusy     = RegInit(false.B)
  val foldInFlight = RegInit(false.B)

  val reserveOnAccept   = Mux(nextIsRowStart, 1.U, 0.U) +& Mux(tap, 1.U, 0.U)
  val acceptDuringIssue = haveBeat  && lastSub        && overlapOK
  val acceptWhenIdle    = !haveBeat && (stall === 0.U)
  val creditOK          = credit    >= reserveOnAccept
  val tapBlock          = tap       && nextIsRowStart && tapDraining
  val foldBlock         = nextIsRowStart && (foldInFlight || (haveBeat && lastSub && lastBeatInRow)) &&
                          (if (foldParallel) tap else true.B)
  ext_data_i.ready := (acceptDuringIssue || acceptWhenIdle) && creditOK && !tapBlock && !foldBlock && !treeBusy && !ext_start_i
  val accept = ext_data_i.fire

  // ---- issue ----
  val issuing = haveBeat
  val res     = Wire(Vec(computeLanes, UInt(accWidth.W)))
  for (j <- 0 until computeLanes)
    res(j) := accLane(carrierOf(li(sub, j)), regs(bankIdx(issueBank, li(sub, j))), firstInBank)

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
  val bankRetire    = ShiftRegister(issueBank, accLat)
  val retireValid   = ShiftRegister(issuing, accLat, false.B, true.B)
  val rowLastIssue  = issuing && lastSub && lastBeatInRow
  val rowLastRetire = clrPipe(rowLastIssue, accLat)
  val outNowBanked  = WireInit(regs)
  when(retireValid) {
    for (j <- 0 until computeLanes) {
      regs(bankIdx(bankRetire, li(subRetire, j)))         := res(j)
      outNowBanked(bankIdx(bankRetire, li(subRetire, j))) := res(j)
    }
  }

  // ---- bank collapse: combine the nBanks per-lane partials into ONE partial per lane (masked identity for
  // banks a short row never wrote). Combinational, captured at rowLastRetire. ----
  val collapsed = Wire(Vec(lanes, UInt(accWidth.W)))
  if (nBanks == 1) {
    for (L <- 0 until lanes) collapsed(L) := outNowBanked(L)
  } else {
    val validBanks = Mux(operandCount > nBanks.U, nBanks.U, operandCount)
    def collapseAdd(x: UInt, y: UInt): UInt = {
      val m = Module(new FpAdd(FP32, FP32, FP32, 0)); m.io.in_a := x; m.io.in_b := y; m.io.out
    }
    def collapseCombine(x: UInt, y: UInt): UInt =
      if (multiOp) Mux(wantMax, fp32max(x, y), collapseAdd(x, y))
      else if (hasCompare) fp32max(x, y)
      else collapseAdd(x, y)
    val identity =
      if (multiOp) Mux(wantMax, NEG_INF, FP32_ZERO)
      else if (hasCompare) NEG_INF
      else FP32_ZERO
    for (L <- 0 until lanes) {
      val bankVals = (0 until nBanks).map(b => Mux(b.U < validBanks, outNowBanked(b * lanes + L), identity))
      collapsed(L) := bankVals.reduceLeft((x, y) => collapseCombine(x, y))
    }
  }

  // ---- lane mask for the horizontal fold: at FP16/BF16 only the low maxLanes/2 lanes carry data; force the
  // high lanes to the op identity so a reduction over an all-negative FP16 row is not corrupted by the padding
  // (MAX would otherwise pick the padding 0.0). At FP8 all lanes are valid (pass-through). ----
  val foldIdentity =
    if (multiOp) Mux(wantMax, NEG_INF, FP32_ZERO)
    else if (hasCompare) NEG_INF
    else FP32_ZERO
  val foldIn = Wire(Vec(lanes, UInt(accWidth.W)))
  for (L <- 0 until lanes) {
    if (L < maxLanes / 2) foldIn(L) := collapsed(L)
    else foldIn(L) := Mux(fmt === FMT_FP8.U, collapsed(L), foldIdentity)
  }

  // ---- horizontal fold: parallel pipelined tree OR time-muxed log-fold (both feed `foldIn`) ----
  // scalarIdx = the argmax winner's lane index (single-beat ARGMAX; multi-beat vertical index is a noted
  // extension). Only meaningful when hasArgmax; else driven 0 and DCE'd.
  val scalarIdx = Wire(UInt(idxWidth.W))
  val (scalarFP32, scalarValid) = if (foldParallel) {
    val foldLatP = if (pipelined) scala.math.max(treePipe, 1) else 0
    def combineP(x: UInt, y: UInt): UInt = {
      val addA = if (hasAdd || hasSumsq) ShiftRegister(treeAdd(x, y), foldLatP - treePipe) else FP32_ZERO
      val maxv = if (hasCompare) ShiftRegister(fp32max(x, y), foldLatP) else FP32_ZERO
      if (multiOp) Mux(wantMax, maxv, addA) else if (hasCompare) maxv else addA
    }
    var lvl: Seq[UInt] = (0 until lanes).map(i => foldIn(i))
    for (_ <- 0 until numLevels) lvl = (0 until (lvl.length / 2)).map(i => combineP(lvl(2 * i), lvl(2 * i + 1)))
    // parallel argmax INDEX tree: leaf index = lane L; at each node keep the winner's index (fp32aWins is the
    // SAME comparator fp32max uses), with the identical per-level ShiftRegister so the index rides the value.
    if (hasArgmax) {
      var vlvl: Seq[UInt] = (0 until lanes).map(i => foldIn(i))
      var ilvl: Seq[UInt] = (0 until lanes).map(i => i.U(idxWidth.W))
      for (_ <- 0 until numLevels) {
        val vN = (0 until vlvl.length / 2).map(i => ShiftRegister(fp32max(vlvl(2 * i), vlvl(2 * i + 1)), foldLatP))
        val iN = (0 until ilvl.length / 2).map(i =>
          ShiftRegister(Mux(fp32aWins(vlvl(2 * i), vlvl(2 * i + 1)), ilvl(2 * i), ilvl(2 * i + 1)), foldLatP))
        vlvl = vN; ilvl = iN
      }
      scalarIdx := ilvl.head
    } else scalarIdx := 0.U
    treeBusy := false.B
    (lvl.head, clrPipe(rowLastRetire, numLevels * foldLatP))
  } else {
    scalarIdx := 0.U
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
        when(rowLastRetire) { treeBuf := foldIn; treeBusy := true.B; foldRound := 0.U; foldCol := 0.U; foldWait := 0.U }
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
    assert(!(treeBusy && rowLastRetire), "StreamReduceRt fold: a row completed while the fold was still busy")
    (treeBuf(0), treeDone)
  }

  // ---- output packing: narrowRt the FP32 scalar to the runtime transport grid and SPLAT across the beat, or
  // (fp32out) splat the raw FP32 scalar (fmt-independent). FP8 -> 64 byte copies; FP16/BF16 -> 32 halfwords. ----
  val nCarrier     = narrowRt(scalarFP32, fmt)                    // 16b carrier (FP8 in low 8)
  val splat8       = Cat(Seq.fill(maxLanes)(nCarrier(7, 0)))      // 64 * 8  = 512
  val splat16      = Cat(Seq.fill(maxLanes / 2)(nCarrier(15, 0))) // 32 * 16 = 512
  val narrowedBeat = Mux(fmt === FMT_FP8.U, splat8, splat16)
  val fp32Beat     = Cat(Seq.fill(extensionParam.dataWidth / accWidth)(scalarFP32))
  // ARGMAX: emit the winning lane INDEX (raw 32-bit int, splatted; read the low 32 bits as an integer),
  // fmt-independent — the MoE router wants the expert index, not a value.
  val idxBeat      = Cat(Seq.fill(extensionParam.dataWidth / idxWidth)(scalarIdx))
  val scalarBeat   =
    if (hasArgmax) Mux(argmaxMode, idxBeat, Mux(fp32out, fp32Beat, narrowedBeat))
    else Mux(fp32out, fp32Beat, narrowedBeat)

  // ---- push to the output queue: passthrough beat (tap, on accept) then the row's scalar (scalarValid) ----
  val passPush = tap && accept
  outQ.io.enq.valid := (scalarValid || passPush) && !ext_start_i
  outQ.io.enq.bits  := Mux(passPush, ext_data_i.bits, scalarBeat)
  assert(!(scalarValid && passPush), "StreamReduceRt: passthrough/scalar output collision (tap ordering bug)")
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "StreamReduceRt: output queue overflow (credit bug)")
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
    when(scalarValid) { foldInFlight := false.B }.elsewhen(rowLastIssue) { foldInFlight := true.B }
    credit := credit - Mux(accept, reserveOnAccept, 0.U) + Mux(deq, 1.U, 0.U)
  }

  ext_busy_o := (credit =/= Qdepth.U) || haveBeat || (stall =/= 0.U) || tapDraining || treeBusy || foldInFlight
}
