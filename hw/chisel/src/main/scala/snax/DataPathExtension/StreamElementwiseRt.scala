package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._
import fp_native._

/** StreamElementwiseRt: RUNTIME-precision StreamElementwise. One elaborated netlist combines `operandCount`
  * consecutive input beats element-wise (out(i) = beat0(i) op beat1(i) op ...) at FP16 / BF16 / FP8 selected
  * at RUNTIME by a 2-bit `fmt` CSR field, instead of StreamElementwise's elaboration-time precision (one
  * element width => one netlist per precision).
  *
  *   - Internal math is FP32 (format-agnostic), so runtime precision is an EDGE problem only: `widenRt` on
  *     input, `narrowRt` + a runtime beat packer on output (3 compile-time converters muxed by `fmt`). The
  *     FP32 combine (FMA), the accumulator banking, credit/queue and the streaming FSM are UNCHANGED from
  *     StreamElementwise. See FpHelpers.widenRt/narrowRt and dev_docs/xdma_ext/10-runtime-precision.md.
  *   - Built for the MAX lane count `maxLanes = dataWidth/8 = 64` (FP8's element count). At FP16/BF16 only 32
  *     of the 64 slots carry data; the slicer/packer pick the layout by `fmt`. With computeLanes = maxLanes
  *     (=64) subCycles=1, so BOTH FP16 and FP8 stream at 1 beat/cycle (full 512 b/cyc).
  *   - The single fused FP32 FMA covers BOTH combines (MUL = acc*x+0, ADD = acc*1+x); the `op` CSR selects
  *     at runtime, same as StreamElementwise's `bothOps` path.
  *
  * CSR layout: csr(0) bits[15:0]=operandCount (0->1), bits[17:16]=fmt (0=FP16,1=BF16,2=FP8);
  * csr(1) bits[7:0]=op (0=MUL, 1=ADD).
  */
class HasStreamElementwiseRt(
  computeLanes: Int,
  dataWidth:    Int = 512,
  fpPipe:       Int = 1,
  accBanks:     Int = 0 // per-lane partial banks (0 = auto = nextPow2(accLat+1); 1 = legacy single partial)
) extends HasDataPathExtension {
  require(computeLanes > 0, "HasStreamElementwiseRt: computeLanes must be > 0")
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(moduleName = "StreamElementwiseRt", userCsrNum = 2, dataWidth = dataWidth)

  def instantiate(clusterName: String): StreamElementwiseRt =
    Module(new StreamElementwiseRt(computeLanes, fpPipe, accBanks) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamElementwiseRt(
  computeLanesParam: Int     = 8,
  fpPipeParam:       Int     = 1,
  accBanksParam:     Int     = 0,
  pipelined:         Boolean = true
)(implicit
  extensionParam:    DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers._

  val accWidth     = 32 // FP32 internal
  val maxLanes     = extensionParam.dataWidth / 8 // 64: FP8 element count = the max lanes/beat
  val lanes        = maxLanes
  val computeLanes = if (computeLanesParam <= 0 || computeLanesParam > lanes) lanes else computeLanesParam
  require(lanes % computeLanes == 0, "StreamElementwiseRt: maxLanes must be a multiple of computeLanes")
  val subCycles = lanes / computeLanes

  // runtime op: always build the FMA path (MUL = acc*x+0, ADD = acc*1+x), op selected by CSR.
  def OP_MUL = 0.U
  def OP_ADD = 1.U
  val FP32_ZERO = 0.U(accWidth.W)
  val FP32_ONE  = "h3F800000".U(accWidth.W)

  val csr0         = ext_csr_i(0)
  val operandCount = Mux(csr0(15, 0) === 0.U, 1.U(16.W), csr0(15, 0))
  val fmt          = csr0(17, 16) // 0=FP16, 1=BF16, 2=FP8
  val opcode       = ext_csr_i(1)(7, 0)

  // ---- pipeline depth (timing) — identical to StreamElementwise (widenRt(fpPipe) == widen(fpPipe)) ----
  val fpPipe  = if (pipelined) fpPipeParam else 0
  val laneLat = if (pipelined) 1 else 0
  val accLat  = laneLat + 2 * fpPipe
  val nBanks =
    if (accBanksParam <= 0) (1 << log2Ceil(accLat + 1))
    else if (accBanksParam == 1) 1
    else (1 << log2Ceil(accBanksParam))
  def bankOf(idx: UInt): UInt = if (nBanks == 1) 0.U(1.W) else idx(log2Ceil(nBanks) - 1, 0)
  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  def fmul(a: UInt, b: UInt):          UInt = {
    val m = Module(new FpMul(FP32, FP32, FP32, fpPipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }
  def fadd(a: UInt, b: UInt):          UInt = {
    val m = Module(new FpAdd(FP32, FP32, FP32, fpPipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
  }
  def ffma(a: UInt, b: UInt, c: UInt): UInt = {
    val m = Module(new FpFma(FP32, FP32, FP32, fpPipe)); m.io.in_a := a; m.io.in_b := b; m.io.in_c := c; m.io.out
  }

  // one lane's new partial in FP32: widenRt(carrier,fmt) -> FP32, then combine into `prev` via the FP32 FMA.
  // first beat SEEDS the partial with the widened input; later beats fold in. Pipeline: register the lane
  // select (laneLat), widenRt (fpPipe), combine (fpPipe); seed + accumulator ShiftRegister-aligned. Total
  // issue->result latency = accLat (== StreamElementwise), so the streaming FSM below is reused verbatim.
  def accLane(carrier: UInt, prev: UInt, first: Bool): UInt = {
    val carR    = sr(carrier)                        // +laneLat
    val widened = widenRt(carR, fmt, fpPipe)         // +fpPipe -> FP32
    val prevA   = ShiftRegister(prev,  laneLat + fpPipe)
    val firstA  = ShiftRegister(first, laneLat + fpPipe)
    val combined =                                   // +fpPipe : MUL=prev*x+0, ADD=prev*1+x
      ffma(prevA, Mux(opcode === OP_ADD, FP32_ONE, widened), Mux(opcode === OP_ADD, widened, FP32_ZERO))
    val seedD   = ShiftRegister(widened, fpPipe)     // first-beat seed, aligned to the combine output
    val firstAA = ShiftRegister(firstA, fpPipe)
    Mux(firstAA, seedD, combined)
  }

  // ---- per-lane FP32 partials (nBanks per lane, round-robin by beat) ----
  val regs = RegInit(VecInit(Seq.fill(lanes * nBanks)(0.U(accWidth.W))))
  def bankIdx(bank: UInt, laneSel: UInt): UInt = (bank * lanes.U + laneSel)(log2Ceil(lanes * nBanks) - 1, 0)

  // ---- streaming time-mux + pipeline FSM (identical to StreamElementwise; lanes = maxLanes) ----
  val gap      = scala.math.max(0, accLat + 1 - nBanks * subCycles)
  val beatSpan = subCycles + gap

  val inFlightMax     = (accLat + beatSpan - 1) / beatSpan + 1
  val creditRoundTrip = accLat + 4
  val Qdepth = scala.math.max(2, scala.math.max(inFlightMax, (creditRoundTrip + beatSpan - 1) / beatSpan + 1))
  val outQ   = Module(new Queue(UInt(extensionParam.dataWidth.W), entries = Qdepth))

  def li(s: UInt, j: Int): UInt = (s * computeLanes.U + j.U)(log2Ceil(lanes) - 1, 0)

  val inBeat   = Reg(UInt(extensionParam.dataWidth.W)) // raw beat; sliced by fmt at issue
  // two static views of the input beat; carrierOf muxes by fmt (cheap Vec index, not a dynamic shift)
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
  val credit   = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  val lastSub        = sub === (subCycles - 1).U
  val firstInBank    = beatIdx < nBanks.U
  val issueBank      = bankOf(beatIdx)
  val lastBeatInRow  = beatIdx === (operandCount - 1.U)
  val nextIsRowStart = nextIdx === 0.U
  val overlapOK      = if (gap == 0) true.B else nextIsRowStart

  val acceptDuringIssue = haveBeat  && lastSub && overlapOK
  val acceptWhenIdle    = !haveBeat && (stall === 0.U)
  val creditOK          = !nextIsRowStart || (credit =/= 0.U)
  ext_data_i.ready := (acceptDuringIssue || acceptWhenIdle) && creditOK && !ext_start_i
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

  // ---- retire: fold the partials into regs; assemble the row result at its last-sub retire ----
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
  // collapse the nBanks per-lane partials into the per-lane FP32 result (masked: banks unwritten when
  // operandCount<nBanks contribute the op identity, 1.0 for MUL / +0.0 for ADD). Combinational, captured at
  // rowLastRetire, then narrowRt + runtime packer -> output beat.
  val collapsed = Wire(Vec(lanes, UInt(accWidth.W)))
  if (nBanks == 1) {
    for (L <- 0 until lanes) collapsed(L) := outNowBanked(L)
  } else {
    val validBanks = Mux(operandCount > nBanks.U, nBanks.U, operandCount)
    def cAdd(x: UInt, y: UInt): UInt = { val m = Module(new FpAdd(FP32, FP32, FP32, 0)); m.io.in_a := x; m.io.in_b := y; m.io.out }
    def cMul(x: UInt, y: UInt): UInt = { val m = Module(new FpMul(FP32, FP32, FP32, 0)); m.io.in_a := x; m.io.in_b := y; m.io.out }
    def cCombine(x: UInt, y: UInt): UInt = Mux(opcode === OP_ADD, cAdd(x, y), cMul(x, y))
    val identity = Mux(opcode === OP_ADD, FP32_ZERO, FP32_ONE)
    for (L <- 0 until lanes) {
      val bankVals = (0 until nBanks).map(b => Mux(b.U < validBanks, outNowBanked(b * lanes + L), identity))
      collapsed(L) := bankVals.reduceLeft((x, y) => cCombine(x, y))
    }
  }

  // runtime packer: FP8 -> 64 bytes; FP16/BF16 -> 32 halfwords (slots 0..31). Cat puts index 0 in the low bits.
  val packed8  = Cat((0 until lanes).map(i => narrowRt(collapsed(lanes - 1 - i), fmt)(7, 0)))
  val packed16 = Cat((0 until lanes / 2).map(i => narrowRt(collapsed(lanes / 2 - 1 - i), fmt)(15, 0)))
  val outBeat  = Mux(fmt === FMT_FP8.U, packed8, packed16)

  outQ.io.enq.valid := rowLastRetire && !ext_start_i
  outQ.io.enq.bits  := outBeat
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "StreamElementwiseRt: output queue overflow (credit bug)")
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
        when(overlapOK) { haveBeat := false.B }
          .otherwise { haveBeat := false.B; stall := gap.U }
      }.otherwise { sub := sub + 1.U }
    }
    when(doReserve =/= deq) { credit := Mux(doReserve, credit - 1.U, credit + 1.U) }
  }

  ext_busy_o := (credit =/= Qdepth.U) || haveBeat || (stall =/= 0.U)
}
