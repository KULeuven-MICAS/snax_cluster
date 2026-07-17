package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._
import fp_native._

/** StreamMapRt: RUNTIME-precision StreamMap -- the NONLINEAR ACTIVATION primitive. One elaborated netlist maps
  * `act(a*x+b)` over a streamed row at FP16 / BF16 / FP8 selected at runtime by a 2-bit `fmt` CSR field,
  * instead of the elaboration-time precision of StreamMap (one element width => one netlist per precision).
  *
  *   - Internal compute is FP32 (format-agnostic), so runtime precision is purely an EDGE problem:
  *     `widenRt`/`narrowRt` (3 compile-time converters muxed by `fmt`) at the input/output, plus a runtime
  *     beat slicer/packer. The FP32 MAC core (ffma) and the FpActivation LUTs are UNCHANGED.
  *   - Built for the MAX lane count `maxLanes = dataWidth/8 = 64` (FP8's element count). At FP16/BF16 only 32
  *     of the 64 slots carry data; the slicer/packer select the layout by `fmt`. Runtime-variable subCycles
  *     issues only the ACTIVE lanes of the current fmt (FP16/BF16 = maxLanes/2, FP8 = maxLanes), so every
  *     format reaches computeUtil=1.0 and the narrow ones run proportionally FASTER at a given computeLanes.
  *   - func in {LINEAR, EXP, SILU, GELU} selected at runtime (CSR). Precisions FP16/BF16/FP8 are ALL built.
  *
  *   DESIGN NOTE -- why this block is FP16/BF16/FP8 only (no MX): the activation (exp/silu/gelu) is a BF16
  *   operation in recent models; MX/FP8 is the GEMM-operand precision. So MX (microscaling) quantization is
  *   factored OUT into the dedicated StreamCastRt (the in-transit quantizer), and the expensive FpActivation
  *   LUT is never replicated onto the wide (128-lane) MX path. Chain them on the datapath for a fused epilogue:
  *   reader -> StreamMapRt (act, BF16) -> StreamCastRt (BF16 -> MX/FP8) -> writer, in one streaming pass.
  *
  * CSR layout: csr(0)=a (FP32 bits), csr(1)=b (FP32 bits), csr(2)= bits[1:0]=func (0=LINEAR,1=EXP,2=SILU,
  * 3=GELU), bits[3:2]=fmt (0=FP16,1=BF16,2=FP8).
  */
class HasStreamMapRt(
  computeLanes: Int,
  func:         Seq[String], // activation names, no precision tag: subset of {LINEAR, EXP, SILU, GELU}
  dataWidth:    Int = 512,
  fpPipe:       Int = 1
) extends HasDataPathExtension {
  require(computeLanes > 0, "HasStreamMapRt: computeLanes must be > 0")
  private val allowed = Set("LINEAR", "EXP", "SILU", "GELU")
  require(func.nonEmpty && func.toSet.subsetOf(allowed),
          s"HasStreamMapRt: func must be a non-empty subset of $allowed, got $func")
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(moduleName = "StreamMapRt", userCsrNum = 3, dataWidth = dataWidth)

  def instantiate(clusterName: String): StreamMapRt =
    Module(new StreamMapRt(computeLanes, func, fpPipe) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamMapRt(
  computeLanesParam: Int         = 8,
  func:              Seq[String] = Seq("LINEAR", "EXP", "SILU"),
  fpPipeParam:       Int         = 1,
  pipelined:         Boolean     = true
)(implicit
  extensionParam:    DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers._

  val maxLanes     = extensionParam.dataWidth / 8 // 64: FP8 element count = the max lanes/beat
  val computeLanes = if (computeLanesParam > maxLanes) maxLanes else computeLanesParam
  require(maxLanes % computeLanes == 0, "StreamMapRt: maxLanes must be a multiple of computeLanes")
  require(isPow2(computeLanes), "StreamMapRt: computeLanes must be a power of two (runtime-variable subCycles shift)")
  val subCycles = maxLanes / computeLanes // BUILD-max (widest format); the RUNTIME subCycles is `runSub` below

  val hasExp    = func.contains("EXP")
  val hasSilu   = func.contains("SILU")
  val hasGelu   = func.contains("GELU")
  val hasLinear = func.contains("LINEAR")
  val hasAct    = hasExp || hasSilu || hasGelu
  val expLutN   = 32
  val siluN     = 256
  def ACT_EXP  = 1.U
  def ACT_SILU = 2.U
  def ACT_GELU = 3.U

  val a        = ext_csr_i(0)
  val b        = ext_csr_i(1)
  val actField = ext_csr_i(2)
  val act      = actField(1, 0) // 0=LINEAR, 1=EXP, 2=SILU, 3=GELU
  val fmt      = actField(3, 2) // 0=FP16, 1=BF16, 2=FP8
  // RUNTIME-variable subCycles: issue only the ACTIVE lanes of the current fmt (FP16/BF16 use maxLanes/2, FP8
  // uses maxLanes), so every format reaches computeUtil=1.0 and the narrow ones run proportionally FASTER at a
  // given computeLanes (area).
  val activeLanes = Mux(fmt <= FMT_BF16.U, (maxLanes / 2).U, maxLanes.U)
  val runSub      = Mux(activeLanes < computeLanes.U, 1.U, activeLanes >> log2Ceil(computeLanes))

  // ---- pipeline depth (mirrors StreamMap so the streaming FSM is reused verbatim) ----
  val fpPipe  = if (pipelined) fpPipeParam else 0
  val actLat  = if (pipelined && hasAct) FpActivation.PipeLatency else 0
  val preLat  = if (pipelined) 2 * fpPipe + 1 else 0 // widenRt(fpPipe) -> ffma(fpPipe) -> register
  val postLat = if (pipelined) fpPipe + 1 else 0     // narrowRt(fpPipe) -> register
  val P       = preLat + actLat + postLat
  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  // one runtime-precision compute lane: widenRt(fmt) -> FP32 a*x+b -> act -> narrowRt(fmt). `carrier16` is
  // the 16-bit carrier (FP8 in the low 8) for this slot.
  def computeLane(carrier16: UInt): UInt = {
    val xF32   = widenRt(carrier16, fmt, fpPipe)         // +fpPipe: runtime transport -> FP32
    val t      = sr(ffma(a, xF32, b, fpPipe))            // +fpPipe+1: FP32 affine a*x+b
    val actES  = if (hasAct) {
      val m = Module(new FpActivation(pipelined, hasExp, hasSilu, hasGelu, expLutN, siluN))
      m.io.in := t
      m.io.func := (act === ACT_SILU) || (act === ACT_GELU)      // g-family select (vs exp)
      m.io.gelu := (if (hasGelu) act === ACT_GELU else false.B)  // gelu vs silu within the g-family
      m.io.out
    } else t
    val tD     = if (actLat > 0) ShiftRegister(t, actLat) else t
    val selAct = (if (hasExp) act === ACT_EXP else false.B) ||
      (if (hasSilu) act === ACT_SILU else false.B) || (if (hasGelu) act === ACT_GELU else false.B)
    val r      =
      if (!hasAct) tD
      else if (!hasLinear) actES
      else Mux(selAct, actES, tD)
    sr(narrowRt(r, fmt, fpPipe))                         // +fpPipe+1: FP32 -> runtime transport (16b carrier)
  }

  // ---- streaming time-mux + pipeline FSM (identical to StreamMap; lanes = maxLanes) ----
  val inBeat  = Reg(UInt(extensionParam.dataWidth.W))              // raw beat; sliced by fmt at issue
  val outSlot = Reg(Vec(maxLanes, UInt(16.W)))                     // per-slot 16b result carriers (FP8 low 8)

  // two static views of the input beat; carrierOf muxes by fmt (cheap Vec index, not a dynamic shift)
  val fp8View = inBeat.asTypeOf(Vec(maxLanes, UInt(8.W)))          // 64 FP8 slots
  val f16View = inBeat.asTypeOf(Vec(maxLanes / 2, UInt(16.W)))     // 32 FP16/BF16 slots
  def carrierOf(slot: UInt): UInt = {
    val c8  = Cat(0.U(8.W), fp8View(slot))                        // FP8: low 8 bits
    val c16 = Mux(slot < (maxLanes / 2).U, f16View(slot(log2Ceil(maxLanes / 2) - 1, 0)), 0.U(16.W))
    Mux(fmt === FMT_FP8.U, c8, c16)
  }

  // Size the queue / credit round-trip for the SMALLEST runtime subCycles (narrowest fmt FP16 = maxLanes/2
  // active lanes): a smaller runSub accepts beats FASTER -> more in flight -> a DEEPER queue. Using the build
  // subCycles under-sizes it and the narrow formats plateau below the roofline (FP16 stuck at 0.75 @ cl=32).
  val minRunSub       = scala.math.max(1, (maxLanes / 2) / computeLanes) // narrowest fmt = FP16 = maxLanes/2
  val inFlightMax     = (P + minRunSub - 1) / minRunSub + 1
  val creditRoundTrip = P + 4
  val Qdepth = scala.math.max(2,
    scala.math.max(inFlightMax, (creditRoundTrip + minRunSub - 1) / minRunSub + 1))
  val outQ = Module(new Queue(UInt(extensionParam.dataWidth.W), entries = Qdepth))

  def li(s: UInt, j: Int): UInt = (s * computeLanes.U + j.U)(log2Ceil(maxLanes) - 1, 0)

  val haveBeat = RegInit(false.B)
  val sub      = RegInit(0.U(log2Ceil(subCycles).max(1).W))
  val lastSub  = sub === (runSub - 1.U) // runtime count: narrow fmts finish in fewer subgroups
  val credit   = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  ext_data_i.ready := (!haveBeat || lastSub) && (credit =/= 0.U) && !ext_start_i
  val accept = ext_data_i.fire

  val issuing = haveBeat
  val res     = Wire(Vec(computeLanes, UInt(16.W)))
  for (j <- 0 until computeLanes) res(j) := computeLane(carrierOf(li(sub, j)))

  def clrPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(ext_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(ext_start_i, false.B, r(i - 1))
      r(n - 1)
    }

  val subRetire   = ShiftRegister(sub, P)
  val retireValid = clrPipe(issuing, P)
  val lastRetire  = subRetire === (runSub - 1.U) // fmt (hence runSub) is stable per task, so this aligns
  val outNow      = WireInit(outSlot)
  when(retireValid) {
    for (j <- 0 until computeLanes) {
      outSlot(li(subRetire, j)) := res(j)
      outNow(li(subRetire, j))  := res(j)
    }
  }

  // runtime packer: FP8 -> 64 bytes; FP16/BF16 -> 32 halfwords (slots 0..31). Cat puts index 0 in the low bits.
  val packed8  = Cat((0 until maxLanes).map(i => outNow(maxLanes - 1 - i)(7, 0)))
  val packed16 = Cat((0 until maxLanes / 2).map(i => outNow(maxLanes / 2 - 1 - i)(15, 0)))
  val outBeat  = Mux(fmt === FMT_FP8.U, packed8, packed16)

  outQ.io.enq.valid := retireValid && lastRetire && !ext_start_i
  outQ.io.enq.bits  := outBeat
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "StreamMapRt: output queue overflow (credit accounting bug)")
  outQ.io.deq.ready := ext_data_o.ready
  ext_data_o.valid  := outQ.io.deq.valid
  ext_data_o.bits   := outQ.io.deq.bits

  val deq = outQ.io.deq.fire
  when(ext_start_i) {
    haveBeat := false.B; sub := 0.U; credit := Qdepth.U
  }.otherwise {
    when(accept) {
      inBeat   := ext_data_i.bits; haveBeat := true.B; sub := 0.U
    }.elsewhen(issuing) {
      when(lastSub) { haveBeat := false.B; sub := 0.U }.otherwise { sub := sub + 1.U }
    }
    when(accept =/= deq) { credit := Mux(accept, credit - 1.U, credit + 1.U) }
  }

  ext_busy_o := (credit =/= Qdepth.U) || haveBeat
}
