// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._

/** Fp16ToInt8PE: symmetric per-tensor quantize of one FP16 lane to a signed int8.
  *
  * q = sat_[-127,127]( round_rne( x * inv_scale ) ) // x an FP16 lane, computed in FP32
  *
  * The mirror image of Int32ToFp16PE. FP16 transport in, FP32 internal, SInt(8) out. The scale is a MIXED multiply —
  * FpMul(FP32, FP16, FP32) fed the raw FP16 lane (exact in FP32), so it needs no widen and a smaller 24x11 multiplier.
  * The fp32->int round uses the 1.5*2^23 magic-number add (no F2I unit), the same trick as FpActivation. The product is
  * first clamped into [-128, 128] so the magic add stays exact (|value| << 2^23) and the rounded integer lands in a
  * tiny signed range; the result is then saturated to the SYMMETRIC int8 range [-127, 127] (zero-point = 0).
  *
  * PIPELINING (timing): the mixed-mul->clamp->fadd->saturate chain is ~2 FP ops deep; when `pipelined` is set it is cut
  * into `latency` (=2) register stages (after the multiply and after the magic add) so it fits the clock.
  * `pipelined=false` keeps the original combinational PE (used by the standalone PE tester for an exact same-cycle
  * compare).
  */
class Fp16ToInt8PE(pipelined: Boolean = false, fpPipeParam: Int = 0) extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in        = Input(UInt(16.W)) // FP16
    val inv_scale = Input(UInt(32.W)) // FP32 (= 1 / quant_scale)
    val out       = Output(SInt(8.W))
  })

  import FpHelpers._

  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u
  val fpPipe = if (pipelined) fpPipeParam else 0 // internal FP-unit pipeline depth (cfg cut knob)

  val MAGIC = f32lit(12582912.0f) // 1.5 * 2^23 = 0x4B400000
  val HI    = f32lit(128.0f)      // pre-round clamp window (keeps the RNE exact, bounds iM)
  val LO    = f32lit(-128.0f)

  // ---- stage 0: multiply the raw fp16 lane by the FP32 inv_scale (mixed FP32*FP16 -> FP32: 24x11 mult, no
  // widen; the fp16 value is exact in FP32 so this is bit-identical). ShiftRegister keeps the old
  // widen(fpPipe)+fmul(fpPipe) latency so the streaming pack FSM is unchanged. ----
  val inD    = ShiftRegister(io.in, fpPipe)
  val scaled = sr(fmulT(io.inv_scale, inD, FP16, fpPipe)) // reg0

  // ---- stage 1: clamp into [-128, 128] then round to nearest (ties to even) via the magic add ----
  val clamped = fp32max(fp32min(scaled, HI), LO)
  val rM      = sr(fadd(clamped, MAGIC, fpPipe)) // reg1
  val iM      = rM.asSInt - 0x4b400000.S         // |iM| <= 128 after the clamp

  // ---- symmetric saturate to [-127, 127] ----
  val q = WireDefault(0.S(8.W))
  when(iM > 127.S) {
    q := 127.S(8.W)
  }.elsewhen(iM < (-127).S) {
    q := (-127).S(8.W)
  }.otherwise {
    q := iM(7, 0).asSInt // exact: |iM| <= 127 in this branch
  }

  io.out := q
}

object Fp16ToInt8PE {

  /** Register-stage latency of the pipelined PE: the 2 sr() cuts (after the multiply and the magic add) plus `fpPipe`
    * internal registers in each of the input-align shift / mixed-mul / fadd (= 2 + 3*fpPipe). 0 if not pipelined.
    */
  def pipeLatency(pipelined: Boolean, fpPipe: Int): Int = if (pipelined) 2 + 3 * fpPipe else 0
}

/** Fp16ToInt8: stream FP16 -> INT8 quantize extension. Packs `pack` input beats (e.g. 2 beats of 32 FP16 = 512b each)
  * into 1 output beat (64 INT8 = 512b). One FP32 CSR carries inv_scale. Chains LAST in the reader extension list so a
  * vector op (StreamMap / StreamElementwise) computes fp16 and the cast narrows it to int8 before it leaves the stream.
  *
  * TIME-MULTIPLEXED (area): only `computeLanes` quantize PEs are built and swept over `subCycles` (= nPE/computeLanes)
  * cycles per input beat, so a 512-bit beat needs `computeLanes` PEs instead of one per element. computeLanes >= nPE ⇒
  * fully parallel (subCycles=1). The issue FSM STREAMS: it feeds one sub-group per cycle across back-to-back beats and
  * the PE pipeline latency is a one-time fill, so steady-state throughput is 1 input beat / `subCycles` cycles (packing
  * `pack` input beats into each output beat via a small skid Queue), independent of the PE depth.
  *
  * NOTE: the pack ratio requires an integer multiple of `pack` input beats (a lone trailing beat is latched but never
  * emitted). All current SIMD apps use row lengths that are a multiple of 64 FP16.
  */
class Fp16ToInt8(
  in_elementWidth:   Int     = 16,
  out_elementWidth:  Int     = 8,
  computeLanesParam: Int     = 0,
  fpPipeParam:       Int     = 1,
  pipelined:         Boolean = true
)(implicit extensionParam: DataPathExtensionParam)
    extends DataPathExtension {

  require(
    extensionParam.dataWidth % in_elementWidth == 0,
    s"Fp16ToInt8: dataWidth (${extensionParam.dataWidth}) must be a multiple of in_elementWidth ($in_elementWidth)"
  )
  require(
    in_elementWidth == 16   && out_elementWidth == 8,
    s"Fp16ToInt8: only FP16(16)->INT8(8) supported, got $in_elementWidth->$out_elementWidth"
  )

  val nPE          = extensionParam.dataWidth / in_elementWidth  // input elements per beat (e.g. 32)
  val pack         = in_elementWidth / out_elementWidth          // input beats per output beat (e.g. 2)
  val outElems     = extensionParam.dataWidth / out_elementWidth // int8 per output beat (e.g. 64)
  val computeLanes = if (computeLanesParam <= 0 || computeLanesParam > nPE) nPE else computeLanesParam
  require(nPE % computeLanes == 0, "Fp16ToInt8: nPE must be a multiple of computeLanes")
  val subCycles = nPE / computeLanes
  val fpPipe    = if (pipelined) fpPipeParam else 0
  val Ppe       = Fp16ToInt8PE.pipeLatency(pipelined, fpPipe)

  val inv_scale = WireInit(ext_csr_i(0).asUInt)

  // ---- streaming time-mux + pipeline FSM (continuous-issue; see StreamMap for the rationale) -----------
  // Was: accept 1 beat -> issue subCycles -> DRAIN Ppe idle -> pack -> emit -> re-accept (per-beat bubble).
  // Now: the `computeLanes` PEs quantize one sub-group per cycle across back-to-back beats; each input
  // beat's int8 results retire into a DISTINCT nPE-wide slice of the output beat, and after `pack` input
  // beats fill it the beat is pushed into a small skid Queue. Each element is independent (NO FP
  // accumulator recurrence), so -- unlike StreamReduce/StreamElementwise -- there is no `gap`: it streams
  // at 1 input beat / subCycles cycles for ANY computeLanes. A credit counter reserves one output slot per
  // `pack` beats so the FP pipeline (no stall input) never drops a retired beat under backpressure.
  // CONTRACT: ext_start_i only asserts when idle (ext_busy_o low, guaranteed by the orchestration).
  val inBeat  = Reg(UInt((nPE * in_elementWidth).W))
  val inLanes = inBeat.asTypeOf(Vec(nPE, UInt(in_elementWidth.W)))
  val outRegs = Reg(Vec(outElems, SInt(out_elementWidth.W)))

  // valid-pulse pipeline CLEARED by ext_start_i (a pack-emit still draining from the previous task must
  // not push an unreserved beat after the credit reset).
  def clrPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(ext_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(ext_start_i, false.B, r(i - 1))
      r(n - 1)
    }

  val inFlightMax = (Ppe + subCycles - 1) / subCycles + 1 // beats still draining after issue stops
  // minimum credit-safe depth (see StreamMap): the credit caps outstanding, so the queue never overflows
  // below inFlightMax; slack only helps under sustained backpressure the fast writer never causes.
  val Qdepth      = scala.math.max(2, inFlightMax)
  val outQ        = Module(new Queue(UInt(extensionParam.dataWidth.W), entries = Qdepth))
  val credit      = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  val haveBeat        = RegInit(false.B)
  val sub             = RegInit(0.U(log2Ceil(subCycles).max(1).W))
  val packIdx         = RegInit(0.U(log2Ceil(pack).max(1).W)) // pack-index of the beat being issued
  val nextPack        = RegInit(0.U(log2Ceil(pack).max(1).W)) // pack-index of the NEXT beat to accept
  val lastSub         = sub === (subCycles - 1).U
  val lastInPack      = packIdx === (pack - 1).U
  val nextIsPackStart = nextPack === 0.U

  // accept a new beat when finishing the current one this cycle (overlap; no recurrence => no gap) or idle;
  // reserve one output-beat credit only when starting a new pack.
  val creditOK = !nextIsPackStart || (credit =/= 0.U)
  ext_data_i.ready := ((haveBeat && lastSub) || !haveBeat) && creditOK && !ext_start_i
  val accept = ext_data_i.fire

  // int8 output index for (pack beat p, sub s, lane j) and input-lane index, width-exact to silence W004
  def oi(p: UInt, s: UInt, j: Int): UInt = (p * nPE.U + s * computeLanes.U + j.U)(log2Ceil(outElems) - 1, 0)
  def li(s: UInt, j: Int): UInt = (s * computeLanes.U + j.U)(log2Ceil(nPE) - 1, 0)

  val issuing = haveBeat
  val res     = Wire(Vec(computeLanes, SInt(out_elementWidth.W)))
  for (j <- 0 until computeLanes) {
    val PE = Module(new Fp16ToInt8PE(pipelined, fpPipe) {
      override def desiredName = extensionParam.moduleName + "_fp16_to_int8_pe"
    })
    PE.io.in := inLanes(li(sub, j))
    PE.io.inv_scale := inv_scale
    res(j)          := PE.io.out
  }

  val subRetire      = ShiftRegister(sub, Ppe)
  val packRetire     = ShiftRegister(packIdx, Ppe)
  val retireValid    = ShiftRegister(issuing, Ppe, false.B, true.B)
  val packLastIssue  = issuing && lastSub && lastInPack
  val packLastRetire = clrPipe(packLastIssue, Ppe)
  val outNow         = WireInit(outRegs)
  when(retireValid) {
    for (j <- 0 until computeLanes) {
      outRegs(oi(packRetire, subRetire, j)) := res(j)
      outNow(oi(packRetire, subRetire, j))  := res(j)
    }
  }

  outQ.io.enq.valid := packLastRetire && !ext_start_i // no stale push on the restart cycle
  outQ.io.enq.bits  := outNow.asTypeOf(UInt(extensionParam.dataWidth.W))
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "Fp16ToInt8: output queue overflow (credit bug)")
  outQ.io.deq.ready := ext_data_o.ready
  ext_data_o.valid  := outQ.io.deq.valid
  ext_data_o.bits   := outQ.io.deq.bits

  val deq       = outQ.io.deq.fire
  val doReserve = accept && nextIsPackStart
  when(ext_start_i) {
    haveBeat := false.B; sub := 0.U; packIdx := 0.U; nextPack := 0.U; credit := Qdepth.U
  }.otherwise {
    when(accept) {
      inBeat   := ext_data_i.bits; haveBeat := true.B; sub := 0.U
      packIdx  := nextPack
      nextPack := Mux(nextPack === (pack - 1).U, 0.U, nextPack + 1.U)
    }.elsewhen(issuing) {
      when(lastSub) { haveBeat := false.B }.otherwise { sub := sub + 1.U }
    }
    when(doReserve =/= deq) { credit := Mux(doReserve, credit - 1.U, credit + 1.U) }
  }

  ext_busy_o := (credit =/= Qdepth.U) || haveBeat
}

class HasFp16ToInt8(
  in_elementWidth:  Int = 16,
  out_elementWidth: Int = 8,
  dataWidth:        Int = 512,
  computeLanes:     Int = 0,
  fpPipe:           Int = 1 // internal pipeline depth of the quantize-PE FP units (timing cut knob)
) extends HasDataPathExtension {
  require(
    in_elementWidth == 16 && out_elementWidth == 8,
    s"HasFp16ToInt8: only FP16(16)->INT8(8) supported, got $in_elementWidth->$out_elementWidth"
  )

  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "Fp16ToInt8", // -> READER_EXT_FP16TOINT8 (keep stable: never width-encode the name)
      userCsrNum = 1,            // inv_scale (FP32 bits)
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): Fp16ToInt8 =
    Module(
      new Fp16ToInt8(in_elementWidth, out_elementWidth, computeLanes, fpPipe) {
        override def desiredName = clusterName + namePostfix
      }
    )
}
