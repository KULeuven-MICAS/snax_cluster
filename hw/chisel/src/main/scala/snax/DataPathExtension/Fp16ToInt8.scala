// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._

/** Fp16ToInt8PE: symmetric per-tensor quantize of one FP16 lane to a signed int8.
  *
  *   q = sat_[-127,127]( round_rne( widen_fp32(x) * inv_scale ) )
  *
  * The mirror image of Int32ToFp16PE. FP16 transport in, FP32 internal (fpnew mixed-format
  * blackboxes), SInt(8) out. The fp32->int round uses the 1.5*2^23 magic-number add (no F2I unit),
  * the same trick as FpExp/FpSilu. The product is first clamped into [-128, 128] so the magic add
  * stays exact (|value| << 2^23) and the rounded integer lands in a tiny signed range; the result is
  * then saturated to the SYMMETRIC int8 range [-127, 127] (zero-point = 0).
  *
  * PIPELINING (timing): the widen->fmul->clamp->fadd->saturate chain is ~3 FP ops deep; when
  * `pipelined` is set it is cut into `latency` (=2) register stages (after the multiply and after the
  * magic add) so it fits the clock. `pipelined=false` keeps the original combinational PE (used by the
  * standalone PE tester for an exact same-cycle compare).
  */
class Fp16ToInt8PE(pipelined: Boolean = false, fpPipeParam: Int = 0) extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in        = Input(UInt(16.W))  // FP16
    val inv_scale = Input(UInt(32.W))  // FP32 (= 1 / quant_scale)
    val out       = Output(SInt(8.W))
  })

  import FpHelpers._

  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u
  val fpPipe = if (pipelined) fpPipeParam else 0 // internal FP-unit pipeline depth (cfg cut knob)

  val MAGIC = f32lit(12582912.0f) // 1.5 * 2^23 = 0x4B400000
  val HI    = f32lit(128.0f)      // pre-round clamp window (keeps the RNE exact, bounds iM)
  val LO    = f32lit(-128.0f)

  // ---- stage 0: widen fp16 -> fp32, then multiply by the FP32 inv_scale ----
  val xf     = widen(io.in, FP16, fpPipe)
  val scaled = sr(fmul(xf, io.inv_scale, fpPipe)) // reg0

  // ---- stage 1: clamp into [-128, 128] then round to nearest (ties to even) via the magic add ----
  val clamped = fp32max(fp32min(scaled, HI), LO)
  val rM      = sr(fadd(clamped, MAGIC, fpPipe))  // reg1
  val iM      = rM.asSInt - 0x4b400000.S  // |iM| <= 128 after the clamp

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
  /** Register-stage latency of the pipelined PE: the 2 sr() cuts (after the multiply and the magic add)
    * plus `fpPipe` internal registers in each of widen/fmul/fadd (= 2 + 3*fpPipe). 0 if not pipelined. */
  def pipeLatency(pipelined: Boolean, fpPipe: Int): Int = if (pipelined) 2 + 3 * fpPipe else 0
}

/** Fp16ToInt8: stream FP16 -> INT8 quantize extension. Packs `pack` input beats (e.g. 2 beats of 32
  * FP16 = 512b each) into 1 output beat (64 INT8 = 512b). One FP32 CSR carries inv_scale. Chains LAST in
  * the reader extension list so a vector op (StreamMap / StreamElementwise) computes fp16 and the cast
  * narrows it to int8 before it leaves the stream.
  *
  * TIME-MULTIPLEXED (area): only `computeLanes` quantize PEs are built and swept over
  * `subCycles` (= nPE/computeLanes) cycles per input beat, so a 512-bit beat needs `computeLanes` PEs
  * instead of one per element. computeLanes >= nPE ⇒ fully parallel (subCycles=1). The PEs are pipelined
  * (timing), so each input beat costs `subCycles + PE.latency` cycles; results retire into the output
  * accumulator and the packed beat is emitted once `pack` input beats are folded in.
  *
  * NOTE: the pack ratio requires an integer multiple of `pack` input beats (a lone trailing beat is
  * latched but never emitted). All current SIMD apps use row lengths that are a multiple of 64 FP16.
  */
class Fp16ToInt8(
  in_elementWidth:   Int = 16,
  out_elementWidth:  Int = 8,
  computeLanesParam: Int = 0,
  fpPipeParam:       Int = 1,
  pipelined:         Boolean = true
)(implicit extensionParam: DataPathExtensionParam)
    extends DataPathExtension {

  require(
    extensionParam.dataWidth % in_elementWidth == 0,
    s"Fp16ToInt8: dataWidth (${extensionParam.dataWidth}) must be a multiple of in_elementWidth ($in_elementWidth)"
  )
  require(
    in_elementWidth == 16 && out_elementWidth == 8,
    s"Fp16ToInt8: only FP16(16)->INT8(8) supported, got $in_elementWidth->$out_elementWidth"
  )

  val nPE   = extensionParam.dataWidth / in_elementWidth      // input elements per beat (e.g. 32)
  val pack  = in_elementWidth / out_elementWidth              // input beats per output beat (e.g. 2)
  val outElems = extensionParam.dataWidth / out_elementWidth  // int8 per output beat (e.g. 64)
  val computeLanes = if (computeLanesParam <= 0 || computeLanesParam > nPE) nPE else computeLanesParam
  require(nPE % computeLanes == 0, "Fp16ToInt8: nPE must be a multiple of computeLanes")
  val subCycles = nPE / computeLanes
  val fpPipe = if (pipelined) fpPipeParam else 0
  val Ppe    = Fp16ToInt8PE.pipeLatency(pipelined, fpPipe)

  val inv_scale = WireInit(ext_csr_i(0).asUInt)

  // ---- the `computeLanes` quantize PEs (pipelined) ----
  val inBeat   = Reg(UInt((nPE * in_elementWidth).W))
  val inLanes  = inBeat.asTypeOf(Vec(nPE, UInt(in_elementWidth.W)))
  val outRegs  = Reg(Vec(outElems, SInt(out_elementWidth.W)))
  val busy     = RegInit(false.B)
  val outValid = RegInit(false.B)
  val packCnt  = RegInit(0.U(log2Ceil(pack).W))
  val total    = subCycles - 1 + Ppe
  val step     = RegInit(0.U(log2Ceil(total + 1).max(1).W))

  // index of element (s*computeLanes + j) into the `nPE`-wide Vec, width-exact to silence W004
  def li(s: UInt, j: Int): UInt = (s * computeLanes.U + j.U)(log2Ceil(nPE) - 1, 0)

  val issuing  = busy && (step < subCycles.U)
  val subIssue = step
  val res = Wire(Vec(computeLanes, SInt(out_elementWidth.W)))
  for (j <- 0 until computeLanes) {
    val PE = Module(new Fp16ToInt8PE(pipelined, fpPipe) {
      override def desiredName = extensionParam.moduleName + "_fp16_to_int8_pe"
    })
    PE.io.in        := inLanes(li(subIssue, j))
    PE.io.inv_scale := inv_scale
    res(j)          := PE.io.out
  }

  val subRetire   = ShiftRegister(subIssue, Ppe)
  val retireValid = ShiftRegister(issuing, Ppe, false.B, true.B)
  when(busy && retireValid) {
    for (j <- 0 until computeLanes)
      outRegs(packCnt * nPE.U + li(subRetire, j)) := res(j)
  }

  ext_data_i.ready := !busy && !outValid

  when(ext_start_i) {
    busy := false.B; outValid := false.B; step := 0.U; packCnt := 0.U
  }.elsewhen(busy) {
    when(step === total.U) {
      busy := false.B; step := 0.U
      when(packCnt === (pack - 1).U) { outValid := true.B; packCnt := 0.U }
        .otherwise { packCnt := packCnt + 1.U }
    }.otherwise {
      step := step + 1.U
    }
  }.elsewhen(ext_data_i.fire) {
    inBeat := ext_data_i.bits; busy := true.B; step := 0.U
  }
  when(outValid && ext_data_o.ready && !ext_start_i) { outValid := false.B }

  ext_data_o.bits  := outRegs.asTypeOf(ext_data_o.bits)
  ext_data_o.valid := outValid
  ext_busy_o       := busy || outValid
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
