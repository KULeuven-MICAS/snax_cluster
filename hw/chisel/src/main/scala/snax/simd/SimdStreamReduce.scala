package snax.simd

import chisel3._
import chisel3.util._

import fp_native._
import fp_unit._
import snax.DataPathExtension.DataPathExtensionParam
import snax.DataPathExtension.FpHelpers
import snax.DataPathExtension.HasDataPathExtension
import snax.DataPathExtension.OpSpec

/** SIMD core reduction of a streamed row, one beat per cycle.
  *
  * A row is `operandCount` beats. Every lane accumulates its own column ACROSS the row's beats (op = MAX, ADD or SUMSQ,
  * FP32 inside); at the row's end the per-lane results are either emitted as one beat (LANEWISE) or folded ACROSS the
  * lanes into one scalar that is splatted over the output beat. Rows follow each other with no bubble.
  *
  * CSRs: csr(0) = operandCount (beats per row, 0 = 1). csr(1): [7:0] op (0 MAX, 1 ADD, 2 SUMSQ); [8] TAP -- pass every
  * input beat through and emit the row's result after it (N + 1 beats out); [9] FP32OUT -- splat the scalar as FP32
  * instead of narrowing it; [10] LANEWISE -- emit the per-lane results, skip the fold across lanes.
  *
  * THE RECURRENCE. Beat k of a row reads the partial that beat k - 1 of the same lane wrote, and the accumulate takes
  * `accLat` cycles from that read until its result can be read again. At one beat per cycle each lane therefore keeps
  * `accLat` partials and beat k accumulates into partial k mod accLat: the same partial is next read exactly when its
  * previous result has landed. The first `accLat` beats of a row seed their partial instead of accumulating, and at the
  * row's end the partials are folded together -- a short row (fewer beats than partials) folds only the ones it wrote.
  * So the partial count is not a knob: it is the FMA latency.
  *
  * Pipeline: input register | accumulate (2 fpPipe) | fold the partials (ceil(log2 accLat) treePipe) | fold the lanes
  * (log2 lanes treePipe, skipped by LANEWISE). Both folds are balanced pairwise trees with one FP32 node per pair.
  *
  * `op` lists the reductions to build: "FMA_<P>" provides ADD and SUMSQ (one FMA: x*1 + acc and x*x + acc, exact
  * product, one rounding), "MAX_<P>" the compare. `dataWidth` is the SIMD beat and is supplied by the SIMD generator.
  */
class HasSimdStreamReduce(
  op:           Seq[String],
  elementWidth: Int,
  dataWidth:    Int,
  fpPipe:       Int = 1, // internal pipeline depth of the accumulate, 0..2 (sets the partial count)
  treePipe:     Int = 1  // internal pipeline depth of each fold node, 0..2
) extends HasDataPathExtension {
  private val (opNames, transport) = OpSpec.parse(op, SimdStreamReduce.opSet, "HasSimdStreamReduce")
  OpSpec.checkWidth(elementWidth, transport, "HasSimdStreamReduce")
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName   = "StreamReduce", // the software name: SIMD_EXT_STREAMREDUCE_*
      userCsrNum   = 2,
      dataWidth    = dataWidth,
      capabilities = OpSpec.runtimeOps(opNames, Seq("ADD", "SUMSQ"))
    )

  def instantiate(clusterName: String): SimdStreamReduce =
    Module(new SimdStreamReduce(op, elementWidth, fpPipe, treePipe) {
      override def desiredName = clusterName + namePostfix
    })
}

object SimdStreamReduce {
  val opSet = Set("FMA", "MAX")
}

class SimdStreamReduce(op: Seq[String], elementWidth: Int, fpPipe: Int, treePipe: Int)(implicit
  extensionParam: DataPathExtensionParam
) extends SimdStreamOp {

  import FpHelpers._
  import SimdStreamOp.{treeLevels, treeReduce}

  val (ops, transport) = OpSpec.parse(op, SimdStreamReduce.opSet, "SimdStreamReduce")
  OpSpec.checkWidth(elementWidth, transport, "SimdStreamReduce")
  require(extensionParam.dataWidth % elementWidth == 0, "SimdStreamReduce: the beat must hold whole lanes")
  require(fpPipe >= 0 && fpPipe <= 2 && treePipe >= 0 && treePipe <= 2, "SimdStreamReduce: pipe depths are 0..2")
  val lanes = extensionParam.dataWidth / elementWidth

  private val runtimeOps = OpSpec.runtimeOps(ops, Seq("ADD", "SUMSQ"))
  val hasMax             = runtimeOps.contains("MAX")
  val hasFma             = ops.contains("FMA")
  val bothKind           = hasMax && hasFma

  def OP_MAX   = 0.U
  def OP_SUMSQ = 2.U

  val operandCount = Mux(ext_csr_i(0) === 0.U, 1.U(16.W), ext_csr_i(0)(15, 0))
  val opField      = ext_csr_i(1)
  val isMax        = if (bothKind) opField(7, 0) === OP_MAX else hasMax.B
  val isSumsq      = opField(7, 0) === OP_SUMSQ
  val tap          = opField(8).asBool
  val fp32out      = opField(9).asBool
  val lanewise     = opField(10).asBool

  val FP32_NEG_INF = "hFF800000".U(32.W)
  val ONE_T        = ((((1 << (transport.expWidth - 1)) - 1) << transport.sigWidth)).U(transport.width.W)

  // ---- latencies -------------------------------------------------------------------------------------------
  val accLat      = 2 * fpPipe + 1            // partial read -> its result readable again
  val partials    = accLat
  val accOutLat   = 1 + 2 * fpPipe            // accept -> accumulate result
  val partFoldLat = treeLevels(partials) * treePipe
  val laneFoldLat = treeLevels(lanes) * treePipe
  val lanewiseLat = accOutLat + partFoldLat   // accept of a row's last beat -> its LANEWISE beat
  val scalarLat   = lanewiseLat + laneFoldLat // ... -> its scalar beat

  // ---- row bookkeeping at accept --------------------------------------------------------------------------
  val beatIdx  = RegInit(0.U(16.W))                        // beat of the row being accepted
  val part     = RegInit(0.U(log2Ceil(partials).max(1).W)) // = beatIdx mod partials
  val rowStart = beatIdx === 0.U
  val rowLast  = beatIdx === (operandCount - 1.U)
  val seeds    = beatIdx < partials.U                      // first write of its partial in this row

  // TAP: a row's passthrough beats leave at accept and its result `scalarLat` later, so the next row's first beat
  // waits until the result is out -- otherwise its passthrough would overtake it.
  val tapDraining = RegInit(false.B)
  val out         = Module(new SimdOutputStage(extensionParam.dataWidth, scalarLat, maxReserve = 2))
  val claim       = Mux(rowStart, 1.U, 0.U) +& Mux(tap, 1.U, 0.U)
  ext_data_i.ready := out.io.credit >= claim && !(tap && rowStart && tapDraining) && !ext_start_i
  val accept = ext_data_i.fire

  when(ext_start_i) {
    beatIdx := 0.U; part := 0.U
  }.elsewhen(accept) {
    beatIdx := Mux(rowLast, 0.U, beatIdx + 1.U)
    part    := Mux(rowLast || part === (partials - 1).U, 0.U, part + 1.U)
  }

  // ---- accumulate ------------------------------------------------------------------------------------------
  val valid1 = RegNext(accept && !ext_start_i, false.B)
  val x      = RegNext(ext_data_i.bits).asTypeOf(Vec(lanes, UInt(elementWidth.W)))
  val part1  = RegNext(part)
  val seeds1 = RegNext(seeds)

  val regs = Reg(Vec(partials, Vec(lanes, UInt(32.W))))

  // one lane: acc' = op(prev, x), read at stage 1, result 2 fpPipe later
  def accumulate(xi: UInt, prev: UInt): UInt = {
    val fmaPipe = scala.math.min(2, 2 * fpPipe)
    val sum     =
      if (hasFma) {
        val m = Module(new FpFma(transport, transport, FP32, fmaPipe))
        m.io.in_a := xi
        m.io.in_b := Mux(isSumsq, xi, ONE_T)
        m.io.in_c := Mux(seeds1, FP32_ZERO, prev)
        ShiftRegister(m.io.out, 2 * fpPipe - fmaPipe)
      } else FP32_ZERO
    val max     =
      if (hasMax) {
        val w = widen(xi, transport, fpPipe)
        ShiftRegister(Mux(ShiftRegister(seeds1, fpPipe), w, fp32max(w, ShiftRegister(prev, fpPipe))), fpPipe)
      } else FP32_ZERO
    if (bothKind) Mux(isMax, max, sum) else if (hasMax) max else sum
  }
  val acc = VecInit((0 until lanes).map(i => accumulate(x(i), regs(part1)(i))))

  val retire     = ShiftRegister(valid1, 2 * fpPipe, false.B, true.B)
  val partRetire = ShiftRegister(part1, 2 * fpPipe)
  when(retire) { regs(partRetire) := acc }

  // ---- the row's end -----------------------------------------------------------------------------------------
  // At the retire of a row's last beat every partial holds its final value, except the one being written this
  // cycle, which is taken from `acc`. A partial the row never wrote holds an older row's value and folds in as the
  // op's identity instead.
  val rowRetire = validPipe(accept && rowLast, accOutLat)
  val written   = Mux(operandCount < partials.U, operandCount, partials.U)
  def node(a: UInt, b: UInt): UInt = {
    val s  = if (hasFma) {
      val m = Module(new FpAdd(FP32, FP32, FP32, treePipe)); m.io.in_a := a; m.io.in_b := b; m.io.out
    } else FP32_ZERO
    val mx = if (hasMax) ShiftRegister(fp32max(a, b), treePipe) else FP32_ZERO
    if (bothKind) Mux(isMax, mx, s) else if (hasMax) mx else s
  }
  val identity = if (bothKind) Mux(isMax, FP32_NEG_INF, FP32_ZERO) else if (hasMax) FP32_NEG_INF else FP32_ZERO
  val perLane = (0 until lanes).map { i =>
    val ps = (0 until partials).map { p =>
      val v = Mux(partRetire === p.U, acc(i), regs(p)(i))
      if (p == 0) v else Mux(p.U < written, v, identity)
    }
    treeReduce(ps, treePipe)(node)
  }
  val scalar  = treeReduce(perLane, treePipe)(node)

  val lanewiseDone = validPipe(rowRetire, partFoldLat)
  val scalarDone   = validPipe(lanewiseDone, laneFoldLat)
  val resultValid  = Mux(lanewise, lanewiseDone, scalarDone)

  val lanewiseBeat = Cat(perLane.map(narrow(_, transport)).reverse)
  val narrowBeat   = Cat(Seq.fill(lanes)(narrow(scalar, transport)))
  val scalarBeat   =
    if (transport.width >= 32) narrowBeat
    else Mux(fp32out, Cat(Seq.fill(extensionParam.dataWidth / 32)(scalar)), narrowBeat)

  when(ext_start_i) {
    tapDraining := false.B
  }.elsewhen(tap && accept && rowLast) {
    tapDraining := true.B
  }.elsewhen(resultValid) {
    tapDraining := false.B
  }

  // ---- output: the passthrough at accept (TAP), the row's result when it is ready ----------------------------
  val pass = tap && accept
  assert(!(pass && resultValid), "SimdStreamReduce: a passthrough and a row result in the same cycle")
  out.io.clear := ext_start_i
  out.io.reserve   := Mux(accept, claim, 0.U)
  out.io.enq.valid := (pass || resultValid) && !ext_start_i
  out.io.enq.bits  := Mux(pass, ext_data_i.bits, Mux(lanewise, lanewiseBeat, scalarBeat))
  ext_data_o.valid := out.io.deq.valid
  ext_data_o.bits  := out.io.deq.bits
  out.io.deq.ready := ext_data_o.ready
  ext_busy_o       := !out.io.idle
}
