package snax.simd

import chisel3._
import chisel3.util._

import snax.DataPathExtension.DataPathExtensionParam
import snax.DataPathExtension.Fp16ToInt8PE
import snax.DataPathExtension.HasDataPathExtension

/** SIMD core FP16 -> INT8 quantiser, one input beat per cycle.
  *
  * Every lane computes q = sat[-127,127](rne(x * inv_scale)) (Fp16ToInt8PE). An INT8 beat holds two input beats, so the
  * output runs at half the input rate:
  *
  *   - PACK (default): input beats 2j and 2j+1 fill output beat j, low half first.
  *   - INTERLEAVE (csr(1)[16], build with `interleave = 1`): four input beats fill two output beats so that every
  *     4-byte atom is ONE lane across the four beats, out[q / (L/2)][4 (q % (L/2)) + b] = quant(in_b[q]) for L lanes.
  *     That is the GEMM's B-operand atom (four consecutive K of one N column) built from beats that each carry one K
  *     for every column.
  *   - TAIL (csr(1)[15:0] = period, build with `tailPassthrough = 1`): input beat `period` of every period + 1 is
  *     passed through unquantised -- the trailing result beat a StreamReduce TAP appends to a row stays FP16 while the
  *     row is narrowed. The period must be a multiple of the group (2, or 4 when interleaving).
  *
  * csr(0) = inv_scale (FP32 bits). `dataWidth` is the SIMD beat and is supplied by the SIMD generator.
  */
class HasSimdFp16ToInt8(
  dataWidth:       Int,
  fpPipe:          Int = 1, // internal pipeline depth of the quantise PE's FP units
  tailPassthrough: Int = 0,
  interleave:      Int = 0
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName   = "Fp16ToInt8", // the software name: SIMD_EXT_FP16TOINT8_*
      userCsrNum   = if (tailPassthrough != 0 || interleave != 0) 2 else 1,
      dataWidth    = dataWidth,
      capabilities = (if (tailPassthrough != 0) Seq("TAILPASSTHROUGH") else Nil) ++
        (if (interleave != 0) Seq("INTERLEAVE4") else Nil)
    )

  def instantiate(clusterName: String): SimdFp16ToInt8 =
    Module(new SimdFp16ToInt8(fpPipe, tailPassthrough != 0, interleave != 0) {
      override def desiredName = clusterName + namePostfix
    })
}

class SimdFp16ToInt8(fpPipe: Int, hasTail: Boolean, hasIlv: Boolean)(implicit extensionParam: DataPathExtensionParam)
    extends SimdStreamOp {

  val dw = extensionParam.dataWidth
  require(dw % 32 == 0, "SimdFp16ToInt8: the beat must hold an even number of FP16 lanes")
  val lanes    = dw / 16
  val outBytes = dw / 8 // INT8 per output beat = 2 lanes' worth of input beats

  val inv_scale = ext_csr_i(0)
  val ilv       = if (hasIlv) ext_csr_i(1)(16).asBool else false.B
  val period    = if (hasTail) ext_csr_i(1)(15, 0) else 0.U(16.W)

  // A GROUP is the input beats of one FIFO entry: two (one output beat) or, interleaved, four (two output beats).
  val ilvGroup    = 4
  val atomsPerOut = outBytes / ilvGroup // lanes per interleaved output beat
  val entryW      = if (hasIlv) 2 * dw + 1 else dw
  val groupLast   = Mux(ilv, (ilvGroup - 1).U, 1.U)

  // ---- group / tail bookkeeping at accept ------------------------------------------------------------------
  val slot      = RegInit(0.U(2.W))  // beat within the group
  val sinceTail = RegInit(0.U(16.W)) // data beats since the last tail
  val isTail    = (period =/= 0.U) && sinceTail === period
  val tailReg   = Reg(UInt(dw.W))    // the tail beat, waiting for the group ahead of it to drain
  val tailBusy  = RegInit(false.B)

  val latency = 1 + Fp16ToInt8PE.pipeLatency(true, fpPipe)
  val out     = Module(new SimdOutputStage(entryW, latency, beatsPerEntry = 2))
  val claims  = isTail || slot === 0.U
  // A tail waits only for the previous tail, which still owns tailReg.
  ext_data_i.ready := (!claims || out.io.credit =/= 0.U) && !(isTail && tailBusy) && !ext_start_i
  val accept = ext_data_i.fire

  when(ext_start_i) {
    slot := 0.U; sinceTail := 0.U
  }.elsewhen(accept) {
    when(isTail) {
      sinceTail := 0.U
    }.otherwise {
      sinceTail := sinceTail + 1.U
      slot      := Mux(slot === groupLast, 0.U, slot + 1.U)
    }
  }
  when(accept && isTail) { tailReg := ext_data_i.bits }
  assert(!(accept && isTail) || slot === 0.U, "SimdFp16ToInt8: the tail period must be a multiple of the group")

  // ---- quantise every lane -------------------------------------------------------------------------------
  val x          = RegNext(ext_data_i.bits).asTypeOf(Vec(lanes, UInt(16.W)))
  val q          = (0 until lanes).map { i =>
    val pe = Module(new Fp16ToInt8PE(true, fpPipe))
    pe.io.in        := x(i)
    pe.io.inv_scale := inv_scale
    pe.io.out
  }
  val dataRetire = validPipe(accept && !isTail, latency)
  val tailRetire = validPipe(accept && isTail, latency)
  val slotRetire = ShiftRegister(slot, latency)

  when(ext_start_i) { tailBusy := false.B }
    .elsewhen(accept && isTail) { tailBusy := true.B }
    .elsewhen(tailRetire) { tailBusy := false.B }

  // The group's bytes: beat b of the group lands at byte b*lanes + q (pack) or at atom q, byte b (interleave).
  // Earlier beats wait in `bytes`; the last one is merged in as it retires.
  val bytes = Reg(Vec(if (hasIlv) 2 * outBytes else outBytes, UInt(8.W)))
  val now   = WireInit(bytes)
  for {
    b <- 0 until ilvGroup
    i <- 0 until lanes
  } {
    val packAt = b * lanes + i
    val ilvAt  = (i / atomsPerOut) * outBytes + (i % atomsPerOut) * ilvGroup + b
    if (b < 2) when(dataRetire && slotRetire === b.U && !ilv) { now(packAt) := q(i).asUInt }
    if (hasIlv) when(dataRetire && slotRetire === b.U && ilv) { now(ilvAt) := q(i).asUInt }
  }
  when(dataRetire) { bytes := now }
  val groupDone = dataRetire && slotRetire === groupLast

  val flat = now.asUInt
  out.io.clear     := ext_start_i
  out.io.reserve   := (accept                   && claims).asUInt
  out.io.enq.valid := (groupDone || tailRetire) && !ext_start_i
  assert(!(groupDone && tailRetire), "SimdFp16ToInt8: a group and a tail in the same cycle")
  ext_busy_o       := !out.io.idle

  if (!hasIlv) {
    out.io.enq.bits  := Mux(tailRetire, tailReg, flat(dw - 1, 0))
    ext_data_o.valid := out.io.deq.valid
    ext_data_o.bits  := out.io.deq.bits
    out.io.deq.ready := ext_data_o.ready
  } else {
    // entry = {two beats?, beat 1, beat 0}; the serializer hands an interleaved entry out as two beats and pops it
    // (releasing its credit) with the second.
    out.io.enq.bits := Cat(ilv && !tailRetire, flat(2 * dw - 1, dw), Mux(tailRetire, tailReg, flat(dw - 1, 0)))
    val second = RegInit(false.B)
    val head   = out.io.deq.bits
    val two    = head(2 * dw)
    ext_data_o.valid := out.io.deq.valid
    ext_data_o.bits  := Mux(second, head(2 * dw - 1, dw), head(dw - 1, 0))
    out.io.deq.ready := ext_data_o.ready && (!two || second)
    when(ext_start_i) { second := false.B }
      .elsewhen(ext_data_o.fire) { second := two && !second }
  }
}
