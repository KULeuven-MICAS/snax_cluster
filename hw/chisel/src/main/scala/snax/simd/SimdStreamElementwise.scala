package snax.simd

import chisel3._
import chisel3.util._

import fp_native._
import fp_unit._
import snax.DataPathExtension.DataPathExtensionParam
import snax.DataPathExtension.FpHelpers
import snax.DataPathExtension.HasDataPathExtension
import snax.DataPathExtension.OpSpec

/** SIMD core elementwise combine, `out = B op x` on every lane, op = MUL | ADD, one beat per cycle.
  *
  * One beat is operand B and every other beat is an x combined against it. Which beat is B is set by the CSRs:
  *
  *   - csr(0) = 2: PAIRS. The stream alternates B, x, B, x, ... (the AGU interleaves the two operand regions), and each
  *     pair emits one beat.
  *   - csr(1)[8] STICKY-B, with csr(0) = 1: the task's FIRST beat is B and every later beat combines against it -- a
  *     broadcast operand read once instead of replicated per data beat. N + 1 beats in, N out.
  *   - csr(0) = 1 without sticky-B: every beat passes through unchanged.
  *
  * csr(1)[7:0] = op (0 MUL, 1 ADD).
  *
  * B is latched RAW from the input stream, not from the result of a previous beat, so no beat depends on another beat's
  * arithmetic: there is no accumulator, no recurrence and nothing to stall for. Both ops run through one fused
  * multiply-add per lane on two transport operands (exact product) with an FP32 addend -- MUL is B*x + 0 and ADD is x*1
  * + B -- rounded once to FP32 and then once to the transport type.
  *
  * `op` must be ["FMA_<PRECISION>"]: the one FMA build that provides both run-time ops. `dataWidth` is the SIMD beat
  * and is supplied by the SIMD generator.
  */
class HasSimdStreamElementwise(
  op:           Seq[String],
  elementWidth: Int,
  dataWidth:    Int,
  fpPipe:       Int = 1 // internal pipeline depth of the widen and the FMA, 0..2
) extends HasDataPathExtension {
  private val (ops, transport) = OpSpec.parse(op, Set("FMA"), "HasSimdStreamElementwise")
  OpSpec.checkWidth(elementWidth, transport, "HasSimdStreamElementwise")
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName   = "StreamElementwise", // the software name: SIMD_EXT_STREAMELEMENTWISE_*
      userCsrNum   = 2,
      dataWidth    = dataWidth,
      capabilities = OpSpec.runtimeOps(ops, Seq("MUL", "ADD"))
    )

  def instantiate(clusterName: String): SimdStreamElementwise =
    Module(new SimdStreamElementwise(op, elementWidth, fpPipe) {
      override def desiredName = clusterName + namePostfix
    })
}

class SimdStreamElementwise(op: Seq[String], elementWidth: Int, fpPipe: Int)(implicit
  extensionParam: DataPathExtensionParam
) extends SimdStreamOp {

  import FpHelpers._

  val (ops, transport) = OpSpec.parse(op, Set("FMA"), "SimdStreamElementwise")
  OpSpec.checkWidth(elementWidth, transport, "SimdStreamElementwise")
  require(extensionParam.dataWidth % elementWidth == 0, "SimdStreamElementwise: the beat must hold whole lanes")
  val lanes = extensionParam.dataWidth / elementWidth

  def OP_MUL = 0.U
  def OP_ADD = 1.U

  val operandCount = Mux(ext_csr_i(0) === 0.U, 1.U, ext_csr_i(0)(15, 0))
  val isAdd        = ext_csr_i(1)(7, 0) === OP_ADD
  val stickyB      = ext_csr_i(1)(8).asBool
  val pairs        = operandCount === 2.U && !stickyB
  val passThrough  = operandCount === 1.U && !stickyB

  // ---- which beat is operand B --------------------------------------------------------------------------
  val phase  = RegInit(false.B) // pairs: the next beat is the x of a pair
  val seeded = RegInit(false.B) // sticky-B: B has been latched for this task
  val isB    = Mux(stickyB, !seeded, pairs && !phase)
  val opB    = Reg(UInt(extensionParam.dataWidth.W))

  val out = Module(new SimdOutputStage(extensionParam.dataWidth, 1 + 2 * fpPipe))
  // A B beat produces nothing, so it needs no slot.
  ext_data_i.ready := (isB || out.io.credit =/= 0.U) && !ext_start_i
  val accept = ext_data_i.fire

  when(ext_start_i) {
    phase := false.B; seeded := false.B
  }.elsewhen(accept) {
    when(isB) { opB := ext_data_i.bits }
    when(stickyB) { seeded := true.B }
    when(pairs) { phase := !phase }
  }
  assert(!(accept && operandCount > 2.U), "SimdStreamElementwise: operandCount must be 1 or 2")

  // ---- one lane: input register | widen + align | FMA -------------------------------------------------------
  // B is read in the register stage: a later B beat can only overwrite it from the next cycle on.
  val ONE_T = ((((1 << (transport.expWidth - 1)) - 1) << transport.sigWidth)).U(transport.width.W)
  def lane(x: UInt, b: UInt): UInt = {
    val bW  = widen(b, transport, fpPipe)
    val xD  = ShiftRegister(x, fpPipe)
    val bD  = ShiftRegister(b, fpPipe)
    val fma = Module(new FpFma(transport, transport, FP32, fpPipe))
    fma.io.in_a := xD
    fma.io.in_b := Mux(isAdd, ONE_T, bD)
    fma.io.in_c := Mux(isAdd, bW, FP32_ZERO)
    val xW = ShiftRegister(widen(x, transport, fpPipe), fpPipe)
    narrow(Mux(passThrough, xW, fma.io.out), transport)
  }
  val latency = 1 + 2 * fpPipe

  val x  = RegNext(ext_data_i.bits).asTypeOf(Vec(lanes, UInt(elementWidth.W)))
  val bv = opB.asTypeOf(Vec(lanes, UInt(elementWidth.W)))

  out.io.clear     := ext_start_i
  out.io.reserve   := (accept                            && !isB).asUInt
  out.io.enq.valid := validPipe(accept && !isB, latency) && !ext_start_i
  out.io.enq.bits  := Cat((0 until lanes).map(i => lane(x(i), bv(i))).reverse)
  ext_data_o.valid := out.io.deq.valid
  ext_data_o.bits  := out.io.deq.bits
  out.io.deq.ready := ext_data_o.ready
  ext_busy_o       := !out.io.idle
}
