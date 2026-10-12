package snax.simd

import chisel3._
import chisel3.util._

import snax.DataPathExtension.DataPathExtensionParam
import snax.DataPathExtension.FpActivation
import snax.DataPathExtension.FpHelpers
import snax.DataPathExtension.HasDataPathExtension
import snax.DataPathExtension.OpSpec

/** SIMD core pointwise map, `out = act(a * x + b)` on every lane of a beat, one beat per cycle.
  *
  * `x` is a transport-precision lane; `a` and `b` are FP32 CSR immediates; the math is FP32 and the result is narrowed
  * back to the transport type. `func` selects the activation at run time: LINEAR (the affine alone), EXP, SILU or
  * RSQRT. The three non-linear functions share ONE FpActivation per lane (they are mutually exclusive at run time).
  *
  * CSRs: csr(0) = a (FP32 bits), csr(1) = b (FP32 bits), csr(2)[1:0] = func (0 LINEAR, 1 EXP, 2 SILU, 3 RSQRT).
  *
  * `func` lists the functions to build ("<FUNC>_<PRECISION>"); selecting one that was not built returns the LINEAR
  * result, so the generated header publishes the list as `SIMD_EXT_STREAMMAP_HAS_<FUNC>`. `dataWidth` is the SIMD beat
  * and is supplied by the SIMD generator, not the cfg.
  */
class HasSimdStreamMap(
  func:         Seq[String],
  elementWidth: Int,
  dataWidth:    Int,
  fpPipe:       Int = 1 // internal pipeline depth of the affine FMA and the narrow, 0..2
) extends HasDataPathExtension {
  private val (funcNames, transport) = OpSpec.parse(func, SimdStreamMap.funcSet, "HasSimdStreamMap")
  OpSpec.checkWidth(elementWidth, transport, "HasSimdStreamMap")
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName   = "StreamMap", // the software name: SIMD_EXT_STREAMMAP_*
      userCsrNum   = 3,
      dataWidth    = dataWidth,
      capabilities = funcNames
    )

  def instantiate(clusterName: String): SimdStreamMap =
    Module(new SimdStreamMap(func, elementWidth, fpPipe) {
      override def desiredName = clusterName + namePostfix
    })
}

object SimdStreamMap {
  val funcSet = Set("LINEAR", "EXP", "SILU", "RSQRT")
}

class SimdStreamMap(func: Seq[String], elementWidth: Int, fpPipe: Int)(implicit
  extensionParam: DataPathExtensionParam
) extends SimdStreamOp {

  import FpHelpers._

  val (funcs, transport) = OpSpec.parse(func, SimdStreamMap.funcSet, "SimdStreamMap")
  OpSpec.checkWidth(elementWidth, transport, "SimdStreamMap")
  require(extensionParam.dataWidth % elementWidth == 0, "SimdStreamMap: the beat must hold whole lanes")
  val lanes = extensionParam.dataWidth / elementWidth

  val hasExp    = funcs.contains("EXP")
  val hasSilu   = funcs.contains("SILU")
  val hasRsqrt  = funcs.contains("RSQRT")
  val hasLinear = funcs.contains("LINEAR")
  val hasAct    = hasExp || hasSilu || hasRsqrt

  // FpActivation ROM depths, the smallest that keep each function within 1 FP16 ULP:
  //   exp:   2^(i/N) with linear interpolation, 32 nodes.
  //   silu:  sigmoid(-|x|) over [0,16] reflected for x > 0; the tail near x = -10.8 needs 256 nodes.
  //   rsqrt: a table of the significand over [1,4), 64 nodes (the exponent is exact).
  val expLutN = 32
  val siluN   = 256
  val rsqN    = 64

  val a   = ext_csr_i(0)
  val b   = ext_csr_i(1)
  val act = ext_csr_i(2)(1, 0)

  // ---- one lane: affine | activation | narrow -------------------------------------------------------------
  // The affine feeds x RAW into a mixed FMA (FP32 * transport + FP32): x is exact in FP32, so this equals widening
  // first, with a 24 x 11 multiplier at FP16. Each stage ends in a register, and LINEAR is delayed by the
  // activation's latency so the run-time func select stays aligned.
  val actLat    = if (hasAct) FpActivation.PipeLatency else 0
  val affineLat = fpPipe + 1
  val narrowLat = fpPipe + 1
  def lane(x: UInt): UInt = {
    val t      = RegNext(ffmaT(a, x, b, transport, fpPipe))
    val actOut = if (hasAct) {
      val m = Module(new FpActivation(true, hasExp, hasSilu, expLutN, siluN, hasRsqrt, rsqN))
      m.io.in := t; m.io.func := act; m.io.out
    } else t
    val tD     = ShiftRegister(t, actLat)
    val selAct = (if (hasExp) act === FpActivation.EXP.U else false.B) ||
      (if (hasSilu) act === FpActivation.SILU.U else false.B) ||
      (if (hasRsqrt) act === FpActivation.RSQRT.U else false.B)
    val r      = if (!hasAct) tD else if (!hasLinear) actOut else Mux(selAct, actOut, tD)
    RegNext(narrow(r, transport, fpPipe))
  }

  // input register, then the lane pipeline
  val latency = 1 + affineLat + actLat + narrowLat

  val out = Module(new SimdOutputStage(extensionParam.dataWidth, latency))

  val accept = ext_data_i.fire
  val x      = RegNext(ext_data_i.bits).asTypeOf(Vec(lanes, UInt(elementWidth.W)))

  out.io.clear     := ext_start_i
  out.io.reserve   := accept.asUInt
  out.io.enq.valid := validPipe(accept, latency) && !ext_start_i
  out.io.enq.bits  := Cat(x.map(lane).reverse)
  ext_data_i.ready := out.io.credit =/= 0.U      && !ext_start_i
  ext_data_o.valid := out.io.deq.valid
  ext_data_o.bits  := out.io.deq.bits
  out.io.deq.ready := ext_data_o.ready
  ext_busy_o       := !out.io.idle
}
