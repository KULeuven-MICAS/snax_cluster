package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** StreamScalar: applies a selectable scalar op to the row-scalar of a stream. It reads the row-scalar (in
  * the configured transport precision) from the low lane of the incoming beat, computes the op in FP32,
  * narrows back to the transport type and splats the result across the output beat (1 beat in / 1 beat out,
  * combinational). The Stream-family sibling of StreamMap / StreamReduce, for the scalar a reduction produces.
  *
  * `op` is the LIST of supported scalar ops (a subset of {RECIP, RSQRT} — 1/x and 1/√x via the shared
  * Newton-Raphson `FpRecipRsqrt` primitive); only the listed path(s) are built. When more than one is
  * listed the runtime CSR(0) selects (userCsrNum=1); when a single op is listed it is fixed and the CSR
  * is dropped (userCsrNum=0). `op` is REQUIRED (no default) — the config must spell it out, e.g.
  * {op:["RECIP_FP16","RSQRT_FP16"]}.
  *
  * HARDWARE UNIT COUNT — there is exactly ONE FpRecipRsqrt core regardless of the lane count, NOT one per
  * lane (e.g. 1, not 32, for a 512-bit FP16 input). StreamScalar operates on a single ROW-SCALAR (the value
  * a reduction produces — e.g. Σexp or mean(x²) — which arrives splatted across all lanes): it reads ONLY
  * the low lane (`ext_data_i.bits(width-1,0)`), computes once, and broadcasts the result to all `lanes`
  * output lanes (`Cat(Seq.fill(lanes)(...))`); the other input lanes are ignored. A per-lane DivSqrt would
  * be 16–32× the area for zero benefit. With op=["RECIP","RSQRT"] the one core builds both
  * NR chains + a select mux; a single-op list builds only that chain. Contrast StreamMap/StreamReduce: those
  * are genuinely per-lane (distinct value per lane) and so time-multiplex `computeLanes` units — StreamScalar
  * needs no time-mux because there is only one value.
  *
  * NUMERICAL EXAMPLE — RSQRT, FP16, 32 lanes (1 beat in / 1 beat out, combinational — NO time-mux).
  * ONE shared FpRecipRsqrt reads the low-lane scalar and BROADCASTS its result to all 32 output lanes —
  * the inverse of StreamMap/StreamReduce's per-lane units, so there is nothing reused across cycles:
  *
  *   CC  phase  datapath
  *   --  -----  --------------------------------------------------------------------------
  *    0  in     scalar = bits[15:0] = 4.0 (FP16) → widen → 4.0 (FP32)
  *              → FpRecipRsqrt(RSQRT) = 1/√4.0 = 0.5 → narrow → 0.5 (FP16)
  *    1  out    ext_data_o = [0.5, 0.5, …, 0.5]   (the scalar splatted across all 32 lanes)
  * The host's -||> pipeline cut gives the 1-cycle in→out latency; the recip/rsqrt math is combinational.
  */
class HasStreamScalar(
  op:        Seq[String], // each entry "<OP>_<PRECISION>", e.g. "RSQRT_FP16"
  dataWidth: Int = 512
) extends HasDataPathExtension {
  private val (ops, _) = OpSpec.parse(op, Set("RECIP", "RSQRT"), "HasStreamScalar")
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "StreamScalar",
      userCsrNum = if (ops.size > 1) 1 else 0, // drop the op-select CSR when fixed
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): StreamScalar =
    Module(new StreamScalar(op) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamScalar(op: Seq[String] = Seq("RECIP_FP16", "RSQRT_FP16"))(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {
  import FpHelpers._
  // transport (element) precision from the op-set; the recip/rsqrt math stays FP32
  val (ops, transport) = OpSpec.parse(op, Set("RECIP", "RSQRT"), "StreamScalar")
  val lanes = extensionParam.dataWidth / transport.width
  val both  = ops.size > 1

  val scalarIn = ext_data_i.bits(transport.width - 1, 0)
  val core     = Module(new FpRecipRsqrt(ops))
  core.io.in   := widen(scalarIn, transport)
  // runtime op-select only when both are built; otherwise fixed (and CSR(0) does not exist)
  core.io.mode := (if (both) ext_csr_i(0)(0) else if (ops.contains("RSQRT")) 1.U else 0.U)
  val outEl = narrow(core.io.out, transport)

  ext_data_o.bits  := Cat(Seq.fill(lanes)(outEl))
  ext_data_o.valid := ext_data_i.valid
  ext_data_i.ready := ext_data_o.ready
  ext_busy_o       := false.B
}
