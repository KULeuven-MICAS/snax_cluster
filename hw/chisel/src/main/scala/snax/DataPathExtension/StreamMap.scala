package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** StreamMap: per-lane pointwise map act(a*x + b) over a streamed row.
  *
  *   - x is a transport-precision lane (the element type — FP16/BF16/FP8/FP32 — is set by the op-set);
  *     a, b are FP32 CSR-immediates; the math is FP32; the output is narrowed back to the transport type.
  *   - func in {LINEAR, EXP} selected at runtime (CSR), so one instance serves softmax pass-2
  *     (a=1, b=-max, EXP) and pass-4 (a=inv_sum, b=0, LINEAR — identity activation, just a*x+b).
  *     EXP routes through FpExp.
  *
  * TIME-MULTIPLEXED (area-reduction point): only `computeLanes` compute units are built and reused over
  * `subCycles` (= lanes/computeLanes) cycles per beat, so a 512-bit beat (e.g. 32 FP16 lanes) needs only
  * `computeLanes` FpExp instead of one per lane. A small FSM latches the input beat, sweeps the units across
  * the sub-groups, then emits the assembled beat. (computeLanes >= lanes ⇒ fully parallel / combinational.)
  * Throughput drops ~`subCycles`x — acceptable since the datapath is far from the orchestration critical path.
  *
  * CSR layout: csr(0)=a (FP32 bits), csr(1)=b (FP32 bits), csr(2)= bits[0]=func (0=LINEAR,1=EXP),
  * bits[9:8]=operandMode (0=a as-is, 1=a:=1/a, 2=a:=1/√a). operandMode transforms the scalar `a` ONCE
  * (a single shared FpRecipRsqrt, not per-lane) so the normalize map can fold the reciprocal in:
  * softmax x/Σ = StreamMap(a=Σ, operandMode=1); rmsnorm x·rms = StreamMap(a=meanSq, operandMode=2).
  *
  * Configurable from the hjson: `computeLanes` (time-mux width), `func` (the LIST of supported funcs, a
  * subset of {LINEAR, EXP}), and `elementWidth` (the transport element width in bits — must match the
  * func-set precision: FP16/BF16⇒16, FP8⇒8, FP32⇒32). FpExp is built only if "EXP" is listed; the act
  * CSR bit selects among the listed funcs at runtime only when more than one is listed. All are REQUIRED
  * (no defaults) — the config must spell them out explicitly, e.g.
  * {elementWidth:16, computeLanes:8, func:["LINEAR_FP16","EXP_FP16"]}.
  *
  * HARDWARE UNIT COUNT — for a 512-bit input only `computeLanes` physical compute units exist, NOT one per
  * lane (lanes = dataWidth/elementWidth, e.g. 32 at FP16 or 64 at FP8): each does one FFMA (a·x+b) and, when
  * "EXP" is listed, one FpExp. They are time-multiplexed over `subCycles` (= lanes/computeLanes) cycles to
  * cover all lanes, so func=["EXP"] builds `computeLanes` FpExp, not one per lane. (The shared operandMode
  * FpRecipRsqrt is also ONE unit, not per-lane — it transforms the scalar `a` once.) computeLanes = lanes ⇒
  * subCycles=1 ⇒ one unit per lane in parallel (the combinational path).
  *
  * NUMERICAL EXAMPLE — affine map, FP16 transport, 32 lanes, computeLanes=8 ⇒ subCycles=4.
  * CSR a=2.0, b=1.0, func=LINEAR ⇒ out[i]=2·x[i]+1 (same op every lane). The 8 physical compute units U0..U7
  * are REUSED across the 4 sub-cycles; on sub-cycle s, unit Uj processes logical lane (s·8+j), so one
  * 32-lane beat takes 4 compute cycles (+1 to emit):
  *
  *   CC  phase    U0   U1   U2   U3   U4   U5   U6   U7   → outBeat lanes written
  *   --  -------  ------------------------------------    ----------------------
  *    0  accept   latch the 512-bit input x[0..31]        —
  *    1  sub=0    x0   x1   x2   x3   x4   x5   x6   x7    [0..7]
  *    2  sub=1    x8   x9  x10  x11  x12  x13  x14  x15    [8..15]
  *    3  sub=2   x16  x17  x18  x19  x20  x21  x22  x23    [16..23]
  *    4  sub=3   x24  x25  x26  x27  x28  x29  x30  x31    [24..31]   (last sub)
  *    5  emit    drive ext_data_o = [2·x0+1, …, 2·x31+1]
  * e.g. x0=3, x1=5 ⇒ out0=7, out1=11.  Lane reuse: unit Uj owns logical lanes {j, 8+j, 16+j, 24+j}. For
  * func=EXP each Uj additionally holds an FpExp (8 instead of 32). computeLanes=32 ⇒ subCycles=1 ⇒ the
  * combinational path: all 32 lanes in one cycle, no sub-cycling.
  */
class HasStreamMap(
  computeLanes: Int,
  func:         Seq[String], // each entry "<FUNC>_<PRECISION>", e.g. "EXP_FP16"
  elementWidth: Int,         // transport element width (bits); must match the func-set precision
  dataWidth:    Int = 512
) extends HasDataPathExtension {
  require(computeLanes > 0, "HasStreamMap: computeLanes must be > 0")
  private val (_, transport) = OpSpec.parse(func, Set("LINEAR", "EXP"), "HasStreamMap") // validate func names + precision
  OpSpec.checkWidth(elementWidth, transport, "HasStreamMap") // explicit width must match the precision tag
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "StreamMap",
      userCsrNum = 3,
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): StreamMap =
    Module(new StreamMap(computeLanes, func, elementWidth) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamMap(
  computeLanesParam: Int = 8,
  func:              Seq[String] = Seq("LINEAR_FP16", "EXP_FP16"),
  elementWidth:      Int = 16
)(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers._

  // transport (element) precision comes from the func-set; a/b and the per-lane math stay FP32
  val (funcs, transport) = OpSpec.parse(func, Set("LINEAR", "EXP"), "StreamMap")
  OpSpec.checkWidth(elementWidth, transport, "StreamMap") // config width must match the func-set precision
  val lanes = extensionParam.dataWidth / elementWidth
  val computeLanes = if (computeLanesParam > lanes) lanes else computeLanesParam // time-mux width
  require(lanes % computeLanes == 0, "StreamMap: lanes must be a multiple of computeLanes")
  val subCycles    = lanes / computeLanes

  val hasExp    = funcs.contains("EXP")
  val hasLinear = funcs.contains("LINEAR")

  def ACT_EXP = 1.U

  val a        = ext_csr_i(0)
  val b        = ext_csr_i(1)
  val actField = ext_csr_i(2)
  val act         = actField(0)    // func: 0=LINEAR, 1=EXP
  val operandMode = actField(9, 8) // 0=a as-is, 1=1/a (recip), 2=1/sqrt(a) (rsqrt)

  // transform the scalar operand a ONCE via a single shared scalar unit (not per-lane)
  val recipUnit = Module(new FpRecipRsqrt)
  recipUnit.io.in   := a
  recipUnit.io.mode := operandMode(1) // operandMode 1 -> RECIP (mode 0), 2 -> RSQRT (mode 1)
  val aEff = Mux(operandMode === 0.U, a, recipUnit.io.out)

  // one compute lane: aEff*x + b (FP32), optional EXP (built only if listed), narrow back to transport
  def computeLane(laneIn: UInt): UInt = {
    val t = ffma(aEff, widen(laneIn, transport), b)
    val r =
      if (hasExp) {
        val e = Module(new FpExp)
        e.io.in := t
        if (hasLinear) Mux(act === ACT_EXP, e.io.out, t) else e.io.out
      } else t
    narrow(r, transport)
  }

  if (subCycles == 1) {
    // fully parallel: combinational 1 beat in / 1 beat out
    val in_lanes  = ext_data_i.bits.asTypeOf(Vec(lanes, UInt(elementWidth.W)))
    val out_lanes = VecInit((0 until lanes).map(i => computeLane(in_lanes(i))))
    ext_data_o.bits  := Cat(out_lanes.reverse)
    ext_data_o.valid := ext_data_i.valid
    ext_data_i.ready := ext_data_o.ready
    ext_busy_o       := false.B
  } else {
    // time-multiplexed: `computeLanes` units sweep the beat over `subCycles` cycles
    val inBeat   = Reg(Vec(lanes, UInt(elementWidth.W)))
    val outBeat  = Reg(Vec(lanes, UInt(elementWidth.W)))
    val sub      = RegInit(0.U(log2Ceil(subCycles).W))
    val busy     = RegInit(false.B)
    val outValid = RegInit(false.B)

    // index of lane (sub*computeLanes + j) into the `lanes`-wide Vec, width-exact to silence W004
    def li(j: Int): UInt = (sub * computeLanes.U + j.U)(log2Ceil(lanes) - 1, 0)

    // combinational compute of the current sub-group's `computeLanes` lanes
    val res = Wire(Vec(computeLanes, UInt(elementWidth.W)))
    for (j <- 0 until computeLanes)
      res(j) := computeLane(inBeat(li(j)))

    ext_data_i.ready := !busy && !outValid

    when(ext_start_i) {
      busy := false.B; outValid := false.B; sub := 0.U
    }.elsewhen(busy) {
      for (j <- 0 until computeLanes) outBeat(li(j)) := res(j)
      when(sub === (subCycles - 1).U) {
        busy := false.B; outValid := true.B; sub := 0.U
      }.otherwise {
        sub := sub + 1.U
      }
    }.elsewhen(ext_data_i.fire) {
      inBeat := ext_data_i.bits.asTypeOf(Vec(lanes, UInt(elementWidth.W)))
      busy := true.B; sub := 0.U
    }
    when(outValid && ext_data_o.ready && !ext_start_i) { outValid := false.B }

    ext_data_o.bits  := Cat(outBeat.reverse)
    ext_data_o.valid := outValid
    ext_busy_o       := busy || outValid
  }
}
