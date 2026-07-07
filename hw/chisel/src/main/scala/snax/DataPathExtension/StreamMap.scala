package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** StreamMap: per-lane pointwise map act(a*x + b) over a streamed row.
  *
  *   - x is a transport-precision lane (the element type — FP16/BF16/FP8/FP32 — is set by the op-set);
  *     a, b are FP32 CSR-immediates; the math is FP32; the output is narrowed back to the transport type.
  *   - func in {LINEAR, EXP, SILU} selected at runtime (CSR), so one instance serves softmax pass-2
  *     (a=1, b=-max, EXP), pass-4 (a=inv_sum, b=0, LINEAR — identity activation, just a*x+b), and silu
  *     (a=1, b=0, SILU). EXP and SILU are computed by ONE merged FpActivation core (they are mutually
  *     exclusive at runtime; a func bit selects exp vs silu, sharing all its FP units — see FpActivation).
  *
  * TIME-MULTIPLEXED (area-reduction point): only `computeLanes` compute units are built and reused over
  * `subCycles` (= lanes/computeLanes) cycles per beat, so a 512-bit beat (e.g. 32 FP16 lanes) needs only
  * `computeLanes` activation cores instead of one per lane. (computeLanes >= lanes ⇒ fully parallel, one sub-cycle.)
  * The issue FSM STREAMS: it feeds one sub-group per cycle across back-to-back beats and reassembles the
  * results into whole output beats through a small skid FIFO, so the per-lane pipeline latency P is a
  * one-time fill rather than a per-beat bubble. Steady-state throughput = 1 beat / `subCycles` cycles
  * (1 beat/cycle at computeLanes == lanes), independent of P.
  *
  * CSR layout: csr(0)=a (FP32 bits), csr(1)=b (FP32 bits), csr(2)= bits[1:0]=func (0=LINEAR,1=EXP,2=SILU).
  * a, b are plain FP32 immediates; the normalize maps (softmax x/Σ, rmsnorm x·rms) pass the already-inverted
  * scalar in `a` as a LINEAR multiply — the host computes the 1/Σ resp. 1/√mean reciprocal/rsqrt.
  *
  * Configurable from the hjson: `computeLanes` (time-mux width), `func` (the LIST of supported funcs, a
  * subset of {LINEAR, EXP, SILU}), and `elementWidth` (the transport element width in bits — must match the
  * func-set precision: FP16/BF16⇒16, FP8⇒8, FP32⇒32). The FpActivation's exp / silu tables+path are built
  * only if "EXP"/"SILU" is listed; the act CSR field selects among the listed funcs at runtime when >1 listed.
  * All are REQUIRED (no defaults) — the config must spell them out explicitly, e.g.
  * {elementWidth:16, computeLanes:8, func:["LINEAR_FP16","EXP_FP16","SILU_FP16"]}.
  *
  * HARDWARE UNIT COUNT — for a 512-bit input only `computeLanes` physical compute units exist, NOT one per
  * lane (lanes = dataWidth/elementWidth, e.g. 32 at FP16 or 64 at FP8): each does one mixed FFMA (a·x+b) and,
  * when EXP/SILU is listed, one merged FpActivation. They are time-multiplexed over `subCycles`
  * (= lanes/computeLanes) cycles to cover all lanes, so a func-set builds `computeLanes` cores. computeLanes = lanes ⇒
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
  * func=EXP/SILU each Uj additionally holds an FpActivation (8 instead of 32). computeLanes=32 ⇒ subCycles=1 ⇒ the
  * combinational path: all 32 lanes in one cycle, no sub-cycling.
  */
class HasStreamMap(
  computeLanes: Int,
  func:         Seq[String], // each entry "<FUNC>_<PRECISION>", e.g. "EXP_FP16"
  elementWidth: Int,         // transport element width (bits); must match the func-set precision
  dataWidth:    Int = 512,
  fpPipe:       Int = 1      // internal pipeline depth of the affine (a*x+b) FP units (timing cut knob)
) extends HasDataPathExtension {
  require(computeLanes > 0, "HasStreamMap: computeLanes must be > 0")
  private val (_, transport) = OpSpec.parse(func, Set("LINEAR", "EXP", "SILU"), "HasStreamMap") // validate func names + precision
  OpSpec.checkWidth(elementWidth, transport, "HasStreamMap") // explicit width must match the precision tag
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "StreamMap",
      userCsrNum = 3,
      dataWidth  = dataWidth
    )

  def instantiate(clusterName: String): StreamMap =
    Module(new StreamMap(computeLanes, func, elementWidth, fpPipe) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamMap(
  computeLanesParam: Int = 8,
  func:              Seq[String] = Seq("LINEAR_FP16", "EXP_FP16"),
  elementWidth:      Int = 16,
  fpPipeParam:       Int = 1,
  pipelined:         Boolean = true
)(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers._

  // transport (element) precision comes from the func-set; a/b and the per-lane math stay FP32
  val (funcs, transport) = OpSpec.parse(func, Set("LINEAR", "EXP", "SILU"), "StreamMap")
  OpSpec.checkWidth(elementWidth, transport, "StreamMap") // config width must match the func-set precision
  val lanes = extensionParam.dataWidth / elementWidth
  val computeLanes = if (computeLanesParam > lanes) lanes else computeLanesParam // time-mux width
  require(lanes % computeLanes == 0, "StreamMap: lanes must be a multiple of computeLanes")
  val subCycles    = lanes / computeLanes

  val hasExp    = funcs.contains("EXP")
  val hasSilu   = funcs.contains("SILU")
  val hasLinear = funcs.contains("LINEAR")

  // FpActivation ROM depths (measured via a ULP-vs-depth sweep, 2026-07-08):
  //   exp: 2^(i/N) linear interp stays at 1 FP16 ULP down to lutN=16 (budget <=2) -> 32 keeps margin cheaply.
  //   silu: odd-symmetry tabulation g(m)=sigmoid(-|x|) (reflect via 1-g for x>0) over [0,16]; the deep tail
  //   (x~-10.8) needs 256 nodes for <=1 ULP (128 = 3 ULP with linear interp). Quadratic interp holds <=1 ULP
  //   at 128 but only trades 25% ROM for +1 FMA/lane + a pipeline stage (a wash), so it stays linear/256.
  val expLutN = 32
  val siluN   = 256

  def ACT_EXP  = 1.U
  def ACT_SILU = 2.U

  val a        = ext_csr_i(0)
  val b        = ext_csr_i(1)
  val actField = ext_csr_i(2)
  val act      = actField(1, 0) // func: 0=LINEAR, 1=EXP, 2=SILU

  // ---- pipeline depth (timing): the per-lane chain affine-ffma -> act -> narrow is cut into P register
  // stages so no single combinational path crosses a clock period; the FSM below drains it. ----
  // fpPipe = internal pipeline depth of the affine FP units (mixed ffma / narrow). The EXP/SILU activation
  // keeps its own fixed FpActivation.PipeLatency (its LUT chain has no numPipe knob).
  val fpPipe  = if (pipelined) fpPipeParam else 0
  val actLat  = if (pipelined && (hasExp || hasSilu)) FpActivation.PipeLatency else 0
  // affine feeds x RAW (no widen) into a mixed FMA; a fpPipe-deep input ShiftRegister keeps preLat = 2*fpPipe+1
  val preLat  = if (pipelined) 2 * fpPipe + 1 else 0 // sr(x, fpPipe) -> mixed ffma(fpPipe) -> register
  val postLat = if (pipelined) fpPipe + 1 else 0     // narrow(fpPipe) -> register
  val P       = preLat + actLat + postLat
  private def sr[T <: Data](u: T): T = if (pipelined) RegNext(u) else u

  // one compute lane: a*x + b (FP32), then the CSR-selected activation (built only if listed), narrow back
  // to transport. LINEAR is the affine result itself; EXP/SILU go through the merged FpActivation core.
  // When `pipelined`, all three func branches share `actLat` so the runtime func-mux stays aligned.
  def computeLane(laneIn: UInt): UInt = {
    // affine a*x+b: feed x RAW (transport precision) into a mixed FMA (FP32*FP16+FP32) instead of widening
    // it to FP32 first — bit-identical (x is exact in FP32) but a 24x11 multiplier and no widen. The
    // ShiftRegister keeps x's arrival (and thus preLat = 2*fpPipe+1) unchanged, so the streaming FSM is intact.
    val t = sr(ffmaT(a, ShiftRegister(laneIn, fpPipe), b, transport, fpPipe))
    // EXP and SILU share ONE merged activation core (mutually exclusive at runtime; func picks exp vs silu).
    val actES = if (hasExp || hasSilu) {
      val m = Module(new FpActivation(pipelined, hasExp, hasSilu, expLutN, siluN))
      m.io.in := t; m.io.func := (act === ACT_SILU); m.io.out
    } else t
    val tD = if (actLat > 0) ShiftRegister(t, actLat) else t // delay LINEAR to match EXP/SILU latency
    // pick the activation output only when `act` selects a BUILT non-linear func; else LINEAR (=tD).
    val selAct = (if (hasExp) act === ACT_EXP else false.B) || (if (hasSilu) act === ACT_SILU else false.B)
    val r =
      if (!hasExp && !hasSilu) tD
      else if (!hasLinear) actES
      else Mux(selAct, actES, tD)
    sr(narrow(r, transport, fpPipe)) // postLat stage (narrow cut by fpPipe)
  }

  // ---- streaming time-mux + pipeline FSM --------------------------------------------------------------
  // Continuous-issue rewrite (was: accept 1 beat -> issue subCycles -> DRAIN P cycles idle -> emit ->
  // re-accept). The `computeLanes` pipelined units are kept busy every cycle: a free-running (beat, sub)
  // issue pointer feeds one sub-group per cycle across back-to-back beats, so the P-cycle compute latency
  // is a one-time fill instead of a per-beat bubble. Results retire P cycles later, are reassembled into
  // whole beats, and land in a small output Queue (skid buffer) that absorbs the consumer's backpressure.
  // Steady-state throughput = 1 beat / subCycles cycles (== 1 beat/cycle when computeLanes == lanes),
  // independent of P. StreamMap is stateless per lane (no cross-beat accumulator), so beats stream freely.
  //
  // A credit counter caps in-flight + queued beats to the queue depth so no retired beat is ever dropped:
  // once issue stops (consumer stalls / input gap) the P-deep FP pipeline — which has no stall input —
  // keeps draining up to ceil(P/subCycles)+1 more beats, and the queue must have room for all of them.
  // CONTRACT: ext_start_i is asserted only when the extension is idle (ext_busy_o low, guaranteed by the
  // orchestration), so no pre-start in-flight result can leak into the new stream.
  val inBeat  = Reg(Vec(lanes, UInt(elementWidth.W)))
  val outBeat = Reg(Vec(lanes, UInt(elementWidth.W))) // assembles a beat across its subCycles retires

  val inFlightMax = (P + subCycles - 1) / subCycles + 1 // beats still draining after issue stops
  // minimum credit-safe depth: the credit caps outstanding at Qdepth and the FP pipeline retires at most
  // inFlightMax beats after issue stops, so the queue never overflows below inFlightMax (assert guards it).
  // Slack only helps under sustained backpressure; the writer drains faster than the time-muxed producer.
  val Qdepth      = scala.math.max(2, inFlightMax)
  val outQ = Module(new Queue(UInt((lanes * elementWidth).W), entries = Qdepth))

  // index of lane (s*computeLanes + j) into the `lanes`-wide Vec, width-exact to silence W004
  def li(s: UInt, j: Int): UInt = (s * computeLanes.U + j.U)(log2Ceil(lanes) - 1, 0)

  // issue pointer: `sub` sweeps 0..subCycles-1, one sub-group issued per cycle while a beat is latched.
  val haveBeat = RegInit(false.B)
  val sub      = RegInit(0.U(log2Ceil(subCycles).max(1).W))
  val lastSub  = sub === (subCycles - 1).U
  // credit = free output slots; reserve one per accepted beat, release one per dequeue.
  val credit = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  // accept a new beat when idle or finishing the current one this cycle, if a credit is free.
  ext_data_i.ready := (!haveBeat || lastSub) && (credit =/= 0.U) && !ext_start_i
  val accept = ext_data_i.fire

  val issuing = haveBeat
  val res = Wire(Vec(computeLanes, UInt(elementWidth.W)))
  for (j <- 0 until computeLanes)
    res(j) := computeLane(inBeat(li(sub, j)))

  // A valid-pulse pipeline CLEARED by ext_start_i, so an emit still in flight from the previous task cannot
  // fire — and push an unreserved beat past the credit reset — after the controller restarts the next task
  // while this pipeline is still draining.
  def clrPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(ext_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(ext_start_i, false.B, r(i - 1))
      r(n - 1)
    }

  // retire: the group issued P cycles ago lands now (P==0 ⇒ same cycle). Reassemble the beat: prior subs
  // sit in `outBeat`, the current sub is overlaid combinationally so the whole beat is captured before the
  // next beat's sub 0 (one cycle later) overwrites lane 0.
  val subRetire   = ShiftRegister(sub, P)
  val retireValid = clrPipe(issuing, P)
  val lastRetire  = subRetire === (subCycles - 1).U
  val outNow = WireInit(outBeat)
  when(retireValid) {
    for (j <- 0 until computeLanes) {
      outBeat(li(subRetire, j)) := res(j)
      outNow(li(subRetire, j))  := res(j)
    }
  }

  outQ.io.enq.valid := retireValid && lastRetire && !ext_start_i // no stale push on the restart cycle
  outQ.io.enq.bits  := Cat(outNow.reverse)
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "StreamMap: output queue overflow (credit accounting bug)")
  outQ.io.deq.ready := ext_data_o.ready
  ext_data_o.valid  := outQ.io.deq.valid
  ext_data_o.bits   := outQ.io.deq.bits

  val deq = outQ.io.deq.fire
  when(ext_start_i) {
    haveBeat := false.B; sub := 0.U; credit := Qdepth.U
  }.otherwise {
    when(accept) {
      inBeat := ext_data_i.bits.asTypeOf(Vec(lanes, UInt(elementWidth.W)))
      haveBeat := true.B; sub := 0.U
    }.elsewhen(issuing) {
      when(lastSub) { haveBeat := false.B; sub := 0.U }.otherwise { sub := sub + 1.U }
    }
    when(accept =/= deq) { credit := Mux(accept, credit - 1.U, credit + 1.U) }
  }

  ext_busy_o := (credit =/= Qdepth.U) || haveBeat
}
