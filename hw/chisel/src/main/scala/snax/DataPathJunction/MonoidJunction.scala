package snax.DataPathJunction

import chisel3._
import chisel3.util._

/** ============================================================================================================
  * `MonoidJunction` -- the nonlinear two-stream collective fold.
  * ============================================================================================================
  *
  * One `MonoidCombine` per slot, driven by a live operand PAIR. Mode-selected per transfer by the 3-bit
  * `combineMode` CSR field, so one netlist covers the whole arithmetic-reduction family:
  *
  * {{{
  *   SUM      (Sx, Sx^2)                distributed LayerNorm / RMSNorm statistics
  *   MOMENT   (m, l)                    the online-softmax normalizer
  *   ATTN     (m, l, O[dHead])          the flash-attention triple
  *   MAXPOOL  (m)                       max-reduce
  *   ARGMAX   (m, idx)                  top-1 / greedy decode / top-1 MoE routing
  *   MOMENT2  (m, l, Se^s v, Se^s v^2)  exp-weighted moment bank
  * }}}
  *
  * MOMENT, ATTN and MOMENT2 are nonlinear in the operand pair: they need a shared rescale
  * alpha = exp(m_loser - m*) derived from the pair itself, so they are not expressible as a per-element reduction.
  *
  * ---- THE JOIN IS CUT-THROUGH, NOT STORE-AND-FORWARD ----
  *
  * Each beat pair traverses the combine independently and is emitted `latency` cycles later, so the node starts
  * emitting as soon as its first operand pair is available and first-output latency does not grow with payload
  * size. Per-hop cost: 1 (input skid) + `MonoidCombine.latency` (the exp LUT) + 1 (output queue) + 1 (issue
  * register).
  *
  * Because the combine is stateless in the operand pair, the fold is fully pipelined at one beat pair per cycle.
  *
  * ---- BEAT LAYOUT ----
  *
  * PAIRED modes (SUM / MOMENT / MAXPOOL / ARGMAX): the beat carries `pairSlots` INDEPENDENT two-field partials,
  * field0_k = lane k, field1_k = lane pairSlots+k. Slot k of stream A folds with slot k of stream B: the reduction
  * is SIMD across slots and monoid across nodes -- e.g. `pairSlots` rows' softmax normalizers reduced across P
  * clusters in one pass. `nValid` masks the trailing slots to the monoid identity.
  *
  * SINGLE-PARTIAL modes (ATTN / MOMENT2): the beat is ONE (m, l, ...) partial of `Fmax = 2 + dHead` fields laid
  * across lanes 0..Fmax-1, folded by a single wide combine node.
  *
  * Slot 0's node is elaborated at the full `Fmax` field count and serves BOTH topologies (paired mode drives only
  * its low two fields, and the role decode parks the rest), so the wide single-partial mode costs no extra combine
  * unit.
  *
  * ---- CSR(0) ---- (one user CSR, so the inter-cluster cfg serdes carries it unchanged)
  * {{{
  *   [7:0]    nValid       live partials in the beat (paired modes); slots >= nValid get the monoid identity
  *   [15:13]  combineMode  SUM | MOMENT | ATTN | MAXPOOL | ARGMAX | MOMENT2
  * }}}
  * Bits [12:8] are the accumulate-slot fields of the extension placement and are IGNORED here -- a junction has no
  * slot state -- so one software CSR encoder serves both placements.
  */
class HasMonoidJunction(
  dataWidth:   Int = 512,
  dHead:       Int = 8,
  expLutN:     Int = 128,
  skidDepth:   Int = 4,
  pairSlots:   Int = 0, // 0 = auto = dataWidth / 64
  starveLimit: Int = 4096
) extends HasDataPathJunction {
  implicit val junctionParam: JunctionParam =
    new JunctionParam(
      moduleName  = "MonoidJunction",
      userCsrNum  = 1,
      dataWidth   = dataWidth,
      starveLimit = starveLimit
    )

  def instantiate(clusterName: String): MonoidJunction =
    Module(new MonoidJunction(dHead = dHead, expLutN = expLutN, skidDepth = skidDepth, pairSlotsParam = pairSlots) {
      override def desiredName = clusterName + namePostfix
    })
}

class MonoidJunction(
  dHead:          Int = 8,
  expLutN:        Int = 128,
  skidDepth:      Int = 4,
  pairSlotsParam: Int = 0
)(implicit
  junctionParam:  JunctionParam
) extends DataPathJunction {

  import MonoidCombine._

  val accWidth  = 32
  val nLanes    = junctionParam.dataWidth / accWidth        // 16 FP32 lanes at 512-bit
  val pairSlots = if (pairSlotsParam <= 0) junctionParam.dataWidth / (2 * accWidth) else pairSlotsParam
  val Fmax      = 2 + dHead                                  // fields in one (m, l, O) partial (ATTN)
  require(Fmax <= nLanes, s"MonoidJunction: 2+dHead ($Fmax) must fit one beat ($nLanes lanes)")
  require(2 * pairSlots <= nLanes, s"MonoidJunction: 2*pairSlots (${2 * pairSlots}) must fit one beat ($nLanes lanes)")

  val nValid      = jct_csr_i(0)(7, 0)
  val combineMode = jct_csr_i(0)(15, 13)
  val single      = isSingle(combineMode)

  // ---- elastic (skid) FIFOs on both operand streams --------------------------------------------------------
  // Depth >= 1 is required for CORRECTNESS (the join must not be a combinational loop between the two producers);
  // depth >= the link round-trip buys THROUGHPUT, since the two streams arrive with a skew set by the remote hop.
  // They let a middle gather node accept remote beats before its local reader catches up.
  val aQ = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = skidDepth, pipe = true, flow = false))
  val bQ = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = skidDepth, pipe = true, flow = false))
  aQ.io.enq <> jct_a_i
  bQ.io.enq <> jct_b_i

  // ---- output queue + credit, so the combine pipeline never has to be stalled mid-flight -------------------
  val lat    = MonoidCombine.latency
  val Qdepth = scala.math.max(2, lat + 4)
  val outQ   = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = Qdepth))
  val credit = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  // A pair fires whenever BOTH operands are present and the output pipeline has room: one pair per cycle. The
  // combine holds no state between pairs, so nothing serializes the issue.
  val fire = aQ.io.deq.valid && bQ.io.deq.valid && (credit =/= 0.U) && !jct_start_i
  aQ.io.deq.ready := fire
  bQ.io.deq.ready := fire

  val aLanes = aQ.io.deq.bits.asTypeOf(Vec(nLanes, UInt(accWidth.W)))
  val bLanes = bQ.io.deq.bits.asTypeOf(Vec(nLanes, UInt(accWidth.W)))

  // ---- the combine bank ------------------------------------------------------------------------------------
  // Slot 0 is elaborated at Fmax fields and is shared by both topologies; slots 1.. are two-field nodes.
  // Slot k >= nValid is fed the monoid identity on BOTH sides, so it contributes the identity to the output.
  def maskA(k: Int, f: Int): UInt =
    Mux(k.U < nValid, if (f == 0) aLanes(k) else aLanes(pairSlots + k), identityOf(combineMode, f))
  def maskB(k: Int, f: Int): UInt =
    Mux(k.U < nValid, if (f == 0) bLanes(k) else bLanes(pairSlots + k), identityOf(combineMode, f))

  // slot 0 operands: the single-partial beat (lanes 0..Fmax-1) or the paired slot-0 partial widened with identities
  val a0 = (0 until Fmax).map(f => Mux(single, aLanes(f), if (f < 2) maskA(0, f) else identityOf(combineMode, f)))
  val b0 = (0 until Fmax).map(f => Mux(single, bLanes(f), if (f < 2) maskB(0, f) else identityOf(combineMode, f)))
  val (out0, _) = MonoidCombine(a0, b0, combineMode, dHead, expLutN)

  val outPairs = (1 until pairSlots).map { k =>
    val (o, _) = MonoidCombine(Seq(maskA(k, 0), maskA(k, 1)), Seq(maskB(k, 0), maskB(k, 1)), combineMode, dHead, expLutN)
    o
  }

  // ---- output beat -----------------------------------------------------------------------------------------
  // single-partial: lay the merged (m, l, ...) across lanes 0..Fmax-1.
  // paired: slot k's (field0, field1) go back to lanes (k, pairSlots+k) -- the same layout the operands arrived in,
  // so a chain of gather nodes is closed under its own beat format.
  val singleBeat = Wire(Vec(nLanes, UInt(accWidth.W)))
  for (f <- 0 until Fmax) singleBeat(f) := out0(f)
  for (f <- Fmax until nLanes) singleBeat(f) := F32_ZERO

  val pairedBeat = Wire(Vec(nLanes, UInt(accWidth.W)))
  for (l <- 0 until nLanes) pairedBeat(l) := F32_ZERO
  pairedBeat(0)         := out0(0)
  pairedBeat(pairSlots) := out0(1)
  for (k <- 1 until pairSlots) {
    pairedBeat(k)             := outPairs(k - 1)(0)
    pairedBeat(pairSlots + k) := outPairs(k - 1)(1)
  }

  val outBeat = Mux(ShiftRegister(single, lat), singleBeat.asUInt, pairedBeat.asUInt)

  // The combine bank is a fixed-latency pipeline, so the retire pulse is the issue pulse delayed. `jct_start_i`
  // flushes it, so a new stream never inherits a half-traversed pair from the previous one.
  def clrPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(jct_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(jct_start_i, false.B, r(i - 1))
      r(n - 1)
    }
  val retire = clrPipe(fire, lat)

  outQ.io.enq.valid := retire && !jct_start_i
  outQ.io.enq.bits  := outBeat
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "MonoidJunction: output queue overflow (credit bug)")
  outQ.io.deq.ready := jct_data_o.ready
  jct_data_o.valid  := outQ.io.deq.valid
  jct_data_o.bits   := outQ.io.deq.bits

  val deq = outQ.io.deq.fire
  when(jct_start_i) { credit := Qdepth.U }
    .otherwise { when(fire =/= deq) { credit := Mux(fire, credit - 1.U, credit + 1.U) } }

  jct_busy_o := (credit =/= Qdepth.U) || aQ.io.deq.valid || bQ.io.deq.valid
}
