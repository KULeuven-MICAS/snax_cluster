package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** UnifiedMonoidMergeRt: ONE configurable combine cell for the whole in-fabric arithmetic-reduction family.
  *
  * StreamMomentMergeRt (softmax normalizer), StreamNormStatMergeRt (LayerNorm/RMSNorm stats) and
  * StreamAttnMergeRt (the flash-attention (m, ℓ, O) triple) are three copies of the SAME chassis -- the accEn
  * slot, the credit/Queue streaming FSM, the enable-gated bypass -- differing ONLY in the per-node combine.
  * This module folds all three (plus a degenerate max-pool) into one netlist whose combine is mode-selected
  * per transfer by a 3-bit CSR field. "One netlist for the distributed nonlinear collective" made literal.
  *
  * THE REPRESENTATION. Every combine in the family is the same op over an F-field FP32 partial, with each
  * field given a ROLE:
  *   KEY   out = max(a,b) (= m*)                     one distinguished field; produces the shared rescale α=exp(Δ)
  *   RSUM  out = winner + α·loser                    exponent-aligned sum (softmax ℓ, attention O)  [one FMA]
  *   SUM   out = a + b                               plain sum (norm Σx, Σx²)                        [the SAME FMA, α=1]
  *   OFF   out = identity (0)                         lane parked
  * SUM and RSUM are the SAME FMA (`a+b = ffma(b, 1.0, a)`), so a value field is one FMA + two config muxes
  * (a winner-swap and a scale ∈ {α, 1} select); KEY taps the front-end max; ONE exp LUT per combine unit feeds
  * every RSUM lane (Lemma 3 -- adding lanes never adds exp hardware).
  *
  * combineMode (csr(0)[15:13]) selects the role vector; `roleOf` is a tiny runtime decode of (mode, field):
  *   SUM      -> NormStat        : (SUM, SUM)                      no key, α idle
  *   MOMENT   -> softmax (m, ℓ)  : (KEY, RSUM)
  *   ATTN     -> (m, ℓ, O[dHead]): (KEY, RSUM, RSUM×dHead)        one α shared across ℓ and all O lanes
  *   MAXPOOL  -> max-reduce      : (KEY, OFF…)
  *
  * TWO wrapper topologies around the one combine (both always built; the mode picks which drives the output):
  *   - PAIRED-TREE (SUM / MOMENT / MAXPOOL): the beat packs 8 two-field partials (field0 = lanes 0..7, field1 =
  *     lanes 8..15); an 8->4->2->1 tree of identical combine nodes reduces them intra-beat. MOMENT and SUM share
  *     this tree verbatim, differing only in the per-node role -- the clean 2-for-1.
  *   - SINGLE-PARTIAL (ATTN): the beat is one (m, ℓ, O) partial (F = 2+dHead lanes); there is no intra-beat tree,
  *     the fold IS the accEn slot merge. The beat is delayed to the tree's retire point so the slot merge sees one
  *     aligned Fmax-field partial regardless of mode.
  *
  * The intra-beat result (`arm`, an Fmax-field partial: the tree's (m*,ℓ*) padded, or the ATTN beat) then folds
  * into the persistent accEn slot under ONE more combine node -- so P producers push straight at the merger
  * (associative + commutative monoid, arrival order irrelevant), exactly as the three fixed modules.
  *
  * CSR(0) -- still ONE user CSR, so the inter-cluster / D2D cfg serdes is untouched:
  *   [7:0]    nValid        live partials in the beat (paired modes); lanes >= nValid get the monoid identity
  *   [8]      accEn         1 = accumulate into a slot; 0 = stateless per beat
  *   [9]      accInit       1 = this push arms (overwrites) the slot; 0 = folds into it
  *   [12:10]  accSlot       which accumulator (0..numAccSlots-1)
  *   [15:13]  combineMode   SUM | MOMENT | ATTN | MAXPOOL
  *
  * accEn slots init to the KEY-mode identity (m = most-negative-finite, values = 0); SUM mode with accEn must
  * arm (accInit=1) on its first push -- the same "arming is software's job" contract the fixed modules carry.
  * FP32-internal throughout, so the crossing stats stay exact at any transport precision (they travel FP32).
  */
object MonoidRole { val OFF = 0; val KEY = 1; val SUM = 2; val RSUM = 3 }

class HasUnifiedMonoidMergeRt(
  dataWidth:   Int = 512,
  dHead:       Int = 8,
  expLutN:     Int = 128,
  numAccSlots: Int = 8
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(moduleName = "UnifiedMonoidMergeRt", userCsrNum = 1, dataWidth = dataWidth)

  def instantiate(clusterName: String): UnifiedMonoidMergeRt =
    Module(new UnifiedMonoidMergeRt(dHead = dHead, expLutN = expLutN, numAccSlots = numAccSlots) {
      override def desiredName = clusterName + namePostfix
    })
}

class UnifiedMonoidMergeRt(
  dHead:       Int = 8,
  expLutN:     Int = 128,
  numAccSlots: Int = 8
)(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers._

  val accWidth = 32
  val nLanes   = extensionParam.dataWidth / accWidth       // 16 FP32 lanes at 512-bit
  val maxPairs = extensionParam.dataWidth / (2 * accWidth)  // 8 (field0, field1) partials per beat
  val Fmax     = 2 + dHead                                  // fields in a single (m, ℓ, O) partial (ATTN)
  require(Fmax <= nLanes, s"UnifiedMonoidMergeRt: 2+dHead ($Fmax) must fit one beat ($nLanes lanes)")
  require(numAccSlots >= 1 && numAccSlots <= 8, "UnifiedMonoidMergeRt: numAccSlots must be in 1..8 (csr(0)[12:10])")

  val F32_ONE   = "h3F800000".U(accWidth.W)  // 1.0f  (scale for SUM / the winner factor)
  val ID_M      = "hFF7FFFFF".U(accWidth.W)  // most-negative finite fp32: loses every max, id-id delta = 0
  val FP32_ZERO = 0.U(accWidth.W)

  // combineMode encoding (csr(0)[15:13])
  val MODE_SUM     = 0
  val MODE_MOMENT  = 1
  val MODE_ATTN    = 2
  val MODE_MAXPOOL = 3

  val nValid      = ext_csr_i(0)(7, 0)
  val accEn       = ext_csr_i(0)(8)
  val accInit     = ext_csr_i(0)(9)
  val accSlot     = ext_csr_i(0)(12, 10)
  val combineMode = ext_csr_i(0)(15, 13)

  val isAttn     = combineMode === MODE_ATTN.U
  def modeHasKey: Bool = combineMode =/= MODE_SUM.U    // only SUM lacks a key (max + exp idle => α = 1)

  // role(mode, field): a runtime decode of the structured role vectors (field f is elaboration-time)
  def roleOf(f: Int): UInt = {
    val sumRole  = if (f < 2) MonoidRole.SUM.U else MonoidRole.OFF.U
    val momRole  = if (f == 0) MonoidRole.KEY.U else if (f == 1) MonoidRole.RSUM.U else MonoidRole.OFF.U
    val attnRole = if (f == 0) MonoidRole.KEY.U else if (f <= 1 + dHead) MonoidRole.RSUM.U else MonoidRole.OFF.U
    val maxRole  = if (f == 0) MonoidRole.KEY.U else MonoidRole.OFF.U
    MuxLookup(combineMode, sumRole)(Seq(
      MODE_SUM.U     -> sumRole,
      MODE_MOMENT.U  -> momRole,
      MODE_ATTN.U    -> attnRole,
      MODE_MAXPOOL.U -> maxRole
    ))
  }

  // ONE configurable combine over two F-field partials. Shares one exp LUT across every RSUM lane; returns the
  // merged partial + the exp-LUT pipeline latency (the value path is aligned to it via ShiftRegister).
  def combineNode(a: Seq[UInt], b: Seq[UInt]): (Seq[UInt], Int) = {
    val aWins  = fp32aWins(a(0), b(0))               // a.key >= b.key (only meaningful when field0 is KEY)
    val mStar  = Mux(aWins, a(0), b(0))
    val loser0 = Mux(aWins, b(0), a(0))
    val delta  = fadd(loser0, fneg32(mStar))         // loser.key - m*  (<= 0)
    val exp    = Module(new FpActivation(true, true, false, false, expLutN, 256))
    exp.io.in := delta; exp.io.func := false.B; exp.io.gelu := false.B
    val lat    = FpActivation.PipeLatency
    val alpha  = Mux(modeHasKey, exp.io.out, F32_ONE) // no key => α = 1.0 => the FMA is a plain add

    val out = (0 until a.length).map { f =>
      val role  = roleOf(f)
      val isKey = role === MonoidRole.KEY.U
      val isOff = role === MonoidRole.OFF.U
      val rsum  = role === MonoidRole.RSUM.U
      val swap  = rsum && !aWins                      // RSUM winner-swap; SUM is commutative (no swap)
      val los   = ShiftRegister(Mux(swap, a(f), b(f)), lat)
      val win   = ShiftRegister(Mux(swap, b(f), a(f)), lat)
      val scale = Mux(rsum, alpha, F32_ONE)           // α (RSUM) or 1.0 (SUM): the whole SUM/RSUM difference
      // 0-loser guard (as in the fixed modules): a 0 value adds exactly the winner, avoiding 0*exp(-huge)=NaN
      val fma   = Mux(los === FP32_ZERO, win, ffma(los, scale, win))
      Mux(isKey, ShiftRegister(mStar, lat), Mux(isOff, FP32_ZERO, fma))  // KEY | OFF | (SUM & RSUM => the FMA)
    }
    (out, lat)
  }

  // ---- accept one beat ----
  val inBeat = Reg(UInt(extensionParam.dataWidth.W))
  val lanes  = inBeat.asTypeOf(Vec(nLanes, UInt(accWidth.W)))

  // PAIRED-TREE intra-beat fold (SUM / MOMENT / MAXPOOL). Lanes >= nValid get the monoid identity: the KEY
  // field's identity is mode-dependent (ID_M for key modes, 0 for the keyless SUM), value fields' is 0.
  def f0Of(k: Int): UInt = Mux(k.U < nValid, lanes(k), Mux(modeHasKey, ID_M, FP32_ZERO))
  def f1Of(k: Int): UInt = Mux(k.U < nValid, lanes(maxPairs + k), FP32_ZERO)
  var parts: Seq[(UInt, UInt)] = (0 until maxPairs).map(k => (f0Of(k), f1Of(k)))
  var foldLat = 0
  while (parts.length > 1) {
    val next = scala.collection.mutable.ArrayBuffer[(UInt, UInt)]()
    var lvlLat = 0
    var i = 0
    while (i < parts.length) {
      val (o, lat) = combineNode(Seq(parts(i)._1, parts(i)._2), Seq(parts(i + 1)._1, parts(i + 1)._2))
      next += ((o(0), o(1))); lvlLat = lat; i += 2
    }
    foldLat += lvlLat
    parts = next.toSeq
  }
  val (treeM, treeL) = parts.head

  // SINGLE-PARTIAL (ATTN): the beat's own Fmax-field partial, delayed to the tree's retire point so the slot
  // merge sees ONE aligned partial regardless of mode.
  val attnFields = (0 until Fmax).map(f => ShiftRegister(lanes(f), foldLat))

  // `arm` = this beat's intra-beat-folded partial (tree result padded for paired modes; the beat for ATTN).
  val arm = (0 until Fmax).map { f =>
    if (f == 0) Mux(isAttn, attnFields(0), treeM)
    else if (f == 1) Mux(isAttn, attnFields(1), treeL)
    else Mux(isAttn, attnFields(f), FP32_ZERO)
  }

  // ---- accumulate-on-arrival: fold `arm` into a persistent slot under one more combine node ----
  // Slots hold the KEY-mode identity at RESET only; they persist across streams/tasks (ext_start_i note below).
  val slotInit = VecInit((0 until Fmax).map(f => if (f == 0) ID_M else FP32_ZERO))
  val accSlots = RegInit(VecInit(Seq.fill(numAccSlots)(slotInit)))
  val slotSel  = if (numAccSlots == 1) 0.U else accSlot(log2Ceil(numAccSlots) - 1, 0)
  val slotCur  = (0 until Fmax).map(f => accSlots(slotSel)(f))
  val (slotMerged, accLat) = combineNode(slotCur, arm)

  // ---- streaming FSM (credit + Queue + accBusy), identical to the fixed collective modules ----
  val P      = foldLat + 1                      // inBeat registered on accept; tree retires foldLat later
  val Pacc   = P + accLat                        // accumulate mode additionally traverses the slot merge
  val Qdepth = scala.math.max(2, Pacc + 4)
  val outQ   = Module(new Queue(UInt(extensionParam.dataWidth.W), entries = Qdepth))
  val credit = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  val accBusy = RegInit(false.B)                 // one accumulate in flight (the slot is a RMW across beats)
  ext_data_i.ready := (credit =/= 0.U) && !ext_start_i && !accBusy
  val accept = ext_data_i.fire
  when(accept) { inBeat := ext_data_i.bits }

  def clrPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(ext_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(ext_start_i, false.B, r(i - 1))
      r(n - 1)
    }
  val outValid    = clrPipe(accept, P)           // stateless: emit the intra-beat fold
  val outValidAcc = clrPipe(accept, Pacc)        // accumulate: additionally after the slot merge

  // slot write-back / emitted partial: arm (accInit) with this beat's partial, else the slot-merged one.
  val newFields  = (0 until Fmax).map(f => Mux(accInit, arm(f), slotMerged(f)))
  val emitFields = (0 until Fmax).map(f => Mux(accEn, newFields(f), arm(f)))
  // Slots deliberately SURVIVE ext_start_i (which pulses for every incoming remote push); arming is software's
  // job via accInit=1 on exactly one push. Same contract as the fixed collective modules.
  when(outValidAcc && accEn) {
    for (f <- 0 until Fmax) accSlots(slotSel)(f) := newFields(f)
  }

  when(ext_start_i) { accBusy := false.B }
    .elsewhen(accept && accEn) { accBusy := true.B }
    .elsewhen(outValidAcc) { accBusy := false.B }

  // output beat: ATTN lays the (m, ℓ, O) partial into lanes 0,1,2..; paired modes splat (field0, field1) across
  // the beat's low 64 bits for the writer (m*/S1 in 0..31, ℓ*/S2 in 32..63), as the fixed modules do.
  val attnLanes = Wire(Vec(nLanes, UInt(accWidth.W)))
  for (f <- 0 until Fmax) attnLanes(f) := emitFields(f)
  for (f <- Fmax until nLanes) attnLanes(f) := FP32_ZERO
  val pairedBeat = Cat(Seq.fill(extensionParam.dataWidth / 64)(Cat(emitFields(1), emitFields(0))))
  val outBeat    = Mux(isAttn, attnLanes.asUInt, pairedBeat)

  outQ.io.enq.valid := Mux(accEn, outValidAcc, outValid) && !ext_start_i
  outQ.io.enq.bits  := outBeat
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "UnifiedMonoidMergeRt: output queue overflow (credit bug)")
  outQ.io.deq.ready := ext_data_o.ready
  ext_data_o.valid  := outQ.io.deq.valid
  ext_data_o.bits   := outQ.io.deq.bits

  val deq = outQ.io.deq.fire
  when(ext_start_i) { credit := Qdepth.U }
    .otherwise { when(accept =/= deq) { credit := Mux(accept, credit - 1.U, credit + 1.U) } }

  ext_busy_o := credit =/= Qdepth.U
}
