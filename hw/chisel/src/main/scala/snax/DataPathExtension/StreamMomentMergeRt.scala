package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** StreamMomentMergeRt: the in-transit NONLINEAR collective. Folds one 512-bit beat carrying
  * up to `maxPairs` flash statistics (m_k, l_k) = (running max, Sexp) — one per shard/cluster — into the
  * merged pair (m*, l*) under the online-softmax moment-merge monoid
  *   (m,l) = ( max(m_a,m_b), l_winner + l_loser*exp(m_loser - m_winner) ).
  * This is the cross-shard combine SHARP provably cannot express (a max coupled to a rescaled sum). It reuses
  * the FP32 exp LUT and is FP32-internal, so the crossing stats stay exact at any transport precision. The
  * merged pair lands in the low 64 bits of the output beat (m* then l*), splatted for the writer.
  *
  * Beat layout: 16 FP32 lanes; m_k = lane k, l_k = lane (maxPairs + k), for k in 0..maxPairs-1 (maxPairs=8 at
  * dataWidth=512). One beat = one set of partials -> one merged pair (operandCount is implicit 1). The fold is
  * fully pipelined so beats stream at the 512 b/cyc roofline (F3: run this on the HeMAiA 512-bit inter-cluster
  * bus to combine the P clusters' partials in-transit).
  *
  * CSR layout: csr(0) bits[7:0] = nValid (number of live partials in the beat, 0->maxPairs); the rest are the
  * op identity (-inf, 0) so a short beat merges correctly.
  *
  * ACCUMULATE-ON-ARRIVAL (`accEn`, added 2026-07-20) -- the no-assemble / P>maxPairs mode.
  * --------------------------------------------------------------------------------------
  * In the base mode the fold is STATELESS PER BEAT, so the P partials must be made co-resident in ONE beat
  * before the single triggering push, and P independent pushes at one destination clobber rather than compose.
  * That forces an explicit assemble phase and caps a collective at P <= maxPairs (8).
  *
  * With `accEn` the extension keeps `numAccSlots` persistent (m,l) accumulators. Each arriving beat is first
  * folded intra-beat exactly as before, then merged INTO the selected slot under the same monoid:
  *   accInit=1 -> slot := this beat's folded pair   (the first producer arms the slot)
  *   accInit=0 -> slot := merge(slot, this beat's folded pair)
  * The output beat carries the running slot value, so the destination always holds the up-to-date partial
  * merge. P producers can each push STRAIGHT AT the merger -- no assemble phase, no P<=8 wall. The monoid is
  * associative and commutative, so arrival order does not matter.
  *
  * Extended CSR layout (csr(0); still ONE user CSR, so the inter-cluster cfg serdes is untouched):
  *   [7:0]   nValid
  *   [8]     accEn    1 = accumulate into a slot; 0 = stateless per beat (default, unchanged behaviour)
  *   [9]     accInit  1 = this push OVERWRITES the slot (arms it); 0 = folds into it
  *   [12:10] accSlot  which accumulator (0..numAccSlots-1)
  *
  * Accumulate mode admits ONE beat in flight (the slot is a read-modify-write hazard across beats), so
  * `ext_data_i.ready` is gated while an accumulate is in flight. That costs nothing for the intended use,
  * where each producer's push is a single-beat task. Stateless mode is untouched and still streams at the
  * roofline. Slots reset to the monoid identity (ID_M, 0) so an un-armed slot merges harmlessly.
  */
class HasStreamMomentMergeRt(
  dataWidth: Int = 512,
  numAccSlots: Int = 8
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(moduleName = "StreamMomentMergeRt", userCsrNum = 1, dataWidth = dataWidth)

  def instantiate(clusterName: String): StreamMomentMergeRt =
    Module(new StreamMomentMergeRt(numAccSlots = numAccSlots) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamMomentMergeRt(
  pipelined: Boolean = true,
  numAccSlots: Int = 8
)(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers._

  val accWidth = 32
  val maxPairs = extensionParam.dataWidth / (2 * accWidth) // 8 pairs (m,l) per 512-bit beat
  val numLevels = log2Ceil(maxPairs)
  // identity m = the most-negative FINITE FP32 (not -inf): it loses every max, and identity-identity gives
  // delta = id - id = 0 (not -inf - -inf = NaN), so a beat of all-identity lanes merges to (id, 0) cleanly.
  val ID_M      = "hFF7FFFFF".U(accWidth.W)
  val FP32_ZERO = 0.U(accWidth.W)

  val nValid = ext_csr_i(0)(7, 0) // number of live partials; lanes >= nValid get the identity (ID_M, 0)

  // accumulate-on-arrival controls (see the header comment). accSlot is sized to numAccSlots but always
  // read from the same fixed CSR field, so the cfg layout does not depend on the parameter.
  require(numAccSlots >= 1 && numAccSlots <= 8, "StreamMomentMergeRt: numAccSlots must be in 1..8 (csr(0)[12:10])")
  val accEn   = ext_csr_i(0)(8)
  val accInit = ext_csr_i(0)(9)
  val accSlot = ext_csr_i(0)(12, 10)

  // one moment-merge combine (pipelined via the exp LUT); returns (m, l, lat)
  private def merge(ma: UInt, la: UInt, mb: UInt, lb: UInt): (UInt, UInt, Int) =
    momentMerge(ma, la, mb, lb, pipelined, expLutN = 128)

  // ---- accept one beat; the fold latency threads through, output enqueued when the merge retires ----
  val inBeat = Reg(UInt(extensionParam.dataWidth.W))
  val lanes  = inBeat.asTypeOf(Vec(2 * maxPairs, UInt(accWidth.W)))
  // mask lanes beyond nValid to the monoid identity (-inf, 0) so short beats merge correctly
  def mOf(k: Int): UInt = Mux(k.U < nValid, lanes(k), ID_M)
  def lOf(k: Int): UInt = Mux(k.U < nValid, lanes(maxPairs + k), FP32_ZERO)

  // balanced moment-merge fold tree over the maxPairs pairs
  var ms: Seq[UInt] = (0 until maxPairs).map(mOf)
  var ls: Seq[UInt] = (0 until maxPairs).map(lOf)
  var foldLat = 0
  while (ms.length > 1) {
    val nm = scala.collection.mutable.ArrayBuffer[UInt]()
    val nl = scala.collection.mutable.ArrayBuffer[UInt]()
    var lvlLat = 0
    var i = 0
    while (i < ms.length) {
      val (mm, ll, lat) = merge(ms(i), ls(i), ms(i + 1), ls(i + 1))
      nm += mm; nl += ll; lvlLat = lat; i += 2
    }
    foldLat += lvlLat
    ms = nm.toSeq; ls = nl.toSeq
  }
  val mStar = ms.head
  val lStar = ls.head

  // ---- accumulate-on-arrival: fold the beat's pair INTO a persistent slot ----
  // Slots hold the monoid identity at reset and at every task start, so an un-armed slot merges harmlessly.
  val accM = RegInit(VecInit(Seq.fill(numAccSlots)(ID_M)))
  val accL = RegInit(VecInit(Seq.fill(numAccSlots)(FP32_ZERO)))
  val slotSel = if (numAccSlots == 1) 0.U else accSlot(log2Ceil(numAccSlots) - 1, 0)
  // The second merge, slot (+) this beat. Its inputs are stable for the whole accumulate window because
  // accumulate mode keeps a single beat in flight (see accBusy below), so its result is valid accLat cycles
  // after the intra-beat fold has retired.
  val (accMergedM, accMergedL, accLat) = merge(accM(slotSel), accL(slotSel), mStar, lStar)

  // ---- streaming FSM (credit + Queue; stateless per beat, like StreamMapRt) ----
  // +1: inBeat is registered on accept (available the next cycle), so the fold result lands foldLat cycles
  // after that -> the output valid pulse must be foldLat+1 cycles after the accept pulse.
  val P = foldLat + 1
  // In accumulate mode the result additionally traverses the slot merge, so it retires accLat cycles later.
  val Pacc   = P + accLat
  val Qdepth = scala.math.max(2, Pacc + 4)
  val outQ   = Module(new Queue(UInt(extensionParam.dataWidth.W), entries = Qdepth))
  val credit = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  // One accumulate in flight at a time: the slot is a read-modify-write across beats. Stateless mode is
  // unaffected (accBusy can only be set when accEn).
  val accBusy = RegInit(false.B)
  ext_data_i.ready := (credit =/= 0.U) && !ext_start_i && !accBusy
  val accept = ext_data_i.fire
  when(accept) { inBeat := ext_data_i.bits }

  // valid pulse cleared by ext_start_i so an in-flight merge can't push after a restart
  def clrPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(ext_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(ext_start_i, false.B, r(i - 1))
      r(n - 1)
    }
  val outValid    = clrPipe(accept, P)     // stateless mode: retires after the intra-beat fold
  val outValidAcc = clrPipe(accept, Pacc)  // accumulate mode: additionally after the slot merge

  // The value written back to the slot (and emitted): arm it on accInit, else fold into it. Both operands are
  // stable across the accumulate window because only one beat is in flight, so no extra alignment is needed.
  val newM = Mux(accInit, mStar, accMergedM)
  val newL = Mux(accInit, lStar, accMergedL)
  when(ext_start_i) {
    // a task start re-arms every slot to the monoid identity
    for (s <- 0 until numAccSlots) { accM(s) := ID_M; accL(s) := FP32_ZERO }
  }.elsewhen(outValidAcc && accEn) {
    accM(slotSel) := newM
    accL(slotSel) := newL
  }

  // accBusy holds ready low for the accumulate round trip so the slot is never read while an update is pending
  when(ext_start_i) { accBusy := false.B }
    .elsewhen(accept && accEn) { accBusy := true.B }
    .elsewhen(outValidAcc) { accBusy := false.B }

  // output beat: (m*, l*) in the low 64 bits, splatted across the 512-bit beat for the writer.
  // In accumulate mode the emitted pair is the RUNNING slot value, so the destination always holds the
  // up-to-date partial merge after every producer's push.
  val pair    = Mux(accEn, Cat(newL, newM), Cat(lStar, mStar)) // low 64 bits = m (0..31), l (32..63)
  val outBeat = Cat(Seq.fill(extensionParam.dataWidth / 64)(pair))
  outQ.io.enq.valid := Mux(accEn, outValidAcc, outValid) && !ext_start_i
  outQ.io.enq.bits  := outBeat
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "StreamMomentMergeRt: output queue overflow (credit bug)")
  outQ.io.deq.ready := ext_data_o.ready
  ext_data_o.valid  := outQ.io.deq.valid
  ext_data_o.bits   := outQ.io.deq.bits

  val deq = outQ.io.deq.fire
  when(ext_start_i) { credit := Qdepth.U }
    .otherwise { when(accept =/= deq) { credit := Mux(accept, credit - 1.U, credit + 1.U) } }

  ext_busy_o := credit =/= Qdepth.U
}
