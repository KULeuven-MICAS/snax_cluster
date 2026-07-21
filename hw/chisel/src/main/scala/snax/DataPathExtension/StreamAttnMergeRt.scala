package snax.DataPathExtension

import chisel3._
import chisel3.util._

import fp_unit._
import fp_native._

/** StreamAttnMergeRt: the COMPLETE distributed flash-attention collective — the `(m, ℓ, O)` triple.
  *
  * StreamMomentMergeRt folds the softmax NORMALIZER `(m, ℓ)` across shards. But distributed attention also
  * needs the attention OUTPUT: each shard `k` computes a local partial `O_k = Σ_j exp(s_j − m_k)·v_j` over
  * its KV block, and the global output is `O* = (Σ_k O_k·exp(m_k − m*)) / ℓ*`. This extension folds the full
  * flash-attention partial `(m_k, ℓ_k, O_k)` in-transit, on the SAME accumulate-on-arrival substrate — so the
  * whole distributed attention (not just its normalizer) is computed in the fabric; the host does one final
  * divide `O_star / ell_star`. This is the "complete the collective" upgrade (`07 §2b`, `10`).
  *
  * THE MONOID (associative + commutative — the flash-attention merge). State = `(m, ℓ, O[dHead])`.
  *   combine( (m_a,ℓ_a,O_a), (m_b,ℓ_b,O_b) ):
  *     m* = max(m_a, m_b)
  *     ℓ* = ℓ_win + ℓ_los·exp(m_los − m*)                 (winner factor exp(0)=1, free)
  *     O*[j] = O_win[j] + O_los[j]·exp(m_los − m*)          (SAME exp(Δ) applied to every O lane)
  * identity = (−∞_finite, 0, 0⃗). The `exp(Δ)` is shared across ℓ and all dHead lanes of O, so the extension
  * is StreamMomentMergeRt + carry O + one extra FMA per O lane. Beyond-SHARP for the same reason the softmax
  * fold is (a max coupled to a data-dependent rescaled sum, now vector-valued).
  *
  * Beat layout: lane 0 = m, lane 1 = ℓ, lanes 2..(2+dHead−1) = O[0..dHead−1] (dHead ≤ dataWidth/32 − 2 so one
  * shard's partial fits in one 512-bit beat; larger dHead spans multiple beats — a documented extension). One
  * beat = one shard's partial, folded into the accEn slot on arrival, so P producers push straight at the
  * merger. CSR(0): [7:0] nValid (unused; one partial/beat) | [8] accEn | [9] accInit | [12:10] accSlot —
  * identical to the other collective monoids.
  */
class HasStreamAttnMergeRt(
  dataWidth:   Int = 512,
  dHead:       Int = 8,
  numAccSlots: Int = 8
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(moduleName = "StreamAttnMergeRt", userCsrNum = 1, dataWidth = dataWidth)

  def instantiate(clusterName: String): StreamAttnMergeRt =
    Module(new StreamAttnMergeRt(dHead = dHead, numAccSlots = numAccSlots) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamAttnMergeRt(
  dHead:       Int = 8,
  numAccSlots: Int = 8
)(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers._

  val accWidth = 32
  val nLanes   = extensionParam.dataWidth / accWidth // 16 at 512-bit
  require(2 + dHead <= nLanes, s"StreamAttnMergeRt: 2+dHead ($dHead) must fit one beat ($nLanes lanes)")
  require(numAccSlots >= 1 && numAccSlots <= 8, "StreamAttnMergeRt: numAccSlots in 1..8")
  val ID_M      = "hFF7FFFFF".U(accWidth.W) // most-negative finite fp32 (loses every max; identity)
  val FP32_ZERO = 0.U(accWidth.W)

  val accEn   = ext_csr_i(0)(8)
  val accInit = ext_csr_i(0)(9)
  val accSlot = ext_csr_i(0)(12, 10)

  // one flash-attention triple combine: (m,ℓ,O) ⊕ (m,ℓ,O). Shares one exp(Δ) across ℓ and all O lanes.
  private def combine(mS: UInt, lS: UInt, oS: Seq[UInt],
                      mB: UInt, lB: UInt, oB: Seq[UInt]): (UInt, UInt, Seq[UInt], Int) = {
    val aWins  = fp32aWins(mS, mB)                 // slot >= incoming
    val m      = Mux(aWins, mS, mB)
    val loserM = Mux(aWins, mB, mS)
    val winL   = Mux(aWins, lS, lB); val losL = Mux(aWins, lB, lS)
    val winO   = (0 until dHead).map(k => Mux(aWins, oS(k), oB(k)))
    val losO   = (0 until dHead).map(k => Mux(aWins, oB(k), oS(k)))
    val delta  = fadd(loserM, fneg32(m))           // loser − winner ≤ 0
    val exp    = Module(new FpActivation(true, true, false, false, 128, 256))
    exp.io.in := delta; exp.io.func := false.B; exp.io.gelu := false.B
    val lat    = FpActivation.PipeLatency
    val losLA  = ShiftRegister(losL, lat); val winLA = ShiftRegister(winL, lat)
    // gate the loser's ℓ when it is 0 (identity/empty shard): add exactly the winner (avoid 0*exp(-huge)=NaN)
    val l      = Mux(losLA === FP32_ZERO, winLA, ffma(losLA, exp.io.out, winLA))
    val oStar  = (0 until dHead).map { k =>
      val wa = ShiftRegister(winO(k), lat); val la = ShiftRegister(losO(k), lat)
      ffma(la, exp.io.out, wa)                      // O_win + O_los·exp(Δ)
    }
    (ShiftRegister(m, lat), l, oStar, lat)
  }

  // ---- accept one beat = one shard's (m, ℓ, O) partial ----
  val inBeat = Reg(UInt(extensionParam.dataWidth.W))
  val lanes  = inBeat.asTypeOf(Vec(nLanes, UInt(accWidth.W)))
  val mIn = lanes(0); val lIn = lanes(1); val oIn = (0 until dHead).map(k => lanes(2 + k))

  // ---- accumulate-on-arrival slot: (m, ℓ, O) per slot; identity at reset; survives ext_start_i ----
  val accM = RegInit(VecInit(Seq.fill(numAccSlots)(ID_M)))
  val accL = RegInit(VecInit(Seq.fill(numAccSlots)(FP32_ZERO)))
  val accO = RegInit(VecInit(Seq.fill(numAccSlots)(VecInit(Seq.fill(dHead)(FP32_ZERO)))))
  val slotSel = if (numAccSlots == 1) 0.U else accSlot(log2Ceil(numAccSlots) - 1, 0)
  val mS = accM(slotSel); val lS = accL(slotSel); val oS = (0 until dHead).map(k => accO(slotSel)(k))
  val (mStar, lStar, oStar, accLat) = combine(mS, lS, oS, mIn, lIn, oIn)

  // ---- streaming FSM (credit + Queue + accBusy), identical to StreamMomentMergeRt ----
  val foldLat = 0                       // one partial per beat (no intra-beat tree); the fold is the slot merge
  val P       = foldLat + 1
  val Pacc    = P + accLat
  val Qdepth  = scala.math.max(2, Pacc + 4)
  val outQ    = Module(new Queue(UInt(extensionParam.dataWidth.W), entries = Qdepth))
  val credit  = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  val accBusy = RegInit(false.B)
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
  val outValid    = clrPipe(accept, P)     // stateless: pass the beat's (m,ℓ,O) through
  val outValidAcc = clrPipe(accept, Pacc)  // accumulate: after the slot combine

  // slot write-back: arm (accInit) with the beat's own (m,ℓ,O), else the combined triple. Inputs are stable
  // across the accumulate window (one beat in flight via accBusy), so no extra alignment is needed.
  val newM = Mux(accInit, mIn, mStar)
  val newL = Mux(accInit, lIn, lStar)
  val newO = (0 until dHead).map(k => Mux(accInit, oIn(k), oStar(k)))
  when(outValidAcc && accEn) {
    accM(slotSel) := newM
    accL(slotSel) := newL
    for (k <- 0 until dHead) accO(slotSel)(k) := newO(k)
  }

  when(ext_start_i) { accBusy := false.B }
    .elsewhen(accept && accEn) { accBusy := true.B }
    .elsewhen(outValidAcc) { accBusy := false.B }

  // output beat: (m, ℓ, O) in lanes 0,1,2..; accEn -> the running slot, stateless -> this beat. NOT splatted.
  val emitM = Mux(accEn, newM, mIn)
  val emitL = Mux(accEn, newL, lIn)
  val emitO = (0 until dHead).map(k => Mux(accEn, newO(k), oIn(k)))
  val outLanes = Wire(Vec(nLanes, UInt(accWidth.W)))
  outLanes(0) := emitM; outLanes(1) := emitL
  for (k <- 0 until dHead) outLanes(2 + k) := emitO(k)
  for (k <- (2 + dHead) until nLanes) outLanes(k) := FP32_ZERO
  val outBeat = outLanes.asUInt

  outQ.io.enq.valid := Mux(accEn, outValidAcc, outValid) && !ext_start_i
  outQ.io.enq.bits  := outBeat
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "StreamAttnMergeRt: output queue overflow (credit bug)")
  outQ.io.deq.ready := ext_data_o.ready
  ext_data_o.valid  := outQ.io.deq.valid
  ext_data_o.bits   := outQ.io.deq.bits

  val deq = outQ.io.deq.fire
  when(ext_start_i) { credit := Qdepth.U }
    .otherwise { when(accept =/= deq) { credit := Mux(accept, credit - 1.U, credit + 1.U) } }

  ext_busy_o := credit =/= Qdepth.U
}
