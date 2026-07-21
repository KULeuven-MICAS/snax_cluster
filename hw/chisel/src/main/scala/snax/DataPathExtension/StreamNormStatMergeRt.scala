package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** StreamNormStatMergeRt: the SECOND monoid of the in-fabric collective algebra -- distributed
  * LayerNorm / RMSNorm statistics.
  *
  * Each cluster reduces its shard of the hidden dim to a pair (S1, S2) = (Σx, Σx²); this writer
  * extension folds the pairs across clusters, in-transit, into the running global (Σx, Σx²). The host
  * then projects: mean = Σx / n, var = Σx²/n − mean² (LayerNorm), or rms = sqrt(Σx²/n) (RMSNorm) --
  * the same host-side reciprocal/rsqrt already used for the single-cluster norm epilogues.
  *
  * HONEST SCOPE. Unlike the softmax moment-merge (`StreamMomentMergeRt`), the norm fold is LINEAR in
  * (Σx, Σx²) -- a plain sum -- so it is expressible by a linear all-reduce (SHARP could do it too).
  * Its value in the algebra is therefore NOT "beyond-SHARP by expressibility"; it is (a) the in-fabric
  * LOCUS (computed as the shards move, no gather/all-reduce round on the cores), (b) FP32-internal
  * accumulation -- the Σx² sum is done in FP32 in the mover, avoiding the catastrophic cancellation that
  * an FP16-transport sum-of-squares suffers at scale -- and (c) UNIFICATION: the transformer's THREE
  * distributed reductions (softmax, norm, top-k routing) all fold on the SAME accumulate-on-arrival
  * substrate, one monoid each. The genuinely beyond-SHARP (expressibility) monoids are softmax
  * (finite-precision overflow necessity) and top-k routing (not a linear reduce at all).
  *
  * Structure MIRRORS StreamMomentMergeRt exactly (fold tree + accEn slot + credit/Queue FSM); only the
  * combine (a pair-add) and the identity ((0,0)) differ, so it inherits all the proven flow control,
  * including the accEn semantics: slots persist across ext_start_i, arming is software's job (accInit=1).
  *
  * Beat layout: 16 FP32 lanes; S1_k = lane k, S2_k = lane (maxPairs+k), k in 0..maxPairs-1 (maxPairs=8).
  * CSR(0): [7:0] nValid | [8] accEn | [9] accInit | [12:10] accSlot -- identical to StreamMomentMergeRt.
  */
class HasStreamNormStatMergeRt(
  dataWidth:   Int = 512,
  fpPipe:      Int = 1,
  numAccSlots: Int = 8
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(moduleName = "StreamNormStatMergeRt", userCsrNum = 1, dataWidth = dataWidth)

  def instantiate(clusterName: String): StreamNormStatMergeRt =
    Module(new StreamNormStatMergeRt(fpPipe = fpPipe, numAccSlots = numAccSlots) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamNormStatMergeRt(
  fpPipe:      Int = 1,
  numAccSlots: Int = 8
)(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers._

  val accWidth  = 32
  val maxPairs  = extensionParam.dataWidth / (2 * accWidth) // 8 (S1,S2) pairs per 512-bit beat
  val FP32_ZERO = 0.U(accWidth.W)

  val nValid  = ext_csr_i(0)(7, 0)
  require(numAccSlots >= 1 && numAccSlots <= 8, "StreamNormStatMergeRt: numAccSlots must be in 1..8")
  val accEn   = ext_csr_i(0)(8)
  val accInit = ext_csr_i(0)(9)
  val accSlot = ext_csr_i(0)(12, 10)

  // the additive monoid combine (S1a+S1b, S2a+S2b); returns (S1, S2, lat)
  private def merge(s1a: UInt, s2a: UInt, s1b: UInt, s2b: UInt): (UInt, UInt, Int) =
    (fadd(s1a, s1b, fpPipe), fadd(s2a, s2b, fpPipe), fpPipe)

  // ---- accept one beat; fold its live pairs ----
  val inBeat = Reg(UInt(extensionParam.dataWidth.W))
  val lanes  = inBeat.asTypeOf(Vec(2 * maxPairs, UInt(accWidth.W)))
  // lanes >= nValid get the additive identity (0, 0) so a short beat sums correctly
  def s1Of(k: Int): UInt = Mux(k.U < nValid, lanes(k), FP32_ZERO)
  def s2Of(k: Int): UInt = Mux(k.U < nValid, lanes(maxPairs + k), FP32_ZERO)

  // balanced additive fold tree over the maxPairs pairs
  var a1: Seq[UInt] = (0 until maxPairs).map(s1Of)
  var a2: Seq[UInt] = (0 until maxPairs).map(s2Of)
  var foldLat = 0
  while (a1.length > 1) {
    val n1 = scala.collection.mutable.ArrayBuffer[UInt]()
    val n2 = scala.collection.mutable.ArrayBuffer[UInt]()
    var lvlLat = 0
    var i = 0
    while (i < a1.length) {
      val (s1, s2, lat) = merge(a1(i), a2(i), a1(i + 1), a2(i + 1))
      n1 += s1; n2 += s2; lvlLat = lat; i += 2
    }
    foldLat += lvlLat
    a1 = n1.toSeq; a2 = n2.toSeq
  }
  val s1Star = a1.head
  val s2Star = a2.head

  // ---- accumulate-on-arrival: fold the beat's pair INTO a persistent slot ----
  // Slots hold the additive identity (0,0) at RESET only; they persist across streams/tasks (see the
  // ext_start_i note below). Software arms a slot for a fresh reduction with accInit=1.
  val accS1 = RegInit(VecInit(Seq.fill(numAccSlots)(FP32_ZERO)))
  val accS2 = RegInit(VecInit(Seq.fill(numAccSlots)(FP32_ZERO)))
  val slotSel = if (numAccSlots == 1) 0.U else accSlot(log2Ceil(numAccSlots) - 1, 0)
  val (accMergedS1, accMergedS2, accLat) = merge(accS1(slotSel), accS2(slotSel), s1Star, s2Star)

  // ---- streaming FSM (credit + Queue), identical to StreamMomentMergeRt ----
  val P = foldLat + 1
  val Pacc   = P + accLat
  val Qdepth = scala.math.max(2, Pacc + 4)
  val outQ   = Module(new Queue(UInt(extensionParam.dataWidth.W), entries = Qdepth))
  val credit = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

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
  val outValid    = clrPipe(accept, P)
  val outValidAcc = clrPipe(accept, Pacc)

  val newS1 = Mux(accInit, s1Star, accMergedS1)
  val newS2 = Mux(accInit, s2Star, accMergedS2)
  // Slots deliberately SURVIVE ext_start_i (which pulses for every incoming remote push); arming is
  // software's job via accInit=1 on exactly one push. Same contract as StreamMomentMergeRt.
  when(outValidAcc && accEn) {
    accS1(slotSel) := newS1
    accS2(slotSel) := newS2
  }

  when(ext_start_i) { accBusy := false.B }
    .elsewhen(accept && accEn) { accBusy := true.B }
    .elsewhen(outValidAcc) { accBusy := false.B }

  // output beat: (S1, S2) in the low 64 bits, splatted across the 512-bit beat for the writer
  val pair    = Mux(accEn, Cat(newS2, newS1), Cat(s2Star, s1Star)) // low 64 = S1 (0..31), S2 (32..63)
  val outBeat = Cat(Seq.fill(extensionParam.dataWidth / 64)(pair))
  outQ.io.enq.valid := Mux(accEn, outValidAcc, outValid) && !ext_start_i
  outQ.io.enq.bits  := outBeat
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "StreamNormStatMergeRt: output queue overflow (credit bug)")
  outQ.io.deq.ready := ext_data_o.ready
  ext_data_o.valid  := outQ.io.deq.valid
  ext_data_o.bits   := outQ.io.deq.bits

  val deq = outQ.io.deq.fire
  when(ext_start_i) { credit := Qdepth.U }
    .otherwise { when(accept =/= deq) { credit := Mux(accept, credit - 1.U, credit + 1.U) } }

  ext_busy_o := credit =/= Qdepth.U
}
