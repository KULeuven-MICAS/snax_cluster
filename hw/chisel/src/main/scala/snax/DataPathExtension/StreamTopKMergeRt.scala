package snax.DataPathExtension

import chisel3._
import chisel3.util._

/** StreamTopKMergeRt: the THIRD monoid of the in-fabric collective algebra -- distributed MoE top-k
  * expert routing.
  *
  * An MoE router scores E experts per token; when the experts are sharded across clusters, each cluster
  * scores its OWN experts and holds a local list of the best (logit, expert-id) candidates. Global routing
  * needs the top-k experts by logit, WITH their indices, merged across clusters. This writer extension folds
  * the shards' candidate lists in-transit into the running GLOBAL top-M sorted list; the host then reads the
  * top-k prefix and softmaxes those k logits into the gating weights.
  *
  * WHY THIS IS GENUINELY BEYOND-SHARP (by EXPRESSIBILITY, unlike the norm monoid). SHARP-style in-network
  * reduction computes a single associative scalar fold -- sum, max. Selecting an ORDERED top-k SET with its
  * indices is not a linear reduce: it is a compare-exchange SORTING network. A max-tree gives you the top-1
  * value but neither the runner-up nor the winners' indices; there is no linear all-reduce that yields the
  * global top-k routing table. So top-k joins softmax (finite-precision overflow necessity) as the two members
  * of the algebra that are beyond-SHARP by EXPRESSIBILITY, not merely by locus/stability (the norm monoid).
  *
  * THE MONOID.
  *   element  = a list of up to M candidate (value: fp32 logit, index: uint32 expert-id) pairs
  *   identity = M copies of (ID_V = most-negative FINITE fp32 [loses every max], INVALID_IDX = 0xFFFFFFFF)
  *   merge(A,B) = sort(A concat B) DESCENDING by value, keep the top M
  * The order is TOTAL -- value descending, ties broken by SMALLER index -- so merge is associative and
  * commutative (arrival order is irrelevant, exactly what an in-fabric collective needs). The extension keeps
  * the running top-M=8; SOFTWARE reads the first k. This is exact for any global k <= M because
  *   top_k(A cup B) = top_k( top_M(A) cup top_M(B) )   for k <= M,
  * so a shard may pre-reduce to its own local top-M with no loss. M=8 covers every real MoE (Mixtral k=2,
  * DeepSeek-V3 k=8). A global k > 8 would need a wider slot (not a new mechanism); it does not occur in MoE.
  *
  * Structure MIRRORS StreamMomentMergeRt / StreamNormStatMergeRt exactly (accEn slot + credit/Queue FSM);
  * only the combine differs -- here a Batcher bitonic sorting network over (value, index) pairs instead of a
  * reduce tree. It inherits the accEn semantics verbatim: slots persist across ext_start_i, arming is
  * software's job (accInit=1 on exactly one producer).
  *
  * Beat layout: 16 FP32 lanes; value_k = lane k, index_k = lane (M+k), k in 0..M-1 (M=8 at 512 bit). nValid
  * (CSR[7:0]) live candidates; lanes >= nValid get the identity. The OUTPUT beat carries the sorted top-M list
  * in the SAME layout (value lane k, index lane M+k) -- it is NOT splatted, the beat *is* the sorted table.
  *
  * CSR(0): [7:0] nValid | [8] accEn | [9] accInit | [12:10] accSlot -- identical to the other two monoids, so
  * the single-user-CSR inter-cluster cfg serdes is untouched and a remote push carries these bits verbatim.
  */
class HasStreamTopKMergeRt(
  dataWidth:   Int = 512,
  numAccSlots: Int = 8
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(moduleName = "StreamTopKMergeRt", userCsrNum = 1, dataWidth = dataWidth)

  def instantiate(clusterName: String): StreamTopKMergeRt =
    Module(new StreamTopKMergeRt(numAccSlots = numAccSlots) {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamTopKMergeRt(
  numAccSlots: Int = 8
)(implicit
  extensionParam: DataPathExtensionParam
) extends DataPathExtension {

  import FpHelpers._

  val accWidth    = 32
  val M           = extensionParam.dataWidth / (2 * accWidth) // 8 (value,index) candidates per 512-bit beat
  // identity value = the most-negative FINITE fp32: it loses every max, so identity pairs always sink to the
  // bottom of the sort and never displace a real candidate. index = INVALID sentinel (surfaces only when a
  // collective has fewer than M real candidates, which the host detects by the sentinel / by nValid).
  val ID_V        = "hFF7FFFFF".U(accWidth.W)
  val INVALID_IDX = "hFFFFFFFF".U(accWidth.W)

  val nValid = ext_csr_i(0)(7, 0)
  require(numAccSlots >= 1 && numAccSlots <= 8, "StreamTopKMergeRt: numAccSlots must be in 1..8 (csr(0)[12:10])")
  val accEn   = ext_csr_i(0)(8)
  val accInit = ext_csr_i(0)(9)
  val accSlot = ext_csr_i(0)(12, 10)

  // ---- compare-exchange on (value, index) pairs under the total order (value asc, tie -> smaller index) ----
  // asc is a COMPILE-TIME Scala Boolean (the bitonic direction of this comparator), so the branch resolves at
  // elaboration -- no runtime mux on a constant.
  private type Pair = (UInt, UInt) // (value, index)
  private def cex(x: Pair, y: Pair, asc: Boolean): (Pair, Pair) = {
    val (vx, ix) = x; val (vy, iy) = y
    val aWinsXY = fp32aWins(vx, vy)                 // vx >= vy (finite fp32, sign-aware)
    val aWinsYX = fp32aWins(vy, vx)                 // vy >= vx
    // x <= y in the ASCENDING total order. On a VALUE tie, the LARGER index is the ascending-min, so after
    // sortDesc's `.reverse` the SMALLER index wins the tie in the delivered descending table -- matching the
    // numpy golden (sortBy (-value, index)) and the house argmax convention (StreamReduceRt keeps the smaller
    // index on a tie). (Distinct-logit datagen makes ties measure-zero in practice, but the convention must
    // still agree with software's local top-k so the top-8 decomposition stays exact.)
    val xLEy    = !aWinsXY || (aWinsYX && (ix >= iy))
    val keepXlo = if (asc) xLEy else !xLEy          // asc: smaller value at the lower wire; desc: larger
    val loV = Mux(keepXlo, vx, vy); val loI = Mux(keepXlo, ix, iy)
    val hiV = Mux(keepXlo, vy, vx); val hiI = Mux(keepXlo, iy, ix)
    ((loV, loI), (hiV, hiI))
  }

  // Batcher bitonic sort, ASCENDING by value. n must be a power of two. Each (k, j) stage's comparators touch
  // disjoint wire pairs (i, i^j), so the in-place update is a proper network stage (each wire written once per
  // stage). Fully unrolled at elaboration -> combinational compare-exchange logic.
  private def bitonicSortAsc(in: Seq[Pair]): Seq[Pair] = {
    val n = in.length
    require((n & (n - 1)) == 0 && n >= 1, s"bitonicSortAsc: length must be a power of two, got $n")
    val a = scala.collection.mutable.ArrayBuffer[Pair](in: _*)
    var k = 2
    while (k <= n) {
      var j = k / 2
      while (j > 0) {
        for (i <- 0 until n) {
          val l = i ^ j
          if (l > i) {
            val asc = (i & k) == 0                  // bitonic direction for this comparator
            val (lo, hi) = cex(a(i), a(l), asc)
            a(i) = lo; a(l) = hi
          }
        }
        j /= 2
      }
      k *= 2
    }
    a.toSeq
  }
  // DESCENDING (largest first) = ascending reversed (valid because the sort is TOTAL).
  private def sortDesc(in: Seq[Pair]): Seq[Pair] = bitonicSortAsc(in).reverse

  // ---- accept one beat; unpack its candidate lanes, masking beyond nValid to the identity ----
  val inBeat = Reg(UInt(extensionParam.dataWidth.W))
  val lanes  = inBeat.asTypeOf(Vec(2 * M, UInt(accWidth.W)))
  def vOf(k: Int): UInt = Mux(k.U < nValid, lanes(k), ID_V)
  def iOf(k: Int): UInt = Mux(k.U < nValid, lanes(M + k), INVALID_IDX)
  val cand: Seq[Pair] = (0 until M).map(k => (vOf(k), iOf(k)))

  // intra-beat fold: sort the beat's candidates -> the beat's local top-M (descending). Combinational.
  val beatSorted: Seq[Pair] = sortDesc(cand)
  val foldLat = 0

  // ---- accumulate-on-arrival: merge the beat's top-M INTO a persistent M-pair slot ----
  // Slots hold the identity at RESET only; they persist across streams/tasks (see the ext_start_i note below).
  // Two parallel Vecs (value, index) per the existing monoids' accM/accL convention, one dimension wider (M).
  val accV = RegInit(VecInit(Seq.fill(numAccSlots)(VecInit(Seq.fill(M)(ID_V)))))
  val accI = RegInit(VecInit(Seq.fill(numAccSlots)(VecInit(Seq.fill(M)(INVALID_IDX)))))
  val slotSel  = if (numAccSlots == 1) 0.U else accSlot(log2Ceil(numAccSlots) - 1, 0)
  val slotPairs: Seq[Pair] = (0 until M).map(k => (accV(slotSel)(k), accI(slotSel)(k)))
  // merge = sort(slot's M ++ this beat's top-M) descending, keep the top M
  val mergedTopM: Seq[Pair] = sortDesc(slotPairs ++ beatSorted).take(M)
  val accLat = 0

  // ---- streaming FSM (credit + Queue), identical to StreamMomentMergeRt / StreamNormStatMergeRt ----
  val P      = foldLat + 1
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

  // slot write-back: arm on accInit (slot := this beat's top-M), else fold (slot := merged top-M).
  val newV = (0 until M).map(k => Mux(accInit, beatSorted(k)._1, mergedTopM(k)._1))
  val newI = (0 until M).map(k => Mux(accInit, beatSorted(k)._2, mergedTopM(k)._2))
  // Slots deliberately SURVIVE ext_start_i (which pulses for every incoming remote push); arming is software's
  // job via accInit=1 on exactly one push. Same contract as StreamMomentMergeRt / StreamNormStatMergeRt.
  when(outValidAcc && accEn) {
    for (k <- 0 until M) { accV(slotSel)(k) := newV(k); accI(slotSel)(k) := newI(k) }
  }

  when(ext_start_i) { accBusy := false.B }
    .elsewhen(accept && accEn) { accBusy := true.B }
    .elsewhen(outValidAcc) { accBusy := false.B }

  // output beat: the sorted top-M table (value lane k, index lane M+k). accEn -> the running slot value;
  // stateless -> this beat's sorted candidates. NOT splatted -- the beat is the whole table.
  val emitV = (0 until M).map(k => Mux(accEn, newV(k), beatSorted(k)._1))
  val emitI = (0 until M).map(k => Mux(accEn, newI(k), beatSorted(k)._2))
  val outLanes = Wire(Vec(2 * M, UInt(accWidth.W)))
  for (k <- 0 until M) { outLanes(k) := emitV(k); outLanes(M + k) := emitI(k) }
  val outBeat = outLanes.asUInt

  outQ.io.enq.valid := Mux(accEn, outValidAcc, outValid) && !ext_start_i
  outQ.io.enq.bits  := outBeat
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "StreamTopKMergeRt: output queue overflow (credit bug)")
  outQ.io.deq.ready := ext_data_o.ready
  ext_data_o.valid  := outQ.io.deq.valid
  ext_data_o.bits   := outQ.io.deq.bits

  val deq = outQ.io.deq.fire
  when(ext_start_i) { credit := Qdepth.U }
    .otherwise { when(accept =/= deq) { credit := Mux(accept, credit - 1.U, credit + 1.U) } }

  ext_busy_o := credit =/= Qdepth.U
}
