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
  */
class HasStreamMomentMergeRt(
  dataWidth: Int = 512
) extends HasDataPathExtension {
  implicit val extensionParam: DataPathExtensionParam =
    new DataPathExtensionParam(moduleName = "StreamMomentMergeRt", userCsrNum = 1, dataWidth = dataWidth)

  def instantiate(clusterName: String): StreamMomentMergeRt =
    Module(new StreamMomentMergeRt() {
      override def desiredName = clusterName + namePostfix
    })
}

class StreamMomentMergeRt(
  pipelined: Boolean = true
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

  // ---- streaming FSM (credit + Queue; stateless per beat, like StreamMapRt) ----
  // +1: inBeat is registered on accept (available the next cycle), so the fold result lands foldLat cycles
  // after that -> the output valid pulse must be foldLat+1 cycles after the accept pulse.
  val P = foldLat + 1
  val Qdepth = scala.math.max(2, P + 4)
  val outQ   = Module(new Queue(UInt(extensionParam.dataWidth.W), entries = Qdepth))
  val credit = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  ext_data_i.ready := (credit =/= 0.U) && !ext_start_i
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
  val outValid = clrPipe(accept, P)

  // output beat: (m*, l*) in the low 64 bits, splatted across the 512-bit beat for the writer
  val pair    = Cat(lStar, mStar) // low 64 bits = m* (0..31), l* (32..63)
  val outBeat = Cat(Seq.fill(extensionParam.dataWidth / 64)(pair))
  outQ.io.enq.valid := outValid && !ext_start_i
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
