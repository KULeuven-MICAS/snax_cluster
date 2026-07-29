package snax.DataPathJunction

import chisel3._
import chisel3.util._

import snax.DataPathExtension.FpHelpers._

/** ============================================================================================================
  * `TopKJunction` -- an operator from OUTSIDE the classified monoid family, on the same socket.
  * ============================================================================================================
  *
  * State is a DESCENDING SORTED k-tuple of values, each with a payload:
  * {{{
  *   combine(A, B) = the k largest of A union B, sorted descending
  *   identity      = ( ID_M, ID_M, ... )        k copies of the most-negative finite FP32
  * }}}
  * Associative and commutative (both sides are "the k largest of the union"), bounded (k values + k payloads,
  * fixed), and format-closed by construction -- a sorted k-tuple in, a sorted k-tuple out, same layout.
  *
  * WHY THIS OPERATOR. It meets the classified family at exactly one point: `k = 1` with a payload IS argmax, the
  * selection corner. For `k > 1` there is no single key and no scalar rescale, so it is **outside the family
  * entirely** -- which makes it the evidence that the socket's contract, not the monoid cell, is what composes.
  * It also uses a beat layout that is NOT `lane = field*S + slot`, and does not have to: the contract requires
  * closure, not a geometry.
  *
  * ---- HOW IT MEETS THE CONTRACT ----
  * {{{
  *   O1 declared fixed latency   1 + log2(k) registered stages, published to the chassis
  *   O2 format closure           a descending sorted k-tuple with payloads, in and out, same lanes
  *   O3 identity tolerance       entries at or above nValid are ID_M, which loses every comparison
  *   O4 no state between pairs   pure feed-forward; nothing but the stage registers survives a pair
  * }}}
  *
  * ---- BEAT LAYOUT ---- (its own, and closed)
  * {{{
  *   partial p occupies lanes 2k*p .. 2k*p + 2k-1
  *     lanes 2k*p + 0 .. 2k*p + k-1     values, DESCENDING
  *     lanes 2k*p + k .. 2k*p + 2k-1    payloads; payload j pairs with value j
  *   2k lanes per partial => floor(nLanes / 2k) partials per beat
  *   k = 8 -> 1 partial/beat (16 lanes, 100% packed) ; k = 4 -> 2 partials/beat
  * }}}
  *
  * ---- THE MERGE IS NOT A SORT ----
  *
  * Both inputs are ALREADY sorted, so this is a bitonic MERGE, not a sorting network:
  * {{{
  *   stage 0            c_i = max( A_i , B_{k-1-i} )   -- the top-k SET, in bitonic order
  *   stages 1..log2(k)  bitonic merge of {c_i}         -- put it back in descending order
  * }}}
  * Stage 0 is k selects; each later stage is k/2 compare-exchanges. At k = 8 that is 8 + 12 = 20 comparators in
  * 4 registered stages -- NOT the fully combinational 16-input Batcher network that is the hazard on record.
  * There is no key front end, no exponential unit and no FMA anywhere in this operator.
  *
  * ---- CSR(0) ----
  * {{{
  *   [7:0]  nValid   live entries per input list; entries at or above it are the identity
  * }}}
  *
  * TIES: the comparator is `>=`, so an exact tie keeps the A-side entry -- the same deterministic rule `ARGMAX`
  * uses. With distinct values the operator is exactly commutative; on tied values the VALUE is still
  * commutative and only the carried payload can differ.
  */
class HasTopKJunction(
  dataWidth:   Int = 512,
  k:           Int = 8,
  skidDepth:   Int = 4,
  starveLimit: Int = 4096
) extends HasDataPathJunction {
  implicit val junctionParam: JunctionParam =
    new JunctionParam(
      moduleName  = "TopKJunction",
      userCsrNum  = 1,
      dataWidth   = dataWidth,
      starveLimit = starveLimit
    )

  def instantiate(clusterName: String): TopKJunction =
    Module(new TopKJunction(k = k, skidDepth = skidDepth) {
      override def desiredName = clusterName + namePostfix
    })
}

class TopKJunction(
  k:             Int = 8,
  skidDepth:     Int = 4
)(implicit
  junctionParam: JunctionParam
) extends DataPathJunction {

  val accWidth = 32
  val nLanes   = junctionParam.dataWidth / accWidth
  require(isPow2(k) && k >= 2, s"TopKJunction: k ($k) must be a power of two and at least 2")
  require(2 * k <= nLanes, s"TopKJunction: one partial needs 2k = ${2 * k} lanes, beat has $nLanes")
  val nParts = nLanes / (2 * k) // partials per beat

  /** the most-negative finite FP32 -- loses every comparison, so it is the top-k identity. Finite on purpose:
    * a true -Inf would make an identity-vs-identity comparison meaningless in the same way it does for the
    * monoid operator.
    */
  val ID_M = MonoidCombine.ID_M

  val nValid = jct_csr_i(0)(7, 0)

  // ---- socket plumbing ---------------------------------------------------------------------------------------
  val aQ = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = skidDepth, pipe = true, flow = false))
  val bQ = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = skidDepth, pipe = true, flow = false))
  aQ.io.enq <> jct_a_i
  bQ.io.enq <> jct_b_i

  /** O1: one register per merge stage -- stage 0 plus the log2(k) bitonic stages */
  val latency = 1 + log2Ceil(k)
  val Qdepth  = scala.math.max(2, latency + 4)
  val outQ    = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = Qdepth))
  val credit  = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  val fire = aQ.io.deq.valid && bQ.io.deq.valid && (credit =/= 0.U) && !jct_start_i
  aQ.io.deq.ready := fire
  bQ.io.deq.ready := fire

  assert(!(fire && nValid === 0.U), "TopKJunction: fired with nValid = 0 -- both lists are entirely identity")

  val aLanes = aQ.io.deq.bits.asTypeOf(Vec(nLanes, UInt(accWidth.W)))
  val bLanes = bQ.io.deq.bits.asTypeOf(Vec(nLanes, UInt(accWidth.W)))

  /** one (value, payload) entry of a sorted list */
  private case class Entry(v: UInt, p: UInt)
  private def regEntry(e: Entry): Entry = Entry(RegNext(e.v), RegNext(e.p))

  /** keep the larger of two entries; the payload follows its value */
  private def pick(x: Entry, y: Entry): Entry = {
    val xw = fp32aWins(x.v, y.v) // x >= y, so an exact tie keeps x
    Entry(Mux(xw, x.v, y.v), Mux(xw, x.p, y.p))
  }

  /** descending compare-exchange: larger to the low index */
  private def ce(x: Entry, y: Entry): (Entry, Entry) = {
    val xw = fp32aWins(x.v, y.v)
    (Entry(Mux(xw, x.v, y.v), Mux(xw, x.p, y.p)), Entry(Mux(xw, y.v, x.v), Mux(xw, y.p, x.p)))
  }

  val outLanes = Wire(Vec(nLanes, UInt(accWidth.W)))

  for (p <- 0 until nParts) {
    val base = 2 * k * p
    // O3: an entry at or above nValid is the identity on BOTH sides, so it loses every comparison and cannot
    // perturb the result -- and a short list emits identity in its tail, which is still a legal input partial.
    def entryOf(ln: Vec[UInt], i: Int): Entry =
      Entry(Mux(i.U < nValid, ln(base + i), ID_M), Mux(i.U < nValid, ln(base + k + i), 0.U(accWidth.W)))

    // ---- stage 0: the top-k SET, in bitonic order -----------------------------------------------------------
    // Both lists are descending, so pairing A_i against B_{k-1-i} and keeping the larger yields exactly the k
    // largest of the union. Nothing is sorted here -- the result is bitonic, which is what the next stages fix.
    var cur: Seq[Entry] =
      (0 until k).map(i => regEntry(pick(entryOf(aLanes, i), entryOf(bLanes, k - 1 - i))))

    // ---- stages 1..log2(k): bitonic merge back into descending order ----------------------------------------
    var s = k / 2
    while (s >= 1) {
      val nxt = Array.fill[Entry](k)(null)
      for (i <- 0 until k if (i & s) == 0) {
        val (hi, lo) = ce(cur(i), cur(i + s))
        nxt(i) = hi
        nxt(i + s) = lo
      }
      cur = nxt.toSeq.map(regEntry)
      s /= 2
    }

    for (i <- 0 until k) {
      outLanes(base + i)     := cur(i).v
      outLanes(base + k + i) := cur(i).p
    }
  }

  // ---- retire --------------------------------------------------------------------------------------------
  def clrPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(jct_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(jct_start_i, false.B, r(i - 1))
      r(n - 1)
    }
  val retire = clrPipe(fire, latency)

  outQ.io.enq.valid := retire && !jct_start_i
  outQ.io.enq.bits  := outLanes.asUInt
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "TopKJunction: output queue overflow (credit bug)")
  outQ.io.deq.ready := jct_data_o.ready
  jct_data_o.valid  := outQ.io.deq.valid
  jct_data_o.bits   := outQ.io.deq.bits

  val deq = outQ.io.deq.fire
  when(jct_start_i) { credit := Qdepth.U }
    .otherwise { when(fire =/= deq) { credit := Mux(fire, credit - 1.U, credit + 1.U) } }

  jct_busy_o := (credit =/= Qdepth.U) || aQ.io.deq.valid || bQ.io.deq.valid
}
