package snax.DataPathJunction

import chisel3._
import chisel3.util._

import fp_native._
import fp_unit._


/** ============================================================================================================
  * `MonoidJunction` -- the TWISTED operator on the junction socket.
  * ============================================================================================================
  *
  * One operator, one algebra. A partial is `(m, v_1..v_n)`: a key in a max-monoid and n values, combined as
  * {{{
  *   (m_a, v_a) (+) (m_b, v_b)  =  ( max(m_a, m_b),  alpha_a * v_a + alpha_b * v_b ),  alpha_k = exp(m_k - m*)
  * }}}
  * a semidirect product. The whole classified family -- norm statistics, the online-softmax normalizer, the
  * flash-attention triple, exp-weighted moment banks, max-pool, argmax/argmin -- is this one datapath under a
  * different geometry word. `MonoidCombine` holds the algebra; this file is the socket plumbing around it.
  *
  * Every partial carries exactly one key, which is what makes it a member of this family. A partial with no key
  * has nothing to compare and folds to the direct product -- a different algebra, and `ElementwiseJunction`'s
  * operator on this socket.
  *
  * The key lives in one of two monoids, chosen by `keyMul`. At `keyMul = 0` it is `(R, max)` and the twist is
  * `exp(m_lose - m*)`: the commutative family above. At `keyMul = 1` it is `(R, x)` and the twist is the B-side
  * key, which makes the operator the ORDERED SCAN
  * `(k_A, v_A) (+) (k_B, v_B) = ( k_A*k_B , k_B*v_A + v_B )` -- the chunked recurrence of linear attention and
  * of state-space models. That one is NOT commutative: `A` must be the earlier chunk. See `MonoidCombine` for
  * the algebra and for why both reach the same FMA.
  *
  * ---- HOW IT MEETS THE SOCKET CONTRACT ----
  * {{{
  *   O1 declared fixed latency   latMon = MonoidCombine.latency + fpPipe, published to the chassis below
  *   O2 format closure           the output beat is a legal INPUT beat: lane l carries (field(l), slot(l)) on
  *                               the way out exactly as it did on the way in, and dead slots carry the field
  *                               IDENTITY rather than zero, so a downstream hop reads the same answer at ANY
  *                               nValid. This is the obligation a 1->1 extension never has.
  *   O3 identity tolerance       a slot at or above nValid is fed its field's identity on BOTH sides
  *   O4 no state between pairs    nothing but the retire pipeline survives a beat pair
  * }}}
  *
  * ---- THE LANE MAP ---- (see `MonoidCombine` for why field-major, and why 8 key front ends is a theorem)
  * {{{
  *   lane = field * S + slot        slot(l) = l & (S-1)      field(l) = l >> sigma      S = 1 << sigma
  * }}}
  * `sigma` is a configuration field, so the layout is one rule with one parameter and there is no per-operator
  * special case anywhere in the block.
  *
  * ---- CSR(0) ---- (one user CSR, so the inter-cluster / D2D cfg serdes is untouched)
  * {{{
  *   [7:0]    nValid   live partials in the beat; slots >= nValid get their field's identity
  *   [11:8]   n        value coordinates; the partial is (m, v_1..v_n), so F = n + 1
  *   [21:18]  nExp     fields 1 .. nExp take the twist
  *   [25:22]  nAdd     fields nExp+1 .. nExp+nAdd are plainly summed; the rest carry the winner's payload
  *   [27:26]  sigma    beat geometry, saturated to the widest legal value for F
  *   [28]     keyPol   0 = max-monoid, 1 = min-monoid            (read only when keyMul = 0)
  *   [29]     keyMul   0 = the key monoid is (R, max), 1 = it is (R, x) -- the ordered scan
  * }}}
  * The word names a GEOMETRY, not an operator, which is what keeps the head width out of the netlist: a
  * twelve-wide flash-attention head is `n = 13, nExp = 13, sigma = 0` on the same hardware.
  */
class HasMonoidJunction(
  dataWidth:   Int = 512,
  fpPipe:      Int = 1,
  expLutN:     Int = 128,
  hasScan:     Boolean = true,
  skidDepth:   Int = 4,
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
    Module(new MonoidJunction(fpPipe = fpPipe, expLutN = expLutN, hasScan = hasScan,
                              skidDepth = skidDepth) {
      override def desiredName = clusterName + namePostfix
    })
}

class MonoidJunction(
  fpPipe:        Int = 1,
  expLutN:       Int = 128,
  hasScan:       Boolean = true,
  skidDepth:     Int = 4
)(implicit
  junctionParam: JunctionParam
) extends DataPathJunction {

  import MonoidCombine._

  // ---- geometry --------------------------------------------------------------------------------------------
  val accWidth = 32
  val nLanes   = junctionParam.dataWidth / accWidth // FP32 lanes in one beat (16 at 512-bit)
  val nKey     = nLanes / 2                         // key front ends; see the theorem below
  // "nLanes/2 key front ends is exactly sufficient" is a THEOREM, not a coincidence of dataWidth/64: `n` is a
  // 4-bit field so F = n+1 <= nLanes, and every partial has a key so F >= 1; `sigma <= SIGMA_MAX` then forces
  // S <= nLanes/2, so a key lane can never leave the low half of the beat, for ANY n.
  require((1 << SIGMA_MAX) <= nKey, s"MonoidJunction: nKey ($nKey) must cover the widest legal S")
  require(isPow2(nKey), s"MonoidJunction: nKey ($nKey) must be a power of two for the slot-class fold")

  // ---- CSR decode: the geometry IS the configuration --------------------------------------------------------
  val nValid = jct_csr_i(0)(7, 0)
  val geom   = geomOf(jct_csr_i(0))
  val sigma  = geom.sigma
  // The key monoid is (R, x) rather than (R, max): the ordered scan. `hasScan = false` ties it off at
  // elaboration, so a build that does not want the scan pays nothing for it -- see the alignment note below.
  val keyMul = if (hasScan) geom.keyMul else false.B

  // ---- socket plumbing: elastic input, credit-gated output ---------------------------------------------------
  // Skid depth >= 1 is required for CORRECTNESS (the join must not be a combinational loop between the two
  // producers); depth >= the link round-trip buys THROUGHPUT, since the two streams arrive with a skew set by
  // the remote hop.
  val aQ = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = skidDepth, pipe = true, flow = false))
  val bQ = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = skidDepth, pipe = true, flow = false))
  aQ.io.enq <> jct_a_i
  bQ.io.enq <> jct_b_i

  /** O1: the latency this operator publishes to the chassis -- the exp LUT, then the value lane's FMA */
  val latency = MonoidCombine.latency + fpPipe
  val Qdepth  = scala.math.max(2, latency + 4)
  val outQ    = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = Qdepth))
  val credit  = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  // O4: a pair fires whenever both operands are present and the output pipeline has room. The operator holds no
  // state between pairs, so nothing serializes the issue -- one pair per cycle.
  val fire = aQ.io.deq.valid && bQ.io.deq.valid && (credit =/= 0.U) && !jct_start_i
  aQ.io.deq.ready := fire
  bQ.io.deq.ready := fire

  // ---- O5: what this operator cannot honour --------------------------------------------------------------
  // All three are stream-constant, read from the CSR before a pair fires, and all three would otherwise be
  // SILENT: the beat that comes out is finite, format-legal and wrong in a way no downstream hop can detect.
  val cfgSigmaSaturated = jct_csr_i(0)(27, 26) > sigmaMaxOf(geom.F) // asked for a layout that does not exist
  val cfgRolesOverflow  = (geom.nExp +& geom.nAdd) > jct_csr_i(0)(11, 8) // more value roles than value fields
  val cfgNoLiveSlot     = nValid === 0.U                            // every slot masked to the identity
  jct_cfgerr_o := cfgSigmaSaturated || cfgRolesOverflow || cfgNoLiveSlot

  // `nValid = 0` masks EVERY slot to the identity, so the fold emits an identity partial: finite, format-legal
  // and numerically empty. Legal, but indistinguishable from a stale CSR. Simulation-only, so it cannot break a
  // real transfer; `jct_cfgerr_o` above is what reports it to software.
  assert(!(fire && nValid === 0.U),
         "MonoidJunction: fired with nValid = 0 -- every slot is masked to the identity")

  // ---- the operator: nLanes identical value lanes, nKey identical key front ends -----------------------------
  //
  // No layout mux, no slot-0 special case, no repack. Lane `l` reads `a(l)` and `b(l)` and writes `y(l)`: NO
  // DATA OPERAND EVER CROSSES A LANE. The only cross-lane wires are the twist and the winner-swap bit, which
  // one-exp-per-key forces to exist anyway.
  val aLanes = aQ.io.deq.bits.asTypeOf(Vec(nLanes, UInt(accWidth.W)))
  val bLanes = bQ.io.deq.bits.asTypeOf(Vec(nLanes, UInt(accWidth.W)))

  // SLOT-CLASS SHARING. `S <= nKey` and `nKey` is a power of two, so `S-1` only ever masks bits that `nKey-1`
  // already masks: `l & (S-1) == (l & (nKey-1)) & (S-1)`. Lane `l` and lane `l & (nKey-1)` are therefore ALWAYS
  // in the same slot, whatever sigma is, so everything that depends on the slot is built once per class.
  def cls(l: Int): Int = l & (nKey - 1)
  val cLive = (0 until nKey).map(c => slotOf(c, sigma) < nValid)

  /** the broadcast tap for slot class `c`: front end `c & (S-1)` across the legal sigmas. NESTED PREFIX MASKS,
    * so most classes have a single candidate and collapse to a plain wire -- the whole reason for field-major.
    */
  def tapOf[T <: Data](c: Int, sel: Int => T): T = {
    val cand = (0 to SIGMA_MAX).map(s => c & ((1 << s) - 1))
    if (cand.distinct.length == 1) sel(cand.head) // no mux at all
    else MuxLookup(sigma, sel(cand.head))((0 to SIGMA_MAX).map(s => s.U -> sel(cand(s))))
  }

  // per-lane decode -- all of it stream-constant, so none of it is in a data path
  val lField    = (0 until nLanes).map(l => fieldOf(l, sigma))
  val lInRange  = (0 until nLanes).map(l => lField(l) < geom.F)
  // `sigma <= SIGMA_MAX` means `field(l) >= 1` for every lane at or above `1 << SIGMA_MAX`, so no such lane can
  // EVER hold the key. State it, because the field decode lowers to a Vec lookup the tool cannot see through.
  val lIsKey    = (0 until nLanes).map(l =>
    if (l >= (1 << SIGMA_MAX)) false.B else lField(l) === 0.U)
  // In the scan the key lane is scaled too, because `k_A * k_B` is the same `los * scale` the value lanes use.
  val lUseAlpha = (0 until nLanes).map(l =>
    (lIsKey(l) && keyMul) || (lField(l) >= 1.U && lField(l) <= geom.nExp))
  // No `&& lInRange(l)`: an out-of-range lane retires zero from the outer select in `valueLane` regardless.
  val lIsSel    = (0 until nLanes).map(l => lField(l) > (geom.nExp +& geom.nAdd))
  // ... and it therefore stops SELECTING a winner: in the scan it computes one.
  val lSelOrKey = (0 until nLanes).map(l => (lIsKey(l) && !keyMul) || lIsSel(l))
  val lLive     = (0 until nLanes).map(l => cLive(cls(l)))
  // O3 on the INPUT, per lane: a dead slot is fed its field's identity on BOTH sides, so it contributes the
  // identity and the emitted beat stays a legal input partial at any downstream nValid.
  // The scan's key identity is 1.0, not the losing extreme: a dead slot must be NEUTRAL under multiplication.
  val padKey    = keyIdentity(geom.keyPol, keyMul)
  val lPad      = (0 until nLanes).map(l => Mux(lIsKey(l), padKey, F32_ZERO))
  // O3 strengthened: `lPad` is already the identity partial, lane by lane -- key fields get the monoid's neutral
  // element and value fields get zero -- so publishing it costs nothing but the wire.
  jct_identity_o := VecInit(lPad).asUInt

  val amL       = (0 until nLanes).map(l => Mux(lLive(l), aLanes(l), lPad(l)))
  val bmL       = (0 until nLanes).map(l => Mux(lLive(l), bLanes(l), lPad(l)))

  // key front ends: front end `s` sits ON lane `s` and reads that lane's own masked operands (lane s IS
  // (field 0, slot s) for every sigma, which is the property field-major buys).
  // In the ordered scan the twist is a beat lane and the swap is forced, so NOTHING downstream reads this block:
  // neither its lookup output nor its comparison. Hold its operands still rather than let eight exponential
  // units -- the most expensive arithmetic in the operator -- switch for a result no lane consumes.
  def feOperand(x: UInt): UInt = if (hasScan) Mux(keyMul, F32_ZERO, x) else x
  val fe = (0 until nKey).map(s => keyFrontEnd(feOperand(amL(s)), feOperand(bmL(s)), geom.keyPol, expLutN))

  // The two cross-lane wires, and the only place the key monoid is visible outside the lane decode.
  //   twist: the lookup's output, or -- in the scan -- the B-side key itself, straight off lane `s`.
  //   swap:  whichever side lost, or -- in the scan -- always A, because A is always the earlier chunk.
  // The lookup still runs in the scan and its output is simply unread, the same way it is for a partial whose
  // value coordinates are all plainly summed.
  //
  // ALIGNMENT. `valueLane` delays its operands by the lookup's depth, because in the twisted family the twist
  // comes OUT of the lookup and is already that old. The scan's twist is a raw beat lane, so it must be aged to
  // match or it arrives at the multiplier several cycles ahead of the value it is meant to scale. That delay is
  // the scan's whole marginal cost: nKey * 32 * MonoidCombine.latency flip-flops, built only when `hasScan`.
  //
  // (There is a cheaper form available if this ever matters: lane `s`'s own `win` register already holds exactly
  // this value, aged exactly this much. Reaching it means `valueLane` returning its delayed winner, which is a
  // change to the one function every lane of every configuration shares -- not worth it for a knob.)
  val bDel     = if (hasScan) (0 until nKey).map(s => ShiftRegister(bmL(s), MonoidCombine.latency)) else Seq()
  val alphaSrc = (s: Int) => if (hasScan) Mux(keyMul, bDel(s), fe(s)._1) else fe(s)._1
  val swSrc    = (s: Int) => fe(s)._2 || keyMul
  val alphaTap = (0 until nKey).map(c => tapOf(c, alphaSrc))
  val swTap    = (0 until nKey).map(c => tapOf(c, swSrc))

  // one FMA per lane, by an identity map: lane `l` drives FMA `l`. No pool and no allocator.
  val outBeat = VecInit((0 until nLanes).map { l =>
    valueLane(
      am       = amL(l),
      // Driving the key lane's B operand to zero is what turns `los*scale + win` into the product `k_A * k_B`.
      // Only the value lane sees this; the twist tap above still reads the real key off lane `s`.
      bm       = (if (l >= (1 << SIGMA_MAX)) bmL(l) else Mux(keyMul && lIsKey(l), F32_ZERO, bmL(l))),
      sw       = swTap(cls(l)),
      alpha    = alphaTap(cls(l)),
      useAlpha = lUseAlpha(l),
      selOrKey = lSelOrKey(l),
      inRange  = lInRange(l),
      fma      = (x: UInt, y: UInt, z: UInt) => {
        val m = Module(new FpFma(FP32, FP32, FP32, fpPipe))
        m.io.in_a := x; m.io.in_b := y; m.io.in_c := z; m.io.out
      },
      fmaLat   = fpPipe
    )
  }).asUInt

  // ---- retire ------------------------------------------------------------------------------------------------
  // O1 made structural: a fixed-latency pipeline, so the retire pulse is the issue pulse delayed by exactly the
  // published constant. `jct_start_i` flushes it, so a new stream never inherits a half-traversed pair.
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
