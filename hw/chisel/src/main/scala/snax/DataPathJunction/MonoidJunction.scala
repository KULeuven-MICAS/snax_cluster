package snax.DataPathJunction

import chisel3._
import chisel3.util._

import fp_native._
import fp_unit._

import snax.DataPathExtension.FpHelpers

/** \============================================================================================================
  * `MonoidJunction` -- the TWISTED operator on the junction socket.
  * \============================================================================================================
  *
  * One operator, one algebra. A partial is `(m, v_1..v_n)`: a key in a max-monoid and n values, combined as
  * {{{
  *   (m_a, v_a) (+) (m_b, v_b)  =  ( max(m_a, m_b),  alpha_a * v_a + alpha_b * v_b ),  alpha_k = exp(m_k - m*)
  * }}}
  * a semidirect product. The whole classified family -- norm statistics, the online-softmax normalizer, the
  * flash-attention triple, exp-weighted moment banks, max-pool, argmax/argmin -- is this one datapath under a different
  * geometry word. `MonoidCombine` holds the algebra; this file is the socket plumbing around it.
  *
  * Every partial carries exactly one key, which is what makes it a member of this family. A partial with no key has
  * nothing to compare and folds to the direct product -- a different algebra, and `ElementwiseJunction`'s operator on
  * this socket.
  *
  * The key lives in one of two monoids, chosen by `keyMul`. At `keyMul = 0` it is `(R, max)` and the twist is
  * `exp(m_lose - m*)`: the commutative family above. At `keyMul = 1` it is `(R, x)` and the twist is the B-side key,
  * which makes the operator the ORDERED SCAN `(k_A, v_A) (+) (k_B, v_B) = ( k_A*k_B , k_B*v_A + v_B )` -- the chunked
  * recurrence of linear attention and of state-space models. That one is NOT commutative: `A` must be the earlier
  * chunk. See `MonoidCombine` for the algebra and for why both reach the same FMA.
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
  * `sigma` is a configuration field, so the layout is one rule with one parameter and there is no per-operator special
  * case anywhere in the block.
  *
  * ---- TRANSPORT FORMAT ---- (the same split `ElementwiseJunction` has on this socket)
  * `elemWidth` (elaboration) sets the lane count `nLanes = dataWidth / elemWidth`; the ARITHMETIC is always FP32, so
  * `fmt` (runtime) selects only how a beat is sliced on the way in and packed on the way out. FP16 here is a TRANSPORT
  * format, not an arithmetic one: every key compare, exp lookup and FMA is bit-for-bit what it was before. `elemWidth =
  * 32` is the default and reproduces the FP32-only module exactly, down to the published latency.
  *
  * ---- CSR(0) ---- (one user CSR, so the inter-cluster / D2D cfg serdes is untouched)
  * {{{
  *   [7:0]    nValid   live partials in the beat; slots >= nValid get their field's identity
  *   [11:8]   n        value coordinates; the partial is (m, v_1..v_n), so F = n + 1
  *   [14:12]  fmt      TRANSPORT format: 0 = FP16, 1 = BF16, 2 = FP8, 3 = FP32   (FpHelpers.FMT_*)
  *   [21:18]  nExp     fields 1 .. nExp take the twist
  *   [25:22]  nAdd     fields nExp+1 .. nExp+nAdd are plainly summed; the rest carry the winner's payload
  *   [27:26]  sigma    beat geometry, saturated to the widest legal value for F
  *   [28]     keyPol   0 = max-monoid, 1 = min-monoid            (read only when keyMul = 0)
  *   [29]     keyMul   0 = the key monoid is (R, max), 1 = it is (R, x) -- the ordered scan
  * }}}
  * The word names a GEOMETRY, not an operator, which is what keeps the head width out of the netlist: a twelve-wide
  * flash-attention head is `n = 13, nExp = 13, sigma = 0` on the same hardware.
  */
class HasMonoidJunction(
  dataWidth:   Int     = 512,
  // The transport element width. 32 is the FP32-only module this block shipped as; 16 slices the beat into 32
  // lanes and lets `fmt` carry FP16/BF16 at full beat rate (and FP32 on lanes 0..15).
  elemWidth:   Int     = 32,
  fpPipe:      Int     = 1,
  expLutN:     Int     = 128,
  hasScan:     Boolean = true,
  skidDepth:   Int     = 4,
  starveLimit: Int     = 4096
) extends HasDataPathJunction {
  implicit val junctionParam: JunctionParam =
    new JunctionParam(
      moduleName  = "MonoidJunction",
      userCsrNum  = 1,
      dataWidth   = dataWidth,
      starveLimit = starveLimit
    )

  def instantiate(clusterName: String): MonoidJunction =
    Module(new MonoidJunction(
      elemWidth = elemWidth,
      fpPipe    = fpPipe,
      expLutN   = expLutN,
      hasScan   = hasScan,
      skidDepth = skidDepth
    ) {
      override def desiredName = clusterName + namePostfix
    })
}

class MonoidJunction(
  elemWidth:     Int     = 32,
  fpPipe:        Int     = 1,
  expLutN:       Int     = 128,
  hasScan:       Boolean = true,
  skidDepth:     Int     = 4
)(implicit
  junctionParam: JunctionParam
) extends DataPathJunction {

  import MonoidCombine._

  // ---- geometry --------------------------------------------------------------------------------------------
  val accWidth = 32 // the ARITHMETIC width. It is NOT the transport width -- see `elemWidth` below.
  require(
    isPow2(elemWidth) && elemWidth >= 8 && elemWidth <= 32,
    s"MonoidJunction: elemWidth ($elemWidth) must be 8, 16 or 32"
  )
  val nLanes   = junctionParam.dataWidth / elemWidth // FP32 lanes in one beat (32 at 512-bit, elemWidth 16)
  // "nLanes/2 key front ends is exactly sufficient" is a THEOREM, not a coincidence of dataWidth/64: `n` is a
  // 4-bit field so F = n+1 <= nLanes, and every partial has a key so F >= 1; `sigma <= SIGMA_MAX` then forces
  // S <= nLanes/2, so a key lane can never leave the low half of the beat, for ANY n.
  //
  // It is a theorem with a CEILING, and the min below is where that matters. The derivation bounds S by
  // `1 << SIGMA_MAX` = 8 REGARDLESS of nLanes, so only 8 front ends are ever reachable. Halving `elemWidth` must
  // therefore double the FMAs and NOT the exponential units -- the key front end is where the exp LUT lives, and
  // it is the most expensive arithmetic in the block.
  val nKey     = scala.math.min(nLanes / 2, 1 << SIGMA_MAX)
  require((1 << SIGMA_MAX) <= nKey, s"MonoidJunction: nKey ($nKey) must cover the widest legal S")
  require(isPow2(nKey), s"MonoidJunction: nKey ($nKey) must be a power of two for the slot-class fold")

  // ---- transport formats -------------------------------------------------------------------------------------
  // The same table and the same codes `ElementwiseJunction` and the SIMD extensions already use, rather than a
  // second encoding of the same thing. INTEGER formats are deliberately absent: the twist is `exp(m_lose - m*)`,
  // so a partial's key and its twisted coordinates are inherently floating point.
  val supported  = ElementwiseJunction.formats.filter(_._2 >= elemWidth)
  require(supported.nonEmpty, s"MonoidJunction: no transport format is >= elemWidth=$elemWidth")
  val multiFmt   = supported.length > 1
  // O1, and the one place a format change can break it. The narrow is a rounding step and costs a stage; the
  // widen is pure bit-manipulation and costs none. `latency` is published to the chassis at ELABORATION while
  // `fmt` is a RUNTIME field, so the stage is taken on every arm of the format mux, FP32 included. An
  // FP32-only build has nothing to round and pays nothing -- which is what keeps `elemWidth = 32`
  // bit-identical, latency included, to the FP32-only module.
  //
  // Keyed on WHICH formats are built, not on how many: a build trimmed to FP16 alone is still a real
  // rounding narrow and still needs its stage, and `multiFmt` would say no.
  val needsNarrow = supported.exists(_._3 != FP32)
  val narrowPipe  = if (needsNarrow) fpPipe else 0

  // ---- CSR decode: the geometry IS the configuration --------------------------------------------------------
  val nValid = jct_csr_i(0)(7, 0)
  val geom   = geomOf(jct_csr_i(0), nLanes)
  val sigma  = geom.sigma
  val fmt    = jct_csr_i(0)(14, 12)
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

  /** O1: the latency this operator publishes to the chassis -- the exp LUT, the value lane's FMA, and (only on a
    * multi-format build) the narrow back to the transport format.
    */
  val latency = MonoidCombine.latency + fpPipe + narrowPipe
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
  val cfgSigmaSaturated = jct_csr_i(0)(27, 26) > sigmaMaxOf(geom.F, nLanes) // asked for a layout that does not exist
  val cfgRolesOverflow  = (geom.nExp +& geom.nAdd) > jct_csr_i(0)(11, 8) // more value roles than value fields
  val cfgNoLiveSlot     = nValid === 0.U                                 // every slot masked to the identity
  // A transport format this instance was never elaborated for. On a multi-format build it would fall through the
  // repack MuxLookup and reinterpret the beat in a different number system; on a single-format build the datapath
  // ignores `fmt` entirely, and this is what catches a CSR word WRITTEN BEFORE THE FIELD EXISTED -- whose zero
  // bits name FP16, not the FP32 the word was computed for.
  val cfgBadFmt         =
    if (multiFmt) !VecInit(supported.map { case (c, _, _) => fmt === c.U }).asUInt.orR
    else fmt =/= supported.head._1.U
  jct_cfgerr_o := cfgSigmaSaturated || cfgRolesOverflow || cfgNoLiveSlot || cfgBadFmt

  // `nValid = 0` masks EVERY slot to the identity, so the fold emits an identity partial: finite, format-legal
  // and numerically empty. Legal, but indistinguishable from a stale CSR. Simulation-only, so it cannot break a
  // real transfer; `jct_cfgerr_o` above is what reports it to software.
  assert(!(fire && nValid === 0.U), "MonoidJunction: fired with nValid = 0 -- every slot is masked to the identity")

  // ---- the operator: nLanes identical value lanes, nKey identical key front ends -----------------------------
  //
  // No layout mux, no slot-0 special case, no repack. Lane `l` reads `a(l)` and `b(l)` and writes `y(l)`: NO
  // DATA OPERAND EVER CROSSES A LANE. The only cross-lane wires are the twist and the winner-swap bit, which
  // one-exp-per-key forces to exist anyway.
  //
  // INGRESS. The beat is sliced by `fmt` and widened EXACTLY to FP32 -- every sub-FP32 format has <= 23 mantissa
  // and <= 8 exponent bits, so the convert is a pure bit-manipulation with no rounding and no pipeline stage.
  // Everything past this point is the FP32 operator, unchanged. A format WIDER than `elemWidth` simply lights up
  // fewer lanes: the rest park at zero and are out of range for every legal geometry, so they retire zero and are
  // dropped by the repack.
  def widenedLanes(beat: UInt, w: Int, t: FpType): IndexedSeq[UInt] = {
    val n = junctionParam.dataWidth / w
    (0 until nLanes).map { i =>
      if (i < n) { val slice = beat(w * i + w - 1, w * i); if (t == FP32) slice else FpHelpers.widen(slice, t) }
      else F32_ZERO
    }
  }
  def laneMux(beat: UInt): IndexedSeq[UInt] =
    if (!multiFmt) widenedLanes(beat, supported.head._2, supported.head._3)
    else {
      val perFmt = supported.map { case (c, w, t) => c -> widenedLanes(beat, w, t) }
      (0 until nLanes).map(i => MuxLookup(fmt, perFmt.head._2(i))(perFmt.map { case (c, v) => c.U -> v(i) }))
    }
  val aLanes = laneMux(aQ.io.deq.bits)
  val bLanes = laneMux(bQ.io.deq.bits)

  // SLOT-CLASS SHARING. `S <= nKey` and `nKey` is a power of two, so `S-1` only ever masks bits that `nKey-1`
  // already masks: `l & (S-1) == (l & (nKey-1)) & (S-1)`. Lane `l` and lane `l & (nKey-1)` are therefore ALWAYS
  // in the same slot, whatever sigma is, so everything that depends on the slot is built once per class.
  def cls(l: Int): Int = l & (nKey - 1)
  val cLive = (0 until nKey).map(c => slotOf(c, sigma) < nValid)

  /** the broadcast tap for slot class `c`: front end `c & (S-1)` across the legal sigmas. NESTED PREFIX MASKS, so most
    * classes have a single candidate and collapse to a plain wire -- the whole reason for field-major.
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
  val lIsKey    = (0 until nLanes).map(l => if (l >= (1 << SIGMA_MAX)) false.B else lField(l) === 0.U)
  // In the scan the key lane is scaled too, because `k_A * k_B` is the same `los * scale` the value lanes use.
  val lUseAlpha = (0 until nLanes).map(l => (lIsKey(l) && keyMul) || (lField(l) >= 1.U && lField(l) <= geom.nExp))
  // No `&& lInRange(l)`: an out-of-range lane retires zero from the outer select in `valueLane` regardless.
  val lIsSel    = (0 until nLanes).map(l => lField(l) > (geom.nExp +& geom.nAdd))
  // ... and it therefore stops SELECTING a winner: in the scan it computes one.
  val lSelOrKey = (0 until nLanes).map(l => (lIsKey(l) && !keyMul) || lIsSel(l))
  val lLive     = (0 until nLanes).map(l => cLive(cls(l)))
  // O3 on the INPUT, per lane: a dead slot is fed its field's identity on BOTH sides, so it contributes the
  // identity and the emitted beat stays a legal input partial at any downstream nValid.
  // The scan's key identity is 1.0, not the losing extreme: a dead slot must be NEUTRAL under multiplication.
  //
  // O3 strengthened, AND THE TRAP A FORMAT CHANGE OPENS. The key identity does two jobs that used to coincide:
  // it MASKS the internal operands, which are FP32 whatever `fmt` says, and it is the identity partial the
  // chassis PUBLISHES, which has to be in the TRANSPORT format or a padded partial is read as garbage by the
  // next hop. The two must diverge the moment the transport is not FP32 -- and not only in the published beat:
  // the FP32 extreme -3.4e38 is NOT REPRESENTABLE in FP16, so masking with it and narrowing on the way out
  // returns whatever the narrow does on overflow. The mask is therefore the TRANSPORT identity WIDENED, which
  // survives the round trip exactly and still loses every comparison.
  def keyIdOf(w: Int, t: FpType): UInt =
    if (t == FP32) keyIdentity(geom.keyPol, keyMul)
    else {
      val (ew, mw) = (t.expWidth, t.sigWidth)
      val bias     = (BigInt(1) << (ew - 1)) - 1
      val maxFin   = (((BigInt(1) << ew) - 2) << mw) | ((BigInt(1) << mw) - 1) // largest finite, sign 0
      // the same three values `keyIdentity` names in FP32: 1.0 for the scan, +maxFin for the min-monoid and
      // -maxFin for the max-monoid. In FP16 that is 0x3C00 / 0x7BFF / 0xFBFF.
      Mux(keyMul, (bias << mw).U(w.W), Mux(geom.keyPol, maxFin.U(w.W), (maxFin | (BigInt(1) << (w - 1))).U(w.W)))
    }
  def padKeyOf(w: Int, t: FpType): UInt =
    if (t == FP32) keyIdOf(w, t) else FpHelpers.widen(keyIdOf(w, t), t)
  val padKey    =
    if (!multiFmt) padKeyOf(supported.head._2, supported.head._3)
    else {
      val per = supported.map { case (c, w, t) => c.U -> padKeyOf(w, t) }
      MuxLookup(fmt, padKeyOf(supported.head._2, supported.head._3))(per)
    }
  val lPad      = (0 until nLanes).map(l => Mux(lIsKey(l), padKey, F32_ZERO))
  def identityOf(w: Int, t: FpType): UInt = {
    val n = junctionParam.dataWidth / w
    val k = keyIdOf(w, t)
    Cat((0 until n).map(i => Mux(lIsKey(i), k, 0.U(w.W))).reverse)
  }
  jct_identity_o :=
    (if (!multiFmt) identityOf(supported.head._2, supported.head._3)
     else {
       val idPerFmt = supported.map { case (c, w, t) => c -> identityOf(w, t) }
       MuxLookup(fmt, idPerFmt.head._2)(idPerFmt.map { case (c, v) => c.U -> v })
     })

  val amL = (0 until nLanes).map(l => Mux(lLive(l), aLanes(l), lPad(l)))
  val bmL = (0 until nLanes).map(l => Mux(lLive(l), bLanes(l), lPad(l)))

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
  val laneOut = (0 until nLanes).map { l =>
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
  }

  // ---- egress: narrow back to the transport format and repack the beat ---------------------------------------
  // O2, format closure, under a format change. The beat that leaves must be a legal beat to bring back IN, so it
  // is narrowed with exactly the rounding the next hop's widen will read. A 4-hop chain is therefore
  // `narrow(fold(widen(.)))` three times: deterministic for a fixed route, but NOT associative in the last bit --
  // a different chain ORDER can move the LSB. Worth knowing before re-deriving a golden.
  def packedOf(w: Int, t: FpType): UInt = {
    val n = junctionParam.dataWidth / w
    Cat((0 until n).map { i =>
      if (t == FP32) ShiftRegister(laneOut(i), narrowPipe) else FpHelpers.narrow(laneOut(i), t, narrowPipe)
    }.reverse)
  }
  val outBeat =
    if (!multiFmt) packedOf(supported.head._2, supported.head._3)
    else {
      val outPacked = supported.map { case (c, w, t) => c -> packedOf(w, t) }
      MuxLookup(fmt, outPacked.head._2)(outPacked.map { case (c, p) => c.U -> p })
    }

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
  when(jct_start_i) { credit := Qdepth.U }.otherwise {
    when(fire =/= deq) { credit := Mux(fire, credit - 1.U, credit + 1.U) }
  }

  jct_busy_o := (credit =/= Qdepth.U) || aQ.io.deq.valid || bQ.io.deq.valid
}
