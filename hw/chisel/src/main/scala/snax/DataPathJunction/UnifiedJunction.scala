package snax.DataPathJunction

import chisel3._
import chisel3.util._

import fp_native._
import fp_unit._

import snax.DataPathExtension.FpHelpers
import snax.DataPathExtension.FpHelpers._

/** ============================================================================================================
  * `UnifiedJunction` -- ONE 2->1 netlist for the LINEAR and the NONLINEAR fold family, over one FMA pool.
  * ============================================================================================================
  *
  * `ElementwiseJunction` (per-element ADD/MUL/MAX/MIN at runtime FP16/BF16/FP8/FP32) and `MonoidJunction` (the
  * structured serving-reduction family: online-softmax (m, l), the flash-attention triple, exp-weighted moments,
  * argmax) are two operator classes with the SAME junction ABI, the same skid/credit chassis and the same
  * cut-through join. What actually differs is the shape of the value lane -- and both shapes are an FP32 FMA:
  * {{{
  *   linear ADD   : a*1 + b            linear MUL : a*b + 0
  *   monoid SUM   : loser*1   + winner  (= a + b)
  *   monoid RSUM  : loser*alpha + winner (alpha = exp(m_loser - m*), the whole nonlinear difference)
  * }}}
  * So this module builds ONE pool of `nPool` FP32 FMA lanes and lets whichever mode is armed drive it. The pool
  * is sized by the wider claimant -- the linear grid needs `dataWidth/elemWidth` lanes, the combine bank needs
  * one per FP32 beat lane -- instead of the SUM of the two, which is what two separate junctions cost.
  *
  * What stays private to a class is only what has no counterpart: the monoid side's key front-end (max, delta,
  * ONE exp LUT per combine node) and the linear side's runtime widen/narrow grid and MAX/MIN comparators.
  *
  * ---- CSR(0) ---- (still ONE user CSR, so the inter-cluster / D2D cfg serdes is untouched)
  * {{{
  *   [16]     opClass      0 = MONOID (structured), 1 = ELEMENTWISE (per-element linear)
  *
  *   MONOID class -- byte-identical to `MonoidJunction`'s encoding:
  *     [7:0]    nValid     live partials in the beat; slots >= nValid get the monoid identity
  *     [15:13]  combineMode  SUM | MOMENT | ATTN | MAXPOOL | ARGMAX | MOMENT2
  *
  *   ELEMENTWISE class -- byte-identical to `ElementwiseJunction`'s encoding:
  *     [3:0]    op         0 = ADD, 1 = MUL, 2 = MAX, 3 = MIN
  *     [6:4]    fmt        0 = FP16, 1 = BF16, 2 = FP8, 3 = FP32   (must be >= elemWidth wide)
  * }}}
  * The two decodes are disjoint and bit [16] is unused by both predecessors, so an existing MONOID CSR word is
  * already a valid word here and an existing ELEMENTWISE word needs only that one bit set. Bits [12:8] are left
  * unread -- a junction is stateless in the operand pair and has no accumulator slot to name.
  *
  * ---- LATENCY ----
  *
  * The two classes retire at different fixed depths (`2*fpPipe` for the linear grid: FMA then narrow; the exp
  * LUT plus the FMA for the monoid bank), so both retire pipelines are built and the armed class selects one.
  * `combineMode` / `op` are held for a whole stream, so the selection never changes mid-flight.
  */
class HasUnifiedJunction(
  dataWidth:   Int = 512,
  elemWidth:   Int = 16,
  fpPipe:      Int = 1,
  dHead:       Int = 8,
  expLutN:     Int = 128,
  skidDepth:   Int = 4,
  starveLimit: Int = 4096
) extends HasDataPathJunction {
  implicit val junctionParam: JunctionParam =
    new JunctionParam(
      moduleName  = "UnifiedJunction",
      userCsrNum  = 1,
      dataWidth   = dataWidth,
      starveLimit = starveLimit
    )

  def instantiate(clusterName: String): UnifiedJunction =
    Module(
      new UnifiedJunction(
        elemWidth      = elemWidth,
        fpPipe         = fpPipe,
        dHead          = dHead,
        expLutN        = expLutN,
        skidDepth      = skidDepth,
      ) {
        override def desiredName = clusterName + namePostfix
      }
    )
}

object UnifiedJunction {
  val CLASS_MONOID      = 0
  val CLASS_ELEMENTWISE = 1
  val OPCLASS_BIT       = 16
}

class UnifiedJunction(
  elemWidth:      Int = 16,
  fpPipe:         Int = 1,
  dHead:          Int = 8,
  expLutN:        Int = 128,
  skidDepth:      Int = 4,
)(implicit
  junctionParam:  JunctionParam
) extends DataPathJunction {

  import ElementwiseJunction._
  import MonoidCombine._
  import UnifiedJunction._

  // ---- geometry --------------------------------------------------------------------------------------------
  require(isPow2(elemWidth) && elemWidth >= 8 && elemWidth <= 32, "UnifiedJunction: elemWidth must be 8, 16 or 32")
  val accWidth  = 32
  val lanesEw   = junctionParam.dataWidth / elemWidth            // linear grid width (32 at elemWidth=16)
  val nLanes    = junctionParam.dataWidth / accWidth             // FP32 lanes in one beat (16 at 512-bit)
  val nKey      = nLanes / 2                                     // key front ends; see the theorem below
  require(2 + dHead <= nLanes, s"UnifiedJunction: 2+dHead (${2 + dHead}) must fit one beat ($nLanes lanes)")
  // "eight key front ends is exactly sufficient" is a THEOREM, not a coincidence of dataWidth/64: any partial
  // with a key has F >= 2, which forces sigma <= SIGMA_MAX, which forces S <= nLanes/2 -- so a key lane can
  // never leave the low half of the beat, for ANY n and ANY dHead.
  require((1 << MonoidCombine.SIGMA_MAX) <= nKey,
          s"UnifiedJunction: nKey ($nKey) must cover the widest legal S (${1 << MonoidCombine.SIGMA_MAX})")
  val supported = formats.filter(_._2 >= elemWidth)
  require(supported.nonEmpty, s"UnifiedJunction: no transport format is >= elemWidth=$elemWidth")

  // ---- CSR decode ------------------------------------------------------------------------------------------
  val linear      = jct_csr_i(0)(OPCLASS_BIT) === CLASS_ELEMENTWISE.U
  val nValid      = jct_csr_i(0)(7, 0)
  val combineMode = jct_csr_i(0)(15, 13)
  val opcode      = jct_csr_i(0)(3, 0)
  val fmt         = jct_csr_i(0)(6, 4)
  // The geometry IS the configuration: (hasKey, F, sigma, nExp, nAdd). There is no layout predicate and no
  // mode-shaped special case anywhere below -- `combineMode` is read exactly once, here.
  val geom  = geomOf(jct_csr_i(0), dHead)
  val sigma = geom.sigma

  // ---- elastic (skid) FIFOs on both operand streams ---------------------------------------------------------
  // Depth >= 1 is required for CORRECTNESS (the join must not be a combinational loop between the two producers);
  // depth >= the link round-trip buys THROUGHPUT, since the two streams arrive with a skew set by the remote hop.
  val aQ = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = skidDepth, pipe = true, flow = false))
  val bQ = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = skidDepth, pipe = true, flow = false))
  aQ.io.enq <> jct_a_i
  bQ.io.enq <> jct_b_i

  val latEw   = 2 * fpPipe                                 // FMA depth + narrow depth (widen is combinational)
  val latMon  = MonoidCombine.latency + fpPipe             // exp LUT depth + the pooled FMA's depth
  val Qdepth  = scala.math.max(2, scala.math.max(latEw, latMon) + 4)
  val outQ    = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = Qdepth))
  val credit  = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  // A pair fires whenever BOTH operands are present and the output pipeline has room: one pair per cycle. Neither
  // class holds state between pairs, so nothing serializes the issue.
  val fire = aQ.io.deq.valid && bQ.io.deq.valid && (credit =/= 0.U) && !jct_start_i
  aQ.io.deq.ready := fire
  bQ.io.deq.ready := fire

  // `nValid = 0` masks EVERY slot to the identity, so the fold emits an identity partial: finite, format-legal,
  // and numerically empty. It is a legal configuration, but it is indistinguishable from a stale CSR -- and the
  // single-partial geometries used to ignore `nValid` altogether, so a word that worked before this rewrite can
  // now silently return nothing on the flagship operator. Simulation-only, so it cannot break a real transfer.
  assert(!(fire && !linear && nValid === 0.U),
         "UnifiedJunction: the monoid fold fired with nValid = 0 -- every slot is masked to the identity")

  // ---- THE SHARED FP32 FMA POOL ----------------------------------------------------------------------------
  // Every value lane of either class is `x*y + z`. Both classes drive the pool's operand ports and the armed
  // class wins the mux, so the multipliers are built ONCE instead of once per class.
  // The monoid class claims exactly one lane per FP32 beat lane -- an identity map, no allocator.
  val nPool = scala.math.max(lanesEw, nLanes)

  val ewX = Wire(Vec(nPool, UInt(accWidth.W)))
  val ewY = Wire(Vec(nPool, UInt(accWidth.W)))
  val ewZ = Wire(Vec(nPool, UInt(accWidth.W)))
  val mnX = Wire(Vec(nPool, UInt(accWidth.W)))
  val mnY = Wire(Vec(nPool, UInt(accWidth.W)))
  val mnZ = Wire(Vec(nPool, UInt(accWidth.W)))
  for (i <- 0 until nPool) {
    ewX(i) := F32_ZERO; ewY(i) := F32_ZERO; ewZ(i) := F32_ZERO
    mnX(i) := F32_ZERO; mnY(i) := F32_ZERO; mnZ(i) := F32_ZERO
  }
  val poolOut = (0 until nPool).map { i =>
    val fma = Module(new FpFma(FP32, FP32, FP32, fpPipe))
    fma.io.in_a := Mux(linear, ewX(i), mnX(i))
    fma.io.in_b := Mux(linear, ewY(i), mnY(i))
    fma.io.in_c := Mux(linear, ewZ(i), mnZ(i))
    fma.io.out
  }

  /** monoid lane `l` drives pool lane `l` */
  private def monFma(l: Int)(x: UInt, y: UInt, z: UInt): UInt = {
    mnX(l) := x; mnY(l) := y; mnZ(l) := z
    poolOut(l)
  }

  // ---- ELEMENTWISE class: runtime slice + exact widen, one pooled lane per element --------------------------
  // For each supported format, slice the beat into dataWidth/w elements and widen. Lanes beyond a format's
  // element count are parked at zero, so a wide format simply uses fewer lanes at the same beat rate.
  def widenedLanes(beat: UInt, w: Int, t: FpType): IndexedSeq[UInt] = {
    val n = junctionParam.dataWidth / w
    (0 until lanesEw).map { i =>
      if (i < n) {
        val slice = beat(w * i + w - 1, w * i)
        if (t == FP32) slice else FpHelpers.widen(slice, t)
      } else F32_ZERO
    }
  }
  def laneMux(beat: UInt): IndexedSeq[UInt] = {
    val perFmt = supported.map { case (code, w, t) => code -> widenedLanes(beat, w, t) }
    (0 until lanesEw).map { i =>
      MuxLookup(fmt, perFmt.head._2(i))(perFmt.map { case (code, v) => code.U -> v(i) })
    }
  }

  val aW = laneMux(aQ.io.deq.bits)
  val bW = laneMux(bQ.io.deq.bits)

  // ADD and MUL are the SAME pooled FMA (ADD = a*1 + b, MUL = a*b + 0), so a lane costs two operand muxes on top
  // of the shared unit. MAX/MIN are pure comparison logic, aligned to the FMA's depth so the lane retires at one
  // fixed latency regardless of op.
  val isMul    = opcode === OP_MUL.U
  val isMax    = opcode === OP_MAX.U
  val isMin    = opcode === OP_MIN.U
  val isMinMax = isMax || isMin

  for (i <- 0 until lanesEw) {
    ewX(i) := aW(i)
    ewY(i) := Mux(isMul, bW(i), F32_ONE)
    ewZ(i) := Mux(isMul, F32_ZERO, bW(i))
  }
  val ewRes32 = (0 until lanesEw).map { i =>
    val mm = ShiftRegister(Mux(isMax, fp32max(aW(i), bW(i)), fp32min(aW(i), bW(i))), fpPipe)
    Mux(isMinMax, mm, poolOut(i))
  }

  // narrow back to the transport format and repack the beat
  def packed(w: Int, t: FpType): UInt = {
    val n     = junctionParam.dataWidth / w
    val elems = (0 until n).map { i =>
      if (t == FP32) ShiftRegister(ewRes32(i), fpPipe) else FpHelpers.narrow(ewRes32(i), t, fpPipe)
    }
    Cat(elems.reverse)
  }
  // Elaborate each format's repack ONCE and let the default arm REUSE the first one. Calling `packed` again for
  // the default builds a second, identical converter grid for that format -- 32 FP32->FP16 rounders that no
  // configuration can ever reach, which was a third of every FP adder in this block. (`laneMux` above already
  // has this right: it materialises `perFmt` once and indexes it for both the default and the arms.)
  val ewPacked = supported.map { case (code, w, t) => code -> packed(w, t) }
  val ewBeat   = MuxLookup(fmt, ewPacked.head._2)(ewPacked.map { case (code, p) => code.U -> p })

  // ---- MONOID class: one lane map, `nLanes` identical value lanes, `nKey` identical key front ends ----------
  //
  // There is no layout mux, no slot-0 special case and no repack. Lane `l` reads `a(l)` and `b(l)` and writes
  // `y(l)`: NO DATA OPERAND EVER CROSSES A LANE. The only cross-lane wires are the twist and the winner-swap
  // bit, which C2 forces to exist anyway.
  val aLanes = aQ.io.deq.bits.asTypeOf(Vec(nLanes, UInt(accWidth.W)))
  val bLanes = bQ.io.deq.bits.asTypeOf(Vec(nLanes, UInt(accWidth.W)))

  // ---- SLOT-CLASS SHARING ----------------------------------------------------------------------------------
  // `S <= nKey` (the theorem above) and `nKey` is a power of two, so `S-1` only ever masks bits that
  // `nKey-1` already masks:  l & (S-1)  ==  (l & (nKey-1)) & (S-1).
  // Lane `l` and lane `l & (nKey-1)` therefore ALWAYS sit in the same slot, whatever sigma is. Everything that
  // depends on the slot rather than the field is built ONCE PER CLASS and read by the two lanes that share it,
  // which halves the slot decode and the whole broadcast network for free.
  require(isPow2(nKey), s"UnifiedJunction: nKey ($nKey) must be a power of two for the slot-class fold")
  def cls(l: Int): Int = l & (nKey - 1)

  val cLive = (0 until nKey).map(c => slotOf(c, sigma) < nValid)

  /** the broadcast tap for slot class `c`: front end `c & (S-1)` across the legal sigmas. The candidates are
    * NESTED PREFIX MASKS, so most classes have a single candidate and collapse to a plain wire -- that is the
    * whole reason the lane map is field-major rather than slot-major.
    */
  def tapOf[T <: Data](c: Int, sel: Int => T): T = {
    val cand = (0 to SIGMA_MAX).map(s => c & ((1 << s) - 1))
    if (cand.distinct.length == 1) sel(cand.head) // no mux at all
    else MuxLookup(sigma, sel(cand.head))((0 to SIGMA_MAX).map(s => s.U -> sel(cand(s))))
  }

  // per-lane decode -- all of it stream-constant, so none of it is in a data path
  val lField    = (0 until nLanes).map(l => fieldOf(l, sigma))
  val lInRange  = (0 until nLanes).map(l => lField(l) < geom.F)
  // `sigma <= SIGMA_MAX` means `field(l) = l >> sigma >= 1` for every lane at or above `1 << SIGMA_MAX`, so no
  // such lane can EVER hold the key. That is the same theorem the key front ends are sized by; state it here
  // too, because the field decode lowers to a Vec lookup and the tool cannot recover it from the index.
  val lIsKey    = (0 until nLanes).map(l =>
    if (l >= (1 << MonoidCombine.SIGMA_MAX)) false.B else geom.hasKey && lField(l) === 0.U)
  val lUseAlpha = (0 until nLanes).map(l => geom.hasKey && lField(l) >= 1.U && lField(l) <= geom.nExp)
  // No `&& lInRange(l)` here: an out-of-range lane retires F32_ZERO from the outer select in `valueLane`
  // regardless of what this predicate says, so the term only ever ANDs a signal that is about to be ignored.
  val lIsSel    = (0 until nLanes).map(l => geom.hasKey && (lField(l) > (geom.nExp +& geom.nAdd)))
  val lLive     = (0 until nLanes).map(l => cLive(cls(l)))
  // C4 on the INPUT, per lane: a dead slot is fed its field's identity on BOTH sides, so it contributes the
  // identity and the emitted beat stays a legal input partial at any downstream nValid. The identity itself is
  // stream-constant (it flips with the key polarity), so it is selected once and fanned out.
  val padKey    = keyIdentity(geom.keyPol)
  val lPad      = (0 until nLanes).map(l => Mux(lIsKey(l), padKey, F32_ZERO))
  val amL       = (0 until nLanes).map(l => Mux(lLive(l), aLanes(l), lPad(l)))
  val bmL       = (0 until nLanes).map(l => Mux(lLive(l), bLanes(l), lPad(l)))

  // key front ends: front end `s` sits ON lane `s` and reads that lane's own masked operands (lane s IS
  // (field 0, slot s) for every sigma, which is the property field-major buys).
  val fe       = (0 until nKey).map(s => keyFrontEnd(amL(s), bmL(s), geom.hasKey, geom.keyPol, expLutN))
  val alphaTap = (0 until nKey).map(c => tapOf(c, (s: Int) => fe(s)._1))
  val swTap    = (0 until nKey).map(c => tapOf(c, (s: Int) => fe(s)._2))

  val monLanes = (0 until nLanes).map { l =>
    valueLane(
      am       = amL(l),
      bm       = bmL(l),
      sw       = swTap(cls(l)),
      alpha    = alphaTap(cls(l)),
      useAlpha = lUseAlpha(l),
      selOrKey = lIsKey(l) || lIsSel(l),
      inRange  = lInRange(l),
      fma      = monFma(l),
      fmaLat   = fpPipe
    )
  }
  val monBeat = VecInit(monLanes).asUInt

  // ---- retire ----------------------------------------------------------------------------------------------
  // Each class is a fixed-latency pipeline, so its retire pulse is the issue pulse delayed by its own depth.
  // `jct_start_i` flushes both, so a new stream never inherits a half-traversed pair from the previous one.
  def clrPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(jct_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(jct_start_i, false.B, r(i - 1))
      r(n - 1)
    }
  val retire = Mux(linear, clrPipe(fire, latEw), clrPipe(fire, latMon))

  outQ.io.enq.valid := retire && !jct_start_i
  outQ.io.enq.bits  := Mux(linear, ewBeat, monBeat)
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "UnifiedJunction: output queue overflow (credit bug)")
  outQ.io.deq.ready := jct_data_o.ready
  jct_data_o.valid  := outQ.io.deq.valid
  jct_data_o.bits   := outQ.io.deq.bits

  val deq = outQ.io.deq.fire
  when(jct_start_i) { credit := Qdepth.U }
    .otherwise { when(fire =/= deq) { credit := Mux(fire, credit - 1.U, credit + 1.U) } }

  jct_busy_o := (credit =/= Qdepth.U) || aQ.io.deq.valid || bQ.io.deq.valid
}
