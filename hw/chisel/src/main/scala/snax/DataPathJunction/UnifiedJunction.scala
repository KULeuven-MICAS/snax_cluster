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
  * `Fmax + 2*(pairSlots-1)` -- instead of the SUM of the two, which is what two separate junctions cost.
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
  pairSlots:   Int = 0, // 0 = auto = dataWidth / 64
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
        pairSlotsParam = pairSlots
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
  pairSlotsParam: Int = 0
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
  val pairSlots = if (pairSlotsParam <= 0) junctionParam.dataWidth / (2 * accWidth) else pairSlotsParam
  val Fmax      = 2 + dHead                                      // fields in one (m, l, O) partial (ATTN)
  require(Fmax <= nLanes, s"UnifiedJunction: 2+dHead ($Fmax) must fit one beat ($nLanes lanes)")
  require(2 * pairSlots <= nLanes, s"UnifiedJunction: 2*pairSlots (${2 * pairSlots}) must fit one beat ($nLanes)")
  val supported = formats.filter(_._2 >= elemWidth)
  require(supported.nonEmpty, s"UnifiedJunction: no transport format is >= elemWidth=$elemWidth")

  // ---- CSR decode ------------------------------------------------------------------------------------------
  val linear      = jct_csr_i(0)(OPCLASS_BIT) === CLASS_ELEMENTWISE.U
  val nValid      = jct_csr_i(0)(7, 0)
  val combineMode = jct_csr_i(0)(15, 13)
  val opcode      = jct_csr_i(0)(3, 0)
  val fmt         = jct_csr_i(0)(6, 4)
  val single      = isSingle(combineMode)

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

  // ---- THE SHARED FP32 FMA POOL ----------------------------------------------------------------------------
  // Every value lane of either class is `x*y + z`. Both classes drive the pool's operand ports and the armed
  // class wins the mux, so the multipliers are built ONCE instead of once per class.
  val nMonFma = Fmax + 2 * (pairSlots - 1) // slot 0 is elaborated at Fmax fields; the paired slots at 2 each
  val nPool   = scala.math.max(lanesEw, nMonFma)

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

  /** hand the combine bank the next free pool lane (elaboration-time allocation, deterministic order) */
  private var mnNext = 0
  private def monFma(x: UInt, y: UInt, z: UInt): UInt = {
    val i = mnNext
    mnNext += 1
    require(i < nPool, s"UnifiedJunction: the combine bank claimed more than $nPool FMA lanes")
    mnX(i) := x; mnY(i) := y; mnZ(i) := z
    poolOut(i)
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
  val ewBeat = MuxLookup(fmt, packed(supported.head._2, supported.head._3))(
    supported.map { case (code, w, t) => code.U -> packed(w, t) }
  )

  // ---- MONOID class: the combine bank, drawing its value lanes from the pool --------------------------------
  val aLanes = aQ.io.deq.bits.asTypeOf(Vec(nLanes, UInt(accWidth.W)))
  val bLanes = bQ.io.deq.bits.asTypeOf(Vec(nLanes, UInt(accWidth.W)))

  // Slot k >= nValid is fed the monoid identity on BOTH sides, so it contributes the identity to the output.
  def maskA(k: Int, f: Int): UInt =
    Mux(k.U < nValid, if (f == 0) aLanes(k) else aLanes(pairSlots + k), identityOf(combineMode, f))
  def maskB(k: Int, f: Int): UInt =
    Mux(k.U < nValid, if (f == 0) bLanes(k) else bLanes(pairSlots + k), identityOf(combineMode, f))

  // slot 0 operands: the single-partial beat (lanes 0..Fmax-1) or the paired slot-0 partial widened with identities
  val a0 = (0 until Fmax).map(f => Mux(single, aLanes(f), if (f < 2) maskA(0, f) else identityOf(combineMode, f)))
  val b0 = (0 until Fmax).map(f => Mux(single, bLanes(f), if (f < 2) maskB(0, f) else identityOf(combineMode, f)))
  val (out0, monLat) = MonoidCombine(a0, b0, combineMode, dHead, expLutN, Some(monFma _), fpPipe)
  require(monLat == latMon, s"UnifiedJunction: combine latency $monLat does not match the accounted $latMon")

  val outPairs = (1 until pairSlots).map { k =>
    val (o, _) = MonoidCombine(Seq(maskA(k, 0), maskA(k, 1)), Seq(maskB(k, 0), maskB(k, 1)),
                               combineMode, dHead, expLutN, Some(monFma _), fpPipe)
    o
  }

  // single-partial: lay the merged (m, l, ...) across lanes 0..Fmax-1.
  // paired: slot k's (field0, field1) go back to lanes (k, pairSlots+k) -- the same layout the operands arrived
  // in, so a chain of gather nodes is closed under its own beat format.
  val singleBeat = Wire(Vec(nLanes, UInt(accWidth.W)))
  for (f <- 0 until Fmax) singleBeat(f) := out0(f)
  for (f <- Fmax until nLanes) singleBeat(f) := F32_ZERO

  val pairedBeat = Wire(Vec(nLanes, UInt(accWidth.W)))
  for (l <- 0 until nLanes) pairedBeat(l) := F32_ZERO
  pairedBeat(0)         := out0(0)
  pairedBeat(pairSlots) := out0(1)
  for (k <- 1 until pairSlots) {
    pairedBeat(k)             := outPairs(k - 1)(0)
    pairedBeat(pairSlots + k) := outPairs(k - 1)(1)
  }
  val monBeat = Mux(single, singleBeat.asUInt, pairedBeat.asUInt)

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
