package snax.DataPathJunction

import chisel3._
import chisel3.util._

import fp_native._
import fp_unit._

import snax.DataPathExtension.FpHelpers
import snax.DataPathExtension.FpHelpers._

/** ============================================================================================================
  * `ElementwiseJunction` -- the linear two-stream fold.
  * ============================================================================================================
  *
  * A 2-input, per-element reduction (ADD / MUL / MAX / MIN) on the Junction ABI: same arity, same chain position
  * and same per-beat throughput as the monoid operator, but per element and with no coupling between lanes.
  *
  * This is one of the two operators shipped on the junction socket. It satisfies the same four obligations
  * `MonoidJunction` does -- declared latency, format closure, identity tolerance, no state between pairs --
  * with an entirely different internal shape, which is the point: the socket constrains behaviour, not structure.
  *
  * ADD / MUL / MAX / MIN commute with routing, so a chain of these nodes computes the same result regardless of
  * how the reduction is ordered along the route. The nonlinear family in `MonoidCombine` -- the online-softmax
  * (m, l) merge, the flash-attention (m, l, O) triple, the exp-weighted moment bank -- is NOT expressible this
  * way at any element granularity, because it needs a shared rescale alpha = exp(m_loser - m*) derived from the
  * operand pair itself.
  *
  * ---- TWO GRIDS ----
  *
  * FLOATING POINT. `elemWidth` (elaboration) sets the lane count `lanes = dataWidth / elemWidth`; the math is
  * always FP32, so `fmt` (runtime) selects any transport format at least `elemWidth` wide and the beat is sliced
  * accordingly. Every supported format runs at ONE BEAT PER CYCLE -- narrower formats simply light up more lanes.
  * At the default `elemWidth = 16` that is 32 FP32 lanes covering FP16/BF16 at full beat rate, and FP32 on lanes
  * 0..15.
  *
  * INTEGER. `intWidths` (elaboration) selects which of INT8 / INT16 / INT32 are built. Integers are combined at
  * their NATIVE width -- no widen, no narrow, no rounding -- so an integer format is exact and the beat is
  * `dataWidth / w` lanes wide whatever `elemWidth` is. Two properties follow, and both are the point of having
  * this grid at all:
  *
  *   - `ADD` and `MUL` WRAP, i.e. they are the group and ring operations of Z_2^w. Wrapping addition is
  *     EXACTLY associative and exactly commutative, so a chain of these nodes returns one answer whatever the
  *     route -- which floating-point addition, on the grid above, does not.
  *   - `MAX` / `MIN` are SIGNED comparisons.
  *
  * That makes the integer grid the merge rule of the standard mergeable summaries: a HyperLogLog merges by
  * per-register `MAX` over INT8, a Count-Min sketch by elementwise `ADD` over INT32, a G-Counter by per-replica
  * `MAX`, and a tropical `(min, +)` distance vector by elementwise `MIN`.
  *
  * ---- CSR(0) ----
  * {{{
  *   [3:0]   op    0 = ADD, 1 = MUL, 2 = MAX, 3 = MIN
  *   [6:4]   fmt   0 = FP16, 1 = BF16, 2 = FP8, 3 = FP32   (must be >= elemWidth wide)
  *                 4 = INT8, 5 = INT16, 6 = INT32          (must be listed in intWidths)
  * }}}
  */
class HasElementwiseJunction(
  dataWidth:   Int = 512,
  elemWidth:   Int = 16,
  fpPipe:      Int = 1,
  intWidths:   Seq[Int] = Seq(8, 16, 32),
  skidDepth:   Int = 4,
  starveLimit: Int = 4096
) extends HasDataPathJunction {
  implicit val junctionParam: JunctionParam =
    new JunctionParam(
      moduleName  = "ElementwiseJunction",
      userCsrNum  = 1,
      dataWidth   = dataWidth,
      starveLimit = starveLimit
    )

  def instantiate(clusterName: String): ElementwiseJunction =
    Module(new ElementwiseJunction(elemWidth = elemWidth, fpPipe = fpPipe, intWidths = intWidths,
                                   skidDepth = skidDepth) {
      override def desiredName = clusterName + namePostfix
    })
}

object ElementwiseJunction {
  val OP_ADD = 0
  val OP_MUL = 1
  val OP_MAX = 2
  val OP_MIN = 3

  /** runtime transport formats, reusing the codes the SIMD extensions already use (FpHelpers.FMT_*) */
  val FMT_FP32 = 3
  val formats: Seq[(Int, Int, FpType)] = Seq(
    (FpHelpers.FMT_FP8, 8, FP8),
    (FpHelpers.FMT_FP16, 16, FP16),
    (FpHelpers.FMT_BF16, 16, BF16),
    (FMT_FP32, 32, FP32)
  )

  /** integer transport formats, combined at their native width */
  val FMT_INT8  = 4
  val FMT_INT16 = 5
  val FMT_INT32 = 6
  val intFormats: Seq[(Int, Int)] = Seq((FMT_INT8, 8), (FMT_INT16, 16), (FMT_INT32, 32))
}

class ElementwiseJunction(
  elemWidth:     Int = 16,
  fpPipe:        Int = 1,
  intWidths:     Seq[Int] = Seq(8, 16, 32),
  skidDepth:     Int = 4
)(implicit
  junctionParam: JunctionParam
) extends DataPathJunction {

  import ElementwiseJunction._

  require(isPow2(elemWidth) && elemWidth >= 8 && elemWidth <= 32, "ElementwiseJunction: elemWidth must be 8, 16 or 32")
  val lanes = junctionParam.dataWidth / elemWidth
  // the transport formats this grid can carry at one beat per cycle (anything at least as wide as a lane slot)
  val supported = formats.filter(_._2 >= elemWidth)
  require(supported.nonEmpty, s"ElementwiseJunction: no transport format is >= elemWidth=$elemWidth")
  // The integer grid is independent of `elemWidth`: it never touches the FP32 lanes, so an INT8 merge runs on
  // dataWidth/8 native lanes whatever the float grid was elaborated for.
  val supportedInt = intFormats.filter { case (_, w) => intWidths.contains(w) }
  require(intWidths.forall(w => Seq(8, 16, 32).contains(w)),
          s"ElementwiseJunction: intWidths must be drawn from 8, 16, 32 (got $intWidths)")

  val FP32_ONE  = "h3F800000".U(32.W)
  val FP32_ZERO = 0.U(32.W)

  val opcode = jct_csr_i(0)(3, 0)
  val fmt    = jct_csr_i(0)(6, 4)

  // ---- skid FIFOs + credit: each beat pair traverses the lane grid independently, so first-output latency is
  // independent of payload size (cut-through, not store-and-forward)
  val aQ = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = skidDepth, pipe = true, flow = false))
  val bQ = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = skidDepth, pipe = true, flow = false))
  aQ.io.enq <> jct_a_i
  bQ.io.enq <> jct_b_i

  val lat    = 2 * fpPipe // FMA depth + narrow depth (widen is a combinational bit-manipulation)
  val Qdepth = scala.math.max(2, lat + 4)
  val outQ   = Module(new Queue(UInt(junctionParam.dataWidth.W), entries = Qdepth))
  val credit = RegInit(Qdepth.U(log2Ceil(Qdepth + 1).W))

  // O5: an `fmt` naming a format this instance was not elaborated for would fall through the repack MuxLookup
  // to the first arm and silently reinterpret the beat in a different number system. That is the single most
  // dangerous word this operator can be given, and it is the reason the port exists.
  val builtFmts = (supported.map(_._1) ++ supportedInt.map(_._1)).distinct
  jct_cfgerr_o := (opcode > OP_MIN.U) || !VecInit(builtFmts.map(c => fmt === c.U)).asUInt.orR

  val fire = aQ.io.deq.valid && bQ.io.deq.valid && (credit =/= 0.U) && !jct_start_i
  aQ.io.deq.ready := fire
  bQ.io.deq.ready := fire

  // ---- runtime slice + exact widen to FP32 --------------------------------------------------------------
  // For each supported format, slice the beat into dataWidth/w elements and widen. Lanes beyond a format's
  // element count are parked at zero, so a wide format simply uses fewer lanes at the same beat rate.
  def widenedLanes(beat: UInt, w: Int, t: FpType): IndexedSeq[UInt] = {
    val n = junctionParam.dataWidth / w
    (0 until lanes).map { i =>
      if (i < n) {
        val slice = beat(w * i + w - 1, w * i)
        if (t == FP32) slice else FpHelpers.widen(slice, t)
      } else FP32_ZERO
    }
  }
  def laneMux(beat: UInt): IndexedSeq[UInt] = {
    val perFmt = supported.map { case (code, w, t) => code -> widenedLanes(beat, w, t) }
    (0 until lanes).map { i =>
      MuxLookup(fmt, perFmt.head._2(i))(perFmt.map { case (code, v) => code.U -> v(i) })
    }
  }

  // The float grid and the integer grid are MUTUALLY EXCLUSIVE -- one `fmt` selects one of them, and the other's
  // result is discarded at the output multiplexer. Left ungated, both compute on every beat, so half this
  // operator's arithmetic switches for a result nothing reads. Holding the idle grid's operands still costs one
  // 512-bit mask per operand per grid and stops that entirely.
  //
  // An `fmt` naming no built format falls to the float grid, which is where the output multiplexer's default arm
  // sends it too, so the gating does not change what an invalid word produces. O5 reports it either way.
  val intArmed =
    if (supportedInt.isEmpty) false.B
    else VecInit(supportedInt.map { case (c, _) => fmt === c.U }).asUInt.orR
  val aFp  = Mux(intArmed, 0.U, aQ.io.deq.bits)
  val bFp  = Mux(intArmed, 0.U, bQ.io.deq.bits)
  val aInt = Mux(intArmed, aQ.io.deq.bits, 0.U)
  val bInt = Mux(intArmed, bQ.io.deq.bits, 0.U)

  val aW = laneMux(aFp)
  val bW = laneMux(bFp)

  // ---- the 2-input reduction, one unit per lane ---------------------------------------------------------
  // ADD and MUL share ONE fused FMA (ADD = a*1 + b, MUL = a*b + 0), so a lane costs one FMA plus two operand
  // muxes. MAX/MIN are pure comparison logic (finite operands), aligned to the FMA's depth so the lane retires in
  // a single fixed latency regardless of op.
  val isMul   = opcode === OP_MUL.U
  val isMax   = opcode === OP_MAX.U
  val isMin   = opcode === OP_MIN.U
  val isMinMax = isMax || isMin

  val res32 = (0 until lanes).map { i =>
    val fmaB = Mux(isMul, bW(i), FP32_ONE)
    val fmaC = Mux(isMul, FP32_ZERO, bW(i))
    val fma  = Module(new FpFma(FP32, FP32, FP32, fpPipe))
    fma.io.in_a := aW(i)
    fma.io.in_b := fmaB
    fma.io.in_c := fmaC
    val mm = ShiftRegister(Mux(isMax, fp32max(aW(i), bW(i)), fp32min(aW(i), bW(i))), fpPipe)
    Mux(isMinMax, mm, fma.io.out)
  }

  // ---- narrow back to the transport format and repack the beat ------------------------------------------
  def packed(w: Int, t: FpType): UInt = {
    val n     = junctionParam.dataWidth / w
    val elems = (0 until n).map { i =>
      if (t == FP32) ShiftRegister(res32(i), fpPipe) else FpHelpers.narrow(res32(i), t, fpPipe)
    }
    Cat(elems.reverse)
  }
  // ---- the integer grid ---------------------------------------------------------------------------------
  // Native width, so there is nothing to widen and nothing to round: the result of a lane IS the transport
  // element. ADD and MUL wrap (Z_2^w), MAX and MIN are signed. Delay-matched to `lat` so the operator retires at
  // one published latency whatever `fmt` selects -- O1 must not depend on a runtime field.
  def packedInt(w: Int): UInt = {
    val n = junctionParam.dataWidth / w
    val elems = (0 until n).map { i =>
      val a   = aInt(w * i + w - 1, w * i).asSInt
      val b   = bInt(w * i + w - 1, w * i).asSInt
      val add = (a + b).asUInt                 // wraps: exactly associative, exactly commutative
      val mul = ((a * b).asUInt)(w - 1, 0)     // low half, wraps
      val mx  = Mux(a > b, a, b).asUInt
      val mn  = Mux(a < b, a, b).asUInt
      ShiftRegister(MuxLookup(opcode, add)(
        Seq(OP_ADD.U -> add, OP_MUL.U -> mul, OP_MAX.U -> mx, OP_MIN.U -> mn)
      ), lat)
    }
    Cat(elems.reverse)
  }

  // ---- O3 strengthened: the identity beat, which is a function of BOTH the op and the format --------------
  // ADD wants zero, MUL wants one, MAX wants the format's most-negative value and MIN its most-positive. None of
  // those is a constant the chassis could know, because "one" and "the most negative value" are different bit
  // patterns in every format this operator carries.
  def idElem(w: Int, t: FpType, isInt: Boolean): UInt = {
    val (one, lo, hi) =
      if (isInt) (BigInt(1), BigInt(1) << (w - 1), (BigInt(1) << (w - 1)) - 1) // two's complement
      else {
        val (ew, mw) = (t.expWidth, t.sigWidth)
        val bias     = (BigInt(1) << (ew - 1)) - 1
        val maxFin   = (((BigInt(1) << ew) - 2) << mw) | ((BigInt(1) << mw) - 1) // largest finite, sign 0
        (bias << mw, maxFin | (BigInt(1) << (w - 1)), maxFin)                    // 1.0, -maxFin, +maxFin
      }
    MuxLookup(opcode, 0.U(w.W))(Seq(
      OP_ADD.U -> 0.U(w.W), OP_MUL.U -> one.U(w.W), OP_MAX.U -> lo.U(w.W), OP_MIN.U -> hi.U(w.W)
    ))
  }
  val idPerFmt = supported.map { case (code, w, t) =>
    code -> Fill(junctionParam.dataWidth / w, idElem(w, t, isInt = false))
  } ++ supportedInt.map { case (code, w) =>
    code -> Fill(junctionParam.dataWidth / w, idElem(w, FP32, isInt = true))
  }
  jct_identity_o := MuxLookup(fmt, idPerFmt.head._2)(idPerFmt.map { case (c, v) => c.U -> v })

  // Elaborate each format's repack ONCE; the default arm reuses the first rather than building a second,
  // unreachable copy of it.
  val outPacked = supported.map { case (code, w, t) => code -> packed(w, t) } ++
    supportedInt.map { case (code, w) => code -> packedInt(w) }
  val outBeat   = MuxLookup(fmt, outPacked.head._2)(outPacked.map { case (code, p) => code.U -> p })

  def clrPipe(in: Bool, n: Int): Bool =
    if (n <= 0) in
    else {
      val r = RegInit(VecInit(Seq.fill(n)(false.B)))
      r(0) := Mux(jct_start_i, false.B, in)
      for (i <- 1 until n) r(i) := Mux(jct_start_i, false.B, r(i - 1))
      r(n - 1)
    }
  val retire = clrPipe(fire, lat)

  outQ.io.enq.valid := retire && !jct_start_i
  outQ.io.enq.bits  := outBeat
  assert(!outQ.io.enq.valid || outQ.io.enq.ready, "ElementwiseJunction: output queue overflow (credit bug)")
  outQ.io.deq.ready := jct_data_o.ready
  jct_data_o.valid  := outQ.io.deq.valid
  jct_data_o.bits   := outQ.io.deq.bits

  val deq = outQ.io.deq.fire
  when(jct_start_i) { credit := Qdepth.U }
    .otherwise { when(fire =/= deq) { credit := Mux(fire, credit - 1.U, credit + 1.U) } }

  jct_busy_o := (credit =/= Qdepth.U) || aQ.io.deq.valid || bQ.io.deq.valid
}
