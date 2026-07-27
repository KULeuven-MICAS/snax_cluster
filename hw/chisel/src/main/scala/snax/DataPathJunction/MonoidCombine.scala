package snax.DataPathJunction

import chisel3._
import chisel3.util._

import snax.DataPathExtension.FpActivation
import snax.DataPathExtension.FpHelpers._

/** The role a field of an F-field FP32 partial plays inside one monoid combine.
  *
  * OFF  out = identity (0) -- the lane is parked KEY out = max(a,b) (= m*) -- the distinguished field that produces the
  * shared rescale alpha = exp(delta) SUM out = a + b -- plain sum (norm Sx, Sx^2) RSUM out = winner + alpha * loser --
  * exponent-aligned sum (softmax l, attention O) WSEL out = the winner's payload -- the KEY's argmax carry (top-1
  * index)
  */
object MonoidRole { val OFF = 0; val KEY = 1; val SUM = 2; val RSUM = 3; val WSEL = 4 }

/** `MonoidCombine` -- the two-input combine of the arithmetic-reduction family, as a pure function with no I/O
  * opinion.
  *
  * It is kept separate from `MonoidJunction` so the arithmetic can be reasoned about, tested and reused
  * independently of the join and flow control wrapped around it.
  *
  * THE REPRESENTATION. Every combine in the family is the same op over an F-field FP32 partial, each field given a
  * role (above). SUM and RSUM are the SAME FMA (`a + b = ffma(b, 1.0, a)`), so a value field costs one FMA plus two
  * config muxes (a winner-swap and a scale in {alpha, 1}); WSEL taps the same winner-swap WITHOUT the add; KEY taps
  * the front-end max; ONE exp LUT per combine unit feeds every RSUM lane, so adding value lanes adds no exp hardware.
  *
  * `combineMode` selects the role vector:
  * {{{
  *   SUM      -> NormStat        : (SUM, SUM)                    no key, alpha idle
  *   MOMENT   -> softmax (m, l)  : (KEY, RSUM)
  *   ATTN     -> (m, l, O[dHead]): (KEY, RSUM, RSUM x dHead)     one alpha shared across l and all O lanes
  *   MAXPOOL  -> max-reduce      : (KEY, OFF...)
  *   ARGMAX   -> top-1 (max,idx) : (KEY, WSEL)
  *   MOMENT2  -> exp-wtd moments : (KEY, RSUM, RSUM, RSUM)       (m, l=Se^s, A=Se^s v, B=Se^s v^2)
  * }}}
  */
object MonoidCombine {

  // combineMode encoding (3 bits, carried in the user CSR of whichever placement instantiates the combine)
  val MODE_SUM     = 0
  val MODE_MOMENT  = 1
  val MODE_ATTN    = 2
  val MODE_MAXPOOL = 3
  val MODE_ARGMAX  = 4
  val MODE_MOMENT2 = 5

  /** fields carried by the single-partial MOMENT2 moment bank (m, l, A, B): KEY then 3 RSUM lanes */
  val MOMENT2_RSUM_HI = 3

  val F32_ONE = "h3F800000".U(32.W) // 1.0f  (the scale for SUM / the winner factor)
  val ID_M    = "hFF7FFFFF".U(32.W) // most-negative finite fp32: loses every max, id-id delta = 0
  val F32_ZERO = 0.U(32.W)

  /** the value-path latency of one combine node (the exp LUT's pipeline depth) */
  def latency: Int = FpActivation.PipeLatency

  /** only SUM lacks a key (max + exp idle => alpha = 1) */
  def modeHasKey(mode: UInt): Bool = mode =/= MODE_SUM.U

  /** SINGLE-PARTIAL modes carry one multi-field (m, l, ...) partial per beat; the rest carry paired partials. */
  def isSingle(mode: UInt): Bool = (mode === MODE_ATTN.U) || (mode === MODE_MOMENT2.U)

  /** the monoid identity of field `f` -- KEY fields take the most-negative finite value, everything else takes 0. */
  def identityOf(mode: UInt, f: Int): UInt =
    if (f == 0) Mux(modeHasKey(mode), ID_M, F32_ZERO) else F32_ZERO

  /** role(mode, field): a runtime decode of the structured role vectors (field `f` is elaboration-time). */
  def roleOf(mode: UInt, f: Int, dHead: Int): UInt = {
    val sumRole  = if (f < 2) MonoidRole.SUM.U else MonoidRole.OFF.U
    val momRole  = if (f == 0) MonoidRole.KEY.U else if (f == 1) MonoidRole.RSUM.U else MonoidRole.OFF.U
    val attnRole = if (f == 0) MonoidRole.KEY.U else if (f <= 1 + dHead) MonoidRole.RSUM.U else MonoidRole.OFF.U
    val maxRole  = if (f == 0) MonoidRole.KEY.U else MonoidRole.OFF.U
    val argRole  = if (f == 0) MonoidRole.KEY.U else if (f == 1) MonoidRole.WSEL.U else MonoidRole.OFF.U
    val mom2Role = if (f == 0) MonoidRole.KEY.U else if (f <= MOMENT2_RSUM_HI) MonoidRole.RSUM.U else MonoidRole.OFF.U
    MuxLookup(mode, sumRole)(
      Seq(
        MODE_SUM.U     -> sumRole,
        MODE_MOMENT.U  -> momRole,
        MODE_ATTN.U    -> attnRole,
        MODE_MAXPOOL.U -> maxRole,
        MODE_ARGMAX.U  -> argRole,
        MODE_MOMENT2.U -> mom2Role
      )
    )
  }

  /** ONE configurable combine over two F-field partials. Shares one exp LUT across every RSUM lane; returns the merged
    * partial and the exp-LUT pipeline latency (the value path is aligned to it via ShiftRegister).
    *
    * Must be called from inside a Module body (it instantiates the exp LUT and the FP units at the call site).
    */
  def apply(a: Seq[UInt], b: Seq[UInt], mode: UInt, dHead: Int, expLutN: Int): (Seq[UInt], Int) = {
    require(a.length == b.length, s"MonoidCombine: operand field counts differ (${a.length} vs ${b.length})")
    val aWins  = fp32aWins(a(0), b(0)) // a.key >= b.key (only meaningful when field0 is KEY)
    val mStar  = Mux(aWins, a(0), b(0))
    val loser0 = Mux(aWins, b(0), a(0))
    val delta  = fadd(loser0, fneg32(mStar)) // loser.key - m*  (<= 0)
    val exp    = Module(new FpActivation(true, true, false, false, expLutN, 256))
    exp.io.in   := delta
    exp.io.func := false.B
    exp.io.gelu := false.B
    val lat   = FpActivation.PipeLatency
    val alpha = Mux(modeHasKey(mode), exp.io.out, F32_ONE) // no key => alpha = 1.0 => the FMA is a plain add

    val out = (0 until a.length).map { f =>
      val role  = roleOf(mode, f, dHead)
      val isKey = role === MonoidRole.KEY.U
      val isOff = role === MonoidRole.OFF.U
      val rsum  = role === MonoidRole.RSUM.U
      val wsel  = role === MonoidRole.WSEL.U
      // RSUM and WSEL both take the winner-swap (so `win` is the winner's field); SUM is commutative (no swap)
      val swap  = (rsum || wsel) && !aWins
      val los   = ShiftRegister(Mux(swap, a(f), b(f)), lat)
      val win   = ShiftRegister(Mux(swap, b(f), a(f)), lat)
      val scale = Mux(rsum, alpha, F32_ONE) // alpha (RSUM) or 1.0 (SUM): the whole SUM/RSUM difference
      // 0-loser guard: a 0 value adds exactly the winner, avoiding 0*exp(-huge) = NaN at the LUT's underflow edge.
      // A real shard's l = Sexp >= 1, so l == 0 uniquely tags the monoid identity.
      val fma   = Mux(los === F32_ZERO, win, ffma(los, scale, win))
      // KEY -> m* ; OFF -> 0 ; WSEL -> the winner's carried payload (no add) ; SUM & RSUM -> the shared FMA
      Mux(isKey, ShiftRegister(mStar, lat), Mux(isOff, F32_ZERO, Mux(wsel, win, fma)))
    }
    (out, lat)
  }
}
