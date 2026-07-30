package snax.DataPathJunction

import chisel3._
import chisel3.util._

import snax.DataPathExtension.FpActivation
import snax.DataPathExtension.FpHelpers._

/** ============================================================================================================
  * `MonoidCombine` -- the serving-reduction algebra as ONE lane map, ONE key front end and ONE value lane.
  * ============================================================================================================
  *
  * THE ALGEBRA. Every operator in the family is the same object: a partial is `(m, v_1..v_n)`, the key `m` lives
  * in the max-monoid and the values in `(R^n, +)`, joined by a twist -- the combine is
  * {{{
  *   (m_a, v_a) (+) (m_b, v_b)  =  ( max(m_a, m_b),  alpha_a * v_a + alpha_b * v_b )
  *   alpha_k = exp(m_k - m*)    so the winner's factor is exp(0) = 1 and only the LOSER is rescaled
  * }}}
  * i.e. a semidirect product `Max (x)_phi (R^n, +)`. The twist is a HOMOTHETY -- one scalar alpha times the
  * identity, not an arbitrary diagonal -- which is precisely why ONE exponential serves a whole partial however
  * many value coordinates it carries. (A genuinely diagonal action would need one exp per lane and would destroy
  * that property. Do not "generalise" it.)
  *
  * THE LANE MAP. A beat carries `S` partials of `F` fields, flattened FIELD-MAJOR:
  * {{{
  *   lane = field * S + slot        slot(l) = l & (S-1)      field(l) = l >> sigma      S = 1 << sigma
  * }}}
  * Field-major is not a convention, it is the cheap one: lane `l` takes its twist from front end `l & (S-1)` --
  * the LOW bits of `l` -- so the candidate sets across sigma are nested prefix masks and collapse to 12
  * two-input merges. Slot-major (`l & ~(Fp-1)`, the high bits) needs 32 merges and shares nothing between `l`
  * and `l+8`.
  *
  * It also places the structure map (the key factor) at lanes `0..S-1` INDEPENDENTLY of `F`, which is what makes
  * "eight key front ends" a theorem rather than a coincidence of `dataWidth/64`: `F >= 2` forces `sigma <= 3`
  * forces `S <= 8`, so a key lane can never leave lanes 0..7.
  *
  * THE ROLES ARE GONE. What was a five-value role enum decoded per field is now two lane predicates and one
  * control bit, because:
  *   - KEY and WSEL are the SAME WIRE. With a mode-uniform winner swap, field 0's own `win` register already
  *     holds `m*`, so the key is a COORDINATE OF THE OUTPUT PARTIAL, not a value broadcast to every field.
  *   - A twisted value coordinate and a plainly-summed one are the SAME FMA, differing only in whether `scale`
  *     is `alpha` or `1.0`. `nExp` counts the first kind, `nAdd` the second.
  *   - OFF is not a role at all: it is `field(l) >= F`.
  */
object MonoidCombine {

  // EVERY PARTIAL THIS FILE FOLDS HAS EXACTLY ONE KEY, so the field count is `F = n + 1` for every configuration
  // it accepts. The key is what the twist is a function of: it decides the winner, and `exp(m_k - m*)` is the
  // factor the loser's value coordinates are scaled by. A partial without one has nothing to compare and folds to
  // the direct product instead -- a different algebra, and `ElementwiseJunction`'s operator on the same socket.

  val F32_ONE  = "h3F800000".U(32.W) // 1.0f  (the scale of a plain sum / the winner's factor)
  val ID_M     = "hFF7FFFFF".U(32.W) // most-negative FINITE fp32: loses every max, and id-vs-id delta = +0.
  val ID_MAX   = "h7F7FFFFF".U(32.W) // most-POSITIVE finite fp32: the min-monoid's identity (see keyIdentity)
  val F32_ZERO = 0.U(32.W)
  // ID_M MUST stay finite. A "cleanup" to true -Inf makes the identity-vs-identity fold compute
  // fadd(-Inf, +Inf) = NaN in the front end below, which then poisons every short beat.
  require(!java.lang.Float.intBitsToFloat(0xff7fffff).isInfinite, "MonoidCombine: ID_M must be finite")

  /** the exp LUT's pipeline depth -- the latency of the twist, and the depth the value path aligns to */
  def latency: Int = FpActivation.PipeLatency

  val SIGMA_MAX = 3 // S <= 8; see the "eight key front ends is a theorem" note above

  /** The runtime geometry of one partial: `nExp` value coordinates take the twist, the next `nAdd` are plainly
    * summed, and anything above that but still in range carries the winner's payload instead of being added.
    * `keyPol` picks the MIN-monoid instead of the max (argmin / softmin).
    */
  case class Geom(F: UInt, sigma: UInt, nExp: UInt, nAdd: UInt, keyPol: Bool)

  /** The largest legal sigma for `F` fields: `F * S <= nLanes`, i.e. sigma <= 4 - ceil(log2 F), capped at
    * SIGMA_MAX. Applying it as a SATURATION rather than a check is what makes every one of the 2^k possible CSR
    * words legal: an over-large sigma degrades to fewer partials per beat, which is benign, instead of
    * interleaving two partials into each other's lanes, which is silent corruption.
    */
  def sigmaMaxOf(F: UInt): UInt =
    MuxCase(0.U(2.W), Seq((F <= 2.U) -> 3.U(2.W), (F <= 4.U) -> 2.U(2.W), (F <= 8.U) -> 1.U(2.W)))

  // ---- CSR(0): the geometry IS the configuration -------------------------------------------------------------
  // {{{
  //   [7:0]    nValid   live partials in the beat; slots at or above it are fed their field's identity
  //   [11:8]   n        value coordinates. The partial is `(m, v_1..v_n)`, so the field count is `F = n + 1`
  //   [21:18]  nExp     how many value coordinates take the twist -- fields 1 .. nExp
  //   [25:22]  nAdd     how many are plainly summed -- fields nExp+1 .. nExp+nAdd
  //   [27:26]  sigma    beat geometry; saturated to `sigmaMaxOf(F)` so no word can name an illegal layout
  //   [28]     keyPol   0 = max-monoid, 1 = min-monoid
  // }}}
  // Any field above `nExp + nAdd` and still below `F` carries the winner's payload instead of being combined,
  // so the three ranges partition a partial's value coordinates with no further encoding.
  //
  // Naming the geometry rather than an operator is what keeps `dHead` out of the netlist: a `dHead = 12` flash
  // attention is `n = 13, nExp = 13, sigma = 0`, and a flash merge carrying an argmax index alongside its
  // normalizer is the same word with a larger `n`. Both run on this hardware with no elaboration parameter.

  /** The geometry of the armed operator, read straight out of the configuration word. */
  def geomOf(csr: UInt): Geom = {
    // `F = n + 1`, so F >= 1 for every one of the 2^k possible words: the smallest partial is a bare key, which
    // is a max-reduction. There is no configuration that makes the field count zero, and therefore none that
    // puts `field(l) < F` false on every lane and retires an all-zero beat.
    val F = (csr(11, 8) +& 1.U).pad(5)
    val g = Geom(
      F      = F,
      sigma  = csr(27, 26),
      nExp   = csr(21, 18),
      nAdd   = csr(25, 22),
      keyPol = csr(28)
    )
    g.copy(sigma = Mux(g.sigma < sigmaMaxOf(g.F), g.sigma, sigmaMaxOf(g.F))) // saturate: no illegal geometry
  }

  /** `slot(l)` and `field(l)` are ELABORATION constants once sigma is fixed, so each is a small mux of literals
    * -- and folds away entirely wherever sigma is pinned.
    */
  def slotOf(l: Int, sigma: UInt): UInt =
    MuxLookup(sigma, 0.U(4.W))((0 to SIGMA_MAX).map(s => s.U -> (l & ((1 << s) - 1)).U(4.W)))
  def fieldOf(l: Int, sigma: UInt): UInt =
    MuxLookup(sigma, l.U(5.W))((0 to SIGMA_MAX).map(s => s.U -> (l >> s).U(5.W)))

  /** ONE key front end, living on lane `s` and reading that lane's OWN masked operands. Returns the twist and
    * the winner-swap bit, both broadcast to the value lanes of slot `s`.
    *
    * The lookup runs on every configuration. A partial whose value coordinates are all plainly summed has no
    * lane with `useAlpha`, so its output is simply unread -- which costs a LUT that idles and saves a mux level
    * on the twist path of every configuration that does use it.
    */
  def keyFrontEnd(am: UInt, bm: UInt, keyPol: Bool, expLutN: Int): (UInt, Bool) = {
    val aw   = fp32aWins(am, bm) ^ keyPol                  // a >= b, or a <= b in the min-monoid
    val d0   = fadd(Mux(aw, bm, am), fneg32(Mux(aw, am, bm))) // loser - m*
    // TWO KEYS OF THE SAME INFINITE SIGN PRODUCE NaN HERE, AND NaN MUST NOT REACH THE LUT.
    // `fadd(+Inf, -Inf)` returns the canonical *positive* qNaN, and the exp unit's input clamp is built from
    // `fp32max`/`fp32min`, which are integer comparisons valid for FINITE operands only -- so the clamp does not
    // bound a NaN, it LAUNDERS it into the largest legal argument, and the twist comes back as ~2.4e38 where it
    // should be 1.0. Small value fields then retire finite, plausible, badly wrong numbers. Equal keys mean
    // delta = 0 by definition, so force it: exp(+0) is exactly 1.0, the identity the algebra asks for.
    // Test the mantissa too -- `expAll1` alone would also catch +-Inf, and delta = -Inf is already CORRECT
    // (it clamps to LO_E, whose exponent flushes the twist to +0, which is the right limit).
    val d0IsNaN = d0(30, 23).andR && d0(22, 0).orR
    val delta   = Mux(d0IsNaN, F32_ZERO, Mux(keyPol, fneg32(d0), d0))
    val exp   = Module(new FpActivation(true, true, false, false, expLutN, 256))
    exp.io.in   := delta
    exp.io.func := false.B
    exp.io.gelu := false.B
    (exp.io.out, !aw)
  }

  /** The identity of a KEY field, which is NOT a constant: it is whichever value loses every comparison. In the
    * max-monoid that is the most-negative finite float; in the min-monoid it is the most-POSITIVE one.
    *
    * This is load-bearing and easy to get wrong. Flipping the comparison without flipping the identity makes
    * `ID_M` WIN every min, so argmin/softmin would return the PAD on every short beat -- a silent, format-legal
    * wrong answer on exactly the operators the raw encoding exists to unlock. `fadd(MAXF, -MAXF) = +0` keeps the
    * identity-vs-identity delta at zero, and a padded key still loses, so a winner-select payload stays
    * unreachable.
    */
  def keyIdentity(keyPol: Bool): UInt = Mux(keyPol, ID_MAX, ID_M)

  /** ONE value lane -- identical for every lane of every operator; only the stream-constant control bits differ.
    * `am`/`bm` arrive already masked to the field identity, so C4 is enforced on the INPUT, per lane.
    */
  def valueLane(
    am:       UInt,
    bm:       UInt,
    sw:       Bool,
    alpha:    UInt,
    useAlpha: Bool,
    selOrKey: Bool,
    inRange:  Bool,
    fma:      (UInt, UInt, UInt) => UInt,
    fmaLat:   Int
  ): UInt = {
    val los   = ShiftRegister(Mux(sw, am, bm), latency)
    val win   = ShiftRegister(Mux(sw, bm, am), latency)
    val scale = Mux(useAlpha, alpha, F32_ONE) // the WHOLE difference between a twisted and a plain sum
    val winA  = ShiftRegister(win, fmaLat)
    // 0-loser guard: a 0 value adds exactly the winner, avoiding 0*exp(-huge) at the LUT's underflow edge. A
    // real shard's l = Sexp >= 1, so l == 0 uniquely tags the monoid identity. The select and the bypassed
    // winner ride the FMA's own depth so the guard lands on the same beat as the product.
    val zg    = ShiftRegister(los === F32_ZERO, fmaLat)
    val res   = Mux(zg, winA, fma(los, scale, win))
    // out of range -> 0 ; key or winner-select -> the winner's field ; everything else -> the shared FMA
    Mux(inRange, Mux(selOrKey, winA, res), F32_ZERO)
  }
}
