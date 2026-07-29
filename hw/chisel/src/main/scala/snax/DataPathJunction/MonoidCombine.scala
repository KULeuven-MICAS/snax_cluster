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
  *   - SUM and RSUM are the SAME FMA, differing only in whether `scale` is `alpha` or `1.0`.
  *   - OFF is not a role at all: it is `field(l) >= F`.
  */
object MonoidCombine {

  // combineMode encoding (3 bits, carried in the user CSR of whichever placement instantiates the combine)
  val MODE_SUM     = 0
  val MODE_MOMENT  = 1
  val MODE_ATTN    = 2
  val MODE_MAXPOOL = 3
  val MODE_ARGMAX  = 4
  val MODE_MOMENT2 = 5

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
  case class Geom(hasKey: Bool, F: UInt, sigma: UInt, nExp: UInt, nAdd: UInt, keyPol: Bool)

  /** The largest legal sigma for `F` fields: `F * S <= nLanes`, i.e. sigma <= 4 - ceil(log2 F), capped at
    * SIGMA_MAX. Applying it as a SATURATION rather than a check is what makes every one of the 2^k possible CSR
    * words legal: an over-large sigma degrades to fewer partials per beat, which is benign, instead of
    * interleaving two partials into each other's lanes, which is silent corruption.
    */
  def sigmaMaxOf(F: UInt): UInt =
    MuxCase(0.U(2.W), Seq((F <= 2.U) -> 3.U(2.W), (F <= 4.U) -> 2.U(2.W), (F <= 8.U) -> 1.U(2.W)))

  /** the legacy 6-mode enum as an elaboration-time ROM: (hasKey, F, sigma, nExp, nAdd) */
  private def geomTable(dHead: Int): Seq[(Int, (Int, Int, Int, Int, Int))] = Seq(
    MODE_SUM     -> ((0, 2, 3, 0, 2)),                 // (Sx, Sx^2)       no key, both fields plainly summed
    MODE_MOMENT  -> ((1, 2, 3, 1, 0)),                 // (m, l)           one twisted coordinate
    MODE_ATTN    -> ((1, 2 + dHead, 0, 1 + dHead, 0)), // (m, l, O[dHead]) one alpha across l and every O lane
    MODE_MAXPOOL -> ((1, 1, 3, 0, 0)),                 // (m)              key only; field 1 is out of range
    MODE_ARGMAX  -> ((1, 2, 3, 0, 0)),                 // (m, payload)     field 1 > nExp+nAdd => winner-select
    MODE_MOMENT2 -> ((1, 4, 0, 3, 0))                  // (m, l, A, B)     three twisted coordinates, one alpha
  )

  // ---- CSR(0) geometry fields -------------------------------------------------------------------------------
  val RAWGEOM_BIT = 17 // 0 = the 6-mode enum at [15:13]; 1 = the raw geometry below
  // raw fields: [12] hasKey | [11:8] n | [21:18] nExp | [25:22] nAdd | [27:26] sigma | [28] keyPol

  /** the legacy 6-mode enum, decoded from a ROM that exists only at elaboration */
  private def legacyGeom(mode: UInt, dHead: Int): Geom = {
    val t = geomTable(dHead)
    def pick(sel: ((Int, Int, Int, Int, Int)) => Int, w: Int): UInt =
      MuxLookup(mode, sel(t.head._2).U(w.W))(t.map { case (m, g) => m.U -> sel(g).U(w.W) })
    Geom(pick(_._1, 1) === 1.U, pick(_._2, 5), pick(_._3, 2), pick(_._4, 5), pick(_._5, 5), false.B)
  }

  /** The geometry of the armed operator. The legacy enum and the raw encoding are two SPELLINGS of the same
    * five numbers, so every existing CSR word keeps decoding exactly as before while the raw path removes
    * `dHead` from the netlist entirely -- a dHead=12 flash attention is just `n=13, F=14, sigma=0` on the same
    * hardware, and operators outside the enum (flash carrying an argmax index, argmin, softmin) need no RTL.
    */
  def geomOf(csr: UInt, dHead: Int): Geom = {
    val raw = csr(RAWGEOM_BIT)
    val leg = legacyGeom(csr(15, 13), dHead)
    val rawHasKey = csr(12)
    // F = n + hasKey. `n = 0, hasKey = 0` would give F = 0, and F = 0 is a CLIFF, not a degenerate case:
    // `sigmaMaxOf`'s first arm is `F <= 2`, so it hands back sigma_max = 3, and then `field(l) < F` is false on
    // EVERY lane because the comparison is unsigned -- the block retires an all-zero 512-bit beat, which for an
    // additive fold is a legal identity, so an entire collective converges to zero in silence. Clamp to the
    // smallest partial that means anything.
    //
    // NOTE this makes the geometry defined, not the configuration correct: F = 0 is still a nonsense word, and
    // the honest fix is to REPORT it. That needs a config-error output on the junction ABI, which is a change to
    // `DataPathJunction` and its host, so it is deliberately not smuggled in here.
    val rawF      = Mux((csr(11, 8) +& rawHasKey) === 0.U, 1.U, (csr(11, 8) +& rawHasKey)).pad(5)
    val g = Geom(
      hasKey = Mux(raw, rawHasKey, leg.hasKey),
      F      = Mux(raw, rawF, leg.F),
      sigma  = Mux(raw, csr(27, 26), leg.sigma),
      nExp   = Mux(raw, csr(21, 18), leg.nExp),
      nAdd   = Mux(raw, csr(25, 22), leg.nAdd),
      keyPol = raw && csr(28)
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
    * `alpha` is deliberately NOT gated by `hasKey`: a keyless mode has no lane with `useAlpha`, so the LUT
    * output is simply unread -- which recovers a mux level on the twist path.
    */
  def keyFrontEnd(am: UInt, bm: UInt, hasKey: Bool, keyPol: Bool, expLutN: Int): (UInt, Bool) = {
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
    (exp.io.out, hasKey && !aw)
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
