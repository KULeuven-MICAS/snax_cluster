package snax.DataPathJunction

import chisel3._
import chisel3.util._

import fp_native._
import fp_unit._

import snax.DataPathExtension.FpActivation
import snax.DataPathExtension.FpHelpers._

/** \============================================================================================================
  * `SoftmaxStatsJunction` -- the online-softmax statistics merge, and nothing else.
  * \============================================================================================================
  *
  * A softmax over a row that is spread over several clusters leaves each cluster with two numbers per query row: the
  * key `m`, the largest score it saw, and the value `l`, the sum of `exp(score - m)` over its slice. Two such partials
  * merge as
  * {{{
  *   m*   = max(m_a, m_b)
  *   l*   = alpha_a * l_a + alpha_b * l_b,      alpha_k = exp(scale * (m_k - m*))
  * }}}
  * where the winner's factor is `exp(0) = 1`, so only the loser is rescaled. This is the one merge `MonoidJunction` runs
  * with `F = 2`, built here as a fixed-function operator on the socket, with no geometry word, no slot map and no
  * ordered scan. It reads what the SIMD engine leaves in the scratchpad, so no software step converts or repacks it.
  *
  * ---- THE PARTIAL STREAM ---- (what the xDMA reader moves, byte for byte)
  * {{{
  *   a group = two beats on each operand stream:   beat 0 = the keys    m[0..31]   FP16, lane r = row r
  *                                                 beat 1 = the values  l[0..31]   FP16, lane r = row r
  * }}}
  * 32 rows is `dataWidth / 16`. The flash-attention kernel keeps `m` and `l` as two vectors of FP16, one lane per query
  * row, so a 32-row tile is exactly one key beat and one value beat, and a transfer of more rows is more groups.
  * The output is a group in the same form, so a chain of any length folds hop by hop.
  *
  * ---- CONFIGURATION ---- (two user CSRs)
  * {{{
  *   CSR(0) [7:0]   nValid   rows 0 .. nValid-1 are folded; every other row of the output is the identity
  *   CSR(1) [31:0]  scale    FP32 bits of the factor that multiplies the key difference before the exponential. The
  *                           softmax computes `exp(a * (s - m))`, so the merge needs the same `a`; 0 reads as 1.0
  * }}}
  *
  * ---- HOW IT MEETS THE SOCKET CONTRACT ----
  * {{{
  *   O2 format closure      the output group is a legal input group
  *   O3 identity tolerance  rows at or above nValid are the identity, and a published identity beat of all-0xFBFF
  *                          lanes is neutral in BOTH roles: as keys it is the lowest finite FP16, and as values it is
  *                          multiplied by an exponential that underflows to exactly 0 against any real key
  *   O4 no state between groups
  *   O5 cfgerr              nValid = 0 or above 32, or a scale that is negative, infinite or NaN
  * }}}
  * Unlike `MonoidJunction` it is not one pair per cycle: it takes `rowsPerCycle` rows a cycle, so a group costs about
  * `nValid / rowsPerCycle + 12` cycles. The statistics of a tile are one group, and a hop of a chain costs about 60
  * cycles whatever the payload, so the serial form trades a few cycles a hop for a fraction of the area.
  *
  * ---- DATAPATH, per row ----
  * {{{
  *   issue      compare keys, loser - winner (FP32), the winner key is written back over the A key
  *   +1         delta * scale                       (one multiplier stage)
  *   +1 .. +4   exp lookup
  *   +5         FMA: loser_l * alpha + winner_l     (the loser/winner l are read here, indexed by the delayed chunk)
  *   +6         narrow to FP16, written back over the A value
  * }}}
  * Keys are compared and selected as FP16 and widened only for the subtraction, so the key output is exact. The l of a
  * row is widened exactly, folded in FP32 with the same exponential and FMA as `MonoidJunction`, and rounded once to
  * FP16 on the way out.
  */
class HasSoftmaxStatsJunction(
  dataWidth:    Int = 512,
  // Rows folded per cycle. A group of `dataWidth / 16` rows takes `rows / rowsPerCycle` issue cycles.
  rowsPerCycle: Int = 4,
  fpPipe:       Int = 1,
  expLutN:      Int = 128,
  starveLimit:  Int = 4096
) extends HasDataPathJunction {
  implicit val junctionParam: JunctionParam =
    new JunctionParam(
      moduleName  = "SoftmaxStatsJunction",
      userCsrNum  = 2,
      dataWidth   = dataWidth,
      starveLimit = starveLimit
    )

  def instantiate(clusterName: String): SoftmaxStatsJunction =
    Module(new SoftmaxStatsJunction(
      rowsPerCycle = rowsPerCycle,
      fpPipe       = fpPipe,
      expLutN      = expLutN
    ) {
      override def desiredName = clusterName + namePostfix
    })
}

class SoftmaxStatsJunction(
  rowsPerCycle: Int = 4,
  fpPipe:       Int = 1,
  expLutN:      Int = 128
)(implicit
  junctionParam: JunctionParam
) extends DataPathJunction {

  // ---- geometry --------------------------------------------------------------------------------------------
  val nRows   = junctionParam.dataWidth / 16 // FP16 lanes of one beat: the rows of one group
  val R       = rowsPerCycle
  val nChunks = nRows / R
  require(junctionParam.dataWidth % 16 == 0, "SoftmaxStatsJunction: dataWidth must be a multiple of 16")
  require(isPow2(R) && R >= 1 && nRows % R == 0, s"SoftmaxStatsJunction: rowsPerCycle ($R) must divide $nRows")
  require(nChunks >= 2, s"SoftmaxStatsJunction: rowsPerCycle ($R) must leave at least two chunks of $nRows rows")

  // cycles from a chunk's issue to its key / value results
  val mulPipe = 1
  val keyLat  = mulPipe + MonoidCombine.latency // delta*scale, then the exp lookup
  val resLat  = keyLat + fpPipe                 // the FMA; the narrow is combinational

  val ID_KEY16 = "hFBFF".U(16.W) // the lowest finite FP16: loses every comparison

  // ---- CSR decode ------------------------------------------------------------------------------------------
  val nValid    = jct_csr_i(0)(7, 0)
  val scaleCsr  = jct_csr_i(1)
  val scale     = Mux(scaleCsr === 0.U, MonoidCombine.F32_ONE, scaleCsr)
  val nValidBad = nValid === 0.U || nValid > nRows.U
  val scaleBad  = scaleCsr(31) || scaleCsr(30, 23).andR // negative, infinite or NaN
  jct_cfgerr_o := nValidBad || scaleBad
  // the last chunk that holds a live row (nValid = 0 is reported by cfgerr and folds one chunk)
  val nVc       = Mux(nValid > nRows.U, nRows.U, nValid)
  val lastChunk = Mux(nVc === 0.U, 0.U, (nVc - 1.U) >> log2Ceil(R))

  // ---- storage: the two input groups, held as FP16 rows; the A side is overwritten with the result -----------
  val aK = Reg(Vec(nRows, UInt(16.W)))
  val aV = Reg(Vec(nRows, UInt(16.W)))
  val bK = Reg(Vec(nRows, UInt(16.W)))
  val bV = Reg(Vec(nRows, UInt(16.W)))
  val kAv, vAv, kBv, vBv = RegInit(false.B)
  val pA, pB             = RegInit(false.B) // the next beat on this stream is a value beat

  def rowsOf(beat: UInt): Seq[UInt] = (0 until nRows).map(r => beat(16 * r + 15, 16 * r))

  jct_a_i.ready := Mux(pA, !vAv, !kAv)
  jct_b_i.ready := Mux(pB, !vBv, !kBv)

  // ---- control ---------------------------------------------------------------------------------------------
  val sIdle :: sRun :: sDrain :: sOut :: Nil = Enum(4)
  val state  = RegInit(sIdle)
  val cnt    = RegInit(0.U((log2Ceil(nChunks) + 1).W))
  val drain  = RegInit(0.U(log2Ceil(resLat + 1).W))
  val outIdx = RegInit(false.B)

  val issue = state === sRun
  def clrPipe(in: Bool, n: Int): Bool = {
    val r = RegInit(VecInit(Seq.fill(n)(false.B)))
    r(0) := Mux(jct_start_i, false.B, in)
    for (i <- 1 until n) r(i) := Mux(jct_start_i, false.B, r(i - 1))
    r(n - 1)
  }
  val issueW = clrPipe(issue, resLat) // a chunk's value result is written this cycle
  val cntV   = ShiftRegister(cnt, keyLat) // the chunk whose values the FMA reads this cycle
  val cntW   = ShiftRegister(cnt, resLat) // the chunk whose results are written this cycle

  def pick[T <: Data](idx: UInt, xs: Seq[T]): T = MuxLookup(idx, xs.head)(xs.zipWithIndex.map { case (x, j) => j.U -> x })

  // ---- per-row datapath: R lanes ---------------------------------------------------------------------------
  val keyOut = Wire(Vec(R, UInt(16.W)))
  val resOut = Wire(Vec(R, UInt(16.W)))
  for (i <- 0 until R) {
    // issue: the keys of this chunk
    val ka   = pick(cnt, (0 until nChunks).map(j => aK(j * R + i)))
    val kb   = pick(cnt, (0 until nChunks).map(j => bK(j * R + i)))
    val kaF  = widen(ka, FP16)
    val kbF  = widen(kb, FP16)
    val aw   = fp32aWins(kaF, kbF) // a >= b
    val d0   = fadd(Mux(aw, kbF, kaF), fneg32(Mux(aw, kaF, kbF))) // loser - winner
    // Equal infinite keys subtract to NaN, and a NaN must not reach the exponential: equal keys mean delta = 0.
    val d0IsNaN = d0(30, 23).andR && d0(22, 0).orR
    val delta   = Mux(d0IsNaN, MonoidCombine.F32_ZERO, d0)
    val scaled  = fmul(delta, scale, mulPipe)
    val exp     = Module(new FpActivation(true, true, false, expLutN, 256))
    exp.io.in   := scaled
    exp.io.func := FpActivation.EXP.U
    keyOut(i)   := Mux(aw, ka, kb)

    // the FMA stage: the loser and winner values of the row, aligned with the exponential
    val swD  = ShiftRegister(!aw, keyLat) // B won, so A is the loser
    val la   = widen(pick(cntV, (0 until nChunks).map(j => aV(j * R + i))), FP16)
    val lb   = widen(pick(cntV, (0 until nChunks).map(j => bV(j * R + i))), FP16)
    val los  = Mux(swD, la, lb)
    val win  = Mux(swD, lb, la)
    val fma  = Module(new FpFma(FP32, FP32, FP32, fpPipe))
    fma.io.in_a := los
    fma.io.in_b := exp.io.out
    fma.io.in_c := win
    // A zero loser adds exactly the winner, which avoids 0 * exp(-huge) at the lookup's underflow edge.
    val winA = ShiftRegister(win, fpPipe)
    val zg   = ShiftRegister(los === MonoidCombine.F32_ZERO, fpPipe)
    resOut(i) := narrow(Mux(zg, winA, fma.io.out), FP16, 0)
  }

  // ---- input beats, the in-place results, and the group sequencing -----------------------------------------
  when(jct_a_i.fire) {
    when(!pA) { aK.zip(rowsOf(jct_a_i.bits)).foreach { case (r, v) => r := v }; kAv := true.B; pA := true.B }
      .otherwise { aV.zip(rowsOf(jct_a_i.bits)).foreach { case (r, v) => r := v }; vAv := true.B; pA := false.B }
  }
  when(jct_b_i.fire) {
    when(!pB) { bK.zip(rowsOf(jct_b_i.bits)).foreach { case (r, v) => r := v }; kBv := true.B; pB := true.B }
      .otherwise { bV.zip(rowsOf(jct_b_i.bits)).foreach { case (r, v) => r := v }; vBv := true.B; pB := false.B }
  }
  for (j <- 0 until nChunks; i <- 0 until R) {
    when(issue && cnt === j.U) { aK(j * R + i) := keyOut(i) }
    when(issueW && cntW === j.U) { aV(j * R + i) := resOut(i) }
  }

  val inReady = kAv && vAv && kBv && vBv
  switch(state) {
    is(sIdle) {
      when(inReady) { state := sRun; cnt := 0.U }
    }
    is(sRun) {
      when(cnt === lastChunk) { state := sDrain; drain := 0.U }.otherwise { cnt := cnt + 1.U }
    }
    is(sDrain) {
      drain := drain + 1.U
      when(drain === (resLat - 1).U) { state := sOut; outIdx := false.B; kBv := false.B; vBv := false.B }
    }
    is(sOut) {
      when(jct_data_o.fire) {
        when(outIdx) { state := sIdle; kAv := false.B; vAv := false.B; outIdx := false.B }.otherwise { outIdx := true.B }
      }
    }
  }
  when(jct_start_i) {
    state := sIdle; cnt := 0.U; outIdx := false.B
    kAv := false.B; vAv := false.B; kBv := false.B; vBv := false.B
    pA := false.B; pB := false.B
  }

  // ---- the output group ------------------------------------------------------------------------------------
  def live(r: Int): Bool = r.U < nValid
  val kBeat = Cat((0 until nRows).reverse.map(r => Mux(live(r), aK(r), ID_KEY16)))
  val vBeat = Cat((0 until nRows).reverse.map(r => Mux(live(r), aV(r), 0.U(16.W))))
  jct_data_o.valid := state === sOut
  jct_data_o.bits  := Mux(outIdx, vBeat, kBeat)

  jct_identity_o := Fill(nRows, ID_KEY16)
  jct_busy_o     := state =/= sIdle || kAv || vAv || kBv || vBv
}
