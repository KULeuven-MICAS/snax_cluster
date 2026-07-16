package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Roofline / bandwidth-utilization sweep for the streaming-SIMD primitives.
  *
  * Measures STEADY-STATE input-accept throughput (input beats accepted / cycle; peak 1.0 = 512 bit/cycle)
  * as a function of computeLanes, per primitive. This is the fraction of the reader's 512 b/cyc the
  * primitive can actually sink = its bandwidth utilization / roofline point.
  *
  * Methodology notes (per the adversarial review of the design plan):
  *  - Throughput is counted from INPUT-ACCEPT events only (data_i.ready & fed), never output-retire
  *    events, so a reducer's 1-scalar-per-row output can't masquerade as the input rate.
  *  - A warm-up prefix of accepts is discarded so the pipeline-fill is excluded from the steady window.
  *  - Reduce is measured two ways: (a) ONE large row (operandCount = rowLen) to isolate the per-lane
  *    accumulator recurrence (the tree-fold happens once, amortised away); (b) many short rows to expose
  *    the fold-bound regime. beatSpan_ideal = subCycles = lanes/computeLanes; the accumulator recurrence
  *    floors beatSpan at accLat+1 (=4 at defaults), so naive lane-scaling saturates near cl=8.
  *  - Timing only: no value goldens here (the FP datapath latency is data-independent; correctness lives
  *    in the per-primitive testers). Feeds small FP16 values so nothing is NaN/inf.
  */
class RooflineTester extends AnyFlatSpec with ChiselScalatestTester {

  val W     = 512
  val flags = VerilatorFlags(Seq("--build-jobs", "1")) // fpnew lzc.sv UNOPTFLAT PCH race -> serial build

  // ---- packing / FP helpers (mirrors the sibling testers) ----
  def packBeat(ls: Seq[BigInt], ew: Int): BigInt =
    ls.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) => acc | ((v & ((BigInt(1) << ew) - 1)) << (ew * i)) }
  def f32bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)
  def f32ToF16bits(f: Float): Int = {
    val bits = java.lang.Float.floatToIntBits(f)
    val sign = (bits >>> 16) & 0x8000
    val rawe = (bits >>> 23) & 0xff
    val mant = bits & 0x7fffff
    if (rawe == 0xff) return sign | 0x7c00 | (if (mant != 0) 0x200 else 0)
    val exp = rawe - 127 + 15
    if (exp >= 0x1f) return sign | 0x7c00
    if (exp <= 0) {
      if (exp < -10) return sign
      val m = mant | 0x800000; val shift = 14 - exp
      val half = m >>> shift; val rem = m & ((1 << shift) - 1); val halfway = 1 << (shift - 1)
      var r = half; if (rem > halfway || (rem == halfway && (half & 1) == 1)) r += 1; return sign | r
    }
    var h = sign | (exp << 10) | (mant >>> 13)
    val rem = mant & 0x1fff; if (rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1)) h += 1; h
  }
  def randF16Beat(rng: Random, lanes: Int): BigInt =
    packBeat((0 until lanes).map(_ => BigInt(f32ToF16bits(rng.between(-3, 3)) & 0xffff)), 16)

  // ---- one measured config ----
  case class Roof(name: String, cl: Int, lanes: Int, subCycles: Int, nIn: Int, warmup: Int,
                  feedSpan: Long, firstOutCyc: Long, totalCyc: Long) {
    val measBeatSpan  = feedSpan.toDouble / math.max(1, nIn - warmup) // steady cyc / accepted input beat
    val inThroughput  = math.max(1, nIn - warmup).toDouble / feedSpan // input beats/cycle (<=1)
    val utilization   = inThroughput                                  // fraction of 512 b/cyc reader feed
    val effBW         = inThroughput * W                              // bit/cycle
    val idealBeatSpan = subCycles                                     // no-recurrence ideal
    val computeUtil   = subCycles.toDouble / measBeatSpan             // 1.0 = time-mux fully used (no bubble)
  }

  /** Drive an extension at full input rate (valid high, output ready high) and count input-accept cycles.
    * `nOut` is the expected number of output beats (loop terminates when they've all retired). */
  def measure(makeDut: => DataPathExtensionHarness, setCsr: DataPathExtensionHarness => Unit,
              beats: Seq[BigInt], nOut: Int, cl: Int, lanes: Int, name: String, warmup: Int): Roof = {
    val subCycles = lanes / cl
    val nIn       = beats.length
    var cyc       = 0L; var fed = 0; var got = 0
    var warmFeed  = -1L; var lastFeed = -1L; var firstOut = -1L
    test(makeDut).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      setCsr(dut)
      dut.io.enable_i.poke(true)
      dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
      dut.io.data_o.ready.poke(true)
      dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(beats(0))
      val maxCyc = nIn.toLong * 60 + 800
      while (got < nOut && cyc < maxCyc) {
        val canFeed = fed < nIn && dut.io.data_i.ready.peekBoolean()
        val outNow  = dut.io.data_o.valid.peekBoolean()
        dut.clock.step(1); cyc += 1
        if (canFeed) {
          fed += 1
          if (fed == warmup) warmFeed = cyc
          if (fed < nIn) dut.io.data_i.bits.poke(beats(fed)) else dut.io.data_i.valid.poke(false)
          lastFeed = cyc
        }
        if (outNow) { if (firstOut < 0) firstOut = cyc; got += 1 }
      }
      assert(got == nOut, s"$name: only $got/$nOut outputs after $cyc cyc (HANG / credit bug)")
      var w = 0; while (dut.io.busy_o.peekBoolean() && w < 800) { dut.clock.step(1); w += 1 }
      assert(!dut.io.busy_o.peekBoolean(), s"$name: busy_o STUCK HIGH after drain (completion/credit bug)")
    }
    val feedSpan = math.max(1L, lastFeed - warmFeed)
    Roof(name, cl, lanes, subCycles, nIn, warmup, feedSpan, firstOut, cyc)
  }

  // ---- config builders ----
  def roofMap(cl: Int): Roof = {
    val lanes = 32; val rng = new Random(0x100 + cl)
    val beats = Seq.fill(256)(randF16Beat(rng, lanes))
    measure(new DataPathExtensionHarness(
              new HasStreamMap(dataWidth = W, elementWidth = 16, computeLanes = cl, func = Seq("LINEAR_FP16"))),
            d => { d.io.csr_i(0).poke(f32bits(2.0f).U); d.io.csr_i(1).poke(f32bits(-1.5f).U)
                   if (d.io.csr_i.length > 2) d.io.csr_i(2).poke(0.U) },
            beats, nOut = 256, cl, lanes, s"Map.cl$cl", warmup = 8)
  }

  def roofEw(cl: Int, accBanks: Int): Roof = { // operandCount=2 -> 256 in / 128 out
    val lanes = 32; val rng = new Random(0x200 + cl)
    val beats = Seq.fill(256)(randF16Beat(rng, lanes))
    measure(new DataPathExtensionHarness(
              new HasStreamElementwise(dataWidth = W, elementWidth = 16, computeLanes = cl, op = Seq("FMA_FP16"),
                                       accBanks = accBanks)),
            d => { d.io.csr_i(0).poke(2.U); if (d.io.csr_i.length > 1) d.io.csr_i(1).poke(1.U) },
            beats, nOut = 128, cl, lanes, s"Ew.cl$cl", warmup = 8)
  }

  // accBanks: 1 = legacy single partial (recurrence gap); 0 = auto banked (breaks the recurrence)
  def roofReduceRow(cl: Int, rowLen: Int, op: Int, accBanks: Int): Roof = { // ONE big row: isolate recurrence
    val lanes = 32; val rng = new Random(0x300 + cl)
    val beats = Seq.fill(rowLen)(randF16Beat(rng, lanes))
    measure(new DataPathExtensionHarness(
              new HasStreamReduce(computeLanes = cl, op = Seq("FMA_FP16", "MAX_FP16"), elementWidth = 16, accBanks = accBanks)),
            d => { d.io.csr_i(0).poke(rowLen.U); d.io.csr_i(1).poke(op.U) },
            beats, nOut = 1, cl, lanes, s"Red1row.cl$cl", warmup = 8)
  }

  def roofReduceFold(cl: Int, oc: Int, nRows: Int, op: Int, accBanks: Int, treeLanes: Int = 2, foldPar: Int = 0): Roof = {
    val lanes = 32; val rng = new Random(0x400 + cl + treeLanes + foldPar)
    val beats = Seq.fill(oc * nRows)(randF16Beat(rng, lanes))
    measure(new DataPathExtensionHarness(
              new HasStreamReduce(computeLanes = cl, op = Seq("FMA_FP16", "MAX_FP16"), elementWidth = 16,
                                  accBanks = accBanks, treeLanes = treeLanes, foldParallel = foldPar)),
            d => { d.io.csr_i(0).poke(oc.U); d.io.csr_i(1).poke(op.U) },
            beats, nOut = nRows, cl, lanes, s"RedFold.cl$cl.oc$oc", warmup = 8)
  }

  // Single-line result, greppable from the log. One scalatest test PER (primitive, cl) so each `test(...)`
  // elaboration gets its OWN test_run_dir (multiple test() calls in one scalatest test collide on the shared
  // verilated/ dir -> DirectoryNotEmptyException).
  def printRoof(tag: String, r: Roof): Unit =
    println(f"[ROOF $tag%-9s cl=${r.cl}%2d sub=${r.subCycles}%2d] beatSpan=${r.measBeatSpan}%5.2f " +
            f"util=${r.utilization}%5.3f effBW=${r.effBW}%6.0f compUtil=${r.computeUtil}%5.3f " +
            f"lat=${r.totalCyc}%5d firstOut=${r.firstOutCyc}%4d")

  // ---- precision-generic builders (timing-only: FP datapath latency is data-independent, so random beats
  //      of the right element width suffice to measure the roofline) ----
  def randBeatEw(rng: Random, lanes: Int, ew: Int): BigInt =
    packBeat((0 until lanes).map(_ => BigInt(rng.nextInt(1 << (ew - 2)))), ew) // small magnitudes, no inf/nan
  def precTag(ew: Int): String = if (ew == 8) "FP8" else if (ew == 16) "FP16" else "FP32"

  def roofMapP(cl: Int, ew: Int): Roof = {
    val lanes = W / ew; val rng = new Random(0x500 + cl + ew)
    val beats = Seq.fill(256)(randBeatEw(rng, lanes, ew))
    measure(new DataPathExtensionHarness(
              new HasStreamMap(dataWidth = W, elementWidth = ew, computeLanes = cl, func = Seq(s"LINEAR_${precTag(ew)}"))),
            d => { d.io.csr_i(0).poke(f32bits(2.0f).U); d.io.csr_i(1).poke(f32bits(-1.5f).U)
                   if (d.io.csr_i.length > 2) d.io.csr_i(2).poke(0.U) },
            beats, nOut = 256, cl, lanes, s"MapP.ew$ew.cl$cl", warmup = 8)
  }
  def roofReduceRowP(cl: Int, ew: Int, accBanks: Int): Roof = {
    val lanes = W / ew; val rng = new Random(0x600 + cl + ew)
    val beats = Seq.fill(128)(randBeatEw(rng, lanes, ew))
    measure(new DataPathExtensionHarness(
              new HasStreamReduce(computeLanes = cl, op = Seq(s"FMA_${precTag(ew)}", s"MAX_${precTag(ew)}"),
                                  elementWidth = ew, accBanks = accBanks)),
            d => { d.io.csr_i(0).poke(128.U); d.io.csr_i(1).poke(1.U) },
            beats, nOut = 1, cl, lanes, s"Red1rowP.ew$ew.cl$cl", warmup = 8)
  }

  // FP8: lanes=64, so computeLanes sweeps up to 64 (subCycles=1 at cl=64 => full-bandwidth point)
  for (cl <- Seq(8, 16, 32, 64)) {
    s"Roofline_MAP_FP8_cl$cl" should "report FP8 Map roofline (lanes=64)" in { printRoof("MAP-FP8", roofMapP(cl, 8)) }
  }
  for (cl <- Seq(8, 16, 32, 64)) {
    s"Roofline_RED1ROW_FP8bk_cl$cl" should "report FP8 banked Reduce roofline (lanes=64)" in {
      printRoof("RED1ROW-FP8bk", roofReduceRowP(cl, 8, accBanks = 0))
    }
  }

  // ---- FP16 computeLanes sweeps (lanes=32), one test per cl ----
  for (cl <- Seq(4, 8, 16, 32)) {
    s"Roofline_MAP_cl$cl" should "report roofline (no recurrence -> util->1.0 at cl=32)" in {
      printRoof("MAP", roofMap(cl))
    }
  }
  for (cl <- Seq(4, 8, 16, 32)) {
    s"Roofline_EWnb_cl$cl" should "report roofline (UNBANKED: accumulate recurrence)" in {
      printRoof("EW-nb", roofEw(cl, accBanks = 1))
    }
  }
  for (cl <- Seq(4, 8, 16, 32)) {
    s"Roofline_EWbk_cl$cl" should "report roofline (BANKED: recurrence removed)" in {
      printRoof("EW-bk", roofEw(cl, accBanks = 0))
    }
  }
  for (cl <- Seq(4, 8, 16, 32)) {
    s"Roofline_RED1ROWnb_cl$cl" should "report roofline (1 big row, UNBANKED: recurrence gap)" in {
      printRoof("RED1ROW-nb", roofReduceRow(cl, 128, 1, accBanks = 1))
    }
  }
  for (cl <- Seq(4, 8, 16, 32)) {
    s"Roofline_RED1ROWbk_cl$cl" should "report roofline (1 big row, BANKED: recurrence removed)" in {
      printRoof("RED1ROW-bk", roofReduceRow(cl, 128, 1, accBanks = 0))
    }
  }
  for (cl <- Seq(4, 8, 16, 32)) {
    s"Roofline_REDFOLDnb_cl$cl" should "report roofline (many short rows, UNBANKED: fold-bound)" in {
      printRoof("REDFOLD-nb", roofReduceFold(cl, 4, 32, 1, accBanks = 1))
    }
  }
  for (cl <- Seq(4, 8, 16, 32)) {
    s"Roofline_REDFOLDbk_cl$cl" should "report roofline (many short rows, BANKED)" in {
      printRoof("REDFOLD-bk", roofReduceFold(cl, 4, 32, 1, accBanks = 0))
    }
  }
  // Fix-C lever: banked + wider fold (treeLanes=16 vs 2) shrinks the per-row fold bubble
  for (cl <- Seq(16, 32)) {
    s"Roofline_REDFOLDtl16_cl$cl" should "report roofline (banked, treeLanes=16 fast fold)" in {
      printRoof("REDFOLD-tl16", roofReduceFold(cl, 4, 32, 1, accBanks = 0, treeLanes = 16))
    }
  }
  // Realistic multi-row: oc=32 (D=1024) rows amortise the once-per-row fold (vs the oc=4 tiny-reduction worst case)
  for (cl <- Seq(32)) {
    s"Roofline_REDFOLDreal_cl$cl" should "report roofline (realistic multi-row oc=32, treeLanes=16)" in {
      printRoof("REDFOLD-real-oc32", roofReduceFold(cl, 32, 4, 1, accBanks = 0, treeLanes = 16))
    }
  }
  for (cl <- Seq(32)) {
    s"Roofline_REDFOLDreal64_cl$cl" should "report roofline (realistic multi-row oc=64, treeLanes=16)" in {
      printRoof("REDFOLD-real-oc64", roofReduceFold(cl, 64, 2, 1, accBanks = 0, treeLanes = 16))
    }
  }
  // foldParallel: fully-pipelined tree -> even the tiny oc=4 multi-row pipelines to the roofline
  for (cl <- Seq(16, 32)) {
    s"Roofline_REDFOLDpar_cl$cl" should "report roofline (banked, foldParallel: rows pipeline)" in {
      printRoof("REDFOLD-par", roofReduceFold(cl, 4, 32, 1, accBanks = 0, foldPar = 1))
    }
  }
}
