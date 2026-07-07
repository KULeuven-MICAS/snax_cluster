package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import firrtl2.options.TargetDirAnnotation
import org.scalatest.flatspec.AnyFlatSpec

/** App-level Tier-1 tests: replicate each xDMA app's ACTUAL pass sequence (the same extensions, op-sets,
  * computeLanes and per-pass CSRs as the deployed cfg) purely in the chisel/Verilator env — no vsim, no elf.
  *
  * Each pass runs the real extension RTL through DataPathExtensionHarness; the captured output beats are fed
  * as the next pass's input (exactly how the app chains passes through L1), and any host scalar step
  * (softmax 1/Σ, rmsnorm 1/rms) is done here in Scala like the DM core does. The final FP16 output is checked
  * against a double-precision golden within the app's <=4 FP16 ULP budget. Lets us catch app-level datapath
  * regressions in ~seconds instead of a multi-minute vsim build+run.
  *
  * Extensions are built with the cfg op-sets/lanes: StreamMap func=[LINEAR,EXP,SILU] cl4, StreamReduce
  * op=[FMA,MAX] cl4, StreamElementwise op=[FMA] cl4.
  */
class XdmaAppTester extends AnyFlatSpec with ChiselScalatestTester {

  // ---- FP16 <-> FP32 helpers (bit-exact to the host path) ----
  def f16bitsToF32(h: Int): Float = {
    val sign = (h >> 15) & 0x1; val exp = (h >> 10) & 0x1f; val mant = h & 0x3ff
    val v =
      if (exp == 0) mant * math.pow(2, -24)
      else if (exp == 0x1f) if (mant == 0) Double.PositiveInfinity else Double.NaN
      else (1024 + mant) * math.pow(2, exp - 25)
    (if (sign == 1) -v else v).toFloat
  }
  def f32ToF16bits(f: Float): Int = {
    val bits = java.lang.Float.floatToIntBits(f)
    val sign = (bits >>> 16) & 0x8000
    val e    = ((bits >>> 23) & 0xff) - 127 + 15
    val mant = bits & 0x7fffff
    if (((bits >>> 23) & 0xff) == 0xff) return sign | 0x7c00 | (if (mant != 0) 0x200 else 0)
    if (e >= 0x1f) return sign | 0x7c00
    if (e <= 0) { if (e < -10) return sign; val m = (mant | 0x800000) >> (14 - e); return sign | m }
    // round-to-nearest-even on the 13 dropped mantissa bits (matches the HW narrow; truncation would
    // compound across multi-pass app goldens like rope)
    val m10  = mant >> 13
    val rndUp = ((mant >> 12) & 1) == 1 && (((mant & 0xfff) != 0) || (m10 & 1) == 1)
    var mm   = m10 + (if (rndUp) 1 else 0); var ee = e
    if (mm == 0x400) { mm = 0; ee += 1 }
    if (ee >= 0x1f) return sign | 0x7c00
    sign | (ee << 10) | (mm & 0x3ff)
  }
  def f32bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)
  val LANES = 32
  def packBeat(ls: Seq[Int]): BigInt =
    ls.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (h, i)) => acc | (BigInt(h & 0xffff) << (16 * i)) }
  def unpackBeat(b: BigInt): Seq[Int] = (0 until LANES).map(i => ((b >> (16 * i)) & 0xffff).toInt)
  def f16mono(h: Int): Int = { val mag = h & 0x7fff; if ((h & 0x8000) != 0) 0x8000 - mag else 0x8000 + mag }

  // op / func CSR constants (match the apps)
  val ACT_LINEAR = 0; val ACT_EXP = 1; val ACT_SILU = 2
  val RED_MAX = 0; val RED_ADD = 1; val RED_SUMSQ = 2; val RED_TAP = 0x100
  val EW_MUL = 0; val EW_ADD = 1

  // cfg-matched extension builders
  def mkMap    = new HasStreamMap(dataWidth = 512, elementWidth = 16, computeLanes = 4,
                                  func = Seq("LINEAR_FP16", "EXP_FP16", "SILU_FP16"))
  def mkReduce = new HasStreamReduce(computeLanes = 4, op = Seq("FMA_FP16", "MAX_FP16"), elementWidth = 16)
  def mkElem   = new HasStreamElementwise(dataWidth = 512, elementWidth = 16, computeLanes = 4, op = Seq("FMA_FP16"))

  private var passId = 0
  /** Run ONE extension pass through the harness and return its output beats. `tag` -> a unique sim dir so
    * several passes can run inside one test without colliding. */
  def runExt(tag: String, mk: => HasDataPathExtension, csr: Seq[BigInt], inBeats: Seq[BigInt], nOut: Int): Seq[BigInt] = {
    passId += 1
    var outs = Seq[BigInt]()
    test(new DataPathExtensionHarness(mk))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")),
        TargetDirAnnotation(s"test_run_dir/xapp_${tag}_$passId"))) { dut =>
        csr.zipWithIndex.foreach { case (v, i) => dut.io.csr_i(i).poke(v.U) }
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        var th = new chiseltest.internal.TesterThreadList(Seq())
        th = th.fork {
          dut.io.data_i.valid.poke(true)
          for (bt <- inBeats) {
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
            dut.io.data_i.bits.poke(bt.U); dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
        }
        th = th.fork {
          for (_ <- 0 until nOut) {
            while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
            outs = outs :+ dut.io.data_o.bits.peekInt()
            dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false)
          }
        }
        th.joinAndStep()
      }
    outs
  }

  def redScalar(beats: Seq[BigInt]): Float = f16bitsToF32(unpackBeat(beats.last).head) // splatted scalar, lane 0
  def interleave(a: Seq[BigInt], b: Seq[BigInt]): Seq[BigInt] = a.zip(b).flatMap { case (x, y) => Seq(x, y) }
  def randRow(n: Int, rng: Random, lo: Float, hi: Float): Seq[BigInt] =
    Seq.fill(n)(packBeat(Seq.fill(LANES)(f32ToF16bits(lo + rng.nextFloat() * (hi - lo)))))
  def flat16(beats: Seq[BigInt]): Seq[Float] = beats.flatMap(unpackBeat).map(f16bitsToF32)

  /** compare HW vs golden (both flat FP16-value seqs), report worst ULP over significant (normal) goldens. */
  def check(name: String, hw: Seq[Float], gold: Seq[Float], maxUlp: Int = 4): Unit = {
    var worst = 0; var at = 0
    for (i <- hw.indices) {
      val gB = f32ToF16bits(gold(i))
      if ((gB & 0x7c00) != 0) { // significant
        val u = math.abs(f16mono(f32ToF16bits(hw(i))) - f16mono(gB)); if (u > worst) { worst = u; at = i }
      }
    }
    if (worst > maxUlp) { // debug: dump the biggest offenders
      hw.indices.map(i => (i, math.abs(f16mono(f32ToF16bits(hw(i))) - f16mono(f32ToF16bits(gold(i))))))
        .sortBy(-_._2).take(5).foreach { case (i, u) =>
          println(f"[$name] idx=$i hw=${hw(i)}%.5f (0x${f32ToF16bits(hw(i))}%04x) gold=${gold(i)}%.5f (0x${f32ToF16bits(gold(i))}%04x) ulp=$u") }
    }
    println(f"[XdmaApp:$name] worst FP16 ULP=$worst at idx=$at over ${hw.length} elems")
    assert(worst <= maxUlp, s"$name exceeds $maxUlp ULP: $worst at $at")
  }

  // ============================== apps ==============================

  "XdmaApp_silu" should "match golden silu(x) = x*sigmoid(x)" in {
    val rng = new Random(1); val N = 2
    val x = randRow(N, rng, -6f, 6f)
    // T1: StreamMap(SILU, a=1, b=0)  1 beat -> 1 beat
    val out = runExt("silu", mkMap, Seq(f32bits(1.0f), f32bits(0.0f), ACT_SILU), x, N)
    val gold = flat16(x).map(v => (v.toDouble / (1.0 + math.exp(-v.toDouble))).toFloat)
    check("silu", flat16(out), gold)
  }

  "XdmaApp_swiglu" should "match golden silu(gate)*up" in {
    val rng = new Random(2); val N = 2
    val gate = randRow(N, rng, -6f, 6f); val up = randRow(N, rng, -3f, 3f)
    // T1: StreamMap(SILU) on gate -> sg
    val sg = runExt("swiglu_silu", mkMap, Seq(f32bits(1.0f), f32bits(0.0f), ACT_SILU), gate, N)
    // T2: StreamElementwise(MUL, operandCount=2) on interleaved {sg, up} -> out
    val out = runExt("swiglu_mul", mkElem, Seq(BigInt(2), EW_MUL), interleave(sg, up), N)
    val gold = flat16(gate).zip(flat16(up)).map { case (g, u) =>
      ((g.toDouble / (1.0 + math.exp(-g.toDouble))) * u.toDouble).toFloat
    }
    check("swiglu", flat16(out), gold)
  }

  "XdmaApp_rmsnorm" should "match golden x/sqrt(mean(x^2))" in {
    val rng = new Random(3); val N = 2
    val x = randRow(N, rng, -3f, 3f); val D = N * LANES
    // T1: StreamReduce(SUMSQ) N beats -> 1 scalar
    val ssqBeats = runExt("rms_sumsq", mkReduce, Seq(BigInt(N), RED_SUMSQ), x, 1)
    val ssq = redScalar(ssqBeats)
    val invRms = (1.0 / math.sqrt(ssq.toDouble / D + 1e-6)).toFloat // host step
    // T2: StreamMap(LINEAR, a=invRms, b=0)
    val out = runExt("rms_scale", mkMap, Seq(f32bits(invRms), f32bits(0.0f), ACT_LINEAR), x, N)
    val gold = flat16(x).map(v => (v.toDouble * invRms.toDouble).toFloat)
    check("rmsnorm", flat16(out), gold)
  }

  "XdmaApp_softmax" should "match golden softmax(x)" in {
    val rng = new Random(4); val N = 2
    val x = randRow(N, rng, -4f, 4f)
    // T1: StreamReduce(MAX) -> max scalar
    val maxBeats = runExt("sm_max", mkReduce, Seq(BigInt(N), RED_MAX), x, 1)
    val mx = redScalar(maxBeats)
    // T2a: StreamMap(EXP, a=1, b=-max) -> exp row
    val expRow = runExt("sm_exp", mkMap, Seq(f32bits(1.0f), f32bits(-mx), ACT_EXP), x, N)
    // T2b: StreamReduce(ADD, tap) on exp row -> N passthrough + 1 sum scalar
    val addTap = runExt("sm_sum", mkReduce, Seq(BigInt(N), RED_ADD | RED_TAP), expRow, N + 1)
    val sum = redScalar(addTap) // last beat = scalar
    val invSum = (1.0 / sum.toDouble).toFloat
    // T3: StreamMap(LINEAR, a=invSum) on exp row
    val out = runExt("sm_norm", mkMap, Seq(f32bits(invSum), f32bits(0.0f), ACT_LINEAR), expRow, N)
    val xs = flat16(x); val gmax = xs.max
    val exps = xs.map(v => math.exp(v.toDouble - gmax)); val gsum = exps.sum
    val gold = exps.map(e => (e / gsum).toFloat)
    check("softmax", flat16(out), gold)
  }

  "XdmaApp_rope" should "match golden x*cos + rotate_half(x)*sin" in {
    val rng = new Random(5); val N = 2
    val x = randRow(N, rng, -3f, 3f)
    // cos_full = [c0,c0,c1,c1,...], sin_signed = [-s0,+s0,-s1,+s1,...] over each 32-lane beat (pairwise angles)
    def angles(seed: Int) = Seq.fill(LANES / 2)((seed + rng.nextInt(100)) * 0.01)
    val cosB = x.indices.map { _ => val a = angles(0); packBeat(a.flatMap(t => Seq(f32ToF16bits(math.cos(t).toFloat), f32ToF16bits(math.cos(t).toFloat)))) }
    val sinB = x.indices.map { _ => val a = angles(1); packBeat(a.flatMap(t => Seq(f32ToF16bits(-math.sin(t).toFloat), f32ToF16bits(math.sin(t).toFloat)))) }
    // rotate_half: swap each adjacent pair (x0,x1)->(x1,x0) within a beat (iDMA does this in the app)
    def rotHalf(b: BigInt): BigInt = { val l = unpackBeat(b); packBeat(l.grouped(2).flatMap { case Seq(a, c) => Seq(c, a); case o => o }.toSeq) }
    val xswap = x.map(rotHalf)
    // P1: MUL x (.) cos ; P2: MUL xswap (.) sin ; P3: ADD tmp1 (+) tmp2
    val tmp1 = runExt("rope_p1", mkElem, Seq(BigInt(2), EW_MUL), interleave(x, cosB), N)
    val tmp2 = runExt("rope_p2", mkElem, Seq(BigInt(2), EW_MUL), interleave(xswap, sinB), N)
    val out  = runExt("rope_p3", mkElem, Seq(BigInt(2), EW_ADD), interleave(tmp1, tmp2), N)
    def f16(d: Double): Float = f16bitsToF32(f32ToF16bits(d.toFloat)) // model each pass's FP16 narrowing
    val gold = (0 until N * LANES).map { i =>
      val xf = flat16(x)(i); val xsf = flat16(xswap)(i)
      val cf = f16bitsToF32(unpackBeat(cosB(i / LANES))(i % LANES))
      val sf = f16bitsToF32(unpackBeat(sinB(i / LANES))(i % LANES))
      val t1 = f16(xf.toDouble * cf.toDouble)   // P1: fp16(x * cos)
      val t2 = f16(xsf.toDouble * sf.toDouble)  // P2: fp16(rotate_half(x) * sin)
      f16(t1.toDouble + t2.toDouble)            // P3: fp16(t1 + t2)
    }
    check("rope", flat16(out), gold)
  }

  // ---- multi-row apps: S rows in ONE dispatch; per-row reduce (operandCount=N) + per-row broadcast MUL/ADD.
  def splat(v: Float): BigInt = packBeat(Seq.fill(LANES)(f32ToF16bits(v)))
  // interleave each row's beats with that row's splatted scalar: [x_r0, splat(s_r), x_r1, splat(s_r), ...]
  def bcastIL(rows: Seq[Seq[BigInt]], sc: Seq[Float]): Seq[BigInt] =
    rows.zip(sc).flatMap { case (bs, s) => bs.flatMap(b => Seq(b, splat(s))) }
  def scalars(beats: Seq[BigInt]): Seq[Float] = beats.map(b => f16bitsToF32(unpackBeat(b).head))

  "XdmaApp_reduce_multirow" should "match per-row SUMSQ/MAX over S rows in one dispatch" in {
    val rng = new Random(6); val S = 3; val N = 2
    val rows = Seq.fill(S)(randRow(N, rng, -3f, 3f))
    val ssq = scalars(runExt("redmr_ssq", mkReduce, Seq(BigInt(N), RED_SUMSQ), rows.flatten, S))
    check("reduce_mr_sumsq", ssq, rows.map(r => flat16(r).map(v => v.toDouble * v.toDouble).sum.toFloat), 4)
    val mx = scalars(runExt("redmr_max", mkReduce, Seq(BigInt(N), RED_MAX), rows.flatten, S))
    check("reduce_mr_max", mx, rows.map(r => flat16(r).max), 0)
  }

  "XdmaApp_rmsnorm_multirow" should "match per-row x/rms via SUMSQ + per-row MUL broadcast" in {
    val rng = new Random(7); val S = 2; val N = 2; val D = N * LANES
    val rows = Seq.fill(S)(randRow(N, rng, -3f, 3f))
    val ssq = scalars(runExt("rmsmr_ssq", mkReduce, Seq(BigInt(N), RED_SUMSQ), rows.flatten, S))
    val invRms = ssq.map(s => (1.0 / math.sqrt(s.toDouble / D + 1e-6)).toFloat) // per-row host step
    val out = runExt("rmsmr_scale", mkElem, Seq(BigInt(2), EW_MUL), bcastIL(rows, invRms), S * N)
    val gold = rows.zip(invRms).flatMap { case (r, ir) => flat16(r).map(v => (v.toDouble * ir.toDouble).toFloat) }
    check("rmsnorm_mr", flat16(out), gold, 4)
  }

  "XdmaApp_softmax_multirow" should "match per-row softmax via max/sub/exp/sum/mul" in {
    val rng = new Random(8); val S = 2; val N = 2
    val rows = Seq.fill(S)(randRow(N, rng, -4f, 4f))
    val negmax = scalars(runExt("smmr_max", mkReduce, Seq(BigInt(N), RED_MAX), rows.flatten, S)).map(-_)
    val xs   = runExt("smmr_sub", mkElem, Seq(BigInt(2), EW_ADD), bcastIL(rows, negmax), S * N) // x - max
    val expb = runExt("smmr_exp", mkMap, Seq(f32bits(1.0f), f32bits(0.0f), ACT_EXP), xs, S * N)
    val invSum = scalars(runExt("smmr_sum", mkReduce, Seq(BigInt(N), RED_ADD), expb, S)).map(s => (1.0 / s.toDouble).toFloat)
    val out = runExt("smmr_norm", mkElem, Seq(BigInt(2), EW_MUL), bcastIL(expb.grouped(N).toSeq, invSum), S * N)
    val gold = rows.flatMap { r =>
      val xf = flat16(r); val m = xf.max; val e = xf.map(v => math.exp(v.toDouble - m)); val sm = e.sum
      e.map(x => (x / sm).toFloat)
    }
    // 5-pass FP16 chain vs a double-precision golden (with LUT-approximated exp); the softmax-multirow app
    // itself uses a <=8 FP16 ULP criterion, so match it.
    check("softmax_mr", flat16(out), gold, 8)
  }
}
