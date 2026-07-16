package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** SoftmaxChainRtDemo — the runtime-precision FULL CHAIN.
  *
  * Runs a complete softmax epilogue,  y = softmax(x),  through the THREE runtime-precision xDMA netlists
  * (StreamReduceRt, StreamMapRt) at FP16 and then at FP8, chosen purely by the `fmt` CSR field — ONE set of
  * netlists, two precisions, no re-elaboration. Softmax is a SW-orchestrated multi-pass:
  *
  *   pass 1  m = max_i x_i                         StreamReduceRt (op=MAX)
  *   pass 2  e_i = exp(x_i - m)                     StreamMapRt    (func=EXP, a=1, b=-m)
  *   pass 3  s = Σ_i e_i                            StreamReduceRt (op=ADD)
  *   pass 4  y_i = e_i * (1/s)                      StreamMapRt    (func=LINEAR, a=1/s, b=0)
  *
  * The scalar glue between passes (−m, 1/s) is Snitch's job in the real kernel; here the test JVM plays
  * Snitch. Each pass reuses ONE netlist for BOTH precisions (the whole point of runtime precision), so the
  * four scalatest blocks below build only two DUT types. Suite vars thread the TCDM-resident intermediates
  * between passes (FlatSpec runs blocks sequentially).
  *
  * Reference: float64 softmax of the FP-grid-snapped inputs (so the comparison isolates the LUT/quantization
  * error the hardware actually incurs, not an unfair full-precision gold).
  */
class SoftmaxChainRtDemo extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  // ---- FP16 / FP8(e5m2) codecs ----
  def f16bitsToF32(h: Int): Float = {
    val sign = if ((h & 0x8000) != 0) -1.0 else 1.0
    val exp  = (h >> 10) & 0x1f; val mant = h & 0x3ff
    (if (exp == 0) sign * mant * math.pow(2, -24)
     else if (exp == 0x1f) if (mant == 0) sign * Double.PositiveInfinity else Double.NaN
     else sign * (1024 + mant) * math.pow(2, exp - 25)).toFloat
  }
  def f32ToF16bits(f: Float): Int = {
    val bits = java.lang.Float.floatToIntBits(f); val sign = (bits >>> 16) & 0x8000
    val rawe = (bits >>> 23) & 0xff; val mant = bits & 0x7fffff
    if (rawe == 0xff) return sign | 0x7c00 | (if (mant != 0) 0x200 else 0)
    val exp = rawe - 127 + 15
    if (exp >= 0x1f) return sign | 0x7c00
    if (exp <= 0) { if (exp < -10) return sign
      val m = mant | 0x800000; val shift = 14 - exp; val half = m >>> shift
      val rem = m & ((1 << shift) - 1); val hw = 1 << (shift - 1)
      var r = half; if (rem > hw || (rem == hw && (half & 1) == 1)) r += 1; return sign | r }
    var h = sign | (exp << 10) | (mant >>> 13); val rem = mant & 0x1fff
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1)) h += 1; h
  }
  def f8ToF32(b: Int): Float = { // e5m2
    val sign = if ((b & 0x80) != 0) -1.0 else 1.0
    val exp  = (b >> 2) & 0x1f; val mant = b & 0x3
    (if (exp == 0) sign * mant * math.pow(2, -16) else sign * (4 + mant) * math.pow(2, exp - 17)).toFloat
  }
  def f32ToF8bits(f: Float): Int = { // e5m2, round-to-nearest-even, flush subnormal-underflow to 0
    if (f == 0.0f) return 0
    val sign = if (f < 0) 0x80 else 0; val a = math.abs(f).toDouble
    val e = math.floor(math.log(a) / math.log(2)).toInt
    var exp = e + 15
    if (exp >= 0x1f) return sign | 0x7c // clamp to max finite (exp=30,mant=3 -> 57344); 0x7c=inf guard unused here
    if (exp < 1) return sign // underflow -> 0
    val frac = a / math.pow(2, e) - 1.0 // in [0,1)
    var mant = math.round(frac * 4).toInt
    if (mant == 4) { mant = 0; exp += 1; if (exp >= 0x1f) return sign | 0x78 }
    sign | (exp << 2) | mant
  }
  def packF16(v: Seq[Int]): BigInt = v.zipWithIndex.foldLeft(BigInt(0)) { case (a, (h, i)) => a | (BigInt(h & 0xffff) << (16 * i)) }
  def packF8(v: Seq[Int]):  BigInt = v.zipWithIndex.foldLeft(BigInt(0)) { case (a, (x, i)) => a | (BigInt(x & 0xff) << (8 * i)) }
  def f32bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)
  def rCsr0(oc: Int, fmt: Int): BigInt = BigInt(oc) | (BigInt(fmt) << 16) // reduce csr0
  def mCsr2(fmt: Int, func: Int): BigInt = BigInt((fmt << 2) | func)      // map  csr2
  val FUNC_LINEAR = 0; val FUNC_EXP = 1
  val OP_MAX = 0; val OP_ADD = 1

  /** Drive ONE beat through a StreamMapRt DUT: y = act(a*x+b). */
  def runMap(dut: DataPathExtensionHarness, a: Float, b: Float, fmt: Int, func: Int, beat: BigInt): BigInt =
    driveOne(dut, Seq(f32bits(a), f32bits(b), mCsr2(fmt, func)), Seq(beat))

  /** Drive `beats` through a StreamReduceRt DUT (op in csr1) and return the splatted-scalar output beat. */
  def runReduce(dut: DataPathExtensionHarness, fmt: Int, op: Int, beats: Seq[BigInt]): BigInt =
    driveOne(dut, Seq(rCsr0(beats.length, fmt), BigInt(op)), beats)

  /** Common CSR-poke + feed-N-beats + read-one-output driver. */
  def driveOne(dut: DataPathExtensionHarness, csr: Seq[BigInt], beats: Seq[BigInt]): BigInt = {
    for (i <- csr.indices) dut.io.csr_i(i).poke(csr(i).U)
    dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    var out = BigInt(0)
    var threads = new chiseltest.internal.TesterThreadList(Seq())
    threads = threads.fork {
      for (bt <- beats) {
        dut.io.data_i.bits.poke(bt.U); dut.io.data_i.valid.poke(true)
        while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
        dut.clock.step(1)
      }
      dut.io.data_i.valid.poke(false)
    }
    threads = threads.fork {
      while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
      out = dut.io.data_o.bits.peekInt()
      dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false)
    }
    threads.joinAndStep()
    dut.io.data_o.ready.poke(true)
    var w = 0; while (dut.io.busy_o.peekBoolean() && w < 300) { dut.clock.step(1); w += 1 }
    dut.io.data_o.ready.poke(false)
    out
  }

  // ---- suite state (threaded between the 4 passes; FlatSpec executes blocks sequentially) ----
  val N16 = 32; val N8 = 64
  val rng = new Random(0x50f7)
  // raw inputs, snapped onto each format's grid (so the reference is a faithful gold)
  val rawX16 = Seq.fill(N16)((rng.nextDouble() * 2.0 - 1.0).toFloat)
  val rawX8  = Seq.fill(N8)((rng.nextDouble() * 2.0 - 1.0).toFloat)
  lazy val bitsX16 = rawX16.map(f32ToF16bits); lazy val bitsX8 = rawX8.map(f32ToF8bits)
  lazy val qX16 = bitsX16.map(f16bitsToF32);   lazy val qX8 = bitsX8.map(f8ToF32) // grid-snapped inputs
  var m16 = 0.0f; var m8 = 0.0f
  var eBits16: Seq[Int] = Seq(); var eBits8: Seq[Int] = Seq()
  var s16 = 0.0f; var s8 = 0.0f

  "SoftmaxChainRt pass1" should "m = max(x) via StreamReduceRt at FP16 and FP8 (ONE netlist)" in {
    test(new DataPathExtensionHarness(new HasStreamReduceRt(computeLanes = 64, op = Seq("FMA", "MAX"))))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        m16 = f16bitsToF32((runReduce(dut, 0, OP_MAX, Seq(packF16(bitsX16))) & 0xffff).toInt)
        m8  = f8ToF32((runReduce(dut, 2, OP_MAX, Seq(packF8(bitsX8))) & 0xff).toInt)
        println(f"[chain] pass1 max: FP16=$m16%.4f (gold ${qX16.max}%.4f)  FP8=$m8%.4f (gold ${qX8.max}%.4f)")
        assert(m16 == qX16.max && m8 == qX8.max, "max mismatch")
      }
  }

  "SoftmaxChainRt pass2" should "e = exp(x - m) via StreamMapRt at FP16 and FP8 (ONE netlist)" in {
    test(new DataPathExtensionHarness(new HasStreamMapRt(computeLanes = 64, func = Seq("LINEAR", "EXP"))))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val o16 = runMap(dut, 1.0f, -m16, 0, FUNC_EXP, packF16(bitsX16))
        val o8  = runMap(dut, 1.0f, -m8, 2, FUNC_EXP, packF8(bitsX8))
        eBits16 = (0 until N16).map(i => ((o16 >> (16 * i)) & 0xffff).toInt)
        eBits8  = (0 until N8).map(i => ((o8 >> (8 * i)) & 0xff).toInt)
        val eF16 = eBits16.map(f16bitsToF32); val gF16 = qX16.map(x => math.exp(x - qX16.max).toFloat)
        val err16 = (eF16 zip gF16).map { case (a, b) => math.abs(a - b) }.max
        println(f"[chain] pass2 exp: FP16 maxErr=$err16%.4g (LUT), FP8 lanes=${eBits8.length}")
        assert(err16 < 0.06f, s"FP16 exp LUT error too large: $err16")
      }
  }

  "SoftmaxChainRt pass3" should "s = Σ e via StreamReduceRt at FP16 and FP8 (ONE netlist)" in {
    test(new DataPathExtensionHarness(new HasStreamReduceRt(computeLanes = 64, op = Seq("FMA", "MAX"))))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        s16 = f16bitsToF32((runReduce(dut, 0, OP_ADD, Seq(packF16(eBits16))) & 0xffff).toInt)
        s8  = f8ToF32((runReduce(dut, 2, OP_ADD, Seq(packF8(eBits8))) & 0xff).toInt)
        println(f"[chain] pass3 sum: FP16 s=$s16%.4f (gold ${eBits16.map(f16bitsToF32).sum}%.4f)  FP8 s=$s8%.4f")
        assert(s16 > 0.0f && s8 > 0.0f, "sum must be positive")
      }
  }

  "SoftmaxChainRt pass4" should "y = e/s via StreamMapRt => full softmax correct at FP16 AND FP8 (ONE netlist)" in {
    test(new DataPathExtensionHarness(new HasStreamMapRt(computeLanes = 64, func = Seq("LINEAR", "EXP"))))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val y16 = runMap(dut, 1.0f / s16, 0.0f, 0, FUNC_LINEAR, packF16(eBits16))
        val y8  = runMap(dut, 1.0f / s8, 0.0f, 2, FUNC_LINEAR, packF8(eBits8))
        val hw16 = (0 until N16).map(i => f16bitsToF32(((y16 >> (16 * i)) & 0xffff).toInt))
        val hw8  = (0 until N8).map(i => f8ToF32(((y8 >> (8 * i)) & 0xff).toInt))
        // reference softmax of the grid-snapped inputs
        def ref(q: Seq[Float]): Seq[Float] = { val mx = q.max; val ex = q.map(x => math.exp(x - mx)); val sm = ex.sum; ex.map(e => (e / sm).toFloat) }
        val r16 = ref(qX16); val r8 = ref(qX8)
        val err16 = (hw16 zip r16).map { case (a, b) => math.abs(a - b) }.max
        val err8  = (hw8 zip r8).map { case (a, b) => math.abs(a - b) }.max
        val sum16 = hw16.sum; val sum8 = hw8.sum
        println("=" * 78)
        println(f"[SOFTMAX CHAIN — ONE set of runtime-precision netlists]")
        println(f"  FP16 (32 lanes): maxErr=$err16%.4g   Σy=$sum16%.4f   (should be ~1.0)")
        println(f"  FP8  (64 lanes): maxErr=$err8%.4g   Σy=$sum8%.4f   (should be ~1.0)")
        println(f"  chain = ReduceRt(MAX) |> MapRt(EXP) |> ReduceRt(ADD) |> MapRt(LINEAR), fmt = runtime CSR")
        println("=" * 78)
        assert(err16 < 0.05f, s"FP16 softmax error too large: $err16")
        assert(err8 < 0.10f, s"FP8 softmax error too large: $err8")
        assert(math.abs(sum16 - 1.0f) < 0.05f, s"FP16 probs must sum to ~1: $sum16")
        assert(math.abs(sum8 - 1.0f) < 0.10f, s"FP8 probs must sum to ~1: $sum8")
      }
  }
}
