package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** StreamMapRt: one elaborated netlist maps act(a*x+b) at FP16 / BF16 / FP8 chosen at runtime by the `fmt`
  * CSR field. Proves (a) correctness of FP16 a*x+b (32 lanes) and FP8 identity (64 lanes) on the SAME dut
  * instance back-to-back, and (b) full 512 b/cyc throughput at computeLanes=maxLanes (=64).
  */
class StreamMapRtTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

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
  def f32bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)
  def packF16(v: Seq[Int]): BigInt = v.zipWithIndex.foldLeft(BigInt(0)) { case (a, (h, i)) => a | (BigInt(h & 0xffff) << (16 * i)) }
  def packF8(v: Seq[Int]):  BigInt = v.zipWithIndex.foldLeft(BigInt(0)) { case (a, (x, i)) => a | (BigInt(x & 0xff) << (8 * i)) }
  def csr2(fmt: Int, func: Int): BigInt = BigInt((fmt << 2) | func)

  /** Drive ONE 512-bit beat through the extension and return the output beat. */
  def runBeat(dut: DataPathExtensionHarness, a: Float, b: Float, fmt: Int, func: Int, beat: BigInt): BigInt = {
    dut.io.csr_i(0).poke(f32bits(a).U); dut.io.csr_i(1).poke(f32bits(b).U); dut.io.csr_i(2).poke(csr2(fmt, func).U)
    dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    var out = BigInt(0)
    var threads = new chiseltest.internal.TesterThreadList(Seq())
    threads = threads.fork {
      dut.io.data_i.valid.poke(true)
      while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
      dut.io.data_i.bits.poke(beat.U); dut.clock.step(1)
      dut.io.data_i.valid.poke(false)
    }
    threads = threads.fork {
      while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
      out = dut.io.data_o.bits.peekInt()
      dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false)
    }
    threads.joinAndStep()
    dut.io.data_o.ready.poke(true)
    var w = 0; while (dut.io.busy_o.peekBoolean() && w < 200) { dut.clock.step(1); w += 1 }
    dut.io.data_o.ready.poke(false)
    out
  }

  "StreamMapRt_runtime" should "map a*x+b at FP16 and identity at FP8 on ONE netlist (cl=64)" in {
    test(new DataPathExtensionHarness(new HasStreamMapRt(computeLanes = 64, func = Seq("LINEAR"))))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x5a)
        // --- FP16 task: a*x+b over 32 lanes ---
        val (a, b) = (2.0f, -1.5f)
        val xs   = Seq.fill(32)(f32ToF16bits(rng.between(-4, 4) + rng.nextInt(4) * 0.25f))
        val o16  = runBeat(dut, a, b, fmt = 0, func = 0, packF16(xs))
        var maxErr = 0.0f
        for (i <- 0 until 32) {
          val got  = f16bitsToF32(((o16 >> (16 * i)) & 0xffff).toInt)
          val gold = f16bitsToF32(f32ToF16bits(a * f16bitsToF32(xs(i)) + b))
          val e = math.abs(got - gold); if (e > maxErr) maxErr = e
          assert(e <= math.max(math.abs(gold) * 0.01f, 0.02f), s"FP16 lane $i hw=$got gold=$gold")
        }
        // --- FP8 task on the SAME instance: identity (a=1,b=0), normal FP8 values ---
        val bytes = Seq.fill(64) { (if (rng.nextBoolean()) 0x80 else 0) | ((1 + rng.nextInt(30)) << 2) | rng.nextInt(4) }
        val o8    = runBeat(dut, 1.0f, 0.0f, fmt = 2, func = 0, packF8(bytes))
        assert(o8 == packF8(bytes), "FP8 identity round-trip mismatch on the shared netlist")
        // --- and back to FP16 again, to prove the fmt switch is stateless ---
        val o16b = runBeat(dut, 1.0f, 0.0f, fmt = 0, func = 0, packF16(xs))
        assert(o16b == packF16(xs), "FP16 identity round-trip mismatch after an FP8 task")
        println(f"[StreamMapRt] ONE netlist: FP16 a*x+b maxErr=$maxErr%.4g, FP8 identity OK, FP16-after-FP8 OK")
      }
  }

  "StreamMapRt_roofline" should "reach ~1 beat/cycle (512 b/cyc) at computeLanes=64" in {
    test(new DataPathExtensionHarness(new HasStreamMapRt(computeLanes = 64, func = Seq("LINEAR"))))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng   = new Random(0x6b)
        val nIn   = 256
        val beats = Seq.fill(nIn)(packF8(Seq.fill(64)(rng.nextInt(120))))
        dut.io.csr_i(0).poke(f32bits(2.0f).U); dut.io.csr_i(1).poke(f32bits(-1.5f).U); dut.io.csr_i(2).poke(csr2(2, 0).U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.data_o.ready.poke(true)
        dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(beats(0).U)
        var cyc = 0L; var fed = 0; var got = 0; var warmFeed = -1L; var lastFeed = -1L
        val maxCyc = nIn.toLong * 40 + 400
        while (got < nIn && cyc < maxCyc) {
          val canFeed = fed < nIn && dut.io.data_i.ready.peekBoolean()
          val outNow  = dut.io.data_o.valid.peekBoolean()
          dut.clock.step(1); cyc += 1
          if (canFeed) { fed += 1; if (fed == 8) warmFeed = cyc
            if (fed < nIn) dut.io.data_i.bits.poke(beats(fed).U) else dut.io.data_i.valid.poke(false); lastFeed = cyc }
          if (outNow) got += 1
        }
        assert(got == nIn, s"only $got/$nIn outputs after $cyc cyc")
        val beatSpan = (lastFeed - warmFeed).toDouble / (nIn - 8)
        val util = 1.0 / beatSpan
        println(f"[StreamMapRt roofline cl=64] beatSpan=$beatSpan%.2f util=$util%.3f effBW=${util * 512}%.0f b/cyc")
        assert(util > 0.9, s"expected ~1.0 util at cl=64, got $util")
      }
  }

  def erf(x: Double): Double = {
    val t = 1.0 / (1.0 + 0.3275911 * math.abs(x))
    val y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t * math.exp(-x * x)
    if (x >= 0) y else -y
  }
  def geluRef(x: Double): Double = x * 0.5 * (1.0 + erf(x / math.sqrt(2.0)))
  def f16mono(h: Int): Int = { val m = h & 0x7fff; if ((h & 0x8000) != 0) 0x8000 - m else 0x8000 + m }

  "StreamMapRt_gelu" should "compute GELU at FP16 (<=2 ULP) and hold the roofline at cl=64" in {
    test(new DataPathExtensionHarness(new HasStreamMapRt(computeLanes = 64, func = Seq("LINEAR", "GELU"))))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x9e)
        // --- correctness: gelu(x) over 32 FP16 lanes (func=3, a=1, b=0) ---
        val xs  = Seq.fill(32)(f32ToF16bits(rng.between(-5f, 5f)))
        val o16 = runBeat(dut, 1.0f, 0.0f, fmt = 0, func = 3, packF16(xs))
        var worst = 0
        for (i <- 0 until 32) {
          val g16 = f32ToF16bits(geluRef(f16bitsToF32(xs(i)).toDouble).toFloat)
          if ((g16 & 0x7c00) != 0) {
            val h16 = ((o16 >> (16 * i)) & 0xffff).toInt
            val ulp = math.abs(f16mono(h16) - f16mono(g16)); if (ulp > worst) worst = ulp
            assert(ulp <= 2, s"GELU lane $i hw=0x$h16%04x gold=0x$g16%04x ulp=$ulp")
          }
        }
        // --- roofline: GELU (func=3) at cl=64, FP8, must still stream 1 beat/cycle ---
        val nIn   = 256
        val beats = Seq.fill(nIn)(packF8(Seq.fill(64)(rng.nextInt(120))))
        dut.io.csr_i(0).poke(f32bits(1.0f).U); dut.io.csr_i(1).poke(f32bits(0.0f).U); dut.io.csr_i(2).poke(csr2(2, 3).U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.data_o.ready.poke(true)
        dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(beats(0).U)
        var cyc = 0L; var fed = 0; var got = 0; var warmFeed = -1L; var lastFeed = -1L
        val maxCyc = nIn.toLong * 40 + 400
        while (got < nIn && cyc < maxCyc) {
          val canFeed = fed < nIn && dut.io.data_i.ready.peekBoolean()
          val outNow  = dut.io.data_o.valid.peekBoolean()
          dut.clock.step(1); cyc += 1
          if (canFeed) { fed += 1; if (fed == 8) warmFeed = cyc
            if (fed < nIn) dut.io.data_i.bits.poke(beats(fed).U) else dut.io.data_i.valid.poke(false); lastFeed = cyc }
          if (outNow) got += 1
        }
        assert(got == nIn, s"only $got/$nIn outputs")
        val util = (nIn - 8).toDouble / (lastFeed - warmFeed)
        println(f"[StreamMapRt GELU] FP16 worst ULP=$worst; roofline cl=64 util=$util%.3f")
        assert(util > 0.9, s"GELU roofline broke: util=$util")
      }
  }

  "StreamMapRt_runsub_area" should "let FP16 reach util=1.0 at cl=32 (half the lanes) via runtime subCycles" in {
    test(new DataPathExtensionHarness(new HasStreamMapRt(computeLanes = 32, func = Seq("LINEAR"))))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        def measure(fmt: Int, beat: BigInt): Double = {
          val nIn = 200
          dut.io.csr_i(0).poke(f32bits(1.0f).U); dut.io.csr_i(1).poke(f32bits(0.0f).U); dut.io.csr_i(2).poke(csr2(fmt, 0).U)
          dut.io.enable_i.poke(true)
          dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
          dut.io.data_o.ready.poke(true)
          dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(beat.U)
          var cyc = 0L; var fed = 0; var got = 0; var warm = -1L; var last = -1L
          while (got < nIn && cyc < nIn * 40 + 400) {
            val canFeed = fed < nIn && dut.io.data_i.ready.peekBoolean()
            val outNow  = dut.io.data_o.valid.peekBoolean()
            dut.clock.step(1); cyc += 1
            if (canFeed) { fed += 1; if (fed == 8) warm = cyc; last = cyc
              if (fed < nIn) dut.io.data_i.bits.poke(beat.U) else dut.io.data_i.valid.poke(false) }
            if (outNow) got += 1
          }
          dut.io.data_i.valid.poke(false)
          var w = 0; while (dut.io.busy_o.peekBoolean() && w < 200) { dut.clock.step(1); w += 1 }
          (nIn - 8).toDouble / (last - warm)
        }
        val rng = new Random(0x1c)
        val u16 = measure(0, packF16(Seq.fill(32)(f32ToF16bits(rng.between(-2f, 2f)))))
        val u8  = measure(2, packF8(Seq.fill(64)(rng.nextInt(120))))
        println(f"[StreamMapRt runSub cl=32] FP16 util=$u16%.3f (32 active lanes -> roofline), FP8 util=$u8%.3f (64 lanes -> 0.5)")
        assert(u16 > 0.9, s"FP16 should hit util=1.0 at cl=32 via runSub, got $u16")
        assert(u8 > 0.4 && u8 < 0.65, s"FP8 should sit ~0.5 at cl=32 (needs cl=64), got $u8")
      }
  }

  "StreamMapRt_mxfp8_roofline" should "hold the 512 b/cyc roofline at cl=64 for MXFP8 (block-scaled)" in {
    test(new DataPathExtensionHarness(new HasStreamMapRt(computeLanes = 64, func = Seq("LINEAR"))))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng   = new Random(0xa5)
        val nIn   = 256
        val beats = Seq.fill(nIn)(packF8(Seq.fill(64)(rng.nextInt(120))))
        // csr2 = func(0=LINEAR) | fmt(3=MXFP8_E5M2)<<2 | scale(130 = 2^3)<<5  (MX input widened by 2^(130-127))
        val csr2mx = BigInt(0) | (BigInt(3) << 2) | (BigInt(130) << 5)
        dut.io.csr_i(0).poke(f32bits(1.0f).U); dut.io.csr_i(1).poke(f32bits(0.0f).U); dut.io.csr_i(2).poke(csr2mx.U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.data_o.ready.poke(true)
        dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(beats(0).U)
        var cyc = 0L; var fed = 0; var got = 0; var warmFeed = -1L; var lastFeed = -1L
        val maxCyc = nIn.toLong * 40 + 400
        while (got < nIn && cyc < maxCyc) {
          val canFeed = fed < nIn && dut.io.data_i.ready.peekBoolean()
          val outNow  = dut.io.data_o.valid.peekBoolean()
          dut.clock.step(1); cyc += 1
          if (canFeed) { fed += 1; if (fed == 8) warmFeed = cyc
            if (fed < nIn) dut.io.data_i.bits.poke(beats(fed).U) else dut.io.data_i.valid.poke(false); lastFeed = cyc }
          if (outNow) got += 1
        }
        assert(got == nIn, s"only $got/$nIn outputs")
        val util = (nIn - 8).toDouble / (lastFeed - warmFeed)
        println(f"[StreamMapRt MXFP8] roofline cl=64 util=$util%.3f (edge-only widenMX -> roofline preserved)")
        assert(util > 0.9, s"MXFP8 roofline broke: util=$util")
      }
  }
}
