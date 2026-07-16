package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** StreamElementwiseRt: one elaborated netlist combines `operandCount` beats element-wise (out = b0 op b1 op
  * ...) at FP16 / BF16 / FP8 chosen at runtime by the `fmt` CSR field. Proves (a) correctness of the FP32
  * combine at FP16 for MUL and ADD (32 lanes), (b) an FP8 identity round-trip on the SAME dut instance (the
  * runtime fmt switch is stateless), and (c) full 512 b/cyc input throughput at computeLanes=maxLanes (=64).
  */
class StreamElementwiseRtTester extends AnyFlatSpec with ChiselScalatestTester {

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
  def packF16(v: Seq[Int]): BigInt = v.zipWithIndex.foldLeft(BigInt(0)) { case (a, (h, i)) => a | (BigInt(h & 0xffff) << (16 * i)) }
  def packF8(v: Seq[Int]):  BigInt = v.zipWithIndex.foldLeft(BigInt(0)) { case (a, (x, i)) => a | (BigInt(x & 0xff) << (8 * i)) }
  // csr(0) = operandCount | fmt<<16 ; csr(1) = op (0=MUL, 1=ADD)
  def csr0(oc: Int, fmt: Int): BigInt = BigInt(oc) | (BigInt(fmt) << 16)
  def normalF8(rng: Random): Int = (if (rng.nextBoolean()) 0x80 else 0) | ((1 + rng.nextInt(30)) << 2) | rng.nextInt(4)

  /** Drive `beats.length` operand beats through the extension and return the single output beat. */
  def runEw(dut: DataPathExtensionHarness, fmt: Int, op: Int, beats: Seq[BigInt]): BigInt = {
    dut.io.csr_i(0).poke(csr0(beats.length, fmt).U); dut.io.csr_i(1).poke(BigInt(op).U)
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
    var w = 0; while (dut.io.busy_o.peekBoolean() && w < 200) { dut.clock.step(1); w += 1 }
    dut.io.data_o.ready.poke(false)
    out
  }

  "StreamElementwiseRt_runtime" should "combine at FP16 (MUL/ADD) and identity at FP8 on ONE netlist (cl=64)" in {
    test(new DataPathExtensionHarness(new HasStreamElementwiseRt(computeLanes = 64)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng = new Random(0x3c)
        // --- FP16 MUL: out(i) = x0(i) * x1(i) over 32 lanes ---
        val x0 = Seq.fill(32)(f32ToF16bits(rng.between(-3f, 3f)))
        val x1 = Seq.fill(32)(f32ToF16bits(rng.between(-3f, 3f)))
        val oMul = runEw(dut, fmt = 0, op = 0, Seq(packF16(x0), packF16(x1)))
        var maxErrMul = 0.0f
        for (i <- 0 until 32) {
          val got  = f16bitsToF32(((oMul >> (16 * i)) & 0xffff).toInt)
          val gold = f16bitsToF32(f32ToF16bits(f16bitsToF32(x0(i)) * f16bitsToF32(x1(i))))
          val e = math.abs(got - gold); if (e > maxErrMul) maxErrMul = e
          assert(e <= math.max(math.abs(gold) * 0.02f, 0.02f), s"FP16 MUL lane $i hw=$got gold=$gold")
        }
        // --- FP16 ADD on the SAME instance: out(i) = x0(i) + x1(i) ---
        val oAdd = runEw(dut, fmt = 0, op = 1, Seq(packF16(x0), packF16(x1)))
        var maxErrAdd = 0.0f
        for (i <- 0 until 32) {
          val got  = f16bitsToF32(((oAdd >> (16 * i)) & 0xffff).toInt)
          val gold = f16bitsToF32(f32ToF16bits(f16bitsToF32(x0(i)) + f16bitsToF32(x1(i))))
          val e = math.abs(got - gold); if (e > maxErrAdd) maxErrAdd = e
          assert(e <= math.max(math.abs(gold) * 0.02f, 0.02f), s"FP16 ADD lane $i hw=$got gold=$gold")
        }
        // --- FP8 identity (operandCount=1) on the SAME instance: seed then narrow => round-trip ---
        val bytes = Seq.fill(64)(normalF8(rng))
        val o8    = runEw(dut, fmt = 2, op = 0, Seq(packF8(bytes)))
        assert(o8 == packF8(bytes), "FP8 identity round-trip mismatch on the shared netlist")
        // --- back to FP16 identity (operandCount=1) to prove the fmt switch is stateless ---
        val o16b = runEw(dut, fmt = 0, op = 0, Seq(packF16(x0)))
        assert(o16b == packF16(x0), "FP16 identity round-trip mismatch after an FP8 task")
        println(f"[StreamElementwiseRt] ONE netlist: FP16 MUL maxErr=$maxErrMul%.4g ADD maxErr=$maxErrAdd%.4g, FP8 identity OK")
      }
  }

  "StreamElementwiseRt_roofline" should "reach ~1 input-beat/cycle (512 b/cyc) at computeLanes=64" in {
    test(new DataPathExtensionHarness(new HasStreamElementwiseRt(computeLanes = 64)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        val rng   = new Random(0x77)
        val oc    = 4                       // 4 beats combined per output row
        val nRows = 64
        val nIn   = oc * nRows
        val beats = Seq.fill(nIn)(packF8(Seq.fill(64)(normalF8(rng))))
        dut.io.csr_i(0).poke(csr0(oc, 2).U); dut.io.csr_i(1).poke(BigInt(0).U) // FP8, MUL
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
        dut.io.data_o.ready.poke(true)
        dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(beats(0).U)
        var cyc = 0L; var fed = 0; var warmFeed = -1L; var lastFeed = -1L
        val maxCyc = nIn.toLong * 40 + 400
        while (fed < nIn && cyc < maxCyc) {
          val canFeed = dut.io.data_i.ready.peekBoolean()
          dut.clock.step(1); cyc += 1
          if (canFeed) { fed += 1; if (fed == 8) warmFeed = cyc
            if (fed < nIn) dut.io.data_i.bits.poke(beats(fed).U) else dut.io.data_i.valid.poke(false); lastFeed = cyc }
        }
        assert(fed == nIn, s"only fed $fed/$nIn after $cyc cyc")
        val beatSpan = (lastFeed - warmFeed).toDouble / (nIn - 8)
        val util = 1.0 / beatSpan
        println(f"[StreamElementwiseRt roofline cl=64] beatSpan=$beatSpan%.2f util=$util%.3f effBW=${util * 512}%.0f b/cyc")
        assert(util > 0.9, s"expected ~1.0 input util at cl=64, got $util")
      }
  }
}
