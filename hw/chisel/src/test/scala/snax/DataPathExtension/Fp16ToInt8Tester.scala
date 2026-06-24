// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Tier-1 test for Fp16ToInt8: symmetric per-tensor quantize q = sat[-127,127](round_rne(x*inv_scale)).
  * Verilator (fpnew blackboxes) with --build-jobs 1 (UNOPTFLAT parallel-build race). Two levels:
  *   1) PE-level: exact integer compare against a Scala golden (RNE + clamp-then-round + symmetric sat).
  *   2) Full harness: the 2:1 packing FSM (2 input beats of 32 FP16 -> 1 output beat of 64 INT8).
  */
class Fp16ToInt8Tester extends AnyFlatSpec with ChiselScalatestTester {

  // --- FP helpers (shared style with StreamMapTester) ---
  def f16bitsToF32(h: Int): Float = {
    val sign = if ((h & 0x8000) != 0) -1.0 else 1.0
    val exp  = (h >> 10) & 0x1f
    val mant = h & 0x3ff
    val v: Double =
      if (exp == 0) sign * mant * math.pow(2, -24)
      else if (exp == 0x1f) if (mant == 0) sign * Double.PositiveInfinity else Double.NaN
      else sign * (1024 + mant) * math.pow(2, exp - 25)
    v.toFloat
  }
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
    var h   = sign | (exp << 10) | (mant >>> 13)
    val rem = mant & 0x1fff; if (rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1)) h += 1; h
  }
  def f32bits(f: Float): BigInt = BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)

  // --- Golden: clamp-then-round, exactly mirroring the HW PE ---
  def quantRef(h: Int, invScaleBits: Long): Int = {
    val x = f16bitsToF32(h).toDouble
    val s = java.lang.Float.intBitsToFloat(invScaleBits.toInt).toDouble
    val v = math.max(-128.0, math.min(128.0, x * s)) // clamp before the round, like the PE
    val r = math.rint(v).toInt                        // round-to-nearest-even
    math.max(-127, math.min(127, r))                  // symmetric saturate
  }

  val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  // FP16 bit patterns drawn from a bounded range (avoid NaN/Inf), plus exact edges.
  def sampleFp16(rng: Random): Int =
    f32ToF16bits((rng.between(-4.0, 4.0) + rng.nextInt(8) * 0.125).toFloat)

  val scales = Seq(
    f32bits(1.0f).toLong,   // identity
    f32bits(16.0f).toLong,  // silu / swiglu measurement scale
    f32bits(64.0f).toLong,  // rmsnorm measurement scale
    f32bits(127.0f).toLong  // softmax (P in [0,1])
  )

  behavior of "Fp16ToInt8"

  it should "quantize a single FP16 lane (RNE + symmetric saturate) bit-exactly" in {
    test(new Fp16ToInt8PE).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng     = new Random(0xF16)
      // edge inputs: +0, -0, +1.0, -1.0, +0.5, -0.5, a large value (saturates), smallest normal
      val edges   = Seq(0x0000, 0x8000, 0x3c00, 0xbc00, 0x3800, 0xb800, 0x6000, 0xe000, 0x0400)
      val randoms = Seq.fill(400)(sampleFp16(rng))
      for (s <- scales; h <- edges ++ randoms) {
        dut.io.in.poke(h.U(16.W))
        dut.io.inv_scale.poke(BigInt(s).U(32.W))
        dut.clock.step()
        val hw = dut.io.out.peek().litValue.toInt
        val sw = quantRef(h, s)
        assert(
          hw == sw,
          f"in=0x$h%04x (${f16bitsToF32(h)}) scaleBits=0x$s%08x : HW=$hw SW=$sw"
        )
      }
    }
  }

  // Pack `beats` 512b FP16 input beats (even count) into beats/2 INT8 output beats.
  def runHarness(invScaleBits: BigInt, inputBeats: Seq[BigInt], computeLanes: Int = 0): Seq[BigInt] = {
    var outs = Seq[BigInt]()
    test(new DataPathExtensionHarness(new HasFp16ToInt8(16, 8, 512, computeLanes)))
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, flags)) { dut =>
        dut.io.csr_i(0).poke(invScaleBits.U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)

        var threads = new chiseltest.internal.TesterThreadList(Seq())
        threads = threads.fork {
          dut.io.data_i.valid.poke(true)
          for (bt <- inputBeats) {
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
            dut.io.data_i.bits.poke(bt); dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
        }
        threads = threads.fork {
          for (_ <- 0 until inputBeats.length / 2) {
            while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
            outs = outs :+ dut.io.data_o.bits.peekInt()
            dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false)
          }
        }
        threads.joinAndStep()
      }
    outs
  }

  def packFp16(lanes: Seq[Int]): BigInt =
    lanes.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (h, i)) => acc | (BigInt(h & 0xffff) << (16 * i)) }
  def lanesOf(beat: BigInt): Seq[Int] = (0 until 32).map(i => ((beat >> (16 * i)) & 0xffff).toInt)
  def packInt8(bytes: Seq[Int]): BigInt =
    bytes.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (b, i)) => acc | (BigInt(b & 0xff) << (8 * i)) }

  it should "pack 2 FP16 beats into 1 INT8 beat (32 lanes each, little-endian)" in {
    val rng        = new Random(0xBEE)
    val scaleBits  = f32bits(16.0f)
    val nBeats     = 6 // even
    val inputBeats = Seq.fill(nBeats)(packFp16(Seq.fill(32)(sampleFp16(rng))))

    val hw = runHarness(scaleBits, inputBeats)
    val gd = inputBeats.grouped(2).map { pair =>
      val bytes = lanesOf(pair(0)).map(h => quantRef(h, scaleBits.toLong)) ++
        lanesOf(pair(1)).map(h => quantRef(h, scaleBits.toLong))
      packInt8(bytes)
    }.toSeq

    assert(hw.length == gd.length, s"got ${hw.length} output beats, expected ${gd.length}")
    for (i <- hw.indices)
      assert(hw(i) == gd(i), f"beat $i mismatch:\n  HW=0x${hw(i).toString(16)}\n  SW=0x${gd(i).toString(16)}")
  }

  // Time-mux config used by the cfg (computeLanes=8 -> 4 sub-cycles per input beat): same packed result.
  it should "pack correctly in the time-mux FSM (computeLanes=8)" in {
    val rng        = new Random(0xBEE)
    val scaleBits  = f32bits(16.0f)
    val nBeats     = 6 // even
    val inputBeats = Seq.fill(nBeats)(packFp16(Seq.fill(32)(sampleFp16(rng))))

    val hw = runHarness(scaleBits, inputBeats, computeLanes = 8)
    val gd = inputBeats.grouped(2).map { pair =>
      val bytes = lanesOf(pair(0)).map(h => quantRef(h, scaleBits.toLong)) ++
        lanesOf(pair(1)).map(h => quantRef(h, scaleBits.toLong))
      packInt8(bytes)
    }.toSeq

    assert(hw.length == gd.length, s"got ${hw.length} output beats, expected ${gd.length}")
    for (i <- hw.indices)
      assert(hw(i) == gd(i), f"beat $i (cl=8) mismatch:\n  HW=0x${hw(i).toString(16)}\n  SW=0x${gd(i).toString(16)}")
  }
}
