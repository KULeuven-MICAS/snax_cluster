// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi <xiaoling.yi@kuleuven.be>

package snax.DataPathExtension

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

class Int32ToFp16Spec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "Int32ToFp16"

  // -----------------------------
  // Reference FP16 encoding (Scala)
  // -----------------------------
  def intToFp16Ref(x: Int): Int = intToFp16RefShift(x, 0)

  // RNE(x * 2^-k) in FP16, k clamped to 14 like the RTL. A power of two only lowers the exponent, so this is the
  // unshifted reference with (expRaw - k); with k <= 14 the result is never subnormal.
  def intToFp16RefShift(x: Int, kIn: Int): Int = {
    val k = math.min(kIn, 14)
    // Handle signed → absolute value
    val sign = if (x < 0) 1 else 0
    val abs  = Math.abs(x.toLong) // use long to avoid overflow on Int.MinValue

    if (abs == 0) return sign << 15

    // Find MSB index
    val msbIndex = 63 - java.lang.Long.numberOfLeadingZeros(abs)

    val expUnbiased = msbIndex
    val expBias     = 15
    val expRaw      = expUnbiased + expBias - k

    // Overflow → ±Inf
    if (expRaw >= 31)
      return (sign << 15) | (0x1F << 10)

    // Normalize abs (shift so MSB goes to bit 31)
    val shiftAmt = 31 - msbIndex
    val magNorm  = (abs << shiftAmt) & 0xFFFFFFFFL

    // Extract fraction + GRS
    val frac      = ((magNorm >> 21) & 0x3FF).toInt
    val guard     = ((magNorm >> 20) & 1) != 0
    val roundBit  = ((magNorm >> 19) & 1) != 0
    val sticky    = (magNorm & ((1 << 19) - 1)) != 0
    val lsb       = (frac & 1) != 0

    val increment = guard && (roundBit || sticky || lsb)

    var fracRounded = frac + (if (increment) 1 else 0)
    var expField    = expRaw

    // Mantissa overflow
    if (fracRounded == 1024) {
      fracRounded = 0
      expField += 1
    }

    // Overflow to Inf
    if (expField >= 31)
      return (sign << 15) | (0x1F << 10)

    (sign << 15) | (expField << 10) | fracRounded
  }

  // -----------------------------
  // TESTS
  // -----------------------------
  /** The handshake, under a SLOW consumer.
    *
    * The tests below drive the bare PE, so this is the one that covers the extension's own valid/ready logic.
    * The hazard it guards appears only when the consumer back-pressures: if the "group complete" condition
    * were gated on ext_data_i.valid while the counter advances on .fire, a completed group waiting for a slow
    * consumer would re-raise the output valid every stalled cycle and emit DUPLICATE beats. Feeding the writer
    * that almost never happens; feeding a time-muxed StreamReduce (one beat every 4 cycles) it floods, and the
    * frame count downstream never closes -- a hang, not a wrong answer.
    *
    * The assertion that matters is the output beat COUNT: 2 INT32 beats in must be exactly 1 FP16 beat out,
    * no matter how slowly the consumer drains.
    */
  it should "emit exactly one beat per two inputs under a stalling consumer" in {
    test(new DataPathExtensionHarness(new HasInt32ToFp16Converter(dataWidth = 512)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        val nIn = 16 // 16 INT32 beats -> 8 FP16 beats
        dut.clock.setTimeout(0)
        dut.io.csr_i(0).poke(0.U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)

        var produced = 0
        var threads  = new chiseltest.internal.TesterThreadList(Seq())
        threads = threads.fork {
          dut.io.data_i.valid.poke(true)
          for (b <- 0 until nIn) {
            // 16 lanes of int32 per 512-bit beat
            val beat = (0 until 16).foldLeft(BigInt(0)) { (acc, i) =>
              acc | (BigInt(b * 16 + i) << (32 * i))
            }
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
            dut.io.data_i.bits.poke(beat)
            dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
        }
        threads = threads.fork {
          // Drain slowly -- ready one cycle in four, like a cl=8 operator downstream -- and keep going
          // well past the last expected beat, so a duplicate would be COUNTED rather than missed.
          var idle = 0
          while (idle < 60) {
            if (dut.io.data_o.valid.peekBoolean()) {
              dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false)
              produced += 1; idle = 0
              dut.clock.step(3) // the stall that exposed the bug
            } else { dut.clock.step(1); idle += 1 }
          }
        }
        threads.joinAndStep()
        assert(produced == nIn / 2, s"expected ${nIn / 2} output beats, got $produced (duplicates = handshake bug)")
      }
  }

  it should "convert int32 to fp16 correctly for several values" in {
    test(new Int32ToFp16PE) { dut =>
      val testValues = Seq(
        0, 1, -1,
        123, -123,
        1000, -1000,
        65504, -65504,   // largest fp16 finite
        Int.MaxValue,
        Int.MinValue,
        1 << 20,
        1 << 30
      )

      dut.io.shift.poke(0.U)
      for (v <- testValues) {
        dut.io.in.poke(v.S)
        dut.clock.step()
        val hw = dut.io.out.peek().litValue.toInt
        val sw = intToFp16Ref(v)
        assert(hw == sw, f"Input: $v HW=${hw.toHexString} SW=${sw.toHexString}")
      }
    }
  }

  it should "pass 10k randomized tests" in {
    test(new Int32ToFp16PE) { dut =>
      val rand = new scala.util.Random(0)
      dut.io.shift.poke(0.U)
      for (_ <- 0 until 10000) {
        val v = rand.nextInt()
        dut.io.in.poke(v.S)
        dut.clock.step()
        val hw = dut.io.out.peek().litValue.toInt
        val sw = intToFp16Ref(v)
        assert(hw == sw, f"Random input: $v HW=${hw.toHexString} SW=${sw.toHexString}")
      }
    }
  }

  // -----------------------------
  // POWER-OF-TWO SCALE (shift)
  // -----------------------------
  /** Hard-coded against numpy: float16(float64(v) * 2^-k), a single RNE of the exactly scaled value. These pin the
    * three things the shift is for: a full-range INT8 dot product (2,064,512) that is Inf at k = 0 fits once
    * shifted; ties still go to even after the shift; and k = 14 on the smallest input gives exactly FP16's
    * smallest NORMAL number, so no subnormal path is needed. k = 15 must behave as 14 (the clamp).
    */
  it should "apply a power-of-two scale: RNE(v * 2^-k), matching numpy" in {
    test(new Int32ToFp16PE) { dut =>
      val cases = Seq(
        (3001, 9, 0x45dc),        // 5.859375 (tie -> even)
        (2064512, 0, 0x7c00),     // +Inf without the shift
        (2064512, 6, 0x77e0),     // 32256
        (-2064512, 6, 0xf7e0),    // -32256
        (2064512, 14, 0x57e0),    // 126
        (65520, 0, 0x7c00),       // the first integer that overflows
        (65520, 1, 0x7800),       // 32768: tie rounds up to the even fraction
        (40016, 5, 0x64e2),       // 1250 (1250.5, tie -> even)
        (1, 14, 0x0400),          // 2^-14 = smallest normal
        (-1, 14, 0x8400),
        (1, 15, 0x0400),          // clamped to 14
        (Int.MaxValue, 14, 0x7c00),
        (Int.MinValue, 14, 0xfc00)
      )
      for ((v, k, want) <- cases) {
        dut.io.in.poke(v.S)
        dut.io.shift.poke(k.U)
        dut.clock.step()
        val hw = dut.io.out.peek().litValue.toInt
        assert(hw == want, f"v=$v k=$k HW=0x${hw}%04x want=0x${want}%04x")
      }
    }
  }

  it should "match the reference on 20k random (value, shift) pairs" in {
    test(new Int32ToFp16PE) { dut =>
      val rand = new scala.util.Random(1)
      for (i <- 0 until 20000) {
        // half full-range INT32, half in the INT8-dot-product range where the shift matters
        val v = if (i % 2 == 0) rand.nextInt() else rand.nextInt(4200000) - 2100000
        val k = rand.nextInt(16)
        dut.io.in.poke(v.S)
        dut.io.shift.poke(k.U)
        dut.clock.step()
        val hw = dut.io.out.peek().litValue.toInt
        val sw = intToFp16RefShift(v, k)
        assert(hw == sw, f"v=$v k=$k HW=${hw.toHexString} SW=${sw.toHexString}")
      }
    }
  }

  /** The CSR path end to end: with `shift = 1` built, csr(1)[3:0] reaches every PE. Two INT32 beats in, one FP16
    * beat out, every lane RNE(x * 2^-k).
    */
  it should "take the shift from csr(1) in the shift build" in {
    test(new DataPathExtensionHarness(new HasInt32ToFp16Converter(dataWidth = 512, shift = 1)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        for (k <- Seq(0, 6, 9, 14)) {
          val rand = new scala.util.Random(k)
          val vals = Seq.fill(32)(rand.nextInt(4128000) - 2064000)
          dut.io.csr_i(0).poke(0.U)
          dut.io.csr_i(1).poke(k.U)
          dut.io.enable_i.poke(true)
          dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
          dut.io.data_o.ready.poke(true)
          var got: Option[BigInt] = None
          for (b <- 0 until 2) {
            val beat = (0 until 16).foldLeft(BigInt(0)) { (acc, i) =>
              acc | ((BigInt(vals(b * 16 + i)) & 0xFFFFFFFFL) << (32 * i))
            }
            dut.io.data_i.bits.poke(beat)
            dut.io.data_i.valid.poke(true)
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
            dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
          var t = 0
          while (got.isEmpty && t < 20) {
            if (dut.io.data_o.valid.peekBoolean()) got = Some(dut.io.data_o.bits.peek().litValue)
            dut.clock.step(1); t += 1
          }
          assert(got.isDefined, s"k=$k: no output beat")
          for (i <- 0 until 32) {
            val hw = ((got.get >> (16 * i)) & 0xFFFF).toInt
            val sw = intToFp16RefShift(vals(i), k)
            assert(hw == sw, f"k=$k lane $i v=${vals(i)} HW=0x${hw}%04x SW=0x${sw}%04x")
          }
        }
      }
  }
}
