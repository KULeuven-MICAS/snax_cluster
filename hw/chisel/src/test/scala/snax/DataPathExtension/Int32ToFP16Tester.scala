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
  def intToFp16Ref(x: Int): Int = {
    // Handle signed → absolute value
    val sign = if (x < 0) 1 else 0
    val abs  = Math.abs(x.toLong) // use long to avoid overflow on Int.MinValue

    if (abs == 0) return sign << 15

    // Find MSB index
    val msbIndex = 63 - java.lang.Long.numberOfLeadingZeros(abs)

    val expUnbiased = msbIndex
    val expBias     = 15
    val expRaw      = expUnbiased + expBias

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
    * The tests below drive the bare PE, so the extension's own valid/ready logic was never covered. It had
    * a bug that only appears when the consumer back-pressures: the "group complete" condition was gated on
    * ext_data_i.valid while the counter advanced on .fire, so a completed group waiting for a slow consumer
    * re-raised the output valid every stalled cycle and emitted DUPLICATE beats. Feeding the writer that
    * almost never happens; feeding a time-muxed StreamReduce (one beat every 4 cycles) it floods and the
    * frame count downstream never closes -- which is a hang, not a wrong answer.
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
}
