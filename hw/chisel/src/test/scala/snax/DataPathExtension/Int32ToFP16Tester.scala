// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi <xiaoling.yi@kuleuven.be>

package snax.DataPathExtension

import chisel3._
import chiseltest._
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
  it should "convert int32 to fp16 correctly for several values" in {
    test(new Int32ToFp16) { dut =>
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
    test(new Int32ToFp16) { dut =>
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
