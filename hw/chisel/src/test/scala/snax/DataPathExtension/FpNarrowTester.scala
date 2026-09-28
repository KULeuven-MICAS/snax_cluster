// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.DataPathExtension

import java.math.{BigDecimal => JBigDecimal, RoundingMode}

import chisel3._
import chiseltest._
import fp_unit._
import org.scalatest.flatspec.AnyFlatSpec

/** FpHelpers.narrow, the FP32 -> transport conversion at the output of every SIMD operator, checked against two
  * independent references:
  *
  *   - an EXACT one: the FP32 value as a BigDecimal, divided by the target's grid spacing at its exponent and
  *     rounded half-to-even (IEEE round-to-nearest-even, subnormals included) -- for FP16, BF16 and FP8 (E5M2);
  *   - for FP16 also java.lang.Float.floatToFloat16, the JDK's IEEE conversion.
  *
  * The inputs are random bit patterns (every exponent, NaN and Inf included) plus the cases a narrowing gets wrong:
  * exact ties, ties decided by the last FP32 bit, every subnormal binade, the half-ULP boundaries of the smallest
  * subnormal and of the largest finite value.
  */
class NarrowHarness(t: FpType) extends Module {
  val io = IO(new Bundle {
    val in  = Input(UInt(32.W))
    val out = Output(UInt(t.width.W))
  })
  io.out := FpHelpers.narrow(io.in, t)
}

class FpNarrowSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "FpHelpers.narrow"

  /** IEEE RNE from FP32 bits to a format with `e` exponent and `m` fraction bits, by exact arithmetic. */
  def refNarrow(bits: Int, e: Int, m: Int): Int = {
    val width   = 1 + e + m
    val bias    = (1 << (e - 1)) - 1
    val expMax  = (1 << e) - 1
    val sign    = (bits >>> 31) & 1
    val exp32   = (bits >>> 23) & 0xFF
    val man32   = bits & 0x7FFFFF
    val signBit = sign << (width - 1)
    if (exp32 == 0xFF && man32 != 0) return (expMax << m) | (1 << (m - 1)) // canonical qNaN, sign 0
    if (exp32 == 0xFF) return signBit | (expMax << m)
    if (exp32 == 0 && man32 == 0) return signBit
    val v       = new JBigDecimal(java.lang.Float.intBitsToFloat(bits & 0x7FFFFFFF).toDouble) // exact, |x|
    // the leading bit's unbiased exponent of |x|
    val lead    = if (exp32 != 0) exp32 - 127 else (-126 - 23 + (31 - Integer.numberOfLeadingZeros(man32)))
    val minNorm = 1 - bias
    val q       = if (lead >= minNorm) lead - m else minNorm - m // the grid's exponent at |x|
    val two     = new JBigDecimal(2)
    val spacing = if (q >= 0) two.pow(q) else JBigDecimal.ONE.divide(two.pow(-q))
    val n       = v.divide(spacing).setScale(0, RoundingMode.HALF_EVEN).toBigInteger // exact division: a power of 2
    if (n.signum == 0) return signBit
    // re-encode n * 2^q: the rounding may have carried into the next binade
    val nb      = n.bitLength
    val lead2   = q + nb - 1
    if (lead2 > expMax - 1 - bias) return signBit | (expMax << m)
    if (lead2 < minNorm) return signBit | n.intValue // subnormal: the field is n itself
    val frac    = n.shiftLeft(m + 1 - nb).intValue & ((1 << m) - 1)
    signBit | ((lead2 + bias) << m) | frac
  }

  def f32(x: Float): Int = java.lang.Float.floatToRawIntBits(x)

  /** The inputs: random patterns plus every hazard of a narrowing, for a format with `e`/`m`. */
  def vectors(e: Int, m: Int, seed: Int): Seq[Int] = {
    val rnd   = new scala.util.Random(seed)
    val bias  = (1 << (e - 1)) - 1
    val out   = scala.collection.mutable.ArrayBuffer[Int]()
    out ++= Seq.fill(20000)(rnd.nextInt())
    // every binade from well below the smallest subnormal to past the largest finite value, random fractions
    for (lead <- (-bias - m - 6) to (bias + 3); _ <- 0 until 40) {
      val exp32 = lead + 127
      if (exp32 >= 1 && exp32 <= 254) out += ((rnd.nextInt(2) << 31) | (exp32 << 23) | rnd.nextInt(1 << 23))
    }
    // ties at the target's rounding bit: fraction bits below it = 100..0, decided by the target LSB, and the same
    // with only the LAST FP32 bit set below the tie (it must round up)
    val sh = 23 - m
    for (lead <- (1 - bias) to bias; lsb <- 0 to 1; last <- 0 to 1) {
      val exp32 = lead + 127
      if (exp32 >= 1 && exp32 <= 254) {
        val top = (rnd.nextInt(1 << m) & ~1) | lsb
        out += (exp32 << 23) | (top << sh) | (1 << (sh - 1)) | last
      }
    }
    // around the smallest subnormal 2^(1-bias-m): half of it (tie to 0), just above half, 1.5x (tie to 2)
    val tiny = 1 - bias - m
    for (k <- Seq(tiny - 1, tiny, tiny + 1)) {
      val exp32 = k + 127
      if (exp32 >= 1 && exp32 <= 254) {
        out += (exp32 << 23)
        out += (exp32 << 23) | 1
        out += (exp32 << 23) | (1 << 22)
      }
    }
    // the largest finite value and the half-ULP above it (rounds to Inf)
    val maxLead = (1 << e) - 2 - bias
    if (maxLead + 127 <= 254) {
      val allOnes = ((1 << m) - 1) << sh
      out += ((maxLead + 127) << 23) | allOnes
      out += ((maxLead + 127) << 23) | allOnes | (1 << (sh - 1))
      out += ((maxLead + 127) << 23) | allOnes | (1 << (sh - 1)) - 1
    }
    out ++= Seq(0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFFFFFFF, 1, 0x80000001, 0x00800000)
    out.toSeq
  }

  def check(t: FpType, seed: Int): Unit =
    test(new NarrowHarness(t)) { c =>
      var bad = 0
      for (b <- vectors(t.expWidth, t.sigWidth, seed)) {
        c.io.in.poke((b.toLong & 0xFFFFFFFFL).U)
        val got  = c.io.out.peek().litValue.toInt
        val want = refNarrow(b, t.expWidth, t.sigWidth)
        if (got != want) {
          if (bad < 8) println(f"narrow to e${t.expWidth}m${t.sigWidth}: in 0x$b%08x got 0x$got%x want 0x$want%x")
          bad += 1
        }
      }
      assert(bad == 0, s"$bad conversions differ from IEEE round-to-nearest-even")
    }

  it should "be IEEE round-to-nearest-even into FP16, subnormals included" in check(FP16, 1)
  it should "be IEEE round-to-nearest-even into BF16" in check(BF16, 2)
  it should "be IEEE round-to-nearest-even into FP8 (E5M2)" in check(FP8, 3)

  it should "agree with the JDK's floatToFloat16, the independent FP16 reference" in {
    val rnd = new scala.util.Random(4)
    var bad = 0
    for (b <- vectors(5, 10, 5) ++ Seq.fill(20000)(rnd.nextInt())) {
      val x    = java.lang.Float.intBitsToFloat(b)
      val want = if (x.isNaN) 0x7E00 else java.lang.Float.floatToFloat16(x).toInt & 0xFFFF
      if (refNarrow(b, 5, 10) != want) bad += 1
    }
    assert(bad == 0, s"the exact reference disagrees with the JDK on $bad inputs")
  }
}
