// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// FpAdd against IEEE round-to-nearest-even (host float addition) when one operand is more than PREC_C + 1
// binades below the other. fp_add.sv rounds the effective subtractions of this case one ulp low (e.g.
// -2^15 + 1.5 * 2^-15 = 0xc6ffffff instead of 0xc7000000), so FpAddDiffTest skips those pairs.

package fp_native

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import fp_unit._

object FpAddFarOperandTest {
  private def field(t: FpType, x: BigInt) =
    (
      x.testBit(t.width - 1),
      ((x >> t.sigWidth) & ((BigInt(1) << t.expWidth) - 1)).toInt,
      x & ((BigInt(1) << t.sigWidth) - 1)
    )

  /** IEEE RNE sum of two finite operands, as FP32 bits. Every input format is exact in FP32. */
  def rne(tA: FpType, tB: FpType, a: BigInt, b: BigInt): BigInt = {
    def toFloat(t: FpType, x: BigInt): Float = {
      val (s, e, m) = field(t, x)
      val bias      = FpCommon.bias(t)
      val mag       =
        if (e == 0) m.toDouble * math.pow(2, 1 - bias - t.sigWidth)
        else (m.toDouble + math.pow(2, t.sigWidth)) * math.pow(2, e - bias - t.sigWidth)
      (if (s) -mag else mag).toFloat
    }
    BigInt(java.lang.Float.floatToRawIntBits(toFloat(tA, a) + toFloat(tB, b))) & 0xffffffffL
  }

  /** Non-zero finite operands with opposite signs whose exponents differ by more than PREC_C + 1, for the format
    * combinations where FpAdd reduces the far operand to a sticky bit (inputs not wider than the output).
    */
  def farSub(tA: FpType, tB: FpType, tC: FpType, a: BigInt, b: BigInt): Boolean = {
    val (sa, ea, ma) = field(tA, a)
    val (sb, eb, mb) = field(tB, b)
    val precC        = tC.sigWidth + 1
    val ok           = tA.sigWidth + 1 <= precC && tB.sigWidth + 1 <= precC
    def finite(t: FpType, e: Int) = e != (1 << t.expWidth) - 1
    val nonZero = (ea != 0 || ma != 0) && (eb != 0 || mb != 0)
    val d       = (math.max(ea, 1) - FpCommon.bias(tA)) - (math.max(eb, 1) - FpCommon.bias(tB))
    ok && nonZero && finite(tA, ea) && finite(tB, eb) && sa != sb && math.abs(d) > precC + 1
  }
}

class FpAddFarOperandTest extends AnyFlatSpec with ChiselScalatestTester {
  import FpAddFarOperandTest._

  // larger operand: random sign and exponent, a power of two half of the time; smaller operand: PREC_C - 1 to
  // PREC_C + 41 binades below it (subnormal when that is below the normal range), random sign and mantissa
  def pairs(t: FpType, precC: Int, n: Int, seed: Int): Seq[(BigInt, BigInt)] = {
    val rng    = new scala.util.Random(seed)
    val expMax = (1 << t.expWidth) - 2
    Seq.fill(n) {
      val eBig  = precC + 3 + rng.nextInt(math.max(1, expMax - precC - 2))
      val eSm   = math.max(0, eBig - (precC - 1) - rng.nextInt(43))
      val mBig  = if (rng.nextBoolean()) BigInt(0) else BigInt(t.sigWidth, rng)
      val mSm   = BigInt(t.sigWidth, rng) | (if (eSm == 0) BigInt(1) else BigInt(0))
      val big   = (BigInt(rng.nextInt(2)) << (t.width - 1)) | (BigInt(eBig) << t.sigWidth) | mBig
      val small = (BigInt(rng.nextInt(2)) << (t.width - 1)) | (BigInt(eSm) << t.sigWidth) | mSm
      if (rng.nextBoolean()) (big, small) else (small, big)
    }
  }

  def run(t: FpType, extra: Seq[(BigInt, BigInt)] = Nil): Unit = {
    val all = extra ++ pairs(t, FP32.sigWidth + 1, 4000, 0x5eed)
    test(new FpAdd(t, t, FP32))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.clock.step(1)
        for ((a, b) <- all) {
          dut.io.in_a.poke(a.U); dut.io.in_b.poke(b.U)
          dut.clock.step(1)
          val n = dut.io.out.peekInt(); val r = rne(t, t, a, b)
          assert(
            n == r,
            s"FpAdd ${t.width}+${t.width}->32: a=0x${a.toString(16)} b=0x${b.toString(16)} " +
              s"got=0x${n.toString(16)} rne=0x${r.toString(16)}"
          )
        }
      }
  }

  "FpAdd FP32+FP32->FP32" should "round like IEEE RNE when one operand is far below the other" in {
    run(FP32, Seq((BigInt(0x38400000L), BigInt(0xc7000000L)), (BigInt(0xb9600000L), BigInt(0x47000000L))))
  }
  "FpAdd FP16+FP16->FP32" should "round like IEEE RNE when one operand is far below the other" in { run(FP16) }
  "FpAdd BF16+BF16->FP32" should "round like IEEE RNE when one operand is far below the other" in { run(BF16) }
  "FpAdd FP8+FP8->FP32" should "round like IEEE RNE when one operand is far below the other" in { run(FP8) }
}
