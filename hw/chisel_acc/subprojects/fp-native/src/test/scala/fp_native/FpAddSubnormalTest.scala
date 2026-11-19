// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// FpAdd against IEEE round-to-nearest-even (host float addition) with a subnormal operand. fp_add.sv takes the
// exponent of a subnormal operand one too low (e.g. FP16 0x0001 + 0 = 2^-25 in FP32 instead of 2^-24), so
// FpAddDiffTest skips those pairs.

package fp_native

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import fp_unit._

object FpAddSubnormalTest {
  private def expField(t: FpType, x: BigInt): Int = ((x >> t.sigWidth) & ((BigInt(1) << t.expWidth) - 1)).toInt

  def isSubnormal(t: FpType, x: BigInt): Boolean = expField(t, x) == 0 && (x & ((BigInt(1) << t.sigWidth) - 1)) != 0

  /** The pairs where FpAdd and fp_add.sv differ because of the subnormal exponent: a subnormal operand, or (for inputs
    * with a wider exponent range than the output) both operands below the output's normal range.
    */
  def svDiffers(tA: FpType, tB: FpType, tC: FpType, a: BigInt, b: BigInt): Boolean = {
    def belowC(t: FpType, x: BigInt) = expField(t, x) <= FpCommon.bias(t) - FpCommon.bias(tC)
    isSubnormal(tA, a) || isSubnormal(tB, b) || (belowC(tA, a) && belowC(tB, b))
  }
}

class FpAddSubnormalTest extends AnyFlatSpec with ChiselScalatestTester {

  // one subnormal operand (random sign and mantissa); the other one subnormal, in the lowest normal binades, or
  // any finite value
  def pairs(t: FpType, n: Int, seed: Int): Seq[(BigInt, BigInt)] = {
    val rng    = new scala.util.Random(seed)
    val expMax = (1 << t.expWidth) - 2
    def sign   = BigInt(rng.nextInt(2)) << (t.width - 1)
    Seq.fill(n) {
      val sub   = sign | BigInt(t.sigWidth, rng) | 1
      val e     = rng.nextInt(4) match { case 0 => 0; case 1 => 1; case 2 => 2; case _ => rng.nextInt(expMax + 1) }
      val other = sign | (BigInt(e) << t.sigWidth) | BigInt(t.sigWidth, rng)
      if (rng.nextBoolean()) (sub, other) else (other, sub)
    }
  }

  def run(t: FpType, extra: Seq[(BigInt, BigInt)]): Unit = {
    test(new FpAdd(t, t, FP32))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.clock.step(1)
        for ((a, b) <- extra ++ pairs(t, 4000, 0x5eed)) {
          dut.io.in_a.poke(a.U); dut.io.in_b.poke(b.U)
          dut.clock.step(1)
          val n  = dut.io.out.peekInt(); val r = FpAddFarOperandTest.rne(t, t, a, b)
          // the sign of an exact zero (x + -x) follows fp_add.sv, see the rounding step of FpAdd
          val ok = if ((r & 0x7fffffff) == 0) (n & 0x7fffffff) == 0 else n == r
          assert(
            ok,
            s"FpAdd ${t.width}+${t.width}->32: a=0x${a.toString(16)} b=0x${b.toString(16)} " +
              s"got=0x${n.toString(16)} rne=0x${r.toString(16)}"
          )
        }
      }
  }

  "FpAdd FP32+FP32->FP32" should "round like IEEE RNE with a subnormal operand" in {
    run(FP32, Seq((BigInt(0x00400000L), BigInt(0x00800000L)), (BigInt(0x00400000L), BigInt(0x00400000L))))
  }
  "FpAdd FP16+FP16->FP32" should "round like IEEE RNE with a subnormal operand" in {
    run(FP16, Seq((BigInt(0x0001), BigInt(0)), (BigInt(0x03ff), BigInt(0x8001))))
  }
  "FpAdd BF16+BF16->FP32" should "round like IEEE RNE with a subnormal operand" in {
    run(BF16, Seq((BigInt(0x0001), BigInt(0)), (BigInt(0x0040), BigInt(0x0080))))
  }
  "FpAdd FP8+FP8->FP32" should "round like IEEE RNE with a subnormal operand" in {
    run(FP8, Seq((BigInt(0x01), BigInt(0)), (BigInt(0x03), BigInt(0x83))))
  }
}
