// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// FpFma against IEEE round-to-nearest-even (exact a * b + c on the host). fp_fma.sv at chisel-float 8b0b24b
// rounds some FP16 * FP16 + FP32 cases wrong (an addend wider than the product, a subnormal FP16 operand, a
// zero product with a small addend), so FpFmaDiffTest accepts the native result there when it is the IEEE one.

package fp_native

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import fp_unit._

object FpFmaRoundingTest {
  private def field(t: FpType, x: BigInt) =
    (
      x.testBit(t.width - 1),
      ((x >> t.sigWidth) & ((BigInt(1) << t.expWidth) - 1)).toInt,
      x & ((BigInt(1) << t.sigWidth) - 1)
    )

  def finite(t: FpType, x: BigInt): Boolean = field(t, x)._2 != (1 << t.expWidth) - 1

  // a finite operand as (sign, integer significand, exponent of its LSB)
  private def decode(t: FpType, x: BigInt): (Boolean, BigInt, Int) = {
    val (s, e, m) = field(t, x)
    val bias      = FpCommon.bias(t)
    if (e == 0) (s, m, 1 - bias - t.sigWidth) else (s, m | (BigInt(1) << t.sigWidth), e - bias - t.sigWidth)
  }

  /** IEEE RNE of a * b + c for finite operands, as bits of format tC. */
  def rne(tA: FpType, tB: FpType, tC: FpType, a: BigInt, b: BigInt, c: BigInt): BigInt = {
    val (sa, ma, ea) = decode(tA, a)
    val (sb, mb, eb) = decode(tB, b)
    val (sc, mc, ec) = decode(tC, c)
    val sp           = sa ^ sb
    val mp           = ma * mb
    val e0           = math.min(ea + eb, ec)
    val n            = (if (sp) -mp else mp) * (BigInt(1) << (ea + eb - e0)) + (if (sc) -mc else mc) * (BigInt(1) << (ec - e0))
    val signBit      = BigInt(1) << (tC.width - 1)
    if (n == 0) return if (mp == 0 && mc == 0 && sp && sc) signBit else BigInt(0)
    val mag  = n.abs
    val man  = tC.sigWidth
    val bias = FpCommon.bias(tC)
    val e    = mag.bitLength - 1 + e0           // exponent of the leading bit
    val q    = math.max(e, 1 - bias) - man      // exponent of the result LSB
    val sh   = q - e0
    var r    =
      if (sh <= 0) mag << -sh
      else {
        val f = mag >> sh; val rem = mag - (f << sh); val half = BigInt(1) << (sh - 1)
        if (rem > half || (rem == half && f.testBit(0))) f + 1 else f
      }
    var ef   = if (e < 1 - bias) 0 else e + bias // a rounded-up subnormal r = 2^man encodes the smallest normal
    if (ef > 0 && r == (BigInt(1) << (man + 1))) { r = r >> 1; ef += 1 }
    val abs  =
      if (ef >= (1 << tC.expWidth) - 1) BigInt((1 << tC.expWidth) - 1) << man
      else if (ef == 0) r
      else (BigInt(ef) << man) | (r - (BigInt(1) << man))
    if (n < 0) abs | signBit else abs
  }
}

class FpFmaRoundingTest extends AnyFlatSpec with ChiselScalatestTester {
  import FpFmaRoundingTest._

  private def mk(t: FpType, s: Int, e: Int, m: BigInt) =
    (BigInt(s) << (t.width - 1)) | (BigInt(e) << t.sigWidth) | (m & ((BigInt(1) << t.sigWidth) - 1))

  // finite triples: a subnormal a or b, a zero product, an addend of the other sign 1 to 8 binades below the
  // product, and uniform finite operands
  def triples(tA: FpType, tB: FpType, tC: FpType, n: Int, seed: Int): Seq[(BigInt, BigInt, BigInt)] = {
    val rng                             = new scala.util.Random(seed)
    def r(t: FpType, lo: Int, hi: Int) = mk(t, rng.nextInt(2), lo + rng.nextInt(hi - lo + 1), BigInt(t.sigWidth, rng))
    def mid(t: FpType)                  = { val b = FpCommon.bias(t); r(t, b - b / 2, b + b / 2) }
    def any(t: FpType)                  = r(t, 0, (1 << t.expWidth) - 2)
    Seq.fill(n) {
      rng.nextInt(4) match {
        case 0 => (mk(tA, rng.nextInt(2), 0, BigInt(tA.sigWidth, rng) | 1), mid(tB), any(tC))
        case 1 => (mk(tA, rng.nextInt(2), 0, 0), any(tB), r(tC, 1, 60))
        case 2 =>
          val a  = mid(tA); val b = mid(tB)
          val pe = (((a >> tA.sigWidth).toInt & ((1 << tA.expWidth) - 1)) - FpCommon.bias(tA)) +
            (((b >> tB.sigWidth).toInt & ((1 << tB.expWidth) - 1)) - FpCommon.bias(tB))
          val s  = (if (a.testBit(tA.width - 1) ^ b.testBit(tB.width - 1)) 0 else 1)
          val ce = math.max(1, pe - 1 - rng.nextInt(8) + FpCommon.bias(tC))
          (a, b, mk(tC, s, ce, BigInt(tC.sigWidth, rng)))
        case _ => (any(tA), any(tB), any(tC))
      }
    }
  }

  def run(tA: FpType, tB: FpType, extra: Seq[(BigInt, BigInt, BigInt)] = Nil): Unit = {
    val all = extra ++ triples(tA, tB, FP32, 4000, 0x5eed)
    test(new FpFma(tA, tB, FP32))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.clock.step(1)
        for ((a, b, c) <- all) {
          dut.io.in_a.poke(a.U); dut.io.in_b.poke(b.U); dut.io.in_c.poke(c.U)
          dut.clock.step(1)
          val o = dut.io.out.peekInt(); val e = rne(tA, tB, FP32, a, b, c)
          assert(
            o == e,
            s"FpFma ${tA.width}*${tB.width}+32: a=0x${a.toString(16)} b=0x${b.toString(16)} c=0x${c.toString(16)} " +
              s"got=0x${o.toString(16)} rne=0x${e.toString(16)}"
          )
        }
      }
  }

  "FpFma FP16*FP16+FP32->FP32" should "round like IEEE RNE" in {
    run(
      FP16,
      FP16,
      Seq(
        (BigInt(0x743b), BigInt(0x3c00), BigInt(0xc595c001L)), // addend below the product: 0x4643e000
        (BigInt(0x8089), BigInt(0xc9e5), BigInt(0xb8c9e342L)), // subnormal operand: 0xad800000
        (BigInt(0x0000), BigInt(0xb97d), BigInt(0x0701b98bL))  // zero product: the addend, 0x0701b98b
      )
    )
  }
  "FpFma FP32*FP16+FP32->FP32" should "round like IEEE RNE" in { run(FP32, FP16) }
  "FpFma FP32*FP32+FP32->FP32" should "round like IEEE RNE" in { run(FP32, FP32) }
}
