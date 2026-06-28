// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Differential test: native FpFma must be BIT-EXACT to the fp_fma.sv blackbox (FpFmaFp) over random
// + corner-case inputs. Production uses FP32^3 (ffma); a mixed combo is included for breadth.

package fp_native

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import fp_unit._

class FpFmaDiffHarness(typeA: FpType, typeB: FpType, typeC: FpType, numPipe: Int) extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in_a    = Input(UInt(typeA.W))
    val in_b    = Input(UInt(typeB.W))
    val in_c    = Input(UInt(typeC.W))
    val out_nat = Output(UInt(typeC.W))
    val out_bb  = Output(UInt(typeC.W))
  })
  val nat = Module(new FpFma(typeA, typeB, typeC, numPipe))
  val bb  = Module(new FpFmaFp(typeA, typeB, typeC))
  nat.io.in_a := io.in_a; nat.io.in_b := io.in_b; nat.io.in_c := io.in_c
  bb.io.in_a  := io.in_a; bb.io.in_b  := io.in_b; bb.io.in_c  := io.in_c
  io.out_nat  := nat.io.out
  io.out_bb   := ShiftRegister(bb.io.out, numPipe)
}

class FpFmaDiffTest extends AnyFlatSpec with ChiselScalatestTester {

  def corners(t: FpType): Seq[BigInt] = {
    val e = t.expWidth; val s = t.sigWidth; val w = t.width
    val allExp = (BigInt(1) << e) - 1
    val allMan = (BigInt(1) << s) - 1
    val sign   = BigInt(1) << (w - 1)
    val inf    = allExp << s
    val one    = BigInt((1 << (e - 1)) - 1) << s
    Seq(
      BigInt(0), sign, inf, sign | inf, inf | (BigInt(1) << (s - 1)), inf | BigInt(1),
      (allExp - 1) << s | allMan, BigInt(1) << s, allMan, BigInt(1), one, sign | one, one | (BigInt(1) << s)
    ).map(_ & ((BigInt(1) << w) - 1))
  }

  def run(typeA: FpType, typeB: FpType, typeC: FpType, numPipe: Int, nRand: Int = 4000): Unit = {
    val rng   = new scala.util.Random(0x5eed + numPipe)
    val mA = (BigInt(1) << typeA.width) - 1; val mB = (BigInt(1) << typeB.width) - 1; val mC = (BigInt(1) << typeC.width) - 1
    val ca = corners(typeA); val cb = corners(typeB); val cc = corners(typeC)
    // full corner cross-product is large; sample corners for a/b and sweep c corners, plus randoms
    val cornerTriples = for (a <- ca; b <- cb; c <- Seq(cc(0), cc(10), cc(6))) yield (a, b, c)
    val randTriples = Seq.fill(nRand)(
      (BigInt(typeA.width, rng) & mA, BigInt(typeB.width, rng) & mB, BigInt(typeC.width, rng) & mC))
    val triples = cornerTriples ++ randTriples
    test(new FpFmaDiffHarness(typeA, typeB, typeC, numPipe))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.clock.step(1)
        for ((a, b, c) <- triples) {
          dut.io.in_a.poke(a.U); dut.io.in_b.poke(b.U); dut.io.in_c.poke(c.U)
          dut.clock.step(numPipe + 1)
          val n = dut.io.out_nat.peekInt(); val r = dut.io.out_bb.peekInt()
          assert(n == r, s"FpFma ${typeA.width}*${typeB.width}+${typeC.width} pipe$numPipe: " +
            s"a=0x${a.toString(16)} b=0x${b.toString(16)} c=0x${c.toString(16)} native=0x${n.toString(16)} bb=0x${r.toString(16)}")
        }
      }
  }

  for (np <- 0 to 2) {
    s"FpFma FP32*FP32+FP32->FP32 (pipe$np)" should "match fp_fma.sv bit-exactly" in { run(FP32, FP32, FP32, np) }
  }
  "FpFma FP16*FP16+FP32->FP32 (pipe0)" should "match fp_fma.sv bit-exactly" in { run(FP16, FP16, FP32, 0) }
}
