// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Differential test: native FpAdd must be BIT-EXACT to the fp_add.sv blackbox (FpAddFp) over random
// + corner-case inputs, for every production format combo (incl. the narrowing path) and numPipe.

package fp_native

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import fp_unit._

class FpAddDiffHarness(typeA: FpType, typeB: FpType, typeC: FpType, numPipe: Int) extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in_a    = Input(UInt(typeA.W))
    val in_b    = Input(UInt(typeB.W))
    val out_nat = Output(UInt(typeC.W))
    val out_bb  = Output(UInt(typeC.W))
  })
  val nat = Module(new FpAdd(typeA, typeB, typeC, numPipe))
  val bb  = Module(new FpAddFp(typeA, typeB, typeC))
  nat.io.in_a := io.in_a; nat.io.in_b := io.in_b
  bb.io.in_a  := io.in_a; bb.io.in_b  := io.in_b
  io.out_nat  := nat.io.out
  io.out_bb   := ShiftRegister(bb.io.out, numPipe)
}

class FpAddDiffTest extends AnyFlatSpec with ChiselScalatestTester {

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

  def run(typeA: FpType, typeB: FpType, typeC: FpType, numPipe: Int, nRand: Int = 2000): Unit = {
    val rng   = new scala.util.Random(0x5eed + numPipe)
    val maskA = (BigInt(1) << typeA.width) - 1
    val maskB = (BigInt(1) << typeB.width) - 1
    val pairs = (for (a <- corners(typeA); b <- corners(typeB)) yield (a, b)) ++
      Seq.fill(nRand)((BigInt(typeA.width, rng) & maskA, BigInt(typeB.width, rng) & maskB))
    test(new FpAddDiffHarness(typeA, typeB, typeC, numPipe))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.clock.step(1)
        for ((a, b) <- pairs) {
          dut.io.in_a.poke(a.U); dut.io.in_b.poke(b.U)
          dut.clock.step(numPipe + 1)
          val n = dut.io.out_nat.peekInt(); val r = dut.io.out_bb.peekInt()
          assert(n == r, s"FpAdd ${typeA.width}+${typeB.width}->${typeC.width} pipe$numPipe: " +
            s"a=0x${a.toString(16)} b=0x${b.toString(16)} native=0x${n.toString(16)} bb=0x${r.toString(16)}")
        }
      }
  }

  for (np <- 0 to 2) {
    s"FpAdd FP32+FP32->FP32 (pipe$np)" should "match fp_add.sv bit-exactly" in { run(FP32, FP32, FP32, np) }
    s"FpAdd FP16+FP16->FP32 (pipe$np)" should "match fp_add.sv bit-exactly" in { run(FP16, FP16, FP32, np) }
    s"FpAdd FP32+FP32->FP16 (pipe$np)" should "match fp_add.sv bit-exactly" in { run(FP32, FP32, FP16, np) }
  }
  "FpAdd BF16+BF16->FP32 (pipe0)" should "match fp_add.sv bit-exactly" in { run(BF16, BF16, FP32, 0) }
  "FpAdd FP8+FP8->FP32 (pipe0)"   should "match fp_add.sv bit-exactly" in { run(FP8, FP8, FP32, 0) }
}
