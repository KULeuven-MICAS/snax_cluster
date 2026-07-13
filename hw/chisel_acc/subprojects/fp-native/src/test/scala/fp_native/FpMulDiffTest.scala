// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Differential test: native FpMul must be BIT-EXACT to the fp_mul.sv blackbox (FpMulFp) over random
// + corner-case inputs, for every production format combo and numPipe in {0,1,2}.

package fp_native

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

import fp_unit._

/** Drives identical raw inputs into the native FpMul and the FpMulFp blackbox; exposes both outputs
  * (blackbox delayed by numPipe so they align). */
class FpMulDiffHarness(typeA: FpType, typeB: FpType, typeC: FpType, numPipe: Int) extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in_a    = Input(UInt(typeA.W))
    val in_b    = Input(UInt(typeB.W))
    val out_nat = Output(UInt(typeC.W))
    val out_bb  = Output(UInt(typeC.W))
  })
  val nat = Module(new FpMul(typeA, typeB, typeC, numPipe))
  val bb  = Module(new FpMulFp(typeA, typeB, typeC))
  nat.io.in_a := io.in_a; nat.io.in_b := io.in_b
  bb.io.in_a  := io.in_a; bb.io.in_b  := io.in_b
  io.out_nat  := nat.io.out
  io.out_bb   := ShiftRegister(bb.io.out, numPipe)
}

class FpMulDiffTest extends AnyFlatSpec with ChiselScalatestTester {

  // corner-case bit patterns for a format
  def corners(t: FpType): Seq[BigInt] = {
    val e = t.expWidth; val s = t.sigWidth; val w = t.width
    val allExp = (BigInt(1) << e) - 1
    val allMan = (BigInt(1) << s) - 1
    val sign   = BigInt(1) << (w - 1)
    val inf    = allExp << s
    val one    = BigInt((1 << (e - 1)) - 1) << s
    val base = Seq(
      BigInt(0),              // +0
      sign,                   // -0
      inf,                    // +inf
      sign | inf,             // -inf
      inf | (BigInt(1) << (s - 1)), // qNaN
      inf | BigInt(1),        // sNaN
      (allExp - 1) << s | allMan,   // max normal
      BigInt(1) << s,         // min normal
      allMan,                 // max subnormal
      BigInt(1),              // min subnormal
      one,                    // 1.0
      sign | one,             // -1.0
      one | BigInt(1) << s    // 2.0
    )
    base.map(_ & ((BigInt(1) << w) - 1))
  }

  def run(typeA: FpType, typeB: FpType, typeC: FpType, numPipe: Int, nRand: Int = 2000): Unit = {
    val rng    = new scala.util.Random(0x5eed + numPipe)
    val maskA  = (BigInt(1) << typeA.width) - 1
    val maskB  = (BigInt(1) << typeB.width) - 1
    val ca     = corners(typeA); val cb = corners(typeB)
    val pairs  = (for (a <- ca; b <- cb) yield (a, b)) ++
      Seq.fill(nRand)((BigInt(typeA.width, rng) & maskA, BigInt(typeB.width, rng) & maskB))
    test(new FpMulDiffHarness(typeA, typeB, typeC, numPipe))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.clock.step(1)
        for ((a, b) <- pairs) {
          dut.io.in_a.poke(a.U); dut.io.in_b.poke(b.U)
          dut.clock.step(numPipe + 1)
          val n = dut.io.out_nat.peekInt(); val r = dut.io.out_bb.peekInt()
          assert(n == r, f"FpMul ${typeA.width}x${typeB.width}->${typeC.width} pipe$numPipe: a=0x$a%x b=0x$b%x  native=0x$n%x  bb=0x$r%x")
        }
      }
  }

  for (np <- 0 to 2) {
    s"FpMul FP32xFP32->FP32 (pipe$np)" should "match fp_mul.sv bit-exactly" in { run(FP32, FP32, FP32, np) }
    s"FpMul FP16xFP16->FP32 (pipe$np)" should "match fp_mul.sv bit-exactly" in { run(FP16, FP16, FP32, np) }
  }
  "FpMul BF16xBF16->FP32 (pipe0)" should "match fp_mul.sv bit-exactly" in { run(BF16, BF16, FP32, 0) }
  "FpMul FP8xFP8->FP32 (pipe0)"   should "match fp_mul.sv bit-exactly" in { run(FP8, FP8, FP32, 0) }
}
