// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Int32ColumnScale: y = rne(x * f[col]) per INT32 lane, col = lane % 16 + 16 * ((beat / 8) % N).
  *
  * The golden multiplies in double -- exact, since |x| < 2^31 and an FP16 significand has 11 bits -- and rounds with
  * Math.rint (round-half-even): a different path from the RTL's shift-and-sticky rounding, so a shared mistake cannot
  * hide. Inputs and outputs both stall irregularly.
  */
class Int32ColumnScaleTester extends AnyFlatSpec with ChiselScalatestTester {

  val flags     = VerilatorFlags(Seq("--build-jobs", "1"))
  val lanes     = 32
  val cols      = 16
  val bpb       = 8
  val maxBlocks = 2

  def f16ToDouble(h: Int): Double = {
    val e   = (h >> 10) & 0x1f
    val m   = h & 0x3ff
    val mag = if (e == 0) m * math.pow(2, -24) else (1024 + m) * math.pow(2, e - 25)
    if (((h >> 15) & 1) == 1) -mag else mag
  }

  def scaleRef(x: Int, h: Int): Int =
    if (((h >> 10) & 0x1f) == 31) { // Inf/NaN: saturate by the sign of the product
      if (x == 0) 0 else if ((x < 0) != (((h >> 15) & 1) == 1)) Int.MinValue else Int.MaxValue
    } else {
      val r = math.rint(x.toDouble * f16ToDouble(h))
      if (r >= Int.MaxValue.toDouble) Int.MaxValue else if (r <= Int.MinValue.toDouble) Int.MinValue else r.toInt
    }

  def packI32(v: Seq[Int]): BigInt =
    v.zipWithIndex.map { case (x, l) => BigInt(x.toLong & 0xffffffffL) << (32 * l) }.reduce(_ | _)
  def lanesOf(beat: BigInt): Seq[Int] = (0 until lanes).map(l => ((beat >> (32 * l)) & BigInt(0xffffffffL)).toLong.toInt)

  def golden(n: Int, factors: Seq[Int], beats: Seq[BigInt]): Seq[BigInt] =
    beats.zipWithIndex.map { case (bt, b) =>
      val blk = (b / bpb) % n
      packI32(lanesOf(bt).zipWithIndex.map { case (x, l) => scaleRef(x, factors(blk * cols + l % cols)) })
    }

  def sampleI32(rng: Random): Int = rng.nextInt(6) match {
    case 0 => rng.nextInt()                    // full range
    case 1 => rng.nextInt(1 << 16) - (1 << 15) // a typical O accumulator
    case 2 => rng.nextInt(64) - 32             // tiny: rounding ties everywhere
    case 3 => if (rng.nextBoolean()) Int.MaxValue - rng.nextInt(4) else Int.MinValue + rng.nextInt(4)
    case _ => rng.nextInt(1 << 24) - (1 << 23)
  }

  // corr = exp(m_old - m_new) lies in (0, 1]; sample (0, 2) plus exact 1.0 and subnormals.
  def sampleCorr(rng: Random): Int = rng.nextInt(8) match {
    case 0 => 0x3c00
    case 1 => rng.nextInt(0x400)
    case _ => ((rng.nextInt(15) + 1) << 10) | rng.nextInt(0x400)
  }

  def beatsOf(nBeats: Int, rng: Random): Seq[BigInt] = Seq.fill(nBeats)(packI32(Seq.fill(lanes)(sampleI32(rng))))

  /** One task: program the CSRs, pulse start, stream `beats` in with random gaps, take them out with a slow ready. */
  def runTask(dut: DataPathExtensionHarness, n: Int, factors: Seq[Int], beats: Seq[BigInt], rng: Random): Seq[BigInt] = {
    dut.io.csr_i(0).poke(n.U)
    for (i <- 0 until cols * maxBlocks / 2)
      dut.io.csr_i(1 + i).poke(((factors(2 * i) & 0xffff).toLong | ((factors(2 * i + 1) & 0xffff).toLong << 16)).U)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    var outs    = Seq[BigInt]()
    var threads = new chiseltest.internal.TesterThreadList(Seq())
    threads = threads.fork {
      for (bt <- beats) {
        if (rng.nextInt(4) == 0) { dut.io.data_i.valid.poke(false); dut.clock.step(1 + rng.nextInt(3)) }
        dut.io.data_i.valid.poke(true); dut.io.data_i.bits.poke(bt)
        while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
        dut.clock.step(1)
      }
      dut.io.data_i.valid.poke(false)
    }
    threads = threads.fork {
      for (k <- beats.indices) {
        while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
        if (k % 3 == 2) dut.clock.step(2)
        outs = outs :+ dut.io.data_o.bits.peekInt()
        dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false)
      }
    }
    threads.joinAndStep()
    dut.io.data_o.ready.poke(true)
    var w = 0; while (dut.io.busy_o.peekBoolean() && w < 100) { dut.clock.step(1); w += 1 }
    dut.io.data_o.ready.poke(false)
    assert(!dut.io.busy_o.peekBoolean(), "Int32ColumnScale: busy_o stuck high after the drain")
    assert(!dut.io.data_o.valid.peekBoolean(), "Int32ColumnScale: an extra output beat is pending")
    outs
  }

  def check(tag: String, hw: Seq[BigInt], gd: Seq[BigInt]): Unit = {
    assert(hw.length == gd.length, s"$tag: got ${hw.length} beats, expected ${gd.length}")
    for (b <- hw.indices) {
      val (h, g) = (lanesOf(hw(b)), lanesOf(gd(b)))
      val bad    = (0 until lanes).filter(l => h(l) != g(l))
      assert(bad.isEmpty, s"$tag beat $b: lanes ${bad.take(4).mkString(",")} got ${bad.take(4).map(h).mkString(",")}" +
        s" expected ${bad.take(4).map(g).mkString(",")}")
    }
  }

  def withDut(body: DataPathExtensionHarness => Unit): Unit =
    test(new DataPathExtensionHarness(new HasInt32ColumnScale(1024, cols, bpb, maxBlocks)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        dut.io.enable_i.poke(true)
        dut.io.data_i.valid.poke(false)
        dut.io.data_o.ready.poke(false)
        body(dut)
      }

  it should "scale every lane by its own column's factor, blocks n-inner (N=2)" in {
    val rng     = new Random(0xC015)
    val factors = Seq.fill(cols * maxBlocks)(sampleCorr(rng))
    val beats   = beatsOf(3 * 2 * bpb, rng) // 3 row blocks x 2 column blocks
    withDut(dut => check("N=2", runTask(dut, 2, factors, beats, rng), golden(2, factors, beats)))
  }

  it should "use only the first 16 factors when N=1" in {
    val rng     = new Random(0xC016)
    val factors = Seq.fill(cols * maxBlocks)(sampleCorr(rng))
    val beats   = beatsOf(4 * bpb, rng)
    withDut(dut => check("N=1", runTask(dut, 1, factors, beats, rng), golden(1, factors, beats)))
  }

  it should "round, saturate and sign edge factors like IEEE (0, 0.5 ties, subnormal, max, negative, Inf, NaN)" in {
    val rng   = new Random(0xC017)
    val edges = Seq(0x0000, 0x8000, 0x3c00, 0x3800, 0xb800, 0x0001, 0x03ff, 0x0400, 0x3bff, 0x3555, 0x4000, 0x5bff,
                    0x7bff, 0xbc00, 0x7c00, 0xfc00, 0x7e00, 0x3400, 0x2e66, 0x1000, 0x6400, 0x6800, 0x6c00, 0x7000,
                    0x3a00, 0x3e00, 0xc100, 0x0200, 0x3c01, 0x37ff, 0x6fff, 0x13ff)
    val specials = Seq(0, 1, -1, 2, -2, 3, -3, 5, -5, 7, -7, Int.MaxValue, Int.MinValue, 1 << 30, -(1 << 30), 12345679)
    val beats    = Seq.tabulate(2 * bpb) { b =>
      packI32(Seq.tabulate(lanes)(l => if ((b + l) % 3 == 0) specials((b * 7 + l) % specials.length) else sampleI32(rng)))
    }
    withDut(dut => check("edges", runTask(dut, 2, edges, beats, rng), golden(2, edges, beats)))
  }

  it should "restart the column count at every start, back to back" in {
    val rng = new Random(0xC018)
    withDut { dut =>
      // Task A stops after ONE block, leaving the counter mid-way; B and C must still begin at column block 0.
      for ((n, blocks, tag) <- Seq((2, 1, "A"), (2, 4, "B"), (1, 2, "C"), (2, 2, "D"))) {
        val factors = Seq.fill(cols * maxBlocks)(sampleCorr(rng))
        val beats   = beatsOf(blocks * bpb, rng)
        check(s"task $tag", runTask(dut, n, factors, beats, rng), golden(n, factors, beats))
      }
    }
  }
}
