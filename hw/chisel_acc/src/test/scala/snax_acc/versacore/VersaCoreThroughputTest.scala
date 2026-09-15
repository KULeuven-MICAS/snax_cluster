// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax_acc.versacore

import scala.util.Random

import chisel3._

import chiseltest._
import fp_unit._
import org.scalatest.flatspec.AnyFlatSpec
import snax_acc.utils.CommonTestUtils.toSInt
import snax_acc.utils.MatrixLibBlock.temporalToSpatialIndicesAB

/** Sustained array throughput with an ideal operand feed, in the shapes FlashAttention runs.
  *
  * FA leaves 4597 cycles of array idle inside the GEMM tasks (1.24 and 1.31 cyc/pass against a
  * floor of 1.00). That idle is either inside VersaCore or outside it in the streamer / TCDM.
  * This test removes the outside entirely: A and B are held valid every cycle and D is drained
  * with no backpressure, so whatever rate the array reaches here is the rate it is capable of.
  *
  * The knob that separates the two candidates is the C/D serial ratio. The cluster runs A and B
  * at ratio 1 (512/512) but C and D at ratio 4 (8192/2048), so every output block deserialises C
  * over 4 beats and serialises D over 4, through C_s2p and D_p2s. Running the same shapes at
  * ratio 1 and ratio 4 charges the difference to those converters.
  */
class VersaCoreThroughputTest extends AnyFlatSpec with ChiselScalatestTester {

  val Mu = 4
  val Ku = 4
  val Nu = 4

  // one C/D element per lane, Mu*Nu lanes of 32 bit
  val cdParallelWidth = Mu * Nu * 32

  def params(cdRatio: Int) = SpatialArrayParam(
    multiplierNum          = Seq(Mu * Ku * Nu),
    inputTypeA             = Seq(Int8),
    inputTypeB             = Seq(Int8),
    inputTypeC             = Seq(Int32),
    outputTypeD            = Seq(Int32),
    arrayInputAWidth       = Mu * Ku * 8,
    arrayInputBWidth       = Ku * Nu * 8,
    arrayInputCWidth       = cdParallelWidth,
    arrayOutputDWidth      = cdParallelWidth,
    serialInputADataWidth  = Mu * Ku * 8,
    serialInputBDataWidth  = Ku * Nu * 8,
    serialInputCDataWidth  = cdParallelWidth / cdRatio,
    serialOutputDDataWidth = cdParallelWidth / cdRatio,
    arrayDim               = Seq(Seq(Seq(Mu, Ku, Nu))),
    // the cluster declares a single temporal unrolling (output_stationary); the default is all
    // three, and multi-dataflow forbids a serial ratio other than 1
    dataflow               = Seq("output_stationary")
  )

  def until(cond: => Boolean, dut: VersaCoreHarness, budget: Int, what: String): Unit = {
    var n = 0
    while (!cond) {
      assert(n < budget, s"HANG: timed out after $budget cycles waiting for $what")
      dut.clock.step(1)
      n += 1
    }
  }

  /** blocks output blocks, each accumulating K passes. Returns cycles per array pass. */
  def run(cdRatio: Int, blocks: Int, K: Int): Double = {
    val M      = blocks
    val N      = 1
    val passes = M * N * K
    val budget = 200000
    var result = 0.0

    test(new VersaCoreHarness(params(cdRatio))) { dut =>
      val rand    = new Random(7)
      val aValues = Array.fill(Mu * Ku * M * K)(rand.nextInt(1 << 8))
      val bValues = Array.fill(Ku * Nu * N * K)(rand.nextInt(1 << 8))
      val cValues = Array.fill(Mu * Nu * M * N)(rand.nextInt(1 << 16))

      val golden = Array.tabulate(M * N) { b =>
        val acc = Array.fill(Mu * Nu)(0)
        for (k2 <- 0 until K; m1 <- 0 until Mu; n1 <- 0 until Nu) {
          var sum = 0
          for (k1 <- 0 until Ku) {
            val a = toSInt(aValues(b * K * Mu * Ku + k2 * Mu * Ku + m1 * Ku + k1), 8)
            val bb = toSInt(bValues(k2 * Nu * Ku + n1 * Ku + k1), 8)
            sum += a * bb
          }
          acc(m1 * Nu + n1) += sum
        }
        for (i <- 0 until Mu * Nu) acc(i) += toSInt(cValues(b * Mu * Nu + i), 32)
        acc
      }

      dut.clock.setTimeout(0)
      dut.clock.step(5)
      dut.io.ctrl.bits.fsmCfg.take_in_new_c.poke(1.U)
      dut.io.ctrl.bits.fsmCfg.temporal_accumulation_times.poke(K.U)
      dut.io.ctrl.bits.fsmCfg.output_times.poke((M * N).U)
      dut.io.ctrl.bits.fsmCfg.subtraction_constant_i.poke(0.U)
      dut.io.ctrl.bits.arrayCfg.arrayShapeCfg.poke(0.U)
      dut.io.ctrl.bits.arrayCfg.dataTypeCfg.poke(0.U)
      dut.io.ctrl.valid.poke(true.B)
      until(dut.io.ctrl.ready.peekBoolean(), dut, budget, "ctrl.ready")
      dut.clock.step(1)
      dut.io.ctrl.valid.poke(false.B)

      dut.io.versacore_data.out_d.ready.poke(true.B)

      var threads = new chiseltest.internal.TesterThreadList(Seq())

      threads = threads.fork {
        for (t <- 0 until passes) {
          val (ia, _) = temporalToSpatialIndicesAB(t, K = K, N = N)
          val w = aValues.slice(ia * Mu * Ku, ia * Mu * Ku + Mu * Ku)
            .zipWithIndex.map { case (v, i) => BigInt(v) << (i * 8) }.sum
          dut.io.versacore_data.in_a.bits.poke(w.U)
          dut.io.versacore_data.in_a.valid.poke(true.B)
          until(dut.io.versacore_data.in_a.ready.peekBoolean(), dut, budget, s"in_a at $t")
          dut.clock.step(1)
        }
        dut.io.versacore_data.in_a.valid.poke(false.B)
      }

      threads = threads.fork {
        for (t <- 0 until passes) {
          val (_, ib) = temporalToSpatialIndicesAB(t, K = K, N = N)
          val w = bValues.slice(ib * Nu * Ku, ib * Nu * Ku + Nu * Ku)
            .zipWithIndex.map { case (v, i) => BigInt(v) << (i * 8) }.sum
          dut.io.versacore_data.in_b.bits.poke(w.U)
          dut.io.versacore_data.in_b.valid.poke(true.B)
          until(dut.io.versacore_data.in_b.ready.peekBoolean(), dut, budget, s"in_b at $t")
          dut.clock.step(1)
        }
        dut.io.versacore_data.in_b.valid.poke(false.B)
      }

      // C: cdRatio beats per output block, low lanes first
      val lanesPerBeat = Mu * Nu / cdRatio
      threads = threads.fork {
        for (b <- 0 until M * N; beat <- 0 until cdRatio) {
          val w = (0 until lanesPerBeat).map { l =>
            BigInt(cValues(b * Mu * Nu + beat * lanesPerBeat + l)) << (l * 32)
          }.sum
          dut.io.versacore_data.in_c.bits.poke(w.U)
          dut.io.versacore_data.in_c.valid.poke(true.B)
          until(dut.io.versacore_data.in_c.ready.peekBoolean(), dut, budget, s"in_c block $b beat $beat")
          dut.clock.step(1)
        }
        dut.io.versacore_data.in_c.valid.poke(false.B)
      }

      // D: cdRatio beats per output block
      val got = Array.fill(M * N)(Array.fill(Mu * Nu)(0L))
      threads = threads.fork {
        for (b <- 0 until M * N; beat <- 0 until cdRatio) {
          until(dut.io.versacore_data.out_d.valid.peekBoolean(), dut, budget, s"out_d block $b beat $beat")
          val w = dut.io.versacore_data.out_d.bits.peek().litValue
          for (l <- 0 until lanesPerBeat) {
            got(b)(beat * lanesPerBeat + l) = ((w >> (l * 32)) & 0xffffffffL).toLong
          }
          dut.clock.step(1)
        }
      }

      threads.join()
      until(!dut.io.busy_o.peekBoolean(), dut, budget, "busy_o to fall")

      val cycles = dut.io.performance_counter.peek().litValue
      result = cycles.toDouble / passes
      println(f"THROUGHPUT cdRatio=$cdRatio%d blocks=$blocks%-4d accum=$K%-4d " +
        f"passes=$passes%-5d cycles=$cycles%-6s ${result}%.3f cyc/pass")

      for (b <- 0 until M * N; i <- 0 until Mu * Nu) {
        val exp = golden(b)(i).toLong & 0xffffffffL
        assert(got(b)(i) == exp, f"block $b lane $i: got 0x${got(b)(i)}%X expected 0x$exp%X")
      }
    }
    result
  }

  behavior of "VersaCore throughput"

  it should "sustain full rate in the FA shapes with an ideal operand feed" in {
    // FA runs 256 blocks x 32 accum and 64 blocks x 128 accum, both 8192 passes.
    // Same structure, scaled so the fixed fill/drain cost stays negligible.
    val shapes = Seq((16, 32), (4, 128))
    // ratio 4 is what the cluster runs today (8192/2048). 8 and 16 are the candidates for
    // narrowing the C/D port so it stops claiming all 32 TCDM banks in one beat.
    for (cdRatio <- Seq(1, 4, 8, 16); (blocks, k) <- shapes) run(cdRatio, blocks, k)
  }
}
