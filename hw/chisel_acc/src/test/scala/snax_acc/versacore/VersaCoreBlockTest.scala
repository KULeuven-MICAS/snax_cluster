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

/** Output-stationary multi-block test with the D output enabled and the operands driven
  * back to back.
  *
  * VersaCoreTest cannot observe the datapath cut on the multiplier -> adder tree path: it sets
  * output_times = 0 (so D_p2s.in.valid is false, out_d.ready is constant true and the whole D
  * path is dead), it runs a single output block (M = N = 1, so neither pass counter ever
  * wraps), and it spaces every operand by Random.between(1, 5) cycles so the array never runs
  * at full rate. This test drives M*N > 1 blocks with output_times = M*N and valid held high,
  * which is what the cluster does, and fails with a cycle budget instead of hanging.
  */
class VersaCoreBlockTest extends AnyFlatSpec with ChiselScalatestTester {

  val params = SpatialArrayParam(
    multiplierNum          = Seq(8),
    inputTypeA             = Seq(Int8),
    inputTypeB             = Seq(Int8),
    inputTypeC             = Seq(Int32),
    outputTypeD            = Seq(Int32),
    arrayInputAWidth       = 32,
    arrayInputBWidth       = 32,
    arrayInputCWidth       = 128,
    arrayOutputDWidth      = 128,
    serialInputADataWidth  = 32,
    serialInputBDataWidth  = 32,
    serialInputCDataWidth  = 128,
    serialOutputDDataWidth = 128,
    // Mu, Ku, Nu
    arrayDim               = Seq(Seq(Seq(2, 2, 2)))
  )

  val Mu = 2
  val Ku = 2
  val Nu = 2
  val M  = 2
  val N  = 2
  val K  = 4

  val passes = M * N * K
  val budget = 4000

  /** Step until cond holds, failing with a useful message instead of spinning forever. */
  def until(cond: => Boolean, dut: VersaCoreHarness, what: String): Unit = {
    var n = 0
    while (!cond) {
      assert(n < budget, s"HANG: timed out after $budget cycles waiting for $what")
      dut.clock.step(1)
      n += 1
    }
  }

  behavior of "VersaCore"

  it should "run multiple output blocks back to back and emit every D block" in {
    test(new VersaCoreHarness(params)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      val rand = new Random(42)

      val aValues = Array.fill(Mu * Ku * M * K)(rand.nextInt(1 << 8))
      val bValues = Array.fill(Ku * Nu * N * K)(rand.nextInt(1 << 8))
      val cValues = Array.fill(Mu * Nu * M * N)(rand.nextInt(1 << 16))

      // golden: D(m2,n2) = C(m2,n2) + sum_k A(m2,k) . B(n2,k)
      val golden = Array.tabulate(M, N) { (m2, n2) =>
        val acc = Array.fill(Mu, Nu)(0)
        for (k2 <- 0 until K; m1 <- 0 until Mu; n1 <- 0 until Nu) {
          var sum = 0
          for (k1 <- 0 until Ku) {
            val a = toSInt(aValues(m2 * K * Mu * Ku + k2 * Mu * Ku + m1 * Ku + k1), 8)
            val b = toSInt(bValues(n2 * K * Nu * Ku + k2 * Nu * Ku + n1 * Ku + k1), 8)
            sum += a * b
          }
          acc(m1)(n1) += sum
        }
        for (m1 <- 0 until Mu; n1 <- 0 until Nu) {
          acc(m1)(n1) += toSInt(cValues(m2 * N * Mu * Nu + n2 * Mu * Nu + m1 * Nu + n1), 32)
        }
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
      until(dut.io.ctrl.ready.peekBoolean(), dut, "ctrl.ready")
      dut.clock.step(1)
      dut.io.ctrl.valid.poke(false.B)

      // D is drained with no backpressure
      dut.io.versacore_data.out_d.ready.poke(true.B)

      var threads = new chiseltest.internal.TesterThreadList(Seq())

      // A, back to back: valid stays high across the whole stream
      threads = threads.fork {
        for (t <- 0 until M * K * N) {
          val (ia, _) = temporalToSpatialIndicesAB(t, K = K, N = N)
          val word    = aValues
            .slice(ia * Mu * Ku, ia * Mu * Ku + Mu * Ku)
            .zipWithIndex
            .map { case (v, i) => BigInt(v) << (i * 8) }
            .sum
          dut.io.versacore_data.in_a.bits.poke(word.U)
          dut.io.versacore_data.in_a.valid.poke(true.B)
          until(dut.io.versacore_data.in_a.ready.peekBoolean(), dut, s"in_a.ready at pass $t")
          dut.clock.step(1)
        }
        dut.io.versacore_data.in_a.valid.poke(false.B)
      }

      threads = threads.fork {
        for (t <- 0 until M * K * N) {
          val (_, ib) = temporalToSpatialIndicesAB(t, K = K, N = N)
          val word    = bValues
            .slice(ib * Nu * Ku, ib * Nu * Ku + Nu * Ku)
            .zipWithIndex
            .map { case (v, i) => BigInt(v) << (i * 8) }
            .sum
          dut.io.versacore_data.in_b.bits.poke(word.U)
          dut.io.versacore_data.in_b.valid.poke(true.B)
          until(dut.io.versacore_data.in_b.ready.peekBoolean(), dut, s"in_b.ready at pass $t")
          dut.clock.step(1)
        }
        dut.io.versacore_data.in_b.valid.poke(false.B)
      }

      // C, one word per output block
      threads = threads.fork {
        for (t <- 0 until M * N) {
          val word = cValues
            .slice(t * Mu * Nu, t * Mu * Nu + Mu * Nu)
            .zipWithIndex
            .map { case (v, i) => BigInt(v) << (i * 32) }
            .sum
          dut.io.versacore_data.in_c.bits.poke(word.U)
          dut.io.versacore_data.in_c.valid.poke(true.B)
          until(dut.io.versacore_data.in_c.ready.peekBoolean(), dut, s"in_c.ready at block $t")
          dut.clock.step(1)
        }
        dut.io.versacore_data.in_c.valid.poke(false.B)
      }

      // D, one word per output block, checked against the golden
      val got = Array.fill(M * N)(Array.fill(Mu * Nu)(0L))
      threads = threads.fork {
        for (t <- 0 until M * N) {
          until(dut.io.versacore_data.out_d.valid.peekBoolean(), dut, s"out_d.valid for block $t")
          val word = dut.io.versacore_data.out_d.bits.peek().litValue
          for (i <- 0 until Mu * Nu) {
            got(t)(i) = ((word >> (i * 32)) & 0xffffffffL).toLong
          }
          dut.clock.step(1)
        }
      }

      threads.join()

      until(!dut.io.busy_o.peekBoolean(), dut, "busy_o to fall")

      val cycles = dut.io.performance_counter.peek().litValue
      println(f"VersaCoreBlockTest: $passes%d passes in $cycles%s cycles " +
        f"(${cycles.toDouble / passes}%.2f cyc/pass, $M%dx$N%d blocks x $K%d accum)")

      val want = golden.flatten.map(_.flatten)
      for (t <- 0 until M * N; i <- 0 until Mu * Nu) {
        val exp = (want(t)(i).toLong & 0xffffffffL)
        assert(got(t)(i) == exp, f"block $t lane $i: got 0x${got(t)(i)}%X expected 0x$exp%X")
      }
    }
  }
}
