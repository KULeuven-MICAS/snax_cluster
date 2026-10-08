// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax_acc.versacore

import scala.util.Random

import chisel3._

import chiseltest._
import chiseltest.simulator.VerilatorBackendAnnotation
import fp_unit._
import org.scalatest.flatspec.AnyFlatSpec

/** snax_split_cluster's VersaCore, its exact parameters: 1024 INT8 multipliers, four array shapes picked per task.
  *
  *   0  (16, 4, 16)  the GEMM
  *   1  (1, 4, 32)   one-token GEMV
  *   2  (4, 4, 32)   four-token GEMV: rows 0..3 of A's first 16 bytes, the same 128 weights as shape 1, and a 4 x 32
  *                   output block that leaves as four 1024-bit D beats, one per row
  *   3  (8, 8, 16)   the few-row GEMM: all 64 A bytes and all 128 B bytes, Ku = 8 (a three-level adder tree), and an
  *                   8 x 16 output block that leaves as four D beats, two rows each
  *
  * One instance runs a sequence of tasks, so the runtime shape switch is covered: shape 2 without C (take_in_new_c =
  * 0, the accumulator cleared at each block's first pass -- how a GEMV runs with no C stream), shape 2 with C, then
  * shapes 1 and 0; shape 3 likewise, between shapes 0..2. A and B bytes past what a shape reads are random, so a shape
  * that read them would fail. Operands are driven back to back; D is drained without backpressure.
  *
  * A second sequence offers a full C stream to take_in_new_c = 0 tasks: the array must drain it (no hang), ignore it
  * (D = the fresh product), count every beat in c_dropped, and leave the next task's C untouched.
  */
class VersaCoreShapeTest extends AnyFlatSpec with ChiselScalatestTester {

  val shapes = Seq(Seq(16, 4, 16), Seq(1, 4, 32), Seq(4, 4, 32), Seq(8, 8, 16))

  val params = SpatialArrayParam(
    multiplierNum          = Seq(1024),
    inputTypeA             = Seq(Int8),
    inputTypeB             = Seq(Int8),
    inputTypeC             = Seq(Int32),
    outputTypeD            = Seq(Int32),
    arrayInputAWidth       = 512,
    arrayInputBWidth       = 1024,
    arrayInputCWidth       = 8192,
    arrayOutputDWidth      = 8192,
    serialInputADataWidth  = 512,
    serialInputBDataWidth  = 1024,
    serialInputCDataWidth  = 1024,
    serialOutputDDataWidth = 1024,
    arrayDim               = Seq(shapes),
    dataflow               = Seq("output_stationary")
  )

  val budget = 20000

  def until(cond: => Boolean, dut: VersaCoreHarness, what: String): Unit = {
    var n = 0
    while (!cond) {
      assert(n < budget, s"HANG: timed out after $budget cycles waiting for $what")
      dut.clock.step(1)
      n += 1
    }
  }

  def s8(v: Int):  Int  = if (v >= 128) v - 256 else v
  def word(bytes: Seq[Int], bits: Int): BigInt =
    bytes.zipWithIndex.map { case (v, i) => BigInt(v & 0xff) << (8 * i) }.sum & ((BigInt(1) << bits) - 1)
  def word32(vals: Seq[Long]): BigInt = vals.zipWithIndex.map { case (v, i) => BigInt(v & 0xffffffffL) << (32 * i) }.sum

  /** One task of `blocks` output blocks, `K` passes each, in array shape `shape`. Returns the D beats' INT32 lanes and
    * checks them against the golden: D(m, n) = C(m, n) + sum_k sum_kk A[k](m, kk) * B[k](n, kk), with A's pass word
    * holding A(m, kk) at byte m * Ku + kk and B's holding B(n, kk) at byte n * Ku + kk.
    */
  def runTask(
    dut:    VersaCoreHarness,
    shape:  Int,
    blocks: Int,
    K:      Int,
    withC:  Boolean,
    rng:    Random,
    strayC: Boolean = false // offer a C stream to a task with take_in_new_c = 0
  ): Double = {
    require(!(withC && strayC), "strayC is a C stream the task does not take")
    val Seq(mu, ku, nu) = shapes(shape)
    val beatsPerBlock   = math.max(1, mu * nu * 32 / 1024)
    val lanesPerBeat    = 32
    // every byte of every pass word is random; the shape reads only the first mu*ku (A) and nu*ku (B)
    val aw = Seq.fill(blocks * K)(Seq.fill(64)(rng.nextInt(256)))
    val bw = Seq.fill(blocks * K)(Seq.fill(128)(rng.nextInt(256)))
    val cw = Seq.fill(blocks)(Seq.fill(mu * nu)(rng.nextInt().toLong))
    val golden = Seq.tabulate(blocks) { b =>
      Seq.tabulate(mu * nu) { e =>
        val (m, n) = (e / nu, e % nu)
        var acc    = if (withC) cw(b)(e) else 0L
        for (k <- 0 until K; kk <- 0 until ku)
          acc += s8(aw(b * K + k)(m * ku + kk)).toLong * s8(bw(b * K + k)(n * ku + kk)).toLong
        acc.toInt.toLong & 0xffffffffL
      }
    }

    val droppedBefore = dut.io.c_dropped.peek().litValue
    dut.io.ctrl.bits.fsmCfg.take_in_new_c.poke((if (withC) 1 else 0).U)
    dut.io.ctrl.bits.fsmCfg.temporal_accumulation_times.poke(K.U)
    dut.io.ctrl.bits.fsmCfg.output_times.poke(blocks.U)
    dut.io.ctrl.bits.fsmCfg.subtraction_constant_i.poke(0.U)
    dut.io.ctrl.bits.arrayCfg.arrayShapeCfg.poke(shape.U)
    dut.io.ctrl.bits.arrayCfg.dataTypeCfg.poke(0.U)
    dut.io.ctrl.valid.poke(true.B)
    until(dut.io.ctrl.ready.peekBoolean(), dut, "ctrl.ready")
    dut.clock.step(1)
    dut.io.ctrl.valid.poke(false.B)
    dut.io.versacore_data.out_d.ready.poke(true.B)

    var threads = new chiseltest.internal.TesterThreadList(Seq())
    threads = threads.fork {
      for (t <- 0 until blocks * K) {
        dut.io.versacore_data.in_a.bits.poke(word(aw(t), 512).U)
        dut.io.versacore_data.in_a.valid.poke(true.B)
        until(dut.io.versacore_data.in_a.ready.peekBoolean(), dut, s"in_a.ready at pass $t")
        dut.clock.step(1)
      }
      dut.io.versacore_data.in_a.valid.poke(false.B)
    }
    threads = threads.fork {
      for (t <- 0 until blocks * K) {
        dut.io.versacore_data.in_b.bits.poke(word(bw(t), 1024).U)
        dut.io.versacore_data.in_b.valid.poke(true.B)
        until(dut.io.versacore_data.in_b.ready.peekBoolean(), dut, s"in_b.ready at pass $t")
        dut.clock.step(1)
      }
      dut.io.versacore_data.in_b.valid.poke(false.B)
    }
    if (withC || strayC) threads = threads.fork {
      for (b <- 0 until blocks; beat <- 0 until beatsPerBlock) {
        val lanes = cw(b).slice(beat * lanesPerBeat, (beat + 1) * lanesPerBeat)
        dut.io.versacore_data.in_c.bits.poke(word32(lanes).U)
        dut.io.versacore_data.in_c.valid.poke(true.B)
        until(dut.io.versacore_data.in_c.ready.peekBoolean(), dut, s"in_c.ready at block $b beat $beat")
        dut.clock.step(1)
      }
      dut.io.versacore_data.in_c.valid.poke(false.B)
    }
    val got = Array.fill(blocks, mu * nu)(0L)
    threads = threads.fork {
      for (b <- 0 until blocks; beat <- 0 until beatsPerBlock) {
        until(dut.io.versacore_data.out_d.valid.peekBoolean(), dut, s"out_d.valid at block $b beat $beat")
        val w = dut.io.versacore_data.out_d.bits.peek().litValue
        for (l <- 0 until lanesPerBeat if beat * lanesPerBeat + l < mu * nu)
          got(b)(beat * lanesPerBeat + l) = ((w >> (32 * l)) & 0xffffffffL).toLong
        dut.clock.step(1)
      }
    }
    threads.join()
    until(!dut.io.busy_o.peekBoolean(), dut, "busy_o to fall")
    val cycles = dut.io.performance_counter.peek().litValue.toDouble
    val dropped = dut.io.c_dropped.peek().litValue - droppedBefore
    val expectDropped = if (strayC) blocks * beatsPerBlock else 0
    assert(dropped == expectDropped, s"shape $shape: c_dropped moved by $dropped, expected $expectDropped")

    for (b <- 0 until blocks; e <- 0 until mu * nu)
      assert(
        got(b)(e) == golden(b)(e),
        f"shape $shape ${if (withC) "with" else "without"} C, block $b row ${e / nu} col ${e % nu}: " +
          f"got 0x${got(b)(e)}%08X expected 0x${golden(b)(e)}%08X"
      )
    val cpp = cycles / (blocks * K)
    println(f"VersaCoreShapeTest: shape $shape ($mu, $ku, $nu) ${if (withC) "with" else "without"} C: " +
      f"${blocks * K} passes in ${cycles.toLong} cycles ($cpp%.3f cyc/pass)")
    cpp
  }

  behavior of "snax_split_cluster's VersaCore"

  it should "run (4, 4, 32) with and without C, then switch back to (1, 4, 32) and (16, 4, 16)" in {
    test(new VersaCoreHarness(params)).withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
      val rng = new Random(0x4432)
      dut.clock.setTimeout(0)
      dut.clock.step(5)
      val c2  = runTask(dut, shape = 2, blocks = 5, K = 64, withC = false, rng)
      runTask(dut, shape = 2, blocks = 3, K = 16, withC = true, rng)
      runTask(dut, shape = 1, blocks = 4, K = 16, withC = false, rng)
      runTask(dut, shape = 0, blocks = 2, K = 8, withC = true, rng)
      runTask(dut, shape = 2, blocks = 2, K = 8, withC = false, rng)
      // a GEMV block drains its four D beats while the next block's passes run: the rate is the array's
      assert(c2 < 1.05, f"(4, 4, 32) should run near one pass a cycle, measured $c2%.3f")
    }
  }

  it should "run (8, 8, 16) with and without C, between the other three shapes" in {
    test(new VersaCoreHarness(params)).withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
      val rng = new Random(0x8816)
      dut.clock.setTimeout(0)
      dut.clock.step(5)
      val c3 = runTask(dut, shape = 3, blocks = 5, K = 64, withC = false, rng)
      runTask(dut, shape = 3, blocks = 3, K = 16, withC = true, rng)
      runTask(dut, shape = 0, blocks = 2, K = 8, withC = true, rng)
      runTask(dut, shape = 3, blocks = 4, K = 2, withC = true, rng)
      runTask(dut, shape = 2, blocks = 2, K = 8, withC = false, rng)
      runTask(dut, shape = 1, blocks = 2, K = 8, withC = false, rng)
      runTask(dut, shape = 3, blocks = 2, K = 8, withC = false, rng)
      // four D beats a block drain while the next block's passes run: the rate is the array's
      assert(c3 < 1.05, f"(8, 8, 16) should run near one pass a cycle, measured $c3%.3f")
    }
  }

  it should "drain and count a C stream offered to a task that takes no C, and leave the next task's C intact" in {
    test(new VersaCoreHarness(params)).withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
      val rng = new Random(0xc0de)
      dut.clock.setTimeout(0)
      dut.clock.step(5)
      runTask(dut, shape = 0, blocks = 2, K = 8, withC = false, rng, strayC = true)
      runTask(dut, shape = 0, blocks = 2, K = 8, withC = true, rng)
      runTask(dut, shape = 2, blocks = 3, K = 16, withC = false, rng, strayC = true)
      runTask(dut, shape = 1, blocks = 4, K = 16, withC = false, rng, strayC = true)
      runTask(dut, shape = 2, blocks = 3, K = 16, withC = true, rng)
      runTask(dut, shape = 3, blocks = 3, K = 16, withC = false, rng, strayC = true)
      runTask(dut, shape = 3, blocks = 2, K = 16, withC = true, rng)
    }
  }
}
