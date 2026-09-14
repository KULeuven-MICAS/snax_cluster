package snax_acc.utils

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import scala.util.Random

class ParallelToSerialExhaustiveTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "ParallelToSerial independent word scoreboard"

  private def initialize(dut: GroupedParallelToSerialTestHarness, factor: Int): Unit = {
    dut.clock.setTimeout(0)
    dut.io.in.valid.poke(false.B)
    dut.io.in.bits.poke(0.U)
    dut.io.out.ready.poke(false.B)
    dut.io.is_busy_cstate.poke(true.B)
    dut.io.counter_value_reset.poke(false.B)
    dut.io.terminate_factor.foreach(_.poke(factor.U))
    dut.reset.poke(true.B)
    dut.clock.step(2)
    dut.reset.poke(false.B)
    dut.io.out.valid.expect(false.B)
  }

  // Expected data is a slice of the original word, independent of the DUT's
  // group layout, shift registers, and counter implementation.
  private def beat(
    dut:       GroupedParallelToSerialTestHarness,
    p:         ParallelAndSerialConverterParams,
    word:      BigInt,
    index:     Int,
    stallTime: Int
  ): Unit = {
    val serialMask   = (BigInt(1) << p.serialWidth) - 1
    val parallelMask = (BigInt(1) << p.parallelWidth) - 1
    val expected     = (word >> (index * p.serialWidth)) & serialMask
    dut.io.in.valid.poke((index == 0).B)
    dut.io.in.bits.poke((if (index == 0) word else word ^ parallelMask).U)
    dut.io.out.ready.poke(false.B)
    for (_ <- 0 until stallTime) {
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.expect(expected.U)
      dut.io.in.ready.expect(false.B)
      dut.clock.step()
    }
    dut.io.out.ready.poke(true.B)
    dut.io.out.valid.expect(true.B)
    dut.io.out.bits.expect(expected.U)
    dut.io.in.ready.expect((index == 0).B)
    dut.clock.step()
  }

  private def transaction(
    dut:       GroupedParallelToSerialTestHarness,
    p:         ParallelAndSerialConverterParams,
    word:      BigInt,
    factor:    Int,
    stallTime: Int => Int
  ): Unit = {
    dut.io.terminate_factor.foreach(_.poke(factor.U))
    for (index <- 0 until factor) {
      beat(dut, p, word, index, stallTime(index))
    }
    dut.io.in.valid.poke(false.B)
    dut.io.out.valid.expect(false.B)
    dut.io.in.ready.expect(true.B)
  }

  // This bounded exhaustive check enumerates every possible input word, every
  // legal termination factor, and every combination of zero/one stalled cycle
  // before each beat. It is not an unbounded formal proof.
  for {
    ratio <- 1 to 5
    group <- Seq(2, 4, 8)
  } {
    it should s"exhaust all $ratio-bit words, factors, and one-cycle stall patterns with group $group" in {
      val p = ParallelAndSerialConverterParams(
        parallelWidth           = ratio,
        serialWidth             = 1,
        earlyTerminate          = true,
        allowedTerminateFactors = 1 to ratio,
        p2sChunksPerGroup       = group
      )
      test(new GroupedParallelToSerialTestHarness(p)) { dut =>
        initialize(dut, 1)
        for {
          word      <- 0 until (1 << ratio)
          factor    <- 1 to ratio
          stallMask <- 0 until (1 << factor)
        } {
          withClue(s"word=$word factor=$factor stallMask=$stallMask: ") {
            transaction(dut, p, BigInt(word), factor, index => (stallMask >> index) & 1)
          }
        }
      }
    }
  }

  // The current HJSON configurations use a 32768-bit word, a 1024-bit output,
  // and group sizes 2, 4, or 8. Exercise all factors (a superset of the actual
  // shapes), plus single-bit words at both ends of every output chunk.
  for (group <- Seq(2, 4, 8)) {
    it should s"preserve bit positions and every factor at 32768-to-1024 with group $group" in {
      val p = ParallelAndSerialConverterParams(
        parallelWidth           = 32768,
        serialWidth             = 1024,
        earlyTerminate          = true,
        allowedTerminateFactors = 1 to 32,
        p2sChunksPerGroup       = group
      )
      test(new GroupedParallelToSerialTestHarness(p)) { dut =>
        val rng = new Random(0x503253L + group)
        initialize(dut, 1)
        for (factor <- (1 to 32) ++ (1 to 32).reverse) {
          transaction(dut, p, BigInt(p.parallelWidth, rng), factor, _ => rng.nextInt(5))
        }
        val bitPositions = (0 until 32).flatMap { chunk =>
          Seq(chunk * p.serialWidth, (chunk + 1) * p.serialWidth - 1)
        }
        val words        = Seq(BigInt(0), (BigInt(1) << p.parallelWidth) - 1) ++
          bitPositions.map(position => BigInt(1) << position)
        for (word <- words) {
          transaction(dut, p, word, 32, index => if (index % group == 0) 3 else 0)
        }
      }
    }
  }

  for (group <- Seq(2, 4, 8)) {
    it should s"discard interrupted words at every beat and recover after either reset with group $group" in {
      val ratio = 9
      val p     = ParallelAndSerialConverterParams(
        parallelWidth     = ratio * 7,
        serialWidth       = 7,
        p2sChunksPerGroup = group
      )
      test(new GroupedParallelToSerialTestHarness(p)) { dut =>
        val rng = new Random(0x5245534554L + group)
        initialize(dut, ratio)
        for {
          nextBeat <- 0 until ratio
          ready    <- Seq(false, true)
          async    <- Seq(false, true)
        } {
          val abandoned = BigInt(p.parallelWidth, rng)
          for (index <- 0 until nextBeat) {
            beat(dut, p, abandoned, index, stallTime = 0)
          }
          dut.io.in.valid.poke(false.B)
          dut.io.out.ready.poke(ready.B)
          dut.io.out.valid.expect((nextBeat != 0).B)
          if (async) {
            dut.reset.poke(true.B)
            // Asynchronous reset must take effect without a clock edge.
            dut.io.out.valid.expect(false.B)
          } else {
            dut.io.counter_value_reset.poke(true.B)
            // Counter reset is synchronous; a ready pending beat can still
            // transfer on this edge before the remaining word is discarded.
            dut.io.out.valid.expect((nextBeat != 0).B)
          }
          dut.clock.step(2)
          dut.reset.poke(false.B)
          dut.io.counter_value_reset.poke(false.B)
          dut.io.out.valid.expect(false.B)
          transaction(dut, p, BigInt(p.parallelWidth, rng), ratio, _ => 2)
        }
      }
    }
  }
}
