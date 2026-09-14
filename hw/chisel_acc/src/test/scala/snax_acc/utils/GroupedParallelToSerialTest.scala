package snax_acc.utils

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import scala.util.Random

// Chiseltest 6.0 can poke a Bool reset but not an AsyncReset port. The cast
// preserves asynchronous reset behavior inside the actual converter.
class GroupedParallelToSerialTestHarness(p: ParallelAndSerialConverterParams) extends Module {
  val converter = withReset(reset.asBool.asAsyncReset) {
    Module(new ParallelToSerial(p))
  }
  val io        = IO(chiselTypeOf(converter.io))
  io <> converter.io
}

class GroupedParallelToSerialTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Grouped ParallelToSerial"

  private def initialize(dut: GroupedParallelToSerialTestHarness, factor: Int): Unit = {
    dut.clock.setTimeout(10000)
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

  // Each expected beat comes directly from the original transaction word. The
  // scoreboard does not reproduce the DUT's grouping or register updates.
  private def transferBeat(
    dut:    GroupedParallelToSerialTestHarness,
    p:      ParallelAndSerialConverterParams,
    word:   BigInt,
    beat:   Int,
    factor: Int,
    rng:    Random,
    stalls: Boolean = true
  ): Unit = {
    val serialMask   = (BigInt(1) << p.serialWidth) - 1
    val parallelMask = (BigInt(1) << p.parallelWidth) - 1
    val expected     = (word >> (beat * p.serialWidth)) & serialMask
    val first        = beat == 0
    // Once the word is accepted, unrelated upstream values must not overwrite
    // groups that have not yet reached the output.
    dut.io.in.bits.poke((if (first) word else word ^ parallelMask).U)
    dut.io.in.valid.poke((first || rng.nextBoolean()).B)
    dut.io.is_busy_cstate.poke(true.B)

    val atBoundary  =
      beat % p.p2sChunksPerGroup == 0 || beat % p.p2sChunksPerGroup == p.p2sChunksPerGroup - 1
    val stallCycles = if (stalls) {
      rng.nextInt(3) + (if (first || beat == factor - 1 || atBoundary) 2 else 0)
    } else 0

    withClue(s"ratio=${p.parallelWidth / p.serialWidth}, group=${p.p2sChunksPerGroup}, factor=$factor, beat=$beat: ") {
      dut.io.out.ready.poke(false.B)
      for (_ <- 0 until stallCycles) {
        dut.io.out.valid.expect(true.B)
        dut.io.out.bits.expect(expected.U)
        dut.io.in.ready.expect(false.B)
        dut.clock.step()
      }

      dut.io.out.ready.poke(true.B)
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.expect(expected.U)
      dut.io.in.ready.expect(first.B)
      dut.clock.step()
    }
  }

  private def transaction(
    dut:    GroupedParallelToSerialTestHarness,
    p:      ParallelAndSerialConverterParams,
    word:   BigInt,
    factor: Int,
    rng:    Random,
    stalls: Boolean = true
  ): Unit = {
    dut.io.terminate_factor.foreach(_.poke(factor.U))
    for (beat <- 0 until factor) {
      transferBeat(dut, p, word, beat, factor, rng, stalls)
    }
    // No idle edge is inserted between calls. With stalls disabled, the final
    // and next first beats transfer on consecutive clock edges.
    dut.io.in.valid.poke(false.B)
    dut.io.out.valid.expect(false.B)
    dut.io.in.ready.expect(true.B)
  }

  private def inputGap(dut: GroupedParallelToSerialTestHarness, rng: Random): Unit = {
    dut.io.in.valid.poke(false.B)
    for (_ <- 0 until 3) {
      val ready = rng.nextBoolean()
      val busy  = rng.nextBoolean()
      dut.io.out.ready.poke(ready.B)
      dut.io.is_busy_cstate.poke(busy.B)
      dut.io.out.valid.expect(false.B)
      dut.io.in.ready.expect((ready && busy).B)
      dut.clock.step()
    }
    dut.io.is_busy_cstate.poke(true.B)
  }

  it should "preserve transactions across all eight-chunk boundaries, termination factors, and resets" in {
    val factors = Seq(1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 32)
    val p       = ParallelAndSerialConverterParams(
      parallelWidth           = 32 * 16,
      serialWidth             = 16,
      earlyTerminate          = true,
      allowedTerminateFactors = factors,
      p2sChunksPerGroup        = 8
    )
    test(new GroupedParallelToSerialTestHarness(p)) { dut =>
      val rng = new Random(0x47524f5550L)
      initialize(dut, factors.head)

      // Preserve the existing busy guard: it gates input ready, but does not
      // independently suppress output valid. Do not transfer while not busy.
      dut.io.in.bits.poke(0x1234.U)
      dut.io.in.valid.poke(true.B)
      dut.io.is_busy_cstate.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.io.in.ready.expect(false.B)
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.expect(0x1234.U)
      dut.io.in.valid.poke(false.B)
      dut.io.is_busy_cstate.poke(true.B)

      for ((factor, index) <- (factors ++ factors.reverse).zipWithIndex) {
        if (index % 3 == 0) inputGap(dut, rng)
        transaction(dut, p, BigInt(p.parallelWidth, rng), factor, rng)
      }

      // Abort in a later group, both while stalled and on a transfer. The
      // external counter reset has priority over advancement on that edge.
      for (readyDuringReset <- Seq(false, true)) {
        val abandoned = BigInt(p.parallelWidth, rng)
        dut.io.terminate_factor.get.poke(32.U)
        for (beat <- 0 until 9) {
          transferBeat(dut, p, abandoned, beat, 32, rng, stalls = false)
        }
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(readyDuringReset.B)
        dut.io.out.valid.expect(true.B)
        dut.io.out.bits.expect(((abandoned >> (9 * p.serialWidth)) & 0xffff).U)
        dut.io.counter_value_reset.poke(true.B)
        dut.clock.step()
        dut.io.counter_value_reset.poke(false.B)
        dut.io.out.valid.expect(false.B)
        transaction(dut, p, BigInt(p.parallelWidth, rng), 32, rng)
      }

      val abandoned = BigInt(p.parallelWidth, rng)
      dut.io.terminate_factor.get.poke(32.U)
      for (beat <- 0 until 17) {
        transferBeat(dut, p, abandoned, beat, 32, rng, stalls = false)
      }
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.io.out.valid.expect(true.B)
      dut.reset.poke(true.B)
      // No clock edge: this specifically checks asynchronous counter reset.
      dut.io.out.valid.expect(false.B)
      dut.clock.step()
      dut.reset.poke(false.B)
      transaction(dut, p, BigInt(p.parallelWidth, rng), 32, rng)
    }
  }

  for ((ratio, group) <- Seq((13, 4), (5, 2), (35, 16), (5, 8), (2, 8))) {
    it should s"handle ratio $ratio with groups of $group and a partial final group" in {
      val factors = Seq(1, group - 1, group, group + 1, ratio - 1, ratio)
        .filter(factor => factor >= 1 && factor <= ratio)
        .distinct
      val p       = ParallelAndSerialConverterParams(
        parallelWidth           = ratio * 13,
        serialWidth             = 13,
        earlyTerminate          = true,
        allowedTerminateFactors = factors,
        p2sChunksPerGroup        = group
      )
      test(new GroupedParallelToSerialTestHarness(p)) { dut =>
        val rng = new Random(1000L + ratio * 31 + group)
        initialize(dut, factors.head)
        for (factor <- factors ++ factors.reverse) {
          transaction(dut, p, BigInt(p.parallelWidth, rng), factor, rng)
        }
      }
    }
  }

  it should "retain the ratio-one combinational passthrough, including its busy and reset behavior" in {
    val p = ParallelAndSerialConverterParams(
      parallelWidth           = 19,
      serialWidth             = 19,
      earlyTerminate          = true,
      allowedTerminateFactors = Seq(1),
      p2sChunksPerGroup        = 8
    )
    test(new GroupedParallelToSerialTestHarness(p)) { dut =>
      val rng = new Random(1L)
      initialize(dut, 1)
      for {
        valid        <- Seq(false, true)
        ready        <- Seq(false, true)
        busy         <- Seq(false, true)
        resetCounter <- Seq(false, true)
      } {
        val word = BigInt(p.parallelWidth, rng)
        dut.io.in.bits.poke(word.U)
        dut.io.in.valid.poke(valid.B)
        dut.io.out.ready.poke(ready.B)
        dut.io.is_busy_cstate.poke(busy.B)
        dut.io.counter_value_reset.poke(resetCounter.B)
        dut.io.out.bits.expect(word.U)
        dut.io.out.valid.expect(valid.B)
        dut.io.in.ready.expect(ready.B)
        dut.clock.step()
      }
    }
  }

  it should "serialize the 32768-to-1024 configuration with stalls and consecutive full words" in {
    val p = ParallelAndSerialConverterParams(
      parallelWidth    = 32768,
      serialWidth      = 1024,
      p2sChunksPerGroup = 8
    )
    test(new GroupedParallelToSerialTestHarness(p)) { dut =>
      val rng = new Random(32768L)
      initialize(dut, 32)
      transaction(dut, p, BigInt(p.parallelWidth, rng), 32, rng)
      transaction(dut, p, BigInt(p.parallelWidth, rng), 32, rng, stalls = false)
    }
  }

  // Serializer widths and runtime factors from the PR 648 and PR 649
  // snax_versacore_to_256KB_cluster configurations. Include the actual default
  // group size and an alternate size so both grouped and single-group layouts
  // are exercised at these application widths.
  for ((pr, parallelWidth, serialWidth, factors, alternateGroup) <- Seq(
    (648, 8192, 1024, Seq(1, 4, 8), 8),
    (649, 2048, 512, Seq(2, 4), 2)
  )) {
    val defaultParams = ParallelAndSerialConverterParams(
      parallelWidth           = parallelWidth,
      serialWidth             = serialWidth,
      earlyTerminate          = true,
      allowedTerminateFactors = factors
    )
    val layouts = Seq(
      defaultParams,
      defaultParams.copy(p2sChunksPerGroup = 4),
      defaultParams.copy(p2sChunksPerGroup = alternateGroup)
    ).distinct
    for (p <- layouts) {
      it should s"serialize PR $pr with groups of ${p.p2sChunksPerGroup}, stalls, and consecutive words" in {
        test(new GroupedParallelToSerialTestHarness(p)) { dut =>
          val rng = new Random(pr.toLong * 100 + p.p2sChunksPerGroup)
          initialize(dut, factors.head)

          for (factor <- factors ++ factors.reverse) {
            transaction(dut, p, BigInt(p.parallelWidth, rng), factor, rng)
          }
          // With no stalls or gaps, accepted final and next first chunks occur
          // on consecutive clock edges while the runtime factor changes.
          for (factor <- factors ++ factors.reverse) {
            transaction(dut, p, BigInt(p.parallelWidth, rng), factor, rng, stalls = false)
          }
        }
      }
    }
  }
}
