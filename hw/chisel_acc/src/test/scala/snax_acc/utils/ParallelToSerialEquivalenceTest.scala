package snax_acc.utils

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable
import scala.util.Random

// The pre-group implementation from commit 41825eae. Keep the single wide
// shift register here: reusing the new grouping logic would weaken this test.
class MonolithicParallelToSerialReference(p: ParallelAndSerialConverterParams)
    extends Module
    with RequireAsyncReset {
  val io = IO(new Bundle {
    val in                  = Flipped(Decoupled(UInt(p.parallelWidth.W)))
    val terminate_factor    =
      if (p.earlyTerminate) Some(Input(UInt(log2Ceil(p.parallelWidth / p.serialWidth + 1).W)))
      else None
    val out                 = Decoupled(UInt(p.serialWidth.W))
    val counter_value_reset = Input(Bool())
    val is_busy_cstate      = Input(Bool())
  })

  val ratio = p.parallelWidth / p.serialWidth

  if (p.earlyTerminate) {
    assert(p.allowedTerminateFactors.map(f => io.terminate_factor.get === f.U).reduce(_ || _))
  }

  if (ratio == 1) {
    io.out.valid := io.in.valid
    io.out.bits  := io.in.bits
    io.in.ready  := io.out.ready
  } else {
    val counter = Module(new BasicCounter(width = log2Ceil(ratio), hasCeil = true))
    if (p.earlyTerminate) {
      counter.io.ceilOpt.get := io.terminate_factor.get
    } else {
      counter.io.ceilOpt.get := ratio.U
    }
    counter.io.reset := io.counter_value_reset
    counter.io.tick  := io.out.fire

    val shiftReg = Reg(UInt((p.parallelWidth - p.serialWidth).W))
    when(io.out.fire) {
      when(counter.io.value === 0.U) {
        shiftReg := io.in.bits(p.parallelWidth - 1, p.serialWidth)
      }.otherwise {
        shiftReg := shiftReg >> p.serialWidth
      }
    }

    when(counter.io.value === 0.U) {
      io.out.valid := io.in.valid
      io.out.bits  := io.in.bits(p.serialWidth - 1, 0)
      io.in.ready  := io.out.ready && io.is_busy_cstate
    }.otherwise {
      io.out.valid := true.B
      io.out.bits  := shiftReg(p.serialWidth - 1, 0)
      io.in.ready  := false.B && io.is_busy_cstate
    }
  }
}

class ParallelToSerialEquivalenceResult(serialWidth: Int) extends Bundle {
  val ready = Bool()
  val valid = Bool()
  val bits  = UInt(serialWidth.W)
}

class ParallelToSerialEquivalenceHarness(p: ParallelAndSerialConverterParams, groups: Seq[Int]) extends Module {
  val io = IO(new Bundle {
    val in_bits          = Input(UInt(p.parallelWidth.W))
    val in_valid         = Input(Bool())
    val out_ready        = Input(Bool())
    val busy             = Input(Bool())
    val counter_reset    = Input(Bool())
    val terminate_factor = Input(UInt(log2Ceil(p.parallelWidth / p.serialWidth + 1).W))
    val reference        = Output(new ParallelToSerialEquivalenceResult(p.serialWidth))
    val grouped          = Output(Vec(groups.size, new ParallelToSerialEquivalenceResult(p.serialWidth)))
  })

  val reference = withReset(reset.asBool.asAsyncReset) {
    Module(new MonolithicParallelToSerialReference(p))
  }
  reference.io.in.bits             := io.in_bits
  reference.io.in.valid            := io.in_valid
  reference.io.out.ready           := io.out_ready
  reference.io.is_busy_cstate      := io.busy
  reference.io.counter_value_reset := io.counter_reset
  reference.io.terminate_factor.foreach(_ := io.terminate_factor)
  io.reference.ready              := reference.io.in.ready
  io.reference.valid              := reference.io.out.valid
  io.reference.bits               := reference.io.out.bits

  for ((group, index) <- groups.zipWithIndex) {
    val converter = withReset(reset.asBool.asAsyncReset) {
      Module(new ParallelToSerial(p.copy(p2sChunksPerGroup = group)))
    }
    converter.io.in.bits             := io.in_bits
    converter.io.in.valid            := io.in_valid
    converter.io.out.ready           := io.out_ready
    converter.io.is_busy_cstate      := io.busy
    converter.io.counter_value_reset := io.counter_reset
    converter.io.terminate_factor.foreach(_ := io.terminate_factor)
    io.grouped(index).ready          := converter.io.in.ready
    io.grouped(index).valid          := converter.io.out.valid
    io.grouped(index).bits           := converter.io.out.bits
  }
}

class ParallelToSerialEquivalenceTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "ParallelToSerial equivalence to the original monolithic serializer"

  private val groups = Seq(2, 4, 8, 16)

  // Include one-bit payloads, non-power-of-two widths/ratios, partial groups,
  // bypass, single-group layouts, both PR widths, and the original wide case.
  private val geometries = Seq(
    (1, 1), (2, 3), (3, 7), (4, 13), (5, 1), (7, 13), (8, 32), (13, 7),
    (16, 1), (17, 13), (32, 16), (35, 7), (4, 512), (8, 1024), (32, 1024)
  )

  for {
    (ratio, serialWidth) <- geometries
    earlyTerminate      <- Seq(false, true)
  } {
    it should s"match every valid cycle for ratio $ratio, width $serialWidth, early termination $earlyTerminate" in {
      val p = ParallelAndSerialConverterParams(
        parallelWidth           = ratio * serialWidth,
        serialWidth             = serialWidth,
        earlyTerminate          = earlyTerminate,
        allowedTerminateFactors = if (earlyTerminate) 1 to ratio else Seq.empty
      )
      test(new ParallelToSerialEquivalenceHarness(p, groups)) { dut =>
        dut.clock.setTimeout(20000)
        val rng     = new Random(0x4551554956L + ratio * 101 + serialWidth * 7 + (if (earlyTerminate) 1 else 0))
        val visited = mutable.Set.empty[Int]
        var phase   = 0
        var cycles  = 0

        def cycle(
          word:         BigInt,
          valid:        Boolean,
          ready:        Boolean,
          busy:         Boolean,
          factor:       Int,
          counterReset: Boolean = false,
          asyncReset:   Boolean = false
        ): Unit = {
          dut.io.in_bits.poke(word.U)
          dut.io.in_valid.poke(valid.B)
          dut.io.out_ready.poke(ready.B)
          dut.io.busy.poke(busy.B)
          dut.io.counter_reset.poke(counterReset.B)
          dut.io.terminate_factor.poke(factor.U)
          dut.reset.poke(asyncReset.B)
          if (asyncReset) phase = 0

          // Compare before the edge, including stalled valid output. After
          // asynchronous reset this also checks its effect without an edge.
          val referenceValid = dut.io.reference.valid.peek().litToBoolean
          val referenceReady = dut.io.reference.ready.peek().litToBoolean
          withClue(s"cycle=$cycles phase=$phase factor=$factor reset=$counterReset async=$asyncReset: ") {
            dut.io.reference.valid.expect((phase != 0 || valid).B)
            dut.io.reference.ready.expect((ready && (ratio == 1 || (phase == 0 && busy))).B)
            val referenceBits = if (referenceValid) Some(dut.io.reference.bits.peek().litValue) else None
            for ((group, index) <- groups.zipWithIndex) {
              withClue(s"group=$group: ") {
                dut.io.grouped(index).valid.expect(referenceValid.B)
                dut.io.grouped(index).ready.expect(referenceReady.B)
                // Both designs intentionally have unreset payload registers.
                // Invalid output is outside the interface's data contract.
                referenceBits.foreach(bits => dut.io.grouped(index).bits.expect(bits.U))
              }
            }
          }
          if (referenceValid && ready) visited += phase
          dut.clock.step()
          if (asyncReset || counterReset || ratio == 1) phase = 0
          else if (referenceValid && ready) phase = if (phase < factor - 1) phase + 1 else 0
          cycles += 1
        }

        def randomWord(): BigInt = BigInt(p.parallelWidth, rng)

        cycle(0, valid = false, ready = false, busy = true, factor = ratio, asyncReset = true)
        cycle(0, valid = false, ready = false, busy = true, factor = ratio)

        // Every possible legal termination factor and every output position,
        // including both sides of each group boundary. The first word remains
        // stable under backpressure; later upstream data deliberately changes.
        for (factor <- (if (earlyTerminate) 1 to ratio else Seq(ratio))) {
          val word = randomWord()
          for (beat <- 0 until factor) {
            assert(phase == beat)
            val input = if (beat == 0) word else randomWord()
            cycle(input, valid = true, ready = false, busy = true, factor = factor)
            cycle(input, valid = true, ready = true, busy = true, factor = factor)
          }
        }

        // Abort at the first beat, group boundaries, and the final beat, both
        // on and off a transfer edge. The following transaction reloads data.
        val abortPoints = (Seq(0, 1, ratio - 1) ++ groups.flatMap(g => Seq(g - 1, g, g + 1)))
          .filter(beat => beat >= 0 && beat < ratio)
          .distinct
        for {
          abortAt <- abortPoints
          ready   <- Seq(false, true)
        } {
          for (_ <- 0 until abortAt) {
            cycle(randomWord(), valid = true, ready = true, busy = true, factor = ratio)
          }
          cycle(randomWord(), valid = true, ready = ready, busy = true, factor = ratio, counterReset = true)
          assert(phase == 0)
        }

        // Differential stress deliberately also exercises the legacy busy
        // behavior: output can transfer while input ready is busy-gated. These
        // cycles check compatibility, rather than prescribing upstream usage.
        var factor = ratio
        for (index <- 0 until 600) {
          if (earlyTerminate && (phase == 0 || index >= 400)) factor = 1 + rng.nextInt(ratio)
          // The first 400 cycles hold factors throughout each transaction.
          // The last 200 also compare legal factor changes during a word.
          cycle(
            randomWord(),
            valid        = rng.nextBoolean(),
            ready        = rng.nextInt(4) != 0,
            busy         = rng.nextBoolean(),
            factor       = factor,
            counterReset = rng.nextInt(31) == 0,
            asyncReset   = rng.nextInt(73) == 0
          )
        }

        assert(visited == (0 until ratio).toSet, s"Missing transferred phases: ${(0 until ratio).toSet -- visited}")
      }
    }
  }
}
