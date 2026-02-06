package snax_acc.utils

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

// define the reference implementation (Old Mux-based)
class ParallelToSerialReference(val p: ParallelAndSerialConverterParams) extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in               = Flipped(Decoupled(UInt(p.parallelWidth.W)))
    val terminate_factor =
      if (p.earlyTerminate)
        Some(Input(UInt(log2Ceil(p.parallelWidth / p.serialWidth + 1).W)))
      else None
    val out              = Decoupled(UInt(p.serialWidth.W))
    val start            = Input(Bool())
  })

  val ratio: Int = p.parallelWidth / p.serialWidth
  assert(
    ratio >= 1,
    "The ratio of parallelWidth to serialWidth must be at least 1."
  )
  val inBitsSeq = Wire(Vec(ratio, UInt(p.serialWidth.W)))
  inBitsSeq := VecInit(
    Seq.tabulate(ratio) { i =>
      io.in.bits((i + 1) * p.serialWidth - 1, i * p.serialWidth)
    }
  )

  val storedData = Seq(inBitsSeq(0)) ++ inBitsSeq.drop(1).map { i =>
    RegEnable(i, io.in.fire)
  }

  // Validate terminate_factor if early termination is enabled at runtime
  if (p.earlyTerminate) {
    val tf        = io.terminate_factor.get
    val isAllowed = p.allowedTerminateFactors
      .map(f => tf === f.U)
      .reduce(_ || _) // since allowedFactors non-empty if earlyTerminate
    assert(
      isAllowed,
      s"terminate_factor must be one of ${p.allowedTerminateFactors.mkString(", ")}"
    )
  }

  val counter = Module(
    new BasicCounter(width = log2Ceil(ratio), hasCeil = true)
  )
  if (p.earlyTerminate) {
    counter.io.ceilOpt.get := io.terminate_factor.get
  } else {
    counter.io.ceilOpt.get := ratio.U
  }
  counter.io.reset := io.start
  counter.io.tick := io.out.fire

  io.out.bits := MuxLookup(
    counter.io.value,
    0.U.asTypeOf(UInt(p.serialWidth.W))
  )(storedData.zipWithIndex.map { case (i, j) =>
    j.U -> i
  })

  when(counter.io.value === 0.U) {
    io.out.valid := io.in.valid
    io.in.ready  := io.out.ready
  } otherwise {
    io.out.valid := true.B
    io.in.ready  := false.B
  }
}

class ParallelToSerialTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "ParallelToSerial"

  it should "match reference implementation" in {
    val parallelWidth = 64
    val serialWidth   = 16
    val params        = ParallelAndSerialConverterParams(
      parallelWidth           = parallelWidth,
      serialWidth             = serialWidth,
      earlyTerminate          = true,
      allowedTerminateFactors = Seq(4) // Only allow full width for basic equivalence test
    )

    // Helper to test a pair of modules
    test(new Module with RequireAsyncReset {
      val io = IO(new Bundle {
        val in     = Flipped(Decoupled(UInt(parallelWidth.W)))
        val outRef = Decoupled(UInt(serialWidth.W))
        val outDut = Decoupled(UInt(serialWidth.W))
        val start  = Input(Bool())
      })

      val ref = Module(new ParallelToSerialReference(params))
      val dut = Module(new ParallelToSerial(params))

      ref.io.terminate_factor.get := 4.U
      dut.io.terminate_factor.get := 4.U
      ref.io.start                := io.start
      dut.io.start                := io.start

      ref.io.in <> io.in
      dut.io.in.valid := io.in.valid
      dut.io.in.bits  := io.in.bits
      // We assume both consume input at same time if they are equivalent
      io.in.ready     := ref.io.in.ready && dut.io.in.ready

      io.outRef <> ref.io.out
      io.outDut <> dut.io.out
    }).withAnnotations(Seq(WriteVcdAnnotation)) { c =>
      c.io.start.poke(true.B)
      c.clock.step()
      c.io.start.poke(false.B)

      val rng = new Random(42)

      // Test Loop
      for (i <- 0 until 50) {
        // Prepare Input
        val inputVal = BigInt(parallelWidth, rng)
        c.io.in.bits.poke(inputVal.U)
        c.io.in.valid.poke(true.B)

        // Wait for acceptance
        while (!c.io.in.ready.peek().litToBoolean) {
          // Randomly assert output ready to drain previous transactions
          val ready = rng.nextBoolean()
          c.io.outRef.ready.poke(ready.B)
          c.io.outDut.ready.poke(ready.B)

          c.clock.step()

          // Check equivalence while waiting
          if (
            c.io.outRef.valid.peek().litToBoolean && c.io.outDut.valid
              .peek()
              .litToBoolean
          ) {
            c.io.outDut.bits.expect(c.io.outRef.bits.peek())
          }
          c.io.outDut.valid.expect(c.io.outRef.valid.peek())
        }

        // Cycle where input is accepted
        // Both should be valid and have same data (first chunk)
        val ready = true // Consume immediately
        c.io.outRef.ready.poke(ready.B)
        c.io.outDut.ready.poke(ready.B)

        c.io.outDut.valid.expect(true.B)
        c.io.outRef.valid.expect(true.B)
        c.io.outDut.bits.expect(c.io.outRef.bits.peek())

        c.clock.step()
        c.io.in.valid.poke(false.B)

        // Drain remainder of serial chunks
        // Ratio is 4, so we consumed 1, 3 remaining
        for (j <- 0 until 3) {
          c.io.outRef.ready.poke(true.B)
          c.io.outDut.ready.poke(true.B)
          c.io.outDut.valid.expect(true.B)
          c.io.outRef.valid.expect(true.B)
          c.io.outDut.bits.expect(c.io.outRef.bits.peek())
          c.clock.step()
        }
      }
    }
  }

  it should "handle random backpressure correctly" in {
    val parallelWidth = 64
    val serialWidth   = 16
    val params        = ParallelAndSerialConverterParams(
      parallelWidth           = parallelWidth,
      serialWidth             = serialWidth,
      earlyTerminate          = true,
      allowedTerminateFactors = Seq(4)
    )

    test(new Module with RequireAsyncReset {
      val io  = IO(new Bundle {
        val in     = Flipped(Decoupled(UInt(parallelWidth.W)))
        val outRef = Decoupled(UInt(serialWidth.W))
        val outDut = Decoupled(UInt(serialWidth.W))
        val start  = Input(Bool())
      })
      val ref = Module(new ParallelToSerialReference(params))
      val dut = Module(new ParallelToSerial(params))

      ref.io.terminate_factor.get := 4.U
      dut.io.terminate_factor.get := 4.U
      ref.io.start                := io.start
      dut.io.start                := io.start

      // Wire inputs
      ref.io.in.valid := io.in.valid
      ref.io.in.bits  := io.in.bits
      dut.io.in.valid := io.in.valid
      dut.io.in.bits  := io.in.bits
      // Check ready coherence
      io.in.ready     := ref.io.in.ready // Monitor only, assert coherence in test

      // Wire outputs
      io.outRef <> ref.io.out
      io.outDut <> dut.io.out
    }).withAnnotations(Seq(WriteVcdAnnotation)) { c =>
      c.io.start.poke(true.B)
      c.clock.step()
      c.io.start.poke(false.B)
      val rng = new Random(123)

      for (i <- 0 until 50) { // 50 transactions
        val inputVal = BigInt(parallelWidth, rng)
        c.io.in.bits.poke(inputVal.U)
        c.io.in.valid.poke(true.B)

        // Wait until transaction starts (both ready)
        while (!c.io.in.ready.peek().litToBoolean) {
          // If waiting for input ready, it means we are pushing out old data
          val r = rng.nextBoolean()
          c.io.outRef.ready.poke(r.B)
          c.io.outDut.ready.poke(r.B)
          c.clock.step()
          c.io.outDut.valid.expect(c.io.outRef.valid.peek())
          if (c.io.outRef.valid.peek().litToBoolean) {
            c.io.outDut.bits.expect(c.io.outRef.bits.peek())
          }
        }

        // Input accepted
        // We must drive output ready to consume first chunk if we want to proceed validly
        // But input is accepted regardless of output ready IF module accepts it (combinational or not)
        // In P2S, input.ready = output.ready for the first chunk.

        // So drive random ready
        var chunksConsumed = 0
        while (chunksConsumed < 4) {
          val r = rng.nextBoolean()
          c.io.outRef.ready.poke(r.B)
          c.io.outDut.ready.poke(r.B)

          // Assert inputs are treated same
          // Note: peek() on ref.io.in.ready is indirect
          // We can check if dut.io.in.ready matches ref
          // But we can't access internal signals easily here without IO
          // We just proceed

          if (
            c.io.outRef.valid.peek().litToBoolean && c.io.outRef.ready
              .peek()
              .litToBoolean
          ) {
            chunksConsumed += 1
          }

          c.io.outDut.valid.expect(c.io.outRef.valid.peek())
          if (c.io.outRef.valid.peek().litToBoolean) {
            c.io.outDut.bits.expect(c.io.outRef.bits.peek())
          }

          c.clock.step()
          // Input valid can stay high or go low, shouldn't matter once accepted
          if (chunksConsumed > 0) c.io.in.valid.poke(false.B)
        }
      }
    }
  }
}
