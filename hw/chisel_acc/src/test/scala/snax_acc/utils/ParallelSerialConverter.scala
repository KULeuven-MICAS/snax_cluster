package snax_acc.utils

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ParallelToSerialTest extends AnyFlatSpec with ChiselScalatestTester {

  "ParallelToSerial" should "convert parallel data into multiple serial chunks" in {
    val parallelWidth = 16
    val serialWidth   = 4

    test(
      new ParallelToSerial(ParallelAndSerialConverterParams(parallelWidth, serialWidth))
    ).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>

      // Default values
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.io.start.poke(false.B)
      dut.clock.step()

      // Pulse start once to initialize the counter
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      def runOnce(parallelData: UInt, expectedChunks: Seq[UInt]): Unit = {
        // Present the parallel word for the first chunk
        dut.io.in.bits.poke(parallelData)
        dut.io.in.valid.poke(true.B)

        for ((exp, idx) <- expectedChunks.zipWithIndex) {
          // Consumer is ready on every cycle while we expect data
          dut.io.out.ready.poke(true.B)

          // For this design, io.out.valid is high whenever data is being
          // produced (first chunk: follows io.in.valid, later chunks: always true)
          dut.io.out.valid.expect(true.B)
          dut.io.out.bits.expect(exp)

          // Complete this transfer
          dut.clock.step()

          // After the first chunk the module has latched all the remaining chunks,
          // so we can de-assert io.in.valid.
          if (idx == 0) {
            dut.io.in.valid.poke(false.B)
          }
        }

        // After the last chunk the internal counter has wrapped back to 0.
        // With io.in.valid low, no more valid data should be present.
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        dut.io.out.valid.expect(false.B)
      }

      // 0xABCD in chunks (LSB first): D, C, B, A
      runOnce("hABCD".U, Seq("hD".U(4.W), "hC".U(4.W), "hB".U(4.W), "hA".U(4.W)))

      // 0xEFEF in chunks (LSB first): F, E, F, E
      runOnce("hEFEF".U, Seq("hF".U(4.W), "hE".U(4.W), "hF".U(4.W), "hE".U(4.W)))
    }
  }
}

class SerialToParallelSpec extends AnyFlatSpec with ChiselScalatestTester with Matchers {

  behavior of "SerialToParallel"

  it should "collect serial data and produce parallel output correctly" in {
    test(
      new SerialToParallel(
        ParallelAndSerialConverterParams(
          serialWidth   = 8,  // 8 bits per serial input
          parallelWidth = 32  // 32 bits parallel output
        )
      )
    ).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>

      // Initial defaults
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.io.start.poke(false.B)
      dut.clock.step()

      // Start the module once (initialise internal counter)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      def sendFrame(bytes: Seq[Int]): Unit = {
        require(bytes.length == 4) // ratio = 32 / 8 = 4

        // LSB comes from the first byte, MSB from the last
        val expectedParallel =
          (bytes(3) << 24) | (bytes(2) << 16) | (bytes(1) << 8) | bytes(0)

        for ((b, idx) <- bytes.zipWithIndex) {
          val last = idx == bytes.length - 1

          // For the last byte we must be ready to accept the parallel word
          dut.io.out.ready.poke(last.B)

          dut.io.in.bits.poke(b.U)
          dut.io.in.valid.poke(true.B)

          if (!last) {
            // Not enough bytes yet: output must not be valid
            dut.io.out.valid.expect(false.B)
          } else {
            // On the cycle of the last byte, the full parallel word is visible
            dut.io.out.valid.expect(true.B)
            dut.io.out.bits.expect(expectedParallel.U)
          }

          // Perform the transfer
          dut.clock.step()
          dut.io.in.valid.poke(false.B)
        }

        // After the last transfer, the counter is back to 0 and
        // with no new input, output should no longer be valid.
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        dut.io.out.valid.expect(false.B)
      }

      // First frame: bytes (LSB→MSB) AB, CD, 12, 34 -> 0x34_12_CD_AB
      sendFrame(Seq(0xab, 0xcd, 0x12, 0x34))

      // Second frame: bytes (LSB→MSB) EF, AA, BB, CC -> 0xCC_BB_AA_EF
      sendFrame(Seq(0xef, 0xaa, 0xbb, 0xcc))
    }
  }
}

class SerialToParallelEarlyTerminateSpec
    extends AnyFlatSpec
    with ChiselScalatestTester
    with Matchers {

  behavior of "SerialToParallel with earlyTerminate"

  it should "operate normally when terminate_factor equals the full ratio" in {
    val params = ParallelAndSerialConverterParams(
      serialWidth            = 8,
      parallelWidth          = 32,
      earlyTerminate         = true,
      allowedTerminateFactors = Seq(4) // ratio = 32 / 8 = 4
    )

    test(new SerialToParallel(params)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      // Default IO
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.io.start.poke(false.B)
      dut.clock.step()

      // Pulse start to initialise the counter
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.clock.step()

      // Use the allowed factor == ratio
      dut.io.terminate_factor.get.poke(4.U)

      def sendFrame(bytes: Seq[Int]): Unit = {
        require(bytes.length == 4)
        val expected =
          (bytes(3) << 24) | (bytes(2) << 16) | (bytes(1) << 8) | bytes(0)

        for ((b, idx) <- bytes.zipWithIndex) {
          val last = idx == bytes.length - 1

          dut.io.in.bits.poke(b.U)
          dut.io.in.valid.poke(true.B)
          dut.io.out.ready.poke(last.B)

          if (!last) {
            // Not enough bytes yet: no valid parallel output
            dut.io.out.valid.expect(false.B)
          } else {
            // On the last byte, the full parallel word is produced
            dut.io.out.valid.expect(true.B)
            dut.io.out.bits.expect(expected.U)
          }

          dut.clock.step()
          dut.io.in.valid.poke(false.B)
        }

        dut.io.out.ready.poke(false.B)
        dut.clock.step()
        dut.io.out.valid.expect(false.B)
      }

      // Frame 1: AB, CD, 12, 34 -> 0x34_12_CD_AB
      sendFrame(Seq(0xab, 0xcd, 0x12, 0x34))

      // Frame 2: EF, AA, BB, CC -> 0xCC_BB_AA_EF
      sendFrame(Seq(0xef, 0xaa, 0xbb, 0xcc))
    }
  }

  it should "assert at runtime when terminate_factor is not allowed" in {
    val params = ParallelAndSerialConverterParams(
      serialWidth            = 8,
      parallelWidth          = 32,
      earlyTerminate         = true,
      allowedTerminateFactors = Seq(4) // only 4 is allowed
    )

    intercept[Exception] {
      test(new SerialToParallel(params)) { dut =>
        // Illegal terminate_factor: 2 not in Seq(4)
        dut.io.terminate_factor.get.poke(2.U)

        // Just tick a bit so the assertion is evaluated
        dut.io.start.poke(false.B)
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(false.B)
        dut.clock.step()
      }
    }
  }
}

object ParallelToSerialConverterEmitter   extends App {
  println(emitVerilog(new ParallelToSerial(ParallelAndSerialConverterParams(16, 4))))
}
object SerialToParallelConverterEmitter extends App {
  println(emitVerilog(new SerialToParallel(ParallelAndSerialConverterParams(16, 4))))
}
