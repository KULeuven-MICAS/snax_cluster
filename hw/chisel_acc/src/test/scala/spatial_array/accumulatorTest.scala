package snax_acc.spatial_array

import chisel3._

import chiseltest._
import org.scalatest.funsuite.AnyFunSuite

class AccumulatorTest extends AnyFunSuite with ChiselScalatestTester {
  test("Accumulator should correctly add or accumulate values") {
    val numElements = 4

    test(new Accumulator(OpType.UIntUIntOp, 8, 16, numElements)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      // Helper function to run tests with different configurations
      def testConfig(
          in1Values: Seq[Int],
          in2Values: Seq[Int],
          addExtIn: Boolean,
          expectedOutput: Seq[Int]
      ): Unit = {
        // Set input values
        in1Values.zipWithIndex.foreach { case (value, idx) =>
          dut.io.in1.bits(idx).poke(value.U)
        }
        in2Values.zipWithIndex.foreach { case (value, idx) =>
          dut.io.in2.bits(idx).poke(value.U)
        }

        // Set control signals
        dut.io.add_ext_in.poke(addExtIn.B)
        dut.io.in1.valid.poke(true.B)
        dut.io.in2.valid.poke(true.B)

        // Step clock
        dut.clock.step()

        // Check expected output
        expectedOutput.zipWithIndex.foreach { case (expected, idx) =>
          dut.io.out(idx).expect(expected.U)
        }
      }

      // Test case 1: Element-wise addition when add_ext_in is true
      testConfig(
        in1Values = Seq(1, 2, 3, 4),
        in2Values = Seq(4, 3, 2, 1),
        addExtIn = true,
        expectedOutput = Seq(5, 5, 5, 5) // [1+4, 2+3, 3+2, 4+1]
      )

      // Test case 2: Accumulate in1 values when add_ext_in is false
      testConfig(
        in1Values = Seq(1, 2, 3, 4),
        in2Values = Seq(0, 0, 0, 0), // Unused
        addExtIn = false,
        expectedOutput = Seq(6, 7, 8, 9) // Initial accumulation
      )
      testConfig(
        in1Values = Seq(1, 2, 3, 4),
        in2Values = Seq(0, 0, 0, 0), // Unused
        addExtIn = false,
        expectedOutput = Seq(7, 9, 11, 13) // Initial accumulation
      )

      // Test case 3: clear accumulator
      testConfig(
        in1Values = Seq(0, 0, 0, 0), // Unused
        in2Values = Seq(0, 0, 0, 0), // Unused
        addExtIn = true,
        expectedOutput = Seq(0, 0, 0, 0) // Initial accumulation
      )

      // Continue accumulating
      testConfig(
        in1Values = Seq(2, 3, 4, 5),
        in2Values = Seq(0, 0, 0, 0), // Unused
        addExtIn = false,
        expectedOutput = Seq(2, 3, 4, 5) // 
      )
    }
  }
}
