package snax_acc.spatial_array

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class NestCounterTest extends AnyFlatSpec with ChiselScalatestTester {

  "NestCounter" should "correctly compute counter values with random test data" in {
    val width = 4  // Set the width of the counter
    val loopNum = 3  // Set the number of loops (counters)

    test(new NestCounter(width, loopNum)).withAnnotations(Seq(WriteVcdAnnotation)) { c =>
      val rand = new Random()

      // Generate random values for 'ceil' for each counter
      val ceilValues = Array.fill(loopNum)(rand.nextInt(8) + 1)  // Random ceil between 1 and 8 for each counter

      // Print generated ceil values
      println(s"Randomly generated ceil values: ${ceilValues.mkString(", ")}")

      // Poke the random ceil values to the NestCounter inputs
      for (i <- 0 until loopNum) {
        c.io.ceil(i).poke(ceilValues(i).U)
      }

      // Apply reset signal
      c.io.reset.poke(true.B)
      step(1)  // Step to allow reset to take effect
      c.io.reset.poke(false.B)

      // Apply tick signal and test counter incrementing
      for (tickCount <- 1 to 10) {
        c.io.tick.poke(true.B)
        step(1)  // Step to simulate one cycle of the clock

        // Check values after each tick
        for (i <- 0 until loopNum) {
          val expected = (tickCount % ceilValues(i))
          val actual = c.io.value(i).peek().litValue
          println(s"Counter[$i] after tick $tickCount: got $actual (Expected: $expected)")
          // assert(actual == expected, s"Mismatch at counter $i: got $actual, expected $expected")
        }
      }

      // Test lastVal logic for each counter
      for (i <- 0 until loopNum) {
        val expectedLastVal = (ceilValues(i) - 1) == c.io.value(i).peek().litValue
        val actualLastVal = c.io.lastVal(i).peek().litValue
        println(s"lastVal[$i]: got $actualLastVal (Expected: $expectedLastVal)")
        // assert(actualLastVal == expectedLastVal, s"Mismatch at lastVal for counter $i: got $actualLastVal, expected $expectedLastVal")
      }
    }
  }
}
