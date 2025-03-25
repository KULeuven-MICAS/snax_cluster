package snax_acc.spatial_array

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class SpatialArrayTest extends AnyFlatSpec with ChiselScalatestTester {

  "SpatialArray" should "correctly compute output with random test data" in {
    val params = SpatialArrayParam(
      macNum = 8,
      inputAType = UInt(8.W),
      inputBType = UInt(8.W),
      inputCType = UInt(32.W),
      mulType = UInt(16.W),
      outType = UInt(32.W),
      inputAWidth = 64,
      inputBWidth = 64,
      inputCWidth = 256,
      outputWidth = 256,
      arrayDim = Seq(Seq(2, 2, 2), Seq(2, 1, 4), Seq(4, 1, 2), Seq(4, 2, 1), Seq(1, 8, 1))
    )

    test(new SpatialArray(params)).withAnnotations(Seq(WriteVcdAnnotation)) { c =>
      val rand = new Random()

      // Generate random values for 'a' and 'b'
      val aValues = Array.fill(8)(rand.nextInt(256))  // 8 random 8-bit values
      val bValues = Array.fill(8)(rand.nextInt(256))  // 8 random 8-bit values

      // Construct 'a' and 'b' from the random values
      val a = aValues.zipWithIndex.map { case (v, i) => BigInt(v) << (i * 8) }.sum
      val b = bValues.zipWithIndex.map { case (v, i) => BigInt(v) << (i * 8) }.sum

      // Print generated test inputs
      println(s"Randomly generated A values: ${aValues.mkString(", ")}")
      println(s"Randomly generated B values: ${bValues.mkString(", ")}")

      val results = Seq(
        Seq(
          aValues(0) * bValues(0) + aValues(1) * bValues(1),
          aValues(0) * bValues(2) + aValues(1) * bValues(3),
          aValues(2) * bValues(0) + aValues(3) * bValues(1),
          aValues(2) * bValues(2) + aValues(3) * bValues(3)
        ),
        Seq(
          aValues(0) * bValues(0),
          aValues(0) * bValues(1),
          aValues(0) * bValues(2),
          aValues(0) * bValues(3),
          aValues(1) * bValues(0),
          aValues(1) * bValues(1),
          aValues(1) * bValues(2),
          aValues(1) * bValues(3)
        ),
        Seq(
          aValues(0) * bValues(0), aValues(0) * bValues(1), aValues(1) * bValues(0), aValues(1) * bValues(1),
          aValues(2) * bValues(0), aValues(2) * bValues(1), aValues(3) * bValues(0), aValues(3) * bValues(1)
        ),
        Seq(
          aValues(0) * bValues(0) + aValues(1) * bValues(1),
          aValues(2) * bValues(0) + aValues(3) * bValues(1),
          aValues(4) * bValues(0) + aValues(5) * bValues(1),
          aValues(6) * bValues(0) + aValues(7) * bValues(1)
        ),
        Seq(
          (0 until 8).map(i => aValues(i) * bValues(i)).sum
        )
      )

      // Poke the random values
      c.io.data.in_a.bits.poke(a.U)
      c.io.data.in_b.bits.poke(b.U)
      c.io.data.in_c.bits.poke(0.U)

      // Enable valid signals
      c.io.data.in_a.valid.poke(true.B)
      c.io.data.in_b.valid.poke(true.B)
      c.io.data.in_c.valid.poke(true.B)

      // Test different configurations
      for (cfg <- results.indices) {
        c.io.ctrl.spatialArrayCfg.poke(cfg.U)
        c.clock.step(1)

        val out_d = c.io.data.out_d.bits.peek().litValue
        val extractedOutputs = Seq(
          out_d & 0xFFFFFFFFL,
          (out_d >> 32)  & 0xFFFFFFFFL,
          (out_d >> 64)  & 0xFFFFFFFFL,
          (out_d >> 96)  & 0xFFFFFFFFL,
          (out_d >> 128) & 0xFFFFFFFFL,
          (out_d >> 160) & 0xFFFFFFFFL,
          (out_d >> 192) & 0xFFFFFFFFL,
          (out_d >> 224) & 0xFFFFFFFFL
        )

        println(s"Checking res_cfg${cfg + 1}...")

        for (i <- results(cfg).indices) {
          val expected = results(cfg)(i)
          val actual = extractedOutputs(i)
          println(s"  Output[$i]: $actual (Expected: $expected)")
          assert(actual == expected, s"Mismatch at index $i: got $actual, expected $expected")
        }
      }
    }
  }
}
