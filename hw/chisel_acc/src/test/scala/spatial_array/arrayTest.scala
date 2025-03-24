package snax_acc.spatial_array

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SpatialArrayTest extends AnyFlatSpec with ChiselScalatestTester {

  "SpatialArray" should "correctly compute output" in {
    // Define test parameters
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
      arrayDim = Seq(Seq(2, 2, 2), Seq(2, 1, 4), Seq(4, 1, 2), Seq(4, 2, 1), Seq(1, 8, 1)), // 2x2x2 array dimensions
    )

    // Instantiate the module
    test(new SpatialArray(params)).withAnnotations(Seq(WriteVcdAnnotation)) { c =>
      // Construct 'a' using shift and OR operations
      val a = (BigInt(8.toByte)  << 56) | 
              (BigInt(7.toByte)  << 48) | 
              (BigInt(6.toByte)  << 40) | 
              (BigInt(5.toByte)  << 32) | 
              (BigInt(4.toByte)  << 24) | 
              (BigInt(3.toByte)  << 16) | 
              (BigInt(2.toByte)  << 8)  | 
              (BigInt(1.toByte))

      // Construct 'b' using shift and OR operations
      val b = (BigInt(16.toByte) << 56) | 
              (BigInt(15.toByte) << 48) | 
              (BigInt(14.toByte) << 40) | 
              (BigInt(13.toByte) << 32) | 
              (BigInt(12.toByte) << 24) | 
              (BigInt(11.toByte) << 16) | 
              (BigInt(10.toByte) << 8)  | 
              (BigInt(9.toByte))

      val results = Seq(
        Seq( // res_cfg1
          1 * 9 + 2 * 10,
          1 * 11 + 2 * 12,
          3 * 9 + 4 * 10,
          3 * 11 + 4 * 12
        ),
        Seq( // res_cfg2
          1 * 9,
          1 * 10,
          1 * 11,
          1 * 12,
          2 * 9,
          2 * 10,
          2 * 11,
          2 * 12
        ),
        Seq( // res_cfg3
          1 * 9, 1 * 10, 2 * 9, 2 * 10,
          3 * 9, 3 * 10, 4 * 9, 4 * 10
        ),
        Seq( // res_cfg4
          1 * 9 + 2 * 10,
          3 * 9 + 4 * 10,
          5 * 9 + 6 * 10,
          7 * 9 + 8 * 10
        ),
        Seq( // res_cfg5
          1 * 9 + 2 * 10 + 3 * 11 + 4 * 12 + 5 * 13 + 6 * 14 + 7 * 15 + 8 * 16
        )
      )

      // Poke the values
      c.io.data.in_a.bits.poke(a.U)
      c.io.data.in_b.bits.poke(b.U)
      c.io.data.in_c.bits.poke(0.U)

      // Test the ready/valid handshake
      c.io.data.in_a.valid.poke(true.B)
      c.io.data.in_b.valid.poke(true.B)
      c.io.data.in_c.valid.poke(true.B)

      // Test configurations 0 to 4
      for (cfg <- 0 until results.length) {
        c.io.ctrl.spatialArrayCfg.poke(cfg.U)
        c.clock.step(1)

        // Read 256-bit output and extract 8 separate 32-bit values
        val out_d = c.io.data.out_d.bits.peek().litValue
        val extractedOutputs = Seq(
          out_d & 0xFFFFFFFFL,          // Lowest 32 bits (LSB)
          (out_d >> 32)  & 0xFFFFFFFFL,
          (out_d >> 64)  & 0xFFFFFFFFL,
          (out_d >> 96)  & 0xFFFFFFFFL,
          (out_d >> 128) & 0xFFFFFFFFL,
          (out_d >> 160) & 0xFFFFFFFFL,
          (out_d >> 192) & 0xFFFFFFFFL,
          (out_d >> 224) & 0xFFFFFFFFL   // Highest 32 bits (MSB)
        )

        // Print and compare results
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
