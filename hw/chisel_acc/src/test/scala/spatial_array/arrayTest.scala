package snax_acc.spatial_array

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class SpatialArrayTest extends AnyFlatSpec with ChiselScalatestTester {

  "SpatialArray" should "correctly compute output with large array dimensions" in {

    def testArray(params: SpatialArrayParam, fgIdx: Int): Unit = {

      test(new SpatialArray(params)).withAnnotations(Seq(WriteVcdAnnotation)) { c =>
        val rand = new Random()

        val Mu = params.arrayDim(fgIdx)(0)
        val Ku = params.arrayDim(fgIdx)(1) 
        val Nu = params.arrayDim(fgIdx)(2)

        // Generate random values for 'a' and 'b'
        val aValues = Array.fill(Mu * Ku)(rand.nextInt(256)) 
        val bValues = Array.fill(Ku * Nu)(rand.nextInt(256)) 

        // Construct 'a' and 'b' from the random values
        val a = aValues.zipWithIndex.map { case (v, i) => BigInt(v) << (i * 8) }.sum
        val b = bValues.zipWithIndex.map { case (v, i) => BigInt(v) << (i * 8) }.sum

        // Compute expected results for (Mu, Ku) x (Ku, Nu) matrix multiplication
        val expectedResult1 = Array.tabulate(Mu, Nu) { (i, j) =>
          (0 until Ku).map(k => aValues(i * Ku + k) * bValues(k + j * Ku)).sum
        }

        // Poke the random values
        c.io.data.in_a.bits.poke(a.U)
        c.io.data.in_b.bits.poke(b.U)
        c.io.data.in_c.bits.poke(0.U)

        // Enable valid signals
        c.io.data.in_a.valid.poke(true.B)
        c.io.data.in_b.valid.poke(true.B)
        c.io.data.in_c.valid.poke(true.B)
        c.io.ctrl.spatialArrayCfg.poke(fgIdx.U)
        c.clock.step(1)

        var out_d = c.io.data.out_d.bits.peek().litValue
        var extractedOutputs = (0 until Mu * Nu).map(i => (out_d >> (i * 32)) & 0xFFFFFFFFL)

        println(s"Checking res_cfg${fgIdx + 1}...")
        var expected = expectedResult1.flatten
        for (i <- expected.indices) {
          val actual = extractedOutputs(i)
          // println(s"  Output[$i]: $actual (Expected: ${expected(i)})")
          assert(actual == expected(i), s"Mismatch at index $i: got $actual, expected ${expected(i)}")
        }

      }
    }
    var params = SpatialArrayParam(
      opType = OpType.SIntSIntOp,
      macNum = 1024,
      inputAElemWidth = 8,
      inputBElemWidth = 8,
      inputCElemWidth = 8,
      mulElemWidth = 16,
      outElemWidth = 32,
      inputAWidth = 1024,
      inputBWidth = 8192,
      inputCWidth = 4096,
      outputDWidth = 4096,
      arrayDim = Seq(Seq(16, 8, 8), Seq(1, 32, 32))
    )

    // Test for each configuration
    for (fgIdx <- 0 until params.arrayDim.length) {
      testArray(params, fgIdx)
    }

    // Test for a different configuration
    params = SpatialArrayParam(
          opType = OpType.SIntSIntOp,
          macNum = 8,
          inputAElemWidth = 8,
          inputBElemWidth = 8,
          inputCElemWidth = 8,
          mulElemWidth = 16,
          outElemWidth = 32,
          inputAWidth = 64,
          inputBWidth = 64,
          inputCWidth = 256,
          outputDWidth = 256,
          arrayDim = Seq(Seq(2, 2, 2), Seq(2, 1, 4), Seq(4, 1, 2), Seq(4, 2, 1), Seq(1, 8, 1))
        )

    for (fgIdx <- 0 until params.arrayDim.length) {
      testArray(params, fgIdx)
    }

  }
}
