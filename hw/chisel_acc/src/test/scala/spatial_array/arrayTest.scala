package snax_acc.spatial_array

import scala.util.Random

import chisel3._

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SpatialArrayTest extends AnyFlatSpec with ChiselScalatestTester {

  "SpatialArray" should "correctly compute output with different array dimensions" in {

    def testArray(params: SpatialArrayParam): Unit = {

      test(new SpatialArray(params)).withAnnotations(Seq(WriteVcdAnnotation)) { c =>
          val rand = new Random()

          (0 until params.opType.length).map{idx =>

            (0 until params.arrayDim(idx).length).map{fgIdx =>
            // Get the parameters for the current configuration
            val inputAElemWidth = params.inputAElemWidth(idx)
            val inputBElemWidth = params.inputBElemWidth(idx)
            val outElemWidth = params.outElemWidth(idx)

            val Mu = params.arrayDim(idx)(fgIdx)(0)
            val Ku = params.arrayDim(idx)(fgIdx)(1)
            val Nu = params.arrayDim(idx)(fgIdx)(2)

            // Generate random values for 'a' and 'b'
            val aValues = Array.fill(Mu * Ku)(rand.nextInt(math.pow(2, inputAElemWidth).toInt))
            val bValues = Array.fill(Ku * Nu)(rand.nextInt(math.pow(2, inputBElemWidth).toInt))

            // Print the generated values in 0x format
            // println(s"Generated aValues: ${aValues.map(v => f"0x$v%X").mkString(", ")}")
            // println(s"Generated bValues: ${bValues.map(v => f"0x$v%X").mkString(", ")}")

            // Construct 'a' and 'b' from the random values
            val a = aValues.zipWithIndex.map { case (v, i) => BigInt(v) << (i * inputAElemWidth) }.sum
            val b = bValues.zipWithIndex.map { case (v, i) => BigInt(v) << (i * inputBElemWidth) }.sum

            // Function to convert UInt to SInt manually
            def toSInt(value: Int, bitWidth: Int, ifTrans: Boolean): Int = {
              if (!ifTrans) return value // No conversion needed
              val signBit = 1 << (bitWidth - 1) // 2^(N-1)
              if (value >= signBit) value - (1 << bitWidth) // Convert to signed
              else value // Keep as is
            }

            // Compute dot product treating aValues and bValues as SInt
            val expectedResult1 = Array.tabulate(Mu, Nu) { (i, j) =>
              (0 until Ku).map { k =>
                val aSInt = toSInt(aValues(i * Ku + k), inputAElemWidth, params.opType(idx) == OpType.SIntSIntOp) // Convert UInt to SInt
                val bSInt = toSInt(bValues(k + j * Ku), inputBElemWidth, params.opType(idx) == OpType.SIntSIntOp) // Convert UInt to SInt
                aSInt * bSInt
              }.sum
            }

            // Poke the random values
            c.io.data.in_a.bits.poke(a.U)
            c.io.data.in_b.bits.poke(b.U)
            c.io.data.in_c.bits.poke(0.U)

            // Enable valid signals
            c.io.data.in_a.valid.poke(true.B)
            c.io.data.in_b.valid.poke(true.B)
            c.io.data.in_c.valid.poke(true.B)
            c.io.data.in_substraction.valid.poke(false.B)
            c.io.data.out_d.ready.poke(true.B)

            c.io.ctrl.spatialArrayCfg.poke(fgIdx.U)
            c.io.ctrl.dataTypeCfg.poke(idx.U)
            c.io.ctrl.accAddExtIn.poke(false.B)
            c.io.ctrl.accClear.poke(false.B)

            c.clock.step(1)

            // Check the output
            c.io.data.out_d.valid.expect(true.B)
            val out_d = c.io.data.out_d.bits.peek().litValue
            val extractedOutputs = (0 until (Mu * Nu)).map { i =>
              ((out_d >> (i * outElemWidth)) & (math.pow(2, outElemWidth).toLong - 1)).toInt
            }

          println(s"Checking opType${idx + 1} res_cfg${fgIdx + 1}...")
          var expected = expectedResult1.flatten
          for (i <- expected.indices) {
            val actual = extractedOutputs(i)
            // println(s"  Output[$i]: $actual (Expected: ${expected(i)})")
            assert(actual == expected(i), f"Mismatch at index $i: got 0x$actual%X, expected 0x${expected(i)}%X")
          }

            c.io.ctrl.accClear.poke(true.B)
            c.clock.step(1)
            c.io.ctrl.accClear.poke(false.B)

        }
      }
    }
  }

  var params = SpatialArrayParam(
    opType = Seq(OpType.UIntUIntOp),
    macNum = Seq(1024),
    inputAElemWidth = Seq(8),
    inputBElemWidth = Seq(8),
    inputCElemWidth = Seq(8),
    mulElemWidth = Seq(16),
    outElemWidth = Seq(32),
    inputAWidth = 1024,
    inputBWidth = 8192,
    arrayInputCWidth = 4096,
    arrayOutputDWidth = 4096,
    arrayDim = Seq(Seq(Seq(16, 8, 8), Seq(1, 32, 32)))
  )

  // Test for each configuration
  testArray(params)

  // Test for a different configuration
  params = SpatialArrayParam(
    opType = Seq(OpType.SIntSIntOp, OpType.UIntUIntOp),
    // opType = Seq(OpType.UIntUIntOp, OpType.UIntUIntOp),
    macNum = Seq(8, 16),
    inputAElemWidth = Seq(8, 4),
    inputBElemWidth = Seq(8, 4),
    inputCElemWidth = Seq(8, 4),
    mulElemWidth = Seq(16, 8),
    outElemWidth = Seq(32, 16),
    inputAWidth = 64,
    inputBWidth = 64,
    arrayInputCWidth = 256,
    arrayOutputDWidth = 256,
    arrayDim = Seq(Seq(Seq(2, 2, 2), Seq(2, 1, 4)),
      Seq(Seq(2, 4, 2), Seq(2, 1, 8))
    )
  )

  testArray(params)

  }
}
