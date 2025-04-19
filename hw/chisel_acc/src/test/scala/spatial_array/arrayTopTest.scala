package snax_acc.spatial_array

import scala.util.Random

import chisel3._

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import snax_acc.utils.DecoupledCut._

class ArrayTopHarness(params: SpatialArrayParam) extends Module with RequireAsyncReset {
  val dut = Module(new ArrayTop(params))
  val io  = IO(chiselTypeOf(dut.io))

  io.ctrl <> dut.io.ctrl
  io.data <> dut.io.data
  io.data.in_a -||> dut.io.data.in_a
  io.data.in_b -||> dut.io.data.in_b
  io.data.in_c -||> dut.io.data.in_c

  io.busy_o              := dut.io.busy_o
  io.performance_counter := dut.io.performance_counter
}

class ArrayTopTest extends AnyFlatSpec with ChiselScalatestTester with GeMMTestUtils {

  behavior of "ArrayTop"

  it should "correctly process configuration and data" in {

    def testArrayTop(params: SpatialArrayParam): Unit = {
      test(new ArrayTopHarness(params)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        val rand = new Random()
        (0 until params.opType.length).map { dataTypeIdx =>

          (0 until params.arrayDim(dataTypeIdx).length).map { arrayShapeIdx =>
            // Get the parameters for the current configuration
            val inputAElemWidth  = params.inputAElemWidth(dataTypeIdx)
            val inputBElemWidth  = params.inputBElemWidth(dataTypeIdx)
            val inputCElemWidth  = params.inputCElemWidth(dataTypeIdx)
            val outputDElemWidth = params.outputDElemWidth(dataTypeIdx)

            val Mu = params.arrayDim(dataTypeIdx)(arrayShapeIdx)(0)
            val Ku = params.arrayDim(dataTypeIdx)(arrayShapeIdx)(1)
            val Nu = params.arrayDim(dataTypeIdx)(arrayShapeIdx)(2)

            val sizeRange = 1
            val M         = rand.nextInt(sizeRange) + 1
            val N         = rand.nextInt(sizeRange) + 1
            val K         = rand.nextInt(sizeRange) + 1
            // val K = 2

            // Generate random values for 'a', 'b' and 'c'

            val aValues = Array.fill(Mu * Ku * M * K)(rand.nextInt(1))
            val bValues = Array.fill(Ku * Nu * N * K)(rand.nextInt(1))
            val cValues = Array.fill(Mu * Nu * M * N)(rand.nextInt(1))
            // println(s"Generated aValues: ${aValues.mkString(", ")}")
            // println(s"Generated bValues: ${bValues.mkString(", ")}")
            // println(s"Generated cValues: ${cValues.mkString(", ")}")

            // Print the generated values in 0x format
            println(s"Generated aValues: ${aValues.map(v => f"0x$v%X").mkString(", ")}")
            println(s"Generated bValues: ${bValues.map(v => f"0x$v%X").mkString(", ")}")
            println(s"Generated cValues: ${cValues.map(v => f"0x$v%X").mkString(", ")}")

            val expectedResult = Array.tabulate(M, N) { (m2, n2) =>
              {
                // multiply a and b and sum up along K dimension
                val mul_res = (0 until K).map { k2 =>
                  {
                    Array.tabulate(Mu, Nu) { (m1, n1) =>
                      (0 until Ku).map { k1 =>
                        val aSInt = toSInt(
                          aValues(m2 * K * Mu * Ku + k2 * Mu * Ku + k1 + m1 * Ku),
                          inputAElemWidth,
                          params.opType(dataTypeIdx) == OpType.SIntSIntOp
                        ) // Convert UInt to SInt
                        val bSInt = toSInt(
                          bValues(n2 * K * Nu * Ku + k2 * Nu * Ku + k1 + n1 * Ku),
                          inputBElemWidth,
                          params.opType(dataTypeIdx) == OpType.SIntSIntOp
                        ) // Convert UInt to SInt
                        aSInt * bSInt
                      }.sum
                    }
                  }
                }.reduce { (a, b) =>
                  a.zip(b).map { case (rowA, rowB) =>
                    rowA.zip(rowB).map { case (x, y) => x + y }
                  }
                }

                val acc_res = Array.tabulate(Mu, Nu) { (m, n) =>
                  val mul  = mul_res(m)(n)
                  val cIdx = m2 * N * Mu * Nu + n2 * Mu * Nu + m * Nu + n
                  val cVal = toSInt(cValues(cIdx), inputCElemWidth, params.opType(dataTypeIdx) == OpType.SIntSIntOp)
                  mul + cVal
                }

                acc_res
              }
            }

            println(s"Checking dataTypeIdx${dataTypeIdx + 1} arrayShapeIdx_cfg${arrayShapeIdx + 1}...")
            print(s"M = $M, N = $N, K = $K\n")
            expectedResult.zipWithIndex.foreach { case (rowBlocks, m) =>
              rowBlocks.zipWithIndex.foreach { case (block, n) =>
                println(s"Block ($m, $n):\n" + block.map(_.mkString(" ")).mkString("\n") + "\n")
              }
            }

            // Set up configuration
            dut.clock.step(5)
            dut.io.ctrl.bits.fsmCfg.K_i.poke(K.U)
            dut.io.ctrl.bits.fsmCfg.N_i.poke(N.U)
            dut.io.ctrl.bits.fsmCfg.M_i.poke(M.U)
            dut.io.ctrl.bits.fsmCfg.subtraction_constant_i.poke(0.U)
            dut.io.ctrl.bits.arrayCfg.arrayShapeCfg.poke(arrayShapeIdx.U)
            dut.io.ctrl.bits.arrayCfg.dataTypeCfg.poke(dataTypeIdx.U)
            dut.io.ctrl.valid.poke(true.B)
            WaitOrTimeout(dut.io.ctrl.ready, dut.clock)
            dut.clock.step(1)
            dut.io.ctrl.valid.poke(false.B)

            var concurrent_threads = new chiseltest.internal.TesterThreadList(Seq())

            // A Input injector
            concurrent_threads = concurrent_threads.fork {

              for (temporalIndexInput <- 0 until M * K * N) {
                val (indexA, indexB) = temporalToSpatialIndicesAB(temporalIndexInput, K = K, N = N)
                // A
                val aValues_cur      = aValues
                  .slice(indexA * Mu * Ku, indexA * Mu * Ku + Mu * Ku)
                  .zipWithIndex
                  .map { case (v, i) => BigInt(v) << (i * inputAElemWidth) }
                  .sum

                // dut.clock.step(Random.between(1, 5))
                dut.io.data.in_a.bits.poke(aValues_cur.U)
                dut.io.data.in_a.valid.poke(true.B)
                WaitOrTimeout(dut.io.data.in_a.ready, dut.clock)
                assert(dut.io.data.in_a.ready.peekBoolean())

                dut.clock.step(1) // Valid needs to be asserted for 1 cycle

                dut.io.data.in_a.valid.poke(false.B)
              }
            }

            // B Input injector
            concurrent_threads = concurrent_threads.fork {

              for (temporalIndexInput <- 0 until M * K * N) {
                val (indexA, indexB) = temporalToSpatialIndicesAB(temporalIndexInput, K = K, N = N)
                // B
                val bValues_cur      = bValues
                  .slice(indexB * Nu * Ku, indexB * Nu * Ku + Nu * Ku)
                  .zipWithIndex
                  .map { case (v, i) => BigInt(v) << (i * inputBElemWidth) }
                  .sum
                // dut.clock.step(Random.between(1, 5))
                dut.io.data.in_b.bits.poke(bValues_cur.U)
                dut.io.data.in_b.valid.poke(true.B)
                WaitOrTimeout(dut.io.data.in_b.ready, dut.clock)
                assert(dut.io.data.in_b.ready.peekBoolean())

                dut.clock.step(1) // Valid needs to be asserted for 1 cycle

                dut.io.data.in_b.valid.poke(false.B)

              }
            }

            // C injector
            concurrent_threads = concurrent_threads.fork {

              for (temporalIndex <- 0 until M * N) {
                val cValues_cur = cValues
                  .slice(temporalIndex * Mu * Nu, temporalIndex * Mu * Nu + Mu * Nu)
                  .zipWithIndex
                  .map { case (v, i) => BigInt(v) << (i * inputCElemWidth) }
                  .sum

                // dut.clock.step(Random.between(1, 5))
                dut.io.data.in_c.bits.poke(cValues_cur.U)
                dut.io.data.in_c.valid.poke(true.B)
                WaitOrTimeout(dut.io.data.in_c.ready, dut.clock)

                dut.clock.step(1) // Valid needs to be asserted for 1 cycle

                dut.io.data.in_c.valid.poke(false.B)
              }
            }

            // Output checker
            concurrent_threads = concurrent_threads.fork {
              for (outputTemporalIndex <- 0 until M * N) {
                WaitOrTimeout(dut.io.data.out_d.valid, dut.clock)

                // Check the output
                val expected = expectedResult.flatten.flatten.flatten
                  .slice(outputTemporalIndex * Mu * Nu, (outputTemporalIndex + 1) * Mu * Nu)
                val out_d    = dut.io.data.out_d.bits.peek().litValue
                val output   = (0 until (Mu * Nu)).map { i =>
                  ((out_d >> (i * outputDElemWidth)) & (math.pow(2, outputDElemWidth).toLong - 1)).toInt
                }
                // println(s"Expected: ${expected.mkString(", ")}")
                // println(s"Output: ${output.mkString(", ")}")
                for (i <- output.indices) {
                  // println(s"output: ${output(i)} (expected: ${expected(i)})")
                  assert(
                    output(i) == expected(i),
                    f"Mismatch at index $i: got 0x${output(i)}%X, expected 0x${expected(i)}%X"
                  )
                }

                // dut.clock.step(Random.between(1, 5))
                dut.io.data.out_d.ready.poke(true.B)
                dut.clock.step(1)
                dut.io.data.out_d.ready.poke(false.B)
              }
            }

            // Wait for all threads to finish
            concurrent_threads.join()

          }
        }
      }
    }

    // Test with custom parameters
    // params = SpatialArrayParam(
    //   opType                 = Seq(OpType.SIntSIntOp, OpType.UIntUIntOp),
    //   // opType                 = Seq(OpType.UIntUIntOp, OpType.UIntUIntOp),
    //   macNum                 = Seq(512, 1024),
    //   inputAElemWidth        = Seq(8, 4),
    //   inputBElemWidth        = Seq(8, 4),
    //   inputCElemWidth        = Seq(32, 16),
    //   mulElemWidth           = Seq(16, 8),
    //   outputDElemWidth       = Seq(32, 16),
    //   arrayInputAWidth       = 512,
    //   arrayInputBWidth       = 4096,
    //   arrayInputCWidth       = 2048,
    //   arrayOutputDWidth      = 2048,
    //   arrayDim               = Seq(Seq(Seq(8, 8, 8), Seq(1, 32, 16)), Seq(Seq(8, 16, 8), Seq(1, 64, 16))),
    //   serialInputCDataWidth  = 2048,
    //   serialOutputDDataWidth = 2048
    // )

    val params = SpatialArrayParam(
      opType = Seq(OpType.SIntSIntOp, OpType.UIntUIntOp),
      macNum                 = Seq(8, 16),
      inputAElemWidth        = Seq(8, 4),
      inputBElemWidth        = Seq(8, 4),
      inputCElemWidth        = Seq(32, 16),
      mulElemWidth           = Seq(16, 8),
      outputDElemWidth           = Seq(32, 16),
      arrayInputAWidth            = 64,
      arrayInputBWidth            = 64,
      arrayInputCWidth       = 256,
      arrayOutputDWidth      = 256,
      arrayDim               = Seq(Seq(Seq(2, 2, 2), Seq(2, 1, 4)), Seq(Seq(2, 4, 2), Seq(2, 1, 8))),
      serialInputCDataWidth  = 256,
      serialOutputDDataWidth = 256
    )

    val repeat_times = 2
    (0 until repeat_times).foreach { _ =>
      testArrayTop(params)
    }

  }
}
