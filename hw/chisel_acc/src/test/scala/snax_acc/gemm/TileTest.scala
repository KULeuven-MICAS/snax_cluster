package snax_acc.gemm


import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class TileUnitTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Tile"

  it should "compute dot product + optional add correctly" in {

    test(new Tile(TestParameters.testConfig)).withAnnotations(Seq(WriteVcdAnnotation)) { c =>
      val a = Seq(1, 2, 3, 4, 5)
      val b = Seq(10, 20, 30, 40, 50)
      val c_in = 1000

      // Set inputs
      for (i <- 0 until TestParameters.testConfig.tileSize) {
        c.io.data_a_i(i).poke(a(i).U)
        c.io.data_b_i(i).poke(b(i).U)
      }

      c.io.data_c_i.poke(c_in.U)

      // Set subtraction offsets to zero (no subtraction)
      c.io.ctrl.subtraction_a_i.poke(0)
      c.io.ctrl.subtraction_b_i.poke(0)

      // Trigger dot product with c addition
      c.io.ctrl.dotprod_a_b.poke(true.B)
      c.io.ctrl.add_c_i.poke(true.B)
      c.io.ctrl.d_ready_i.poke(true.B)

      c.clock.step()

      // Deassert fire after one cycle
      c.io.ctrl.dotprod_a_b.poke(false.B)
      c.io.ctrl.add_c_i.poke(false.B)

      // Output valid the next cycle
      c.clock.step()

      val expectedDot = a.zip(b).map { case (x, y) => x * y }.sum
      val expectedResult = c_in + expectedDot

      c.io.data_d_o.expect(expectedResult.S)

      println(s"\n[Tile Test] Expected result: $expectedResult, Got: " + c.io.data_d_o.peek().litValue)
    }
  }
}
