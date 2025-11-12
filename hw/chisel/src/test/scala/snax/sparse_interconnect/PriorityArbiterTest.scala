package snax.sparse_interconnect

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class PriorityRoundRobinArbiterTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "PriorityRoundRobinArbiter"

  it should "perform basic round-robin arbitration without priorities" in {
    test(new PriorityRoundRobinArbiter(NumInp = 4, priorityWidth = 2)) { dut =>
      // Initialize inputs with two valid requests with same priority
      dut.io.requests.foreach(_.valid.poke(false.B))
      dut.io.selection.ready.poke(true.B)
      dut.io.requests(0).valid.poke(true.B)
      dut.io.requests(0).priority.poke(1.U)
      dut.io.requests(1).valid.poke(true.B)
      dut.io.requests(1).priority.poke(1.U)

      // First valid request is selected
      dut.io.selection.bits.expect(0.U)
      dut.io.selection.valid.expect(true.B)
      dut.clock.step()

      // Second valid request is selected in round-robin fashion
      dut.io.selection.bits.expect(1.U)
      dut.io.selection.valid.expect(true.B)
      dut.clock.step()

      // Select first request again (wrap-around)
      dut.io.selection.bits.expect(0.U)
      dut.io.selection.valid.expect(true.B)
      dut.clock.step()
    }
  }

  it should "perform basic round-robin arbitration with priorities" in {
    test(new PriorityRoundRobinArbiter(NumInp = 4, priorityWidth = 2)) { dut =>
      // Initialize inputs with two valid requests with same priority
      dut.io.requests.foreach(_.valid.poke(false.B))
      dut.io.selection.ready.poke(true.B)
      dut.io.requests(0).valid.poke(true.B)
      dut.io.requests(0).priority.poke(0.U)
      dut.io.requests(1).valid.poke(true.B)
      dut.io.requests(1).priority.poke(1.U)
      dut.io.requests(2).valid.poke(true.B)
      dut.io.requests(2).priority.poke(2.U)

      // Last request is selected first (highest priority)
      dut.io.selection.bits.expect(2.U)
      dut.io.selection.valid.expect(true.B)
      dut.clock.step()
      dut.io.requests(2).valid.poke(false.B)

      // Second request has next priority
      dut.io.selection.bits.expect(1.U)
      dut.io.selection.valid.expect(true.B)
      dut.clock.step()

      // Second request is still selected when first request has lower priority
      dut.io.selection.bits.expect(1.U)
      dut.io.selection.valid.expect(true.B)
      dut.clock.step()
      dut.io.requests(1).valid.poke(false.B)

      // Finally, first request (lowest priority) is selected
      dut.io.selection.bits.expect(0.U)
      dut.io.selection.valid.expect(true.B)
      dut.clock.step()
    }
  }

}
