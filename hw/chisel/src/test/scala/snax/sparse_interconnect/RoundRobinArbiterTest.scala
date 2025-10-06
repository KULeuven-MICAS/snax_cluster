package snax.sparse_interconnect

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class RoundRobinArbiterTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "RoundRobinArbiter"

  it should "perform basic round-robin arbitration" in {
    test(new RoundRobinArbiter(NumInp = 4)) { dut =>
      // Initialize inputs with two valid requests
      dut.io.requests.foreach(_.poke(false.B))
      dut.io.selection.ready.poke(true.B)
      dut.io.requests(0).poke(true.B)
      dut.io.requests(1).poke(true.B)

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

  it should "lock onto a request until it is ready" in {
    test(new RoundRobinArbiter(NumInp = 4)) { dut =>
      // Initialize inputs with two valid requests
      dut.io.requests.foreach(_.poke(false.B))
      dut.io.selection.ready.poke(false.B) // not ready initially
      dut.io.requests(0).poke(true.B)
      dut.io.requests(1).poke(true.B)

      // Test case: ensure first request is selected
      dut.io.selection.bits.expect(0.U)
      dut.io.selection.valid.expect(true.B)
      dut.clock.step()

      // Request is not ready, lock should remain to same request
      dut.io.selection.bits.expect(0.U)
      dut.io.selection.valid.expect(true.B)
      // In this cycle, we accept the request
      dut.io.selection.ready.poke(true.B)
      dut.clock.step()

      // Second request should now be selected
      dut.io.selection.bits.expect(1.U)
    }
  }

  it should "output valid = false when no requests are valid" in {
    test(new RoundRobinArbiter(NumInp = 4)) { dut =>
      // Initialize inputs
      dut.io.requests.foreach(_.poke(false.B))
      dut.io.selection.ready.poke(true.B)

      // Test case: No valid requests
      dut.clock.step()
      dut.io.selection.valid.expect(false.B)
    }
  }
}
