package snax_acc.utils

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** ParallelToSerial must emit exactly `ratio` chunks per parallel word.
  *
  * Every shipped gemmX configuration has meshRow*meshCol*32 == serialC32D32Width, i.e.
  * ratio 1, which takes the pass-through branch and never exercises the counter. A
  * 16x16 mesh makes the ratio 4 for the first time, and the design then hangs: the
  * streamer waits for a fourth beat that never comes.
  */
class ParallelToSerialRatioTest extends AnyFlatSpec with ChiselScalatestTester {

  private def chunksEmitted(parallelWidth: Int, serialWidth: Int): Int = {
    var seen = 0
    test(new ParallelToSerial(ParallelAndSerialConverterParams(parallelWidth, serialWidth))) { dut =>
      dut.io.is_busy_cstate.poke(true.B)
      dut.io.counter_value_reset.poke(true.B)
      dut.clock.step(1)
      dut.io.counter_value_reset.poke(false.B)

      // Offer exactly ONE parallel word and drain until it stops producing.
      dut.io.in.bits.poke(BigInt(1))
      dut.io.in.valid.poke(true.B)
      dut.io.out.ready.poke(true.B)
      var guard = 0
      var accepted = false
      while (guard < 64) {
        if (dut.io.in.ready.peekBoolean() && !accepted) accepted = true
        else if (accepted) dut.io.in.valid.poke(false.B)
        if (dut.io.out.valid.peekBoolean()) seen += 1
        dut.clock.step(1)
        guard += 1
        if (accepted && !dut.io.out.valid.peekBoolean() && seen > 0) guard = 64
      }
    }
    seen
  }

  "ParallelToSerial" should "emit 4 chunks when the ratio is 4" in {
    val n = chunksEmitted(8192, 2048)
    println(s"[ratio 4] chunks emitted = $n (expected 4)")
    assert(n == 4, s"emitted $n chunks, expected 4 -- the ceil port truncated")
  }

  "ParallelToSerial" should "emit 2 chunks when the ratio is 2" in {
    val n = chunksEmitted(4096, 2048)
    println(s"[ratio 2] chunks emitted = $n (expected 2)")
    assert(n == 2, s"emitted $n chunks, expected 2 -- the ceil port truncated")
  }
}
