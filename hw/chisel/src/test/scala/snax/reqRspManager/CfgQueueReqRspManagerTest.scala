package snax.reqRspManager

import chisel3._

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The configuration staging queue.
  *
  * Two properties matter and neither is visible at depth 1. A start write must SNAPSHOT the
  * register file, so a configuration already queued is unaffected by the core reprogramming
  * the registers behind it; and the core must retire that write immediately rather than
  * waiting for the accelerator, right up until the queue is genuinely full.
  */
class CfgQueueReqRspManagerTest
    extends AnyFlatSpec
    with ChiselScalatestTester
    with Matchers
    with HasRegRspManagerTestUtils {

  val nRW  = 10
  val last = nRW - 1

  /** A write that fails loudly instead of hanging when the manager refuses to grant.
    * Returns how many cycles it waited.
    */
  def writeRegBounded[T <: ReqRspManager](dut: T, addr: Int, data: Int, bound: Int = 8): Int = {
    dut.io.reqRspIO.req.bits.strb.poke(((BigInt(1) << (dut.wordsPerBeat * 4)) - 1).U)
    dut.io.reqRspIO.req.bits.write.poke(1.B)
    dut.io.reqRspIO.req.bits.data.poke(data.U)
    dut.io.reqRspIO.req.bits.addr.poke(addr.U)
    dut.io.reqRspIO.req.valid.poke(1.B)
    var waited = 0
    while (!dut.io.reqRspIO.req.ready.peekBoolean()) {
      assert(waited < bound, s"manager never granted the write to $addr within $bound cycles")
      dut.clock.step(1)
      waited += 1
    }
    dut.clock.step(1)
    dut.io.reqRspIO.req.valid.poke(0.B)
    waited
  }

  "A depth-2 config queue" should "stage two tasks behind a busy accelerator and snapshot each" in {
    test(new ReqRspManager(nRW, 2, 32, 32, 32, 2))
      .withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        // The accelerator accepts nothing for the whole staging phase.
        dut.io.readWriteRegIO.ready.poke(0.B)
        dut.clock.step(2)

        // Task A, then task B, reprogramming the same register in between.
        writeRegBounded(dut, 1, 0xAA)
        assert(0 == writeRegBounded(dut, last, 1), "the first start write must not stall")
        writeRegBounded(dut, 1, 0xBB)
        assert(0 == writeRegBounded(dut, last, 1), "the second start write must not stall")
        dut.clock.step(2)

        // A comes out first, carrying the value the register held when A was started --
        // not 0xBB, which the core wrote afterwards.
        dut.io.readWriteRegIO.valid.expect(1.B)
        dut.io.readWriteRegIO.bits(1).expect(0xAA.U)
        dut.io.readWriteRegIO.ready.poke(1.B)
        dut.clock.step(1)
        dut.io.readWriteRegIO.ready.poke(0.B)
        dut.clock.step(1)

        // then B, with its own snapshot
        dut.io.readWriteRegIO.valid.expect(1.B)
        dut.io.readWriteRegIO.bits(1).expect(0xBB.U)
        dut.io.readWriteRegIO.ready.poke(1.B)
        dut.clock.step(1)
        dut.io.readWriteRegIO.ready.poke(0.B)
        dut.clock.step(1)

        dut.io.readWriteRegIO.valid.expect(0.B)
      }
  }

  it should "back-pressure the core once it is full" in {
    test(new ReqRspManager(nRW, 2, 32, 32, 32, 2)) { dut =>
      dut.io.readWriteRegIO.ready.poke(0.B)
      dut.clock.step(2)
      writeRegBounded(dut, last, 1)
      writeRegBounded(dut, last, 1)

      // A third start has nowhere to go: the manager must refuse it.
      dut.io.reqRspIO.req.bits.strb.poke(((BigInt(1) << (dut.wordsPerBeat * 4)) - 1).U)
      dut.io.reqRspIO.req.bits.write.poke(1.B)
      dut.io.reqRspIO.req.bits.data.poke(1.U)
      dut.io.reqRspIO.req.bits.addr.poke(last.U)
      dut.io.reqRspIO.req.valid.poke(1.B)
      dut.clock.step(1)
      dut.io.reqRspIO.req.ready.expect(0.B)

      // ...and grant it as soon as the accelerator drains one.
      dut.io.readWriteRegIO.ready.poke(1.B)
      dut.clock.step(1)
      dut.io.reqRspIO.req.ready.expect(1.B)
    }
  }

  "A depth-1 manager" should "still stall the core while the accelerator is busy" in {
    test(new ReqRspManager(nRW, 2, 32, 32, 32, 1)) { dut =>
      dut.io.readWriteRegIO.ready.poke(0.B)
      dut.clock.step(2)
      dut.io.reqRspIO.req.bits.strb.poke(((BigInt(1) << (dut.wordsPerBeat * 4)) - 1).U)
      dut.io.reqRspIO.req.bits.write.poke(1.B)
      dut.io.reqRspIO.req.bits.data.poke(1.U)
      dut.io.reqRspIO.req.bits.addr.poke(last.U)
      dut.io.reqRspIO.req.valid.poke(1.B)
      dut.clock.step(1)
      // The historical behaviour, unchanged: with no queue to absorb it the manager
      // refuses the write outright. valid stays low with it, because it is gated on the
      // request firing -- the configuration is only ever offered once the write lands.
      dut.io.reqRspIO.req.ready.expect(0.B)
      dut.io.readWriteRegIO.valid.expect(0.B)

      // and it is granted the moment the accelerator can take it
      dut.io.readWriteRegIO.ready.poke(1.B)
      dut.clock.step(1)
      dut.io.reqRspIO.req.ready.expect(1.B)
      dut.io.readWriteRegIO.valid.expect(1.B)
    }
  }
}
