package snax.sparse_interconnect

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** Checks that the user field of a request reaches the bank side with the request that wins. */
class SparseInterconnectUserTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SparseInterconnect"

  // 4 inputs, 4 banks of 16 words of 64 bits: [row (4) | bank (2) | byte offset (3)]
  val NumInp    = 4
  val NumOut    = 4
  val userWidth = 5
  val users     = Seq(0x15, 0x0a, 0x1f, 0x01)

  def dut =
    new SparseInterconnect(
      NumInp        = NumInp,
      NumOut        = NumOut,
      memAddrWidth  = 4,
      tcdmAddrWidth = 9,
      dataWidth     = 64,
      strbWidth     = 8,
      priorityWidth = 1,
      userWidth     = userWidth,
      sparse_config = new SparseConfig(Seq(new SparsePortDefinition(NumInp, 1)))
    )

  def addr(row: Int, bank: Int): UInt = ((row << 5) | (bank << 3)).U

  def drive(c: SparseInterconnect, i: Int, valid: Boolean, bank: Int): Unit = {
    c.io.tcdmReqs(i).valid.poke(valid.B)
    c.io.tcdmReqs(i).bits.addr.poke(addr(i, bank))
    c.io.tcdmReqs(i).bits.write.poke(true.B)
    c.io.tcdmReqs(i).bits.amo.poke(AmoOp.None)
    c.io.tcdmReqs(i).bits.data.poke((0x100 + i).U)
    c.io.tcdmReqs(i).bits.strb.poke(0xff.U)
    c.io.tcdmReqs(i).bits.priority.poke(0.U)
    c.io.tcdmReqs(i).bits.user.poke(users(i).U)
  }

  def init(c: SparseInterconnect): Unit = {
    c.io.memReqs.foreach(_.ready.poke(true.B))
    c.io.memRsps.foreach { r => r.valid.poke(false.B); r.bits.data.poke(0.U) }
    c.io.tcdmRsps.foreach(_.ready.poke(true.B))
  }

  it should "send the winner's user to the bank when requesters contend for one bank" in {
    test(dut) { c =>
      init(c)
      val bank    = 1
      var pending = (0 until NumInp).toSet
      var winners = Seq[Int]()
      // All four ask for bank 1; each winner leaves, so every user value has to reach the bank once
      while (pending.nonEmpty) {
        for (i <- 0 until NumInp) drive(c, i, pending.contains(i), bank)
        val granted = (0 until NumInp).filter(i => c.io.tcdmReqs(i).ready.peek().litToBoolean)
        assert(granted.size == 1, s"expected one grant, got $granted")
        val w       = granted.head
        assert(pending.contains(w), s"input $w granted without a request")
        c.io.memReqs(bank).valid.expect(true.B)
        c.io.memReqs(bank).bits.user.expect(users(w).U)
        c.io.memReqs(bank).bits.addr.expect(w.U)
        c.io.memReqs(bank).bits.data.expect((0x100 + w).U)
        for (b <- 0 until NumOut if b != bank) c.io.memReqs(b).valid.expect(false.B)
        winners = winners :+ w
        pending -= w
        c.clock.step()
      }
      assert(winners.sorted == (0 until NumInp), s"winners $winners")
    }
  }

  it should "send each requester's user to its own bank when the banks differ" in {
    test(dut) { c =>
      init(c)
      // Input i asks for bank (i + 1) % 4, all in the same cycle
      for (i <- 0 until NumInp) drive(c, i, valid = true, bank = (i + 1) % NumOut)
      for (i <- 0 until NumInp) {
        val bank = (i + 1) % NumOut
        c.io.tcdmReqs(i).ready.expect(true.B)
        c.io.memReqs(bank).valid.expect(true.B)
        c.io.memReqs(bank).bits.user.expect(users(i).U)
        c.io.memReqs(bank).bits.addr.expect(i.U)
      }
    }
  }
}
