package snax.DataPathJunction

import chisel3._
import chiseltest._

import snax.utils.DecoupledCut._

/** Test harness for a `DataPathJunction`: wraps the junction and cuts both operand paths and the output, so the
  * testbench exercises the join across register boundaries rather than through a combinational shortcut.
  */
class DataPathJunctionHarness(junction: HasDataPathJunction) extends Module with RequireAsyncReset {
  val dut = junction.instantiate("dma_junction_dut")
  val io  = IO(chiselTypeOf(dut.io))

  io.busy_o       := dut.io.busy_o
  io.starved_o    := dut.io.starved_o
  dut.io.csr_i    := io.csr_i
  dut.io.enable_i := io.enable_i
  dut.io.start_i  := io.start_i

  io.a_i -||> dut.io.a_i
  io.b_i -||> dut.io.b_i
  dut.io.out_o -||> io.out_o
}

/** Shared driver utilities for the junction testers. */
object JunctionTestUtils {
  def f32(f: Double): BigInt = BigInt(java.lang.Float.floatToIntBits(f.toFloat).toLong & 0xffffffffL)
  def dec(b: BigInt): Double = java.lang.Float.intBitsToFloat(b.toInt).toDouble
  def lane(beat: BigInt, i: Int, w: Int = 32): BigInt = (beat >> (w * i)) & ((BigInt(1) << w) - 1)
  def laneF32(beat: BigInt, i: Int): Double = dec(lane(beat, i))

  /** Drive one operand PAIR through the junction and return the folded beat. Both operands are offered
    * concurrently on independent threads, so the join is exercised with real (nonzero, varying) skew.
    */
  def runPair(dut: DataPathJunctionHarness, csr: BigInt, aBeat: BigInt, bBeat: BigInt, bDelay: Int = 0): BigInt = {
    dut.io.csr_i(0).poke(csr.U)
    dut.io.enable_i.poke(true)
    dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
    var out = BigInt(0)
    var th  = new chiseltest.internal.TesterThreadList(Seq())
    th = th.fork {
      dut.io.a_i.bits.poke(aBeat.U); dut.io.a_i.valid.poke(true)
      while (!dut.io.a_i.ready.peekBoolean()) dut.clock.step(1)
      dut.clock.step(1); dut.io.a_i.valid.poke(false)
    }
    th = th.fork {
      if (bDelay > 0) dut.clock.step(bDelay)
      dut.io.b_i.bits.poke(bBeat.U); dut.io.b_i.valid.poke(true)
      while (!dut.io.b_i.ready.peekBoolean()) dut.clock.step(1)
      dut.clock.step(1); dut.io.b_i.valid.poke(false)
    }
    th = th.fork {
      while (!dut.io.out_o.valid.peekBoolean()) dut.clock.step(1)
      out = dut.io.out_o.bits.peekInt()
      dut.io.out_o.ready.poke(true); dut.clock.step(1); dut.io.out_o.ready.poke(false)
    }
    th.joinAndStep()
    dut.io.out_o.ready.poke(true)
    var w = 0; while (dut.io.busy_o.peekBoolean() && w < 400) { dut.clock.step(1); w += 1 }
    dut.io.out_o.ready.poke(false)
    out
  }
}
