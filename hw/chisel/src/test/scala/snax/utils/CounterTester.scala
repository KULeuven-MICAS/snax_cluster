package snax.utils

import chisel3._

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class BasicCounterTester extends AnyFlatSpec with ChiselScalatestTester {
  println(getVerilogString(new BasicCounter(8)))
  "The basic counter" should " pass" in {
    test(new BasicCounter(8)).withAnnotations(
      Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)
    ) { dut =>
      dut.io.ceil.poke(28)
      for (i <- 0 until 128) {
        dut.io.tick.poke(i % 2)
        dut.clock.step()
      }
    }
  }
}

class UpDownCounterTester extends AnyFlatSpec with ChiselScalatestTester {
  println(getVerilogString(new UpDownCounter(8)))
  "The up down counter" should " pass" in {
    test(new UpDownCounter(8)).withAnnotations(
      Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)
    ) { dut =>
      dut.io.ceil.poke(28)
      for (i <- 0 until 128) {
        dut.io.tickUp.poke(i         % 2 == 0)
        dut.io.tickDown.poke((i + 1) % 7 == 0)
        dut.clock.step()
      }
      for (i <- 0 until 128) {
        dut.io.tickUp.poke(i         % 7 == 0)
        dut.io.tickDown.poke((i + 1) % 2 == 0)
        dut.clock.step()
      }
    }
  }
}

class ProgrammableCounterTester extends AnyFlatSpec with ChiselScalatestTester {
  // A 19-bit counter addresses 512 KiB. The step must be as wide as the value: a stride with the
  // top bit set -- two operands 261 KiB apart -- has to advance the value by all of it.
  "The programmable counter" should "step by a stride with its top bit set" in {
    test(new ProgrammableCounter(19, hasCeil = true, "ProgrammableCounterTop")).withAnnotations(
      Seq(VerilatorBackendAnnotation)
    ) { dut =>
      val step = 0x41280
      dut.io.ceil.poke(3)
      dut.io.step.poke(step)
      dut.io.reset.poke(true)
      dut.clock.step()
      dut.io.reset.poke(false)
      dut.io.tick.poke(true)
      for (i <- 0 until 6) {
        val want = (i % 3) * step % (1 << 19)
        assert(dut.io.value.peek().litValue == want, s"tick $i: value ${dut.io.value.peek().litValue}, want $want")
        dut.clock.step()
      }
    }
  }
}
