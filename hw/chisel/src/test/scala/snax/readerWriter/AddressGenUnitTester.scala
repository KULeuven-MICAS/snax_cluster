package snax.readerWriter

import chisel3._

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class AddressGenUnitTester extends AnyFlatSpec with ChiselScalatestTester {

  println(
    getVerilogString(new AddressGenUnit(AddressGenUnitParam()))
  )

  "AddressGenUnit: continuous fetch with first temporal loop disabled" should " pass" in test(
    new AddressGenUnit(
      AddressGenUnitParam(
        spatialBounds     = List(8),
        temporalDimension = 2,
        numChannel        = 8,
        outputBufferDepth = 2,
        tcdmSize          = 128
      )
    )
  )
    .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
      dut.io.cfg.ptr.poke(0x1000.U)
      dut.io.cfg.spatialStrides(0).poke(8)
      dut.io.cfg.temporalStrides(0).poke(64)
      dut.io.cfg.temporalStrides(1).poke(64)
      dut.io.cfg.temporalBounds(0).poke(0)
      dut.io.cfg.temporalBounds(1).poke(16)

      dut.io.start.poke(true)
      dut.clock.step()
      dut.io.start.poke(false)
      for (i <- 0 until 16) {
        dut.clock.step()
      }

      dut.io.addr.foreach(_.ready.poke(true.B))
      for (i <- 0 until 48) {
        dut.clock.step()
      }

    }

  "AddressGenUnit: continuous fetch with first temporal loop enabled" should " pass" in test(
    new AddressGenUnit(
      AddressGenUnitParam(
        spatialBounds     = List(8),
        temporalDimension = 2,
        numChannel        = 8,
        outputBufferDepth = 2,
        tcdmSize          = 128
      )
    )
  )
    .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
      dut.io.cfg.ptr.poke(0x1000.U)
      dut.io.cfg.spatialStrides(0).poke(8)
      dut.io.cfg.temporalStrides(0).poke(64)
      dut.io.cfg.temporalStrides(1).poke(256)
      dut.io.cfg.temporalBounds(0).poke(4)
      dut.io.cfg.temporalBounds(1).poke(4)

      dut.io.start.poke(true)
      dut.clock.step()
      dut.io.start.poke(false)
      for (i <- 0 until 16) {
        dut.clock.step()
      }

      dut.io.addr.foreach(_.ready.poke(true.B))
      for (i <- 0 until 48) {
        dut.clock.step()
      }
    }

  "AddressGenUnit: continuous 1D fetch with memory remapped to non-interleaved in superbank" should " pass" in test(
    new AddressGenUnit(
      AddressGenUnitParam(
        spatialBounds     = List(8),
        temporalDimension = 2,
        numChannel        = 8,
        outputBufferDepth = 2,
        tcdmSize          = 128,
        tcdmPhysWordSize  = 256,
        tcdmLogicWordSize = Seq(256, 128, 64)
      )
    )
  )
    .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
      dut.clock.setTimeout(0)
      dut.io.cfg.ptr.poke(0x0.U)
      dut.io.cfg.spatialStrides(0).poke(8)
      dut.io.cfg.temporalStrides(0).poke(64)
      dut.io.cfg.temporalStrides(1).poke(0)
      dut.io.cfg.temporalBounds(0).poke(2048)
      dut.io.cfg.temporalBounds(1).poke(1)
      dut.io.cfg.addressRemapIndex.poke(2)

      dut.io.start.poke(true)
      dut.clock.step()
      dut.io.start.poke(false)
      for (i <- 0 until 16) {
        dut.clock.step()
      }

      dut.io.addr.foreach(_.ready.poke(true.B))
      for (i <- 0 until 2048) {
        dut.clock.step()
      }
    }
}

class AddressGenUnitStrideTester extends AnyFlatSpec with ChiselScalatestTester {
  // 512 KiB of TCDM: 19-bit addresses. A 2-operand stream reads its operands d apart through the
  // innermost loop; with d = 0x41280 (bit 18 set) every channel must reach the far operand.
  "AddressGenUnit: a temporal stride with the top address bit set" should " pass" in test(
    new AddressGenUnit(
      AddressGenUnitParam(
        spatialBounds     = List(8),
        temporalDimension = 2,
        numChannel        = 8,
        outputBufferDepth = 2,
        tcdmSize          = 512
      )
    )
  ).withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
    val d = 0x41280
    dut.io.cfg.ptr.poke(0x100.U)
    dut.io.cfg.spatialStrides(0).poke(8)
    dut.io.cfg.temporalStrides(0).poke(d)
    dut.io.cfg.temporalStrides(1).poke(0x40)
    dut.io.cfg.temporalBounds(0).poke(2)
    dut.io.cfg.temporalBounds(1).poke(3)
    dut.io.addr.foreach(_.ready.poke(true.B))
    dut.io.start.poke(true)
    dut.clock.step()
    dut.io.start.poke(false)
    val want = for (j <- 0 until 3; i <- 0 until 2) yield BigInt(0x100 + i * d + j * 0x40)
    val got  = Array.fill(8)(Seq.empty[BigInt])
    var cyc  = 0
    while (got.exists(_.length < want.length) && cyc < 200) {
      for (c <- 0 until 8)
        if (dut.io.addr(c).valid.peek().litToBoolean) got(c) :+= dut.io.addr(c).bits.peek().litValue
      dut.clock.step()
      cyc += 1
    }
    for (c <- 0 until 8)
      assert(got(c) == want.map(_ + 8 * c), s"channel $c: ${got(c).map(_.toString(16))}, want ${want.map(w => (w + 8 * c).toString(16))}")
  }
}

object AddressGenUnitEmitter extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new AddressGenUnit(
      AddressGenUnitParam(
        spatialBounds     = List(8),
        temporalDimension = 2,
        numChannel        = 8,
        outputBufferDepth = 8,
        tcdmSize          = 128,
        tcdmPhysWordSize  = 256,
        tcdmLogicWordSize = Seq(256, 128, 64)
      )
    ),
    args = Array("--target-dir", "generated/xdma")
  )
}
