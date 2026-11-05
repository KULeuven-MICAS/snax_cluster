package snax.xdma.xdmaFrontEnd

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import snax.readerWriter.ReaderWriterParam
import snax.xdma.DesignParams._
import snax.xdma.xdmaIO._

private class ByteMaskInterClusterHarness(dimensions: Int) extends Module with RequireAsyncReset {
  private val cross        = new XDMACrossClusterParam(maxTemporalDimension = dimensions)
  private val param        = new XDMAParam(
    new XDMAConfigParam(32, 32),
    new XDMAAXIParam(),
    cross,
    new ReaderWriterParam(configurableByteMask = true)
  )
  val io                   = IO(new Bundle {
    val in  = Flipped(Decoupled(new XDMAInterClusterCfgIO(param, param)))
    val out = Decoupled(new XDMAInterClusterCfgIO(param, param))
  })
  private val serializer   = Module(new XDMAInterClusterCfgIOSerializer(param))
  private val deserializer = Module(new XDMAInterClusterCfgIODeserializer(param))
  serializer.io.cfgIn <> io.in
  deserializer.io.cfgIn <> serializer.io.cfgOut
  io.out <> deserializer.io.cfgOut
}

class ByteMaskInterClusterTest extends AnyFlatSpec with ChiselScalatestTester {
  // Five dimensions fit in one frame; eight require multiple frames.
  for (dimensions <- Seq(5, 8)) {
    "XDMA configuration transport" should s"preserve all channel masks with $dimensions temporal dimensions" in {
      test(new ByteMaskInterClusterHarness(dimensions))
        .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
          val mask = BigInt("8040201008040201", 16)
          dut.io.in.valid.poke(false.B)
          dut.io.out.ready.poke(false.B)
          dut.io.in.bits.taskID.poke(9.U)
          dut.io.in.bits.isWriterSide.poke(true.B)
          dut.io.in.bits.readerPtr.poke(0x1000.U)
          dut.io.in.bits.writerPtr(0).poke(0x2000.U)
          dut.io.in.bits.writerPtr(1).poke(0x3000.U)
          dut.io.in.bits.axiTransferBeatSize.poke(17.U)
          dut.io.in.bits.spatialStride.poke(1.U)
          for (i <- 0 until dimensions) {
            dut.io.in.bits.temporalBounds(i).poke((i + 1).U)
            dut.io.in.bits.temporalStrides(i).poke((i + 11).U)
          }
          dut.io.in.bits.enabledChannel.poke(0xff.U)
          dut.io.in.bits.enabledByte.poke(mask.U)
          dut.io.in.valid.poke(true.B)
          var cycles = 0
          while (!dut.io.in.ready.peekBoolean() && cycles < 50) {
            dut.clock.step()
            cycles += 1
          }
          assert(cycles < 50, "Serializer did not accept configuration")
          dut.clock.step()
          dut.io.in.valid.poke(false.B)
          cycles = 0
          while (!dut.io.out.valid.peekBoolean() && cycles < 80) {
            dut.clock.step()
            cycles += 1
          }
          assert(cycles < 80, "Configuration was not delivered")
          dut.io.out.bits.enabledByte.expect(mask.U)
          dut.io.out.bits.enabledChannel.expect(0xff.U)
          dut.io.out.bits.taskID.expect(9.U)
          dut.io.out.bits.isWriterSide.expect(true.B)
          dut.io.out.bits.readerPtr.expect(0x1000.U)
          dut.io.out.bits.writerPtr(0).expect(0x2000.U)
          dut.io.out.bits.writerPtr(1).expect(0x3000.U)
          dut.io.out.bits.axiTransferBeatSize.expect(17.U)
          for (i <- 0 until dimensions) {
            dut.io.out.bits.temporalBounds(i).expect((i + 1).U)
            dut.io.out.bits.temporalStrides(i).expect((i + 11).U)
          }
          dut.clock.step(2)
          dut.io.out.valid.expect(true.B)
          dut.io.out.bits.enabledByte.expect(mask.U)
          dut.io.out.ready.poke(true.B)
          dut.clock.step(3)
          dut.io.out.valid.expect(false.B)
        }
    }
  }
}
