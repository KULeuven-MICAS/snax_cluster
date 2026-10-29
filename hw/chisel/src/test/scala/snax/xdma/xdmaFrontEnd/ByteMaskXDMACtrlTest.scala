package snax.xdma.xdmaFrontEnd

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import snax.readerWriter.ReaderWriterParam
import snax.xdma.DesignParams._
import snax.xdma.xdmaFrontend.XDMACtrl

class ByteMaskXDMACtrlTest extends AnyFlatSpec with ChiselScalatestTester {
  for (bankWidth <- Seq(512, 1024)) {
    "XDMACtrl" should s"deliver every mask CSR to a $bankWidth-bit writer" in {
      val cfg               = new XDMAConfigParam(addrWidth = 32, dataWidth = 32)
      val axi               = new XDMAAXIParam(dataWidth = bankWidth, addrWidth = 48)
      val cross             = new XDMACrossClusterParam(
        maxMulticastDest     = 1,
        maxTemporalDimension = 1,
        AxiDataWidth         = bankWidth,
        wordlineWidth        = bankWidth
      )
      def rw(mask: Boolean) =
        new ReaderWriterParam(
          spatialBounds        = List(1),
          temporalDimension    = 1,
          tcdmDataWidth        = bankWidth,
          numChannel           = 1,
          configurableChannel  = true,
          configurableByteMask = mask
        )
      test(
        new XDMACtrl(
          new XDMAParam(cfg, axi, cross, rw(false)),
          new XDMAParam(cfg, axi, cross, rw(true))
        )
      ).withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
        dut.io.clusterBaseAddress.poke(0.U)
        dut.io.localXDMACfg.readerBusy.poke(false.B)
        dut.io.localXDMACfg.writerBusy.poke(false.B)
        dut.io.remoteTaskFinished.poke(false.B)
        dut.io.remoteXDMACfg.fromRemote.valid.poke(false.B)
        dut.io.remoteXDMACfg.fromRemote.bits.poke(0.U)
        dut.io.remoteXDMACfg.toRemote.ready.poke(true.B)
        dut.io.csrIO.req.valid.poke(false.B)
        dut.io.csrIO.rsp.ready.poke(true.B)

        def write(address: Int, value: BigInt): Unit = {
          dut.io.csrIO.req.bits.addr.poke(address.U)
          dut.io.csrIO.req.bits.data.poke(value.U)
          dut.io.csrIO.req.bits.strb.poke(15.U)
          dut.io.csrIO.req.bits.write.poke(true.B)
          dut.io.csrIO.req.valid.poke(true.B)
          var cycles = 0
          while (!dut.io.csrIO.req.ready.peekBoolean() && cycles < 50) {
            dut.clock.step()
            cycles += 1
          }
          assert(cycles < 50, "CSR write did not receive a grant")
          dut.clock.step()
          dut.io.csrIO.req.valid.poke(false.B)
        }

        val words  = Seq("89abcdef", "01234567", "76543210", "fedcba98")
          .map(BigInt(_, 16))
          .take(bankWidth / 256)
        // Pointers, reader strides/bound/channel, writer strides/bound/channel, mask.
        val config = Seq(0x1000, 0, 0x2000, 0, bankWidth / 8, 1, bankWidth / 8, 1, bankWidth / 8, 1, bankWidth / 8, 1)
          .map(BigInt(_)) ++ words
        config.zipWithIndex.foreach { case (value, i) => write(i, value) }
        // The start register follows every mask word, not just the first one.
        write(config.length, BigInt(1))
        var cycles = 0
        while (!dut.io.localXDMACfg.writerStart.peekBoolean() && cycles < 50) {
          dut.clock.step()
          cycles += 1
        }
        assert(cycles < 50, "XDMA writer configuration was not delivered")
        val expected = words.zipWithIndex.map { case (word, i) => word << (32 * i) }.reduce(_ | _)
        dut.io.localXDMACfg.writerCfg.readerwriterCfg.enabledByte.expect(expected.U)
        dut.io.localXDMACfg.writerCfg.aguCfg.temporalBounds(0).expect(1.U)
        dut.io.localXDMACfg.writerCfg.aguCfg.ptr.expect(0x2000.U)
      }
    }
  }
}
