package snax.readerWriter

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import snax.utils.SparseTCDMReq

private class ByteMaskWriterHarness(bankWidth: Int, channels: Int = 1) extends Module with RequireAsyncReset {
  private val param  = new ReaderWriterParam(
    spatialBounds        = List(channels),
    temporalDimension    = 1,
    tcdmDataWidth        = bankWidth,
    tcdmPhysWordSize     = bankWidth,
    tcdmLogicWordSize    = Seq(bankWidth),
    numChannel           = channels,
    addressBufferDepth   = 2,
    dataBufferDepth      = 2,
    configurableByteMask = true,
    dynamicPriority      = false
  )
  val io             = IO(new Bundle {
    val csr     = Input(Vec(param.csrNum, UInt(32.W)))
    val start   = Input(Bool())
    val data    = Flipped(Decoupled(UInt((bankWidth * channels).W)))
    val request = Vec(channels, Decoupled(new SparseTCDMReq(param.tcdmParam.addrWidth, bankWidth)))
  })
  private val writer = Module(new Writer(param))
  require(writer.io.connectCfgWithList(io.csr.toIndexedSeq).isEmpty)
  writer.io.start := io.start
  writer.io.data <> io.data
  io.request <> writer.io.tcdmReq
}

class ByteMaskWriterTest extends AnyFlatSpec with ChiselScalatestTester {
  for (bankWidth <- Seq(64, 512, 1024)) {
    "Writer" should s"drive the complete byte strobe from CSR words for a $bankWidth-bit bank" in {
      test(new ByteMaskWriterHarness(bankWidth))
        .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
          val words        = Seq("89abcdef", "01234567", "76543210", "fedcba98")
            .map(BigInt(_, 16))
            .take((bankWidth / 8 + 31) / 32)
          // Two pointer words, spatial stride, one temporal bound and its stride.
          val config       = Seq(BigInt(0), BigInt(0), BigInt(bankWidth / 8), BigInt(1), BigInt(bankWidth / 8)) ++ words
          config.zipWithIndex.foreach { case (word, i) => dut.io.csr(i).poke(word.U) }
          val expectedMask = words.zipWithIndex.map { case (word, i) => word << (32 * i) }.reduce(_ | _) &
            ((BigInt(1) << (bankWidth / 8)) - 1)
          dut.io.start.poke(false.B)
          dut.io.data.bits.poke(0x1234.U)
          dut.io.data.valid.poke(false.B)
          dut.io.request.head.ready.poke(false.B)

          dut.io.start.poke(true.B)
          dut.clock.step()
          dut.io.start.poke(false.B)
          dut.io.data.valid.poke(true.B)
          var cycles = 0
          while (!dut.io.data.ready.peekBoolean() && cycles < 20) {
            dut.clock.step()
            cycles += 1
          }
          assert(cycles < 20, "Writer did not accept input data")
          dut.clock.step()
          dut.io.data.valid.poke(false.B)
          cycles = 0
          while (!dut.io.request.head.valid.peekBoolean() && cycles < 20) {
            dut.clock.step()
            cycles += 1
          }
          assert(cycles < 20, "Writer did not issue a TCDM request")
          dut.io.request.head.bits.addr.expect(0.U)
          dut.io.request.head.bits.data.expect(0x1234.U)
          dut.io.request.head.bits.write.expect(true.B)
          dut.io.request.head.bits.strb.expect(expectedMask.U)
          // Keep the request stalled to verify that the upper strobe bits persist.
          dut.clock.step(2)
          dut.io.request.head.valid.expect(true.B)
          dut.io.request.head.bits.strb.expect(expectedMask.U)
          dut.io.request.head.ready.poke(true.B)
          dut.clock.step(3)
          dut.io.request.head.valid.expect(false.B)
        }
    }
  }

  "Writer" should "give eight narrow channels independent strobes under unequal grants and clear all masks" in {
    test(new ByteMaskWriterHarness(64, channels = 8))
      .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
        dut.io.start.poke(false.B)
        dut.io.data.valid.poke(false.B)
        dut.io.request.foreach(_.ready.poke(false.B))
        val data = (0 until 8).map(i => BigInt(i + 1) << (64 * i)).reduce(_ | _)

        def transfer(words: Seq[BigInt], strobes: Seq[Int]): Unit = {
          val config = Seq(BigInt(0), BigInt(0), BigInt(8), BigInt(1), BigInt(64)) ++ words
          config.zipWithIndex.foreach { case (word, i) => dut.io.csr(i).poke(word.U) }
          dut.io.request.foreach(_.ready.poke(false.B))
          dut.io.start.poke(true.B)
          dut.clock.step()
          dut.io.start.poke(false.B)
          dut.io.data.bits.poke(data.U)
          dut.io.data.valid.poke(true.B)
          var cycles = 0
          while (!dut.io.data.ready.peekBoolean() && cycles < 40) {
            dut.clock.step()
            cycles += 1
          }
          assert(cycles < 40, "Writer did not accept the complete beat")
          dut.clock.step()
          dut.io.data.valid.poke(false.B)
          cycles = 0
          while (!dut.io.request.forall(_.valid.peekBoolean()) && cycles < 40) {
            dut.clock.step()
            cycles += 1
          }
          assert(cycles < 40, "Not all narrow channels issued a request")
          for (channel <- 0 until 8) {
            dut.io.request(channel).bits.addr.expect((channel * 8).U)
            dut.io.request(channel).bits.data.expect((channel + 1).U)
            dut.io.request(channel).bits.write.expect(true.B)
            dut.io.request(channel).bits.strb.expect(strobes(channel).U)
          }
          // Lower channels may complete while the upper channels remain stalled.
          dut.io.request.take(4).foreach(_.ready.poke(true.B))
          dut.clock.step()
          dut.io.request.foreach(_.ready.poke(false.B))
          dut.clock.step(2)
          for (channel <- 0 until 8) {
            dut.io.request(channel).valid.expect((channel >= 4).B)
            if (channel                                   >= 4)
              dut.io.request(channel).bits.strb.expect(strobes(channel).U)
          }
          dut.io.request.drop(4).foreach(_.ready.poke(true.B))
          dut.clock.step(3)
          dut.io.request.foreach(_.valid.expect(false.B))
        }

        transfer(Seq(BigInt("08040201", 16), BigInt("80402010", 16)), (0 until 8).map(1 << _))
        transfer(Seq(BigInt(0), BigInt(0)), Seq.fill(8)(0))
      }
  }

}
