package snax.readerWriter

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

private class ByteMaskCfgHarness(bankWidth: Int, maskWords: Int, enableMask: Boolean = true) extends Module {
  private val param = new ReaderWriterParam(
    spatialBounds        = List(1),
    temporalDimension    = 1,
    tcdmDataWidth        = bankWidth,
    tcdmPhysWordSize     = bankWidth,
    tcdmLogicWordSize    = Seq(bankWidth),
    numChannel           = 1,
    configurableChannel  = false,
    configurableByteMask = enableMask
  )

  val io = IO(new Bundle {
    // A following CSR detects accidental consumption of the next configuration field.
    val csr            = Input(Vec(maskWords + 1, UInt(32.W)))
    val mask           = Output(UInt((bankWidth / 8).W))
    val remainingWords = Output(UInt(32.W))
    val nextWord       = Output(UInt(32.W))
  })

  private val cfg       = Wire(new ReaderWriterCfgIO(param))
  private val remaining = cfg.connectWithList(io.csr.toIndexedSeq)
  io.mask           := (if (enableMask) cfg.enabledByte else Fill(bankWidth / 8, 1.U))
  io.remainingWords := remaining.length.U
  io.nextWord       := remaining.head
}

class ByteMaskCSRConnectionTest extends AnyFlatSpec with ChiselScalatestTester {
  private val cases    = Seq((64, 1), (128, 1), (256, 1), (384, 2), (512, 2), (1024, 4))
  private val patterns = Seq("89abcdef", "01234567", "76543210", "fedcba98").map(BigInt(_, 16))
  private val sentinel = BigInt("13579bdf", 16)

  for ((bankWidth, maskWords) <- cases) {
    "ReaderWriterCfgIO" should s"enable every byte of a $bankWidth-bit bank from $maskWords CSR words" in {
      test(new ByteMaskCfgHarness(bankWidth, maskWords))
        .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
          dut.io.csr.last.poke(sentinel.U)
          val validBits = (BigInt(1) << (bankWidth / 8)) - 1
          dut.io.csr.take(maskWords).foreach(_.poke(BigInt("ffffffff", 16).U))
          dut.io.mask.expect(validBits.U)
          dut.io.remainingWords.expect(1.U)
          dut.io.nextWord.expect(sentinel.U)

          // Distinct words reveal reversed packing; 384 bits also checks a partial final word.
          for ((word, i) <- patterns.take(maskWords).zipWithIndex) {
            dut.io.csr(i).poke(word.U)
          }
          val expectedMask =
            patterns.take(maskWords).zipWithIndex.map { case (word, i) => word << (32 * i) }.reduce(_ | _) & validBits
          dut.io.mask.expect(expectedMask.U)
          dut.io.remainingWords.expect(1.U)
          dut.io.nextWord.expect(sentinel.U)

          // Reconfiguration must clear previously enabled upper bytes as well.
          dut.io.csr.take(maskWords).foreach(_.poke(0.U))
          dut.io.mask.expect(0.U)
        }
    }
  }

  "ReaderWriterCfgIO" should "leave the next CSR untouched when the byte mask is disabled" in {
    test(new ByteMaskCfgHarness(512, 0, enableMask = false))
      .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
        dut.io.csr(0).poke(sentinel.U)
        dut.io.remainingWords.expect(1.U)
        dut.io.nextWord.expect(sentinel.U)
      }
  }
}
