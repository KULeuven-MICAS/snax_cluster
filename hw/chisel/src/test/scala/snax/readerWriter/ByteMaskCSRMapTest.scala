package snax.readerWriter

import java.nio.file.Files
import org.scalatest.flatspec.AnyFlatSpec
import snax.streamer.{Streamer, StreamerParam}

class ByteMaskCSRMapTest extends AnyFlatSpec {
  for ((bankWidth, maskWords) <- Seq((64, 1), (256, 1), (512, 2), (1024, 4))) {
    "Streamer CSR map" should s"reserve $maskWords mask words for a $bankWidth-bit bank" in {
      val directory            = Files.createTempDirectory("byte-mask-csr-map-")
      val header               = directory.resolve("streamer_csr_addr_map.h")
      def mover(mask: Boolean) =
        new ReaderWriterParam(
          spatialBounds        = List(1),
          temporalDimension    = 1,
          tcdmDataWidth        = bankWidth,
          tcdmPhysWordSize     = bankWidth,
          tcdmLogicWordSize    = Seq(bankWidth),
          numChannel           = 1,
          configurableByteMask = mask
        )
      try {
        Streamer.writeCsrAddrMap(
          new StreamerParam(
            readerParams       = Seq(mover(false)),
            writerParams       = Seq(mover(true), mover(false)),
            readerWriterParams = Seq.empty,
            csrAddrWidth       = 32,
            headerFilepath     = directory.toString
          )
        )
        val defines = Files
          .readString(header)
          .linesIterator
          .filter(_.startsWith("#define "))
          .map(_.split("\\s+"))
          .map(parts => parts(1) -> parts(2).toInt)
          .toMap
        // The following writer and start CSR must move past the entire byte mask.
        assert(defines("ENABLED_BYTE_WRITER_0") == 970)
        assert(defines("ENABLED_BYTE_WRITER_0_CSR_NUM") == maskWords)
        assert(defines("BASE_PTR_WRITER_1_LOW") == 970 + maskWords)
        assert(defines("STREAMER_START_CSR") == 975 + maskWords)
      } finally {
        Files.deleteIfExists(header)
        Files.deleteIfExists(directory)
      }
    }
  }
}
