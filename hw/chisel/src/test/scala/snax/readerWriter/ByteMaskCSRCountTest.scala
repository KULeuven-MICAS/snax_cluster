package snax.readerWriter

import org.scalatest.flatspec.AnyFlatSpec

class ByteMaskCSRCountTest extends AnyFlatSpec {
  private def params(bankWidth: Int, channels: Int, enableMask: Boolean): ReaderWriterParam =
    new ReaderWriterParam(
      spatialBounds        = List(channels),
      temporalDimension    = 1,
      tcdmDataWidth        = bankWidth,
      tcdmPhysWordSize     = bankWidth,
      tcdmLogicWordSize    = Seq(bankWidth),
      numChannel           = channels,
      configurableChannel  = false,
      configurableByteMask = enableMask
    )

  // Each CSR controls 32 bytes, since the mask contains one bit per byte.
  private val cases = Seq(
    (64, 1, 1),
    (128, 1, 1),
    (256, 1, 1),
    (384, 1, 2),
    (512, 1, 2),
    (1024, 1, 4),
    (64, 3, 1),
    (64, 5, 2),
    (64, 8, 2),
    (128, 3, 2),
    (256, 2, 2),
    (384, 3, 5),
    (512, 2, 4),
    (1024, 2, 8)
  )

  for ((bankWidth, channels, expectedMaskCSRs) <- cases) {
    "ReaderWriterParam" should s"allocate $expectedMaskCSRs byte-mask CSRs for $channels channels of $bankWidth bits" in {
      val withoutMask    = params(bankWidth, channels, enableMask = false)
      val withMask       = params(bankWidth, channels, enableMask = true)
      // Two pointer words, one spatial stride, one temporal bound and one temporal stride.
      assert(withoutMask.csrNum == 5)
      assert(withoutMask.byteMaskCSRNum == 0)
      assert(withMask.byteMaskCSRNum == expectedMaskCSRs)
      val actualMaskCSRs = withMask.csrNum - withoutMask.csrNum
      assert(
        actualMaskCSRs == expectedMaskCSRs,
        s"$bankWidth-bit bank: expected $expectedMaskCSRs mask CSRs, got $actualMaskCSRs"
      )
    }
  }
}
