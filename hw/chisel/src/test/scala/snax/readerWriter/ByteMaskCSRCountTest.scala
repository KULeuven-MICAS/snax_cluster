package snax.readerWriter

import org.scalatest.flatspec.AnyFlatSpec

class ByteMaskCSRCountTest extends AnyFlatSpec {
  private def params(bankWidth: Int, enableMask: Boolean): ReaderWriterParam =
    new ReaderWriterParam(
      spatialBounds        = List(1),
      temporalDimension    = 1,
      tcdmDataWidth        = bankWidth,
      tcdmPhysWordSize     = bankWidth,
      tcdmLogicWordSize    = Seq(bankWidth),
      numChannel           = 1,
      configurableChannel  = false,
      configurableByteMask = enableMask
    )

  // Each CSR controls 32 bytes, since the mask contains one bit per byte.
  private val cases = Seq((64, 1), (128, 1), (256, 1), (384, 2), (512, 2), (1024, 4))

  for ((bankWidth, expectedMaskCSRs) <- cases) {
    "ReaderWriterParam" should s"allocate $expectedMaskCSRs byte-mask CSRs for a $bankWidth-bit bank" in {
      val withoutMask    = params(bankWidth, enableMask = false)
      val withMask       = params(bankWidth, enableMask = true)
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
