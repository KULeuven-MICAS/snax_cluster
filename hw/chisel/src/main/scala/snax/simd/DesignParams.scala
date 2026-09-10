package snax.simd

import snax.DataPathExtension.HasDataPathExtension
import snax.readerWriter.ReaderWriterParam

/*
 * Parameters for the standalone SIMD block.
 *
 * The SIMD block is the reader-side half of the xDMA datapath -- reader AGU, the DataPathExtension chain, writer
 * AGU -- with a CSR port and TCDM ports and nothing else. It deliberately has NO AXI, no cross-cluster cfg, no
 * data switch and no junctions, so none of the xDMA's cross-cluster parameters appear here.
 *
 * The CSR layout is kept in the same ORDER as the xDMA's (reader, extensions, writer, start) so that the software
 * port from snax-xdma-lib to snax-simd-lib is mechanical. See tmp/xdma-four-engine-decoupling-plan.md §3.3.
 */

/** The SNAX CSR port geometry. `dataWidth` mirrors the xDMA's `cfg_io_width`. */
class SimdConfigParam(val addrWidth: Int = 32, val dataWidth: Int = 32)

class SimdParam(
  val cfgParam:    SimdConfigParam,
  val readerParam: ReaderWriterParam,
  val writerParam: ReaderWriterParam,
  val extParam:    Seq[HasDataPathExtension] = Seq[HasDataPathExtension]()
) {

  /** The beat width shared by the reader, the extension chain and the writer. */
  val dataWidth: Int = readerParam.tcdmParam.dataWidth * readerParam.tcdmParam.numChannel

  require(
    writerParam.tcdmParam.dataWidth * writerParam.tcdmParam.numChannel == dataWidth,
    s"SIMD reader beat (${dataWidth} b) and writer beat " +
      s"(${writerParam.tcdmParam.dataWidth * writerParam.tcdmParam.numChannel} b) must match: the extension chain " +
      "is a single stream, there is no width converter between them."
  )
  require(
    readerParam.configurableByteMask == false,
    "Reader does not support a byte mask (snax.readerWriter.Reader requires configurableByteMask == false)."
  )
  extParam.foreach { ext =>
    require(
      ext.extensionParam.dataWidth == dataWidth,
      s"Extension ${ext.extensionParam.moduleName} is ${ext.extensionParam.dataWidth} b wide but the SIMD beat is " +
        s"${dataWidth} b."
    )
  }

  /** CSRs consumed by one AGU, i.e. exactly what `AddressGenUnitCfgIO.connectWithList` pops. */
  private def aguCsrNum(param: ReaderWriterParam): Int =
    2 +                                     // ptr, always 2 CSRs (Cat(csr(1), csr(0)), truncated to the AGU width)
      param.aguParam.spatialBounds.length +
      param.aguParam.temporalDimension * 2 + // bounds, then strides
      (if (param.aguParam.tcdmLogicWordSize.length > 1) 1 else 0) // addressRemapIndex

  /** CSRs consumed by one `ReaderWriterCfgIO.connectWithList`. */
  private def rwCsrNum(param: ReaderWriterParam): Int =
    (if (param.configurableChannel) (param.tcdmParam.numChannel + 31) / 32 else 0) +
      (if (param.configurableByteMask) 1 else 0)

  def readerCsrNum: Int = aguCsrNum(readerParam) + rwCsrNum(readerParam)
  def writerCsrNum: Int = aguCsrNum(writerParam) + rwCsrNum(writerParam)

  /** User CSRs of the extension chain, without the enable bitmask. */
  def extUserCsrNum: Int = extParam.map(_.extensionParam.userCsrNum).sum

  /** The whole extension region: the enable bitmask plus the user CSRs (0 when there is no extension). */
  def extCsrNum: Int = if (extParam.isEmpty) 0 else extUserCsrNum + 1

  /** Read-write CSRs: the three regions plus the start register, which must stay LAST -- `ReqRspManager` derives
    * its launch pulse from a write to the last read-write address.
    */
  def totalRwCsrNum: Int = readerCsrNum + extCsrNum + writerCsrNum + 1

  /** Read-only CSRs, in order: submitted tasks, finished tasks, last-task cycles, last-reader cycles,
    * last-writer cycles, status.
    */
  def totalRoCsrNum: Int = 6
}
