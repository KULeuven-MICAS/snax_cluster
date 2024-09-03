package snax.xdma.DesignParams

import chisel3.util.log2Up
import snax.xdma.xdmaExtension._
import chisel3.util.log2Ceil

/*
 *  This is the collection of all design Params
 *  Design Params is placed all together with companion object to avoid multiple definition of one config & config conflict
 */

// TCDM Params

class TCDMParam(
    val addrWidth: Int,
    val dataWidth: Int,
    val numChannel: Int,
    val tcdmSize: Int
)

object TCDMParam {
  def apply(dataWidth: Int, numChannel: Int, tcdmSize: Int): TCDMParam =
    new TCDMParam(log2Ceil(tcdmSize) + 10, dataWidth, numChannel, tcdmSize)
  // By default, the TCDM is 128kB, 64bit data width, 8 channels
  def apply(): TCDMParam = apply(dataWidth = 64, numChannel = 8, tcdmSize = 128)
}

// Streamer Params

class AddressGenUnitParam(
    val spatialBounds: List[Int],
    val temporalDimension: Int,
    val addressWidth: Int,
    val numChannel: Int,
    val outputBufferDepth: Int,
    val tcdmSize: Int
)

object AddressGenUnitParam {
  def apply(
      spatialBounds: List[Int],
      temporalDimension: Int,
      numChannel: Int,
      outputBufferDepth: Int,
      tcdmSize: Int
  ): AddressGenUnitParam = new AddressGenUnitParam(
    spatialBounds = spatialBounds,
    temporalDimension = temporalDimension,
    addressWidth = log2Ceil(tcdmSize) + 10,
    numChannel = numChannel,
    outputBufferDepth = outputBufferDepth,
    tcdmSize = tcdmSize
  )

  // The Very Simple instantiation of the Param
  def apply(): AddressGenUnitParam = apply(
    spatialBounds = List(8),
    temporalDimension = 2,
    numChannel = 8,
    outputBufferDepth = 8,
    tcdmSize = 128
  )
}

class ReaderWriterParam(
    spatialBounds: List[Int] = List(8),
    temporalDimension: Int = 2,
    tcdmDataWidth: Int = 64,
    tcdmSize: Int = 128,
    numChannel: Int = 8,
    addressBufferDepth: Int = 8,
    dataBufferDepth: Int = 8,
    val configurableChannel: Boolean = false,
    val configurableByteMask: Boolean = false,
    val hasTranspose: Boolean = false
) {
  val aguParam = AddressGenUnitParam(
    spatialBounds = spatialBounds,
    temporalDimension = temporalDimension,
    numChannel = numChannel,
    outputBufferDepth = addressBufferDepth,
    tcdmSize = tcdmSize
  )

  val tcdmParam = TCDMParam(
    dataWidth = tcdmDataWidth,
    numChannel = numChannel,
    tcdmSize = tcdmSize
  )

  // Data buffer's depth
  val bufferDepth = dataBufferDepth

  val csrNum =
    2 + spatialBounds.length + 2 * temporalDimension + (if (configurableChannel)
                                                          1
                                                        else
                                                          0) + (if (
                                                                  configurableByteMask
                                                                ) 1
                                                                else
                                                                  0) + (if (
                                                                          hasTranspose
                                                                        ) 1
                                                                        else
                                                                          0)
}

// AXI Params
class AXIParam(
    val dataWidth: Int = 512,
    val addrWidth: Int = 48
)

// DMA Params
class DMADataPathParam(
    val axiParam: AXIParam,
    val rwParam: ReaderWriterParam,
    val extParam: Seq[HasDMAExtension] = Seq[HasDMAExtension]()
)

class DMAExtensionParam(
    val moduleName: String,
    val userCsrNum: Int,
    val dataWidth: Int = 512
) {
  require(dataWidth > 0)
  require(userCsrNum >= 0)
}

class DMACtrlParam(
    val readerparam: ReaderWriterParam,
    val writerparam: ReaderWriterParam,
    val readerextparam: Seq[DMAExtensionParam] = Seq[DMAExtensionParam](),
    val writerextparam: Seq[DMAExtensionParam] = Seq[DMAExtensionParam]()
)

class StreamerParam(
    // data mover params
    val readerParams: Seq[ReaderWriterParam],
    val writerParams: Seq[ReaderWriterParam],
    val readerWriterParams: Seq[ReaderWriterParam],

    // csr manager params
    val csrAddrWidth: Int,
    val tagName: String = "Test_"
) {

  // reader, writer, reader-writer number inferred paramters
  val readerNum: Int = readerParams.length
  val writerNum: Int = writerParams.length
  val readerWriterNum: Int = readerWriterParams.length
  val dataMoverNum: Int =
    readerNum + writerNum + readerWriterNum

  // reader, writer, reader-writer tcdm ports inferred parameters
  val readerTcdmPorts: Seq[Int] = readerParams.map(_.aguParam.numChannel)
  val writerTcdmPorts: Seq[Int] = writerParams.map(_.aguParam.numChannel)
  // The tcdm ports for reader-writer, only the even index (reader's) is used
  // reader and writer share the same tcdm ports
  val readerWriterTcdmPorts: Seq[Int] =
    readerWriterParams
      .map(_.aguParam.numChannel)
      .zipWithIndex
      .filter { case (_, index) => index % 2 == 0 }
      .map(_._1)
  val tcdmPortsNum: Int =
    readerTcdmPorts.sum + writerTcdmPorts.sum + readerWriterTcdmPorts.sum

  // inffered parameters for tcdm
  val addrWidth = readerParams(0).tcdmParam.addrWidth
  val tcdmDataWidth = readerParams(0).tcdmParam.dataWidth

  // inffered parameters for data fifos
  val fifoWidthReader: Seq[Int] = readerParams.map(param =>
    param.aguParam.numChannel * param.tcdmParam.dataWidth
  )
  val fifoWidthWriter: Seq[Int] = writerParams.map(param =>
    param.aguParam.numChannel * param.tcdmParam.dataWidth
  )
  val fifoWidthReaderWriter: Seq[Int] = readerWriterParams.map(param =>
    param.aguParam.numChannel * param.tcdmParam.dataWidth
  )

}

object StreamerParam {
  def apply() = new StreamerParam(
    readerParams = Seq(
      new ReaderWriterParam(temporalDimension = 6),
      new ReaderWriterParam(temporalDimension = 3)
    ),
    writerParams = Seq(new ReaderWriterParam(temporalDimension = 3)),
    readerWriterParams = Seq(
      new ReaderWriterParam(temporalDimension = 3, numChannel = 32),
      new ReaderWriterParam(temporalDimension = 3, numChannel = 32)
    ),
    csrAddrWidth = 32
  )
}
