package snax.xdma.DesignParams

import chisel3.util.log2Up
import chisel3.util.log2Ceil
import snax.readerWriter.ReaderWriterParam
import snax.xdma.xdmaExtension._

/*
 *  This is the collection of all design Params
 *  Design Params is placed all together with companion object to avoid multiple definition of one config & config conflict
 */

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
    val tagName: String = ""
) {
  // val readerParams = readerParams
  // val writerParams = writerParams
  // val readerWriterParams = readerWriterParams

  // val csrAddrWidth = csrAddrWidth

  val temporalDim: Seq[Int] =
    readerParams.map(_.aguParam.dimension) ++ writerParams.map(
      _.aguParam.dimension
    ) ++ readerWriterParams.map(_.aguParam.dimension)
  val spatialDim: Seq[Int] =
    readerParams.map(_.aguParam.dimension) ++ writerParams.map(
      _.aguParam.dimension
    ) ++ readerWriterParams.map(_.aguParam.dimension)
  // readerParams.map(_.spatialDim) ++ writerParams.map(
  //   _.spatialDim
  // ) ++ readerWriterParams.map(
  //   _.spatialDim
  // )

  val readerNum: Int = readerParams.length
  val writerNum: Int = writerParams.length
  val readerWriterNum: Int = readerWriterParams.length
  val dataMoverNum: Int =
    readerNum + writerNum + readerWriterNum

  val readerTcdmPorts: Seq[Int] = readerParams.map(_.aguParam.numChannel)
  val writerTcdmPorts: Seq[Int] = writerParams.map(_.aguParam.numChannel)
  val readerWriterTcdmPorts: Seq[Int] =
    readerWriterParams.map(_.aguParam.numChannel)
  val tcdmPortsNum: Int =
    readerTcdmPorts.sum + writerTcdmPorts.sum + readerWriterTcdmPorts.sum

  val addrWidth = readerParams(0).tcdmParam.addrWidth
  val tcdmDataWidth = readerParams(0).tcdmParam.dataWidth

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
    readerParams =
      Seq(new ReaderWriterParam( /* constructor parameters if any */ )),
    writerParams =
      Seq(new ReaderWriterParam( /* constructor parameters if any */ )),
    readerWriterParams = Seq(
      new ReaderWriterParam( /* constructor parameters if any */ ),
      new ReaderWriterParam( /* constructor parameters if any */ )
    ),
    csrAddrWidth = 32
  )
}
