package snax.streamer

import snax.DataPathExtension._
import snax.readerWriter._

/*
 *  This is the collection of all design Params
 *  Design Params is placed all together with companion object to avoid multiple definition of one config & config conflict
 */

// Streamer Module Params

class StreamerParam(
  // data mover params
  val readerParams:       Seq[ReaderWriterParam],
  val writerParams:       Seq[ReaderWriterParam],
  val readerWriterParams: Seq[ReaderWriterParam],

  // datapath extension params
  val readerDatapathExtension:       Seq[Seq[HasDataPathExtension]] = Seq(Seq()),
  val writerDatapathExtension:       Seq[Seq[HasDataPathExtension]] = Seq(Seq()),
  val readerwriterDatapathExtension: Seq[Seq[HasDataPathExtension]] = Seq(
    Seq()
  ),

  // cross clock domain params
  val hasCrossClockDomain:           Boolean                        = false,

  // csr manager params
  val csrAddrWidth:   Int,
  val tagName:        String = "Test",
  val headerFilepath: String = "./generated/"
) {

  // reader, writer, reader-writer number inferred paramters
  val readerNum       = readerParams.length
  val writerNum       = writerParams.length
  val readerWriterNum = readerWriterParams.length
  val dataMoverNum    = readerNum + writerNum + readerWriterNum

  val delayedStartCount: Int = readerParams.map(_.delayedStart).count(identity) +
    writerParams.map(_.delayedStart).count(identity) +
    readerWriterParams.map(_.delayedStart).count(identity)

  // reader, writer, reader-writer tcdm ports inferred parameters
  val readerTcdmPorts       = readerParams.map(_.aguParam.numChannel)
  val writerTcdmPorts       = writerParams.map(_.aguParam.numChannel)
  // The tcdm ports for reader-writer, only the even index (reader's) is used
  // reader and writer share the same tcdm ports
  val readerWriterTcdmPorts =
    readerWriterParams.map(_.aguParam.numChannel).zipWithIndex.filter { case (_, index) => index % 2 == 0 }.map(_._1)
  val tcdmPortsNum          = readerTcdmPorts.sum + writerTcdmPorts.sum + readerWriterTcdmPorts.sum

  // inffered parameters for tcdm
  val addrWidth     = readerParams(0).tcdmParam.addrWidth
  val tcdmDataWidth = readerParams(0).tcdmParam.dataWidth

  // inffered parameters for data fifos
  val fifoWidthReader:       Seq[Int] = readerParams.map(param => param.aguParam.numChannel * param.tcdmParam.dataWidth)
  val fifoWidthWriter:       Seq[Int] = writerParams.map(param => param.aguParam.numChannel * param.tcdmParam.dataWidth)
  val fifoWidthReaderWriter: Seq[Int] =
    readerWriterParams.map(param => param.aguParam.numChannel * param.tcdmParam.dataWidth)

  // design time spatial unrolling factors must match the channel number
  val totalReaderWriterParams =
    readerParams ++ writerParams ++ readerWriterParams
  totalReaderWriterParams.foreach { param =>
    require(
      param.aguParam.spatialBounds.reduce(_ * _) == param.aguParam.numChannel,
      s"spatial unrolling factor product ${param.aguParam.spatialBounds
          .reduce(_ * _)} does not match the channel number ${param.aguParam.numChannel}"
    )
  }

}

object StreamerParam {
  def apply() =
    new StreamerParam(
      readerParams       = Seq(
        new ReaderWriterParam(temporalDimension = 6),
        new ReaderWriterParam(temporalDimension = 3)
      ),
      writerParams       = Seq(new ReaderWriterParam(temporalDimension = 3)),
      readerWriterParams = Seq(
        new ReaderWriterParam(
          temporalDimension = 3,
          numChannel        = 32,
          spatialBounds     = List(32)
        ),
        new ReaderWriterParam(
          temporalDimension = 3,
          numChannel        = 32,
          spatialBounds     = List(32)
        )
      ),
      csrAddrWidth       = 32
    )
  def apply(
    readerParams:                  Seq[ReaderWriterParam],
    writerParams:                  Seq[ReaderWriterParam],
    readerWriterParams:            Seq[ReaderWriterParam],
    readerDatapathExtension:       Seq[Seq[HasDataPathExtension]],
    writerDatapathExtension:       Seq[Seq[HasDataPathExtension]],
    readerwriterDatapathExtension: Seq[Seq[HasDataPathExtension]],
    hasCrossClockDomain:           Boolean,
    csrAddrWidth:                  Int,
    tagName:                       String,
    headerFilepath:                String
  ) =
    new StreamerParam(
      readerParams                  = readerParams,
      writerParams                  = writerParams,
      readerWriterParams            = readerWriterParams,
      readerDatapathExtension       = readerDatapathExtension,
      writerDatapathExtension       = writerDatapathExtension,
      readerwriterDatapathExtension = readerwriterDatapathExtension,
      hasCrossClockDomain           = hasCrossClockDomain,
      csrAddrWidth                  = csrAddrWidth,
      tagName                       = tagName,
      headerFilepath                = headerFilepath
    )
}
