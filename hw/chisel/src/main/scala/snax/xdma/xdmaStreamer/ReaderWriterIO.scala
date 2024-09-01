package snax.xdma.xdmaStreamer

import chisel3._
import chisel3.util._

import snax.utils._
import snax.xdma.DesignParams._

class ReaderWriterCfgIO(val param: ReaderWriterParam) extends Bundle {
  val enabledByte =
    if (param.configurableByteMask)
      UInt((param.tcdmParam.dataWidth / 8).W)
    else UInt(0.W)
  val enabledChannel =
    if (param.configurableChannel)
      UInt(param.tcdmParam.numChannel.W)
    else UInt(0.W)
}

abstract class ReaderWriterCommomIO(val param: ReaderWriterParam)
    extends Bundle {
  // The signal to control address generator
  val aguCfg = Input(new AddressGenUnitCfgIO(param.aguParam))
  // The signal to control which byte is written to TCDM
  val readerwriterCfg = Input(new ReaderWriterCfgIO(param))

  // The signal trigger the start of Address Generator. The non-empty of address generator will cause data requestor to read the data
  val start = Input(Bool())
  // The module is busy if addressgen is busy or fifo in addressgen is not empty
  val busy = Output(Bool())
  // The buffer is empty
  val bufferEmpty = Output(Bool())
}

trait HasTCDMRequestor {
  this: ReaderWriterCommomIO =>
  val tcdmReq = Vec(
    param.tcdmParam.numChannel,
    Decoupled(
      new TcdmReq(
        addrWidth = param.tcdmParam.addrWidth,
        tcdmDataWidth = param.tcdmParam.dataWidth
      )
    )
  )
}

trait HasTCDMResponder {
  this: ReaderWriterCommomIO =>
  val tcdmRsp = Vec(
    param.tcdmParam.numChannel,
    Flipped(Valid(new TcdmRsp(tcdmDataWidth = param.tcdmParam.dataWidth)))
  )
}

trait HasInputDataIO {
  this: ReaderWriterCommomIO =>
  val data = Flipped(
    Decoupled(
      UInt((param.tcdmParam.dataWidth * param.tcdmParam.numChannel).W)
    )
  )
}

trait HasOutputDataIO {
  this: ReaderWriterCommomIO =>
  val data = Decoupled(
    UInt((param.tcdmParam.dataWidth * param.tcdmParam.numChannel).W)
  )
}

class ReaderIO(param: ReaderWriterParam)
    extends ReaderWriterCommomIO(param)
    with HasTCDMRequestor
    with HasTCDMResponder
    with HasOutputDataIO

class WriterIO(param: ReaderWriterParam)
    extends ReaderWriterCommomIO(param)
    with HasTCDMRequestor
    with HasInputDataIO

class ReaderWriterIO(
    readerParam: ReaderWriterParam,
    writerParam: ReaderWriterParam
) extends Bundle {
  // As they share the same TCDM interface, different number of channel is meaningless
  require(
    readerParam.tcdmParam.numChannel == writerParam.tcdmParam.numChannel
  )
  // Full-funcional Reader interface
  val readerInterface = new ReaderIO(readerParam)
  // Writer interface without TCDM port
  val writerInterface = new ReaderWriterCommomIO(writerParam)
    with HasInputDataIO
}
