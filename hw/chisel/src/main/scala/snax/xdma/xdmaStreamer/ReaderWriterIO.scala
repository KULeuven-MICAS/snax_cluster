package snax.xdma.xdmaStreamer

import chisel3._
import chisel3.util._

import snax.utils._
import snax.xdma.CommonCells._
import snax.xdma.DesignParams._

abstract class ReaderWriterCommomIO(val param: ReaderWriterParam)
    extends Bundle {
  // The signal to control address generator
  val cfg = Input(new AddressGenUnitCfgIO(param.aguParam))
  // The signal to control which byte is written to TCDM
  val strb = Input(UInt((param.tcdmParam.dataWidth / 8).W))
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
  val readerInterface = new ReaderWriterCommomIO(readerParam)
    with HasOutputDataIO
  val writerInterface = new ReaderWriterCommomIO(writerParam)
    with HasInputDataIO

  // TCDM Interface
  val tcdmReq = Vec(
    readerParam.tcdmParam.numChannel,
    Decoupled(
      new TcdmReq(
        addrWidth = readerParam.tcdmParam.addrWidth,
        tcdmDataWidth = readerParam.tcdmParam.dataWidth
      )
    )
  )
  val tcdmRsp = Vec(
    readerParam.tcdmParam.numChannel,
    Flipped(Valid(new TcdmRsp(tcdmDataWidth = readerParam.tcdmParam.dataWidth)))
  )
}
