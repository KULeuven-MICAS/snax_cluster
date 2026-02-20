package snax.readerWriter

import chisel3._
import chisel3.util._

import snax.utils._

class ReaderWriterCfgIO(val param: ReaderWriterParam) extends Bundle {
  val enabledByte    =
    if (param.configurableByteMask)
      UInt((param.tcdmParam.dataWidth / 8).W)
    else UInt(0.W)
  val enabledChannel =
    if (param.configurableChannel)
      UInt(param.tcdmParam.numChannel.W)
    else UInt(0.W)

  def connectWithList(csrList: IndexedSeq[UInt]): IndexedSeq[UInt] = {
    var remaincsrList        = csrList
    def enabledChannelCSRNum =
      if (param.configurableChannel)
        ((param.tcdmParam.numChannel + 31) / 32)
      else 0
    if (param.configurableChannel) {
      enabledChannel := remaincsrList.take(enabledChannelCSRNum).reduce(_ ## _)
      remaincsrList = remaincsrList.drop(enabledChannelCSRNum)
    } else {
      enabledChannel := Fill(param.tcdmParam.numChannel, 1.U)
    }
    if (param.configurableByteMask) {
      enabledByte := remaincsrList.head
      remaincsrList = remaincsrList.tail
    } else {
      enabledByte := Fill(param.tcdmParam.dataWidth / 8, 1.U)
    }
    remaincsrList
  }
}

abstract class ReaderWriterCommomIO(val param: ReaderWriterParam) extends Bundle {
  // The signal to control address generator
  val aguCfg          = Input(new AddressGenUnitCfgIO(param.aguParam))
  // The signal to control which byte is written to TCDM
  val readerwriterCfg = Input(new ReaderWriterCfgIO(param))
  // The port to feed in the clock signal from acc
  val accClock        = if (param.crossClockDomain) Some(Input(Clock())) else None

  def connectCfgWithList(csrList: IndexedSeq[UInt]): IndexedSeq[UInt] = {
    var remaincsrList = csrList
    remaincsrList = aguCfg.connectWithList(remaincsrList)
    remaincsrList = readerwriterCfg.connectWithList(remaincsrList)
    remaincsrList
  }

  // The signal trigger the start of Address Generator. The non-empty of address generator will cause data requestor to read the data
  val start       = Input(Bool())
  // The module is busy if addressgen is busy or fifo in addressgen is not empty
  val busy        = Output(Bool())
  // The buffer is empty
  val bufferEmpty = Output(Bool())
}

trait HasTCDMRequestor {
  this: ReaderWriterCommomIO =>
  val tcdmReq = Vec(
    param.tcdmParam.numChannel,
    Decoupled(
      new RegReq(
        addrWidth = param.tcdmParam.addrWidth,
        dataWidth = param.tcdmParam.dataWidth
      )
    )
  )
}

trait HasTCDMResponder {
  this: ReaderWriterCommomIO =>
  val tcdmRsp = Vec(
    param.tcdmParam.numChannel,
    Flipped(Valid(new RegRsp(dataWidth = param.tcdmParam.dataWidth)))
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

trait HasFixedCacheInputIO {
  this: ReaderWriterCommomIO =>
  val fixedCacheInstruction = Flipped(
    Decoupled(new FixedCacheInstructionIO(param.aguParam.fixedCacheDepth))
  )
}

trait HasFixedCacheOutputIO {
  this: ReaderWriterCommomIO =>
  val fixedCacheInstruction = Decoupled(new FixedCacheInstructionIO(param.aguParam.fixedCacheDepth))
}

class ReaderIO(param: ReaderWriterParam, isReaderWriter: Boolean = false)
    extends ReaderWriterCommomIO(param)
    with HasTCDMRequestor
    with HasTCDMResponder
    with HasOutputDataIO
    with HasFixedCacheInputIO {
  // When used in a ReaderWriter configuration, expose the writer-side port so the writer can write
  // incoming data directly into the reader's FixedLevelCache memory.
  val fixedCacheWriterPort =
    if (isReaderWriter)
      Some(new FixedLevelCacheWriterPort(param.aguParam.fixedCacheDepth, param.tcdmParam.dataWidth * param.tcdmParam.numChannel))
    else
      None
}

class WriterIO(param: ReaderWriterParam, isReaderWriter: Boolean = false)
    extends ReaderWriterCommomIO(param)
    with HasTCDMRequestor
    with HasInputDataIO
    with HasFixedCacheOutputIO {
  // When used in a ReaderWriter configuration, expose the port that the writer uses to inject data
  // directly into the reader's FixedLevelCache memory (bypassing TCDM for cached iterations).
  // Flipped so that from inside the Writer module the fields are outputs (driven by Writer).
  val fixedCacheWriterPort =
    if (isReaderWriter)
      Some(Flipped(new FixedLevelCacheWriterPort(param.aguParam.fixedCacheDepth, param.tcdmParam.dataWidth * param.tcdmParam.numChannel)))
    else
      None
}

class ReaderWriterIO(readerParam: ReaderWriterParam, writerParam: ReaderWriterParam) extends Bundle {
  // As they share the same TCDM interface, different number of channel is meaningless
  require(
    readerParam.tcdmParam.numChannel == writerParam.tcdmParam.numChannel
  )
  // Full-funcional Reader interface
  val readerInterface = new ReaderIO(readerParam)
  // Writer interface without TCDM port
  val writerInterface = new ReaderWriterCommomIO(writerParam) with HasInputDataIO
}
