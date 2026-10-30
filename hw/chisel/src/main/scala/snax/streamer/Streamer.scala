package snax.streamer

import scala.reflect.runtime.currentMirror
import scala.tools.reflect.ToolBox

import chisel3._
import chisel3.util._

import play.api.libs.json._
import snax.DataPathExtension._
import snax.readerWriter._
import snax.reqRspManager.ReqRspManager
import snax.reqRspManager.SnaxReqRspIO
import snax.utils._

// data to accelerator interface generator
// a vector of decoupled interface with configurable number and configurable width for each port
class DataToAcceleratorX(param: StreamerParam) extends Bundle {
  val data = MixedVec((0 until param.readerNum).map { i =>
    Decoupled(UInt(param.fifoWidthReader(i).W))
    // even for reader, from streamer to accelerator
  } ++ (0 until param.readerWriterNum / 2).map { i =>
    Decoupled(UInt(param.fifoWidthReaderWriter(i * 2).W))
  })
}

// data from accelerator interface generator
// a vector of decoupled interface with configurable number and configurable width for each port
class DataFromAcceleratorX(param: StreamerParam) extends Bundle {
  val data = MixedVec((0 until param.writerNum).map { i =>
    Flipped(Decoupled(UInt(param.fifoWidthWriter(i).W)))
    // oven for writer, from accelerator to streamer
  } ++ (0 until param.readerWriterNum / 2).map { i =>
    Flipped(Decoupled(UInt(param.fifoWidthReaderWriter(i * 2 + 1).W)))
  })
}

// data related io
class StreamerDataIO(param: StreamerParam) extends Bundle {
  // specify the interface to the accelerator
  val streamer2accelerator =
    new DataToAcceleratorX(param)
  val accelerator2streamer =
    new DataFromAcceleratorX(param)

  // specify the interface to the TCDM
  // request interface with q_valid and q_ready
  val tcdm_req =
    (Vec(
      param.tcdmPortsNum,
      Decoupled(new SparseTCDMReq(param.addrWidth, param.tcdmDataWidth))
    ))
  // response interface with p_valid
  val tcdm_rsp = (Vec(
    param.tcdmPortsNum,
    Flipped(Valid(new SparseTCDMRsp(param.tcdmDataWidth)))
  ))
}

/** This class represents the input and output ports of the streamer top module
  *
  * @param param
  *   the parameters class instantiation for the streamer top module
  */
class StreamerIO(param: StreamerParam) extends Bundle {

  // cross clock domain clock from the accelerator
  val accClock = if (param.hasCrossClockDomain) Some(Input(Clock())) else None

  // ports for csr configuration
  val csr = new SnaxReqRspIO(param.csrAddrWidth, 32)

  // ports for data in and out
  val data = new StreamerDataIO(
    param
  )
}

// streamer generator module
class Streamer(param: StreamerParam) extends Module with RequireAsyncReset {

  override val desiredName = param.tagName + "Streamer"

  val io = IO(
    new StreamerIO(
      param
    )
  )

  require(
    param.readerParams.length == param.readerDatapathExtention.length,
    "The number of reader and reader extension should be the same"
  )
  require(
    param.writerParams.length == param.writerDatapathExtention.length,
    "The number of writer and writer extension should be the same"
  )
  require(
    param.readerWriterParams.length == param.readerWriterDatapathExtention.length,
    "The number of reader_writer and reader_writer extension should be the same"
  )

  // --------------------------------------------------------------------------------
  // ---------------------- csr manager instantiation--------------------------------
  // --------------------------------------------------------------------------------

  val reader_csr        =
    param.readerParams.map(_.csrNum).reduceLeftOption(_ + _).getOrElse(0)
  val writer_csr        =
    param.writerParams.map(_.csrNum).reduceLeftOption(_ + _).getOrElse(0)
  val reader_writer_csr =
    param.readerWriterParams.map(_.csrNum).reduceLeftOption(_ + _).getOrElse(0)

  def get_extension_list_csr_num(
    extensionSeqSeq: Seq[Seq[HasDataPathExtension]]
  ): Int = {
    extensionSeqSeq.map {
      case seq if seq.nonEmpty =>
        seq
          .map(_.extensionParam.userCsrNum)
          .reduceLeftOption(_ + _)
          .getOrElse(0) + 1
      case _                   => 0
    }
      .reduceLeftOption(_ + _)
      .getOrElse(0)
  }

  val reader_extension_csr = get_extension_list_csr_num(
    param.readerDatapathExtention
  )

  val writer_extension_csr = get_extension_list_csr_num(
    param.writerDatapathExtention
  )

  val reader_writer_extension_csr = get_extension_list_csr_num(
    param.readerWriterDatapathExtention
  )

  // extra one is the start csr
  val csrNumReadWrite =
    reader_csr + writer_csr + reader_writer_csr + reader_extension_csr + writer_extension_csr + reader_writer_extension_csr + 1

  // csrManager instantiation
  val csrManager = Module(
    new ReqRspManager(
      numReadWriteReg = csrNumReadWrite,
      // 2 ready only csr for every streamer
      numReadOnlyReg  = 2,
      addrWidth       = param.csrAddrWidth,
      ioDataWidth     = 32,
      regDataWidth    = 32,
      moduleTagName   = param.tagName
    )
  )

  // --------------------------------------------------------------------------------
  // ---------------------- data reader/writer instantiation--------------------------
  // --------------------------------------------------------------------------------

  // data readers instantiation
  // a vector of data reader generator instantiation with different parameters for each module
  val reader = Seq((0 until param.readerNum).map { i =>
    Module(
      new Reader(
        param.readerParams(i),
        param.tagName + "_C" + i.toString()
      )
    )
  }: _*)

  // data writers instantiation
  // a vector of data writer generator instantiation with different parameters for each module
  val writer = Seq((0 until param.writerNum).map { i =>
    Module(
      new Writer(
        param.writerParams(i),
        param.tagName + "_C" + i.toString()
      )
    )
  }: _*)

  // data reader_writers instantiation
  val reader_writer = Seq((0 until param.readerWriterNum / 2).map { i =>
    Module(
      new ReaderWriter(
        param.readerWriterParams(i / 2),
        param.readerWriterParams(i / 2 + 1),
        param.tagName + "_C" + i.toString()
      )
    )
  }: _*)

  // datapath extension module instantiation
  val readerDatapathExtention = (0 until param.readerParams.length).map { i =>
    Module(
      new DataPathExtensionHost(
        extensionList    = param.readerDatapathExtention(i),
        dataWidth        = param.fifoWidthReader(i),
        headCut          = false,
        tailCut          = false,
        halfCut          = false,
        moduleNamePrefix = param.tagName
      )
    )
  }

  val writerDatapathExtention = (0 until param.writerParams.length).map { i =>
    Module(
      new DataPathExtensionHost(
        extensionList    = param.writerDatapathExtention(i),
        dataWidth        = param.fifoWidthWriter(i),
        headCut          = false,
        tailCut          = false,
        halfCut          = false,
        moduleNamePrefix = param.tagName
      )
    )
  }

  val readerWriterDatapathExtention =
    (0 until param.readerWriterParams.length).map { i =>
      Module(
        new DataPathExtensionHost(
          extensionList    = param.readerWriterDatapathExtention(i),
          dataWidth        = param.fifoWidthReaderWriter(i),
          headCut          = false,
          tailCut          = false,
          halfCut          = false,
          moduleNamePrefix = param.tagName
        )
      )
    }

  // --------------------------------------------------------------------------------
  // ---------------------- streamer state machine-----------------------------------
  // --------------------------------------------------------------------------------

  val streamer_config_fire = Wire(Bool())
  val streamer_finish      = Wire(Bool())
  val streamer_busy        = Wire(Bool())
  val streamer_ready       = Wire(Bool())

  // State declaration
  val sIDLE :: sBUSY :: Nil = Enum(2)
  val cstate                = RegInit(sIDLE)
  val nstate                = WireInit(sIDLE)

  // Changing states
  cstate := nstate

  chisel3.dontTouch(cstate)
  switch(cstate) {
    is(sIDLE) {
      when(streamer_config_fire) {
        nstate := sBUSY
      }.otherwise {
        nstate := sIDLE
      }

    }
    is(sBUSY) {
      when(streamer_finish) {
        nstate := sIDLE
      }.otherwise {
        nstate := sBUSY
      }
    }
  }

  var reader_writer_idx:           Int = 0
  var reader_writer_extension_idx: Int = 0

  streamer_config_fire := csrManager.io.readWriteRegIO.fire
  streamer_busy        := cstate === sBUSY
  streamer_ready       := cstate === sIDLE

  // if every data reader/writer is not busy
  streamer_finish := !(reader
    .map(_.io.busy)
    .reduceLeftOption(_ || _)
    .getOrElse(0.B) || writer
    .map(_.io.busy)
    .reduceLeftOption(_ || _)
    .getOrElse(0.B) || reader_writer
    .map(_.io.readerInterface.busy)
    .reduceLeftOption(_ || _)
    .getOrElse(0.B) || reader_writer
    .map(_.io.writerInterface.busy)
    .reduceLeftOption(_ || _)
    .getOrElse(0.B) || readerDatapathExtention
    .map(_.io.busy)
    .reduceLeftOption(_ || _)
    .getOrElse(0.B) || writerDatapathExtention
    .map(_.io.busy)
    .reduceLeftOption(_ || _)
    .getOrElse(0.B) || readerWriterDatapathExtention
    .map(_.io.busy)
    .reduceLeftOption(_ || _)
    .getOrElse(0.B))
  dontTouch(streamer_finish)

  // --------------------------------------------------------------------------------
  // -----------------------data movers start-----------------------------------------
  // --------------------------------------------------------------------------------

  // one clock cycle delay for the start signal to store the configuration first
  for (i <- 0 until param.dataMoverNum) {
    if (i < param.readerNum) {
      reader(i).io.start := streamer_config_fire
    } else {
      if (i < param.readerNum + param.writerNum) {
        writer(i - param.readerNum).io.start := streamer_config_fire
      } else {
        reader_writer_idx = (i - param.readerNum - param.writerNum) / 2
        reader_writer(
          reader_writer_idx
        ).io.readerInterface.start := streamer_config_fire
        reader_writer(
          reader_writer_idx
        ).io.writerInterface.start := streamer_config_fire
      }
    }
  }

  // --------------------------------------------------------------------------------
  // -----------------------extension start-----------------------------------------
  // --------------------------------------------------------------------------------

  for (i <- 0 until param.readerDatapathExtention.length) {
    readerDatapathExtention(i).io.start := streamer_config_fire
  }

  for (i <- 0 until param.writerDatapathExtention.length) {
    writerDatapathExtention(i).io.start := streamer_config_fire
  }
  for (i <- 0 until param.readerWriterDatapathExtention.length) {
    readerWriterDatapathExtention(i).io.start := streamer_config_fire
  }

  // --------------------------------------------------------------------------------
  // ---------------------- csr manager connection----------------------------------------------
  // --------------------------------------------------------------------------------

  // connect the csrManager input and streamertop csr req input
  csrManager.io.reqRspIO.req <> io.csr.req

  // io.csr and csrManager input connection
  csrManager.io.reqRspIO.rsp <> io.csr.rsp

  // connect the reader/writer ready and csrManager output ready
  csrManager.io.readWriteRegIO.ready := streamer_ready

  // add performance counter for streamer
  val streamerIdle2Busy = WireInit(false.B)

  streamerIdle2Busy := streamer_busy && !RegNext(streamer_busy)

  val performance_counter = RegInit(0.U(32.W))
  when(streamerIdle2Busy) {
    performance_counter := 0.U
  }.elsewhen(streamer_busy) {
    performance_counter := performance_counter + 1.U
  }

  // connect the performance counter to the first ready only csr
  csrManager.io.readOnlyReg(0) := streamer_busy
  csrManager.io.readOnlyReg(1) := performance_counter

  // store the configuration csr for each data mover when config fire
  val csrCfgReg = RegInit(VecInit(Seq.fill(csrNumReadWrite)(0.U(32.W))))
  val csrCfg    = Wire(Vec(csrNumReadWrite, UInt(32.W)))
  when(streamer_config_fire) {
    csrCfgReg := csrManager.io.readWriteRegIO.bits
  }
  csrCfg := Mux(
    streamer_config_fire,
    csrManager.io.readWriteRegIO.bits,
    csrCfgReg
  )

  // --------------------------------------------------------------------------------
  // ------------------------------------ csr mapping -------------------------------
  // --------------------------------------------------------------------------------

  var remainingCSR = csrCfg.toIndexedSeq

  // reader
  for (i <- 0 until param.readerNum) {
    remainingCSR = reader(i).io.connectCfgWithList(
      remainingCSR
    )
  }

  // writer
  for (i <- 0 until param.writerNum) {
    remainingCSR = writer(i).io.connectCfgWithList(
      remainingCSR
    )
  }

  // reader_writer
  for (i <- 0 until param.readerWriterNum) {
    if (i % 2 == 0) {
      remainingCSR = reader_writer(i / 2).io.readerInterface.connectCfgWithList(
        remainingCSR
      )
    } else {
      remainingCSR = reader_writer(i / 2).io.writerInterface.connectCfgWithList(
        remainingCSR
      )
    }
  }

  // connect the csr configuration to the extension
  for (i <- 0 until param.readerDatapathExtention.length) {
    remainingCSR = readerDatapathExtention(i).io.connectCfgWithList(
      remainingCSR
    )
  }

  for (i <- 0 until param.writerDatapathExtention.length) {
    remainingCSR = writerDatapathExtention(i).io.connectCfgWithList(remainingCSR)
  }

  for (i <- 0 until param.readerWriterDatapathExtention.length) {
    remainingCSR = readerWriterDatapathExtention(i).io.connectCfgWithList(
      remainingCSR
    )
  }

  // 1 left csr for start signal
  require(remainingCSR.length == 1)

  // --------------------------------------------------------------------------------
  // ---------------------- data reader/writer <> TCDM connection-------------------
  // --------------------------------------------------------------------------------

  def tcdm_read_ports_num  = param.readerTcdmPorts.reduceLeftOption(_ + _).getOrElse(0)
  def tcdm_write_ports_num = param.writerTcdmPorts.reduceLeftOption(_ + _).getOrElse(0)

  def flattenSeq(inputSeq: Seq[Int]): Seq[(Int, Int, Int)] = {
    var flattenedIndex = -1
    val flattenedSeq   = inputSeq.zipWithIndex.flatMap { case (size, dimIndex) =>
      (0 until size).map { innerIndex =>
        flattenedIndex = flattenedIndex + 1
        (dimIndex, innerIndex, flattenedIndex)
      }
    }
    flattenedSeq
  }

  // data reader <> TCDM read ports
  val read_flatten_seq = flattenSeq(param.readerTcdmPorts)
  for ((dimIndex, innerIndex, flattenedIndex) <- read_flatten_seq) {
    // read request to TCDM
    io.data.tcdm_req(flattenedIndex) <> reader(dimIndex).io.tcdmReq(
      innerIndex
    )

    // signals from TCDM responses
    reader(dimIndex).io.tcdmRsp(innerIndex) <> io.data.tcdm_rsp(
      flattenedIndex
    )
  }

  // data writer <> TCDM write ports
  // TCDM request port bias based on the read TCDM ports number
  val write_flatten_seq = flattenSeq(param.writerTcdmPorts)
  for ((dimIndex, innerIndex, flattenedIndex) <- write_flatten_seq) {
    // write request to TCDM
    io.data.tcdm_req(flattenedIndex + tcdm_read_ports_num) <> writer(
      dimIndex
    ).io.tcdmReq(innerIndex)
  }

  // data reader writer <> TCDM read and write ports
  val read_write_flatten_seq = flattenSeq(param.readerWriterTcdmPorts)
  for ((dimIndex, innerIndex, flattenedIndex) <- read_write_flatten_seq) {
    // read request to TCDM
    io.data.tcdm_req(
      flattenedIndex + tcdm_read_ports_num + tcdm_write_ports_num
    ) <> reader_writer(dimIndex).io.readerInterface.tcdmReq(
      innerIndex
    )

    // signals from TCDM responses
    reader_writer(dimIndex).io.readerInterface.tcdmRsp(innerIndex) <> io.data
      .tcdm_rsp(
        flattenedIndex + tcdm_read_ports_num + tcdm_write_ports_num
      )
  }

  // --------------------------------------------------------------------------------
  // ---------------------- data reader/writer <> accelerator data connection-------
  // --------------------------------------------------------------------------------
  // --------------------------------------------------------------------------------
  // connect the data path extension
  // --------------------------------------------------------------------------------
  for (i <- 0 until param.dataMoverNum) {
    if (i < param.readerNum) {
      // reader
      readerDatapathExtention(i).io.data.in <> reader(i).io.data
      readerDatapathExtention(i).io.data.out <> io.data.streamer2accelerator
        .data(
          i
        )
    } else {
      // writer
      if (i < param.readerNum + param.writerNum) {
        io.data.accelerator2streamer
          .data(i - param.readerNum) <> writerDatapathExtention(
          i - param.readerNum
        ).io.data.in
        writer(
          i - param.readerNum
        ).io.data <> writerDatapathExtention(i - param.readerNum).io.data.out
      } else {
        // reader_writer
        reader_writer_idx           = (i - param.readerNum - param.writerNum) / 2
        reader_writer_extension_idx = (i - param.readerNum - param.writerNum)
        // reader first
        if (reader_writer_extension_idx % 2 == 0) {
          reader_writer(
            reader_writer_idx
          ).io.readerInterface.data <> readerWriterDatapathExtention(
            reader_writer_extension_idx
          ).io.data.in
          readerWriterDatapathExtention(
            reader_writer_extension_idx
          ).io.data.out <> io.data.streamer2accelerator.data(
            reader_writer_idx + param.readerNum
          )

        } else {
          // writer
          reader_writer(
            reader_writer_idx
          ).io.writerInterface.data <> readerWriterDatapathExtention(
            reader_writer_extension_idx
          ).io.data.out
          io.data.accelerator2streamer
            .data(
              reader_writer_idx + param.writerNum
            ) <> readerWriterDatapathExtention(
            reader_writer_extension_idx
          ).io.data.in
        }

      }
    }
  }

  // --------------------------------------------------------------------------------
  // ---------------------- data reader/writer <> clock from accelerator connection--
  // --------------------------------------------------------------------------------

  if (param.hasCrossClockDomain) {
    for (i <- 0 until param.dataMoverNum) {
      if (i < param.readerNum) {
        reader(i).io.accClock.get := io.accClock.get
      } else {
        if (i < param.readerNum + param.writerNum) {
          writer(i - param.readerNum).io.accClock.get := io.accClock.get
        } else {
          reader_writer_idx = (i - param.readerNum - param.writerNum) / 2
          reader_writer(
            reader_writer_idx
          ).io.readerInterface.accClock.get := io.accClock.get
          reader_writer(
            reader_writer_idx
          ).io.writerInterface.accClock.get := io.accClock.get
        }
      }
    }
  }

  // --------------------------------------------------------------------------------
  // ------------------ csr address map header file generation-----------------------
  // --------------------------------------------------------------------------------
  // The CSR-map computation and file write are pure functions of `param` and live
  // in `object Streamer` so the SW-only entry point (StreamerSwHeaderGen) can
  // emit the header without paying for full Chisel elaboration. We still call it
  // here to preserve the existing RTL-build side effect.
  Streamer.writeCsrAddrMap(param)

}

// Pure-Scala helpers for CSR map computation. Decoupled from the Chisel Module
// body so SW-only generation (StreamerSwHeaderGen) can run without elaborating
// any hardware.
object Streamer {

  // Per-data-mover CSR layout — base ptr, spatial strides, temporal bounds /
  // strides, address remap, channel/byte enables. Returns the C #define block.
  private def genCSRMap(
    csrBase: Int,
    param:   ReaderWriterParam,
    tag:     String
  ): String = {
    var csrMap    = "// CSR Map for " + tag + "\n"
    var csrOffset = csrBase

    // base pointer
    csrMap    = csrMap + "#define BASE_PTR_" + tag + "_LOW " + csrOffset + "\n"
    csrOffset = csrOffset + 1
    csrMap    = csrMap + "#define BASE_PTR_" + tag + "_HIGH " + csrOffset + "\n"
    csrOffset = csrOffset + 1

    // spatial strides base address for this data mover
    csrMap = csrMap + "#define S_STRIDE_BASE_" + tag + " " + csrOffset + "\n"
    csrMap = csrMap + "#define S_STRIDE_NUM_" + tag + " " + param.aguParam.spatialBounds.length + "\n"

    // spatial strides
    for (i <- 0 until param.aguParam.spatialBounds.length) {
      csrMap    = csrMap + "#define " + "S_STRIDE_" + tag + "_" + i + " " + csrOffset + "\n"
      csrOffset = csrOffset + 1
    }

    // temporal bounds base address for this data mover
    csrMap = csrMap + "#define T_BOUND_BASE_" + tag + " " + csrOffset + "\n"
    csrMap = csrMap + "#define T_BOUND_NUM_" + tag + " " + param.aguParam.temporalDimension + "\n"

    // temporal bounds
    for (i <- 0 until param.aguParam.temporalDimension) {
      csrMap    = csrMap + "#define " + "T_BOUND_" + tag + "_" + i + " " + csrOffset + "\n"
      csrOffset = csrOffset + 1
    }

    // temporal strides base address for this data mover
    csrMap = csrMap + "#define T_STRIDE_BASE_" + tag + " " + csrOffset + "\n"
    csrMap = csrMap + "#define T_STRIDE_NUM_" + tag + " " + param.aguParam.temporalDimension + "\n"

    // temporal stride
    for (i <- 0 until param.aguParam.temporalDimension) {
      csrMap    = csrMap + "#define " + "T_STRIDE_" + tag + "_" + i + " " + csrOffset + "\n"
      csrOffset = csrOffset + 1
    }

    // address remap index
    if (param.aguParam.tcdmLogicWordSize.length > 1) {
      csrMap    = csrMap + "#define " + "ADDR_REMAP_INDEX_" + tag + " " + csrOffset + "\n"
      csrOffset = csrOffset + 1
    }

    // channel enable
    if (param.configurableChannel) {
      csrMap    = csrMap + "#define " + "ENABLED_CHANNEL_" + tag + " " + csrOffset + "\n"
      csrMap    =
        csrMap + "#define " + "ENABLED_CHANNEL_" + tag + "_CSR_NUM " + ((param.aguParam.numChannel + 31) / 32) + "\n"
      csrOffset = csrOffset + ((param.aguParam.numChannel + 31) / 32)
    }

    // byte mask enable
    if (param.configurableByteMask) {
      csrMap    = csrMap + "#define " + "ENABLED_BYTE_" + tag + " " + csrOffset + "\n"
      // Expose the word count so software can program masks wider than 32 bytes.
      csrMap    = csrMap + "#define ENABLED_BYTE_" + tag + "_CSR_NUM " + param.byteMaskCSRNum + "\n"
      csrOffset = csrOffset + param.byteMaskCSRNum
    }

    csrMap
  }

  // Per-extension list contribution to the CSR address space. Mirrors the
  // class-body helper of the same name used to size the CSR manager.
  def get_extension_list_csr_num(
    extensionSeqSeq: Seq[Seq[HasDataPathExtension]]
  ): Int = {
    extensionSeqSeq.map {
      case seq if seq.nonEmpty =>
        seq
          .map(_.extensionParam.userCsrNum)
          .reduceLeftOption(_ + _)
          .getOrElse(0) + 1
      case _                   => 0
    }
      .reduceLeftOption(_ + _)
      .getOrElse(0)
  }

  // Build the streamer_csr_addr_map.h text and write it to disk. Pure function
  // of `param`; safe to call without a Chisel Module context.
  def writeCsrAddrMap(param: StreamerParam): Unit = {
    var csrBase   = 0
    var csrBase_i = 0
    var csrMap    = ""

    // reader csr configuration
    csrBase = 960
    for (i <- 0 until param.readerNum) {
      csrBase_i = csrBase + param.readerParams
        .take(i)
        .map(_.csrNum)
        .reduceLeftOption(_ + _)
        .getOrElse(0)
      csrMap    = csrMap + genCSRMap(csrBase_i, param.readerParams(i), "READER_" + i)
    }
    csrBase = csrBase + param.readerParams.map(_.csrNum).sum

    // writer csr configuration
    for (i <- 0 until param.writerNum) {
      csrBase_i = csrBase + param.writerParams
        .take(i)
        .map(_.csrNum)
        .reduceLeftOption(_ + _)
        .getOrElse(0)
      csrMap    = csrMap + genCSRMap(csrBase_i, param.writerParams(i), "WRITER_" + i)
    }
    csrBase = csrBase + param.writerParams.map(_.csrNum).sum

    // reader_writer csr configuration
    for (i <- 0 until param.readerWriterNum) {
      csrBase_i = csrBase + param.readerWriterParams
        .take(i)
        .map(_.csrNum)
        .reduceLeftOption(_ + _)
        .getOrElse(0)
      csrMap    = csrMap + genCSRMap(
        csrBase_i,
        param.readerWriterParams(i),
        "READER_WRITER_" + i
      )
    }
    csrBase = csrBase + param.readerWriterParams.map(_.csrNum).sum

    // extension csr configuration
    csrMap = csrMap + "// Datapath extension CSRs\n"
    var extension_csr_num = 0

    def get_extension_csr_num(extensionSeq: Seq[HasDataPathExtension]): Int = {
      extension_csr_num = extensionSeq match {
        case seq if seq.nonEmpty =>
          seq
            .map(_.extensionParam.userCsrNum)
            .reduceLeftOption(_ + _)
            .getOrElse(0) + 1
        case _                   => 0
      }
      extension_csr_num
    }

    val reader_extension_csr        = get_extension_list_csr_num(param.readerDatapathExtention)
    val writer_extension_csr        = get_extension_list_csr_num(param.writerDatapathExtention)
    val reader_writer_extension_csr = get_extension_list_csr_num(param.readerWriterDatapathExtention)

    // reader extension csr configuration
    for (i <- 0 until param.readerDatapathExtention.length) {
      extension_csr_num = get_extension_csr_num(param.readerDatapathExtention(i))
      if (extension_csr_num > 0) {
        csrBase_i = csrBase + get_extension_list_csr_num(
          param.readerDatapathExtention.take(i)
        )
        csrMap    = csrMap + "#define READER_EXTENSION_" + i + "_CSR_BASE " + csrBase_i + "\n"
        csrMap    = csrMap + s"#define READER_EXTENSION_${i}_CSR_NUM ${extension_csr_num}\n"
      }
    }
    csrBase = csrBase + reader_extension_csr

    // writer extension csr configuration
    for (i <- 0 until param.writerDatapathExtention.length) {
      extension_csr_num = get_extension_csr_num(param.writerDatapathExtention(i))
      if (extension_csr_num > 0) {
        csrBase_i = csrBase + get_extension_list_csr_num(
          param.writerDatapathExtention.take(i)
        )
        csrMap    = csrMap + "#define WRITER_EXTENSION_" + i + "_CSR_BASE " + csrBase_i + "\n"
        csrMap    = csrMap + s"#define WRITER_EXTENSION_${i}_CSR_NUM ${extension_csr_num}\n"
      }
    }
    csrBase = csrBase + writer_extension_csr

    // reader_writer extension csr configuration
    for (i <- 0 until param.readerWriterDatapathExtention.length) {
      extension_csr_num = get_extension_csr_num(
        param.readerWriterDatapathExtention(i)
      )
      if (extension_csr_num > 0) {
        csrBase_i = csrBase + get_extension_list_csr_num(
          param.readerWriterDatapathExtention.take(i)
        )
        csrMap    = csrMap + "#define READER_WRITER_EXTENSION_" + i + "_CSR_BASE " + csrBase_i + "\n"
        csrMap    = csrMap + s"#define READER_WRITER_EXTENSION_${i}_CSR_NUM ${extension_csr_num}\n"
      }
    }
    csrBase = csrBase + reader_writer_extension_csr

    // start csr
    csrMap  = csrMap + "// Other resgiters\n"
    csrMap  = csrMap + "// Status register\n"
    csrMap  = csrMap + "#define STREAMER_START_CSR " + csrBase + "\n"
    csrBase = csrBase + 1

    // streamer busy csr
    csrMap  = csrMap + "// Read only CSRs\n"
    csrMap  = csrMap + "#define STREAMER_BUSY_CSR " + csrBase + "\n"
    csrBase = csrBase + 1

    // streamer performance counter csr
    csrMap = csrMap + "#define STREAMER_PERFORMANCE_COUNTER_CSR " + csrBase + "\n"

    val macro_dir      = param.headerFilepath + "/streamer_csr_addr_map.h"
    val macro_template =
      s"""// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
// This file is generated by Streamer module in hw/chisel to map the CSR address of Streamer automatically, do not modify it manually
// Generated at ${java.time.Instant.now()}

""" ++ csrMap

    // Ensure parent directory exists when invoked from a fresh checkout where
    // the Chisel RTL elaboration hasn't yet created the destination tree.
    java.nio.file.Files.createDirectories(java.nio.file.Paths.get(param.headerFilepath))
    java.nio.file.Files.write(
      java.nio.file.Paths.get(macro_dir),
      macro_template.getBytes(java.nio.charset.StandardCharsets.UTF_8)
    )
  }
}

object StreamerEmitter extends App {
  emitVerilog(
    new Streamer(StreamerParam()),
    Array("--target-dir", "../../target/snitch_cluster/generated")
  )
}

object StreamerGen {

  // The keys of a data_reader_params / data_writer_params / data_reader_writer_params group
  val groupKeys = Set(
    "spatial_bounds",
    "temporal_dim",
    "num_channel",
    "fifo_depth",
    "tcdm_logic_word_size",
    "configurable_channel",
    "dynamicPriority",
    "higherStaticPriority",
    "datapath_extensions"
  )

  // Parse CLI args + streamer JSON config into a fully-populated StreamerParam,
  // also returning the requested SystemVerilog output directory. Shared between
  // the RTL entry point (StreamerGen.main) and the SW-only entry point
  // (StreamerSwHeaderGen.main); keep them in lockstep on this code path so the
  // CSR map seen by SW always matches the RTL.
  //
  // --streamercfg     the accelerator's snax_streamer_cfg as JSON, inside a one-key object
  //                   ({"<tag>_streamer": {...}} from snaxgen, {"snax_streamer_cfg": {...}} in tests)
  // --tcdm-size       TCDM size in KiB
  // --tcdm-data-width TCDM data width in bits
  // --tag-name        the accelerator's tag; the generated names start with "<tag>_"
  // --header-dir      where streamer_csr_addr_map.h goes
  // --hw-target-dir   where the SystemVerilog goes (default "generated")
  def buildParam(args: Array[String]): (StreamerParam, String) = {

    val parsedArgs = snax.utils.ArgParser.parse(args)

    val outPath = parsedArgs.getOrElse(
      "hw-target-dir",
      "generated"
    )

    def fail(message: String): Nothing =
      throw new IllegalArgumentException(
        s"Streamer generation failed: $message"
      )

    def arg(name: String): String =
      parsedArgs.get(name).filter(_ != "NoArg").getOrElse(fail(s"--$name is not provided"))

    def intArg(name: String): Int = arg(name).toIntOption.getOrElse(fail(s"--$name is not an integer"))

    val tcdmSize      = intArg("tcdm-size")
    val tcdmDataWidth = intArg("tcdm-data-width")

    // The streamercfg region is passed to chisel generator as a JSON string
    val streamercfg = Json.parse(arg("streamercfg")) match {
      case JsObject(wrapper) if wrapper.size == 1 =>
        wrapper.head._2 match {
          case cfg: JsObject => cfg
          case _ => fail(s"--streamercfg: ${wrapper.head._1} is not an object")
        }
      case _                                      => fail("--streamercfg is not a one-key object {\"<name>\": {...}}")
    }

    // The streamer reads the three groups and has_crossClockDomain; the other top-level keys
    // (snax_library_name, has_transpose, ...) belong to snaxgen and the other templates

    // A flag is true or false, or a number as in Python (nonzero is true)
    def flag(value: JsValue, where: String): Boolean =
      value match {
        case JsBoolean(b) => b
        case JsNumber(n)  => n != 0
        case _            => fail(s"$where is not a boolean: $value")
      }

    // constrain: all the reader and writer needs to have same config of crossClockDomain
    val hasCrossClockDomain =
      (streamercfg \ "has_crossClockDomain").toOption.exists(flag(_, "has_crossClockDomain"))

    def genDatapathExtensionPerStreamer(
      input: JsValue
    ): Seq[HasDataPathExtension] = {

      val readerDatapathExtentionStr =
        input match {
          case obj: JsObject =>
            obj.fields.filter { case (k, _) => k.startsWith("Has") }.toSeq.map { case (k, v) =>
              // Attempt to parse the value as a Map[String, Either[Int, Seq[Int]]]
              val parsedValue: Map[String, Either[Int, Seq[Int]]] =
                v.validate[Map[String, JsValue]] match {
                  case JsSuccess(fields, _) =>
                    fields.map { case (key, value) =>
                      // Try to parse each value as Either Int or Seq[Int]
                      key -> (value.validate[Int] match {
                        case JsSuccess(intValue, _) =>
                          Left(intValue) // It's a single Int
                        case JsError(_)             =>
                          value.validate[Seq[Int]] match {
                            case JsSuccess(seqValue, _) =>
                              Right(seqValue) // It's a Seq[Int]
                            case JsError(_)             =>
                              throw new Exception(
                                "Invalid data format for " + key
                              )
                          }
                      })
                    }
                  case JsError(_)           =>
                    throw new Exception("Invalid data format in JSON")
                }

              (k, parsedValue)
            }

          case _ => Seq.empty
        }

      println(
        s"Generating datapath extension for streamer with input: $readerDatapathExtentionStr"
      )

      var readerDatapathExtentionPerStreamer = Seq[HasDataPathExtension]()
      val toolbox                            = currentMirror.mkToolBox()

      readerDatapathExtentionStr.foreach(i => {
        readerDatapathExtentionPerStreamer = readerDatapathExtentionPerStreamer :+ toolbox
          .compile(toolbox.parse(s"""
              import snax.DataPathExtension._
              return new ${i._1}(${i._2.map { case (name, value) =>
              val argument = value match {
                case Left(int: Int)       => int.toString // Matching Left for Int
                case Right(seq: Seq[Int]) =>
                  s"Seq(${seq.mkString(",")})" // Matching Right for Seq[Int]
              }
              s"$name = $argument"
            }
              .mkString(", ")})
                    """))()
          .asInstanceOf[HasDataPathExtension]
      })

      readerDatapathExtentionPerStreamer
    }

    // One group: a ReaderWriterParam and the datapath extensions per entry of spatial_bounds;
    // the other keys hold one value per entry too
    def parseGroup(group: String): (Seq[ReaderWriterParam], Seq[Seq[HasDataPathExtension]]) =
      (streamercfg \ group).toOption match {
        case None                   => (Seq.empty, Seq.empty)
        case Some(params: JsObject) =>
          params.keys.diff(groupKeys).foreach(key => fail(s"unknown key $key in $group"))

          def list(key: String):                       Option[Seq[JsValue]] =
            (params \ key).toOption.map(
              _.asOpt[Seq[JsValue]].getOrElse(fail(s"$group.$key is not a list"))
            )
          def entry(key: String, i: Int):              Option[JsValue]      =
            list(key).map(
              _.lift(i).getOrElse(fail(s"$group.$key has no entry $i"))
            )
          def required[T: Reads](key: String, i: Int): T                    =
            entry(key, i)
              .getOrElse(fail(s"$group has no $key"))
              .asOpt[T]
              .getOrElse(fail(s"$group.$key has a wrong entry $i"))

          val num = list("spatial_bounds").getOrElse(fail(s"$group has no spatial_bounds")).length

          val readerWriterParams = (0 until num).map { i =>
            val fifoDepth = required[Int]("fifo_depth", i)
            new ReaderWriterParam(
              spatialBounds        = required[List[Int]]("spatial_bounds", i),
              temporalDimension    = required[Int]("temporal_dim", i),
              tcdmDataWidth        = tcdmDataWidth,
              tcdmSize             = tcdmSize,
              tcdmLogicWordSize    = entry("tcdm_logic_word_size", i)
                .map(_.asOpt[Seq[Int]].getOrElse(fail(s"$group.tcdm_logic_word_size has a wrong entry $i")))
                .getOrElse(Seq(256)),
              numChannel           = required[Int]("num_channel", i),
              addressBufferDepth   = fifoDepth,
              dataBufferDepth      = fifoDepth,
              configurableChannel  = entry("configurable_channel", i).exists(flag(_, s"$group.configurable_channel")),
              dynamicPriority      = entry("dynamicPriority", i).forall(flag(_, s"$group.dynamicPriority")),
              higherStaticPriority = entry("higherStaticPriority", i).exists(flag(_, s"$group.higherStaticPriority")),
              crossClockDomain     = hasCrossClockDomain
            )
          }

          val extensions = list("datapath_extensions").getOrElse(Seq.empty)
          if (extensions.length > num) fail(s"$group.datapath_extensions has more entries than spatial_bounds")
          (
            readerWriterParams,
            (0 until num).map(i => extensions.lift(i).map(genDatapathExtensionPerStreamer).getOrElse(Seq.empty))
          )
        case Some(_)                => fail(s"$group is not an object")
      }

    val (readerParams, readerDatapathExtention)             = parseGroup("data_reader_params")
    val (writerParams, writerDatapathExtention)             = parseGroup("data_writer_params")
    val (readerWriterParams, readerWriterDatapathExtention) = parseGroup("data_reader_writer_params")

    val param = StreamerParam(
      readerParams                  = readerParams,
      writerParams                  = writerParams,
      readerWriterParams            = readerWriterParams,
      readerDatapathExtention       = readerDatapathExtention,
      writerDatapathExtention       = writerDatapathExtention,
      readerWriterDatapathExtention = readerWriterDatapathExtention,
      hasCrossClockDomain           = hasCrossClockDomain,
      csrAddrWidth                  = 32,
      tagName                       = arg("tag-name") + "_",
      headerFilepath                = arg("header-dir")
    )

    (param, outPath)
  }

  def main(args: Array[String]): Unit = {
    val (param, outPath) = buildParam(args)
    // The CSR header is written as a side effect of `new Streamer(param)` (via
    // Streamer.writeCsrAddrMap in the class body) — keeps `make rtl` self-
    // contained.
    emitVerilog(new Streamer(param), Array("--target-dir", outPath))
  }
}

// SW-only entry point. Produces just streamer_csr_addr_map.h via the pure
// `Streamer.writeCsrAddrMap`, with no Chisel Module instantiation and no
// SystemVerilog emission. Invoked by snaxgen.py when `--sw_only=true` and by
// the snax `sw-snax-gen` make target.
object StreamerSwHeaderGen {
  def main(args: Array[String]): Unit = {
    val (param, _) = StreamerGen.buildParam(args)
    Streamer.writeCsrAddrMap(param)
    println(
      s"[StreamerSwHeaderGen] wrote ${param.headerFilepath}/streamer_csr_addr_map.h"
    )
  }
}
