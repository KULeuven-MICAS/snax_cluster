package snax.xdma.xdmaStreamer

import chisel3._
import chisel3.util._

import snax.utils._

import snax.csr_manager._

import snax.xdma.DesignParams._

// data to accelerator interface generator
// a vector of decoupled interface with configurable number and configurable width for each port
class DataToAcceleratorX(
    param: StreamerParam
) extends Bundle {
  val data = MixedVec((0 until param.readerNum).map { i =>
    Decoupled(UInt(param.fifoWidthReader(i).W))
  } ++ (0 until param.readerWriterNum).map { i =>
    Decoupled(UInt(param.fifoWidthReaderWriter(i).W))
  })
}

// data from accelerator interface generator
// a vector of decoupled interface with configurable number and configurable width for each port
class DataFromAcceleratorX(
    param: StreamerParam
) extends Bundle {
  val data = MixedVec((0 until param.writerNum).map { i =>
    Flipped(Decoupled(UInt(param.fifoWidthWriter(i).W)))
  } ++ (0 until param.readerWriterNum).map { i =>
    Flipped(Decoupled(UInt(param.fifoWidthReaderWriter(i).W)))
  })
}

// data related io
class StreamerDataIO(
    param: StreamerParam
) extends Bundle {
  // specify the interface to the accelerator
  val streamer2accelerator =
    new DataToAcceleratorX(param)
  val accelerator2streamer =
    new DataFromAcceleratorX(param)

  // specify the interface to the TCDM
  // request interface with q_valid and q_ready
  val tcdmReq =
    (Vec(
      param.tcdmPortsNum,
      Decoupled(new TcdmReq(param.addrWidth, param.tcdmDataWidth))
    ))
  // response interface with p_valid
  val tcdmRsp = (Vec(
    param.tcdmPortsNum,
    Flipped(Valid(new TcdmRsp(param.tcdmDataWidth)))
  ))
}

/** This class represents the input and output ports of the streamer top module
  *
  * @param param
  *   the parameters class instantiation for the streamer top module
  */
class StreamerIO(
    param: StreamerParam
) extends Bundle {

  // ports for csr configuration
  val csr = new SnaxCsrIO(param.csrAddrWidth)

  // ports for data in and out
  val data = new StreamerDataIO(
    param
  )
}

// streamer generator module
class Streamer(
    param: StreamerParam
) extends Module
    with RequireAsyncReset {

  override val desiredName = param.tagName + "Streamer"

  val io = IO(
    new StreamerIO(
      param
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
        param.tagName
      )
    )
  }: _*)

  // data writers instantiation
  // a vector of data writer generator instantiation with different parameters for each module
  val writer = Seq((0 until param.writerNum).map { i =>
    Module(
      new Writer(
        param.writerParams(i),
        param.tagName
      )
    )
  }: _*)

  // data reader_writers instantiation
  val reader_writer = Seq((0 until param.readerWriterNum / 2).map { i =>
    Module(
      new ReaderWriter(
        param.readerWriterParams(i / 2),
        param.readerWriterParams(i / 2 + 1),
        param.tagName
      )
    )
  }: _*)

  // --------------------------------------------------------------------------------
  // ---------------------- streamer state machine-----------------------------------
  // --------------------------------------------------------------------------------
  val streamer_config_fire = Wire(Bool())
  val streamer_finish = Wire(Bool())
  val streamer_busy = Wire(Bool())
  val streamer_ready = Wire(Bool())

  // State declaration
  val sIDLE :: sBUSY :: Nil = Enum(2)
  val cstate = RegInit(sIDLE)
  val nstate = WireInit(sIDLE)

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

  var reader_writer_idx: Int = 0

  streamer_config_fire := csrManager.io.csr_config_out.fire
  streamer_busy := cstate === sBUSY
  streamer_ready := cstate === sIDLE

  // if every data reader/writer is not busy
  streamer_finish := !(reader
    .map(_.io.busy)
    .reduce(_ && _) && writer
    .map(_.io.busy)
    .reduce(_ && _) && reader_writer
    .map(_.io.readerInterface.busy)
    .reduce(_ && _)) && reader_writer
    .map(_.io.writerInterface.busy)
    .reduce(_ && _)

  // --------------------------------------------------------------------------------
  // -----------------------data movers start-----------------------------------------
  // --------------------------------------------------------------------------------

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
  // ---------------------- csr manager connection----------------------------------------------
  // --------------------------------------------------------------------------------

  val reader_csr = param.readerParams.map(_.csrNum).reduce(_ + _)
  val writer_csr = param.writerParams.map(_.csrNum).reduce(_ + _)
  val reader_writer_csr = param.readerWriterParams.map(_.csrNum).reduce(_ + _)

  // extra one is the start csr
  val csrNumReadWrite =
    reader_csr + writer_csr + reader_writer_csr + 1

  // csrManager instantiation
  val csrManager = Module(
    new CsrManager(
      csrNumReadWrite,
      // 2 ready only csr for every streamer
      2,
      param.csrAddrWidth,
      param.tagName
    )
  )

  // connect the csrManager input and streamertop csr req input
  csrManager.io.csr_config_in.req <> io.csr.req

  // io.csr and csrManager input connection
  csrManager.io.csr_config_in.rsp <> io.csr.rsp

  // connect the reader/writer ready and csrManager output ready
  csrManager.io.csr_config_out.ready := streamer_ready

  // add performance counter for streamer
  val streamerBusy2Idle = WireInit(false.B)

  streamerBusy2Idle := !streamer_busy && RegNext(streamer_busy)

  val performance_counter = RegInit(0.U(32.W))
  when(streamer_busy) {
    performance_counter := performance_counter + 1.U
  }.elsewhen(streamerBusy2Idle) {
    performance_counter := 0.U
  }

  // connect the performance counter to the first ready only csr
  csrManager.io.read_only_csr(0) := streamer_busy
  csrManager.io.read_only_csr(1) := performance_counter

  // store the configuration csr for each data mover when config fire
  val csrCfgReg = RegInit(VecInit(Seq.fill(csrNumReadWrite)(0.U(32.W))))
  when(streamer_config_fire) {
    csrCfgReg := csrManager.io.csr_config_out.bits
  }

  // --------------------------------------------------------------------------------
  // ------------------------------------ csr mapping -------------------------------
  // --------------------------------------------------------------------------------

  // ptr csr mapping
  for (i <- 0 until param.dataMoverNum) {
    if (i < param.readerNum) {
      reader(i).io.cfg.ptr <> csrCfgReg(i)
    } else {
      if (i < param.readerNum + param.writerNum) {
        writer(i - param.readerNum).io.cfg.ptr <> csrCfgReg(
          i
        )
      } else {
        reader_writer_idx = (i - param.readerNum - param.writerNum) / 2
        reader_writer(
          reader_writer_idx
        ).io.readerInterface.cfg.ptr <> csrCfgReg(
          i
        )
        reader_writer(
          reader_writer_idx
        ).io.writerInterface.cfg.ptr <> csrCfgReg(
          i + 1
        )
      }
    }
  }

  // temporal loopbound csr mapping
  var tBCsrOffset = param.dataMoverNum
  for (i <- 0 until param.dataMoverNum) {
    for (j <- 0 until param.temporalDim(i)) {
      if (i < param.readerNum) {
        reader(i).io.cfg.bounds(j) <> csrCfgReg(
          tBCsrOffset + param.readerParams
            .map(_.aguParam.dimension)
            .take(i)
            .sum + j
        )
      } else {
        if (i < param.readerNum + param.writerNum) {
          writer(i - param.readerNum).io.cfg.bounds(j) <> csrCfgReg(
            tBCsrOffset + param.readerParams
              .map(_.aguParam.dimension)
              .take(i)
              .sum + i
          )
        } else {
          reader_writer_idx = (i - param.readerNum - param.writerNum) / 2
          reader_writer(reader_writer_idx).io.readerInterface.cfg
            .bounds(j) <> csrCfgReg(
            tBCsrOffset + param.readerParams
              .map(_.aguParam.dimension)
              .take(i)
              .sum + i
          )
          reader_writer(reader_writer_idx).io.writerInterface.cfg
            .bounds(j) <> csrCfgReg(
            tBCsrOffset + param.readerParams
              .map(_.aguParam.dimension)
              .take(i)
              .sum + i + 1
          )
        }
      }
    }
  }

// temporal stride csr mapping
  var tSCsrOffset =
    tBCsrOffset + param.readerParams.map(_.aguParam.dimension).sum
  for (i <- 0 until param.dataMoverNum) {
    for (j <- 0 until param.temporalDim(i)) {
      if (i < param.readerNum) {
        reader(i).io.cfg.strides(j) <> csrCfgReg(
          tSCsrOffset + param.readerParams
            .map(_.aguParam.dimension)
            .take(i)
            .sum + j
        )
      } else {
        if (i < param.readerNum + param.writerNum) {
          writer(i - param.readerNum).io.cfg.strides(j) <> csrCfgReg(
            tSCsrOffset + param.readerParams
              .map(_.aguParam.dimension)
              .take(i)
              .sum + i
          )
        } else {
          reader_writer_idx = (i - param.readerNum - param.writerNum) / 2
          reader_writer(reader_writer_idx).io.readerInterface.cfg
            .strides(j) <> csrCfgReg(
            tSCsrOffset + param.readerParams
              .map(_.aguParam.dimension)
              .take(i)
              .sum + i
          )
          reader_writer(reader_writer_idx).io.writerInterface.cfg
            .strides(j) <> csrCfgReg(
            tSCsrOffset + param.readerParams
              .map(_.aguParam.dimension)
              .take(i)
              .sum + i + 1
          )
        }
      }
    }
  }

  //
  // writer

  // reader_writer

  // --------------------------------------------------------------------------------
  // ---------------------- data reader/writer <> TCDM connection-------------------
  // --------------------------------------------------------------------------------
  def tcdm_read_ports_num =
    param.readerTcdmPorts.reduceLeftOption(_ + _).getOrElse(0)
  def tcdm_write_ports_num =
    param.writerTcdmPorts.reduceLeftOption(_ + _).getOrElse(0)

  def flattenSeq(inputSeq: Seq[Int]): Seq[(Int, Int, Int)] = {
    var flattenedIndex = -1
    val flattenedSeq = inputSeq.zipWithIndex.flatMap { case (size, dimIndex) =>
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
    io.data.tcdmReq(flattenedIndex) <> reader(dimIndex).io.tcdmReq(
      innerIndex
    )

    // signals from TCDM responses
    reader(dimIndex).io.tcdmRsp(innerIndex) <> io.data.tcdmRsp(
      flattenedIndex
    )
  }

  // data writer <> TCDM write ports
  // TCDM request port bias based on the read TCDM ports number
  val write_flatten_seq = flattenSeq(param.writerTcdmPorts)
  for ((dimIndex, innerIndex, flattenedIndex) <- write_flatten_seq) {
    // write request to TCDM
    io.data.tcdmReq(flattenedIndex + tcdm_read_ports_num) <> writer(
      dimIndex
    ).io.tcdmReq(innerIndex)
  }

  // data reader writer <> TCDM read and write ports
  val read_write_flatten_seq = flattenSeq(param.readerWriterTcdmPorts)
  for ((dimIndex, innerIndex, flattenedIndex) <- read_write_flatten_seq) {
    // read request to TCDM
    io.data.tcdmReq(
      flattenedIndex + tcdm_read_ports_num + tcdm_write_ports_num
    ) <> reader_writer(dimIndex).io.readerInterface.tcdmReq(
      innerIndex
    )

    // signals from TCDM responses
    reader_writer(dimIndex).io.readerInterface.tcdmRsp(innerIndex) <> io.data
      .tcdmRsp(
        flattenedIndex + tcdm_read_ports_num + tcdm_write_ports_num
      )
  }

// --------------------------------------------------------------------------------
// ---------------------- data reader/writer <> accelerator data connection-------
// --------------------------------------------------------------------------------

  for (i <- 0 until param.dataMoverNum) {
    if (i < param.readerNum) {
      reader(i).io.data <> io.data.streamer2accelerator.data(i)
    } else {
      if (i < param.readerNum + param.writerNum) {
        writer(
          i - param.readerNum
        ).io.data <> io.data.accelerator2streamer
          .data(i - param.readerNum)
      } else {
        reader_writer_idx = (i - param.readerNum - param.writerNum) / 2
        reader_writer(
          reader_writer_idx
        ).io.readerInterface.data <> io.data.streamer2accelerator.data(
          (i - param.readerNum - param.writerNum) / 2 + param.readerNum
        )
        reader_writer(
          reader_writer_idx
        ).io.writerInterface.data <> io.data.accelerator2streamer
          .data(
            (i - param.writerNum - param.readerNum) / 2 + param.writerNum
          )
      }
    }
  }

  val macro_dir = "generated"
  val macro_template =
    s"""// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
"""
  java.nio.file.Files.write(
    java.nio.file.Paths.get(macro_dir),
    macro_template.getBytes(java.nio.charset.StandardCharsets.UTF_8)
  )
}

object StreamerEmitter extends App {
  emitVerilog(
    new Streamer(StreamerParam()),
    Array("--target-dir", "generated")
  )
}
