// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51


package snax.streamer
 
import snax.readerWriter._
import snax.reqRspManager._
import snax.utils._
 
import chisel3._
import chisel3.util._

// Streamer parameters
// tcdm_size in KB
object StreamerParametersGen {

// constrain: all the reader and writer needs to have same config of crossClockDomain
  def hasCrossClockDomain = false

  def readerParams = Seq(
    new ReaderWriterParam(
      spatialBounds = List(
        4
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 4,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        6
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 6,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        2
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 2,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        1
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 1,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        1
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 1,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        1
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 1,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        1
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 1,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        8
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 8,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        1
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 1,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        1
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 1,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        1
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 1,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        1
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 1,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        6
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 6,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        4
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 4,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
    )
  )

  def writerParams = Seq(
    new ReaderWriterParam(
      spatialBounds = List(
        1
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 1,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        1
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 1,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        1
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 1,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        4
      ),
      temporalDimension = 4,
      tcdmDataWidth = 64,
      tcdmSize = 1024,
      tcdmLogicWordSize = Seq(256),
      numChannel = 4,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
    )
  )

  def readerWriterParams = Seq()

  def tagName = "snax_simbacore_"
  def headerFilepath = "../../target/snitch_cluster/sw/snax/simbacore/include"
}
