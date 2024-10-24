// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51


package snax.streamer
 
import snax.readerWriter._
import snax.csr_manager._
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
        8
      ),
      temporalDimension = 6,
      tcdmDataWidth = 64,
      tcdmSize = 4096,
      tcdmLogicWordSize = Seq(
        256,
        128,
        64
      ),
      numChannel = 8,
      addressBufferDepth = 8,
      dataBufferDepth = 8,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        8
      ),
      temporalDimension = 3,
      tcdmDataWidth = 64,
      tcdmSize = 4096,
      tcdmLogicWordSize = Seq(
        256,
        128,
        64
      ),
      numChannel = 8,
      addressBufferDepth = 8,
      dataBufferDepth = 8,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
    )
  )

  def writerParams = Seq(
    new ReaderWriterParam(
      spatialBounds = List(
        8
      ),
      temporalDimension = 3,
      tcdmDataWidth = 64,
      tcdmSize = 4096,
      tcdmLogicWordSize = Seq(
        256,
        128,
        64
      ),
      numChannel = 8,
      addressBufferDepth = 8,
      dataBufferDepth = 8,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
    )
  )

  def readerWriterParams = Seq(
    new ReaderWriterParam(
      spatialBounds = List(
        8,
        4
      ),
      temporalDimension = 3,
      tcdmDataWidth = 64,
      tcdmSize = 4096,
      tcdmLogicWordSize = Seq(
        256,
        128,
        64
      ),
      numChannel = 32,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = true,
      crossClockDomain = hasCrossClockDomain
   ), 
    new ReaderWriterParam(
      spatialBounds = List(
        8,
        4
      ),
      temporalDimension = 3,
      tcdmDataWidth = 64,
      tcdmSize = 4096,
      tcdmLogicWordSize = Seq(
        256,
        128,
        64
      ),
      numChannel = 32,
      addressBufferDepth = 2,
      dataBufferDepth = 2,
      configurableChannel = false,
      crossClockDomain = hasCrossClockDomain
    )
  )

  def hasTranspose = true

  def hasCBroadcast = true

  def headerFilepath = "../../target/snitch_cluster/sw/snax/gemmx/include"
}


object StreamerGen {
  def main(args: Array[String]): Unit = {
    val outPath =
      args.headOption.getOrElse("../../target/snitch_cluster/generated")
    emitVerilog(
      new Streamer(
        StreamerParam(
          readerParams = StreamerParametersGen.readerParams,
          writerParams = StreamerParametersGen.writerParams,
          readerWriterParams = StreamerParametersGen.readerWriterParams,
          hasTranspose = StreamerParametersGen.hasTranspose,
          hasCBroadcast = StreamerParametersGen.hasCBroadcast,
          hasCrossClockDomain = StreamerParametersGen.hasCrossClockDomain,
          csrAddrWidth = 32,
          tagName = "snax_streamer_gemmX_",
          headerFilepath = StreamerParametersGen.headerFilepath
        )
      ),
      Array("--target-dir", outPath)
    )
  }
}

object StreamerHeaderFileGen {
  def main(args: Array[String]): Unit = {
    new StreamerHeaderFile(
      StreamerParam(
        readerParams = StreamerParametersGen.readerParams,
        writerParams = StreamerParametersGen.writerParams,
        readerWriterParams = StreamerParametersGen.readerWriterParams,
        hasTranspose = StreamerParametersGen.hasTranspose,
        hasCBroadcast = StreamerParametersGen.hasCBroadcast,
        hasCrossClockDomain = StreamerParametersGen.hasCrossClockDomain,
        csrAddrWidth = 32,
        tagName = "snax_streamer_gemmX_",
        headerFilepath = StreamerParametersGen.headerFilepath
      )
    )
  }
}
