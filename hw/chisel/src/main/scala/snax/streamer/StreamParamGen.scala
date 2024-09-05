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
object StreamerParametersGen {

  def readerParams = Seq(
    new ReaderWriterParam(
      spatialBounds = List(8),
      temporalDimension = 6,
      numChannel = 8,
      addressBufferDepth = 2,
      dataBufferDepth = 2
    ),
    new ReaderWriterParam(
      spatialBounds = List(8),
      temporalDimension = 3,
      numChannel = 8,
      addressBufferDepth = 2,
      dataBufferDepth = 2
    )
  )
  def writerParams = Seq(
    new ReaderWriterParam(
      spatialBounds = List(8),
      temporalDimension = 3,
      numChannel = 8,
      addressBufferDepth = 2,
      dataBufferDepth = 2
    )
  )
  def readerWriterParams = Seq(
    new ReaderWriterParam(
      spatialBounds = List(32),
      temporalDimension = 3,
      numChannel = 32,
      addressBufferDepth = 2,
      dataBufferDepth = 2
    ),
    new ReaderWriterParam(
      spatialBounds = List(32),
      temporalDimension = 3,
      numChannel = 32,
      addressBufferDepth = 2,
      dataBufferDepth = 2
    )
  )

}

object StreamerGen {
  def main(args: Array[String]): Unit = {
    val outPath =
      args.headOption.getOrElse("../../target/snitch_cluster/generated/.")
    emitVerilog(
      new Streamer(
        StreamerParam(
          readerParams = StreamerParametersGen.readerParams,
          writerParams = StreamerParametersGen.writerParams,
          readerWriterParams = StreamerParametersGen.readerWriterParams,
          csrAddrWidth = 32,
          tagName = "snax_data_reshuffler_"
        )
      ),
      Array("--target-dir", outPath)
    )
  }
}
