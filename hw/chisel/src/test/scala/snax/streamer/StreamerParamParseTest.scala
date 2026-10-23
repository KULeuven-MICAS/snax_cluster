// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.streamer

import org.scalatest.flatspec.AnyFlatSpec
import snax.readerWriter.ReaderWriterParam

class StreamerParamParseTest extends AnyFlatSpec {

  private val headerDir = "../../target/snitch_cluster/sw/snax/test/include"

  private val otherArgs = Array(
    "--tcdm-size",
    "256",
    "--tcdm-data-width",
    "64",
    "--tag-name",
    "snax_test",
    "--header-dir",
    headerDir
  )

  private def parse(cfg: String, wrapper: String = "snax_streamer_cfg"): StreamerParam =
    StreamerGen.buildParam(Array("--streamercfg", s"""{"$wrapper": $cfg}""") ++ otherArgs)._1

  // spatialBounds, temporalDimension, numChannel, addressBufferDepth, dataBufferDepth, tcdmLogicWordSize,
  // configurableChannel, dynamicPriority, higherStaticPriority, crossClockDomain
  private def fields(p: ReaderWriterParam) =
    (
      p.aguParam.spatialBounds,
      p.aguParam.temporalDimension,
      p.aguParam.numChannel,
      p.aguParam.outputBufferDepth,
      p.bufferDepth,
      p.aguParam.tcdmLogicWordSize,
      p.configurableChannel,
      p.dynamicPriority,
      p.higherStaticPriority,
      p.crossClockDomain
    )

  private val full = """{
    "data_reader_params": {
      "spatial_bounds": [[8], [8, 2]],
      "temporal_dim": [6, 3],
      "num_channel": [8, 16],
      "fifo_depth": [4, 2],
      "tcdm_logic_word_size": [[256], [256, 128, 64]],
      "configurable_channel": [0, 1],
      "dynamicPriority": [true, false],
      "higherStaticPriority": [false, true]
    },
    "data_writer_params": {
      "spatial_bounds": [[4]], "temporal_dim": [2], "num_channel": [4], "fifo_depth": [1],
      "configurable_channel": [true]
    },
    "data_reader_writer_params": {
      "spatial_bounds": [[8]], "temporal_dim": [3], "num_channel": [8], "fifo_depth": [3],
      "datapath_extensions": [{"HasBroadcaster": {"inputLength": 256, "outputLength": 2048}}]
    },
    "has_crossClockDomain": true,
    "snax_library_name": "test",
    "has_transpose": true
  }"""

  private val minimal = """{
    "data_reader_params": {"spatial_bounds": [[8]], "temporal_dim": [2], "num_channel": [8], "fifo_depth": [2]}
  }"""

  "StreamerGen.buildParam" should "parse two readers, one writer and one reader-writer into their fields" in {
    val param = parse(full)
    assert(
      param.readerParams.map(fields) == Seq(
        (List(8), 6, 8, 4, 4, Seq(256), false, true, false, true),
        (List(8, 2), 3, 16, 2, 2, Seq(256, 128, 64), true, false, true, true)
      )
    )
    assert(param.writerParams.map(fields) == Seq((List(4), 2, 4, 1, 1, Seq(256), true, true, false, true)))
    assert(param.readerWriterParams.map(fields) == Seq((List(8), 3, 8, 3, 3, Seq(256), false, true, false, true)))
    assert(param.hasCrossClockDomain)
    assert(param.readerDatapathExtention == Seq(Seq(), Seq()))
    assert(param.writerDatapathExtention == Seq(Seq()))
    assert(
      param.readerWriterDatapathExtention.map(_.map(_.extensionParam.moduleName)) == Seq(
        Seq("Broadcaster256to2048")
      )
    )
  }

  it should "use the defaults when the optional keys are absent" in {
    val param = parse(minimal)
    assert(param.readerParams.map(fields) == Seq((List(8), 2, 8, 2, 2, Seq(256), false, true, false, false)))
    assert(!param.hasCrossClockDomain)
    assert(param.writerParams.isEmpty && param.readerWriterParams.isEmpty)
    assert(param.readerDatapathExtention == Seq(Seq()))
  }

  it should "give the TCDM arguments to every ReaderWriterParam" in {
    val param = parse(full)
    val all   = param.readerParams ++ param.writerParams ++ param.readerWriterParams
    assert(all.length == 4)
    for (p <- all) {
      assert(p.tcdmParam.dataWidth == 64)
      assert(p.tcdmParam.tcdmSize == 256)
      assert(p.aguParam.tcdmSize == 256)
    }
  }

  it should "take the tag and the header folder from the arguments" in {
    // snaxgen's wrapper key is "<tag>_streamer"
    val param = parse(minimal, wrapper = "snax_test_streamer")
    assert(param.tagName == "snax_test_")
    assert(param.headerFilepath == headerDir)
  }

  it should "fail on an unknown key in a group, naming the key and the group" in {
    val error = intercept[IllegalArgumentException] {
      parse(minimal.replace("fifo_depth", "fifo_dept"))
    }
    assert(error.getMessage.contains("fifo_dept"))
    assert(error.getMessage.contains("data_reader_params"))
  }

  it should "ignore an unknown top-level key" in {
    val param = parse("""{
      "data_reader_params": {"spatial_bounds": [[8]], "temporal_dim": [2], "num_channel": [8], "fifo_depth": [2]},
      "has_new_wrapper_flag": true
    }""")
    assert(param.readerParams.map(fields) == parse(minimal).readerParams.map(fields))
  }

  it should "fail on a missing required key" in {
    val error = intercept[IllegalArgumentException] {
      parse(minimal.replace(""", "num_channel": [8]""", ""))
    }
    assert(error.getMessage.contains("num_channel"))
    assert(error.getMessage.contains("data_reader_params"))
  }
}
