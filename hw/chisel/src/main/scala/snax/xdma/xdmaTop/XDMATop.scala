package snax.xdma.xdmaTop

import scala.reflect.runtime.currentMirror
import scala.tools.reflect.ToolBox

import chisel3._
import chisel3.util._

import play.api.libs.json._
import snax.DataPathExtension._
import snax.DataPathJunction._
import snax.readerWriter.ReaderWriterParam
import snax.utils._
import snax.xdma.DesignParams._
import snax.xdma.xdmaFrontend._
import snax.xdma.xdmaIO.XDMADataPathCfgIO
import snax.reqRspManager.SnaxReqRspIO

class XDMATopIO(readerParam: XDMAParam, writerParam: XDMAParam) extends Bundle {
  val clusterBaseAddress = Input(
    UInt(writerParam.axiParam.addrWidth.W)
  )
  val csrIO = new SnaxReqRspIO(addrWidth = readerParam.cfgParam.addrWidth, dataWidth = readerParam.cfgParam.dataWidth)

  val remoteXDMACfg = new Bundle {
    val fromRemote = Flipped(
      Decoupled(UInt(readerParam.axiParam.dataWidth.W))
    )
    val toRemote   = Decoupled(UInt(readerParam.axiParam.dataWidth.W))
  }

  val tcdmReader = new Bundle {
    val req = Vec(
      readerParam.rwParam.tcdmParam.numChannel,
      Decoupled(
        new SparseTCDMReq(
          // The address width of the TCDM => Should be equal to axiAddrWidth
          readerParam.rwParam.tcdmParam.addrWidth,
          readerParam.rwParam.tcdmParam.dataWidth
        )
      )
    )
    val rsp = Vec(
      readerParam.rwParam.tcdmParam.numChannel,
      Flipped(
        Valid(
          new SparseTCDMRsp(dataWidth = readerParam.rwParam.tcdmParam.dataWidth)
        )
      )
    )
  }
  val tcdmWriter = new Bundle {
    val req = Vec(
      writerParam.rwParam.tcdmParam.numChannel,
      Decoupled(
        new SparseTCDMReq(
          // The address width of the TCDM => Should be equal to axiAddrWidth
          writerParam.rwParam.tcdmParam.addrWidth,
          writerParam.rwParam.tcdmParam.dataWidth
        )
      )
    )
  }

  val remoteXDMAData = new Bundle {
    val fromRemote               = Flipped(
      Decoupled(
        UInt(
          (writerParam.rwParam.tcdmParam.dataWidth * writerParam.rwParam.tcdmParam.numChannel).W
        )
      )
    )
    val fromRemoteAccompaniedCfg = Output(
      new XDMADataPathCfgIO(
        axiParam          = writerParam.axiParam,
        crossClusterParam = writerParam.crossClusterParam
      )
    )
    val toRemote                 = Decoupled(
      UInt(
        (writerParam.rwParam.tcdmParam.dataWidth * writerParam.rwParam.tcdmParam.numChannel).W
      )
    )
    val toRemoteAccompaniedCfg   = Output(
      new XDMADataPathCfgIO(
        axiParam          = readerParam.axiParam,
        crossClusterParam = readerParam.crossClusterParam
      )
    )
  }

  val remoteTaskFinished = Input(Bool())

  val status = new Bundle {
    val readerBusy = Output(Bool())
    val writerBusy = Output(Bool())
  }
}

class XDMATop(readerParam: XDMAParam, writerParam: XDMAParam, clusterName: String = "unnamed_cluster")
    extends Module
    with RequireAsyncReset {
  override val desiredName = s"${clusterName}_xdma"
  val io                   = IO(
    new XDMATopIO(
      readerParam = readerParam,
      writerParam = writerParam
    )
  )

  val xdmaCtrl = Module(
    new XDMACtrl(
      readerparam = readerParam,
      writerparam = writerParam,
      clusterName = clusterName
    )
  )

  val xdmaDatapath = Module(
    new XDMADataPath(
      readerParam = readerParam,
      writerParam = writerParam,
      clusterName = clusterName
    )
  )

  // Give the dmactrl the current cluster address
  xdmaCtrl.io.clusterBaseAddress := io.clusterBaseAddress

  // IO0: Start to connect datapath to TCDM
  io.tcdmReader <> xdmaDatapath.io.tcdmReader
  io.tcdmWriter <> xdmaDatapath.io.tcdmWriter

  // IO1: Start to connect datapath to axi
  io.remoteXDMAData <> xdmaDatapath.io.remoteXDMAData

  // IO2: Start to coonect ctrl to csr
  io.csrIO <> xdmaCtrl.io.csrIO

  // IO3: Start to connect ctrl to remoteDMADataPath
  io.remoteXDMACfg <> xdmaCtrl.io.remoteXDMACfg

  // Interconnection between ctrl and datapath
  xdmaCtrl.io.localXDMACfg.readerCfg <> xdmaDatapath.io.readerCfg

  xdmaCtrl.io.localXDMACfg.writerCfg <> xdmaDatapath.io.writerCfg

  xdmaDatapath.io.readerStart := xdmaCtrl.io.localXDMACfg.readerStart

  xdmaDatapath.io.writerStart := xdmaCtrl.io.localXDMACfg.writerStart

  xdmaCtrl.io.localXDMACfg.readerBusy := xdmaDatapath.io.readerBusy

  xdmaCtrl.io.localXDMACfg.writerBusy := xdmaDatapath.io.writerBusy
  xdmaCtrl.io.junctionStarved         := xdmaDatapath.io.junctionStarved
  xdmaCtrl.io.junctionCfgErr          := xdmaDatapath.io.junctionCfgErr

  // The status signal
  io.status.readerBusy := xdmaCtrl.io.localXDMACfg.readerBusy

  io.status.writerBusy := xdmaCtrl.io.localXDMACfg.writerBusy

  // The remote task finished signal
  xdmaCtrl.io.remoteTaskFinished := io.remoteTaskFinished
}

/** A holder that instantiates the xDMA and the SIMD block side by side.
  *
  * Its only purpose is to force ONE Chisel elaboration for both blocks. The two share a lot of library
  * hardware -- `BasicCounter`, the native `Fp*` units, the width converters -- and CIRCT names and optimises
  * those per circuit. Elaborated separately, the two files end up with same-named modules of different shapes
  * (measured: `FpAdd` 16-bit-out vs 32-bit-out, a `BasicCounter` with and without its `io_reset` port), which
  * is a hard compile error the moment both files are in one design. Elaborated together, CIRCT dedups what is
  * identical and gives distinct names to what is not.
  *
  * This module itself is never instantiated: the SV wrappers instantiate `<cluster>_xdma` and `<cluster>_simd`
  * directly. It exists only as an elaboration root, so its ports are tied off rather than exposed.
  */
class ClusterBlocks(
  xdmaReaderParam: XDMAParam,
  xdmaWriterParam: XDMAParam,
  simdParam:       snax.simd.SimdParam,
  clusterName:     String
) extends Module
    with RequireAsyncReset {
  override val desiredName = s"${clusterName}_blocks"

  val xdma = Module(new XDMATop(readerParam = xdmaReaderParam, writerParam = xdmaWriterParam, clusterName = clusterName))
  val simd = Module(new snax.simd.SimdTop(param = simdParam, clusterName = clusterName))

  // Tie both children off. DontCare on the inputs and a single OR of the outputs keeps the children from being
  // optimised away without dragging their full port lists onto this holder.
  xdma.io := DontCare
  simd.io := DontCare
  dontTouch(xdma.io)
  dontTouch(simd.io)
}

object XDMATopGen extends App {
  val parsedArgs = snax.utils.ArgParser.parse(args)
  // The xdmaCfg region is passed to chisel generator as a JSON string
  val xdmaCfg    = parsedArgs.find(_._1 == "xdmaCfg")
  if (xdmaCfg.isEmpty) {
    println("xdmaCfg is not provided, generation failed. ")
    sys.exit(-1)
  }
  val parsedXdmaCfg: JsValue = Json.parse(xdmaCfg.get._2)

  /*
  Transferred Parameters:
    snax_xdma_cfg: {
        bender_target: ["snax_KUL_cluster_xdma"],
        reader_buffer: 4,
        writer_buffer: 4,
        reader_agu_spatial_bounds: "8",
        reader_agu_temporal_dimension: 6,
        writer_agu_spatial_bounds: "8",
        writer_agu_temporal_dimension: 6,
        HasVerilogMemset: "",
        HasMaxPool: "",
        HasTransposer: "row=Seq(8), col=Seq(8), elementWidth=Seq(8)",
    }
   */

  val cfgParam = new XDMAConfigParam(
    addrWidth = 32,
    dataWidth = (parsedXdmaCfg \ "cfg_io_width").as[Int]
  )

  val axiParam = new XDMAAXIParam(
    addrWidth = parsedArgs("axiAddrWidth").toInt,
    dataWidth = parsedArgs("axiDataWidth").toInt
  )

  val crossClusterParam = new XDMACrossClusterParam(
    maxMulticastDest     = (parsedXdmaCfg \ "max_multicast").as[Int],
    maxTemporalDimension = (parsedXdmaCfg \ "max_dimension").as[Int],
    // `max_mem_size_kiB` is the unified XDMA-addressable region (KiB).
    tcdmSize             = (parsedXdmaCfg \ "max_mem_size_kiB").as[Int],
    // wordlineWidth (per-bank TCDM data bus width) equals the cluster's
    // `data_width` knob — the schema documents data_width as carrying the
    // TCDM wordline width as well.
    wordlineWidth        = parsedArgs("tcdmDataWidth").toInt,
    AxiAddressWidth      = parsedArgs("axiAddrWidth").toInt
  )

  val readerParam = new ReaderWriterParam(
    spatialBounds        = List(
      parsedArgs("axiDataWidth").toInt / parsedArgs("tcdmDataWidth").toInt
    ),
    temporalDimension    = (parsedXdmaCfg \ "reader_agu_temporal_dimension").as[Int],
    tcdmDataWidth        = parsedArgs("tcdmDataWidth").toInt,
    tcdmSize             = parsedArgs("tcdmSize").toInt,
    numChannel           = parsedArgs("axiDataWidth").toInt / parsedArgs("tcdmDataWidth").toInt,
    addressBufferDepth   = (parsedXdmaCfg \ "reader_buffer").as[Int],
    dataBufferDepth      = (parsedXdmaCfg \ "reader_buffer").as[Int],
    configurableChannel  = true,
    configurableByteMask = false,
    dynamicPriority      = false,
    higherStaticPriority = true
  )

  val writerParam          = new ReaderWriterParam(
    spatialBounds        = List(
      parsedArgs("axiDataWidth").toInt / parsedArgs("tcdmDataWidth").toInt
    ),
    temporalDimension    = (parsedXdmaCfg \ "writer_agu_temporal_dimension").as[Int],
    tcdmDataWidth        = parsedArgs("tcdmDataWidth").toInt,
    tcdmSize             = parsedArgs("tcdmSize").toInt,
    numChannel           = parsedArgs("axiDataWidth").toInt / parsedArgs("tcdmDataWidth").toInt,
    addressBufferDepth   = (parsedXdmaCfg \ "writer_buffer").as[Int],
    dataBufferDepth      = (parsedXdmaCfg \ "writer_buffer").as[Int],
    configurableChannel  = true,
    configurableByteMask = true,
    dynamicPriority      = false,
    higherStaticPriority = true
  )
  var readerExtensionParam = Seq[HasDataPathExtension]()
  var writerExtensionParam = Seq[HasDataPathExtension]()
  // Junctions (2->1 folds) are instantiated at the DATA SWITCH, not in an extension chain, and are configured
  // from the `writer_junctions` hjson object. They occupy a CSR region right after the writer extensions.
  var writerJunctionParam  = Seq[HasDataPathJunction]()

  // The following complex code is to dynamically load the extension modules
  // The target is that: 1) the sequence of the extension can be specified by the user 2) users can add their own extensions in the minimal effort (Does not need to modify the generation code)
  // The mechanism is that a small and temporary scala binary is compiled during the execution, to retrieve the instantiation object from the name
  // E.g. "HasMaxPool" -> HasMaxPool object
  // Thus, the generation function does not need to be modified by the extension developers.
  // Extension developers only need to 1) Add the Extension source code 2) Add Has...: #priority in hjson configuration file

  val toolbox = currentMirror.mkToolBox()

  def renderDatapathExtensionArgs(extensionName: String, args: JsValue): String =
    args match {
      case obj: JsObject =>
        obj.fields.map { case (paramName, paramValue) =>
          val renderedParam = paramValue.validate[Int] match {
            case JsSuccess(intValue, _) => intValue.toString
            case JsError(_)             =>
              paramValue.validate[Seq[Int]] match {
                case JsSuccess(seqValue, _) => s"Seq(${seqValue.mkString(",")})"
                case JsError(_)             =>
                  paramValue.validate[String] match {
                    case JsSuccess(strValue, _) => "\"" + strValue + "\""
                    case JsError(_)             =>
                      paramValue.validate[Seq[String]] match {
                        case JsSuccess(seqStr, _) =>
                          s"Seq(${seqStr.map(s => "\"" + s + "\"").mkString(",")})"
                        case JsError(_)           =>
                          throw new IllegalArgumentException(
                            s"Invalid XDMA datapath extension parameter $extensionName.$paramName: expected Int, Seq[Int], String, or Seq[String]"
                          )
                      }
                  }
              }
          }
          s"$paramName = $renderedParam"
        }
          .mkString(", ")
      case _ =>
        throw new IllegalArgumentException(
          s"Invalid XDMA datapath extension parameters for $extensionName: expected an object"
        )
    }

  // `asOpt` (not `as`) so a cfg that omits the section -- every cfg predating junctions -- yields an empty list
  // instead of throwing.
  def datapathExtensionParams(extensionSide: String): Seq[(String, String)] =
    (parsedXdmaCfg \ extensionSide).asOpt[JsObject] match {
      case Some(obj) =>
        obj.fields.filter { case (k, _) =>
          k.startsWith("Has")
        }.toSeq.map { case (k, v) =>
          (k, renderDatapathExtensionArgs(k, v))
        }
      case _         => Seq.empty
    }

  def instantiateDatapathExtension(extensionName: String, extensionArgs: String): HasDataPathExtension =
    toolbox
      .compile(toolbox.parse(s"""
import snax.DataPathExtension._
return new $extensionName($extensionArgs)
      """))()
      .asInstanceOf[HasDataPathExtension]

  def instantiateDatapathJunction(junctionName: String, junctionArgs: String): HasDataPathJunction =
    toolbox
      .compile(toolbox.parse(s"""
import snax.DataPathJunction._
return new $junctionName($junctionArgs)
      """))()
      .asInstanceOf[HasDataPathJunction]

  // Writer Side
  val writerDatapathExtensionParam = datapathExtensionParams("writer_extensions")

  writerDatapathExtensionParam.foreach { case (extensionName, extensionArgs) =>
    writerExtensionParam = writerExtensionParam :+ instantiateDatapathExtension(extensionName, extensionArgs)
  }

  // Reader Side
  val readerDatapathExtensionParam = datapathExtensionParams("reader_extensions")

  readerDatapathExtensionParam.foreach { case (extensionName, extensionArgs) =>
    readerExtensionParam = readerExtensionParam :+ instantiateDatapathExtension(extensionName, extensionArgs)
  }

  // Switch side: the junction bank (CHAINGATHER / GATHERROOT)
  val writerDatapathJunctionParam = datapathExtensionParams("writer_junctions")

  writerDatapathJunctionParam.foreach { case (junctionName, junctionArgs) =>
    writerJunctionParam = writerJunctionParam :+ instantiateDatapathJunction(junctionName, junctionArgs)
  }

  // SW-header-only mode (--sw-only): skip the RTL/CIRCT elaboration below. The
  // SW #define header further down is derived purely from the parsed
  // params/extensions and does not depend on the Chisel elaboration, so a
  // sw_only build can regenerate the header cheaply without emitting RTL.
  // Mirrors snax.streamer.StreamerSwHeaderGen.
  val swOnly = parsedArgs.contains("sw-only")

  // When a `--simdCfg` is supplied the cluster carries a SIMD block too, and BOTH blocks are emitted from this
  // one elaboration into a single `<cluster>_blocks.sv`. See ClusterBlocks for why they cannot be elaborated
  // separately. Without it, behaviour is exactly as before: XDMATop alone into `<cluster>_xdma.sv`.
  val simdCfgArg     = parsedArgs.find(_._1 == "simdCfg")
  val parsedSimdCfg  = simdCfgArg.map(a => Json.parse(a._2))

  // Generation of the hardware (skipped in --sw-only mode)
  if (!swOnly) {
    val xdmaReaderXParam = new XDMAParam(
      cfgParam,
      axiParam,
      crossClusterParam,
      readerParam,
      readerExtensionParam
    )
    val xdmaWriterXParam = new XDMAParam(
      cfgParam,
      axiParam,
      crossClusterParam,
      writerParam,
      writerExtensionParam,
      writerJunctionParam
    )

    var sv_string = parsedSimdCfg match {
      case Some(simdJson) =>
        getVerilogString(
          new ClusterBlocks(
            xdmaReaderParam = xdmaReaderXParam,
            xdmaWriterParam = xdmaWriterXParam,
            simdParam       = snax.simd.SimdCfgParser(
              simdJson,
              tcdmDataWidth = parsedArgs("tcdmDataWidth").toInt,
              axiDataWidth  = parsedArgs("axiDataWidth").toInt,
              tcdmSize      = parsedArgs("tcdmSize").toInt
            ),
            clusterName     = parsedArgs.getOrElse("clusterName", "")
          )
        )
      case None           =>
        getVerilogString(
          new XDMATop(
            clusterName = parsedArgs.getOrElse("clusterName", ""),
            readerParam = xdmaReaderXParam,
            writerParam = xdmaWriterXParam
          )
        )
    }

    // Perform dirty fix on the Chisel's bug that append the file list at the end of the file
    val truncated = sv_string
      .split("\n")
      .takeWhile(
        !_.contains(
          """// ----- 8< ----- FILE "firrtl_black_box_resource_files.f" ----- 8< -----"""
        )
      )

    // CIRCT inlines blackbox resource files (e.g. fpnew_pkg_snax + fp_add/fp_mul/fp_fma used by the FP
    // SIMD extensions) AFTER the main circuit that instantiates them. SystemVerilog packages must be
    // declared before use, so hoist the inlined resource block (from the first "----- 8< ----- FILE"
    // marker onward) ahead of the main circuit. The resources keep their own dependency order
    // (package -> classifier/rounding/lzc -> fp_* modules).
    val firstResIdx = truncated.indexWhere(_.contains("// ----- 8< ----- FILE"))
    sv_string =
      if (firstResIdx >= 0)
        (truncated.drop(firstResIdx) ++ truncated.take(firstResIdx)).mkString("\n")
      else
        truncated.mkString("\n")

    // Write the sv_string to the SystemVerilog file
    val hardware_dir = parsedArgs.getOrElse(
      "hw-target-dir",
      "generated"
    ) + "/" + s"${parsedArgs.getOrElse("clusterName", "")}" + (if (parsedSimdCfg.isDefined) "_blocks.sv" else "_xdma.sv")
    java.nio.file.Files.write(
      java.nio.file.Paths.get(hardware_dir),
      sv_string.getBytes(java.nio.charset.StandardCharsets.UTF_8)
    )
  }

  // Generation of the software #define macros
  val macro_dir = parsedArgs.getOrElse(
    "sw-target-dir",
    "generated/include/xdma-addr.h"
  )

  var macro_template =
    s"""// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

// This file is automatically generated by XDMA hardware generator corresponding to hardware configuration. 
// Do not modify or commit this file to the repository manually.

#define XDMA_WIDTH ${writerParam.tcdmParam.numChannel * writerParam.tcdmParam.dataWidth / 8}
#define XDMA_SPATIAL_CHAN ${writerParam.tcdmParam.numChannel}

// The base address region of the XDMA
#define XDMA_SRC_ADDR_PTR_LSB 0
#define XDMA_SRC_ADDR_PTR_MSB XDMA_SRC_ADDR_PTR_LSB + 1
#define XDMA_MAX_DST_COUNT ${crossClusterParam.maxMulticastDest}
#define XDMA_DST_ADDR_PTR_LSB XDMA_SRC_ADDR_PTR_MSB + 1
#define XDMA_DST_ADDR_PTR_MSB XDMA_DST_ADDR_PTR_LSB + 1

// The stride and bound region of the reader of XDMA
#define XDMA_SRC_TEMP_DIM ${crossClusterParam.maxTemporalDimension}
#define XDMA_SRC_SPATIAL_STRIDE_PTR XDMA_DST_ADDR_PTR_LSB + XDMA_MAX_DST_COUNT * 2
#define XDMA_SRC_TEMP_BOUND_PTR XDMA_SRC_SPATIAL_STRIDE_PTR + 1
#define XDMA_SRC_TEMP_STRIDE_PTR XDMA_SRC_TEMP_BOUND_PTR + XDMA_SRC_TEMP_DIM

// The channel and strobe region of the reader of XDMA
#define XDMA_SRC_ENABLED_CHAN_PTR XDMA_SRC_TEMP_STRIDE_PTR + XDMA_SRC_TEMP_DIM
#define XDMA_SRC_ENABLE_PTR XDMA_SRC_ENABLED_CHAN_PTR + ${if (readerParam.configurableChannel) 1
      else 0}
#define XDMA_SRC_EXT_NUM ${readerExtensionParam.length}
#define XDMA_SRC_EXT_CSR_PTR XDMA_SRC_ENABLE_PTR + ${if (readerExtensionParam.length > 0) 1
      else 0}
#define XDMA_SRC_EXT_CSR_NUM ${readerExtensionParam
        .map(_.extensionParam.userCsrNum)
        .sum}
#define XDMA_SRC_EXT_CUSTOM_CSR_NUM \\
    { ${readerExtensionParam.map(_.extensionParam.userCsrNum).mkString(", ")} }

// The stride and bound region of the writer of XDMA
#define XDMA_DST_TEMP_DIM ${crossClusterParam.maxTemporalDimension}
#define XDMA_DST_SPATIAL_STRIDE_PTR XDMA_SRC_EXT_CSR_PTR + XDMA_SRC_EXT_CSR_NUM
#define XDMA_DST_TEMP_BOUND_PTR XDMA_DST_SPATIAL_STRIDE_PTR + 1
#define XDMA_DST_TEMP_STRIDE_PTR XDMA_DST_TEMP_BOUND_PTR + XDMA_DST_TEMP_DIM

#define XDMA_DST_ENABLED_CHAN_PTR XDMA_DST_TEMP_STRIDE_PTR + XDMA_DST_TEMP_DIM
#define XDMA_DST_ENABLED_BYTE_PTR XDMA_DST_ENABLED_CHAN_PTR + ${if (writerParam.configurableChannel) 1
      else 0}
#define XDMA_DST_ENABLE_PTR XDMA_DST_ENABLED_BYTE_PTR + ${if (writerParam.configurableByteMask) 1
      else 0}
#define XDMA_DST_EXT_NUM ${writerExtensionParam.length}
#define XDMA_DST_EXT_CSR_PTR XDMA_DST_ENABLE_PTR + ${if (writerExtensionParam.length > 0) 1
      else 0}
#define XDMA_DST_EXT_CSR_NUM ${writerExtensionParam
        .map(_.extensionParam.userCsrNum)
        .sum}
#define XDMA_DST_EXT_CUSTOM_CSR_NUM \\
    { ${writerExtensionParam.map(_.extensionParam.userCsrNum).mkString(", ")} }

// The junction region of the data switch (2->1 collective folds). Laid out exactly like the extension region --
// one enable bitmask followed by the per-junction user CSRs -- immediately after the writer extensions, so with no
// junction configured every pointer below collapses to the pre-junction layout.
#define XDMA_DST_JCT_NUM ${writerJunctionParam.length}
#define XDMA_DST_JCT_ENABLE_PTR XDMA_DST_EXT_CSR_PTR + XDMA_DST_EXT_CSR_NUM
#define XDMA_DST_JCT_CSR_PTR XDMA_DST_JCT_ENABLE_PTR + ${if (writerJunctionParam.length > 0) 1
      else 0}
#define XDMA_DST_JCT_CSR_NUM ${writerJunctionParam
        .map(_.junctionParam.userCsrNum)
        .sum}
#define XDMA_DST_JCT_CUSTOM_CSR_NUM \\
    { ${writerJunctionParam.map(_.junctionParam.userCsrNum).mkString(", ")} }
#define XDMA_START_PTR XDMA_DST_JCT_CSR_PTR + XDMA_DST_JCT_CSR_NUM
#define XDMA_COMMIT_LOCAL_TASK_PTR XDMA_START_PTR + 1
#define XDMA_COMMIT_REMOTE_TASK_PTR XDMA_COMMIT_LOCAL_TASK_PTR + 1
#define XDMA_FINISH_LOCAL_TASK_PTR XDMA_COMMIT_REMOTE_TASK_PTR + 1
#define XDMA_FINISH_REMOTE_TASK_PTR XDMA_FINISH_LOCAL_TASK_PTR + 1
#define XDMA_PERF_CTR_TASK XDMA_FINISH_REMOTE_TASK_PTR + 1
#define XDMA_PERF_CTR_READER XDMA_PERF_CTR_TASK + 1
#define XDMA_PERF_CTR_WRITER XDMA_PERF_CTR_READER + 1
// Junction status, sticky since the last start: [0] the armed operator could not honour its
// configuration word (O5); [1] the join starved; [2] a junction is busy right now.
#define XDMA_JCT_STATUS XDMA_PERF_CTR_WRITER + 1
"""

  // Append CSR Extension Information in to Macro
  macro_template = macro_template + """
// Extension Information
"""

  for ((ext, i) <- readerExtensionParam.zipWithIndex) {
    macro_template = macro_template +
      s"""#define READER_EXT_${ext.extensionParam.moduleName.toUpperCase} ${i}
"""
  }

  for ((ext, i) <- writerExtensionParam.zipWithIndex) {
    macro_template = macro_template +
      s"""#define WRITER_EXT_${ext.extensionParam.moduleName.toUpperCase} ${i}
"""
  }

  for ((jct, i) <- writerJunctionParam.zipWithIndex) {
    macro_template = macro_template +
      s"""#define WRITER_JCT_${jct.junctionParam.moduleName.toUpperCase} ${i}
"""
  }

  // Ensure the directory exists before writing the file
  val macro_dir_path   = java.nio.file.Paths.get(macro_dir)
  val macro_dir_parent = macro_dir_path.getParent
  if (macro_dir_parent != null && !java.nio.file.Files.exists(macro_dir_parent)) {
    java.nio.file.Files.createDirectories(macro_dir_parent)
  }
  java.nio.file.Files.write(
    macro_dir_path,
    macro_template.getBytes(java.nio.charset.StandardCharsets.UTF_8)
  )
}
