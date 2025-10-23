// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi <xiaoling.yi@kuleuven.be>

package snax_acc.versacore

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

import chisel3._

import fp_unit._

// hjson configuration parser, from hjson to SpatialArrayParam
object SpatialArrayParamParser {
  def parseFromHjsonString(hjsonStr: String): SpatialArrayParam = {
    val cfg = ujson.read(hjsonStr)

    def getSeqInt(key: String): Seq[Int] = cfg(key).arr.map(_.num.toInt).toSeq

    def get3DSeq(key: String): Seq[Seq[Seq[Int]]] =
      cfg(key).arr.map(_.arr.map(_.arr.map(_.num.toInt).toSeq).toSeq).toSeq

    require(
      cfg.obj.get("snax_num_rw_csr").map(_.num.toInt) == Some(7),
      "snax_num_rw_csr should be 7 for VersaCore"
    )

    /** Convert input widths to corresponding FP types
      */
    def widthToFpType(width: Int) = {
      width match {
        case 8  => FP8
        case 16 => FP16
        case 32 => FP32
        case _  => throw new NotImplementedError()
      }
    }

    /** Convert input widths to corresponding types */
    def widthToType(width: Int, dataTypeStr: String): DataType = {
      dataTypeStr match {
        case "SInt"  => new IntType(width)
        case "Float" => widthToFpType(width)
        case _       => throw new NotImplementedError()
      }
    }

    /** Convert a sequence of widths and string to a sequence of DataTypes */
    def widthSeqToTypeSeq(widthSeq: Seq[Int], dataTypeStr: Seq[String]): Seq[DataType] =
      widthSeq.zip(dataTypeStr).map { case (a, b) => widthToType(a, b) }

    SpatialArrayParam(
      macNum                 = getSeqInt("snax_versacore_mac_num"),
      inputTypeA             = widthSeqToTypeSeq(
        getSeqInt("snax_versacore_input_a_element_width"),
        cfg("snax_versacore_input_a_data_type").arr.map(_.str).toSeq
      ),
      inputTypeB             = widthSeqToTypeSeq(
        getSeqInt("snax_versacore_input_b_element_width"),
        cfg("snax_versacore_input_b_data_type").arr.map(_.str).toSeq
      ),
      inputTypeC             = widthSeqToTypeSeq(
        getSeqInt("snax_versacore_input_c_element_width"),
        cfg("snax_versacore_input_c_data_type").arr.map(_.str).toSeq
      ),
      outputTypeD            = widthSeqToTypeSeq(
        getSeqInt("snax_versacore_output_d_element_width"),
        cfg("snax_versacore_output_d_data_type").arr.map(_.str).toSeq
      ),
      arrayInputAWidth       = cfg("snax_versacore_array_input_a_width").num.toInt,
      arrayInputBWidth       = cfg("snax_versacore_array_input_b_width").num.toInt,
      arrayInputCWidth       = cfg("snax_versacore_array_input_c_width").num.toInt,
      arrayOutputDWidth      = cfg("snax_versacore_array_output_d_width").num.toInt,
      arrayDim               = get3DSeq("snax_versacore_spatial_unrolling"),
      serialInputADataWidth  = cfg("snax_versacore_serial_a_width").num.toInt,
      serialInputBDataWidth  = cfg("snax_versacore_serial_b_width").num.toInt,
      serialInputCDataWidth  = cfg("snax_versacore_serial_c_d_width").num.toInt,
      serialOutputDDataWidth = cfg("snax_versacore_serial_c_d_width").num.toInt,
      adderTreeDelay         = cfg("snax_versacore_adder_tree_delay").num.toInt,
      dataflow               = cfg("snax_versacore_temporal_unrolling").arr.map(_.str).toSeq
    )
  }
}

// main object to generate the VersaCore module and the wrapper
object VersaCoreGen {
  def main(args: Array[String]): Unit = {

    // parameters parser
    // Helper function to parse command-line arguments into a Map
    def parseArgs(args: Array[String]): Map[String, String] = {
      val parsed_args = args
        .sliding(2, 2)
        .collect {
          case Array(key, value) if key.startsWith("--") => key.drop(2) -> value
        }
        .toMap
      if (parsed_args.size != 2) {
        println(
          "Usage: --versacoreCfg <hjson string> --hw-target-dir <output dir>"
        )
        sys.exit(1)
      }
      parsed_args
    }

    // Parse the command-line arguments
    val parsedArgs = parseArgs(args)

    val outPath = parsedArgs.getOrElse(
      "hw-target-dir",
      "generated/versacore"
    )

    val versacoreCfg = parsedArgs.find(_._1 == "versacoreCfg").get._2

    val params = SpatialArrayParamParser.parseFromHjsonString(versacoreCfg)
    parsedArgs.getOrElse(
      "tag",
      "default"
    )

    // Step 1: Get the SystemVerilog string
    var sv_string = getVerilogString(new VersaCore(params))

    // Step 2: Remove the FIRRTL file list footer
    sv_string = sv_string
      .split("\n")
      .takeWhile(
        !_.contains(
          """// ----- 8< ----- FILE "firrtl_black_box_resource_files.f" ----- 8< -----"""
        )
      )
      .mkString("\n")

    // Step 3: Reorder the package if needed
    val lines: Array[String] = sv_string.split("\n")

    // Find package block range
    val startIdx = lines.indexWhere(_.contains("package fpnew_pkg_snax"))

    if (startIdx != -1) {
      val endIdx = lines.indexWhere(_.trim == "endpackage", startIdx)
      if (endIdx != -1 && endIdx > startIdx) {
        val pkgBlock       = lines.slice(startIdx, endIdx + 1)
        val remainingLines = lines.take(startIdx) ++ lines.drop(endIdx + 1)
        sv_string = (pkgBlock ++ remainingLines).mkString("\n")
      }
    }

    // Step 4: Write to file
    val outFile = Paths.get(s"$outPath/VersaCore.sv")
    Files.write(outFile, sv_string.getBytes(StandardCharsets.UTF_8))

    // generate sv wrapper file
    var macro_template = ""
    val macro_dir      = s"$outPath/snax_versacore_shell_wrapper.sv"
    val header         = s"""// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
// This file is generated by VersaCore module in hw/chisel_acc to wrap the generated verilog code automatically, do not modify it manually
// Generated at ${java.time.Instant.now()}

//-------------------------------
// Accelerator wrapper
//-------------------------------
"""
    val DataWidthA     = params.arrayInputAWidth
    val DataWidthB     = params.arrayInputBWidth
    val DataWidthC     = params.serialInputCDataWidth
    val DataWidthD     = params.serialOutputDDataWidth

    macro_template = header + s"""
module snax_versacore_shell_wrapper #(
    // Custom parameters. As much as possible,
    // these parameters should not be taken from outside
    parameter int unsigned RegRWCount   = ${params.csrNum},
    parameter int unsigned RegROCount   = 2,
    parameter int unsigned DataWidthA   = $DataWidthA,
    parameter int unsigned DataWidthB   = $DataWidthB,
    parameter int unsigned DataWidthC   = $DataWidthC,
    parameter int unsigned DataWidthD = $DataWidthD,
    parameter int unsigned RegDataWidth = 32,
    parameter int unsigned RegAddrWidth = 32
) (
    //-------------------------------
    // Clocks and reset
    //-------------------------------
    input logic clk_i,
    input logic rst_ni,

    //-------------------------------
    // Accelerator ports
    //-------------------------------
    // Note, we maintained the form of these signals
    // just to comply with the top-level wrapper

    // Ports from accelerator to streamer
    output logic [DataWidthD-1:0] acc2stream_0_data_o,
    output logic acc2stream_0_valid_o,
    input logic acc2stream_0_ready_i,

    // Ports from streamer to accelerator
    input logic [DataWidthA-1:0] stream2acc_0_data_i,
    input logic stream2acc_0_valid_i,
    output logic stream2acc_0_ready_o,

    input logic [DataWidthB-1:0] stream2acc_1_data_i,
    input logic stream2acc_1_valid_i,
    output logic stream2acc_1_ready_o,

    input logic [DataWidthC-1:0] stream2acc_2_data_i,
    input logic stream2acc_2_valid_i,
    output logic stream2acc_2_ready_o,

    //-------------------------------
    // CSR manager ports
    //-------------------------------
    input  logic [RegRWCount-1:0][RegDataWidth-1:0] csr_reg_set_i,
    input  logic                                    csr_reg_set_valid_i,
    output logic                                    csr_reg_set_ready_o,
    output logic [RegROCount-1:0][RegDataWidth-1:0] csr_reg_ro_set_o
);
  assign csr_reg_ro_set_o[0][31:1] = 0;

  VersaCore inst_VersaCore (
      .clock(clk_i),
      .reset(~rst_ni),

      .io_data_in_a_ready(stream2acc_0_ready_o),
      .io_data_in_a_valid(stream2acc_0_valid_i),
      .io_data_in_a_bits (stream2acc_0_data_i),

      .io_data_in_b_ready(stream2acc_1_ready_o),
      .io_data_in_b_valid(stream2acc_1_valid_i),
      .io_data_in_b_bits (stream2acc_1_data_i),

      .io_data_in_c_ready(stream2acc_2_ready_o),
      .io_data_in_c_valid(stream2acc_2_valid_i),
      .io_data_in_c_bits (stream2acc_2_data_i),

      .io_data_out_d_ready(acc2stream_0_ready_i),
      .io_data_out_d_valid(acc2stream_0_valid_o),
      .io_data_out_d_bits (acc2stream_0_data_o),

      .io_ctrl_ready(csr_reg_set_ready_o),
      .io_ctrl_valid(csr_reg_set_valid_i),
      .io_ctrl_bits_fsmCfg_take_in_new_c(csr_reg_set_i[0]),
      .io_ctrl_bits_fsmCfg_a_b_input_times_one_output(csr_reg_set_i[1]),
      .io_ctrl_bits_fsmCfg_output_times(csr_reg_set_i[2]),
      .io_ctrl_bits_fsmCfg_subtraction_constant_i(csr_reg_set_i[3]),
      .io_ctrl_bits_arrayCfg_arrayShapeCfg(csr_reg_set_i[4]),
      .io_ctrl_bits_arrayCfg_dataTypeCfg(csr_reg_set_i[5]),

      .io_busy_o(csr_reg_ro_set_o[0][0]),
      .io_performance_counter(csr_reg_ro_set_o[1])

  );

endmodule
"""

    java.nio.file.Files.write(
      java.nio.file.Paths.get(macro_dir),
      macro_template.getBytes(java.nio.charset.StandardCharsets.UTF_8)
    )

    println(s"Generated macro file: $macro_dir")

    // generate the c lib header file
    val headerFile = s"$outPath/../../sw/snax/versacore/include/snax_versacore_stationarity.h"

    var headerContent = s"""// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
"""

    if (params.dataflow.contains("output_stationary") && params.dataflow.length == 1) {
      headerContent += s"""
#define SNAX_VERSACORE_OUTPUT_STATIONARY_ONLY
"""
    } else {
      headerContent += s"""#define SNAX_VERSACORE_MULTI_STATIONARY
"""
    }

    val path: Path = Paths.get(headerFile)
    val parentDir = path.getParent
    if (parentDir != null && !Files.exists(parentDir)) {
      Files.createDirectories(parentDir)
    }

    java.nio.file.Files.write(
      java.nio.file.Paths.get(headerFile),
      headerContent.getBytes(java.nio.charset.StandardCharsets.UTF_8)
    )

    println(s"Generated header file: $headerFile")

  }
}
