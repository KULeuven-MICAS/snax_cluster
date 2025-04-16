package snax_acc.spatial_array

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import scala.sys.process._

object ArrayTopGen {
  def main(args: Array[String]): Unit = {
    // Current time with default format
    var currentTime = LocalDateTime.now()
    println(s"Current time (default): $currentTime")

    // Formatted output (e.g., "2024-01-23 14:30:15")
    var formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
    println(s"Formatted time: ${currentTime.format(formatter)}")

    // 16x8x8, 1x32x32 ISSCC reproduce
    // generation time: ~1sec
    // val params = SpatialArrayParam(
    //     opType = Seq(OpType.SIntSIntOp),
    //     macNum = Seq(1024),
    //     inputAElemWidth = Seq(8),
    //     inputBElemWidth = Seq(8),
    //     inputCElemWidth = Seq(32),
    //     mulElemWidth = Seq(16),
    //     outElemWidth = Seq(32),
    //     inputAWidth = 1024,
    //     inputBWidth = 8192,
    //     arrayInputCWidth = 4096,
    //     arrayOutputDWidth = 4096,
    //     arrayDim = Seq(Seq(Seq(16, 8, 8), Seq(1, 32, 32)))
    // )
    // val tag = "ISSCC"

    // ----------------------------------------------
    // scalability test
    // ----------------------------------------------
    // 256x256 TPU like reproduce
    // val params = SpatialArrayParam(
    //     opType = Seq(OpType.SIntSIntOp),
    //     macNum = Seq(65536),
    //     inputAElemWidth = Seq(8),
    //     inputBElemWidth = Seq(8),
    //     inputCElemWidth = Seq(32),
    //     mulElemWidth = Seq(16),
    //     outElemWidth = Seq(32),
    //     inputAWidth = 2048,
    //     inputBWidth = 2048,
    //     arrayInputCWidth = 2097152,
    //     arrayOutputDWidth = 2097152,
    //     arrayDim = Seq(Seq(Seq(256, 1, 256)))
    // )

    // // 128x128 TPU like reproduce
    // val params = SpatialArrayParam(
    //     opType = Seq(OpType.SIntSIntOp),
    //     macNum = Seq(16384),
    //     inputAElemWidth = Seq(8),
    //     inputBElemWidth = Seq(8),
    //     inputCElemWidth = Seq(32),
    //     mulElemWidth = Seq(16),
    //     outElemWidth = Seq(32),
    //     inputAWidth = 1024,
    //     inputBWidth = 1024,
    //     arrayInputCWidth = 524288,
    //     arrayOutputDWidth = 524288,
    //     arrayDim = Seq(Seq(Seq(128, 1, 128)))
    // )

    // 64x128 TPU like reproduce
    // generation time: ~40mins
    // val params = SpatialArrayParam(
    //     opType = Seq(OpType.SIntSIntOp),
    //     macNum = Seq(8192),
    //     inputAElemWidth = Seq(8),
    //     inputBElemWidth = Seq(8),
    //     inputCElemWidth = Seq(32),
    //     mulElemWidth = Seq(16),
    //     outElemWidth = Seq(32),
    //     inputAWidth = 512,
    //     inputBWidth = 1024,
    //     arrayInputCWidth = 262144,
    //     arrayOutputDWidth = 262144,
    //     arrayDim = Seq(Seq(Seq(64, 1, 128)))
    // )

    // 64x64 TPU like reproduce
    // generation time: ~4mins
    // val params = SpatialArrayParam(
    //     opType = Seq(OpType.SIntSIntOp),
    //     macNum = Seq(4096),
    //     inputAElemWidth = Seq(8),
    //     inputBElemWidth = Seq(8),
    //     inputCElemWidth = Seq(32),
    //     mulElemWidth = Seq(16),
    //     outElemWidth = Seq(32),
    //     inputAWidth = 512,
    //     inputBWidth = 512,
    //     arrayInputCWidth = 131072,
    //     arrayOutputDWidth = 131072,
    //     arrayDim = Seq(Seq(Seq(64, 1, 64)))
    // )

    // Bitwave like reproduce
    // 7 dfs
    val params = SpatialArrayParam(
      opType            = Seq(OpType.SIntSIntOp),
      macNum            = Seq(4096),
      inputAElemWidth   = Seq(8),
      inputBElemWidth   = Seq(8),
      inputCElemWidth   = Seq(32),
      mulElemWidth      = Seq(16),
      outElemWidth      = Seq(32),
      inputAWidth       = 1024,
      inputBWidth       = 8192,
      arrayInputCWidth  = 16384,
      arrayOutputDWidth = 16384,
      arrayDim          = Seq(
        Seq(
          Seq(16, 8, 32),
          Seq(8, 16, 32),
          Seq(4, 32, 32),
          Seq(1, 8, 128),
          Seq(1, 16, 64),
          Seq(1, 32, 32),
          Seq(2, 1, 64)
        )
      )
    )
    val tag    = "BitWave"

    println("arayDim: " + params.arrayDim.toString())

    // generate verilog file
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      new ArrayTop(params),
      Array(
        "--target-dir",
        "generated/SpatialArray"
      )
      // Array(
      //   "--split-verilog",
      //   s"-o=generated/SpatialArray/ArrayTop_${tag}"
      // )
    )

    s"cp generated/SpatialArray/ArrayTop.sv generated/SpatialArray/ArrayTop_${tag}.sv".!

    currentTime = LocalDateTime.now()
    println(s"Current time (default): $currentTime")
    formatter   = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
    println(s"Formatted time: ${currentTime.format(formatter)}")

    // generate sv wrapper file
    var macro_template = ""
    val macro_dir      = "./src/snax_opengemm_shell_wrapper.sv"
    val header         = s"""// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
// This file is generated by ArrayTop module in hw/chisel_acc to wrap the generated verilog code automatically, do not modify it manually
// Generated at ${java.time.Instant.now()}

//-------------------------------
// Accelerator wrapper
//-------------------------------
"""
    val DataWidthA     = params.inputAWidth
    val DataWidthB     = params.inputBWidth
    val DataWidthC     = params.inputCSerialDataWidth
    val DataWidthD     = params.outputDSerialDataWidth

    macro_template = header + s"""
module snax_opengemm_shell_wrapper #(
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

  ArrayTop inst_ArrayTop (
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
      .io_ctrl_bits_fsmCfg_K_i(csr_reg_set_i[0]),
      .io_ctrl_bits_fsmCfg_N_i(csr_reg_set_i[1]),
      .io_ctrl_bits_fsmCfg_M_i(csr_reg_set_i[2]),
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

  }
}
