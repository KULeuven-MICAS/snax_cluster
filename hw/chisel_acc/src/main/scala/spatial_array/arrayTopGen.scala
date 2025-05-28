package snax_acc.spatial_array

object SpatialArrayParamParser {
  def parseFromHjsonString(hjsonStr: String): SpatialArrayParam = {
    val cfg = ujson.read(hjsonStr)

    def getSeqInt(key: String): Seq[Int] = cfg(key).arr.map(_.num.toInt).toSeq

    def get3DSeq(key: String): Seq[Seq[Seq[Int]]] =
      cfg(key).arr.map(_.arr.map(_.arr.map(_.num.toInt).toSeq).toSeq).toSeq

    require(
      cfg.obj.get("snax_num_rw_csr").map(_.num.toInt) == Some(7),
      "snax_num_rw_csr should be 7 for OpenGeMM"
    )

    SpatialArrayParam(
      opType                 = cfg("snax_opengemm_op_type").arr.map(v => OpType.fromString(v.str)).toSeq,
      macNum                 = getSeqInt("snax_opengemm_mac_num"),
      inputAElemWidth        = getSeqInt("snax_opengemm_input_a_element_width"),
      inputBElemWidth        = getSeqInt("snax_opengemm_input_b_element_width"),
      inputCElemWidth        = getSeqInt("snax_opengemm_output_element_width"), // you can adjust if different
      mulElemWidth           = getSeqInt("snax_opengemm_multiply_element_width"),
      outputDElemWidth       = getSeqInt("snax_opengemm_output_element_width"),
      arrayInputAWidth       = cfg("snax_opengemm_array_input_a_width").num.toInt,
      arrayInputBWidth       = cfg("snax_opengemm_array_input_b_width").num.toInt,
      arrayInputCWidth       = cfg("snax_opengemm_array_input_c_width").num.toInt,
      arrayOutputDWidth      = cfg("snax_opengemm_array_output_width").num.toInt,
      arrayDim               = get3DSeq("snax_opengemm_spatial_unrolling"),
      serialInputADataWidth  = cfg("snax_opengemm_serial_a_width").num.toInt,
      serialInputBDataWidth  = cfg("snax_opengemm_serial_b_width").num.toInt,
      serialInputCDataWidth  = cfg("snax_opengemm_serial_c_d_width").num.toInt,
      serialOutputDDataWidth = cfg("snax_opengemm_serial_c_d_width").num.toInt,
      adderTreeDelay         = cfg("snax_opengemm_adder_tree_delay").num.toInt
    )
  }
}

object ArrayTopGen {
  def main(args: Array[String]): Unit = {

    // 16x8x8, 1x32x32 ISSCC reproduce
    // generation time: ~1sec
    // val params = SpatialArrayParam(
    //     opType = Seq(OpType.SIntSIntOp),
    //     macNum = Seq(1024),
    //     inputAElemWidth = Seq(8),
    //     inputBElemWidth = Seq(8),
    //     inputCElemWidth = Seq(32),
    //     mulElemWidth = Seq(16),
    //     outputDElemWidth = Seq(32),
    //     arrayInputAWidth = 1024,
    //     arrayInputBWidth = 8192,
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
    //     outputDElemWidth = Seq(32),
    //     arrayInputAWidth = 2048,
    //     arrayInputBWidth = 2048,
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
    //     outputDElemWidth = Seq(32),
    //     arrayInputAWidth = 1024,
    //     arrayInputBWidth = 1024,
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
    //     outputDElemWidth = Seq(32),
    //     arrayInputAWidth = 512,
    //     arrayInputBWidth = 1024,
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
    //     outputDElemWidth = Seq(32),
    //     arrayInputAWidth = 512,
    //     arrayInputBWidth = 512,
    //     arrayInputCWidth = 131072,
    //     arrayOutputDWidth = 131072,
    //     arrayDim = Seq(Seq(Seq(64, 1, 64)))
    // )

    // Bitwave like reproduce
    // 7 dfs
    // val params = SpatialArrayParam(
    //   opType            = Seq(OpType.SIntSIntOp),
    //   macNum            = Seq(4096),
    //   inputAElemWidth   = Seq(8),
    //   inputBElemWidth   = Seq(8),
    //   inputCElemWidth   = Seq(32),
    //   mulElemWidth      = Seq(16),
    //   outputDElemWidth      = Seq(32),
    //   arrayInputAWidth       = 1024,
    //   arrayInputBWidth       = 8192,
    //   arrayInputCWidth  = 16384,
    //   arrayOutputDWidth = 16384,
    //   arrayDim          = Seq(
    //     Seq(
    //       Seq(16, 8, 32),
    //       Seq(8, 16, 32),
    //       Seq(4, 32, 32),
    //       Seq(1, 8, 128),
    //       Seq(1, 16, 64),
    //       Seq(1, 32, 32),
    //       Seq(2, 1, 64)
    //     )
    //   )
    // )
    // val tag    = "BitWave"

    // opengemm
    // 8x8x8
    // val params = SpatialArrayParam(
    //   opType            = Seq(OpType.SIntSIntOp),
    //   macNum            = Seq(512),
    //   inputAElemWidth   = Seq(8),
    //   inputBElemWidth   = Seq(8),
    //   inputCElemWidth   = Seq(32),
    //   mulElemWidth      = Seq(16),
    //   outputDElemWidth  = Seq(32),
    //   arrayInputAWidth  = 512,
    //   arrayInputBWidth  = 512,
    //   arrayInputCWidth  = 2048,
    //   arrayOutputDWidth = 2048,
    //   arrayDim          = Seq(
    //     Seq(
    //       Seq(8, 8, 8)
    //     )
    //   )
    // )
    // val tag    = "OpenGeMM"

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
          "Usage: --openGeMMCfg <hjson string> --hw-target-dir <output dir>"
        )
        sys.exit(1)
      }
      parsed_args
    }

    // Parse the command-line arguments
    val parsedArgs = parseArgs(args)

    val outPath = parsedArgs.getOrElse(
      "hw-target-dir",
      "generated/SpatialArray"
    )

    val openGeMMCfg = parsedArgs.find(_._1 == "openGeMMCfg").get._2

    val params = SpatialArrayParamParser.parseFromHjsonString(openGeMMCfg)
    parsedArgs.getOrElse(
      "tag",
      "default"
    )

    // generate verilog file
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      new ArrayTop(params),
      Array(
        "--target-dir",
        outPath
      )
    )

    // generate sv wrapper file
    var macro_template = ""
    val macro_dir      = s"$outPath/snax_opengemm_shell_wrapper.sv"
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
    val DataWidthA     = params.arrayInputAWidth
    val DataWidthB     = params.arrayInputBWidth
    val DataWidthC     = params.serialInputCDataWidth
    val DataWidthD     = params.serialOutputDDataWidth

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
