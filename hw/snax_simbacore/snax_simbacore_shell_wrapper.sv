// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
// 
// Author: Robin Geens <robin.geens@kuleuven.be>


//-------------------------------
// Accelerator wrapper
//-------------------------------

module snax_simbacore_shell_wrapper #(
    // NOTE these parameters can be set automatically by running update_simbacore_params.py
    // Acc2stream
    parameter int unsigned OSCoreOutDWidth               = 64,   // 1
    parameter int unsigned SUCoreOutYWidth               = 64,   // 1
    parameter int unsigned SwitchCoreOutWidth            = 64,   // 1
    parameter int unsigned ISCoreOutDWidth               = 256,  // 4
    // Stream2acc
    parameter int unsigned OSCoreInAWidth                = 256,  // 4
    parameter int unsigned OSCoreInBWidth                = 384,  // 6
    //  
    parameter int unsigned SwitchCoreInMatmulWidth       = 128,  // 2
    parameter int unsigned SwitchCoreInWeightWidth       = 64,   // 1
    parameter int unsigned SwitchCoreInBiasWidth         = 64,   // 1
    parameter int unsigned SwitchCoreInMatmulWeightWidth = 64,   // 1
    // 
    parameter int unsigned SUCoreInAWidth                = 64,   // 1
    parameter int unsigned SUCoreInBCWidth               = 512,  // 8
    parameter int unsigned SUCoreInDWidth                = 64,   // 1
    parameter int unsigned SUCoreInXWidth                = 64,   // 1
    parameter int unsigned SUCoreInZWidth                = 64,   // 1
    //
    parameter int unsigned ISCoreInAWidth                = 64,   // 1
    parameter int unsigned ISCoreInBWidth                = 384,  // 6
    parameter int unsigned ISCoreInCWidth                = 256,  // 4
    // CSR
    parameter int unsigned RegRWCount                    = 7,    // +1 for start csr
    parameter int unsigned RegROCount                    = 4,
    parameter int unsigned RegDataWidth                  = 32,
    parameter int unsigned RegAddrWidth                  = 32
) (
    //-------------------------------
    // Clocks and reset
    //-------------------------------
    input logic clk_i,
    input logic rst_ni,

    //-------------------------------
    // Accelerator ports
    //-------------------------------
    // Note, we maintained the form of these signals just to comply with the top-level wrapper

    // Ports from accelerator to streamer
    output logic [   OSCoreOutDWidth-1:0] acc2stream_0_data_o,   // W0
    output logic                          acc2stream_0_valid_o,
    input  logic                          acc2stream_0_ready_i,
    output logic [SwitchCoreOutWidth-1:0] acc2stream_1_data_o,   // W1
    output logic                          acc2stream_1_valid_o,
    input  logic                          acc2stream_1_ready_i,
    output logic [   SUCoreOutYWidth-1:0] acc2stream_2_data_o,   // W2
    output logic                          acc2stream_2_valid_o,
    input  logic                          acc2stream_2_ready_i,
    output logic [   ISCoreOutDWidth-1:0] acc2stream_3_data_o,   // W3
    output logic                          acc2stream_3_valid_o,
    input  logic                          acc2stream_3_ready_i,

    // Ports from streamer to accelerator
    input  logic [               OSCoreInAWidth-1:0] stream2acc_0_data_i,    // R0
    input  logic                                     stream2acc_0_valid_i,
    output logic                                     stream2acc_0_ready_o,
    input  logic [               OSCoreInBWidth-1:0] stream2acc_1_data_i,    // R1
    input  logic                                     stream2acc_1_valid_i,
    output logic                                     stream2acc_1_ready_o,
    input  logic [      SwitchCoreInMatmulWidth-1:0] stream2acc_2_data_i,    // R2
    input  logic                                     stream2acc_2_valid_i,
    output logic                                     stream2acc_2_ready_o,
    input  logic [      SwitchCoreInWeightWidth-1:0] stream2acc_3_data_i,    // R3
    input  logic                                     stream2acc_3_valid_i,
    output logic                                     stream2acc_3_ready_o,
    input  logic [        SwitchCoreInBiasWidth-1:0] stream2acc_4_data_i,    // R4
    input  logic                                     stream2acc_4_valid_i,
    output logic                                     stream2acc_4_ready_o,
    input  logic [SwitchCoreInMatmulWeightWidth-1:0] stream2acc_5_data_i,    // R5
    input  logic                                     stream2acc_5_valid_i,
    output logic                                     stream2acc_5_ready_o,
    input  logic [               SUCoreInAWidth-1:0] stream2acc_6_data_i,    // R6
    input  logic                                     stream2acc_6_valid_i,
    output logic                                     stream2acc_6_ready_o,
    input  logic [              SUCoreInBCWidth-1:0] stream2acc_7_data_i,    // R7
    input  logic                                     stream2acc_7_valid_i,
    output logic                                     stream2acc_7_ready_o,
    input  logic [               SUCoreInDWidth-1:0] stream2acc_8_data_i,    // R8
    input  logic                                     stream2acc_8_valid_i,
    output logic                                     stream2acc_8_ready_o,
    input  logic [               SUCoreInXWidth-1:0] stream2acc_9_data_i,    // R9     
    input  logic                                     stream2acc_9_valid_i,
    output logic                                     stream2acc_9_ready_o,
    input  logic [               SUCoreInZWidth-1:0] stream2acc_10_data_i,   // R10
    input  logic                                     stream2acc_10_valid_i,
    output logic                                     stream2acc_10_ready_o,
    input  logic [               ISCoreInAWidth-1:0] stream2acc_11_data_i,   // R11
    input  logic                                     stream2acc_11_valid_i,
    output logic                                     stream2acc_11_ready_o,
    input  logic [               ISCoreInBWidth-1:0] stream2acc_12_data_i,   // R12
    input  logic                                     stream2acc_12_valid_i,
    output logic                                     stream2acc_12_ready_o,
    input  logic [               ISCoreInCWidth-1:0] stream2acc_13_data_i,   // R13
    input  logic                                     stream2acc_13_valid_i,
    output logic                                     stream2acc_13_ready_o,


    //-------------------------------
    // CSR manager ports
    //-------------------------------
    input  logic [RegRWCount-1:0][RegDataWidth-1:0] csr_reg_set_i,
    input  logic                                    csr_reg_set_valid_i,
    output logic                                    csr_reg_set_ready_o,
    output logic [RegROCount-1:0][RegDataWidth-1:0] csr_reg_ro_set_o
);

  assign csr_reg_ro_set_o[0][31:1] = 0;

  SimbaCore inst_SimbaCore (
      .clock(clk_i),
      .reset(~rst_ni),

      // OS Core
      .io_osCore_in_a_ready(stream2acc_0_ready_o),
      .io_osCore_in_a_valid(stream2acc_0_valid_i),
      .io_osCore_in_a_bits (stream2acc_0_data_i),

      .io_osCore_in_b_ready(stream2acc_1_ready_o),
      .io_osCore_in_b_valid(stream2acc_1_valid_i),
      .io_osCore_in_b_bits (stream2acc_1_data_i),

      .io_osCore_out_d_ready(acc2stream_0_ready_i),
      .io_osCore_out_d_valid(acc2stream_0_valid_o),
      .io_osCore_out_d_bits (acc2stream_0_data_o),

      // Switch Core
      .io_switchCore_in_matmul_ready(stream2acc_2_ready_o),
      .io_switchCore_in_matmul_valid(stream2acc_2_valid_i),
      .io_switchCore_in_matmul_bits (stream2acc_2_data_i),

      .io_switchCore_in_weight_ready(stream2acc_3_ready_o),
      .io_switchCore_in_weight_valid(stream2acc_3_valid_i),
      .io_switchCore_in_weight_bits (stream2acc_3_data_i),

      .io_switchCore_in_bias_ready(stream2acc_4_ready_o),
      .io_switchCore_in_bias_valid(stream2acc_4_valid_i),
      .io_switchCore_in_bias_bits (stream2acc_4_data_i),

      .io_switchCore_in_matmulWeight_ready(stream2acc_5_ready_o),
      .io_switchCore_in_matmulWeight_valid(stream2acc_5_valid_i),
      .io_switchCore_in_matmulWeight_bits (stream2acc_5_data_i),

      .io_switchCore_out_x_valid(acc2stream_1_valid_o),
      .io_switchCore_out_x_ready(acc2stream_1_ready_i),
      .io_switchCore_out_x_bits (acc2stream_1_data_o),


      // State update Core
      .io_suCore_in_A_ready(stream2acc_6_ready_o),
      .io_suCore_in_A_valid(stream2acc_6_valid_i),
      .io_suCore_in_A_bits (stream2acc_6_data_i),

      .io_suCore_in_BC_ready(stream2acc_7_ready_o),
      .io_suCore_in_BC_valid(stream2acc_7_valid_i),
      .io_suCore_in_BC_bits (stream2acc_7_data_i),

      .io_suCore_in_D_ready(stream2acc_8_ready_o),
      .io_suCore_in_D_valid(stream2acc_8_valid_i),
      .io_suCore_in_D_bits (stream2acc_8_data_i),

      .io_suCore_in_x_ready(stream2acc_9_ready_o),
      .io_suCore_in_x_valid(stream2acc_9_valid_i),
      .io_suCore_in_x_bits (stream2acc_9_data_i),

      .io_suCore_in_z_ready(stream2acc_10_ready_o),
      .io_suCore_in_z_valid(stream2acc_10_valid_i),
      .io_suCore_in_z_bits (stream2acc_10_data_i),

      .io_suCore_out_y_ready(acc2stream_2_ready_i),
      .io_suCore_out_y_valid(acc2stream_2_valid_o),
      .io_suCore_out_y_bits (acc2stream_2_data_o),

      // IS Core
      .io_isCore_in_a_ready(stream2acc_11_ready_o),
      .io_isCore_in_a_valid(stream2acc_11_valid_i),
      .io_isCore_in_a_bits (stream2acc_11_data_i),

      .io_isCore_in_b_ready(stream2acc_12_ready_o),
      .io_isCore_in_b_valid(stream2acc_12_valid_i),
      .io_isCore_in_b_bits (stream2acc_12_data_i),

      .io_isCore_in_c_ready(stream2acc_13_ready_o),
      .io_isCore_in_c_valid(stream2acc_13_valid_i),
      .io_isCore_in_c_bits (stream2acc_13_data_i),

      .io_isCore_out_d_ready(acc2stream_3_ready_i),
      .io_isCore_out_d_valid(acc2stream_3_valid_o),
      .io_isCore_out_d_bits (acc2stream_3_data_o),

      // CSR
      .io_config_ready(csr_reg_set_ready_o),
      .io_config_valid(csr_reg_set_valid_i),
      .io_config_bits_mode(csr_reg_set_i[0]),
      .io_config_bits_seqLen(csr_reg_set_i[1]),
      .io_config_bits_dModel(csr_reg_set_i[2]),
      .io_config_bits_dtRank(csr_reg_set_i[3]),
      .io_config_bits_dInner(csr_reg_set_i[4]),
      .io_config_bits_dFinal(csr_reg_set_i[5]),

      .io_busy_o(csr_reg_ro_set_o[0][0]),
      .io_performance_counter(csr_reg_ro_set_o[1]),
      .io_osCoreTileCnt(csr_reg_ro_set_o[2]),
      .io_suCoreOutCnt(csr_reg_ro_set_o[3])
  );

endmodule
