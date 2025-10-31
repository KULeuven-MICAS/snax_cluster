// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51


//-----------------------------
// Streamer wrapper
//-----------------------------
module snax_simbacore_streamer_wrapper #(
  // TCDM typedefs
  parameter type         tcdm_req_t    = logic,
  parameter type         tcdm_rsp_t    = logic,
  // Parameters related to TCDM
  parameter int unsigned TCDMDataWidth = 64,
  parameter int unsigned TCDMNumPorts  = 45,
  parameter int unsigned TCDMAddrWidth = 20
)(
  //-----------------------------
  // Clocks and reset
  //-----------------------------
  // clock from accelerator domain
  input  logic clk_i,
  input  logic rst_ni,
  //-----------------------------
  // Accelerator ports
  //-----------------------------
  // Ports from accelerator to streamer by writer data movers
  input  logic [63:0] acc2stream_0_data_i,
  input  logic acc2stream_0_valid_i,
  output logic acc2stream_0_ready_o,

  input  logic [63:0] acc2stream_1_data_i,
  input  logic acc2stream_1_valid_i,
  output logic acc2stream_1_ready_o,

  input  logic [63:0] acc2stream_2_data_i,
  input  logic acc2stream_2_valid_i,
  output logic acc2stream_2_ready_o,

  input  logic [255:0] acc2stream_3_data_i,
  input  logic acc2stream_3_valid_i,
  output logic acc2stream_3_ready_o,

  // Ports from streamer to accelerator by reader data movers
  output logic [255:0] stream2acc_0_data_o,
  output logic stream2acc_0_valid_o,
  input  logic stream2acc_0_ready_i,

  output logic [383:0] stream2acc_1_data_o,
  output logic stream2acc_1_valid_o,
  input  logic stream2acc_1_ready_i,

  output logic [127:0] stream2acc_2_data_o,
  output logic stream2acc_2_valid_o,
  input  logic stream2acc_2_ready_i,

  output logic [63:0] stream2acc_3_data_o,
  output logic stream2acc_3_valid_o,
  input  logic stream2acc_3_ready_i,

  output logic [63:0] stream2acc_4_data_o,
  output logic stream2acc_4_valid_o,
  input  logic stream2acc_4_ready_i,

  output logic [63:0] stream2acc_5_data_o,
  output logic stream2acc_5_valid_o,
  input  logic stream2acc_5_ready_i,

  output logic [63:0] stream2acc_6_data_o,
  output logic stream2acc_6_valid_o,
  input  logic stream2acc_6_ready_i,

  output logic [511:0] stream2acc_7_data_o,
  output logic stream2acc_7_valid_o,
  input  logic stream2acc_7_ready_i,

  output logic [63:0] stream2acc_8_data_o,
  output logic stream2acc_8_valid_o,
  input  logic stream2acc_8_ready_i,

  output logic [63:0] stream2acc_9_data_o,
  output logic stream2acc_9_valid_o,
  input  logic stream2acc_9_ready_i,

  output logic [63:0] stream2acc_10_data_o,
  output logic stream2acc_10_valid_o,
  input  logic stream2acc_10_ready_i,

  output logic [63:0] stream2acc_11_data_o,
  output logic stream2acc_11_valid_o,
  input  logic stream2acc_11_ready_i,

  output logic [383:0] stream2acc_12_data_o,
  output logic stream2acc_12_valid_o,
  input  logic stream2acc_12_ready_i,

  output logic [255:0] stream2acc_13_data_o,
  output logic stream2acc_13_valid_o,
  input  logic stream2acc_13_ready_i,

  //-----------------------------
  // TCDM ports
  //-----------------------------
  output tcdm_req_t [TCDMNumPorts-1:0] tcdm_req_o,
  input  tcdm_rsp_t [TCDMNumPorts-1:0] tcdm_rsp_i,
  //-----------------------------
  // CSR control ports
  //-----------------------------
  // Request
  input  logic [31:0] csr_req_bits_data_i,
  input  logic [31:0] csr_req_bits_addr_i,
  input  logic        csr_req_bits_write_i,
  input  logic        csr_req_valid_i,
  output logic        csr_req_ready_o,
  // Response
  output logic [31:0] csr_rsp_bits_data_o,
  output logic        csr_rsp_valid_o,
  input  logic        csr_rsp_ready_i
);

  //-----------------------------
  // Wiring and combinational logic
  //-----------------------------

  // TCDM signals
  // Request
  logic [TCDMNumPorts-1:0][  TCDMAddrWidth-1:0] tcdm_req_addr;
  logic [TCDMNumPorts-1:0]                      tcdm_req_write;
  //Note that tcdm_req_amo_i is 4 bits based on reqrsp definition
  logic [TCDMNumPorts-1:0][                3:0] tcdm_req_amo;
  logic [TCDMNumPorts-1:0][  TCDMDataWidth-1:0] tcdm_req_data;
  logic [TCDMNumPorts-1:0][TCDMDataWidth/8-1:0] tcdm_req_strb;
  //Note that tcdm_req_user_core_id_i is 5 bits based on Snitch definition
  logic [TCDMNumPorts-1:0][                4:0] tcdm_req_user_core_id;
  logic [TCDMNumPorts-1:0]                      tcdm_req_user_is_core;
  logic [TCDMNumPorts-1:0]                      tcdm_req_q_valid;

  // Response
  logic [TCDMNumPorts-1:0]                      tcdm_rsp_q_ready;
  logic [TCDMNumPorts-1:0]                      tcdm_rsp_p_valid;
  logic [TCDMNumPorts-1:0][  TCDMDataWidth-1:0] tcdm_rsp_data;

  // Fixed ports that are defaulted to tie-low
  // towards the TCDM from the streamer
  always_comb begin
    for(int i = 0; i < TCDMNumPorts; i++ ) begin
      tcdm_req_amo          [i] = '0;
      tcdm_req_user_core_id [i] = '0;
      tcdm_req_user_is_core [i] = '0;
    end
  end

  // Re-mapping wires for TCDM IO ports
  always_comb begin
    for ( int i = 0; i < TCDMNumPorts; i++) begin
      tcdm_req_o[i].q.addr         = tcdm_req_addr   [i];
      tcdm_req_o[i].q.write        = tcdm_req_write  [i];
      tcdm_req_o[i].q.amo          = reqrsp_pkg::AMONone;
      tcdm_req_o[i].q.data         = tcdm_req_data   [i];
      tcdm_req_o[i].q.strb         = tcdm_req_strb   [i];
      tcdm_req_o[i].q.user.core_id = '0;
      tcdm_req_o[i].q.user.is_core = '0;
      tcdm_req_o[i].q_valid        = tcdm_req_q_valid[i];

      tcdm_rsp_q_ready[i] = tcdm_rsp_i[i].q_ready;
      tcdm_rsp_p_valid[i] = tcdm_rsp_i[i].p_valid;
      tcdm_rsp_data   [i] = tcdm_rsp_i[i].p.data ;
    end
  end
  

  // Streamer module that is generated
  // with template mechanics
  snax_simbacore_Streamer i_snax_simbacore_streamer_top (	
    //-----------------------------
    // Clocks and reset
    //-----------------------------
    // clock from accelerator domain
    .clock ( clk_i   ),
    .reset ( ~rst_ni ),

    //-----------------------------
    // Accelerator ports
    //-----------------------------
    // Ports from accelerator to streamer by writer data movers
    .io_data_accelerator2streamer_data_0_bits  (  acc2stream_0_data_i ),
    .io_data_accelerator2streamer_data_0_valid ( acc2stream_0_valid_i ),
    .io_data_accelerator2streamer_data_0_ready ( acc2stream_0_ready_o ),

    .io_data_accelerator2streamer_data_1_bits  (  acc2stream_1_data_i ),
    .io_data_accelerator2streamer_data_1_valid ( acc2stream_1_valid_i ),
    .io_data_accelerator2streamer_data_1_ready ( acc2stream_1_ready_o ),

    .io_data_accelerator2streamer_data_2_bits  (  acc2stream_2_data_i ),
    .io_data_accelerator2streamer_data_2_valid ( acc2stream_2_valid_i ),
    .io_data_accelerator2streamer_data_2_ready ( acc2stream_2_ready_o ),

    .io_data_accelerator2streamer_data_3_bits  (  acc2stream_3_data_i ),
    .io_data_accelerator2streamer_data_3_valid ( acc2stream_3_valid_i ),
    .io_data_accelerator2streamer_data_3_ready ( acc2stream_3_ready_o ),

    // Ports from streamer to accelerator by reader data movers
    .io_data_streamer2accelerator_data_0_bits  (  stream2acc_0_data_o ),
    .io_data_streamer2accelerator_data_0_valid ( stream2acc_0_valid_o ),
    .io_data_streamer2accelerator_data_0_ready ( stream2acc_0_ready_i ),

    .io_data_streamer2accelerator_data_1_bits  (  stream2acc_1_data_o ),
    .io_data_streamer2accelerator_data_1_valid ( stream2acc_1_valid_o ),
    .io_data_streamer2accelerator_data_1_ready ( stream2acc_1_ready_i ),

    .io_data_streamer2accelerator_data_2_bits  (  stream2acc_2_data_o ),
    .io_data_streamer2accelerator_data_2_valid ( stream2acc_2_valid_o ),
    .io_data_streamer2accelerator_data_2_ready ( stream2acc_2_ready_i ),

    .io_data_streamer2accelerator_data_3_bits  (  stream2acc_3_data_o ),
    .io_data_streamer2accelerator_data_3_valid ( stream2acc_3_valid_o ),
    .io_data_streamer2accelerator_data_3_ready ( stream2acc_3_ready_i ),

    .io_data_streamer2accelerator_data_4_bits  (  stream2acc_4_data_o ),
    .io_data_streamer2accelerator_data_4_valid ( stream2acc_4_valid_o ),
    .io_data_streamer2accelerator_data_4_ready ( stream2acc_4_ready_i ),

    .io_data_streamer2accelerator_data_5_bits  (  stream2acc_5_data_o ),
    .io_data_streamer2accelerator_data_5_valid ( stream2acc_5_valid_o ),
    .io_data_streamer2accelerator_data_5_ready ( stream2acc_5_ready_i ),

    .io_data_streamer2accelerator_data_6_bits  (  stream2acc_6_data_o ),
    .io_data_streamer2accelerator_data_6_valid ( stream2acc_6_valid_o ),
    .io_data_streamer2accelerator_data_6_ready ( stream2acc_6_ready_i ),

    .io_data_streamer2accelerator_data_7_bits  (  stream2acc_7_data_o ),
    .io_data_streamer2accelerator_data_7_valid ( stream2acc_7_valid_o ),
    .io_data_streamer2accelerator_data_7_ready ( stream2acc_7_ready_i ),

    .io_data_streamer2accelerator_data_8_bits  (  stream2acc_8_data_o ),
    .io_data_streamer2accelerator_data_8_valid ( stream2acc_8_valid_o ),
    .io_data_streamer2accelerator_data_8_ready ( stream2acc_8_ready_i ),

    .io_data_streamer2accelerator_data_9_bits  (  stream2acc_9_data_o ),
    .io_data_streamer2accelerator_data_9_valid ( stream2acc_9_valid_o ),
    .io_data_streamer2accelerator_data_9_ready ( stream2acc_9_ready_i ),

    .io_data_streamer2accelerator_data_10_bits  (  stream2acc_10_data_o ),
    .io_data_streamer2accelerator_data_10_valid ( stream2acc_10_valid_o ),
    .io_data_streamer2accelerator_data_10_ready ( stream2acc_10_ready_i ),

    .io_data_streamer2accelerator_data_11_bits  (  stream2acc_11_data_o ),
    .io_data_streamer2accelerator_data_11_valid ( stream2acc_11_valid_o ),
    .io_data_streamer2accelerator_data_11_ready ( stream2acc_11_ready_i ),

    .io_data_streamer2accelerator_data_12_bits  (  stream2acc_12_data_o ),
    .io_data_streamer2accelerator_data_12_valid ( stream2acc_12_valid_o ),
    .io_data_streamer2accelerator_data_12_ready ( stream2acc_12_ready_i ),

    .io_data_streamer2accelerator_data_13_bits  (  stream2acc_13_data_o ),
    .io_data_streamer2accelerator_data_13_valid ( stream2acc_13_valid_o ),
    .io_data_streamer2accelerator_data_13_ready ( stream2acc_13_ready_i ),

    //-----------------------------
    // TCDM Ports
    //-----------------------------
    // Request
    .io_data_tcdm_rsp_0_bits_data  ( tcdm_rsp_data   [0] ),
    .io_data_tcdm_rsp_0_valid      ( tcdm_rsp_p_valid[0] ),
    .io_data_tcdm_req_0_ready      ( tcdm_rsp_q_ready[0] ),

    .io_data_tcdm_rsp_1_bits_data  ( tcdm_rsp_data   [1] ),
    .io_data_tcdm_rsp_1_valid      ( tcdm_rsp_p_valid[1] ),
    .io_data_tcdm_req_1_ready      ( tcdm_rsp_q_ready[1] ),

    .io_data_tcdm_rsp_2_bits_data  ( tcdm_rsp_data   [2] ),
    .io_data_tcdm_rsp_2_valid      ( tcdm_rsp_p_valid[2] ),
    .io_data_tcdm_req_2_ready      ( tcdm_rsp_q_ready[2] ),

    .io_data_tcdm_rsp_3_bits_data  ( tcdm_rsp_data   [3] ),
    .io_data_tcdm_rsp_3_valid      ( tcdm_rsp_p_valid[3] ),
    .io_data_tcdm_req_3_ready      ( tcdm_rsp_q_ready[3] ),

    .io_data_tcdm_rsp_4_bits_data  ( tcdm_rsp_data   [4] ),
    .io_data_tcdm_rsp_4_valid      ( tcdm_rsp_p_valid[4] ),
    .io_data_tcdm_req_4_ready      ( tcdm_rsp_q_ready[4] ),

    .io_data_tcdm_rsp_5_bits_data  ( tcdm_rsp_data   [5] ),
    .io_data_tcdm_rsp_5_valid      ( tcdm_rsp_p_valid[5] ),
    .io_data_tcdm_req_5_ready      ( tcdm_rsp_q_ready[5] ),

    .io_data_tcdm_rsp_6_bits_data  ( tcdm_rsp_data   [6] ),
    .io_data_tcdm_rsp_6_valid      ( tcdm_rsp_p_valid[6] ),
    .io_data_tcdm_req_6_ready      ( tcdm_rsp_q_ready[6] ),

    .io_data_tcdm_rsp_7_bits_data  ( tcdm_rsp_data   [7] ),
    .io_data_tcdm_rsp_7_valid      ( tcdm_rsp_p_valid[7] ),
    .io_data_tcdm_req_7_ready      ( tcdm_rsp_q_ready[7] ),

    .io_data_tcdm_rsp_8_bits_data  ( tcdm_rsp_data   [8] ),
    .io_data_tcdm_rsp_8_valid      ( tcdm_rsp_p_valid[8] ),
    .io_data_tcdm_req_8_ready      ( tcdm_rsp_q_ready[8] ),

    .io_data_tcdm_rsp_9_bits_data  ( tcdm_rsp_data   [9] ),
    .io_data_tcdm_rsp_9_valid      ( tcdm_rsp_p_valid[9] ),
    .io_data_tcdm_req_9_ready      ( tcdm_rsp_q_ready[9] ),

    .io_data_tcdm_rsp_10_bits_data  ( tcdm_rsp_data   [10] ),
    .io_data_tcdm_rsp_10_valid      ( tcdm_rsp_p_valid[10] ),
    .io_data_tcdm_req_10_ready      ( tcdm_rsp_q_ready[10] ),

    .io_data_tcdm_rsp_11_bits_data  ( tcdm_rsp_data   [11] ),
    .io_data_tcdm_rsp_11_valid      ( tcdm_rsp_p_valid[11] ),
    .io_data_tcdm_req_11_ready      ( tcdm_rsp_q_ready[11] ),

    .io_data_tcdm_rsp_12_bits_data  ( tcdm_rsp_data   [12] ),
    .io_data_tcdm_rsp_12_valid      ( tcdm_rsp_p_valid[12] ),
    .io_data_tcdm_req_12_ready      ( tcdm_rsp_q_ready[12] ),

    .io_data_tcdm_rsp_13_bits_data  ( tcdm_rsp_data   [13] ),
    .io_data_tcdm_rsp_13_valid      ( tcdm_rsp_p_valid[13] ),
    .io_data_tcdm_req_13_ready      ( tcdm_rsp_q_ready[13] ),

    .io_data_tcdm_rsp_14_bits_data  ( tcdm_rsp_data   [14] ),
    .io_data_tcdm_rsp_14_valid      ( tcdm_rsp_p_valid[14] ),
    .io_data_tcdm_req_14_ready      ( tcdm_rsp_q_ready[14] ),

    .io_data_tcdm_rsp_15_bits_data  ( tcdm_rsp_data   [15] ),
    .io_data_tcdm_rsp_15_valid      ( tcdm_rsp_p_valid[15] ),
    .io_data_tcdm_req_15_ready      ( tcdm_rsp_q_ready[15] ),

    .io_data_tcdm_rsp_16_bits_data  ( tcdm_rsp_data   [16] ),
    .io_data_tcdm_rsp_16_valid      ( tcdm_rsp_p_valid[16] ),
    .io_data_tcdm_req_16_ready      ( tcdm_rsp_q_ready[16] ),

    .io_data_tcdm_rsp_17_bits_data  ( tcdm_rsp_data   [17] ),
    .io_data_tcdm_rsp_17_valid      ( tcdm_rsp_p_valid[17] ),
    .io_data_tcdm_req_17_ready      ( tcdm_rsp_q_ready[17] ),

    .io_data_tcdm_rsp_18_bits_data  ( tcdm_rsp_data   [18] ),
    .io_data_tcdm_rsp_18_valid      ( tcdm_rsp_p_valid[18] ),
    .io_data_tcdm_req_18_ready      ( tcdm_rsp_q_ready[18] ),

    .io_data_tcdm_rsp_19_bits_data  ( tcdm_rsp_data   [19] ),
    .io_data_tcdm_rsp_19_valid      ( tcdm_rsp_p_valid[19] ),
    .io_data_tcdm_req_19_ready      ( tcdm_rsp_q_ready[19] ),

    .io_data_tcdm_rsp_20_bits_data  ( tcdm_rsp_data   [20] ),
    .io_data_tcdm_rsp_20_valid      ( tcdm_rsp_p_valid[20] ),
    .io_data_tcdm_req_20_ready      ( tcdm_rsp_q_ready[20] ),

    .io_data_tcdm_rsp_21_bits_data  ( tcdm_rsp_data   [21] ),
    .io_data_tcdm_rsp_21_valid      ( tcdm_rsp_p_valid[21] ),
    .io_data_tcdm_req_21_ready      ( tcdm_rsp_q_ready[21] ),

    .io_data_tcdm_rsp_22_bits_data  ( tcdm_rsp_data   [22] ),
    .io_data_tcdm_rsp_22_valid      ( tcdm_rsp_p_valid[22] ),
    .io_data_tcdm_req_22_ready      ( tcdm_rsp_q_ready[22] ),

    .io_data_tcdm_rsp_23_bits_data  ( tcdm_rsp_data   [23] ),
    .io_data_tcdm_rsp_23_valid      ( tcdm_rsp_p_valid[23] ),
    .io_data_tcdm_req_23_ready      ( tcdm_rsp_q_ready[23] ),

    .io_data_tcdm_rsp_24_bits_data  ( tcdm_rsp_data   [24] ),
    .io_data_tcdm_rsp_24_valid      ( tcdm_rsp_p_valid[24] ),
    .io_data_tcdm_req_24_ready      ( tcdm_rsp_q_ready[24] ),

    .io_data_tcdm_rsp_25_bits_data  ( tcdm_rsp_data   [25] ),
    .io_data_tcdm_rsp_25_valid      ( tcdm_rsp_p_valid[25] ),
    .io_data_tcdm_req_25_ready      ( tcdm_rsp_q_ready[25] ),

    .io_data_tcdm_rsp_26_bits_data  ( tcdm_rsp_data   [26] ),
    .io_data_tcdm_rsp_26_valid      ( tcdm_rsp_p_valid[26] ),
    .io_data_tcdm_req_26_ready      ( tcdm_rsp_q_ready[26] ),

    .io_data_tcdm_rsp_27_bits_data  ( tcdm_rsp_data   [27] ),
    .io_data_tcdm_rsp_27_valid      ( tcdm_rsp_p_valid[27] ),
    .io_data_tcdm_req_27_ready      ( tcdm_rsp_q_ready[27] ),

    .io_data_tcdm_rsp_28_bits_data  ( tcdm_rsp_data   [28] ),
    .io_data_tcdm_rsp_28_valid      ( tcdm_rsp_p_valid[28] ),
    .io_data_tcdm_req_28_ready      ( tcdm_rsp_q_ready[28] ),

    .io_data_tcdm_rsp_29_bits_data  ( tcdm_rsp_data   [29] ),
    .io_data_tcdm_rsp_29_valid      ( tcdm_rsp_p_valid[29] ),
    .io_data_tcdm_req_29_ready      ( tcdm_rsp_q_ready[29] ),

    .io_data_tcdm_rsp_30_bits_data  ( tcdm_rsp_data   [30] ),
    .io_data_tcdm_rsp_30_valid      ( tcdm_rsp_p_valid[30] ),
    .io_data_tcdm_req_30_ready      ( tcdm_rsp_q_ready[30] ),

    .io_data_tcdm_rsp_31_bits_data  ( tcdm_rsp_data   [31] ),
    .io_data_tcdm_rsp_31_valid      ( tcdm_rsp_p_valid[31] ),
    .io_data_tcdm_req_31_ready      ( tcdm_rsp_q_ready[31] ),

    .io_data_tcdm_rsp_32_bits_data  ( tcdm_rsp_data   [32] ),
    .io_data_tcdm_rsp_32_valid      ( tcdm_rsp_p_valid[32] ),
    .io_data_tcdm_req_32_ready      ( tcdm_rsp_q_ready[32] ),

    .io_data_tcdm_rsp_33_bits_data  ( tcdm_rsp_data   [33] ),
    .io_data_tcdm_rsp_33_valid      ( tcdm_rsp_p_valid[33] ),
    .io_data_tcdm_req_33_ready      ( tcdm_rsp_q_ready[33] ),

    .io_data_tcdm_rsp_34_bits_data  ( tcdm_rsp_data   [34] ),
    .io_data_tcdm_rsp_34_valid      ( tcdm_rsp_p_valid[34] ),
    .io_data_tcdm_req_34_ready      ( tcdm_rsp_q_ready[34] ),

    .io_data_tcdm_rsp_35_bits_data  ( tcdm_rsp_data   [35] ),
    .io_data_tcdm_rsp_35_valid      ( tcdm_rsp_p_valid[35] ),
    .io_data_tcdm_req_35_ready      ( tcdm_rsp_q_ready[35] ),

    .io_data_tcdm_rsp_36_bits_data  ( tcdm_rsp_data   [36] ),
    .io_data_tcdm_rsp_36_valid      ( tcdm_rsp_p_valid[36] ),
    .io_data_tcdm_req_36_ready      ( tcdm_rsp_q_ready[36] ),

    .io_data_tcdm_rsp_37_bits_data  ( tcdm_rsp_data   [37] ),
    .io_data_tcdm_rsp_37_valid      ( tcdm_rsp_p_valid[37] ),
    .io_data_tcdm_req_37_ready      ( tcdm_rsp_q_ready[37] ),

    .io_data_tcdm_rsp_38_bits_data  ( tcdm_rsp_data   [38] ),
    .io_data_tcdm_rsp_38_valid      ( tcdm_rsp_p_valid[38] ),
    .io_data_tcdm_req_38_ready      ( tcdm_rsp_q_ready[38] ),

    .io_data_tcdm_rsp_39_bits_data  ( tcdm_rsp_data   [39] ),
    .io_data_tcdm_rsp_39_valid      ( tcdm_rsp_p_valid[39] ),
    .io_data_tcdm_req_39_ready      ( tcdm_rsp_q_ready[39] ),

    .io_data_tcdm_rsp_40_bits_data  ( tcdm_rsp_data   [40] ),
    .io_data_tcdm_rsp_40_valid      ( tcdm_rsp_p_valid[40] ),
    .io_data_tcdm_req_40_ready      ( tcdm_rsp_q_ready[40] ),

    .io_data_tcdm_rsp_41_bits_data  ( tcdm_rsp_data   [41] ),
    .io_data_tcdm_rsp_41_valid      ( tcdm_rsp_p_valid[41] ),
    .io_data_tcdm_req_41_ready      ( tcdm_rsp_q_ready[41] ),

    .io_data_tcdm_rsp_42_bits_data  ( tcdm_rsp_data   [42] ),
    .io_data_tcdm_rsp_42_valid      ( tcdm_rsp_p_valid[42] ),
    .io_data_tcdm_req_42_ready      ( tcdm_rsp_q_ready[42] ),

    .io_data_tcdm_rsp_43_bits_data  ( tcdm_rsp_data   [43] ),
    .io_data_tcdm_rsp_43_valid      ( tcdm_rsp_p_valid[43] ),
    .io_data_tcdm_req_43_ready      ( tcdm_rsp_q_ready[43] ),

    .io_data_tcdm_rsp_44_bits_data  ( tcdm_rsp_data   [44] ),
    .io_data_tcdm_rsp_44_valid      ( tcdm_rsp_p_valid[44] ),
    .io_data_tcdm_req_44_ready      ( tcdm_rsp_q_ready[44] ),

    // Response
    .io_data_tcdm_req_0_valid      ( tcdm_req_q_valid[0] ),
    .io_data_tcdm_req_0_bits_addr  ( tcdm_req_addr   [0] ),
    .io_data_tcdm_req_0_bits_write ( tcdm_req_write  [0] ),
    .io_data_tcdm_req_0_bits_data  ( tcdm_req_data   [0] ),
    .io_data_tcdm_req_0_bits_strb  ( tcdm_req_strb   [0] ),

    .io_data_tcdm_req_1_valid      ( tcdm_req_q_valid[1] ),
    .io_data_tcdm_req_1_bits_addr  ( tcdm_req_addr   [1] ),
    .io_data_tcdm_req_1_bits_write ( tcdm_req_write  [1] ),
    .io_data_tcdm_req_1_bits_data  ( tcdm_req_data   [1] ),
    .io_data_tcdm_req_1_bits_strb  ( tcdm_req_strb   [1] ),

    .io_data_tcdm_req_2_valid      ( tcdm_req_q_valid[2] ),
    .io_data_tcdm_req_2_bits_addr  ( tcdm_req_addr   [2] ),
    .io_data_tcdm_req_2_bits_write ( tcdm_req_write  [2] ),
    .io_data_tcdm_req_2_bits_data  ( tcdm_req_data   [2] ),
    .io_data_tcdm_req_2_bits_strb  ( tcdm_req_strb   [2] ),

    .io_data_tcdm_req_3_valid      ( tcdm_req_q_valid[3] ),
    .io_data_tcdm_req_3_bits_addr  ( tcdm_req_addr   [3] ),
    .io_data_tcdm_req_3_bits_write ( tcdm_req_write  [3] ),
    .io_data_tcdm_req_3_bits_data  ( tcdm_req_data   [3] ),
    .io_data_tcdm_req_3_bits_strb  ( tcdm_req_strb   [3] ),

    .io_data_tcdm_req_4_valid      ( tcdm_req_q_valid[4] ),
    .io_data_tcdm_req_4_bits_addr  ( tcdm_req_addr   [4] ),
    .io_data_tcdm_req_4_bits_write ( tcdm_req_write  [4] ),
    .io_data_tcdm_req_4_bits_data  ( tcdm_req_data   [4] ),
    .io_data_tcdm_req_4_bits_strb  ( tcdm_req_strb   [4] ),

    .io_data_tcdm_req_5_valid      ( tcdm_req_q_valid[5] ),
    .io_data_tcdm_req_5_bits_addr  ( tcdm_req_addr   [5] ),
    .io_data_tcdm_req_5_bits_write ( tcdm_req_write  [5] ),
    .io_data_tcdm_req_5_bits_data  ( tcdm_req_data   [5] ),
    .io_data_tcdm_req_5_bits_strb  ( tcdm_req_strb   [5] ),

    .io_data_tcdm_req_6_valid      ( tcdm_req_q_valid[6] ),
    .io_data_tcdm_req_6_bits_addr  ( tcdm_req_addr   [6] ),
    .io_data_tcdm_req_6_bits_write ( tcdm_req_write  [6] ),
    .io_data_tcdm_req_6_bits_data  ( tcdm_req_data   [6] ),
    .io_data_tcdm_req_6_bits_strb  ( tcdm_req_strb   [6] ),

    .io_data_tcdm_req_7_valid      ( tcdm_req_q_valid[7] ),
    .io_data_tcdm_req_7_bits_addr  ( tcdm_req_addr   [7] ),
    .io_data_tcdm_req_7_bits_write ( tcdm_req_write  [7] ),
    .io_data_tcdm_req_7_bits_data  ( tcdm_req_data   [7] ),
    .io_data_tcdm_req_7_bits_strb  ( tcdm_req_strb   [7] ),

    .io_data_tcdm_req_8_valid      ( tcdm_req_q_valid[8] ),
    .io_data_tcdm_req_8_bits_addr  ( tcdm_req_addr   [8] ),
    .io_data_tcdm_req_8_bits_write ( tcdm_req_write  [8] ),
    .io_data_tcdm_req_8_bits_data  ( tcdm_req_data   [8] ),
    .io_data_tcdm_req_8_bits_strb  ( tcdm_req_strb   [8] ),

    .io_data_tcdm_req_9_valid      ( tcdm_req_q_valid[9] ),
    .io_data_tcdm_req_9_bits_addr  ( tcdm_req_addr   [9] ),
    .io_data_tcdm_req_9_bits_write ( tcdm_req_write  [9] ),
    .io_data_tcdm_req_9_bits_data  ( tcdm_req_data   [9] ),
    .io_data_tcdm_req_9_bits_strb  ( tcdm_req_strb   [9] ),

    .io_data_tcdm_req_10_valid      ( tcdm_req_q_valid[10] ),
    .io_data_tcdm_req_10_bits_addr  ( tcdm_req_addr   [10] ),
    .io_data_tcdm_req_10_bits_write ( tcdm_req_write  [10] ),
    .io_data_tcdm_req_10_bits_data  ( tcdm_req_data   [10] ),
    .io_data_tcdm_req_10_bits_strb  ( tcdm_req_strb   [10] ),

    .io_data_tcdm_req_11_valid      ( tcdm_req_q_valid[11] ),
    .io_data_tcdm_req_11_bits_addr  ( tcdm_req_addr   [11] ),
    .io_data_tcdm_req_11_bits_write ( tcdm_req_write  [11] ),
    .io_data_tcdm_req_11_bits_data  ( tcdm_req_data   [11] ),
    .io_data_tcdm_req_11_bits_strb  ( tcdm_req_strb   [11] ),

    .io_data_tcdm_req_12_valid      ( tcdm_req_q_valid[12] ),
    .io_data_tcdm_req_12_bits_addr  ( tcdm_req_addr   [12] ),
    .io_data_tcdm_req_12_bits_write ( tcdm_req_write  [12] ),
    .io_data_tcdm_req_12_bits_data  ( tcdm_req_data   [12] ),
    .io_data_tcdm_req_12_bits_strb  ( tcdm_req_strb   [12] ),

    .io_data_tcdm_req_13_valid      ( tcdm_req_q_valid[13] ),
    .io_data_tcdm_req_13_bits_addr  ( tcdm_req_addr   [13] ),
    .io_data_tcdm_req_13_bits_write ( tcdm_req_write  [13] ),
    .io_data_tcdm_req_13_bits_data  ( tcdm_req_data   [13] ),
    .io_data_tcdm_req_13_bits_strb  ( tcdm_req_strb   [13] ),

    .io_data_tcdm_req_14_valid      ( tcdm_req_q_valid[14] ),
    .io_data_tcdm_req_14_bits_addr  ( tcdm_req_addr   [14] ),
    .io_data_tcdm_req_14_bits_write ( tcdm_req_write  [14] ),
    .io_data_tcdm_req_14_bits_data  ( tcdm_req_data   [14] ),
    .io_data_tcdm_req_14_bits_strb  ( tcdm_req_strb   [14] ),

    .io_data_tcdm_req_15_valid      ( tcdm_req_q_valid[15] ),
    .io_data_tcdm_req_15_bits_addr  ( tcdm_req_addr   [15] ),
    .io_data_tcdm_req_15_bits_write ( tcdm_req_write  [15] ),
    .io_data_tcdm_req_15_bits_data  ( tcdm_req_data   [15] ),
    .io_data_tcdm_req_15_bits_strb  ( tcdm_req_strb   [15] ),

    .io_data_tcdm_req_16_valid      ( tcdm_req_q_valid[16] ),
    .io_data_tcdm_req_16_bits_addr  ( tcdm_req_addr   [16] ),
    .io_data_tcdm_req_16_bits_write ( tcdm_req_write  [16] ),
    .io_data_tcdm_req_16_bits_data  ( tcdm_req_data   [16] ),
    .io_data_tcdm_req_16_bits_strb  ( tcdm_req_strb   [16] ),

    .io_data_tcdm_req_17_valid      ( tcdm_req_q_valid[17] ),
    .io_data_tcdm_req_17_bits_addr  ( tcdm_req_addr   [17] ),
    .io_data_tcdm_req_17_bits_write ( tcdm_req_write  [17] ),
    .io_data_tcdm_req_17_bits_data  ( tcdm_req_data   [17] ),
    .io_data_tcdm_req_17_bits_strb  ( tcdm_req_strb   [17] ),

    .io_data_tcdm_req_18_valid      ( tcdm_req_q_valid[18] ),
    .io_data_tcdm_req_18_bits_addr  ( tcdm_req_addr   [18] ),
    .io_data_tcdm_req_18_bits_write ( tcdm_req_write  [18] ),
    .io_data_tcdm_req_18_bits_data  ( tcdm_req_data   [18] ),
    .io_data_tcdm_req_18_bits_strb  ( tcdm_req_strb   [18] ),

    .io_data_tcdm_req_19_valid      ( tcdm_req_q_valid[19] ),
    .io_data_tcdm_req_19_bits_addr  ( tcdm_req_addr   [19] ),
    .io_data_tcdm_req_19_bits_write ( tcdm_req_write  [19] ),
    .io_data_tcdm_req_19_bits_data  ( tcdm_req_data   [19] ),
    .io_data_tcdm_req_19_bits_strb  ( tcdm_req_strb   [19] ),

    .io_data_tcdm_req_20_valid      ( tcdm_req_q_valid[20] ),
    .io_data_tcdm_req_20_bits_addr  ( tcdm_req_addr   [20] ),
    .io_data_tcdm_req_20_bits_write ( tcdm_req_write  [20] ),
    .io_data_tcdm_req_20_bits_data  ( tcdm_req_data   [20] ),
    .io_data_tcdm_req_20_bits_strb  ( tcdm_req_strb   [20] ),

    .io_data_tcdm_req_21_valid      ( tcdm_req_q_valid[21] ),
    .io_data_tcdm_req_21_bits_addr  ( tcdm_req_addr   [21] ),
    .io_data_tcdm_req_21_bits_write ( tcdm_req_write  [21] ),
    .io_data_tcdm_req_21_bits_data  ( tcdm_req_data   [21] ),
    .io_data_tcdm_req_21_bits_strb  ( tcdm_req_strb   [21] ),

    .io_data_tcdm_req_22_valid      ( tcdm_req_q_valid[22] ),
    .io_data_tcdm_req_22_bits_addr  ( tcdm_req_addr   [22] ),
    .io_data_tcdm_req_22_bits_write ( tcdm_req_write  [22] ),
    .io_data_tcdm_req_22_bits_data  ( tcdm_req_data   [22] ),
    .io_data_tcdm_req_22_bits_strb  ( tcdm_req_strb   [22] ),

    .io_data_tcdm_req_23_valid      ( tcdm_req_q_valid[23] ),
    .io_data_tcdm_req_23_bits_addr  ( tcdm_req_addr   [23] ),
    .io_data_tcdm_req_23_bits_write ( tcdm_req_write  [23] ),
    .io_data_tcdm_req_23_bits_data  ( tcdm_req_data   [23] ),
    .io_data_tcdm_req_23_bits_strb  ( tcdm_req_strb   [23] ),

    .io_data_tcdm_req_24_valid      ( tcdm_req_q_valid[24] ),
    .io_data_tcdm_req_24_bits_addr  ( tcdm_req_addr   [24] ),
    .io_data_tcdm_req_24_bits_write ( tcdm_req_write  [24] ),
    .io_data_tcdm_req_24_bits_data  ( tcdm_req_data   [24] ),
    .io_data_tcdm_req_24_bits_strb  ( tcdm_req_strb   [24] ),

    .io_data_tcdm_req_25_valid      ( tcdm_req_q_valid[25] ),
    .io_data_tcdm_req_25_bits_addr  ( tcdm_req_addr   [25] ),
    .io_data_tcdm_req_25_bits_write ( tcdm_req_write  [25] ),
    .io_data_tcdm_req_25_bits_data  ( tcdm_req_data   [25] ),
    .io_data_tcdm_req_25_bits_strb  ( tcdm_req_strb   [25] ),

    .io_data_tcdm_req_26_valid      ( tcdm_req_q_valid[26] ),
    .io_data_tcdm_req_26_bits_addr  ( tcdm_req_addr   [26] ),
    .io_data_tcdm_req_26_bits_write ( tcdm_req_write  [26] ),
    .io_data_tcdm_req_26_bits_data  ( tcdm_req_data   [26] ),
    .io_data_tcdm_req_26_bits_strb  ( tcdm_req_strb   [26] ),

    .io_data_tcdm_req_27_valid      ( tcdm_req_q_valid[27] ),
    .io_data_tcdm_req_27_bits_addr  ( tcdm_req_addr   [27] ),
    .io_data_tcdm_req_27_bits_write ( tcdm_req_write  [27] ),
    .io_data_tcdm_req_27_bits_data  ( tcdm_req_data   [27] ),
    .io_data_tcdm_req_27_bits_strb  ( tcdm_req_strb   [27] ),

    .io_data_tcdm_req_28_valid      ( tcdm_req_q_valid[28] ),
    .io_data_tcdm_req_28_bits_addr  ( tcdm_req_addr   [28] ),
    .io_data_tcdm_req_28_bits_write ( tcdm_req_write  [28] ),
    .io_data_tcdm_req_28_bits_data  ( tcdm_req_data   [28] ),
    .io_data_tcdm_req_28_bits_strb  ( tcdm_req_strb   [28] ),

    .io_data_tcdm_req_29_valid      ( tcdm_req_q_valid[29] ),
    .io_data_tcdm_req_29_bits_addr  ( tcdm_req_addr   [29] ),
    .io_data_tcdm_req_29_bits_write ( tcdm_req_write  [29] ),
    .io_data_tcdm_req_29_bits_data  ( tcdm_req_data   [29] ),
    .io_data_tcdm_req_29_bits_strb  ( tcdm_req_strb   [29] ),

    .io_data_tcdm_req_30_valid      ( tcdm_req_q_valid[30] ),
    .io_data_tcdm_req_30_bits_addr  ( tcdm_req_addr   [30] ),
    .io_data_tcdm_req_30_bits_write ( tcdm_req_write  [30] ),
    .io_data_tcdm_req_30_bits_data  ( tcdm_req_data   [30] ),
    .io_data_tcdm_req_30_bits_strb  ( tcdm_req_strb   [30] ),

    .io_data_tcdm_req_31_valid      ( tcdm_req_q_valid[31] ),
    .io_data_tcdm_req_31_bits_addr  ( tcdm_req_addr   [31] ),
    .io_data_tcdm_req_31_bits_write ( tcdm_req_write  [31] ),
    .io_data_tcdm_req_31_bits_data  ( tcdm_req_data   [31] ),
    .io_data_tcdm_req_31_bits_strb  ( tcdm_req_strb   [31] ),

    .io_data_tcdm_req_32_valid      ( tcdm_req_q_valid[32] ),
    .io_data_tcdm_req_32_bits_addr  ( tcdm_req_addr   [32] ),
    .io_data_tcdm_req_32_bits_write ( tcdm_req_write  [32] ),
    .io_data_tcdm_req_32_bits_data  ( tcdm_req_data   [32] ),
    .io_data_tcdm_req_32_bits_strb  ( tcdm_req_strb   [32] ),

    .io_data_tcdm_req_33_valid      ( tcdm_req_q_valid[33] ),
    .io_data_tcdm_req_33_bits_addr  ( tcdm_req_addr   [33] ),
    .io_data_tcdm_req_33_bits_write ( tcdm_req_write  [33] ),
    .io_data_tcdm_req_33_bits_data  ( tcdm_req_data   [33] ),
    .io_data_tcdm_req_33_bits_strb  ( tcdm_req_strb   [33] ),

    .io_data_tcdm_req_34_valid      ( tcdm_req_q_valid[34] ),
    .io_data_tcdm_req_34_bits_addr  ( tcdm_req_addr   [34] ),
    .io_data_tcdm_req_34_bits_write ( tcdm_req_write  [34] ),
    .io_data_tcdm_req_34_bits_data  ( tcdm_req_data   [34] ),
    .io_data_tcdm_req_34_bits_strb  ( tcdm_req_strb   [34] ),

    .io_data_tcdm_req_35_valid      ( tcdm_req_q_valid[35] ),
    .io_data_tcdm_req_35_bits_addr  ( tcdm_req_addr   [35] ),
    .io_data_tcdm_req_35_bits_write ( tcdm_req_write  [35] ),
    .io_data_tcdm_req_35_bits_data  ( tcdm_req_data   [35] ),
    .io_data_tcdm_req_35_bits_strb  ( tcdm_req_strb   [35] ),

    .io_data_tcdm_req_36_valid      ( tcdm_req_q_valid[36] ),
    .io_data_tcdm_req_36_bits_addr  ( tcdm_req_addr   [36] ),
    .io_data_tcdm_req_36_bits_write ( tcdm_req_write  [36] ),
    .io_data_tcdm_req_36_bits_data  ( tcdm_req_data   [36] ),
    .io_data_tcdm_req_36_bits_strb  ( tcdm_req_strb   [36] ),

    .io_data_tcdm_req_37_valid      ( tcdm_req_q_valid[37] ),
    .io_data_tcdm_req_37_bits_addr  ( tcdm_req_addr   [37] ),
    .io_data_tcdm_req_37_bits_write ( tcdm_req_write  [37] ),
    .io_data_tcdm_req_37_bits_data  ( tcdm_req_data   [37] ),
    .io_data_tcdm_req_37_bits_strb  ( tcdm_req_strb   [37] ),

    .io_data_tcdm_req_38_valid      ( tcdm_req_q_valid[38] ),
    .io_data_tcdm_req_38_bits_addr  ( tcdm_req_addr   [38] ),
    .io_data_tcdm_req_38_bits_write ( tcdm_req_write  [38] ),
    .io_data_tcdm_req_38_bits_data  ( tcdm_req_data   [38] ),
    .io_data_tcdm_req_38_bits_strb  ( tcdm_req_strb   [38] ),

    .io_data_tcdm_req_39_valid      ( tcdm_req_q_valid[39] ),
    .io_data_tcdm_req_39_bits_addr  ( tcdm_req_addr   [39] ),
    .io_data_tcdm_req_39_bits_write ( tcdm_req_write  [39] ),
    .io_data_tcdm_req_39_bits_data  ( tcdm_req_data   [39] ),
    .io_data_tcdm_req_39_bits_strb  ( tcdm_req_strb   [39] ),

    .io_data_tcdm_req_40_valid      ( tcdm_req_q_valid[40] ),
    .io_data_tcdm_req_40_bits_addr  ( tcdm_req_addr   [40] ),
    .io_data_tcdm_req_40_bits_write ( tcdm_req_write  [40] ),
    .io_data_tcdm_req_40_bits_data  ( tcdm_req_data   [40] ),
    .io_data_tcdm_req_40_bits_strb  ( tcdm_req_strb   [40] ),

    .io_data_tcdm_req_41_valid      ( tcdm_req_q_valid[41] ),
    .io_data_tcdm_req_41_bits_addr  ( tcdm_req_addr   [41] ),
    .io_data_tcdm_req_41_bits_write ( tcdm_req_write  [41] ),
    .io_data_tcdm_req_41_bits_data  ( tcdm_req_data   [41] ),
    .io_data_tcdm_req_41_bits_strb  ( tcdm_req_strb   [41] ),

    .io_data_tcdm_req_42_valid      ( tcdm_req_q_valid[42] ),
    .io_data_tcdm_req_42_bits_addr  ( tcdm_req_addr   [42] ),
    .io_data_tcdm_req_42_bits_write ( tcdm_req_write  [42] ),
    .io_data_tcdm_req_42_bits_data  ( tcdm_req_data   [42] ),
    .io_data_tcdm_req_42_bits_strb  ( tcdm_req_strb   [42] ),

    .io_data_tcdm_req_43_valid      ( tcdm_req_q_valid[43] ),
    .io_data_tcdm_req_43_bits_addr  ( tcdm_req_addr   [43] ),
    .io_data_tcdm_req_43_bits_write ( tcdm_req_write  [43] ),
    .io_data_tcdm_req_43_bits_data  ( tcdm_req_data   [43] ),
    .io_data_tcdm_req_43_bits_strb  ( tcdm_req_strb   [43] ),

    .io_data_tcdm_req_44_valid      ( tcdm_req_q_valid[44] ),
    .io_data_tcdm_req_44_bits_addr  ( tcdm_req_addr   [44] ),
    .io_data_tcdm_req_44_bits_write ( tcdm_req_write  [44] ),
    .io_data_tcdm_req_44_bits_data  ( tcdm_req_data   [44] ),
    .io_data_tcdm_req_44_bits_strb  ( tcdm_req_strb   [44] ),

    //-----------------------------
    // CSR control ports
    //-----------------------------
    // Request
    .io_csr_req_bits_data  ( csr_req_bits_data_i  ),
    .io_csr_req_bits_addr  ( csr_req_bits_addr_i  ),
    .io_csr_req_bits_write ( csr_req_bits_write_i ),
    .io_csr_req_bits_strb  ( '1                   ),
    .io_csr_req_valid      ( csr_req_valid_i      ),
    .io_csr_req_ready      ( csr_req_ready_o      ),

    // Response
    .io_csr_rsp_bits_data  ( csr_rsp_bits_data_o  ),	
    .io_csr_rsp_valid      ( csr_rsp_valid_o      ),
    .io_csr_rsp_ready      ( csr_rsp_ready_i      )
  );

endmodule
