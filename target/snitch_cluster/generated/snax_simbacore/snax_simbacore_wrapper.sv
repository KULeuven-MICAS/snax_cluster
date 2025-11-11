// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51



//-------------------------------
// Streamer-MUL wrapper
// This is the entire accelerator
// That connecst to the TCDM subsystem
//-------------------------------
module snax_simbacore_wrapper # (
  // TCDM typedefs
  parameter type         tcdm_req_t         = logic,
  parameter type         tcdm_rsp_t         = logic,
  // Reconfigurable parameters
  parameter int unsigned DataWidth          = 64,
  parameter int unsigned SnaxTcdmPorts      = 45,
  // Addr width is pre-computed in the generator
  // TCDMAddrWidth = log2(TCDMBankNum * TCDMDepth * (TCDMDataWidth/8))
  parameter int unsigned TCDMAddrWidth      = 20,
  // Don't touch parameters (or modify at your own risk)
  parameter int unsigned RegDataWidth       = 32,
  parameter int unsigned RegAddrWidth       = 32
)(
  //-----------------------------
  // Clocks and reset
  //-----------------------------
  input  logic clk_i,
  input  logic rst_ni,
  //-----------------------------
  // CSR control ports
  //-----------------------------
  // Request
  input  logic [31:0] snax_req_data_i,
  input  logic [31:0] snax_req_addr_i,
  input  logic        snax_req_write_i,
  input  logic        snax_req_valid_i,
  output logic        snax_req_ready_o,
  // Response
  input  logic        snax_rsp_ready_i,
  output logic        snax_rsp_valid_o,
  output logic [31:0] snax_rsp_data_o,
  //-----------------------------
  // Barrier
  //-----------------------------
  output logic        snax_barrier_o,
  //-----------------------------
  // TCDM ports
  //-----------------------------
  output tcdm_req_t [SnaxTcdmPorts-1:0] snax_tcdm_req_o,
  input  tcdm_rsp_t [SnaxTcdmPorts-1:0] snax_tcdm_rsp_i
);

  //-----------------------------
  // Internal local parameters
  //-----------------------------
  localparam int unsigned NumRwCsr = 7;
  localparam int unsigned NumRoCsr = 4;

  //-----------------------------
  // Accelerator ports
  //-----------------------------
  // Note that these have very specific data widths
  // found from the configuration file

  // Ports from accelerator to streamer by writer data movers
  logic [63:0] acc2stream_0_data;
  logic acc2stream_0_valid;
  logic acc2stream_0_ready;

  logic [63:0] acc2stream_1_data;
  logic acc2stream_1_valid;
  logic acc2stream_1_ready;

  logic [63:0] acc2stream_2_data;
  logic acc2stream_2_valid;
  logic acc2stream_2_ready;

  logic [255:0] acc2stream_3_data;
  logic acc2stream_3_valid;
  logic acc2stream_3_ready;

  // Ports from streamer to accelerator by reader data movers
  logic [255:0] stream2acc_0_data;
  logic stream2acc_0_valid;
  logic stream2acc_0_ready;

  logic [383:0] stream2acc_1_data;
  logic stream2acc_1_valid;
  logic stream2acc_1_ready;

  logic [127:0] stream2acc_2_data;
  logic stream2acc_2_valid;
  logic stream2acc_2_ready;

  logic [63:0] stream2acc_3_data;
  logic stream2acc_3_valid;
  logic stream2acc_3_ready;

  logic [63:0] stream2acc_4_data;
  logic stream2acc_4_valid;
  logic stream2acc_4_ready;

  logic [63:0] stream2acc_5_data;
  logic stream2acc_5_valid;
  logic stream2acc_5_ready;

  logic [63:0] stream2acc_6_data;
  logic stream2acc_6_valid;
  logic stream2acc_6_ready;

  logic [511:0] stream2acc_7_data;
  logic stream2acc_7_valid;
  logic stream2acc_7_ready;

  logic [63:0] stream2acc_8_data;
  logic stream2acc_8_valid;
  logic stream2acc_8_ready;

  logic [63:0] stream2acc_9_data;
  logic stream2acc_9_valid;
  logic stream2acc_9_ready;

  logic [63:0] stream2acc_10_data;
  logic stream2acc_10_valid;
  logic stream2acc_10_ready;

  logic [63:0] stream2acc_11_data;
  logic stream2acc_11_valid;
  logic stream2acc_11_ready;

  logic [383:0] stream2acc_12_data;
  logic stream2acc_12_valid;
  logic stream2acc_12_ready;

  logic [255:0] stream2acc_13_data;
  logic stream2acc_13_valid;
  logic stream2acc_13_ready;

  // CSR MUXing
  logic [1:0][RegAddrWidth-1:0] acc_csr_req_addr;
  logic [1:0][RegDataWidth-1:0] acc_csr_req_data;
  logic [1:0]                   acc_csr_req_wen;
  logic [1:0]                   acc_csr_req_valid;
  logic [1:0]                   acc_csr_req_ready;
  logic [1:0][RegDataWidth-1:0] acc_csr_rsp_data;
  logic [1:0]                   acc_csr_rsp_valid;
  logic [1:0]                   acc_csr_rsp_ready;

  // Register set signals
  logic [NumRwCsr-1:0][31:0]    acc_csr_reg_rw_set;
  logic                         acc_csr_reg_set_valid;
  logic                         acc_csr_reg_set_ready;
  logic [NumRoCsr-1:0][31:0]    acc_csr_reg_ro_set;

  //-------------------------------
  // MUX and DEMUX for control signals
  // That separate between streamer CSRs
  // and accelerator CSRs
  //-------------------------------
  snax_csr_mux_demux #(
    // The AddrSelOffset indicates when the MUX switches
    // To streamer or the accelerator CSRs
    // If snax_req_addr_i <  AddrSelOffset, it points
    // to the accelerator CSR manager
    .AddrSelOffSet        ( 203     ), // Streamer CSR number
    .RegDataWidth         ( RegDataWidth      ),
    .RegAddrWidth         ( RegAddrWidth      )
  ) i_snax_csr_mux_demux (
    //-------------------------------
    // Input Core
    //-------------------------------
    .csr_req_addr_i       ( snax_req_addr_i   ),
    .csr_req_data_i       ( snax_req_data_i   ),
    .csr_req_wen_i        ( snax_req_write_i  ),
    .csr_req_valid_i      ( snax_req_valid_i  ),
    .csr_req_ready_o      ( snax_req_ready_o  ),
    .csr_rsp_data_o       ( snax_rsp_data_o   ),
    .csr_rsp_valid_o      ( snax_rsp_valid_o  ),
    .csr_rsp_ready_i      ( snax_rsp_ready_i  ),

    //-------------------------------
    // Output Port
    //-------------------------------
    .acc_csr_req_addr_o   ( acc_csr_req_addr  ),
    .acc_csr_req_data_o   ( acc_csr_req_data  ),
    .acc_csr_req_wen_o    ( acc_csr_req_wen   ),
    .acc_csr_req_valid_o  ( acc_csr_req_valid ),
    .acc_csr_req_ready_i  ( acc_csr_req_ready ),
    .acc_csr_rsp_data_i   ( acc_csr_rsp_data  ),
    .acc_csr_rsp_valid_i  ( acc_csr_rsp_valid ),
    .acc_csr_rsp_ready_o  ( acc_csr_rsp_ready )
  );

  //-----------------------------
  // CSR Manager to control the accelerator
  //-----------------------------
  snax_simbacore_csrman_wrapper #(
    .NumRwCsr             ( NumRwCsr              ),
    .NumRoCsr             ( NumRoCsr              )
  ) i_snax_simbacore_csrman_wrapper (
    //-------------------------------
    // Clocks and reset
    //-------------------------------
    .clk_i                ( clk_i                 ),
    .rst_ni               ( rst_ni                ),
    //-----------------------------
    // CSR control ports
    //-----------------------------
    // Request
    .csr_req_addr_i       ( acc_csr_req_addr [1]  ),
    .csr_req_data_i       ( acc_csr_req_data [1]  ),
    .csr_req_write_i      ( acc_csr_req_wen  [1]  ),
    .csr_req_valid_i      ( acc_csr_req_valid[1]  ),
    .csr_req_ready_o      ( acc_csr_req_ready[1]  ),

    // Response
    .csr_rsp_data_o       ( acc_csr_rsp_data [1]  ),
    .csr_rsp_valid_o      ( acc_csr_rsp_valid[1]  ),
    .csr_rsp_ready_i      ( acc_csr_rsp_ready[1]  ),

    //-----------------------------
    // Packed CSR register signals
    //-----------------------------
    // Read-write CSRs
    .csr_reg_rw_set_o     ( acc_csr_reg_rw_set    ),
    .csr_reg_set_valid_o  ( acc_csr_reg_set_valid ),
    .csr_reg_set_ready_i  ( acc_csr_reg_set_ready ),
    // Read-only CSRs
    .csr_reg_ro_set_i     ( acc_csr_reg_ro_set    )  
  );

  //-----------------------------
  // Accelerator
  //-----------------------------
  // Note: This is the part that needs to be consistent
  // It needs to have the correct connections to the control and data ports!
  // Parameter declarations are custom inside the shell
  // Do not pass it from the top towards here to achieve
  // Uniform shell wrapping :)
  
  snax_simbacore_shell_wrapper i_snax_simbacore_shell_wrapper (
    //-------------------------------
    // Clocks and reset
    //-------------------------------
    .clk_i                ( clk_i                 ),
    .rst_ni               ( rst_ni                ),

    //-----------------------------
    // Accelerator ports
    //-----------------------------
    // Note that these have very specific data widths
    // found from the configuration file
    // Ports from accelerator to streamer by writer data movers
    .acc2stream_0_data_o  ( acc2stream_0_data  ),
    .acc2stream_0_valid_o ( acc2stream_0_valid ),
    .acc2stream_0_ready_i ( acc2stream_0_ready ),

    .acc2stream_1_data_o  ( acc2stream_1_data  ),
    .acc2stream_1_valid_o ( acc2stream_1_valid ),
    .acc2stream_1_ready_i ( acc2stream_1_ready ),

    .acc2stream_2_data_o  ( acc2stream_2_data  ),
    .acc2stream_2_valid_o ( acc2stream_2_valid ),
    .acc2stream_2_ready_i ( acc2stream_2_ready ),

    .acc2stream_3_data_o  ( acc2stream_3_data  ),
    .acc2stream_3_valid_o ( acc2stream_3_valid ),
    .acc2stream_3_ready_i ( acc2stream_3_ready ),

    // Ports from streamer to accelerator by reader data movers
    .stream2acc_0_data_i  ( stream2acc_0_data  ),
    .stream2acc_0_valid_i ( stream2acc_0_valid ),
    .stream2acc_0_ready_o ( stream2acc_0_ready ),

    .stream2acc_1_data_i  ( stream2acc_1_data  ),
    .stream2acc_1_valid_i ( stream2acc_1_valid ),
    .stream2acc_1_ready_o ( stream2acc_1_ready ),

    .stream2acc_2_data_i  ( stream2acc_2_data  ),
    .stream2acc_2_valid_i ( stream2acc_2_valid ),
    .stream2acc_2_ready_o ( stream2acc_2_ready ),

    .stream2acc_3_data_i  ( stream2acc_3_data  ),
    .stream2acc_3_valid_i ( stream2acc_3_valid ),
    .stream2acc_3_ready_o ( stream2acc_3_ready ),

    .stream2acc_4_data_i  ( stream2acc_4_data  ),
    .stream2acc_4_valid_i ( stream2acc_4_valid ),
    .stream2acc_4_ready_o ( stream2acc_4_ready ),

    .stream2acc_5_data_i  ( stream2acc_5_data  ),
    .stream2acc_5_valid_i ( stream2acc_5_valid ),
    .stream2acc_5_ready_o ( stream2acc_5_ready ),

    .stream2acc_6_data_i  ( stream2acc_6_data  ),
    .stream2acc_6_valid_i ( stream2acc_6_valid ),
    .stream2acc_6_ready_o ( stream2acc_6_ready ),

    .stream2acc_7_data_i  ( stream2acc_7_data  ),
    .stream2acc_7_valid_i ( stream2acc_7_valid ),
    .stream2acc_7_ready_o ( stream2acc_7_ready ),

    .stream2acc_8_data_i  ( stream2acc_8_data  ),
    .stream2acc_8_valid_i ( stream2acc_8_valid ),
    .stream2acc_8_ready_o ( stream2acc_8_ready ),

    .stream2acc_9_data_i  ( stream2acc_9_data  ),
    .stream2acc_9_valid_i ( stream2acc_9_valid ),
    .stream2acc_9_ready_o ( stream2acc_9_ready ),

    .stream2acc_10_data_i  ( stream2acc_10_data  ),
    .stream2acc_10_valid_i ( stream2acc_10_valid ),
    .stream2acc_10_ready_o ( stream2acc_10_ready ),

    .stream2acc_11_data_i  ( stream2acc_11_data  ),
    .stream2acc_11_valid_i ( stream2acc_11_valid ),
    .stream2acc_11_ready_o ( stream2acc_11_ready ),

    .stream2acc_12_data_i  ( stream2acc_12_data  ),
    .stream2acc_12_valid_i ( stream2acc_12_valid ),
    .stream2acc_12_ready_o ( stream2acc_12_ready ),

    .stream2acc_13_data_i  ( stream2acc_13_data  ),
    .stream2acc_13_valid_i ( stream2acc_13_valid ),
    .stream2acc_13_ready_o ( stream2acc_13_ready ),


    //-----------------------------
    // Packed CSR register signals
    //-----------------------------
    .csr_reg_set_i        ( acc_csr_reg_rw_set    ),
    .csr_reg_set_valid_i  ( acc_csr_reg_set_valid ),
    .csr_reg_set_ready_o  ( acc_csr_reg_set_ready ),
    .csr_reg_ro_set_o     ( acc_csr_reg_ro_set    )
  );

  //-----------------------------
  // Streamer Wrapper
  //-----------------------------
  snax_simbacore_streamer_wrapper #(
    .tcdm_req_t               ( tcdm_req_t    ),
    .tcdm_rsp_t               ( tcdm_rsp_t    ),
    .TCDMDataWidth            ( DataWidth     ),
    .TCDMNumPorts             ( SnaxTcdmPorts ),
    .TCDMAddrWidth            ( TCDMAddrWidth )
  ) i_snax_simbacore_streamer_wrapper (
    //-----------------------------
    // Clocks and reset
    //-----------------------------
    .clk_i                    ( clk_i  ),
    .rst_ni                   ( rst_ni ),
    //-----------------------------
    // Accelerator ports
    //-----------------------------
    // Note that these have very specific data widths
    // found from the configuration file

    // Ports from accelerator to streamer by writer data movers
    .acc2stream_0_data_i  ( acc2stream_0_data  ),
    .acc2stream_0_valid_i ( acc2stream_0_valid ),
    .acc2stream_0_ready_o ( acc2stream_0_ready ),

    .acc2stream_1_data_i  ( acc2stream_1_data  ),
    .acc2stream_1_valid_i ( acc2stream_1_valid ),
    .acc2stream_1_ready_o ( acc2stream_1_ready ),

    .acc2stream_2_data_i  ( acc2stream_2_data  ),
    .acc2stream_2_valid_i ( acc2stream_2_valid ),
    .acc2stream_2_ready_o ( acc2stream_2_ready ),

    .acc2stream_3_data_i  ( acc2stream_3_data  ),
    .acc2stream_3_valid_i ( acc2stream_3_valid ),
    .acc2stream_3_ready_o ( acc2stream_3_ready ),

    // Ports from streamer to accelerator by reader data movers
    .stream2acc_0_data_o  ( stream2acc_0_data  ),
    .stream2acc_0_valid_o ( stream2acc_0_valid ),
    .stream2acc_0_ready_i ( stream2acc_0_ready ),

    .stream2acc_1_data_o  ( stream2acc_1_data  ),
    .stream2acc_1_valid_o ( stream2acc_1_valid ),
    .stream2acc_1_ready_i ( stream2acc_1_ready ),

    .stream2acc_2_data_o  ( stream2acc_2_data  ),
    .stream2acc_2_valid_o ( stream2acc_2_valid ),
    .stream2acc_2_ready_i ( stream2acc_2_ready ),

    .stream2acc_3_data_o  ( stream2acc_3_data  ),
    .stream2acc_3_valid_o ( stream2acc_3_valid ),
    .stream2acc_3_ready_i ( stream2acc_3_ready ),

    .stream2acc_4_data_o  ( stream2acc_4_data  ),
    .stream2acc_4_valid_o ( stream2acc_4_valid ),
    .stream2acc_4_ready_i ( stream2acc_4_ready ),

    .stream2acc_5_data_o  ( stream2acc_5_data  ),
    .stream2acc_5_valid_o ( stream2acc_5_valid ),
    .stream2acc_5_ready_i ( stream2acc_5_ready ),

    .stream2acc_6_data_o  ( stream2acc_6_data  ),
    .stream2acc_6_valid_o ( stream2acc_6_valid ),
    .stream2acc_6_ready_i ( stream2acc_6_ready ),

    .stream2acc_7_data_o  ( stream2acc_7_data  ),
    .stream2acc_7_valid_o ( stream2acc_7_valid ),
    .stream2acc_7_ready_i ( stream2acc_7_ready ),

    .stream2acc_8_data_o  ( stream2acc_8_data  ),
    .stream2acc_8_valid_o ( stream2acc_8_valid ),
    .stream2acc_8_ready_i ( stream2acc_8_ready ),

    .stream2acc_9_data_o  ( stream2acc_9_data  ),
    .stream2acc_9_valid_o ( stream2acc_9_valid ),
    .stream2acc_9_ready_i ( stream2acc_9_ready ),

    .stream2acc_10_data_o  ( stream2acc_10_data  ),
    .stream2acc_10_valid_o ( stream2acc_10_valid ),
    .stream2acc_10_ready_i ( stream2acc_10_ready ),

    .stream2acc_11_data_o  ( stream2acc_11_data  ),
    .stream2acc_11_valid_o ( stream2acc_11_valid ),
    .stream2acc_11_ready_i ( stream2acc_11_ready ),

    .stream2acc_12_data_o  ( stream2acc_12_data  ),
    .stream2acc_12_valid_o ( stream2acc_12_valid ),
    .stream2acc_12_ready_i ( stream2acc_12_ready ),

    .stream2acc_13_data_o  ( stream2acc_13_data  ),
    .stream2acc_13_valid_o ( stream2acc_13_valid ),
    .stream2acc_13_ready_i ( stream2acc_13_ready ),

    //-----------------------------
    // TCDM ports
    //-----------------------------
    .tcdm_req_o ( snax_tcdm_req_o ),
    .tcdm_rsp_i ( snax_tcdm_rsp_i ),

    //-----------------------------
    // CSR control ports
    //-----------------------------
    // Request
    .csr_req_bits_data_i   ( acc_csr_req_data [0] ),
    .csr_req_bits_addr_i   ( acc_csr_req_addr [0] ),
    .csr_req_bits_write_i  ( acc_csr_req_wen  [0] ),
    .csr_req_valid_i       ( acc_csr_req_valid[0] ),
    .csr_req_ready_o       ( acc_csr_req_ready[0] ),
    // Response
    .csr_rsp_bits_data_o   ( acc_csr_rsp_data [0] ),
    .csr_rsp_valid_o       ( acc_csr_rsp_valid[0] ),
    .csr_rsp_ready_i       ( acc_csr_rsp_ready[0] )
  );

  //-----------------------------
  // Barrier control
  // TODO: Fix me later
  //-----------------------------
  assign snax_barrier_o = '0;

endmodule
