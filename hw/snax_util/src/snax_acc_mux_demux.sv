// Copyright 2020 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Ryan Antonio <ryan.antonio@esat.kuleuven.be>

// verilog_lint: waive-start line-length
// verilog_lint: waive-start no-trailing-spaces

module snax_acc_mux_demux #(
  parameter int unsigned NumAcc               = 2,
  parameter int unsigned CsrWidthList[NumAcc] = '{default: 0},
  parameter int unsigned RegDataWidth         = 32,
  parameter int unsigned RegAddrWidth         = 32
)(
  //------------------------
  // Clock and reset
  //------------------------
  input   logic                   clk_i,
  input   logic                   rst_ni,

  //------------------------
  // Main Snitch Port
  //------------------------
  input  logic [RegAddrWidth-1:0] csr_req_addr_i,
  input  logic [RegDataWidth-1:0] csr_req_data_i,
  input  logic                    csr_req_wen_i,
  input  logic                    csr_req_valid_i,
  output logic                    csr_req_ready_o,
  output logic [RegDataWidth-1:0] csr_rsp_data_o,
  output logic                    csr_rsp_valid_o,
  input  logic                    csr_rsp_ready_i,

  //------------------------
  // Split Ports
  //------------------------
  output logic [NumAcc-1:0][RegAddrWidth-1:0] acc_csr_req_addr_o,
  output logic [NumAcc-1:0][RegDataWidth-1:0] acc_csr_req_data_o,
  output logic [NumAcc-1:0][0:0]              acc_csr_req_wen_o,
  output logic [NumAcc-1:0][0:0]              acc_csr_req_valid_o,
  input  logic [NumAcc-1:0][0:0]              acc_csr_req_ready_i,
  input  logic [NumAcc-1:0][RegDataWidth-1:0] acc_csr_rsp_data_i,
  input  logic [NumAcc-1:0][0:0]              acc_csr_rsp_valid_i,
  output logic [NumAcc-1:0][0:0]              acc_csr_rsp_ready_o
);

  //------------------------------------------
  // Local parameters
  //------------------------------------------
  localparam int unsigned AccNumBitWidth = $clog2(NumAcc);

  //------------------------------------------
  // Accelerator Demux Port
  // For splitting the accelerator into parts
  //------------------------------------------
  logic [AccNumBitWidth-1:0] snax_req_sel;

  logic [31:0] internal_addr_sel;

  assign internal_addr_sel = csr_req_addr_i - 32'd960;

  // Pre-compute static addresses
  int BaseCsrAddress [NumAcc+1];
  
  initial begin
    BaseCsrAddress[0] = 0;
    for ( int i = 1; i < NumAcc; i ++ ) begin
      BaseCsrAddress[i] = BaseCsrAddress[i-1] + CsrWidthList[i-1];
    end
  end

  always_comb begin
    for ( int i = 0; i < NumAcc; i ++ ) begin
      // This one is for setting the control signals for the demuxer
      if((internal_addr_sel >= BaseCsrAddress[i]) &&
         (internal_addr_sel < BaseCsrAddress[i] + CsrWidthList[i]- 1)) begin
        snax_req_sel = i;
      end
      
      // This one broadcasts the control to all request ports
      acc_csr_req_addr_o[i] = csr_req_addr_i - BaseCsrAddress[i];
      acc_csr_req_data_o[i] = csr_req_data_i;
      acc_csr_req_wen_o [i] = csr_req_wen_i;
    end
  end

  stream_demux #(
    .N_OUP        ( NumAcc              )
  ) i_stream_demux_offload (
    .inp_valid_i  ( csr_req_valid_i     ),
    .inp_ready_o  ( csr_req_valid_o     ),
    .oup_sel_i    ( snax_req_sel        ),
    .oup_valid_o  ( acc_csr_req_valid_o ),
    .oup_ready_i  ( acc_csr_req_ready_i )
  );


  //------------------------------------------
  // Accelerator MUX Port
  // For handling multiple transactions
  //------------------------------------------
  typedef logic [RegDataWidth-1:0] data_t;

  stream_arbiter #(
    .DATA_T      ( data_t               ),
    .N_INP       ( NumAcc               )
  ) i_stream_arbiter_offload (
    .clk_i       ( clk_i                ),
    .rst_ni      ( rst_ni               ),
    .inp_data_i  ( acc_csr_rsp_data_i   ),
    .inp_valid_i ( acc_csr_rsp_valid_i  ),
    .inp_ready_o ( acc_csr_rsp_ready_o  ),
    .oup_data_o  ( csr_rsp_data_o       ),
    .oup_valid_o ( csr_rsp_valid_o      ),
    .oup_ready_i ( csr_rsp_ready_i      )
  );

endmodule

// ----- Module Usage -----
// snax_acc_mux_demux #(
//     .NumCsrs      (),
//     .NumAcc       (),
//     .RegDataWidth (),
//     .RegAddrWidth ()
// ) i_snax_acc_mux_demux (
//     //------------------------
//     // Clock and reset
//     //------------------------
//     .clk_i  (),
//     .rst_ni (),
//
//     //------------------------
//     // Main Snitch Port
//     //------------------------
//     .csr_req_addr_i  (),
//     .csr_req_data_i  (),
//     .csr_req_wen_i   (),
//     .csr_req_valid_i (),
//     .csr_req_ready_o (),
//     .csr_rsp_data_o  (),
//     .csr_rsp_valid_o (),
//     .csr_rsp_ready_i (),
//
//     //------------------------
//     // Split Ports
//     //------------------------
//     .acc_csr_req_addr_o  (),
//     .acc_csr_req_data_o  (),
//     .acc_csr_req_wen_o   (),
//     .acc_csr_req_valid_o (),
//     .acc_csr_req_ready_i (),
//     .acc_csr_rsp_data_i  (),
//     .acc_csr_rsp_valid_i (),
//     .acc_csr_rsp_ready_o ()
//   );
