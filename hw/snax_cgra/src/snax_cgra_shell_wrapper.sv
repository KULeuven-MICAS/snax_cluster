// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

//-------------------------------
// CGRA Shell Wrapper
//-------------------------------


module snax_cgra_shell_wrapper # (
  //---------------------------
  // Outer Definitions
  //---------------------------
  parameter int unsigned NarrowDataWidth = 64,
  parameter int unsigned WideDataWidth   = 512,
  parameter int unsigned RegAddrWidth    = 32,
  parameter int unsigned RegDataWidth    = 32,
  parameter int unsigned NumRwCsr = 1,
  parameter int unsigned NumRoCsr = 4,
  //---------------------------
  // General Parameters
  //---------------------------
  parameter int unsigned NumNarrowDataIn     = 8,
  parameter int unsigned NumNarrowDataOut     = 8
)(
  //-----------------------------
  // Clock and reset
  //-----------------------------
  input  logic                       clk_i,
  input  logic                       rst_ni,

  //-----------------------------
  // Accelerator ports
  //-----------------------------
  // Ports from accelerator to streamer by writer data movers
  output logic [NarrowDataWidth-1:0] acc2stream_0_data_o,
  output logic                       acc2stream_0_valid_o,
  input  logic                       acc2stream_0_ready_i,

  output logic [NarrowDataWidth-1:0] acc2stream_1_data_o,
  output logic                       acc2stream_1_valid_o,
  input  logic                       acc2stream_1_ready_i,

  output logic [NarrowDataWidth-1:0] acc2stream_2_data_o,
  output logic                       acc2stream_2_valid_o,
  input  logic                       acc2stream_2_ready_i,

  output logic [NarrowDataWidth-1:0] acc2stream_3_data_o,
  output logic                       acc2stream_3_valid_o,
  input  logic                       acc2stream_3_ready_i,

  output logic [NarrowDataWidth-1:0] acc2stream_4_data_o,
  output logic                       acc2stream_4_valid_o,
  input  logic                       acc2stream_4_ready_i,

  output logic [NarrowDataWidth-1:0] acc2stream_5_data_o,
  output logic                       acc2stream_5_valid_o,
  input  logic                       acc2stream_5_ready_i,

  output logic [NarrowDataWidth-1:0] acc2stream_6_data_o,
  output logic                       acc2stream_6_valid_o,
  input  logic                       acc2stream_6_ready_i,

  output logic [NarrowDataWidth-1:0] acc2stream_7_data_o,
  output logic                       acc2stream_7_valid_o,
  input  logic                       acc2stream_7_ready_i,

  // Ports from streamer to accelerator by reader data movers
  input  logic [NarrowDataWidth-1:0] stream2acc_0_data_i,
  input  logic                       stream2acc_0_valid_i,
  output logic                       stream2acc_0_ready_o,

  input  logic [NarrowDataWidth-1:0] stream2acc_1_data_i,
  input  logic                       stream2acc_1_valid_i,
  output logic                       stream2acc_1_ready_o,

  input  logic [NarrowDataWidth-1:0] stream2acc_2_data_i,
  input  logic                       stream2acc_2_valid_i,
  output logic                       stream2acc_2_ready_o,

  input  logic [NarrowDataWidth-1:0] stream2acc_3_data_i,
  input  logic                       stream2acc_3_valid_i,
  output logic                       stream2acc_3_ready_o,

  input  logic [NarrowDataWidth-1:0] stream2acc_4_data_i,
  input  logic                       stream2acc_4_valid_i,
  output logic                       stream2acc_4_ready_o,

  input  logic [NarrowDataWidth-1:0] stream2acc_5_data_i,
  input  logic                       stream2acc_5_valid_i,
  output logic                       stream2acc_5_ready_o,

  input  logic [NarrowDataWidth-1:0] stream2acc_6_data_i,
  input  logic                       stream2acc_6_valid_i,
  output logic                       stream2acc_6_ready_o,

  input  logic [NarrowDataWidth-1:0] stream2acc_7_data_i,
  input  logic                       stream2acc_7_valid_i,
  output logic                       stream2acc_7_ready_o,

  //-----------------------------
  // Packed CSR register signals
  //-----------------------------
  // Read-write CSRs
  input logic [NumRwCsr-1:0][31:0]    csr_reg_set_i,
  input logic                         csr_reg_set_valid_i,
  output logic                        csr_reg_set_ready_o,
  output logic [NumRoCsr-1:0][31:0]    csr_reg_ro_set_o
);

  //---------------------------
  // IO Port packing
  //---------------------------
  logic [ NarrowDataWidth-1:0] cgra_ni_input_data_i [NumNarrowDataIn];
  logic cgra_ni_input_valid_i [NumNarrowDataIn];
  logic cgra_ni_input_ready_o [NumNarrowDataIn];
  logic [ NarrowDataWidth-1:0] cgra_ni_output_data_o [NumNarrowDataOut];
  logic cgra_ni_output_valid_o [NumNarrowDataOut];
  logic cgra_ni_output_ready_i [NumNarrowDataOut];
  logic [31:0] cgra_csr_reg_ro_set_o [NumRoCsr];
  logic [31:0] cgra_csr_reg_set_i [NumRwCsr];

  assign cgra_ni_input_data_i[0] = stream2acc_0_data_i;
  assign cgra_ni_input_data_i[1] = stream2acc_1_data_i;
  assign cgra_ni_input_data_i[2] = stream2acc_2_data_i;
  assign cgra_ni_input_data_i[3] = stream2acc_3_data_i;
  assign cgra_ni_input_data_i[4] = stream2acc_4_data_i;
  assign cgra_ni_input_data_i[5] = stream2acc_5_data_i;
  assign cgra_ni_input_data_i[6] = stream2acc_6_data_i;
  assign cgra_ni_input_data_i[7] = stream2acc_7_data_i;

  assign cgra_ni_input_valid_i[0] = stream2acc_0_valid_i;
  assign cgra_ni_input_valid_i[1] = stream2acc_1_valid_i;
  assign cgra_ni_input_valid_i[2] = stream2acc_2_valid_i;
  assign cgra_ni_input_valid_i[3] = stream2acc_3_valid_i;
  assign cgra_ni_input_valid_i[4] = stream2acc_4_valid_i;
  assign cgra_ni_input_valid_i[5] = stream2acc_5_valid_i;
  assign cgra_ni_input_valid_i[6] = stream2acc_6_valid_i;
  assign cgra_ni_input_valid_i[7] = stream2acc_7_valid_i;

  assign stream2acc_0_ready_o = cgra_ni_input_ready_o[0];
  assign stream2acc_1_ready_o = cgra_ni_input_ready_o[1];
  assign stream2acc_2_ready_o = cgra_ni_input_ready_o[2];
  assign stream2acc_3_ready_o = cgra_ni_input_ready_o[3];
  assign stream2acc_4_ready_o = cgra_ni_input_ready_o[4];
  assign stream2acc_5_ready_o = cgra_ni_input_ready_o[5];
  assign stream2acc_6_ready_o = cgra_ni_input_ready_o[6];
  assign stream2acc_7_ready_o = cgra_ni_input_ready_o[7];

  assign acc2stream_0_data_o = cgra_ni_output_data_o[0];
  assign acc2stream_1_data_o = cgra_ni_output_data_o[1];
  assign acc2stream_2_data_o = cgra_ni_output_data_o[2];
  assign acc2stream_3_data_o = cgra_ni_output_data_o[3];
  assign acc2stream_4_data_o = cgra_ni_output_data_o[4];
  assign acc2stream_5_data_o = cgra_ni_output_data_o[5];
  assign acc2stream_6_data_o = cgra_ni_output_data_o[6];
  assign acc2stream_7_data_o = cgra_ni_output_data_o[7];

  assign acc2stream_0_valid_o = cgra_ni_output_valid_o[0];
  assign acc2stream_1_valid_o = cgra_ni_output_valid_o[1];
  assign acc2stream_2_valid_o = cgra_ni_output_valid_o[2];
  assign acc2stream_3_valid_o = cgra_ni_output_valid_o[3];
  assign acc2stream_4_valid_o = cgra_ni_output_valid_o[4];
  assign acc2stream_5_valid_o = cgra_ni_output_valid_o[5];
  assign acc2stream_6_valid_o = cgra_ni_output_valid_o[6];
  assign acc2stream_7_valid_o = cgra_ni_output_valid_o[7];

  assign cgra_ni_output_ready_i[0] = acc2stream_0_ready_i;
  assign cgra_ni_output_ready_i[1] = acc2stream_1_ready_i;
  assign cgra_ni_output_ready_i[2] = acc2stream_2_ready_i;
  assign cgra_ni_output_ready_i[3] = acc2stream_3_ready_i;
  assign cgra_ni_output_ready_i[4] = acc2stream_4_ready_i;
  assign cgra_ni_output_ready_i[5] = acc2stream_5_ready_i;
  assign cgra_ni_output_ready_i[6] = acc2stream_6_ready_i;
  assign cgra_ni_output_ready_i[7] = acc2stream_7_ready_i;

  assign csr_reg_ro_set_o[0] = cgra_csr_reg_ro_set_o[0];
  assign csr_reg_ro_set_o[1] = cgra_csr_reg_ro_set_o[1];
  assign csr_reg_ro_set_o[2] = cgra_csr_reg_ro_set_o[2];
  assign csr_reg_ro_set_o[3] = cgra_csr_reg_ro_set_o[3];

  assign cgra_csr_reg_set_i[0] = csr_reg_set_i[0];

  //---------------------------
  // CGRA Top Module
  //---------------------------
  CGRARTL__top i_cgrartl (
    //---------------------------
    // Clocks and reset
    //---------------------------
    .clk                      ( clk_i                    ),
    .reset                    ( !rst_ni                  ),
    //---------------------------
    // CSR RW control signals
    //---------------------------
    .cgra_csr_rw              ( cgra_csr_reg_set_i       ),
    .cgra_csr_rw_ack          ( csr_reg_set_ready_o      ),
    .cgra_csr_rw_valid        ( csr_reg_set_valid_i      ),
    //---------------------------
    // CSR RO control signals
    //---------------------------
    .cgra_csr_ro              ( cgra_csr_reg_ro_set_o   ),
    //---------------------------
    // INPUT ports
    //---------------------------
    .cgra_recv_ni_data__en    ( cgra_ni_input_valid_i   ),
    .cgra_recv_ni_data__msg   ( cgra_ni_input_data_i    ),
    .cgra_recv_ni_data__rdy   ( cgra_ni_input_ready_o   ),
    //---------------------------
    // OUTPUT ports
    //---------------------------
    .cgra_send_ni_data__en    ( cgra_ni_output_valid_o  ),
    .cgra_send_ni_data__msg   ( cgra_ni_output_data_o   ),
    .cgra_send_ni_data__rdy   ( cgra_ni_output_ready_i  )
  );


endmodule
