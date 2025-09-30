// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

module snax_simbacore_csrman_wrapper #(
  parameter int unsigned NumRwCsr = 5,
  parameter int unsigned NumRoCsr = 1
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
  input  logic [31:0] csr_req_data_i,
  input  logic [31:0] csr_req_addr_i,
  input  logic        csr_req_write_i,
  input  logic        csr_req_valid_i,
  output logic        csr_req_ready_o,
  // Response
  output logic [31:0] csr_rsp_data_o,
  input  logic        csr_rsp_ready_i,
  output logic        csr_rsp_valid_o,
  //-----------------------------
  // Packed CSR register signals
  //-----------------------------
  output logic [NumRwCsr-1:0][31:0] csr_reg_rw_set_o,
  output logic                      csr_reg_set_valid_o,
  input  logic                      csr_reg_set_ready_i,
  input  logic [NumRoCsr-1:0][31:0] csr_reg_ro_set_i
);


  //-----------------------------
  // Chisel generated CSR manager
  //-----------------------------
  snax_simbacore_reqrspman_ReqRspManager i_snax_simbacore_reqrspman_ReqRspManager (
    //-----------------------------
    // Clocks and reset
    //-----------------------------
    .clock  ( clk_i  ),
    .reset  ( ~rst_ni ),

    //-----------------------------
    // Req Rsp ports
    //-----------------------------
    // Request ports
    .io_reqRspIO_req_bits_data        ( csr_req_data_i      ),
    .io_reqRspIO_req_bits_addr        ( csr_req_addr_i      ),
    .io_reqRspIO_req_bits_write       ( csr_req_write_i     ),
    .io_reqRspIO_req_bits_strb        ( '1                  ),
    .io_reqRspIO_req_valid            ( csr_req_valid_i     ),
    .io_reqRspIO_req_ready            ( csr_req_ready_o     ),

    // Response ports
    .io_reqRspIO_rsp_bits_data        ( csr_rsp_data_o      ),
    .io_reqRspIO_rsp_valid            ( csr_rsp_valid_o     ),
    .io_reqRspIO_rsp_ready            ( csr_rsp_ready_i     ),

    //-----------------------------
    // CSR configurations
    //-----------------------------
    .io_readWriteRegIO_bits_0      ( csr_reg_rw_set_o[0] ),
    .io_readWriteRegIO_bits_1      ( csr_reg_rw_set_o[1] ),
    .io_readWriteRegIO_bits_2      ( csr_reg_rw_set_o[2] ),
    .io_readWriteRegIO_bits_3      ( csr_reg_rw_set_o[3] ),
    .io_readWriteRegIO_bits_4      ( csr_reg_rw_set_o[4] ),
    .io_readOnlyReg_0              ( csr_reg_ro_set_i[0] ),
    .io_readWriteRegIO_valid          ( csr_reg_set_valid_o ),
    .io_readWriteRegIO_ready          ( csr_reg_set_ready_i )

  );

endmodule
