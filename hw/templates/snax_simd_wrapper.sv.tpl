// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

<%
  # The SIMD block takes num_channel read ports followed by num_channel write ports.
  # Unlike the xDMA wrapper this is a cfg knob, not a fixed dma_data_width/data_width
  # ratio: the SIMD beat width is sized to the lane count, not to the DMA beat.
  simd_num_channel = cfg["simd_num_channel"]
  num_tcdm_ports   = simd_num_channel * 2

  tcdm_addr_width = cfg["tcdm"]["size"].bit_length() - 1 + 10

  try:
    simd_cfg_io_width = cfg["simd_cfg_io_width"]
  except Exception:
    simd_cfg_io_width = 32
%>
//-----------------------------
// SIMD wrapper
//
// The reader-side operator bank of the xDMA, on its own core: CSR in, TCDM in and out, nothing else.
// There is deliberately no AXI adapter here -- the SIMD engine is local TCDM<->TCDM only, which is what
// keeps the cluster's single set of inter-cluster xdma_* ports owned by the xDMA alone.
//-----------------------------
module ${cfg["name"]}_simd_wrapper
#(
  // TCDM typedefs
  parameter type         tcdm_req_t    = logic,
  parameter type         tcdm_rsp_t    = logic,
  // Parameters related to TCDM
  parameter int unsigned TCDMDataWidth = ${cfg["data_width"]},
  parameter int unsigned TCDMNumPorts  = ${num_tcdm_ports},
  parameter int unsigned TCDMAddrWidth = ${tcdm_addr_width}
)(
  //-----------------------------
  // Clocks and reset
  //-----------------------------
  input  logic clk_i,
  input  logic rst_ni,
  //-----------------------------
  // TCDM ports
  //-----------------------------
  output tcdm_req_t [TCDMNumPorts-1:0] tcdm_req_o,
  input  tcdm_rsp_t [TCDMNumPorts-1:0] tcdm_rsp_i,
  //-----------------------------
  // CSR control ports
  //-----------------------------
  // Request
  input  logic [31:0] csr_req_bits_addr_i,
  input  logic        csr_req_bits_write_i,
  input  logic [${simd_cfg_io_width-1}:0] csr_req_bits_data_i,
  input  logic [${int(simd_cfg_io_width/8)-1}:0] csr_req_bits_strb_i,
  input  logic        csr_req_valid_i,
  output logic        csr_req_ready_o,
  // Response
  output logic [${simd_cfg_io_width-1}:0] csr_rsp_bits_data_o,
  output logic        csr_rsp_valid_o,
  input  logic        csr_rsp_ready_i,
  //-----------------------------
  // Status
  //-----------------------------
  output logic        busy_o
);

  //-----------------------------
  // Wiring and combinational logic
  //-----------------------------

  // TCDM signals
  // Request
  logic [TCDMNumPorts-1:0][TCDMAddrWidth-1:0]   tcdm_req_addr;
  logic [TCDMNumPorts-1:0]                      tcdm_req_write;
  logic [TCDMNumPorts-1:0][  TCDMDataWidth-1:0] tcdm_req_data;
  logic [TCDMNumPorts-1:0][TCDMDataWidth/8-1:0] tcdm_req_strb;
  logic [TCDMNumPorts-1:0]                      tcdm_req_q_valid;
  logic [TCDMNumPorts-1:0]                      tcdm_req_priority;

  // Response
  logic [TCDMNumPorts-1:0]                      tcdm_rsp_q_ready;
  logic [TCDMNumPorts-1:0]                      tcdm_rsp_p_valid;
  logic [TCDMNumPorts-1:0][  TCDMDataWidth-1:0] tcdm_rsp_data;

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
      tcdm_req_o[i].q.user.tcdm_priority = tcdm_req_priority [i];
      tcdm_req_o[i].q_valid        = tcdm_req_q_valid[i];

      tcdm_rsp_q_ready[i] = tcdm_rsp_i[i].q_ready;
      tcdm_rsp_p_valid[i] = tcdm_rsp_i[i].p_valid;
      tcdm_rsp_data   [i] = tcdm_rsp_i[i].p.data ;
    end
  end

  // The generated SIMD block
  ${cfg["name"]}_simd i_${cfg["name"]}_simd (
    //-----------------------------
    // Clocks and reset
    //-----------------------------
    .clock ( clk_i   ),
    .reset ( ~rst_ni ),
    //-----------------------------
    // TCDM Ports
    //-----------------------------
    // Reader's Request. As in the xDMA wrapper: the request-side ready lives in the response struct.
% for idx in range(0, simd_num_channel):
    .io_tcdmReader_req_${idx}_ready         (tcdm_rsp_q_ready [${idx}]),
    .io_tcdmReader_req_${idx}_valid         (tcdm_req_q_valid [${idx}]),
    .io_tcdmReader_req_${idx}_bits_addr     (tcdm_req_addr    [${idx}]),
    .io_tcdmReader_req_${idx}_bits_write    (tcdm_req_write   [${idx}]),
    .io_tcdmReader_req_${idx}_bits_data     (tcdm_req_data    [${idx}]),
    .io_tcdmReader_req_${idx}_bits_strb     (tcdm_req_strb    [${idx}]),
    .io_tcdmReader_req_${idx}_bits_priority (tcdm_req_priority[${idx}]),
% endfor
    // Writer's Request
% for idx in range(0, simd_num_channel):
    .io_tcdmWriter_req_${idx}_ready         (tcdm_rsp_q_ready [${idx + simd_num_channel}]),
    .io_tcdmWriter_req_${idx}_valid         (tcdm_req_q_valid [${idx + simd_num_channel}]),
    .io_tcdmWriter_req_${idx}_bits_addr     (tcdm_req_addr    [${idx + simd_num_channel}]),
    .io_tcdmWriter_req_${idx}_bits_write    (tcdm_req_write   [${idx + simd_num_channel}]),
    .io_tcdmWriter_req_${idx}_bits_data     (tcdm_req_data    [${idx + simd_num_channel}]),
    .io_tcdmWriter_req_${idx}_bits_strb     (tcdm_req_strb    [${idx + simd_num_channel}]),
    .io_tcdmWriter_req_${idx}_bits_priority (tcdm_req_priority[${idx + simd_num_channel}]),
% endfor
    // Reader's Response
% for idx in range(0, simd_num_channel):
    .io_tcdmReader_rsp_${idx}_valid         (tcdm_rsp_p_valid [${idx}]),
    .io_tcdmReader_rsp_${idx}_bits_data     (tcdm_rsp_data    [${idx}]),
% endfor
    // Writer has no Response
    //-----------------------------
    // CSR control ports
    //-----------------------------
    // Request
    .io_csrIO_req_bits_data  ( csr_req_bits_data_i  ),
    .io_csrIO_req_bits_addr  ( csr_req_bits_addr_i  ),
    .io_csrIO_req_bits_write ( csr_req_bits_write_i ),
    .io_csrIO_req_bits_strb  ( csr_req_bits_strb_i  ),
    .io_csrIO_req_valid      ( csr_req_valid_i      ),
    .io_csrIO_req_ready      ( csr_req_ready_o      ),
    // Response
    .io_csrIO_rsp_bits_data  ( csr_rsp_bits_data_o  ),
    .io_csrIO_rsp_valid      ( csr_rsp_valid_o      ),
    .io_csrIO_rsp_ready      ( csr_rsp_ready_i      ),
    //-----------------------------
    // Status
    //-----------------------------
    .io_status_busy          ( busy_o               )
  );

  // The writer never issues a read, so its response ports do not exist on the block. Tie the corresponding
  // interconnect response fields off so nothing is left floating.
% for idx in range(0, simd_num_channel):
  /* verilator lint_off UNUSEDSIGNAL */
  logic unused_writer_rsp_${idx};
  assign unused_writer_rsp_${idx} = |{tcdm_rsp_p_valid[${idx + simd_num_channel}], tcdm_rsp_data[${idx + simd_num_channel}]};
  /* verilator lint_on UNUSEDSIGNAL */
% endfor

endmodule
