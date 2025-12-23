// Copyright 2020 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Ryan Antonio <ryan.antonio@esat.kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>
//-------------------------------
// SNAX interface translator for converting
// Snitch accelerator ports to
// CSR ports
//-------------------------------

import riscv_instr::*;
import reqrsp_pkg::*;
import csr_snax_def::*;
module snax_intf_translator #(
  parameter type                        acc_req_t = logic,
  parameter type                        acc_rsp_t = logic,
  parameter type                        csr_req_t = logic,
  parameter type                        csr_rsp_t = logic,
  // Careful! Sensitive parameter that depends
  // On the offset of where the CSRs are placed
  parameter int unsigned                NumOutstandingLoads = 4,
  parameter int unsigned                CsrAddrOffset = 32'h3c0
)(
  //-------------------------------
  // Clocks and reset
  //-------------------------------
  input  logic        clk_i,
  input  logic        rst_ni,

  //-------------------------------
  // Request
  //-------------------------------
  input  acc_req_t    snax_req_i,
  input  logic        snax_qvalid_i,
  output logic        snax_qready_o,

  //-------------------------------
  // Response
  //-------------------------------
  output acc_rsp_t    snax_resp_o,
  output logic        snax_pvalid_o,
  input  logic        snax_pready_i,

  //-----------------------------
  // Simplified CSR control ports
  //-----------------------------
  // [0] To ACC
  // [1] To Top-level
  // Request
  output csr_req_t         snax_csr_req_o,
  output logic             snax_csr_req_acc_valid_o,
  input  logic             snax_csr_req_acc_ready_i,
  output logic             snax_csr_req_top_valid_o,
  input  logic             snax_csr_req_top_ready_i,
  // Response
  input  csr_rsp_t         snax_csr_rsp_acc_i,
  input  logic             snax_csr_rsp_acc_valid_i,
  output logic             snax_csr_rsp_acc_ready_o,
  input  csr_rsp_t         snax_csr_rsp_top_i,
  input  logic             snax_csr_rsp_top_valid_i,
  output logic             snax_csr_rsp_top_ready_o

);

  //-------------------------------
  // Request handler
  //-------------------------------
  logic  write_csr;

  // Combinational logic to detect CSR
  // Write operations
  always_comb begin
    if (snax_qvalid_i) begin
      unique casez (snax_req_i.data_op)
          CSRRS, CSRRSI, CSRRC, CSRRCI: begin
              write_csr = 1'b0;
          end
          default: begin
              write_csr = 1'b1;
          end
        endcase
    end else begin
      write_csr = 1'b0;
    end
  end

  // We need a demux to split the requests
  logic snax_csr_req_sel;
  assign snax_csr_req_sel = (snax_req_i.data_argb[11:0]==CSR_SNAX_READ_TASK_READY_QUEUE) ||
                            (snax_req_i.data_argb[11:0]==CSR_SNAX_WRITE_TASK_DONE_QUEUE);
  stream_demux #(
    .N_OUP(2)
  ) i_snax_csr_req_demux (
    .inp_valid_i ( snax_qvalid_i        ),
    .inp_ready_o ( snax_qready_o        ),
    .oup_sel_i   ( snax_csr_req_sel     ),
    .oup_valid_o ( {snax_csr_req_top_valid_o, snax_csr_req_acc_valid_o}  ),
    .oup_ready_i ( {snax_csr_req_top_ready_i, snax_csr_req_acc_ready_i}  )
  );

  assign snax_csr_req_o.data  = snax_req_i.data_arga[31:0];
  assign snax_csr_req_o.addr  = snax_req_i.data_argb - CsrAddrOffset;
  assign snax_csr_req_o.write = write_csr;


  //-------------------------------
  // Response handler
  //-------------------------------

  // ID needs to be handled with a fifo buffer
  // we know that the responses will always
  // be in order, so we can just use a simple
  // fifo buffer to align the request id

  acc_req_t rsp_fifo_out;
  logic rsp_fifo_full, rsp_fifo_empty;
  logic rsp_fifo_push, rsp_fifo_pop;

  // Combinational logic

  // We push everytime there is a new read request
  // but then the response is not immediatley available
  // and when the fifo is not full!
  assign rsp_fifo_push =   snax_qvalid_i
                        && !write_csr
                        && !snax_csr_rsp_valid_i
                        && !rsp_fifo_full;

  // We pop when the response is valid and the fifo is not empty
  assign rsp_fifo_pop  =   snax_csr_rsp_valid_i
                        && !rsp_fifo_empty;

  // Buffer for aligning request id
  fifo_v3 #(
    .FALL_THROUGH ( 1'b0                ),
    .DEPTH        ( NumOutstandingLoads ),
    .dtype        ( acc_req_t           )
  ) i_rsp_fifo (
    .clk_i        ( clk_i               ),
    .rst_ni       ( rst_ni              ),
    .flush_i      ( 1'b0                ),
    .testmode_i   ( 1'b0                ),
    .full_o       ( rsp_fifo_full       ),
    .empty_o      ( rsp_fifo_empty      ),
    .usage_o      ( /* open */          ),
    .data_i       ( snax_req_i          ),
    .push_i       ( rsp_fifo_push       ),
    .data_o       ( rsp_fifo_out        ),
    .pop_i        ( rsp_fifo_pop        )
  );

  // Ready only when snax is ready and fifo is not full
  assign snax_csr_rsp_ready_o = snax_pready_i && !rsp_fifo_full;

  // We need a mux to select between the two response sources (ACC or Top-level)
  // The selection is based on the fifo out signal
  logic snax_csr_rsp_sel;
  assign snax_csr_rsp_sel = (rsp_fifo_out.data_argb[11:0]==CSR_SNAX_READ_TASK_READY_QUEUE) ||
                            (rsp_fifo_out.data_argb[11:0]==CSR_SNAX_WRITE_TASK_DONE_QUEUE);
  csr_rsp_t snax_csr_rsp;
  stream_mux #(
    .DATA_T   ( csr_rsp_t ),
    .N_INP    ( 2         )
  ) i_snax_csr_rsp_mux (
    .inp_data_i ( {snax_csr_rsp_top_i, snax_csr_rsp_acc_i}                ),
    .inp_valid_i( {snax_csr_rsp_top_valid_i, snax_csr_rsp_acc_valid_i}    ),
    .inp_ready_o( {snax_csr_rsp_top_ready_o, snax_csr_rsp_acc_ready_o}    ),
    .inp_sel_i  ( snax_csr_rsp_sel        ),
    .oup_data_o ( snax_csr_rsp            ),
    .oup_valid_o( snax_pvalid_o           ),
    .oup_ready_i( snax_pready_i           )
  );

  assign snax_resp_o.data     = snax_csr_rsp.data;
  // If fifo is not empty, use the one from the FIFO
  // Else just make it pass through
  assign snax_resp_o.id       = (!rsp_fifo_empty) ? rsp_fifo_out.id: snax_req_i.id;
  // Leave this as always no error for now
  assign snax_resp_o.error    = 1'b0;

endmodule
