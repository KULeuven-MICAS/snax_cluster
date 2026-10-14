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
  // Each request is one of (Zicsr):
  //   a) csrrw/csrrwi, rd == x0          -> one write of data_arga
  //   b) csrrw/csrrwi, rd != x0          -> a read, then a write of data_arga
  //   c) csrrs/c/si/ci, rs1/uimm == 0    -> one read
  //   d) csrrs/c/si/ci, rs1/uimm != 0    -> a read, then a write of
  //                                         old | mask (set) or old & ~mask (clear)
  // In b) and d) the core gets the read's data (the old value).
  logic  write_csr;
  logic  req_set_clear, req_clear;
  logic  req_rmw;

  // Combinational logic to detect CSR
  // set/clear operations
  always_comb begin
    req_set_clear = 1'b0;
    req_clear     = 1'b0;
    if (snax_qvalid_i) begin
      unique casez (snax_req_i.data_op)
          CSRRS, CSRRSI: begin
              req_set_clear = 1'b1;
          end
          CSRRC, CSRRCI: begin
              req_set_clear = 1'b1;
              req_clear     = 1'b1;
          end
          default: ;
        endcase
    end
  end

  // b) and d): read, then write
  assign req_rmw   = snax_qvalid_i && (req_set_clear ? (snax_req_i.data_op[19:15] != '0) :
                                                       (snax_req_i.data_op[11:7]  != '0));
  // a): the only request that goes to the target as a write right away
  assign write_csr = snax_qvalid_i && !req_set_clear && !req_rmw;

  // Read-then-write sequencer for b) and d)
  typedef enum logic [1:0] {
    RmwIdle,    // a) and c) pass through; a b) or d) is accepted here as its read
    RmwWaitRsp, // the b) or d) read is outstanding; wait for its response
    RmwWrite    // issue the write from the latched values until the target takes it
  } rmw_state_e;
  rmw_state_e rmw_state_d, rmw_state_q;

  logic        rmw_idle, rmw_write, rmw_accept, rmw_capture;
  csr_req_t    rmw_req_q;           // addr and mask (data) of the accepted b) or d)
  logic        rmw_sel_q;           // top/acc select of the accepted b) or d)
  logic        rmw_set_clear_q, rmw_clear_q;
  logic [31:0] rmw_old_q;           // the read's data
  logic [31:0] rmw_wdata;

  assign rmw_idle  = (rmw_state_q == RmwIdle);
  assign rmw_write = (rmw_state_q == RmwWrite);

  // We need a demux to split the requests
  logic snax_csr_req_sel;
  logic snax_csr_req_ready;
  logic snax_csr_req_can_accept;
  logic rsp_fifo_full, rsp_fifo_empty;
  logic rsp_fire;
  csr_rsp_t snax_csr_rsp;
  assign snax_csr_req_sel = (snax_req_i.data_argb[11:0]==CSR_SNAX_READ_TASK_READY_QUEUE) ||
                            (snax_req_i.data_argb[11:0]==CSR_SNAX_WRITE_TASK_DONE_QUEUE);
  // Writes do not consume a response ID entry. Reads must only reach a CSR
  // target when there is room to remember their accelerator request ID.
  // A b) or d) waits until no older read is outstanding.
  assign snax_csr_req_can_accept = req_rmw ? rsp_fifo_empty : (write_csr || !rsp_fifo_full);
  stream_demux #(
    .N_OUP(2)
  ) i_snax_csr_req_demux (
    .inp_valid_i ( rmw_write || (rmw_idle && snax_qvalid_i && snax_csr_req_can_accept) ),
    .inp_ready_o ( snax_csr_req_ready   ),
    .oup_sel_i   ( rmw_write ? rmw_sel_q : snax_csr_req_sel ),
    .oup_valid_o ( {snax_csr_req_top_valid_o, snax_csr_req_acc_valid_o}  ),
    .oup_ready_i ( {snax_csr_req_top_ready_i, snax_csr_req_acc_ready_i}  )
  );
  assign snax_qready_o = rmw_idle && snax_csr_req_ready && snax_csr_req_can_accept;

  csr_req_t snax_csr_req_in;
  assign snax_csr_req_in.data  = snax_req_i.data_arga[31:0];
  assign snax_csr_req_in.addr  = snax_req_i.data_argb - CsrAddrOffset;
  assign snax_csr_req_in.write = write_csr;

  always_comb begin
    if (!rmw_set_clear_q) rmw_wdata = rmw_req_q.data;              // b)
    else if (rmw_clear_q) rmw_wdata = rmw_old_q & ~rmw_req_q.data; // d) clear
    else                  rmw_wdata = rmw_old_q |  rmw_req_q.data; // d) set
  end

  always_comb begin
    snax_csr_req_o = snax_csr_req_in;
    if (rmw_write) begin
      snax_csr_req_o       = rmw_req_q;
      snax_csr_req_o.data  = rmw_wdata;
      snax_csr_req_o.write = 1'b1;
    end
  end

  // The b) or d) read is accepted in RmwIdle. Its response is the only one
  // outstanding, so the next rsp_fire (possibly in the same cycle) is its own.
  assign rmw_accept  = snax_qvalid_i && snax_qready_o && req_rmw;
  assign rmw_capture = rsp_fire && (rmw_accept || (rmw_state_q == RmwWaitRsp));

  always_comb begin
    rmw_state_d = rmw_state_q;
    unique case (rmw_state_q)
      RmwIdle:    if (rmw_accept)         rmw_state_d = rsp_fire ? RmwWrite : RmwWaitRsp;
      RmwWaitRsp: if (rsp_fire)           rmw_state_d = RmwWrite;
      RmwWrite:   if (snax_csr_req_ready) rmw_state_d = RmwIdle;
      default:                            rmw_state_d = RmwIdle;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rmw_state_q     <= RmwIdle;
      rmw_req_q       <= '0;
      rmw_sel_q       <= 1'b0;
      rmw_set_clear_q <= 1'b0;
      rmw_clear_q     <= 1'b0;
      rmw_old_q       <= '0;
    end else begin
      rmw_state_q <= rmw_state_d;
      if (rmw_accept) begin
        rmw_req_q       <= snax_csr_req_in;
        rmw_sel_q       <= snax_csr_req_sel;
        rmw_set_clear_q <= req_set_clear;
        rmw_clear_q     <= req_clear;
      end
      if (rmw_capture) rmw_old_q <= snax_csr_rsp.data;
    end
  end


  //-------------------------------
  // Response handler
  //-------------------------------

  // ID needs to be handled with a fifo buffer
  // we know that the responses will always
  // be in order, so we can just use a simple
  // fifo buffer to align the request id

  acc_req_t rsp_fifo_out;
  logic rsp_fifo_push, rsp_fifo_pop;
  logic read_req_fire, rsp_bypass;

  // Combinational logic

  assign read_req_fire = snax_qvalid_i && snax_qready_o && !write_csr;
  assign rsp_fire      = snax_pvalid_o && snax_pready_i;
  assign rsp_bypass    = rsp_fifo_empty && read_req_fire;

  // An immediate response can bypass an empty FIFO. If that response is
  // backpressured, retain the request ID until the response is accepted.
  assign rsp_fifo_push = read_req_fire && !(rsp_bypass && rsp_fire);
  assign rsp_fifo_pop  = rsp_fire && !rsp_fifo_empty;

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

  // We need a mux to select between the two response sources (ACC or Top-level)
  // according to the oldest outstanding request. For an immediate response
  // with an empty FIFO, use the currently accepted read request.
  acc_req_t rsp_req;
  assign rsp_req = rsp_fifo_empty ? snax_req_i : rsp_fifo_out;
  logic snax_csr_rsp_sel;
  assign snax_csr_rsp_sel = (rsp_req.data_argb[11:0]==CSR_SNAX_READ_TASK_READY_QUEUE) ||
                            (rsp_req.data_argb[11:0]==CSR_SNAX_WRITE_TASK_DONE_QUEUE);
  logic snax_csr_rsp_valid, snax_csr_rsp_ready;
  logic rsp_context_valid;
  assign rsp_context_valid = !rsp_fifo_empty || read_req_fire;
  stream_mux #(
    .DATA_T   ( csr_rsp_t ),
    .N_INP    ( 2         )
  ) i_snax_csr_rsp_mux (
    .inp_data_i ( {snax_csr_rsp_top_i, snax_csr_rsp_acc_i}                ),
    .inp_valid_i( {snax_csr_rsp_top_valid_i, snax_csr_rsp_acc_valid_i}    ),
    .inp_ready_o( {snax_csr_rsp_top_ready_o, snax_csr_rsp_acc_ready_o}    ),
    .inp_sel_i  ( snax_csr_rsp_sel        ),
    .oup_data_o ( snax_csr_rsp            ),
    .oup_valid_o( snax_csr_rsp_valid      ),
    .oup_ready_i( snax_csr_rsp_ready      )
  );

  assign snax_pvalid_o      = snax_csr_rsp_valid && rsp_context_valid;
  assign snax_csr_rsp_ready = snax_pready_i && rsp_context_valid;

  assign snax_resp_o.data     = snax_csr_rsp.data;
  assign snax_resp_o.id       = rsp_req.id;
  // Leave this as always no error for now
  assign snax_resp_o.error    = 1'b0;

endmodule
