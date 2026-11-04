// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Ryan Antonio <ryan.antonio@esat.kuleuven.be>

//-------------------------------
// CSR range DEMUX and response MUX
//
// A table version of snax_csr_mux_demux, with the same core-side and target-side ports.
// - Request: range r takes the addresses RangeBase[r] <= addr < RangeBase[r] + RangeSize[r]
//   (the lowest r if ranges overlap). Its target RangeTarget[r] gets the request with
//   addr - RangeBase[r] + RangeTargetOffset[r]; only that target sees valid, and the
//   core's ready is that target's ready.
// - Responses: only reads have one. A FIFO, MaxReads deep, holds each outstanding read's
//   target; responses go back strictly in request order, each taken from the target at
//   the FIFO's head. The core's ready is low while the FIFO is full.
// - An address in no range: a write is accepted and dropped, a read is accepted and
//   answered with 0 in its turn (an "own" FIFO entry). Nothing waits on a target.
//-------------------------------
module snax_csr_range_demux #(
  parameter int unsigned                              NumTargets        = 2,
  parameter int unsigned                              NumRanges         = 2,
  parameter int unsigned                              RegDataWidth      = 32,
  parameter int unsigned                              RegAddrWidth      = 32,
  // Per range r, bit slice [r]
  parameter logic [NumRanges-1:0][RegAddrWidth-1:0]   RangeBase         = '0,
  parameter logic [NumRanges-1:0][RegAddrWidth-1:0]   RangeSize         = '0,
  parameter logic [NumRanges-1:0][31:0]               RangeTarget       = '0,
  parameter logic [NumRanges-1:0][RegAddrWidth-1:0]   RangeTargetOffset = '0,
  // The reads that can be outstanding
  parameter int unsigned                              MaxReads          = 4
)(
  //-------------------------------
  // Clocks and reset
  //-------------------------------
  input  logic clk_i,
  input  logic rst_ni,
  //-------------------------------
  // Input Core
  //-------------------------------
  input  logic [RegAddrWidth-1:0] csr_req_addr_i,
  input  logic [RegDataWidth-1:0] csr_req_data_i,
  input  logic                    csr_req_wen_i,
  input  logic                    csr_req_valid_i,
  output logic                    csr_req_ready_o,
  output logic [RegDataWidth-1:0] csr_rsp_data_o,
  output logic                    csr_rsp_valid_o,
  input  logic                    csr_rsp_ready_i,
  //-------------------------------
  // Output Port
  //-------------------------------
  output logic [NumTargets-1:0][RegAddrWidth-1:0] acc_csr_req_addr_o,
  output logic [NumTargets-1:0][RegDataWidth-1:0] acc_csr_req_data_o,
  output logic [NumTargets-1:0][0:0]              acc_csr_req_wen_o,
  output logic [NumTargets-1:0][0:0]              acc_csr_req_valid_o,
  input  logic [NumTargets-1:0][0:0]              acc_csr_req_ready_i,
  input  logic [NumTargets-1:0][RegDataWidth-1:0] acc_csr_rsp_data_i,
  input  logic [NumTargets-1:0][0:0]              acc_csr_rsp_valid_i,
  output logic [NumTargets-1:0][0:0]              acc_csr_rsp_ready_o
);

  localparam int unsigned TargetWidth = (NumTargets > 1) ? $clog2(NumTargets) : 1;
  localparam int unsigned PtrWidth    = (MaxReads > 1) ? $clog2(MaxReads) : 1;
  localparam int unsigned CountWidth  = $clog2(MaxReads + 1);

  // pragma translate_off
  initial begin
    if (MaxReads == 0) $fatal(1, "snax_csr_range_demux: MaxReads must be at least 1");
    for (int unsigned r = 0; r < NumRanges; r++) begin
      if (RangeTarget[r] >= NumTargets)
        $fatal(1, "snax_csr_range_demux: range %0d targets %0d of %0d targets",
               r, RangeTarget[r], NumTargets);
    end
  end
  // pragma translate_on

  //-------------------------------
  // Range lookup
  //-------------------------------
  logic                    req_hit;
  logic [TargetWidth-1:0]  req_target;
  logic [RegAddrWidth-1:0] req_addr_rebased;

  always_comb begin
    req_hit          = 1'b0;
    req_target       = '0;
    req_addr_rebased = '0;
    for (int unsigned r = 0; r < NumRanges; r++) begin
      if (!req_hit && (csr_req_addr_i >= RangeBase[r]) &&
          ((csr_req_addr_i - RangeBase[r]) < RangeSize[r])) begin
        req_hit          = 1'b1;
        req_target       = RangeTarget[r][TargetWidth-1:0];
        req_addr_rebased = csr_req_addr_i - RangeBase[r] + RangeTargetOffset[r];
      end
    end
  end

  //-------------------------------
  // Outstanding reads: {own, target} per read, oldest at the head
  //-------------------------------
  logic [MaxReads-1:0]                 fifo_own_q;
  logic [MaxReads-1:0][TargetWidth-1:0] fifo_target_q;
  logic [PtrWidth-1:0]                 fifo_wr_ptr_q, fifo_rd_ptr_q;
  logic [CountWidth-1:0]               fifo_count_q;
  logic                                fifo_full, fifo_empty;
  logic                                fifo_push, fifo_pop;
  logic                                head_own;
  logic [TargetWidth-1:0]              head_target;

  assign fifo_full   = (fifo_count_q == CountWidth'(MaxReads));
  assign fifo_empty  = (fifo_count_q == '0);
  assign head_own    = fifo_own_q[fifo_rd_ptr_q];
  assign head_target = fifo_target_q[fifo_rd_ptr_q];

  function automatic logic [PtrWidth-1:0] ptr_next(logic [PtrWidth-1:0] ptr);
    return (ptr == PtrWidth'(MaxReads - 1)) ? '0 : ptr + 1'b1;
  endfunction

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      fifo_own_q    <= '0;
      fifo_target_q <= '0;
      fifo_wr_ptr_q <= '0;
      fifo_rd_ptr_q <= '0;
      fifo_count_q  <= '0;
    end else begin
      if (fifo_push) begin
        fifo_own_q[fifo_wr_ptr_q]    <= !req_hit;
        fifo_target_q[fifo_wr_ptr_q] <= req_target;
        fifo_wr_ptr_q                <= ptr_next(fifo_wr_ptr_q);
      end
      if (fifo_pop) fifo_rd_ptr_q <= ptr_next(fifo_rd_ptr_q);
      if (fifo_push && !fifo_pop) fifo_count_q <= fifo_count_q + 1'b1;
      if (!fifo_push && fifo_pop) fifo_count_q <= fifo_count_q - 1'b1;
    end
  end

  //-------------------------------
  // Request DEMUX
  // Unused target ports are set to 0
  //-------------------------------
  always_comb begin
    for (int unsigned t = 0; t < NumTargets; t++) begin
      if (req_hit && (req_target == TargetWidth'(t))) begin
        acc_csr_req_addr_o [t] = req_addr_rebased;
        acc_csr_req_data_o [t] = csr_req_data_i;
        acc_csr_req_wen_o  [t] = csr_req_wen_i;
        acc_csr_req_valid_o[t] = csr_req_valid_i && !fifo_full;
      end else begin
        acc_csr_req_addr_o [t] = '0;
        acc_csr_req_data_o [t] = '0;
        acc_csr_req_wen_o  [t] = '0;
        acc_csr_req_valid_o[t] = '0;
      end
    end
    // A gap address is taken right away (dropped write or own read)
    csr_req_ready_o = !fifo_full && (req_hit ? acc_csr_req_ready_i[req_target] : 1'b1);
  end

  assign fifo_push = csr_req_valid_i && csr_req_ready_o && !csr_req_wen_i;

  //-------------------------------
  // Response MUX, in request order
  //-------------------------------
  always_comb begin
    acc_csr_rsp_ready_o = '0;
    csr_rsp_valid_o     = 1'b0;
    csr_rsp_data_o      = '0;
    if (!fifo_empty) begin
      if (head_own) begin
        csr_rsp_valid_o = 1'b1;
      end else begin
        csr_rsp_valid_o                  = acc_csr_rsp_valid_i[head_target];
        csr_rsp_data_o                   = acc_csr_rsp_data_i[head_target];
        acc_csr_rsp_ready_o[head_target] = csr_rsp_ready_i;
      end
    end
  end

  assign fifo_pop = csr_rsp_valid_o && csr_rsp_ready_i;

endmodule
