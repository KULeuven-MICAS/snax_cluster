// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Ryan Antonio <ryan.antonio@esat.kuleuven.be>

//-------------------------------
// Testbench of snax_csr_range_demux with snax-forge dot 1's CSR table:
//   0-17 -> target 0 (0-17), 18-21 -> target 2 (0-3), 22-34 -> target 1 (0-12),
//   35-39 -> target 2 (4-8), 40-63 -> no range.
// Seeded random reads and writes over 0-63; three target models with random ready
// stalls and a 1-4 cycle read latency, each answering a read with
// {1, target id, rebased address} and checking each write it gets; the core's response
// ready stalls at random. Checks that writes land on the right target and address (or
// are dropped in the gap), reads come back in order with the right data (0 in the gap),
// only the addressed target sees valid, nothing hangs, and the FIFO-full stall happens.
// Run: hw/snax_util/test/run_snax_csr_range_demux_tb.sh [seed ...]
//-------------------------------
module snax_csr_range_demux_tb #(
  parameter int unsigned NumRequests = 10000
);

  localparam int unsigned NumTargets   = 3;
  localparam int unsigned NumRanges    = 4;
  localparam int unsigned RegDataWidth = 32;
  localparam int unsigned RegAddrWidth = 32;
  localparam int unsigned MaxReads     = 4;
  localparam int unsigned HangCycles   = 2000;

  //-------------------------------
  // Clock, reset, random numbers
  //-------------------------------
  logic clk = 1'b0;
  logic rst_n = 1'b0;
  always #5 clk = ~clk;

  int unsigned seed;
  int unsigned cycle = 0;
  always @(posedge clk) cycle <= cycle + 1;

  // xorshift32: one state per process, seeded from the seed
  function automatic int unsigned rnd(ref int unsigned state);
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  endfunction

  //-------------------------------
  // The reference table (written out separately from the DUT's parameters)
  //-------------------------------
  function automatic void reference(input int unsigned addr, output logic hit,
                                    output int unsigned target, output int unsigned rebased);
    hit = 1'b1;
    if (addr <= 17)      begin target = 0; rebased = addr;          end
    else if (addr <= 21) begin target = 2; rebased = addr - 18;     end
    else if (addr <= 34) begin target = 1; rebased = addr - 22;     end
    else if (addr <= 39) begin target = 2; rebased = addr - 35 + 4; end
    else                 begin hit = 1'b0; target = 0; rebased = 0; end
  endfunction

  function automatic logic [31:0] read_data(int unsigned target, int unsigned rebased);
    return 32'h8000_0000 | (target << 16) | (rebased & 32'hffff);
  endfunction

  //-------------------------------
  // DUT
  //-------------------------------
  logic [RegAddrWidth-1:0] req_addr;
  logic [RegDataWidth-1:0] req_data;
  logic                    req_wen;
  logic                    req_valid;
  logic                    req_ready;
  logic [RegDataWidth-1:0] rsp_data;
  logic                    rsp_valid;
  logic                    rsp_ready;

  logic [NumTargets-1:0][RegAddrWidth-1:0] tgt_req_addr;
  logic [NumTargets-1:0][RegDataWidth-1:0] tgt_req_data;
  logic [NumTargets-1:0][0:0]              tgt_req_wen;
  logic [NumTargets-1:0][0:0]              tgt_req_valid;
  logic [NumTargets-1:0][0:0]              tgt_req_ready;
  logic [NumTargets-1:0][RegDataWidth-1:0] tgt_rsp_data;
  logic [NumTargets-1:0][0:0]              tgt_rsp_valid;
  logic [NumTargets-1:0][0:0]              tgt_rsp_ready;

  snax_csr_range_demux #(
    .NumTargets        ( NumTargets   ),
    .NumRanges         ( NumRanges    ),
    .RegDataWidth      ( RegDataWidth ),
    .RegAddrWidth      ( RegAddrWidth ),
    .RangeBase         ( {32'd35, 32'd22, 32'd18, 32'd0}  ),
    .RangeSize         ( {32'd5,  32'd13, 32'd4,  32'd18} ),
    .RangeTarget       ( {32'd2,  32'd1,  32'd2,  32'd0}  ),
    .RangeTargetOffset ( {32'd4,  32'd0,  32'd0,  32'd0}  ),
    .MaxReads          ( MaxReads     )
  ) i_dut (
    .clk_i               ( clk           ),
    .rst_ni              ( rst_n         ),
    .csr_req_addr_i      ( req_addr      ),
    .csr_req_data_i      ( req_data      ),
    .csr_req_wen_i       ( req_wen       ),
    .csr_req_valid_i     ( req_valid     ),
    .csr_req_ready_o     ( req_ready     ),
    .csr_rsp_data_o      ( rsp_data      ),
    .csr_rsp_valid_o     ( rsp_valid     ),
    .csr_rsp_ready_i     ( rsp_ready     ),
    .acc_csr_req_addr_o  ( tgt_req_addr  ),
    .acc_csr_req_data_o  ( tgt_req_data  ),
    .acc_csr_req_wen_o   ( tgt_req_wen   ),
    .acc_csr_req_valid_o ( tgt_req_valid ),
    .acc_csr_req_ready_i ( tgt_req_ready ),
    .acc_csr_rsp_data_i  ( tgt_rsp_data  ),
    .acc_csr_rsp_valid_i ( tgt_rsp_valid ),
    .acc_csr_rsp_ready_o ( tgt_rsp_ready )
  );

  //-------------------------------
  // Expectations and results
  //-------------------------------
  logic [31:0] exp_rsp[$];                       // each read's data, in request order
  int unsigned presented = 0, n_reads = 0, n_writes = 0, n_gap_reads = 0, n_gap_writes = 0;
  int unsigned n_rsp = 0, n_tgt_writes = 0, full_stalls = 0, errors = 0, idle = 0;

  // Requests still to send, or reads or writes not yet through
  function automatic logic pending();
    return presented < NumRequests || req_valid || exp_rsp.size() != 0 ||
           gen_target[0].exp_wr.size() != 0 || gen_target[1].exp_wr.size() != 0 ||
           gen_target[2].exp_wr.size() != 0;
  endfunction

  task automatic fail(string what);
    errors++;
    if (errors <= 10) $display("[tb] cycle %0d: %s", cycle, what);
  endtask

  //-------------------------------
  // Core-side request driver
  //-------------------------------
  int unsigned drv_rng;
  always @(posedge clk) begin
    if (!rst_n) begin
      req_valid <= 1'b0;
      req_addr  <= '0;
      req_data  <= '0;
      req_wen   <= 1'b0;
    end else if (!req_valid || req_ready) begin
      // Mostly back to back, so reads pile up behind response stalls
      if (presented < NumRequests && (rnd(drv_rng) % 8) != 0) begin
        automatic int unsigned addr = rnd(drv_rng) % 64;
        automatic logic        wen  = (rnd(drv_rng) % 2) != 0;
        automatic logic [31:0] data = rnd(drv_rng);
        automatic logic        hit;
        automatic int unsigned target, rebased;
        reference(addr, hit, target, rebased);
        req_valid <= 1'b1;
        req_addr  <= addr;
        req_wen   <= wen;
        req_data  <= data;
        presented++;
        if (wen) begin
          n_writes++;
          if (hit) begin
            // Each target model keeps its own queue of expected writes {addr, data}
            case (target)
              0:       gen_target[0].exp_wr.push_back({32'(rebased), data});
              1:       gen_target[1].exp_wr.push_back({32'(rebased), data});
              default: gen_target[2].exp_wr.push_back({32'(rebased), data});
            endcase
          end
          else n_gap_writes++;
        end else begin
          n_reads++;
          if (hit) exp_rsp.push_back(read_data(target, rebased));
          else begin
            exp_rsp.push_back('0);
            n_gap_reads++;
          end
        end
      end else begin
        req_valid <= 1'b0;
      end
    end
  end

  //-------------------------------
  // Core-side response checker, with random ready stalls (some long)
  //-------------------------------
  int unsigned mon_rng, stall_left = 0;
  always @(posedge clk) begin
    if (!rst_n) begin
      rsp_ready <= 1'b0;
    end else begin
      if (rsp_valid && rsp_ready) begin
        n_rsp++;
        if (exp_rsp.size() == 0) fail($sformatf("response %h with no read outstanding", rsp_data));
        else begin
          automatic logic [31:0] exp = exp_rsp.pop_front();
          if (rsp_data !== exp)
            fail($sformatf("read %0d: got %h, expected %h", n_rsp, rsp_data, exp));
        end
      end
      if (stall_left > 0) begin
        stall_left--;
        rsp_ready <= 1'b0;
      end else if ((rnd(mon_rng) % 32) == 0) begin
        stall_left = 1 + rnd(mon_rng) % 24;
        rsp_ready <= 1'b0;
      end else begin
        rsp_ready <= (rnd(mon_rng) % 4) != 0;
      end
      // Only the addressed target sees valid; a gap address reaches none
      if ($countones(tgt_req_valid) > 1) fail("more than one target sees valid");
      if (req_valid) begin
        automatic logic        hit;
        automatic int unsigned target, rebased;
        reference(req_addr, hit, target, rebased);
        for (int unsigned t = 0; t < NumTargets; t++) begin
          if (tgt_req_valid[t] && (!hit || t != target))
            fail($sformatf("address %0d reached target %0d", req_addr, t));
        end
        if (!req_ready && i_dut.fifo_full) full_stalls++;
      end else if ($countones(tgt_req_valid) != 0) begin
        fail("a target sees valid with no request");
      end
      // Watchdog: no handshake for HangCycles while work is pending
      if ((req_valid && req_ready) || (rsp_valid && rsp_ready) || !pending()) idle = 0;
      else idle++;
      if (idle > HangCycles) begin
        $display({$sformatf("FAIL seed=%0d: hang, no handshake for %0d cycles at cycle %0d",
                            seed, HangCycles, cycle),
                  $sformatf(" (%0d of %0d requests, req_valid %0d, %0d reads and",
                            presented, NumRequests, req_valid, exp_rsp.size()),
                  $sformatf(" %0d/%0d/%0d writes outstanding, %0d errors)",
                            gen_target[0].exp_wr.size(), gen_target[1].exp_wr.size(),
                            gen_target[2].exp_wr.size(), errors)});
        $finish;
      end
    end
  end

  //-------------------------------
  // Target models
  //-------------------------------
  for (genvar t = 0; t < NumTargets; t++) begin : gen_target
    int unsigned rng;
    logic [63:0] exp_wr[$];                      // the writes it should get {addr, data}, in order
    logic [31:0] rsp_q[$];
    int unsigned due_q[$];
    always @(posedge clk) begin
      if (!rst_n) begin
        rng = seed * 32'hc2b2ae35 + t + 3;
        tgt_req_ready[t] <= 1'b0;
        tgt_rsp_valid[t] <= 1'b0;
        tgt_rsp_data[t]  <= '0;
      end else begin
        // Requests: check writes, schedule reads 1-4 cycles out (in order)
        if (tgt_req_valid[t] && tgt_req_ready[t]) begin
          if (tgt_req_wen[t]) begin
            n_tgt_writes++;
            if (exp_wr.size() == 0)
              fail($sformatf("target %0d: unexpected write %0d <- %h", t, tgt_req_addr[t],
                             tgt_req_data[t]));
            else begin
              automatic logic [63:0] exp = exp_wr.pop_front();
              if ({tgt_req_addr[t], tgt_req_data[t]} !== exp)
                fail($sformatf("target %0d: write %0d <- %h, expected %0d <- %h", t,
                               tgt_req_addr[t], tgt_req_data[t], exp[63:32], exp[31:0]));
            end
          end else begin
            rsp_q.push_back(read_data(t, tgt_req_addr[t]));
            due_q.push_back(cycle + 1 + rnd(rng) % 4);
          end
        end
        tgt_req_ready[t] <= (rnd(rng) % 4) != 0;
        // Responses: hold the head until it is taken, show the next once due
        if (tgt_rsp_valid[t] && tgt_rsp_ready[t]) begin
          void'(rsp_q.pop_front());
          void'(due_q.pop_front());
        end
        if (rsp_q.size() != 0 && due_q[0] <= cycle + 1) begin
          tgt_rsp_valid[t] <= 1'b1;
          tgt_rsp_data[t]  <= rsp_q[0];
        end else begin
          tgt_rsp_valid[t] <= 1'b0;
          tgt_rsp_data[t]  <= '0;
        end
      end
    end
  end

  //-------------------------------
  // Run
  //-------------------------------
  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    drv_rng = seed * 32'h9e3779b9 + 1;
    mon_rng = seed * 32'h85ebca6b + 2;
    repeat (5) @(posedge clk);
    @(negedge clk) rst_n = 1'b1;
    // Wait for every request, response and write to be through
    while (pending()) @(posedge clk);
    repeat (10) @(posedge clk);
    if (rsp_valid) fail("a response with no read outstanding");
    if (full_stalls == 0) fail("the FIFO-full stall never happened");
    if (n_rsp != n_reads) fail($sformatf("%0d responses for %0d reads", n_rsp, n_reads));
    if (n_tgt_writes + n_gap_writes != n_writes)
      fail($sformatf("%0d writes landed + %0d dropped of %0d",
                     n_tgt_writes, n_gap_writes, n_writes));
    if (errors == 0)
      $display({$sformatf("PASS seed=%0d: %0d requests (%0d reads, %0d in the gap;",
                          seed, presented, n_reads, n_gap_reads),
                $sformatf(" %0d writes, %0d dropped), %0d FIFO-full stall cycles, %0d cycles",
                          n_writes, n_gap_writes, full_stalls, cycle)});
    else
      $display("FAIL seed=%0d: %0d errors", seed, errors);
    $finish;
  end

endmodule
