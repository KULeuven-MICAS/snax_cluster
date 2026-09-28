// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// The narrow TCDM interconnect's level arbitration: `snitch_tcdm_interconnect` fed by
// `snitch_tcdm_arb_level`, 8 inputs on 4 banks, under random traffic and five contention
// scenarios. Every cycle it checks that
//   - a bank grants nothing while a higher-level request wants it, and at most one request;
//   - every request is answered once, and every read returns its input's last write to that
//     word (each input owns its own rows, so the expected value is known);
// and per scenario that
//   - the starvation guard bounds a low-urgency victim's wait behind an urgent hog;
//   - without the guard the same victim starves (the negative control);
//   - a software level replaces the guard: a hog pinned above it starves the victim, and a
//     victim pinned at 0 is never promoted.
//
// Build and run from the repository root (Verilator 5):
//   CC=$(bender path common_cells)
//   $ verilator --binary --timing -Wno-fatal --top-module snitch_tcdm_arb_tb \
//     -I$CC/include -Ihw/mem_interface/include -Ihw/tcdm_interface/include \
//     $CC/src/cf_math_pkg.sv hw/reqrsp_interface/src/reqrsp_pkg.sv hw/snitch/src/snitch_pkg.sv \
//     $CC/src/lzc.sv $CC/src/rr_arb_tree.sv $CC/src/shift_reg.sv \
//     hw/snitch_cluster/src/snitch_tcdm_arb_level.sv \
//     hw/snitch_cluster/src/snitch_tcdm_interconnect.sv \
//     hw/snitch_cluster/test/snitch_tcdm_arb_tb.sv
//   obj_dir/Vsnitch_tcdm_arb_tb

`include "mem_interface/typedef.svh"
`include "tcdm_interface/typedef.svh"

module snitch_tcdm_arb_tb;

  localparam int unsigned NumInp       = 8;
  localparam int unsigned NumOut       = 4;
  localparam int unsigned DataWidth    = 64;
  localparam int unsigned MemAddrWidth = 8;
  localparam int unsigned LevelWidth   = 3;
  localparam int unsigned ByteOffset   = 3;
  localparam int unsigned SelWidth     = 2;
  localparam int unsigned RowsPerInp   = (1 << MemAddrWidth) / NumInp;
  localparam int unsigned Threshold    = 8;

  typedef logic [31:0] addr_t;
  typedef logic [DataWidth-1:0] data_t;
  typedef logic [DataWidth/8-1:0] strb_t;
  typedef logic [MemAddrWidth-1:0] mem_addr_t;
  typedef struct packed {logic [1:0] tcdm_priority;} user_t;
  `TCDM_TYPEDEF_ALL(tcdm, addr_t, data_t, strb_t, user_t)
  `MEM_TYPEDEF_ALL(mem, mem_addr_t, data_t, strb_t, user_t)

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;

  // ---- the device ------------------------------------------------------------------------
  tcdm_req_t [NumInp-1:0] req;
  tcdm_rsp_t [NumInp-1:0] rsp;
  mem_req_t  [NumOut-1:0] mreq;
  mem_rsp_t  [NumOut-1:0] mrsp;

  logic urgency_en, guard_en;
  logic [7:0] threshold;
  logic [NumInp-1:0] ov_en, valid, ready;
  logic [NumInp-1:0][LevelWidth-1:0] ov_lvl, level;
  logic [NumInp-1:0][1:0] urgency;

  for (genvar i = 0; i < NumInp; i++) begin : gen_handshake
    assign valid[i] = req[i].q_valid;
    assign ready[i] = rsp[i].q_ready;
  end

  snitch_tcdm_arb_level #(
    .NumInp(NumInp), .UrgencyWidth(2), .LevelWidth(LevelWidth), .AgeWidth(8)
  ) i_level (
    .clk_i(clk), .rst_ni(rst_n), .valid_i(valid), .ready_i(ready), .urgency_i(urgency),
    .urgency_en_i(urgency_en), .guard_en_i(guard_en), .guard_threshold_i(threshold),
    .override_en_i(ov_en), .override_level_i(ov_lvl), .level_o(level)
  );

  snitch_tcdm_interconnect #(
    .NumInp(NumInp), .NumOut(NumOut), .tcdm_req_t(tcdm_req_t), .tcdm_rsp_t(tcdm_rsp_t),
    .mem_req_t(mem_req_t), .mem_rsp_t(mem_rsp_t), .MemAddrWidth(MemAddrWidth),
    .DataWidth(DataWidth), .user_t(user_t), .MemoryResponseLatency(1),
    .LevelWidth(LevelWidth), .Topology(snitch_pkg::LogarithmicInterconnect)
  ) i_dut (
    .clk_i(clk), .rst_ni(rst_n), .req_i(req), .level_i(level), .rsp_o(rsp),
    .mem_req_o(mreq), .mem_rsp_i(mrsp)
  );

  // ---- the banks: one-cycle reads; a bank refuses at random, as when the wide port has it --
  data_t mem [NumOut][1 << MemAddrWidth];
  logic [NumOut-1:0] bank_ready;
  data_t [NumOut-1:0] bank_rdata;
  int unsigned bank_ready_pct;

  for (genvar b = 0; b < NumOut; b++) begin : gen_bank
    assign mrsp[b].q_ready = bank_ready[b];
    assign mrsp[b].p.data  = bank_rdata[b];
    always @(posedge clk) begin
      bank_ready[b] <= ($urandom_range(99) < bank_ready_pct);
      if (mreq[b].q_valid && bank_ready[b]) begin
        if (mreq[b].q.write) mem[b][mreq[b].q.addr] <= mreq[b].q.data;
        bank_rdata[b] <= mem[b][mreq[b].q.addr];
      end
    end
  end

  // ---- the requesters ----------------------------------------------------------------------
  typedef enum int {Drain, Random, Hog, HogNoGuard, HogPinnedHigh, VictimPinnedLow, RandomOverride} scen_e;
  scen_e scen;

  // Each input's shadow of its own rows, the read it waits on, and its waits.
  data_t shadow [NumInp][NumOut][RowsPerInp];
  logic  [NumInp-1:0] rsp_due, rsp_is_read;
  data_t [NumInp-1:0] rsp_expect;
  int unsigned wait_now [NumInp], wait_max [NumInp], served [NumInp];
  int unsigned errors = 0;

  function automatic logic [SelWidth-1:0] bank_of(addr_t a);
    return a[ByteOffset+:SelWidth];
  endfunction
  function automatic logic [MemAddrWidth-1:0] row_of(addr_t a);
    return a[ByteOffset+SelWidth+:MemAddrWidth];
  endfunction

  // A new request for input i: to bank `b`, one of its own rows, a write half the time.
  task automatic issue(int i, int b);
    int unsigned r = $urandom_range(RowsPerInp - 1);
    req[i].q_valid <= 1'b1;
    req[i].q.addr  <= addr_t'((((i * RowsPerInp) + r) << (ByteOffset + SelWidth)) |
                              (b << ByteOffset));
    req[i].q.write <= $urandom_range(1);
    req[i].q.data  <= {$urandom, $urandom};
    req[i].q.strb  <= '1;
    req[i].q.amo   <= reqrsp_pkg::AMONone;
  endtask

  // Whether input i issues this cycle, and to which bank.
  function automatic int want(int i);
    case (scen)
      Drain:                  return -1;
      Random, RandomOverride: return ($urandom_range(3) != 0) ? int'($urandom_range(NumOut - 1)) : -1;
      default:                return (i < 2) ? 0 : -1;  // inputs 0 (the hog) and 1 (the victim)
    endcase
  endfunction

  always @(posedge clk) begin
    if (!rst_n) begin
      req <= '0;
      rsp_due <= '0;
    end else begin
      for (int i = 0; i < NumInp; i++) begin
        int w;
        // the response to last cycle's grant
        if (rsp_due[i]) begin
          if (!rsp[i].p_valid) begin
            $display("ERROR input %0d: granted but no response", i);
            errors++;
          end else if (rsp_is_read[i] && rsp[i].p.data !== rsp_expect[i]) begin
            $display("ERROR input %0d: read %h, expected %h", i, rsp[i].p.data, rsp_expect[i]);
            errors++;
          end
        end else if (rsp[i].p_valid) begin
          $display("ERROR input %0d: a response without a grant", i);
          errors++;
        end
        rsp_due[i] <= valid[i] && ready[i];
        // the grant: record it, update the shadow, issue the next
        if (valid[i] && ready[i]) begin
          int b, r;
          b = bank_of(req[i].q.addr);
          r = row_of(req[i].q.addr) - i * RowsPerInp;
          rsp_is_read[i] <= !req[i].q.write;
          rsp_expect[i]  <= shadow[i][b][r];
          if (req[i].q.write) shadow[i][b][r] = req[i].q.data;
          served[i]++;
          if (wait_now[i] > wait_max[i]) wait_max[i] = wait_now[i];
          wait_now[i] = 0;
          req[i].q_valid <= 1'b0;
          w = want(i);
          if (w >= 0) issue(i, w);  // back to back
        end else if (valid[i]) begin
          wait_now[i]++;
        end else begin
          w = want(i);
          if (w >= 0) issue(i, w);
        end
      end
    end
  end

  // ---- the arbitration checks --------------------------------------------------------------
  always @(posedge clk) begin
    if (rst_n) begin
      for (int b = 0; b < NumOut; b++) begin
        int grants;
        grants = 0;
        for (int i = 0; i < NumInp; i++) begin
          if (valid[i] && ready[i] && bank_of(req[i].q.addr) == b) begin
            grants++;
            for (int j = 0; j < NumInp; j++) begin
              if (j != i && valid[j] && bank_of(req[j].q.addr) == b && level[j] > level[i]) begin
                $display("ERROR bank %0d: granted input %0d at level %0d while input %0d waits at %0d",
                         b, i, level[i], j, level[j]);
                errors++;
              end
            end
          end
        end
        if (grants > 1) begin
          $display("ERROR bank %0d: %0d grants in one cycle", b, grants);
          errors++;
        end
      end
    end
  end

  // ---- the scenarios -----------------------------------------------------------------------
  task automatic run(scen_e s, int unsigned cycles);
    scen = s;
    for (int i = 0; i < NumInp; i++) begin
      wait_max[i] = 0;
      wait_now[i] = 0;
      served[i]   = 0;
    end
    repeat (cycles) begin
      @(negedge clk);
      if (s == Random || s == RandomOverride) begin
        for (int i = 0; i < NumInp; i++) urgency[i] = 2'($urandom_range(3));
      end
      if (s == RandomOverride && $urandom_range(255) == 0) begin
        for (int i = 0; i < NumInp; i++) begin
          ov_en[i]  = ($urandom_range(3) == 0);
          ov_lvl[i] = LevelWidth'($urandom_range(7));
        end
        guard_en  = $urandom_range(1);
        threshold = 8'($urandom_range(1, 20));
      end
    end
    // serve every request still waiting before the next scenario
    scen = Drain;
    ov_en = '0;
    guard_en = 1'b1;
    threshold = Threshold;
    bank_ready_pct = 100;
    repeat (200) @(negedge clk);
    if (valid != '0) begin
      $display("ERROR requests still waiting %b after the drain", valid);
      errors++;
    end
  endtask

  task automatic contend(scen_e s, int unsigned hog_urgency, logic guard, logic [NumInp-1:0] pin,
                         logic [NumInp-1:0][LevelWidth-1:0] pin_level);
    urgency    = '0;
    urgency[0] = 2'(hog_urgency);
    guard_en   = guard;
    ov_en      = pin;
    ov_lvl     = pin_level;
    bank_ready_pct = 100;
    run(s, 2000);
  endtask

  initial begin
    for (int b = 0; b < NumOut; b++) for (int w = 0; w < (1 << MemAddrWidth); w++) mem[b][w] = '0;
    for (int i = 0; i < NumInp; i++)
      for (int b = 0; b < NumOut; b++) for (int r = 0; r < RowsPerInp; r++) shadow[i][b][r] = '0;
    urgency_en = 1'b1;
    guard_en   = 1'b1;
    threshold  = Threshold;
    ov_en      = '0;
    ov_lvl     = '0;
    urgency    = '0;
    bank_ready_pct = 75;
    scen = Random;
    repeat (4) @(negedge clk);
    rst_n = 1'b1;

    run(Random, 20000);
    $display("random: served %p, longest wait %p", served, wait_max);

    // An urgent hog (input 0) and a victim with slack (input 1) on one bank.
    contend(Hog, 3, 1'b1, '0, '0);
    $display("guard on: victim served %0d times, longest wait %0d (threshold %0d)",
             served[1], wait_max[1], Threshold);
    if (served[1] == 0 || wait_max[1] > Threshold + 2) begin
      $display("ERROR the guard did not bound the victim's wait");
      errors++;
    end

    contend(HogNoGuard, 3, 1'b0, '0, '0);
    $display("guard off: victim served %0d times (the negative control: it should starve)",
             served[1]);
    if (served[1] > 1) begin
      $display("ERROR the victim was served without the guard: the control is not a control");
      errors++;
    end

    // software pins the hog above the guard: the victim starves by software's choice
    contend(HogPinnedHigh, 0, 1'b1, 8'b0000_0001, {{7{3'd0}}, 3'd7});
    $display("hog pinned at 7: victim served %0d times", served[1]);
    if (served[1] > 1) begin
      $display("ERROR the guard overrode software's level");
      errors++;
    end

    // software pins the victim at 0: the guard never lifts it over a hog at urgency 1
    contend(VictimPinnedLow, 1, 1'b1, 8'b0000_0010, '0);
    $display("victim pinned at 0: served %0d times", served[1]);
    if (served[1] > 1) begin
      $display("ERROR the guard promoted a request software pinned at 0");
      errors++;
    end

    bank_ready_pct = 75;
    run(RandomOverride, 20000);
    $display("random with overrides: served %p", served);

    repeat (8) @(negedge clk);
    if (errors == 0) $display("PASS");
    else $display("FAIL: %0d errors", errors);
    $finish;
  end

endmodule
