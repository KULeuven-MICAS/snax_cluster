// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "common_cells/registers.svh"
`include "common_cells/assertions.svh"

/// The arbitration level of every narrow TCDM interconnect input, recomputed every cycle.
/// `snitch_tcdm_interconnect` serves, at each bank, the highest level among the requests that
/// want it. The levels:
///
///   0 .. 3  the requester's urgency (`tcdm_priority`): how close it is to stalling the engine
///           it serves, graded by the requester from its own FIFO (`snax.utils.TcdmUrgency`)
///   4       the starvation guard: a request that has waited `guard_threshold_i` cycles, above
///           every urgency, so every request is served within a bounded time
///   0 .. 7  software's level for the input, where software pins one (`override_en_i`). It
///           replaces both of the above: software can put an input above the guard (5 .. 7),
///           among the hardware levels, or hold it at 0 where the guard never lifts it.
module snitch_tcdm_arb_level #(
  parameter int unsigned NumInp       = 1,
  parameter int unsigned UrgencyWidth = 2,
  parameter int unsigned LevelWidth   = 3,
  parameter int unsigned AgeWidth     = 8
) (
  input  logic                                clk_i,
  input  logic                                rst_ni,
  /// Each input presents a request, and it is granted.
  input  logic [NumInp-1:0]                   valid_i,
  input  logic [NumInp-1:0]                   ready_i,
  /// Each requester's urgency; zero everywhere without `urgency_en_i`.
  input  logic [NumInp-1:0][UrgencyWidth-1:0] urgency_i,
  input  logic                                urgency_en_i,
  /// The starvation guard, and the wait in cycles after which it promotes a request.
  input  logic                                guard_en_i,
  input  logic [AgeWidth-1:0]                 guard_threshold_i,
  /// Software's levels.
  input  logic [NumInp-1:0]                   override_en_i,
  input  logic [NumInp-1:0][LevelWidth-1:0]   override_level_i,
  output logic [NumInp-1:0][LevelWidth-1:0]   level_o
);

  localparam logic [LevelWidth-1:0] GuardLevel = LevelWidth'(1 << UrgencyWidth);

  // Cycles each input's pending request has waited, saturating; zero once it is granted or
  // withdrawn.
  logic [NumInp-1:0][AgeWidth-1:0] age_d, age_q;

  for (genvar i = 0; i < NumInp; i++) begin : gen_inp
    logic starved;
    assign age_d[i] = !(valid_i[i] && !ready_i[i]) ? '0 : (&age_q[i]) ? age_q[i] : age_q[i] + 1'b1;
    assign starved  = guard_en_i && valid_i[i] && (age_q[i] >= guard_threshold_i);
    always_comb begin
      if (override_en_i[i]) level_o[i] = override_level_i[i];
      else if (starved)     level_o[i] = GuardLevel;
      else if (urgency_en_i) level_o[i] = LevelWidth'(urgency_i[i]);
      else                  level_o[i] = '0;
    end
  end

  `FF(age_q, age_d, '0, clk_i, rst_ni)

  `ASSERT_INIT(GuardAboveUrgency, LevelWidth > UrgencyWidth)

endmodule
