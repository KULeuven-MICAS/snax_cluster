// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// P=2 with the MONOID junction (the nonlinear (m,l) merge) instead of the linear fold. Every
// other chain testbench here uses JunctionId(0), so the monoid path is only ever exercised by
// `sim-sweep` -- which is the long one. This is the fast gate for it.
module tb_xdma_chaingather_p2mom;
  xdma_chaingather_body #(
      .NumEndpoints(2), .ChainWidth(2), .NumRounds(2), .JunctionId(1), .NumBeats(1)
  ) i_body ();
endmodule
