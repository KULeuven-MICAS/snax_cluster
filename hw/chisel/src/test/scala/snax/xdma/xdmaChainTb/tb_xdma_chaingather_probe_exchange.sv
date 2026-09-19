// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// `sim-exchange` with the head-claim probe on. Diagnostic only.
module tb_xdma_chaingather_probe_exchange;
  xdma_chaingather_body #(
      .NumEndpoints  (4), .NumRounds(1), .MultiSource(2), .MultiWidth(4),
      .NumBeats      (1), .Verbose(1), .ProbeHeadClaim(1'b1)
  ) i_body ();
endmodule
