// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// `sim-tree4` with the head-claim probe on. Diagnostic only.
module tb_xdma_chaingather_probe_tree4;
  xdma_chaingather_body #(
      .NumEndpoints  (16), .NumRounds(2), .JunctionId(0), .Tree(1'b1), .TreeOnlyG(4), .TreeUnsyncG(4),
      .NumBeats      (1), .Verbose(1), .ProbeHeadClaim(1'b1)
  ) i_body ();
endmodule
