// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// `sim-star` with the to-remote window probe on. The control for `probe_exchange`: same
// program_copy, same senders, but no sender is also a receiver -- and this one COMPLETES.
module tb_xdma_chaingather_probe_star;
  xdma_chaingather_body #(
      .NumEndpoints  (4), .NumRounds(1), .MultiSource(1), .MultiWidth(4),
      .NumBeats      (1), .Verbose(1), .ProbeHeadClaim(1'b1)
  ) i_body ();
endmodule
