// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// tree4 with the HANG WATCHDOG armed: stop at the first wide-send stall and dump the
// mechanism, instead of grinding on to SimTimeout with the cause long gone.
module tb_xdma_chaingather_hang_tree4;
  xdma_chaingather_body #(
      .NumEndpoints(16), .NumRounds(2), .JunctionId(0), .Tree(1'b1),
      .TreeOnlyG   (4),  .TreeUnsyncG(4), .NumBeats(1), .Verbose(1),
      .HangWatch   (2000)
  ) i_body ();
endmodule
