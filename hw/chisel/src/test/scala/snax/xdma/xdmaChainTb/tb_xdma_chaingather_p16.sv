// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// P=16: the full 16-endpoint sweep, 14 middle hops. `XDMA_MAX_DST_COUNT = 16` is the hard cap
// on P, so this is the widest chain the hardware can express.

module tb_xdma_chaingather_p16;
  xdma_chaingather_body #(
      .NumEndpoints(16),
      .ChainWidth  (16),
      .NumRounds   (3),
      .JunctionId  (0),
      .NumBeats    (1)
  ) i_body ();
endmodule
