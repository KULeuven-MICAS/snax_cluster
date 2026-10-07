// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// Does a ChainGather middle hop fold correctly while its xDMA runs another task?
//
// Four endpoints: the P=3 chain ep2 -> ep1 -> ep0, and ep3 as the endpoint that pulls from the
// hop in case 3. Each case overlaps the gather with another task on ep1's xDMA (see
// run_overlap); `+OVL_CASE=<n>` runs one case alone.

module tb_xdma_chaingather_overlap;
  xdma_chaingather_body #(
      .NumEndpoints(4),
      .NumRounds   (1),
      .JunctionId  (0),
      .Overlap     (1'b1),
      .NumBeats    (1),
      .Verbose     (1)
  ) i_body ();
endmodule
