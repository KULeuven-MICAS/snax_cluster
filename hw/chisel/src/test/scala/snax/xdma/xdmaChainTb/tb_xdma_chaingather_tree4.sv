// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// The BALANCED tree over 16 endpoints: 4 groups of 4, then a chain of 4.
//
//   ep3  ep2  ep1  -> ep0  \
//   ep7  ep6  ep5  -> ep4   |
//   ep11 ep10 ep9  -> ep8   |   then   ep12 ep8 ep4 -> ep0
//   ep15 ep14 ep13 -> ep12 /
//
// ONE shape per simulation: a wedged run leaves every endpoint stuck, so a second shape measured
// after a first one has failed is only measuring the wreckage.

module tb_xdma_chaingather_tree4;
  xdma_chaingather_body #(
      .NumEndpoints(16),
      .NumRounds   (2),
      .JunctionId  (0),
      .Tree        (1'b1),
      .TreeOnlyG   (4),
      .TreeUnsyncG (4),
      .NumBeats    (1),
      .Verbose     (1)
  ) i_body ();
endmodule
