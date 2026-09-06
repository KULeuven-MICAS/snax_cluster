// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// P=2: the CONTROL. Two endpoints, so the chain has NO middle hop -- ep1 (head) straight to
// ep0 (collector). This configuration has always passed on HeMAiA for TWO rounds; it takes a
// THIRD round to expose the stale-cfg defect. If it fails here the
// testbench itself is wrong (address map, CSR indices, TCDM model), not the DUT.

module tb_xdma_chaingather_p2;
  xdma_chaingather_body #(
      .NumEndpoints(2),
      .ChainWidth  (2),
      .NumRounds   (3),
      .JunctionId  (0),
      .NumBeats    (1)
  ) i_body ();
endmodule
