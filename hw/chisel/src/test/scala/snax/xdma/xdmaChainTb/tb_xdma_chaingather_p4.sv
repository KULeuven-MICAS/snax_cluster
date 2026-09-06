// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// P=4: the configuration the HeMAiA sweep fails on. Two middle hops.
//
//   ep3 (HEAD) -> ep2 (MIDDLE) -> ep1 (MIDDLE) -> ep0 (collector)
//
// Measured on HeMAiA: round 1 byte-exact in 505-551 cycles, round 2 retires in ~26 cycles with
// the 0xDEADBEEF sentinel intact.

module tb_xdma_chaingather_p4;
  xdma_chaingather_body #(
      .NumEndpoints(4),
      .ChainWidth  (4),
      .NumRounds   (3),
      .JunctionId  (0),
      .NumBeats    (1)
  ) i_body ();
endmodule
