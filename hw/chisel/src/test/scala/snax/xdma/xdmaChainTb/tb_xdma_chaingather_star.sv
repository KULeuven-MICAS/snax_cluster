// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// CONCURRENT MULTI-ISSUER, star-in: ep1, ep2 and ep3 all remote-write into ep0 at the same
// time. Every transfer is launched before any is waited on.
//
// Nothing else in this repo has more than one transfer in flight against a shared endpoint.
// A chain is linear -- every node is a hop in exactly one transaction -- and that is the
// assumption the whole grant/finish protocol rests on. This is the smallest thing that breaks
// it, on the real stack: the real Chisel frontend, the real `xdma_axi_adapter_top`, real AXI
// transport, and the real CSR programming sequence.
//
// Two rounds, because a transfer that leaves a context parked shows up on the round after.
module tb_xdma_chaingather_star;
  xdma_chaingather_body #(
      .NumEndpoints(4),
      .NumRounds   (2),
      .MultiSource (1),   // STAR-IN
      .MultiWidth  (4),
      .NumBeats    (1),
      .Verbose     (1)
  ) i_body ();
endmodule
