// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// THE SWEEP: every chain width times both folds, in ONE simulation, with a latency table at the
// end. This is the snax-level twin of the HeMAiA app `snax-xdma-chain-gather-sweep`, and it
// prints the same `P / fold / task_cc / wall_cc / result` table so a number measured here and a
// number measured on silicon-equivalent RTL are directly comparable.
//
// What it does NOT model, and therefore what the numbers do NOT include: D2D hops, routers,
// chip_id address translation, and the cross-chip flag handshake the app needs to stage its
// partials. Every endpoint here reaches every other in one hop. So treat these cycle counts as
// the FABRIC-INTERNAL cost of the fold -- the part the xDMA controls -- and expect HeMAiA's to
// be larger by the per-hop D2D latency times the chain length.
//
// Each configuration is run `NumRounds` times, because the bug class this testbench exists for
// is "the first gather works and the next one does not". The table reports round 1's latency and
// flags the configuration FAIL if any round failed.

module tb_xdma_chaingather_sweep;
  xdma_chaingather_body #(
      .NumEndpoints(16),
      .NumRounds   (3),
      .Sweep       (1'b1),
      .NumBeats    (1),
      .Verbose     (0)
  ) i_body ();
endmodule
