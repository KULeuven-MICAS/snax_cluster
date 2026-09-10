// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// THE COLLECTIVE COMPARISON: three ways to reduce the same data, measured side by side.
//
//   1. sw     -- the naive software baseline. ep0 pulls each remote partial in with an ordinary
//                DMA read and folds it with the cluster SIMD, double-buffered so the fetch of
//                partial k+1 overlaps the accumulate of partial k. No collective at all.
//   2. chain  -- one CHAINGATHER across all P endpoints: the fold happens in the fabric, in the
//                junction at each hop, and nothing lands in memory until the root.
//   3. tree   -- P/G group gathers in parallel, then a chain over the group collectors.
//
// Volume per endpoint is FIXED at NumBeats*64 B and the endpoint count is swept, which is the
// comparison that answers "when is a collective worth it". All three fold the same P partials
// into the same answer at ep0 and are checked against the same golden.
//
// Set NumBeats to change the volume: 64 -> 4 KiB, 16 -> 1 KiB, 1 -> 64 B. The volume sweep
// matters because the three schemes have different fixed costs and different per-byte costs,
// so which one wins is a function of volume as well as of P.

module tb_xdma_chaingather_bench;
  xdma_chaingather_body #(
      .NumEndpoints     (16),
      .Bench            (1'b1),
      .NumBeats         (64),  // 64 x 64 B = 4 KiB per endpoint
      .SimdBytesPerCycle(64),
      .Verbose          (0)
  ) i_body ();
endmodule
