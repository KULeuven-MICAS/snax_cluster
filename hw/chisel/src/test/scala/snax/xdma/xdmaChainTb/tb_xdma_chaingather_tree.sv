// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// TREE vs CHAIN reduction over 16 endpoints, and the synchronisation question.
//
// Stage 1, in PARALLEL: N/G group gathers, group g collected at endpoint g*G.
// Stage 2, SERIAL:      a chain gather over the group collectors, ending at endpoint 0.
//
//   G=4:   ep3  ep2  ep1  -> ep0  \
//          ep7  ep6  ep5  -> ep4   |
//          ep11 ep10 ep9  -> ep8   |   then   ep12 ep8 ep4 -> ep0
//          ep15 ep14 ep13 -> ep12 /
//
// The tree folds all 16 partials, so its answer is byte-identical to the flat P=16 chain --
// same result, different shape, and therefore directly comparable.
//
// ---- Q1: does stage 2 need a barrier behind stage 1? ----
//
// Endpoint 0 issues BOTH its group gather and the stage-2 gather, and its own xDMA task queue
// serialises them, so stage 2 cannot start there before stage 1 has retired. That is the part
// the hardware does give you for free.
//
// What no queue orders is endpoint 4/8/12's stage-1 WRITE of its group result against endpoint
// 0's stage-2 READ of that same address. Different nodes, different queues. So this is a race,
// and it is settled empirically here: the unsynchronised schedule runs against group slots
// pre-filled with 0xDEADBEEF, which as an fp32 is a huge negative number -- if stage 2 ever
// reads a group result early it folds that sentinel and the answer is unmistakably wrong rather
// than plausibly stale. A race is intermittent, so the bench runs it repeatedly and reports how
// many rounds passed; one failure settles it.
//
// ---- Q2: is the tree faster? ----
//
// Compare the SUM OF THE TWO STAGES' xDMA task latencies against the flat chain's, not the
// testbench's wall clock: one stimulus thread here programs all N/G group gathers back to back,
// where real hardware has each group's core programming its own in parallel.

module tb_xdma_chaingather_tree;
  xdma_chaingather_body #(
      .NumEndpoints(16),
      .NumRounds   (3),
      .JunctionId  (0),   // linear fold: byte-exact checking
      .Tree        (1'b1),
      .NumBeats    (1),
      .Verbose     (0)
  ) i_body ();
endmodule
