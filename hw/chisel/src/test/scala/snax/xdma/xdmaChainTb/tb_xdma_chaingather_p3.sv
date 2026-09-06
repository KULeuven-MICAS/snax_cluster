// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// P=3: THE PRIZE. Three endpoints means exactly one MIDDLE hop, which is the smallest
// configuration that can reproduce the "only the first gather works" bug class:
//
//   ep2 (HEAD) -> ep1 (MIDDLE) -> ep0 (collector)
//
// A middle hop is the only node that both receives and forwards, so it is the only one that
// reaches `xdma_grant_manager`'s WRITE_MIDDLE and `xdma_finish_manager`'s WriteMiddleBusy /
// SendToPreviousHop -- and the only one handed a writer-side frame for a local write it must
// never perform.
//
// THREE ROUNDS is the point. Round 1 has always passed; the failure is in round 2 onward.

module tb_xdma_chaingather_p3;
  xdma_chaingather_body #(
      .NumEndpoints(3),
      .ChainWidth  (3),
      .NumRounds   (3),
      .JunctionId  (0),
      .NumBeats    (1)
  ) i_body ();
endmodule
