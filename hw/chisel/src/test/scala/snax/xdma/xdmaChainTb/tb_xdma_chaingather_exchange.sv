// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// CONCURRENT MULTI-ISSUER, exchange: ep0 <-> ep1 and ep2 <-> ep3, all four transfers live at
// once, so EVERY endpoint is simultaneously the source of one remote write and the destination
// of another.
//
// This is the sharper of the two multi-issuer shapes. A chain keeps "sources the payload" and
// "takes delivery of a payload" on different nodes; here they land on the same node at the same
// time, which is the case a head-claim guard keyed on chain position alone cannot judge: a
// spurious claim and a genuine outgoing write present the same position bits.
//
// The failure it guards against is SILENT: data arrives correctly and every FSM returns to
// idle, but a sender's completion never reaches its core. It shows up as `wait_finish` timing
// out on a finish counter, never as a watchdog -- which is why this checks completions rather
// than stalls.
module tb_xdma_chaingather_exchange;
  xdma_chaingather_body #(
      .NumEndpoints(4),
      .NumRounds   (2),
      .MultiSource (2),   // EXCHANGE
      .MultiWidth  (4),
      .NumBeats    (1),
      .Verbose     (1)
  ) i_body ();
endmodule
