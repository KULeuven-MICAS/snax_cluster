// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// Does a node that has COLLECTED a gather still work as a MIDDLE HOP of a later one?
//
// Three endpoints, ~1 minute. This is the minimal form of what blocks the two-stage tree: stage
// 1 makes ep4/8/12 collectors and stage 2 asks them to be middle hops. The control runs the very
// same chain twice with no role change, so a control PASS plus a test FAIL isolates the role
// change itself rather than "a second task".

module tb_xdma_chaingather_role;
  xdma_chaingather_body #(
      .NumEndpoints(3),
      .NumRounds   (2),
      .JunctionId  (0),
      .RoleChange  (1'b1),
      .NumBeats    (1),
      .Verbose     (1)
  ) i_body ();
endmodule
