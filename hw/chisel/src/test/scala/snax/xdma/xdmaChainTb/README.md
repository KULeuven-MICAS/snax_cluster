# N-endpoint ChainGather testbench

A snax-level replacement for the HeMAiA `snax-xdma-chain-gather-sweep` loop, for everything
that does not need a real D2D hop. Target iteration time: **seconds to a minute**, against
~40 min for the 16-chiplet HeMAiA run.

## What is in it

The DUT is `snax_xdma_cluster_xdma_wrapper` instantiated N times behind two AXI crossbars.
That wrapper is the right unit because it already contains **both halves of the stack** — the
Chisel frontend `snax_xdma_cluster_xdma` *and* `xdma_axi_adapter_top` — and every ChainGather
root cause so far has lived in one of them.

| modelled | not modelled |
|---|---|
| the real Chisel xDMA frontend | CVA6 host, boot ROM, snitch cores |
| the real `xdma_axi_adapter_top` | D2D links, routers, `chip_id` address bits |
| real AXI wide + narrow transport | mesh adjacency and X-first routing |
| real TCDM traffic through the reader/writer engines | the memory chiplet |
| the real CSR programming sequence | UART / EOC / `check_finish` |

Both buses are wired: data rides the wide bus, and **cfg, grant and finish all ride the narrow
bus**, so a chain cannot form without it.

The scope cut that matters: **every endpoint reaches every other in one hop.** Chain adjacency
is meaningless here, which removes the topology as a variable. A bug that needs a real D2D hop
will not reproduce here and must go back to HeMAiA.

## Running it

```bash
cd hw/chisel/src/test/scala/snax/xdma/xdmaChainTb
make sim-sweep   # THE SWEEP: P in {2,4,8,16} x {lin,mom}, one run, latency table
make sim-p3      # 3 endpoints, ONE middle hop -- the smallest reproducer
make sim-p2      # the control: no middle hop
make sim-p4      # two middle hops, the HeMAiA failing case
make sim-p16     # 16 endpoints, one width
make sim-role    # ~1 min: can a node that COLLECTED then be a MIDDLE HOP?
make sim-tree4   # the balanced G=4 two-stage tree + the barrier question
make sim-tree    # every tree shape, G in {2,4,8}
make sim-bench   # sw baseline vs chain vs tree: latency, hops, SRAM; P and volume sweeps
make sim-all     # p2, p3, p4
make gui-p3      # same, in the GUI
```

`sim-role` is the fastest gate for the adapter's stranded-AW-descriptor bug (§ *The role change*
below) and `sim-tree4` for the tree; run both after any adapter change.

These are SystemVerilog testbenches living in the Scala test tree so that everything testing the
xDMA sits in one place. **sbt does not see them** -- only `bender` does, through the
`xdma_chain_tb` target -- so `sbt test` neither builds nor runs them; use the Makefile.

`work-vsim/compile.tcl` depends on the actual sources, so an edit cannot leave a stale compiled
design in place. After any **Chisel** edit you must still regenerate the wrapper — what compiles
is `target/snitch_cluster/generated/snax_xdma_cluster_xdma/`, not the Scala.

## The chain

```
endpoint P-1  ->  endpoint P-2  ->  ...  ->  endpoint 1  ->  endpoint 0
   HEAD             MIDDLE                     MIDDLE         TAIL / collector
```

**The collector is always endpoint 0**, mirroring the HeMAiA app whose `SNAKE[0]` is the
collector and whose chain runs from snake index `P-1` down to `1`. Keeping it fixed is what
makes a sweep a real stress: every width re-arms the SAME node, so a node that fails to retire
poisons the next width rather than hiding on an endpoint nobody revisits.

The TCDM map is the app's: each endpoint keeps its linear partial at `0x000` and its moment
partial at `0x080`; the collector's results land at `0x040` and `0x0C0`. Distinct, so a fold
that accidentally writes over an operand shows up as a wrong answer rather than a pass.

## The sweep

`make sim-sweep` runs every width in `{2, 4, 8, 16}` against BOTH junctions in one simulation
and prints the same table the HeMAiA app prints:

```
[Sweep]   P  fold  task_cc  wall_cc  result
```

- **lin** — `ElementwiseJunction`, per-element FP32 ADD, checked **byte-exact** (the operands are
  integer-valued, so the sum is exact).
- **mom** — `MonoidJunction`, the nonlinear online-softmax merge
  `(m1,l1) (+) (m2,l2) = (max(m1,m2), l1*exp(m1-m*) + l2*exp(m2-m*))`. `m*` must be exact; `l*`
  goes through the writer's exp LUT and is checked to a ULP bound, with the measured delta
  printed so the bound can be tightened. Geometry word `0x0C040101` puts `m` at lane 0 and `l`
  at lane 8.

### Measured (16 endpoints, 3 rounds each, all three fixes in)

```
[Sweep]   P  fold  task_cc  wall_cc  result
[Sweep]   2  lin       56       60  PASS
[Sweep]   4  lin      171      176  PASS
[Sweep]   8  lin      415      420  PASS
[Sweep]  16  lin      903      908  PASS
[Sweep]   2  mom       59       64  PASS
[Sweep]   4  mom      180      184  PASS
[Sweep]   8  mom      436      440  PASS
[Sweep]  16  mom      948      952  PASS
```

These are **2 cycles higher at P >= 4** than the numbers this file used to quote (169/413/901,
178/434/946). That is not drift: the `xdma_req_backend` fix moved the write grant from descriptor
emission to descriptor acceptance, which costs one round trip once per transfer. P=2 is
unchanged. If you measure the older numbers, you are running without that fix.

The fold scales linearly in the chain width: about **60 cycles per additional hop**
(`(903-56)/(16-2) = 60.5` for the linear arm), with the nonlinear merge costing a further 3-5%
— consistent with its deeper operator. `task_cc` is `XDMA_PERF_CTR_TASK`; `wall_cc` is the
testbench's own count from the start-CSR write to the finish counter reaching its target, so the
few-cycle gap between them is the CSR round trip.

The monoid `l*` error against a `real`-arithmetic golden was 0, 2, 9 and 8 ULP for P = 2, 4, 8,
16 — the writer's exp LUT is accurate to about 1e-6 relative over this key range.

**What the numbers do and do not include.** There are no D2D hops, routers or `chip_id`
translation here, and every endpoint reaches every other in one hop. So these are the
FABRIC-INTERNAL cost of the fold — the part the xDMA controls. Expect HeMAiA's to be larger by
roughly the per-hop D2D latency times the chain length, and use these to reason about *scaling*
rather than to predict absolute silicon latency.

`P=2` has **no** middle hop and is the control. `P=3` is the smallest configuration with one,
and a middle hop is the only node that both receives and forwards — the only one that reaches
`xdma_grant_manager`'s `WRITE_MIDDLE` and `xdma_finish_manager`'s `WriteMiddleBusy` /
`SendToPreviousHop`, and the only one handed a writer-side frame for a local write it must
never perform.

## The collective comparison (`make sim-bench`)

Three ways to reduce the same data, measured side by side: a **software baseline** (the root DMAs
each partial in and folds it with the cluster SIMD, double-buffered), the **chain**, and the
**tree**. Volume per endpoint is fixed and P is swept, then P is fixed and volume is swept.

Reported per scheme: end-to-end latency, xDMA-busy latency, fabric transactions and beats,
**hop·beats** (beats weighted by Manhattan distance on the 4x4 chiplet array), total TCDM word
accesses and the busiest endpoint's share, and the number of task submissions.

Headline at P=16, 4 KiB per endpoint: chain is **3.4x** faster end-to-end than the baseline and
uses **4.5x** less SRAM in total — **31x** less through the busiest endpoint. The tree is **2.1x**
the chain on xDMA-busy time and is the only scheme whose every transfer is nearest-neighbour
(1.0 hops per beat, against the chain's 1.6 and the baseline's 3.2). Full write-up and the volume
analysis: `tmp/collective-comparison.md`.

Two things about this bench specifically:

- **The fetch is real RTL; the SIMD accumulate is modelled** at `SimdBytesPerCycle` (default 64
  B/cycle, the same width the xDMA datapath gets — the most favourable assumption available, so
  the baseline is never beaten by a strawman). Its TCDM traffic is counted into separate
  `sram_*_model` counters so measured and modelled are never mixed.
- **Volume is a runtime quantity** (`set_volume()`), so the volume sweep is one elaboration. The
  TCDM map is sized for the compile-time `NumBeats`, which is therefore the maximum.

## Traps this bench exposed (all now fixed)

Three latent bugs surfaced the first time anything ran at more than one beat or compared more
than one scheme. Worth knowing about, because two of them make a *wrong* run look right:

1. **The goldens disagreed with the seed for multi-beat transfers.** `seed_partials()` gives every
   lane a distinct value; the checker compared against `golden_lin(width, lane % LanesPerBeat)`.
   Identical at one beat — but at 64 beats the checker would have passed a transfer that replayed
   beat 0 sixty-four times.
2. **The SRAM/fabric counters under-counted by up to 16x.** One `always_ff` per port and per
   endpoint, all assigning the same variable: several drivers, and with nonblocking assignment
   only the last write per cycle survives, so 16 simultaneous TCDM accesses were recorded as one.
   Now one `always_ff` accumulates across the whole array.
3. **The Makefile ran stale designs after a compile error.** The guard grepped `^\*\* Error`, but
   `vsim -c` prefixes echoed lines with `# `, so it never matched — a `vlog` failure went straight
   on to `vopt`/`vsim`, which ran the *previous* design and produced a complete, plausible,
   entirely stale set of results. Fixed, and the same guard added after `vopt`.

## The role change, and the tree

A flat ChainGather never changes a node's role: the collector is fixed and everyone else only
ever forwards to the same neighbour, in every round and at every width. A **tree** is the first
structure that asks a node to collect a group and then be a middle hop of a second stage — and
that turned out to be a whole bug class of its own.

`make sim-role` is the minimal reproducer, three endpoints, about a minute:

```
CONTROL: ep2 -> ep1 -> ep0, twice          (ep1 is a MIDDLE both times)
TEST:    ep1 COLLECTS  (ep2 -> ep1)
         ep2 -> ep1 -> ep0                 (ep1 is now a MIDDLE)
```

The control runs the identical chain twice, so "a second task at that node" is controlled for;
the only difference in the test is the change of role. It **deadlocked** until the fix in
`xdma_axi_adapter/src/xdma_req_backend.sv`: AW descriptors were pushed into an emitter FIFO
unconditionally but popped only once the write was granted, and that FIFO has `flush_i` tied to
`1'b0`, so an ungranted descriptor outlived its own transfer and mis-routed the next write to the
*previous* transfer's destination. Invisible whenever the destination does not change — which is
why the flat sweep never caught it.

`make sim-tree4` then runs the balanced G=4 tree (four group gathers to ep0/4/8/12, then a chain
over those to ep0), which folds all 16 partials and is therefore byte-comparable with the flat
P=16 chain: **342 vs 903 cycles, 2.64x**. Its PHASE 0 runs stage 2's *shape* alone on a fresh
design, which is what separated "the shape is wrong" from "the role change is wrong".

### The barrier probe, and its positive control

`sim-tree4` also answers whether stage 2 needs a barrier behind stage 1. It does not — but it
does need every stage-2 participant's own stage-1 task to be **submitted to that node's queue**
before the stage-2 cfg arrives. Nothing need have completed: a stage-2 cfg queues behind the
node's stage-1 task, which retires only once its group result has landed in that node's TCDM, and
every group result is read by the node that produced it.

Two schedules run, differing only in when the straggler group is programmed. Group slots are
pre-filled with `0xDEADBEEF`, so an early read cannot pass as a plausible stale value:

- **pre-issue** — straggler still programmed before stage 2 is issued: passes at every skew up to
  1600 cycles, more than nine times a whole group gather.
- **post-issue** — straggler programmed after stage 2 is issued: **fails at zero skew.**

The post-issue schedule is *supposed* to fail, so it runs under an `expect_fail` flag: its
failure is printed as `[expected-FAIL]` and does not fail the simulation. It is a **positive
control** — it proves the sentinel and the checker can actually detect a premature read — and the
verdict complains loudly if it ever starts passing, because that would mean the probe has stopped
proving anything. The probe stops at the first failing skew and prints `not run` for the rest
rather than backfilling passes it never measured.

## Why three rounds

The bug class is *"only the first gather through a chain with middle nodes works"*: round 1
byte-exact, round 2 returns having moved nothing. **A sweep that runs each configuration once
cannot see any of it.** `NumRounds` defaults to 3.

Per round the testbench checks:

1. the collector's buffer is byte-exact against a golden computed here;
2. the `0xDEADBEEF` sentinel is gone — the difference between "moved nothing" and "moved the
   wrong thing";
3. `XDMA_JCT_STATUS` shows no cfgerr and no starvation;
4. every endpoint is back at rest (`io_status_readerBusy` / `writerBusy` low, no
   `xdma_stall_error`) **before the next round starts**. A parked FSM is the failure signature
   and it is invisible until the round after, so this is checked explicitly rather than
   inferred from a passing round.

Note that a round reporting a *small* `XDMA_PERF_CTR_TASK` is not a fast completion:
the counter resets on the start-CSR write and only freezes when the task finishes, so with no
finish the value read is just "cycles since the start write". Every verdict here is gated on
the data, never the counter.

## Traps this testbench already encodes

- **`csr_req_bits_addr_i` is a plain 0-based register index.** `SNAX_XDMA_CFG_ADDR = 960` in
  `snax_xdma_lib.h` is the RISC-V CSR *number* base, stripped by the snitch cluster's CSR
  decode, which this testbench bypasses. Drive the `XDMA_*_PTR` value, never `960 +` it.
- **The endpoint stride is 4 MiB, not the 1 MiB a flat map would suggest.** The xDMA's config
  routers decode nothing but a pointer's cluster tag, which for this config sits above bit 17
  (`tcdm.size: 128` KiB). 4 MiB is the real system's `cluster_base_offset`, so the address
  arithmetic here is identical to HeMAiA's. A smaller stride gives two endpoints the same tag
  and the chain silently short-circuits.
- **The junction bank puts the ENABLE BITMASK FIRST**, then the per-junction user CSRs.
  Reversing them arms no junction, the transfer degrades to a plain chained write, and a test
  that only checks "it completed" passes vacuously.
- **`enabledByte` is 8 bits** (`tcdmParam.dataWidth/8`), not the 64 B beat.
- **Unused temporal dims must be 1, not 0.**
- **`MonoidJunction` CSR(0) is a geometry word**, `0x0C040101` for the `(m,l)` merge. The
  plausible-looking `(1<<13)|1` decodes to `n=0, sigma=0` — key-only — so `l` at lane 8 is
  never read and the fold silently returns garbage.
- The CSR map is cross-checked against `ReqRspManager`'s own bounds assertions in the generated
  netlist (`Max: 77` write, `85` read). If a regenerated netlist prints different bounds, the
  config changed and the map in `xdma_chaingather_body.sv` is stale.

## Dependency pin

`xdma_axi_adapter` is pinned **by branch** (`kfc/chaingather-adapter`), so it floats.
`bender update` alone is not enough: with a stale ref cache it prints nothing, exits 0, and
leaves `Bender.lock` on the old commit. Always

```bash
bender update --fetch
grep -A2 '^  xdma_axi_adapter:' Bender.lock    # revision must be the branch head
```

This matters here: the branch carries a finish-manager fix that postdates the lock revision
this repo shipped with (`896f1380` -> `65dd629c`).
