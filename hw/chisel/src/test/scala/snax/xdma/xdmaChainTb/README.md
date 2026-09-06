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
make sim-all     # p2, p3, p4
make gui-p3      # same, in the GUI
```

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

### Measured (16 endpoints, 3 rounds each, both fixes in)

```
[Sweep]   P  fold  task_cc  wall_cc  result
[Sweep]   2  lin       56       60  PASS
[Sweep]   4  lin      169      174  PASS
[Sweep]   8  lin      413      418  PASS
[Sweep]  16  lin      901      906  PASS
[Sweep]   2  mom       59       64  PASS
[Sweep]   4  mom      178      182  PASS
[Sweep]   8  mom      434      438  PASS
[Sweep]  16  mom      946      950  PASS
```

The fold scales linearly in the chain width: about **60 cycles per additional hop**
(`(901-56)/(16-2) = 60.4` for the linear arm), with the nonlinear merge costing a further 3-5%
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
