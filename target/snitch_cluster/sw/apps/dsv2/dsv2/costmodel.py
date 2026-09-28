# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""A cycle model of DeepSeek-V2-Lite layer 1 decode on snax_split_cluster clusters.

    python3 -m dsv2.costmodel             (from target/snitch_cluster/sw/apps/dsv2)

prints the model next to the level 1 runs (the calibration check), then the four-cluster
HeMAiA estimates.

WHAT SETS THE PACE. A decode pass streams every weight once, INT8, through GEMVs. One token's GEMV
runs the (1, 4, 32) array shape, 128 weight bytes a cycle; two or more tokens run (16, 4, 16) with
a row per token, 64 bytes a cycle for all of them, and a streamed GEMV sustains ARRAY_EFF of
either. Weights reach a cluster by one or two routes: the iDMA's AXI reads over its 512-bit port,
and the memory-side xDMA pushing them in as AXI writes. One token takes both, with the TCDM
arbitration serving each requester's own priority bit; two tokens take the iDMA alone, which
already feeds their 64 bytes a cycle. A pass is

    weight bytes / the stream's rate                          the GEMV phases
  + key tiles x the cycles per tile                           attention, array-bound
  + what the stream does not hide                             SIMD stages between phases,
                                                              phase boundaries, the pass's ends

LEVEL 1 CALIBRATION (docs/dsv2_layer1_plan.md, section 8), from the apps' wait profiles:

  STREAM_MLA  B/cycle end to end for the MLA's weights, by routes: the block's cycles less its
              attention tiles and MLA_REST
  TILE        cycles per attention tile of 64 keys and 32 query lanes: the MLA at L + 1 = 4,096
              against L + 1 = 512, 56 more tiles
  MLA_REST    the MLA's other exposed cycles, per tokens per pass: the GEMM's waits for the
              SIMD, its array drains at phase boundaries, the attention's fill
  STREAM_MOE  B/cycle end to end for the MoE's weight stream, by routes: with one, the iDMA's
              port sets the pace; with two, the TCDM the loads and the array share
  MOE_REST    the MoE's other exposed cycles: the input norm before the first GEMV, the last
              down's dequantisation and combine

LEVEL 3, HEMAIA (a model, not a measurement): four clusters behind the routes from L3, PORT_EFF of
64 B/cycle each as at level 1. Each cluster's own link takes at most what one cluster streamed at
level 1, and its array ARRAY_EFF of its shape's rate, so a phase takes the longest of: all its
bytes over the routes, and the busiest cluster's bytes over its link and through its array. The
mapping (the plan's C1-C4):

  projections  W_DKV, W_Q and W_UK split four ways; every cluster normalises x itself
  C1           the query all-gather
  attention    a quarter of the cached tokens per cluster; the cache slices over the routes
  C2           (m, l, o~) merged by head
  W_UV, W_O    by head and by input row; C3 adds W_O's partial sums in the fabric
  MoE          units of 8.25 MiB -- the shared experts' two halves and every slot -- spread over
               the clusters; the router's top-6 hides behind the shared halves; C4 gathers

INT4 halves the bytes but not the array's passes. A two-token (2, 4, 32) shape (the plan's HW-1)
would give two tokens one token's 128 bytes a cycle, and so the second route.
"""

from dataclasses import dataclass

# ---- DeepSeek-V2-Lite layer 1, INT8 bytes ----------------------------------------------------
W_DKV = 2048 * 576
W_Q = 2048 * 3072
W_UK = 16 * 128 * 512
W_UV = 16 * 512 * 128
W_O = 2048 * 2048
W_ROUTER = 2048 * 64
EXPERT = 2048 * 2816 + 1408 * 2048          # gate|up + down: 8.25 MiB; the shared experts are two
FACTORS = 2 * (576 + 3072 + 16 * 512 + 16 * 128 + 2048)   # the MLA's dequant factors, FP16
MLA_W = W_DKV + W_Q + W_UK + W_UV + W_O
ROW_KEY = 576                               # a cached token in the key copy
ROW_VAL = 512                               # ... and in the value copy
N_EXPERTS, TOP_K = 64, 6
BC = 64                                     # keys per attention tile

# ---- level 1 calibration (see the module docstring) -------------------------------------------
STREAM_MLA = {1: 59.6, 2: 82.2}   # by routes
TILE = 3_175
TILE_FILL = 3_000
MLA_REST = {1: 21_400, 2: 35_700}  # by tokens per pass
STREAM_MOE = {1: 63.7, 2: 90.8}   # by routes
MOE_REST = {1: 1_000, 2: 1_000}

# ---- the array and level 3 -------------------------------------------------------------------
ARRAY_1TOK = 128.0     # INT8 B / cycle one token's (1, 4, 32) GEMV takes
ARRAY_NTOK = 64.0      # ... two or more tokens' (16, 4, 16), a row per token
ARRAY_EFF = 0.96       # of it, a streamed GEMV
PORT = 64.0            # B / cycle, one route
PORT_EFF = 0.96        # of it, end to end, as the level 1 iDMA's stream
EXCHANGE = 2_000       # cycles per exchange between the clusters (the page's figure)
MERGE = 3_000          # C2's merge and the normalisation after it
NORM = 1_100           # the input norm and its A operand, before any GEMV (level 1)


def expected_slots(ntok):
    """Distinct experts among ntok tokens' top-6, routing taken as uniform."""
    return N_EXPERTS * (1.0 - (1.0 - TOP_K / N_EXPERTS) ** ntok)


@dataclass
class Pass:
    L: int = 511           # cached tokens
    ntok: int = 1          # tokens per pass
    nslot: int = 6         # expert slots: the union of the tokens' top-6
    int4: bool = False     # INT4 weights: half the bytes, the same array passes
    two_tok_shape: bool = False  # a (2, 4, 32) array shape for two tokens (HW-1)
    value_copy: bool = True   # the cache's second, transposed copy; else transposed on chip (HW-4)

    @property
    def wscale(self):
        return 0.5 if self.int4 else 1.0

    @property
    def array(self):
        """INT8 weight bytes a cycle the GEMV sustains."""
        wide = self.ntok == 1 or (self.two_tok_shape and self.ntok == 2)
        return (ARRAY_1TOK if wide else ARRAY_NTOK) * ARRAY_EFF

    @property
    def routes(self):
        """Routes one cluster loads its weights by: both when its GEMV can take them."""
        return 2 if self.array > ARRAY_NTOK else 1

    def cache_bytes(self):
        return (self.L + self.ntok) * (ROW_KEY + (ROW_VAL if self.value_copy else 0))

    def tiles(self, clusters=1):
        return -(-(self.L + self.ntok) // (clusters * BC))

    def moe_bytes(self):
        return W_ROUTER + (self.nslot + 2) * EXPERT


def one_cluster(p: Pass, routes=None):
    """Level 1: one split cluster, its weights by `routes` (default: p.routes)."""
    w, r = p.wscale, routes or p.routes

    def stream(nbytes, rate):
        return max(nbytes * w / rate, nbytes / p.array)

    t_mla = (stream(MLA_W + FACTORS, STREAM_MLA[r]) + p.tiles() * TILE + TILE_FILL
             + MLA_REST[p.ntok])
    t_moe = stream(p.moe_bytes(), STREAM_MOE[r]) + MOE_REST[p.ntok]
    return dict(mla=t_mla, moe=t_moe, total=t_mla + t_moe, per_token=(t_mla + t_moe) / p.ntok)


def four_clusters(p: Pass, routes=1.0):
    """Level 3: four clusters behind `routes` routes from L3."""
    w = p.wscale
    port = PORT * PORT_EFF * routes
    link = min(port, STREAM_MOE[2])

    def stream(total, busiest):
        return max(total * w / port, busiest * w / link, busiest / p.array)

    p1 = NORM + stream(W_DKV + W_Q + W_UK, (W_DKV + W_Q + W_UK) / 4)
    att_compute = p.tiles(4) * TILE + TILE_FILL
    att_port = p.cache_bytes() / port
    att = max(att_compute, att_port)
    p3 = stream(W_UV + W_O, (W_UV + W_O) / 4)
    units = p.nslot + 2
    moe = stream(p.moe_bytes(), -(-units // 4) * EXPERT) + EXCHANGE
    phases = dict(proj=p1, c1=EXCHANGE, att=att, c2=EXCHANGE + MERGE, out=p3, c3=EXCHANGE,
                  moe=moe)
    total = sum(phases.values())
    return dict(total=total, per_token=total / p.ntok, phases=phases,
                att_bound="compute" if att_compute >= att_port else "routes")


def _k(x):
    return f"{x / 1e3:7.0f} K"


# The level 1 runs this model is calibrated on: (what, pass, routes, block, measured cycles).
LEVEL1 = [
    ("MLA, 1 token, L + 1 = 512, two routes", Pass(L=511), 2, "mla", 217_521),  # snax-dsv2-mla
    ("MLA, 1 token, L + 1 = 512, iDMA alone", Pass(L=511), 1, "mla", 281_355),
    ("MLA, 1 token (layer), two routes", Pass(L=511), 2, "mla", 216_839),        # snax-dsv2-layer
    ("MLA, 2 tokens, L = 511", Pass(L=511, ntok=2), 1, "mla", 298_818),          # snax-dsv2-spec
    ("MoE, 1 token, 6 slots, two routes", Pass(L=511), 2, "moe", 764_226),       # snax-dsv2-layer
    ("MoE, 1 token, 6 slots, iDMA alone", Pass(L=511), 1, "moe", 1_089_107),     # snax-dsv2-moe
    ("MLA, 2 tokens, L = 512", Pass(L=512, ntok=2), 1, "mla", 298_909),
    ("MoE, 2 tokens, 11 slots", Pass(L=511, ntok=2, nslot=11), 1, "moe", 1_833_070),  # -spec
    ("MoE, 2 tokens, 12 slots", Pass(L=512, ntok=2, nslot=12), 1, "moe", 1_974_806),
]

SCENARIOS = [
    ("L = 512, 1 token, INT8", Pass(L=511)),
    ("L = 4,096, 1 token, INT8", Pass(L=4095)),
    ("L = 4,096, 1 token, INT8, one cache copy", Pass(L=4095, value_copy=False)),
    ("L = 512, 2 tokens (12 slots), INT8", Pass(L=511, ntok=2, nslot=12)),
    ("L = 512, 2 tokens (12 slots), INT8, (2, 4, 32)",
     Pass(L=511, ntok=2, nslot=12, two_tok_shape=True)),
    ("L = 4,096, 2 tokens (12 slots), INT8", Pass(L=4095, ntok=2, nslot=12)),
    ("L = 512, 4 tokens (21 slots), INT8", Pass(L=511, ntok=4, nslot=21)),
    ("L = 512, 1 token, INT4", Pass(L=511, int4=True)),
    ("L = 4,096, 2 tokens, INT4, (2, 4, 32)",
     Pass(L=4095, ntok=2, nslot=12, int4=True, two_tok_shape=True)),
]


def report():
    lines = ["LEVEL 1, one cluster: the model against the measured runs (cycles)"]
    for name, p, r, part, m in LEVEL1:
        mdl = one_cluster(p, r)[part]
        err = 100 * (mdl / m - 1)
        lines.append(f"  {name:34s} model {_k(mdl)}  measured {_k(m)}  ({err:+5.1f}%)")
    lines.append("")
    lines.append("LEVEL 3, four clusters on HeMAiA, cycles per token "
                 "(a route = 64 B/cycle at the port; one cluster by its own routes)")
    lines.append(f"  {'scenario':44s} {'1 cluster':>9s} {'1 route':>9s} {'1.5x':>9s} {'2x':>9s}")
    for name, p in SCENARIOS:
        one = one_cluster(p)["per_token"] if p.ntok in MLA_REST else float("nan")
        row = [four_clusters(p, r)["per_token"] for r in (1.0, 1.5, 2.0)]
        one_s = _k(one) if one == one else "      n/a"
        lines.append(f"  {name:44s} {one_s}   {_k(row[0])}   {_k(row[1])}   {_k(row[2])}")
    lines.append("")
    for p, r in ((Pass(L=4095), 1.0), (Pass(L=4095), 2.0)):
        f = four_clusters(p, r)
        lines.append(f"  L = 4,096, 1 token, {r:.0f} route(s), by phase: " + ", ".join(
            f"{k} {v / 1e3:.0f} K" for k, v in f["phases"].items()) +
            f"; attention {f['att_bound']}-bound")
    lines.append(f"  expected slots per pass: 1 token {expected_slots(1):.1f}, 2 tokens "
                 f"{expected_slots(2):.1f}, 4 tokens {expected_slots(4):.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
