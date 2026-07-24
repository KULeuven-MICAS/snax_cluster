# snax-xdma-unified-merge-rt

Local-loopback smoke test for **`UnifiedMonoidMergeRt`** — one configurable combine cell that folds the whole
in-fabric arithmetic-reduction family into a single netlist, mode-selected per transfer by a 3-bit CSR field.
It supersedes the former separate `StreamMomentMergeRt` / `StreamNormStatMergeRt` / `StreamAttnMergeRt` fixed
modules. The app drives every supported mode through the real SW CSR path (`snax_xdma_enable_dst_ext`) as a
TCDM→xDMA-reader→[writer-side merge]→TCDM loopback, and checks each against a datagen golden.

## CSR layout (one user CSR — the inter-cluster / D2D serdes is untouched)

`csr(0)`:

| bits | field | meaning |
|---|---|---|
| `[7:0]` | `nValid` | live partials in the beat (paired modes); lanes ≥ nValid get the monoid identity |
| `[8]` | `accEn` | 1 = accumulate into a persistent slot; 0 = stateless per beat |
| `[9]` | `accInit` | 1 = this push arms (overwrites) the slot; 0 = folds into it |
| `[12:10]` | `accSlot` | which accumulator (0..numAccSlots−1) |
| `[15:13]` | `combineMode` | selects the op / role vector (below) |

## Field roles

The combine over an F-field FP32 partial gives each field a role; `combineMode` selects the role vector.

| role | `out.f =` | uses α? | hardware |
|---|---|---|---|
| **KEY** | `max(a,b)` (= m\*) | produces α = exp(Δ) | the shared front-end max + one exp LUT |
| **RSUM** | `winner + α·loser` | yes | one FMA + winner-swap + scale mux |
| **SUM** | `a + b` | no (α=1) | the **same** FMA (`a+b = ffma(b,1,a)`) |
| **WSEL** | winner's payload | no | the winner-swap mux, **no** FMA |
| **OFF** | identity (0) | — | parked |

One exp LUT per combine node feeds every RSUM lane (adding lanes never adds exp hardware).

## Supported ops

| # | `combineMode` | op | beat / fold | role vector |
|---|---|---|---|---|
| 0 | `SUM` | LayerNorm / RMSNorm `(Σx, Σx²)` | 8 pairs, paired tree | `(SUM, SUM)` |
| 1 | `MOMENT` | softmax normalizer `(m, ℓ)` | 8 pairs, paired tree | `(KEY, RSUM)` |
| 2 | `ATTN` | flash-attention `(m, ℓ, O[dHead])` | 1 partial, accEn slot | `(KEY, RSUM, RSUM×dHead)` |
| 3 | `MAXPOOL` | max-reduce | 8 pairs, paired tree | `(KEY, OFF…)` |
| 4 | `ARGMAX` | top-1 `(max logit, index)` | 8 pairs, paired tree | `(KEY, WSEL)` |
| 5 | `MOMENT2` | exp-weighted moments `(m, ℓ=Σeˢ, A=Σeˢv, B=Σeˢv²)` | 1 partial, accEn slot | `(KEY, RSUM, RSUM, RSUM)` |

**Beat layouts.** *Paired* modes pack 8 two-field partials: `field0_k` = lane `k` (0..7), `field1_k` = lane
`8+k`. *Single-partial* modes lay one partial across low lanes: lane0 = m, lane1 = ℓ, lanes 2.. = the rest.
Output: paired modes splat `(field1, field0)` into the low 64 bits (result in lane 0 / lane 1); single-partial
modes emit the partial in lanes 0,1,2,…

**accEn (accumulate-on-arrival).** For the single-partial modes (ATTN, MOMENT2), each producer pushes its
partial straight at the merger: arm the slot once with `accInit=1`, then fold subsequent pushes with
`accInit=0`. The monoid is associative + commutative, so arrival order doesn't matter. Slots survive
`ext_start_i` (arming is software's job). SUM with accEn must arm first (the slot's key field inits to the
max-identity, not 0).

## Scope boundary (why these ops and not others)

The cell realizes exactly the homomorphic images, under the exp-affine decode, of `max^{0,1} × (ℝ,+)^p`
("LogSumExp-with-payload"). New arithmetic reductions in that fragment are just a new role vector:

- **In-fragment, free redeployments:** distributed cross-entropy / LM-head logsumexp (= `MOMENT`),
  tensor-parallel matmul all-reduce (= `SUM`), MoE expert-output weighted combine (= `SUM`),
  chunked / long-context attention (= `ATTN` + accEn), joint mean+variance+count (= `SUM×3`).
- **Added here:** `ARGMAX` (the k=1 slice of top-k — a single shared max + a selected index, for greedy decode
  and top-1 MoE routing) and `MOMENT2` (the "add a moment" closure demo — Σeˢv² is another RSUM sharing the
  one α). The per-element `v²` is produced upstream by the SIMD **egress square feature**; the merge cell only
  aligned-adds it (zero new merge silicon).
- **Out of fragment (need a sibling primitive, not this cell):** top-k (k>1) routing / median / quantiles
  (a compare-exchange sorting mesh); product / geometric-mean reductions (unless carried in log-domain).

## Notes

- `ARGMAX` assumes distinct keys; ties break toward the first operand, so under fold reordering an exact tie
  could pick either index. The datagen uses distinct logits.
- Requires `cfg/snax_xdma_test.hjson` (defines `WRITER_EXT_UNIFIEDMONOIDMERGERT`); the app `#error`-guards on it.
- Build: `make CFG_OVERRIDE=cfg/snax_xdma_test.hjson`. Run under vsim via `bin/snitch_cluster.vsim <elf> "" -batch`.
