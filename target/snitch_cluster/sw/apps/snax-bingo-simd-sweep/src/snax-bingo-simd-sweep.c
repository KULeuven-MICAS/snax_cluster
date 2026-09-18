// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Cost sweep for the standalone SIMD block: what one whole op COSTS at [rows, cols], in the
// units bingo's cost LUT is keyed in.
//
// WHY THIS IS NOT ONE OF THE snax-simd-* APPS. Those apps report
// `snax_simd_last_task_cycle()` -- the ENGINE's busy counter for one task. bingo's LUT holds
// something else: the whole-kernel span as the driving core sees it, config CSRs and the scalar
// work between passes included (the existing points came from HeMAiA's MGR_RUN_KERNEL counter).
// The two differ by more than a constant -- a 5-pass softmax pays five arming sequences and two
// scalar broadcast loops that no engine counter ever sees -- so a LUT filled from engine spans
// would be a different quantity wearing the same name. Every span below is therefore
// `snrt_mcycle()` on the SIMD hart, around everything the op needs.
//
// WARM, not cold. Each op runs TWICE and the second span is reported, matching the convention
// the HeMAiA sweep used (it dropped its first config as warm-up). The first pass pays the L1I
// miss on the driver and the one-time AGU shape setup; steady state is what a scheduler needs.
// Re-running is safe because every op reads `x_in` and writes only scratch.
//
// NUMERICS ARE NOT CHECKED HERE, deliberately -- see data/datagen.py. The chains below are
// transcribed from the apps that DO check them, pass for pass and AGU shape for AGU shape:
//   silu, streammap        <- snax-simd-silu
//   softmax                <- snax-simd-softmax-multirow
//   rmsnorm                <- snax-simd-rmsnorm-multirow
//   streamreduce           <- snax-simd-reduce-multirow
//   swiglu, streamelementwise <- snax-simd-swiglu
//
// Output, one line per op, parsed by experiments/sweep_split_simd.py:
//   [BINGO] op=<name> rows=<r> cols=<c> cycles=<n>

#include "data.h"
#include "snax-core-roles.h"
#include "snax-simd-lib.h"
#include "snrt.h"

#if !defined(SIMD_EXT_STREAMREDUCE) || !defined(SIMD_EXT_STREAMMAP) || \
    !defined(SIMD_EXT_STREAMELEMENTWISE_1)
#error "Regenerate the SIMD CSR map with StreamReduce, StreamMap and StreamElementwise."
#endif

#define BEAT_BYTES 64
#define FP16_PER_BEAT 32

#define OP_MAX 0u    // StreamReduce: MAX (compare)
#define OP_ADD 1u    // StreamReduce: ADD (fused FMA)
#define OP_SUMSQ 2u  // StreamReduce: SUMSQ (FMA square)
#define EW_MUL 0u    // StreamElementwise fused-FMA: MUL (acc*x)
#define EW_ADD 1u    // StreamElementwise fused-FMA: ADD (acc+x)
#define ACT_LINEAR 0u
#define ACT_EXP 1u
#define ACT_SILU 2u

#define F32_ONE 0x3F800000u
// 0.5 in FP16. A benign input: exp() of it neither overflows nor flushes, and silu's sigmoid
// sits on the steep part of the curve. The FP units are fixed-latency, so the VALUE cannot move
// a cycle count -- this only keeps inf/NaN out of the traces.
#define FP16_HALF 0x3800u

static uint32_t run_task(void) {
    uint32_t id = snax_simd_start();
    snax_simd_wait(id);
    return 0;
}

// The AGU shapes every op below draws from, built once from the sweep point.
typedef struct {
    uint32_t rows, d, beats, row_bytes, rows_bytes, scal_bytes;
    uint32_t red_str[2], red_bnd[2];    // 2-D reader {beats, rows}: a per-row reduce
    uint32_t w_rows_str[1], w_rows_bnd[1];  // writer: one scalar beat per row
    uint32_t flat_str[1], flat_bnd[1];  // 1-D reader/writer over rows*beats
} geom_t;

// ---------------------------------------------------------------- the ops
// Each returns having left the operator chain disabled. `t` is scratch laid out by main().

static void op_streammap(const geom_t* g, uint8_t* in, uint8_t* out) {
    uint32_t csr[3] = {F32_ONE, 0u, ACT_LINEAR};
    snax_simd_enable_ext(SIMD_EXT_STREAMMAP, csr);
    snax_simd_memcpy_nd_fast(in, out, 8, 8, 1, (uint32_t*)g->flat_str,
                             (uint32_t*)g->flat_bnd, 1, (uint32_t*)g->flat_str,
                             (uint32_t*)g->flat_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    run_task();
    snax_simd_disable_ext(SIMD_EXT_STREAMMAP);
}

static void op_streamreduce(const geom_t* g, uint8_t* in, uint8_t* out) {
    uint32_t csr[2] = {g->beats, OP_ADD};
    snax_simd_enable_ext(SIMD_EXT_STREAMREDUCE, csr);
    snax_simd_memcpy_nd_fast(in, out, 8, 8, 2, (uint32_t*)g->red_str, (uint32_t*)g->red_bnd, 1,
                             (uint32_t*)g->w_rows_str, (uint32_t*)g->w_rows_bnd, 0xFFFFFFFF,
                             0xFFFFFFFF, 0xFFFFFFFF);
    run_task();
    snax_simd_disable_ext(SIMD_EXT_STREAMREDUCE);
}

// Two interleaved operands `delta` bytes apart -> one stream. `delta` must be POSITIVE: the AGU
// stride is unsigned, so a second operand below the first wraps and reads outside TCDM while the
// task still reports complete.
static void op_streamelementwise(const geom_t* g, uint8_t* a, uint32_t delta, uint8_t* out,
                                 uint32_t mode) {
    uint32_t str[2] = {delta, BEAT_BYTES};
    uint32_t bnd[2] = {2u, g->rows * g->beats};
    uint32_t csr[2] = {2u, mode};
    snax_simd_enable_ext(SIMD_EXT_STREAMELEMENTWISE_1, csr);
    snax_simd_memcpy_nd_fast(a, out, 8, 8, 2, str, bnd, 1, (uint32_t*)g->flat_str,
                             (uint32_t*)g->flat_bnd, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    run_task();
    snax_simd_disable_ext(SIMD_EXT_STREAMELEMENTWISE_1);
}

static void op_silu(const geom_t* g, uint8_t* in, uint8_t* out) {
    uint32_t csr[3] = {F32_ONE, 0u, ACT_SILU};
    snax_simd_enable_ext(SIMD_EXT_STREAMMAP, csr);
    snax_simd_memcpy_nd_fast(in, out, 8, 8, 1, (uint32_t*)g->flat_str, (uint32_t*)g->flat_bnd, 1,
                             (uint32_t*)g->flat_str, (uint32_t*)g->flat_bnd, 0xFFFFFFFF,
                             0xFFFFFFFF, 0xFFFFFFFF);
    run_task();
    snax_simd_disable_ext(SIMD_EXT_STREAMMAP);
}

// silu(gate) -> `silu_buf`, then silu(gate) (.) up -> out, where `up` is the buffer immediately
// ABOVE silu_buf. The elementwise AGU interleaves its two operands at a positive stride, so the
// pair has to be adjacent and in that order; main() allocates them that way.
static void op_swiglu(const geom_t* g, uint8_t* gate, uint8_t* silu_buf, uint8_t* out) {
    op_silu(g, gate, silu_buf);
    op_streamelementwise(g, silu_buf, g->rows_bytes, out, EW_MUL);
}

// The five-pass per-row softmax, transcribed from snax-simd-softmax-multirow: reduce(MAX),
// scalar -max broadcast, sew(ADD), map(EXP), reduce(ADD), scalar reciprocal broadcast, sew(MUL).
// The two broadcast loops run on this hart and are part of the op's cost -- that is the point.
static void op_softmax(const geom_t* g, uint8_t* x, uint8_t* mx, uint8_t* nbc, uint8_t* xs,
                       uint8_t* expb, uint8_t* sum, uint8_t* rbc, uint8_t* out) {
    uint32_t csr_max[2] = {g->beats, OP_MAX};
    snax_simd_enable_ext(SIMD_EXT_STREAMREDUCE, csr_max);
    snax_simd_memcpy_nd_fast(x, mx, 8, 8, 2, (uint32_t*)g->red_str, (uint32_t*)g->red_bnd, 1,
                             (uint32_t*)g->w_rows_str, (uint32_t*)g->w_rows_bnd, 0xFFFFFFFF,
                             0xFFFFFFFF, 0xFFFFFFFF);
    run_task();
    snax_simd_disable_ext(SIMD_EXT_STREAMREDUCE);

    uint16_t* m = (uint16_t*)mx;
    uint16_t* n = (uint16_t*)nbc;
    for (uint32_t r = 0; r < g->rows; r++) {
        uint16_t neg = (uint16_t)(m[r * FP16_PER_BEAT] ^ 0x8000u);  // low lane of splatted beat
        for (uint32_t c = 0; c < g->d; c++) n[r * g->d + c] = neg;
    }
    op_streamelementwise(g, x, (uint32_t)(nbc - x), xs, EW_ADD);

    uint32_t csr_exp[3] = {F32_ONE, 0u, ACT_EXP};
    snax_simd_enable_ext(SIMD_EXT_STREAMMAP, csr_exp);
    snax_simd_memcpy_nd_fast(xs, expb, 8, 8, 1, (uint32_t*)g->flat_str, (uint32_t*)g->flat_bnd, 1,
                             (uint32_t*)g->flat_str, (uint32_t*)g->flat_bnd, 0xFFFFFFFF,
                             0xFFFFFFFF, 0xFFFFFFFF);
    run_task();
    snax_simd_disable_ext(SIMD_EXT_STREAMMAP);

    uint32_t csr_sum[2] = {g->beats, OP_ADD};
    snax_simd_enable_ext(SIMD_EXT_STREAMREDUCE, csr_sum);
    snax_simd_memcpy_nd_fast(expb, sum, 8, 8, 2, (uint32_t*)g->red_str, (uint32_t*)g->red_bnd, 1,
                             (uint32_t*)g->w_rows_str, (uint32_t*)g->w_rows_bnd, 0xFFFFFFFF,
                             0xFFFFFFFF, 0xFFFFFFFF);
    run_task();
    snax_simd_disable_ext(SIMD_EXT_STREAMREDUCE);

    // 1/Sexp per row. No FPU on this hart, and the reciprocal's VALUE cannot change a cycle
    // count, so the sweep copies the scalar rather than carrying a Newton iteration whose only
    // effect here would be to inflate the span with arithmetic the real kernel does elsewhere.
    uint16_t* s = (uint16_t*)sum;
    uint16_t* rb = (uint16_t*)rbc;
    for (uint32_t r = 0; r < g->rows; r++) {
        uint16_t v = s[r * FP16_PER_BEAT];
        for (uint32_t c = 0; c < g->d; c++) rb[r * g->d + c] = v;
    }
    op_streamelementwise(g, expb, (uint32_t)(rbc - expb), out, EW_MUL);
}

// reduce(SUMSQ) -> per-row scalar -> broadcast -> sew(MUL), the rmsnorm shape.
static void op_rmsnorm(const geom_t* g, uint8_t* x, uint8_t* ss, uint8_t* sbc, uint8_t* out) {
    uint32_t csr[2] = {g->beats, OP_SUMSQ};
    snax_simd_enable_ext(SIMD_EXT_STREAMREDUCE, csr);
    snax_simd_memcpy_nd_fast(x, ss, 8, 8, 2, (uint32_t*)g->red_str, (uint32_t*)g->red_bnd, 1,
                             (uint32_t*)g->w_rows_str, (uint32_t*)g->w_rows_bnd, 0xFFFFFFFF,
                             0xFFFFFFFF, 0xFFFFFFFF);
    run_task();
    snax_simd_disable_ext(SIMD_EXT_STREAMREDUCE);

    uint16_t* s = (uint16_t*)ss;
    uint16_t* sb = (uint16_t*)sbc;
    for (uint32_t r = 0; r < g->rows; r++) {
        uint16_t v = s[r * FP16_PER_BEAT];
        for (uint32_t c = 0; c < g->d; c++) sb[r * g->d + c] = v;
    }
    op_streamelementwise(g, x, (uint32_t)(sbc - x), out, EW_MUL);
}

// ---------------------------------------------------------------- driver

// Run `body` twice and report the SECOND span: cold pays the driver's L1I miss and the one-time
// AGU shape setup, and steady state is what a scheduler needs.
#define MEASURE(name, body)                                                       \
    do {                                                                          \
        uint32_t span = 0;                                                        \
        for (int it = 0; it < 2; it++) {                                           \
            uint32_t t0 = snrt_mcycle();                                          \
            body;                                                                 \
            span = snrt_mcycle() - t0;                                            \
        }                                                                         \
        printf("[BINGO] op=%s rows=%u cols=%u cycles=%u\n", name, g.rows, g.d, span); \
    } while (0)

int main() {
    uint32_t base = snrt_cluster_base_addrl();

    geom_t g;
    g.rows = BSW_ROWS;
    g.d = BSW_D;
    g.beats = BSW_BEATS;
    g.row_bytes = g.beats * BEAT_BYTES;
    g.rows_bytes = g.rows * g.row_bytes;
    g.scal_bytes = g.rows * BEAT_BYTES;
    g.red_str[0] = BEAT_BYTES;
    g.red_str[1] = g.row_bytes;
    g.red_bnd[0] = g.beats;
    g.red_bnd[1] = g.rows;
    g.w_rows_str[0] = BEAT_BYTES;
    g.w_rows_bnd[0] = g.rows;
    g.flat_str[0] = BEAT_BYTES;
    g.flat_bnd[0] = g.rows * g.beats;

    // L1 map. `b` chases the allocations so a size change cannot leave a stale offset behind --
    // the whole point of a sweep is that every one of these moves.
    uint8_t* b = (uint8_t*)base;
    uint8_t* x_in = b;              b += g.rows_bytes;
    uint8_t* up_in = b;             b += g.rows_bytes;   // operand B for the bare elementwise
    uint8_t* sg_a = b;              b += g.rows_bytes;   // swiglu: silu(gate) ...
    uint8_t* sg_b = b;              b += g.rows_bytes;   // ... and the `up` it multiplies
    uint8_t* t0_buf = b;            b += g.rows_bytes;
    uint8_t* t1_buf = b;            b += g.rows_bytes;
    uint8_t* t2_buf = b;            b += g.rows_bytes;
    uint8_t* bc0_buf = b;           b += g.rows_bytes;   // a per-row scalar, broadcast
    uint8_t* bc1_buf = b;           b += g.rows_bytes;
    uint8_t* out_buf = b;           b += g.rows_bytes;
    uint8_t* s0_buf = b;            b += g.scal_bytes;
    uint8_t* s1_buf = b;            b += g.scal_bytes;
    uint32_t span_bytes = (uint32_t)(b - (uint8_t*)base);

    // The input, written here rather than DMA'd in: this app needs a defined bit pattern, not a
    // dataset. Every hart could do it, but only one should.
    //
    // EVERY buffer is written, not just the inputs -- scratch included. TCDM comes up X in
    // simulation, and a pass that reads X propagates it into the datapath; the timing would not
    // change (the FP units are fixed-latency) but the waveform would be full of red herrings for
    // whoever debugs the next failure here.
    if (snax_is_simd_core()) {
        uint16_t* w = (uint16_t*)base;
        for (uint32_t i = 0; i < span_bytes / 2; i++) w[i] = FP16_HALF;
    }
    // Unconditional: the hardware barrier counts every core in the cluster, so a hart that
    // skipped it would hang the ones that did not.
    snrt_cluster_hw_barrier();

    if (snax_is_simd_core()) {
        printf("[BINGO] begin rows=%u cols=%u beats=%u\n", g.rows, g.d, g.beats);

        MEASURE("xdma_streammap", op_streammap(&g, x_in, t0_buf));
        MEASURE("xdma_streamreduce", op_streamreduce(&g, x_in, s0_buf));
        MEASURE("xdma_streamelementwise",
                op_streamelementwise(&g, x_in, (uint32_t)(up_in - x_in), t0_buf, EW_MUL));
        MEASURE("xdma_silu", op_silu(&g, x_in, t0_buf));
        MEASURE("xdma_swiglu", op_swiglu(&g, x_in, sg_a, out_buf));
        MEASURE("xdma_softmax", op_softmax(&g, x_in, s0_buf, bc0_buf, t1_buf, t2_buf, s1_buf,
                                           bc1_buf, out_buf));
        MEASURE("xdma_rmsnorm", op_rmsnorm(&g, x_in, s0_buf, bc0_buf, out_buf));

        // A task the block DROPPED as degenerate reports complete and costs almost nothing, so a
        // bad geometry would otherwise sweep as a suspiciously fast point rather than as a bug.
        if (snax_simd_bad_config()) {
            printf("[BINGO] FAIL degenerate task geometry\n");
            return 1;
        }
        printf("[BINGO] end\n");
    }
    snrt_cluster_hw_barrier();
    return 0;
}
