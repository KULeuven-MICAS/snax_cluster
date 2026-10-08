// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// A GEMM of a few rows -- an MoE expert's routed tokens -- in VersaCore's two GEMM shapes:
//
//   FR_SHAPE_GEMM  (16, 4, 16)   M pads to a multiple of 16
//   FR_SHAPE_FEW   (8, 8, 16)    M pads to a multiple of 8, at the same 1024 MACs a pass
//
//   D (M x N, INT32) = X (M x K, INT8) . W (K x N, INT8, or INT4 through B's converter)
//
// for M = 8, 16, 24, 32 and N = 16, FR_N at K = FR_K. Every task's D is checked word for word
// against the golden (data/datagen.py). Then the array's efficiency per task:
//
//   useful MACs / (1024 x the array's busy cycles)     M x N x K, the padded rows not counted
//
// alone, and again with an L3 -> L1 iDMA stream running for the whole task (TCDM contention).
// Shape 3 runs twice: with A as the datagen's plain A-layout, and with A interleaved (fr_a3i) so
// that A, like shape 3's INT8 B, advances 128 B a pass -- two streams at one stride stay clear
// of each other's banks, at different strides they collide every few passes.
// One more task per shape takes C: D accumulates onto the previous task's D in place, which
// checks the C read path at the shape's C/D beat count.
//
// THE PROGRAMMING (fr_arm). Output stationary, k inner, then n, then m:
//
//   A  Mu x Ku bytes a pass, 64 in either shape, channels 0..7 contiguous; interleaved, the
//      first half of K at 128 B steps, then the second half 64 B on
//   B  Ku x Nu weights a pass: INT8 64 B (shape 0, channels 0..7) or 128 B (shape 3, 0..15);
//      INT4 32 B (0..3) or 64 B (0..7) of nibbles, widened by B's converter. Channel i reads
//      8 i: S_STRIDE_1 = 64 joins the two 8-channel groups into one 128-byte run.
//   C/D  one Mu x Nu INT32 block = Mu*Nu*32/1024 beats (8 or 4) of 128 B; channel i at 8 i
//      (spatial strides 8 and 32), so a block is contiguous and blocks follow n, then m.
//
// Hand-offs between the iDMA hart and the GEMM hart are barriers around whole loads (the
// loader waits for its own transfers first) and, for the contention runs, two monotonic
// counters in L1.

#include "data.h"

#include "snax-core-roles.h"
#include "snax-versacore-to-lib.h"
#include "snrt.h"

#if !SNAX_HAS_GEMM_CORE
#error "this kernel needs a matmul array"
#endif
#ifndef READER_EXTENSION_1_CSR_BASE
#error "INT4 weights need B's HasIntlowToInthighConverter (cfg/snax_split_cluster.hjson)"
#endif

#define FR_NM 4u                                  // M = 8, 16, 24, 32
#define FR_A_BYTES (FR_M_MAX * FR_K)              // one shape's A operand
#define FR_B_BYTES (FR_K * FR_N)                  // the largest B operand, INT8
#define FR_D_WORDS (FR_M_MAX * FR_N)              // one D tile, INT32
#define FR_G16_WORDS (FR_M_MAX * FR_NU)           // a golden of one column block
#define FR_BG_BYTES (64u * 1024u)                 // the background stream's L1 landing zone

// One run's record, for the report.
typedef struct {
    uint32_t var, w4, m, n, bg;  // shape 0 / 3 / 3 with A interleaved, INT4?, rows, columns, load
    uint32_t passes, arr, sa, sb, sd, wall, err;
} fr_rec_t;

#define FR_MAX_RECS 80u

// Arm one task: rows mt row blocks, nt column blocks, D at d. ail: A interleaved (shape 3).
// c_on: C reads D's own block.
__attribute__((always_inline)) static inline void fr_arm(uint32_t few, uint32_t ail, uint32_t w4,
                                                         uint32_t mt, uint32_t nt, uint32_t a,
                                                         uint32_t b, uint32_t d, uint32_t c_on) {
    const uint32_t ku = few ? 8u : 4u, mu = few ? 8u : 16u;
    const uint32_t kt = FR_K / ku;              // passes per output block
    const uint32_t ablk = mu * ku;              // A bytes a pass
    const uint32_t bblk = (ku * FR_NU) >> w4;   // B bytes a pass
    const uint32_t dblk = mu * FR_NU * 4u;      // one INT32 output block
    const uint32_t beats = dblk / 128u;         // its C/D beats
    // A -- k inner (interleaved: two loops over the halves), n broadcast, then m.
    csrw_ss(BASE_PTR_READER_0_LOW, a);
    csrw_ss(ENABLED_CHANNEL_READER_0, 0xFFu);
    csrw_ss(S_STRIDE_READER_0_0, 8);
    csrw_ss(T_BOUND_READER_0_0, ail ? kt / 2u : kt);
    csrw_ss(T_STRIDE_READER_0_0, ail ? 2u * ablk : ablk);
    csrw_ss(T_BOUND_READER_0_1, ail ? 2u : 1u);
    csrw_ss(T_STRIDE_READER_0_1, ablk);
    csrw_ss(T_BOUND_READER_0_2, nt);
    csrw_ss(T_STRIDE_READER_0_2, 0);
    csrw_ss(T_BOUND_READER_0_3, mt);
    csrw_ss(T_STRIDE_READER_0_3, kt * ablk);
    csrw_ss(T_BOUND_READER_0_4, 1);
    csrw_ss(T_STRIDE_READER_0_4, 0);
    csrw_ss(T_BOUND_READER_0_5, 1);
    csrw_ss(T_STRIDE_READER_0_5, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0);
    // B -- k inner, n, then m broadcast; one bblk-byte run a pass.
    csrw_ss(BASE_PTR_READER_1_LOW, b);
    csrw_ss(ENABLED_CHANNEL_READER_1, (1u << (bblk / 8u)) - 1u);
    csrw_ss(S_STRIDE_READER_1_0, 8);
    csrw_ss(S_STRIDE_READER_1_1, 64);
    csrw_ss(T_BOUND_READER_1_0, kt);
    csrw_ss(T_STRIDE_READER_1_0, bblk);
    csrw_ss(T_BOUND_READER_1_1, nt);
    csrw_ss(T_STRIDE_READER_1_1, kt * bblk);
    csrw_ss(T_BOUND_READER_1_2, mt);
    csrw_ss(T_STRIDE_READER_1_2, 0);
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0);
    csrw_ss(READER_EXTENSION_1_CSR_BASE, w4);  // INT4 -> INT8 converter
    // C -- D's own blocks, or masked: a disabled channel presents zero, a fresh accumulator.
    csrw_ss(BASE_PTR_READER_WRITER_0_LOW, d);
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_0, c_on ? 0xFFFFu : 0u);
    csrw_ss(S_STRIDE_READER_WRITER_0_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_0_1, 32);
    csrw_ss(T_BOUND_READER_WRITER_0_0, beats);
    csrw_ss(T_STRIDE_READER_WRITER_0_0, 128);
    csrw_ss(T_BOUND_READER_WRITER_0_1, nt);
    csrw_ss(T_STRIDE_READER_WRITER_0_1, dblk);
    csrw_ss(T_BOUND_READER_WRITER_0_2, mt);
    csrw_ss(T_STRIDE_READER_WRITER_0_2, nt * dblk);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_0, 0);
    csrw_ss(READER_WRITER_EXTENSION_0_CSR_BASE, 0);  // column scale off
    // D -- INT32, the same walk.
    csrw_ss(BASE_PTR_READER_WRITER_1_LOW, d);
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_1, 0xFFFFu);
    csrw_ss(S_STRIDE_READER_WRITER_1_0, 8);
    csrw_ss(S_STRIDE_READER_WRITER_1_1, 32);
    csrw_ss(T_BOUND_READER_WRITER_1_0, beats);
    csrw_ss(T_STRIDE_READER_WRITER_1_0, 128);
    csrw_ss(T_BOUND_READER_WRITER_1_1, nt);
    csrw_ss(T_STRIDE_READER_WRITER_1_1, dblk);
    csrw_ss(T_BOUND_READER_WRITER_1_2, mt);
    csrw_ss(T_STRIDE_READER_WRITER_1_2, nt * dblk);
    csrw_ss(ADDR_REMAP_INDEX_READER_WRITER_1, 0);
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 0, 0);  // INT32 -> FP16 off
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, 0);
    // The array.
    csrw_ss(OVERWRITE_ACCUM, 1);
    csrw_ss(ACCUM_BOUND, kt);
    csrw_ss(OUTPUT_BOUND, nt * mt);
    csrw_ss(SUBTRACTIONS, 0);
    csrw_ss(ARRAY_SHAPE_CFG, few ? FR_SHAPE_FEW : FR_SHAPE_GEMM);
    csrw_ss(DATA_TYPE_CFG, 0);
}

// Launch the armed task as array task `tid` and wait until its D is in L1: the array retires
// it, then the D writer drains. Returns 1 on a timeout.
static uint32_t fr_run(uint32_t tid) {
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(GEMMX_START, 1);
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    uint32_t seen;
    if (snax_versacore_wait_array_task(tid, &seen)) {
        printf("TIMEOUT: array task %u, the counter reads %u\n", tid, seen);
        return 1;
    }
    csrw_ss(GEMMX_START, 0);
    uint32_t spins = 0;
    while (csrr_ss(GEMMX_BUSY) || csrr_ss(STREAMER_BUSY_CSR))
        if (++spins > SNAX_VERSACORE_SPIN_LIMIT) return 1;
    return 0;
}

// Words of D that differ from the golden prefix, times `scale` (2 after a C pass).
static uint32_t fr_check(const int32_t *d, const int32_t *g, uint32_t words, int32_t scale) {
    const volatile int32_t *v = (const volatile int32_t *)d;
    uint32_t err = 0;
    for (uint32_t i = 0; i < words; i++) {
        if (v[i] != scale * g[i]) {
            if (err < 4)
                printf("  word %u: got %d expected %d\n", i, v[i], scale * g[i]);
            err++;
        }
    }
    return err;
}

int main() {
    uint8_t *l1 = (uint8_t *)snrt_l1_next();
    // ---- L1 layout ------------------------------------------------------------------------
    // [shape 0, shape 3, shape 3 interleaved]. The interleaved A starts 16 banks (128 B) off
    // B's alignment: at one stride, that keeps A's 8 banks out of B's 16 from the first pass.
    uint8_t *a[3] = {l1, l1 + FR_A_BYTES, l1 + 2u * FR_A_BYTES + 128u};
    uint8_t *b = l1 + 3u * FR_A_BYTES + 256u;
    int32_t *d = (int32_t *)(b + FR_B_BYTES);
    int32_t *g = d + FR_D_WORDS;                       // [s][w]: 4 x FR_D_WORDS
    int32_t *g16 = g + 4u * FR_D_WORDS;                // [s][w]: 4 x FR_G16_WORDS
    uint8_t *bg = (uint8_t *)(g16 + 4u * FR_G16_WORDS);
    volatile uint32_t *sync = (volatile uint32_t *)(bg + FR_BG_BYTES);
    fr_rec_t *rec = (fr_rec_t *)(sync + 16);
    const uint32_t end = (uint32_t)(rec + FR_MAX_RECS);
    // sync[0] background stream started (phase id)   sync[1] stop it (phase id)
    // sync[2] the stream's transfers, the last phase

    // The goldens, [s][w], s: 0 = shape 0, 1 = shape 3; w: 0 = INT8, 1 = INT4.
    const int32_t *gsrc[4] = {fr_g0_8, fr_g0_4, fr_g3_8, fr_g3_4};
    const int32_t *g16src[4] = {fr_g0_8_n16, fr_g0_4_n16, fr_g3_8_n16, fr_g3_4_n16};
    const int8_t *bsrc[4] = {fr_b0_8, fr_b0_4, fr_b3_8, fr_b3_4};

    if (snrt_is_dm_core()) {
        for (uint32_t i = 0; i < 4; i++) sync[i] = 0;
        snrt_dma_start_1d(a[0], fr_a0, FR_A_BYTES);
        snrt_dma_start_1d(a[1], fr_a3, FR_A_BYTES);
        snrt_dma_start_1d(a[2], fr_a3i, FR_A_BYTES);
        for (uint32_t i = 0; i < 4; i++) {
            snrt_dma_start_1d(g + i * FR_D_WORDS, gsrc[i], FR_D_WORDS * 4u);
            snrt_dma_start_1d(g16 + i * FR_G16_WORDS, g16src[i], FR_G16_WORDS * 4u);
        }
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t nrec = 0, timeouts = 0, tid = 0, phase = 0;
    if (snax_is_gemm_core()) {
        const snrt_allocator_t *al = snrt_l1_allocator();
        if (end > al->base + al->size) {
            printf("FATAL: the L1 layout ends at %x, past the free L1's end %x\n", end,
                   (uint32_t)(al->base + al->size));
            return 1;
        }
        tid = csrr_ss(GEMMX_FINISHED_TASK);
    }

    for (uint32_t w4 = 0; w4 < 2; w4++) {
        for (uint32_t var = 0; var < 3; var++) {
            const uint32_t few = var > 0u, ail = var == 2u;
            const uint32_t sw = 2u * few + w4;
            const uint32_t bbytes = FR_B_BYTES >> w4;
            const uint32_t mu = few ? 8u : 16u, kt = FR_K / (few ? 8u : 4u);
            // ---- B for this shape and width ------------------------------------------------
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(b, bsrc[sw], bbytes);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();

            // ---- alone, then with the background stream -----------------------------------
            for (uint32_t bgon = 0; bgon < 2; bgon++) {
                phase++;
                if (bgon && snrt_is_dm_core()) {
                    // Stream DRAM -> L1 until the GEMM hart is done: B's own blob, in 64 KiB
                    // transfers into a landing zone nothing reads.
                    uint32_t n = 0;
                    snrt_dma_start_1d(bg, bsrc[sw], FR_BG_BYTES);
                    sync[0] = phase;
                    while (sync[1] != phase) {
                        snrt_dma_wait_all();
                        snrt_dma_start_1d(bg, bsrc[sw], FR_BG_BYTES);
                        n++;
                    }
                    snrt_dma_wait_all();
                    sync[2] = n + 1u;
                }
                if (snax_is_gemm_core()) {
                    if (bgon)
                        while (sync[0] != phase) {
                        }
                    for (uint32_t nn = 0; nn < 2; nn++) {
                        const uint32_t n = nn ? FR_N : FR_NU, nt = n / FR_NU;
                        // Under the stream, only the full-width tasks: they are the measurement.
                        if (bgon && !nn) continue;
                        for (uint32_t mi = 0; mi < FR_NM; mi++) {
                            const uint32_t m = 8u * (mi + 1u);
                            const uint32_t mt = (m + mu - 1u) / mu;
                            fr_rec_t *r = &rec[nrec++];
                            r->var = var; r->w4 = w4; r->m = m; r->n = n; r->bg = bgon;
                            r->passes = mt * nt * kt;
                            fr_arm(few, ail, w4, mt, nt, (uint32_t)a[var], (uint32_t)b,
                                   (uint32_t)d, 0);
                            uint32_t t0 = snrt_mcycle();
                            timeouts += fr_run(++tid);
                            r->wall = snrt_mcycle() - t0;
                            r->arr = csrr_ss(GEMMX_PERFORMANCE_COUNTER);
                            r->sa = csrr_ss(GEMMX_STALL_A);
                            r->sb = csrr_ss(GEMMX_STALL_B);
                            r->sd = csrr_ss(GEMMX_STALL_D);
                            // Under the stream, check the full tile only: the check costs
                            // simulated time, and the runs alone already checked the rest.
                            if (!bgon || m == FR_M_MAX) {
                                const int32_t *gg =
                                    nn ? g + sw * FR_D_WORDS : g16 + sw * FR_G16_WORDS;
                                r->err = fr_check(d, gg, mt * mu * n, 1);
                            } else {
                                r->err = 0;
                            }
                        }
                    }
                    // C: accumulate a second X . W onto the full tile in place (INT8, alone).
                    if (!bgon && !w4) {
                        const uint32_t mt = FR_M_MAX / mu, nt = FR_N / FR_NU;
                        fr_rec_t *r = &rec[nrec++];
                        r->var = var; r->w4 = w4; r->m = FR_M_MAX; r->n = FR_N; r->bg = 2;
                        r->passes = mt * nt * kt;
                        fr_arm(few, ail, w4, mt, nt, (uint32_t)a[var], (uint32_t)b, (uint32_t)d,
                               1);
                        uint32_t t0 = snrt_mcycle();
                        timeouts += fr_run(++tid);
                        r->wall = snrt_mcycle() - t0;
                        r->arr = csrr_ss(GEMMX_PERFORMANCE_COUNTER);
                        r->sa = csrr_ss(GEMMX_STALL_A);
                        r->sb = csrr_ss(GEMMX_STALL_B);
                        r->sd = csrr_ss(GEMMX_STALL_D);
                        r->err = fr_check(d, g + sw * FR_D_WORDS, mt * mu * FR_N, 2);
                    }
                    if (bgon) sync[1] = phase;
                }
                snrt_cluster_hw_barrier();
            }
        }
    }

    // ---- the report -----------------------------------------------------------------------
    if (snax_is_gemm_core()) {
        uint32_t errs = 0;
        printf("=== VersaCore few-row GEMM: K %u, shapes (16,4,16) = %u and (8,8,16) = %u ===\n",
               FR_K, FR_SHAPE_GEMM, FR_SHAPE_FEW);
        printf("shape        W    M    N  load  passes  array-cc  pass/cc  MAC-eff  stallA  "
               "stallB  stallD    wall  check\n");
        for (uint32_t i = 0; i < nrec; i++) {
            const fr_rec_t *r = &rec[i];
            // useful MACs / 1024 = M N K / 1024 passes' worth; per mille of the busy cycles
            const uint32_t useful = r->m * r->n * (FR_K / 1024u);
            const uint32_t eff = r->arr ? useful * 1000u / r->arr : 0u;
            const uint32_t rate = r->arr ? r->passes * 1000u / r->arr : 0u;
            errs += r->err;
            printf("%s  %s %4u %4u  %s  %6u  %8u   %3u.%u%%   %3u.%u%%  %6u  %6u  %6u  %6u  %s\n",
                   r->var == 2 ? "(8,8,16)i " : (r->var ? "(8,8,16)  " : "(16,4,16) "),
                   r->w4 ? "W4" : "W8", r->m, r->n,
                   r->bg == 2 ? "+C  " : (r->bg ? "iDMA" : "-   "), r->passes, r->arr,
                   rate / 10u, rate % 10u, eff / 10u, eff % 10u, r->sa, r->sb, r->sd, r->wall,
                   r->err ? "FAIL" : "PASS");
        }
        printf("background stream: %u x %u B in its last phase\n", sync[2], FR_BG_BYTES);
        if (timeouts) printf("WARNING: %u array tasks timed out\n", timeouts);
        printf("%s: %u words wrong over %u tasks\n", errs || timeouts ? "FAIL" : "PASS", errs,
               nrec);
        return errs || timeouts ? 1 : 0;
    }
    return 0;
}
