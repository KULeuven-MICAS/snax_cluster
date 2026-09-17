// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// A tiled matmul as a TWO-ENGINE PIPELINE: the array dispatches while the iDMA fetches the
// operands for the dispatch after next. It exists to measure the pipeline, not the matmul.
//
// WHY THIS APP EXISTS. The other matmuls in this tree -- snax-versacore-to-matmul,
// snax-gemmx-matmul, snax-versacore-to-matmul-profile -- all do
//
//     dma(A); dma(B); dma_wait_all(); dispatch; dma(C)
//
// which is one output block with nothing overlapping anything. That measures a dispatch. It
// cannot measure a PIPELINE, because there isn't one: no double buffering, so no operand
// stall, no buffer-lifetime edge, and no way for the iDMA to be the thing that sets the
// makespan. snax-model-calib-gemm gets closer -- many dispatches, and neighbour engines
// running at a controlled duty -- but its background load is SYNTHETIC padding rather than
// this kernel's own operands, which is exactly the part a pipeline model gets wrong.
//
// So this is the missing instrument. It is the smallest program in which the array can be
// starved by its own operand stream.
//
// WHAT IT DOES. `NBLOCK` output blocks, each accumulated over `NK` dispatches:
//
//   iDMA core (hart SNAX_CORE_IDMA)        GEMM core (hart SNAX_CORE_GEMM)
//     for each step t:                       for each step t:
//       wait: slot t-DEPTH retired             wait: A(t) and B(t) have landed
//       load A(t) -> a_buf[t % DEPTH]          re-point the streamer at those buffers
//       load B(t) -> b_buf[t % DEPTH]          launch, wait for the array to retire
//       (per block) wait: block accumulated    publish "t retired" -- the slot is free
//       store D(block)                         (last k step) publish "block accumulated"
//
// The handoff is COUNTERS in L1, not barriers -- a producer bumps, a consumer spins past a
// threshold -- because that is what the machine does and because a buffer-lifetime edge
// ("the slot I am about to overwrite was last read by the dispatch DEPTH back") is then the
// same kind of edge as a data one, with no separate notion of anti-dependency.
//
// WHAT IS NOT MODELLED, deliberately. Every step reloads BOTH operand tiles from the same
// DRAM source, exactly as snax-flashattn-decode reloads K and V from one buffer: the bytes
// moved and the timing are real, the values are not a true M x N x K product. This is a
// TIMING instrument. The numerical check below therefore checks what is actually claimed --
// that the array computed the dispatch it was configured for -- by verifying the FIRST
// block, whose inputs are the unmodified A and B of data.h, against that dispatch's golden.
//
// NOT STAGED. snax-flashattn-decode writes dispatch t+1's CSRs while dispatch t runs, since
// STREAMER_START latches the bank. This one does not: config, launch, wait, in that order.
// That is the program the cost model describes, and the point here is to compare a model
// against the program it claims to describe. Staging is a real optimisation and it would
// need its own model.

#include "data.h"

#include "snax-core-roles.h"
#include "snax-versacore-to-lib.h"
#include "snrt.h"

#if !SNAX_HAS_GEMM_CORE
#error "this kernel needs a matmul array"
#endif

// The pipeline's shape. Overridable from the Makefile (RISCV_CFLAGS += -DNK=8) so a sweep
// costs a rebuild rather than an edit.
#ifndef NBLOCK
#define NBLOCK 4
#endif
#ifndef NK
#define NK 4
#endif
#ifndef DEPTH
#define DEPTH 2
#endif
#define NSTEP (NBLOCK * NK)

// The store's destination. It must NOT be data.h's `D`: that is the golden the first block
// is checked against, and storing over it would compare the result with itself.
#define OUT_WORDS 32768
static int32_t gemm_out[OUT_WORDS];

// A spin that cannot hang the simulation for ever: the two cores are hand-synchronised, so
// a deadlock is possible, and an unbounded loop burns wall-clock with no diagnosis.
#define SPIN_LIMIT 200000u
#define SPIN_UNTIL(cond, timeout_flag)               \
    do {                                             \
        uint32_t spins__ = 0;                        \
        while (!(cond)) {                            \
            if (++spins__ > SPIN_LIMIT) {            \
                (timeout_flag)++;                    \
                break;                               \
            }                                        \
        }                                            \
    } while (0)

// Re-point an already-configured streamer at new buffers without touching its shape: four
// csrw against the ~60 of a full configure. always_inline so the compile-time-constant CSR
// address folds to a single `csrw <imm>` -- out of line every access pays an indirect jump.
__attribute__((always_inline)) static inline void gemm_set_bases(uint32_t a, uint32_t b,
                                                                uint32_t c, uint32_t d32) {
    csrw_ss(BASE_PTR_READER_0_LOW, a);
    csrw_ss(BASE_PTR_READER_1_LOW, b);
    csrw_ss(BASE_PTR_READER_WRITER_0_LOW, c);
    csrw_ss(BASE_PTR_READER_WRITER_1_LOW, d32);
}

// What changes between dispatches, and nothing else.
//
// The descriptor is STICKY: snax-model-calib-gemm relaunches eight times writing only the
// start registers, so the bounds and strides survive a launch and must NOT be rewritten.
// (An earlier version re-armed them per dispatch the way snax-flashattn-decode does -- it
// switches between two shapes, so it has to -- and the array appeared to retire exactly one
// task. That symptom turned out to be a stale simulator rather than the re-arm; see
// gemm_wait. Keeping the descriptor sticky is still right, it is just not load-bearing.)
//
// `accumulate` picks what C contributes. A disabled C channel still pops its address and
// presents a beat -- it just presents ZERO and issues no TCDM request -- which is precisely
// the seed a fresh output block wants, so the first k step of a block masks C off instead of
// zeroing the accumulator and the steps after it read their own previous output back.
// take_in_new_c stays 1 either way: at 0 the array stops draining the C reader, whose beats
// keep coming, and its depth-1 FIFO fills and never reports empty.
__attribute__((always_inline)) static inline void gemm_accumulate(uint32_t on) {
    csrw_ss(ENABLED_CHANNEL_READER_WRITER_0, on ? (uint32_t)channel_en_C[0] : 0u);
}

__attribute__((always_inline)) static inline void gemm_launch(void) {
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(GEMMX_START, 1);
}

__attribute__((always_inline)) static inline void gemm_ack(void) {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
}

// Wait for ONE named dispatch by id. The streamer's finished counter says its data movers
// retired, which is not the same as the array being done -- a writer's address generator
// finishes issuing before the array finishes producing into it. The array's task counter is
// the signal that the dispatch is over. Bounded, so a stuck engine reports itself.
__attribute__((always_inline)) static inline uint32_t gemm_wait(uint32_t task_id) {
    uint32_t spins = 0;
    while ((int32_t)(csrr_ss(GEMMX_FINISHED_TASK) - task_id) < 0) {
        if (++spins > 100000u) return 1;
    }
    csrw_ss(GEMMX_START, 0);
    // A belt-and-braces drain, NOT a required one. It was added when this app stalled at
    // FINISHED_TASK = 1 with BUSY reading 1 just after, on the theory that the counter rises
    // before the array is idle and ReqRspManager then throttles the next start pulse away.
    //
    // MEASURED, that theory is wrong on this RTL: snax-model-calib-gemm instrumented to count
    // BUSY reads after every counter rise reports a worst case of ZERO over 24 dispatches --
    // when the counter rises the array is already idle, and a tight relaunch is safe.
    //
    // The stall that motivated it was a STALE SIMULATOR. GEMMX_FINISHED_TASK is a read-only
    // CSR the array only gained in `ab1f76ce`; a `work-vsim` built before that returns a stuck
    // value from outside the read-only window, so the counter never tracked anything. BUSY was
    // 1 because the array was still running dispatch ONE. The loop below costs nothing when
    // the array is idle, so it stays -- but the check at startup is what actually catches this.
    while (csrr_ss(GEMMX_BUSY)) {
        if (++spins > 100000u) return 1;
    }
    return 0;
}

// 32-bit throughout: this target links no 64-bit division runtime (__udivdi3), and the
// product overflows only past ~43 M cycles, which is where the other form takes over.
// Sampled either side of the first dispatch, to prove the array's counter is real.
static uint32_t counter_before_first, counter_after_first;

static inline uint32_t pct(uint32_t part, uint32_t whole) {
    if (!whole) return 0u;
    return (whole >= 1000000u) ? (part / (whole / 100u)) : ((part * 100u) / whole);
}

int main() {
    uint8_t *l1 = (uint8_t *)snrt_l1_next();

    // ---- L1 layout ------------------------------------------------------------------
    // Two operand slots per stream (DEPTH), one accumulator. C and D alias: a block
    // accumulates in place, which is what makes the k loop free of a separate reduction.
    const uint32_t alen = (uint32_t)a_data_length, blen = (uint32_t)b_data_length;
    const uint32_t dlen = (uint32_t)d_data_length;
    uint32_t top = 0;
    uint32_t a_off[DEPTH], b_off[DEPTH];
    for (int i = 0; i < DEPTH; i++) { a_off[i] = top; top += alen; }
    for (int i = 0; i < DEPTH; i++) { b_off[i] = top; top += blen; }
    const uint32_t d_off = top; top += dlen;
    top = (top + 63u) & ~63u;
    volatile uint32_t *pub  = (volatile uint32_t *)(l1 + top); top += 1024;
    volatile uint32_t *sync = (volatile uint32_t *)(l1 + top); top += 64;

    // sync[0] A tiles landed      sync[2] dispatches retired -- a slot is free
    // sync[1] B tiles landed      sync[3] output blocks accumulated -- a store may go
    // sync[4] the shared time origin
    if (snrt_cluster_core_idx() == 0)
        for (int i = 0; i < 16; i++) sync[i] = 0;
    snrt_cluster_hw_barrier();

    uint32_t timeouts = 0;

    // ---- one full streamer configuration, on the core that owns the array -------------
    if (snax_is_gemm_core()) {
        int32_t Aslstride[] = {Aslstride0};
        int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2,
                              Atlbound3, Atlbound4, Atlbound5};
        int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2,
                               Atlstride3, Atlstride4, Atlstride5};
        int32_t Bslstride[] = {Bslstride0};
        int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2};
        int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2};
        int32_t Cslstride[] = {Cslstride0};
        int32_t Ctlbound[] = {Ctlbound0, Ctlbound1, Ctlbound2, Ctlbound3};
        int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2, Ctlstride3};
        int32_t D32slstride[] = {D32slstride0};
        int32_t D32tlbound[] = {D32tlbound0, D32tlbound1, D32tlbound2, D32tlbound3};
        int32_t D32tlstride[] = {D32tlstride0, D32tlstride1, D32tlstride2, D32tlstride3};

        set_versacore_streamer_csr(
            (int32_t)a_off[0], Aslstride, Atlbound, Atlstride,
            set_addr_remap_index_A, transposed_A, channel_en_A,
            (int32_t)b_off[0], Bslstride, Btlbound, Btlstride,
            set_addr_remap_index_B, transposed_B, channel_en_B,
            (int32_t)d_off, Cslstride, Ctlbound, Ctlstride,
            set_addr_remap_index_C, channel_en_C,
            (int32_t)d_off, D32slstride, D32tlbound, D32tlstride,
            set_addr_remap_index_D32, channel_en_D, array_shape,
            quantization_enable, shift_i, multiplier_i, input_zp_i, output_zp_i,
            int32tofp16_enable, int4_a_enable, int4_b_enable);

        uint32_t subtraction_setting = gen_subtraction_config(subtraction_a, subtraction_b);
        set_versacore_csr(1, K, N * M, subtraction_setting, array_shape, data_type);
    }
    snrt_cluster_hw_barrier();

    // ONE ORIGIN FOR BOTH LANES. mcycle counts the same clock on every hart, but each core
    // stamping its own after a barrier leaves them hundreds of cycles apart -- enough that
    // a cross-lane reading of the trace is an artefact of the offset rather than a fact.
    if (snax_is_gemm_core()) sync[4] = snrt_mcycle();
    snrt_cluster_hw_barrier();
    const uint32_t t_org = sync[4];


#if DIAG
    // WHICH SETUP BREAKS THE RELAUNCH? snax-model-calib-gemm configures ONCE and then issues
    // REPS = 8 back-to-back dispatches, so relaunching is known to work on this cfg. This app
    // retires exactly one task however little it writes between launches, so the cause is in
    // the one-time configuration, not in the dispatch loop.
    //
    // Probe 0 configures EXACTLY as calib does: data.h's own deltas, and C loaded into L1.
    // Probe 1 configures with this app's layout. Each then launches three times, changing
    // nothing in between. Whichever stalls at 1 names the setup that is wrong.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l1 + (uint32_t)delta_local_a, A, alen); snrt_dma_wait_all();
        snrt_dma_start_1d(l1 + (uint32_t)delta_local_b, B, blen); snrt_dma_wait_all();
        snrt_dma_start_1d(l1 + (uint32_t)delta_local_c, C, (uint32_t)c_data_length);
        snrt_dma_wait_all();
        snrt_dma_start_1d(l1 + a_off[0], A, alen); snrt_dma_wait_all();
        snrt_dma_start_1d(l1 + b_off[0], B, blen); snrt_dma_wait_all();
        sync[0] = 1;
    }
    if (snax_is_gemm_core()) {
        SPIN_UNTIL(sync[0] >= 1, timeouts);
        int32_t Asl[] = {Aslstride0};
        int32_t Atb[] = {Atlbound0, Atlbound1, Atlbound2, Atlbound3, Atlbound4, Atlbound5};
        int32_t Ats[] = {Atlstride0, Atlstride1, Atlstride2, Atlstride3, Atlstride4, Atlstride5};
        int32_t Bsl[] = {Bslstride0};
        int32_t Btb[] = {Btlbound0, Btlbound1, Btlbound2};
        int32_t Bts[] = {Btlstride0, Btlstride1, Btlstride2};
        int32_t Csl[] = {Cslstride0};
        int32_t Ctb[] = {Ctlbound0, Ctlbound1, Ctlbound2, Ctlbound3};
        int32_t Cts[] = {Ctlstride0, Ctlstride1, Ctlstride2, Ctlstride3};
        int32_t Dsl[] = {D32slstride0};
        int32_t Dtb[] = {D32tlbound0, D32tlbound1, D32tlbound2, D32tlbound3};
        int32_t Dts[] = {D32tlstride0, D32tlstride1, D32tlstride2, D32tlstride3};
        uint32_t seq = 0;
        for (uint32_t probe = 0; probe < 2; probe++) {
            int32_t da = probe ? (int32_t)a_off[0] : delta_local_a;
            int32_t db = probe ? (int32_t)b_off[0] : delta_local_b;
            int32_t dc = probe ? (int32_t)d_off   : delta_local_c;
            int32_t dd = probe ? (int32_t)d_off   : delta_local_d;
            set_versacore_streamer_csr(
                da, Asl, Atb, Ats, set_addr_remap_index_A, transposed_A, channel_en_A,
                db, Bsl, Btb, Bts, set_addr_remap_index_B, transposed_B, channel_en_B,
                dc, Csl, Ctb, Cts, set_addr_remap_index_C, channel_en_C,
                dd, Dsl, Dtb, Dts, set_addr_remap_index_D32, channel_en_D, array_shape,
                quantization_enable, shift_i, multiplier_i, input_zp_i, output_zp_i,
                int32tofp16_enable, int4_a_enable, int4_b_enable);
            set_versacore_csr(1, K, N * M,
                              gen_subtraction_config(subtraction_a, subtraction_b),
                              array_shape, data_type);
            for (uint32_t i = 0; i < 3; i++) {
                csrw_ss(STREAMER_START_CSR, 1);
                csrw_ss(GEMMX_START, 1);
                csrw_ss(STREAMER_START_CSR, 0);
                csrw_ss(STREAMER_START_CSR, 0);
                uint32_t tid = ++seq, sp = 0;
                while ((int32_t)(csrr_ss(GEMMX_FINISHED_TASK) - tid) < 0)
                    if (++sp > 20000u) break;
                csrw_ss(GEMMX_START, 0);
                pub[200 + probe * 8 + i * 2] = csrr_ss(GEMMX_FINISHED_TASK);
                pub[201 + probe * 8 + i * 2] = csrr_ss(GEMMX_BUSY);
                (void)csrr_ss(GEMMX_PERFORMANCE_COUNTER);
                (void)csrr_ss(GEMMX_STALL_A);
                (void)csrr_ss(GEMMX_STALL_B);
                (void)csrr_ss(GEMMX_STALL_D);
            }
        }
    }
    snrt_cluster_hw_barrier();
    if (snax_is_gemm_core()) {
        printf("=== DIAG: retired-task counter after each of 3 launches ===\n");
        printf("  probe 0 (calib layout, C loaded): %u %u %u  busy %u %u %u\n",
               pub[200], pub[202], pub[204], pub[201], pub[203], pub[205]);
        printf("  probe 1 (this app's layout):      %u %u %u  busy %u %u %u\n",
               pub[208], pub[210], pub[212], pub[209], pub[211], pub[213]);
        printf("GEMMPIPE done\n");
    }
    return 0;
#endif

    // ---- the operand lane -------------------------------------------------------------
    if (snrt_is_dm_core()) {
        uint32_t dma_busy = 0, dma_block = 0;
        for (uint32_t b = 0; b < NBLOCK; b++) {
            for (uint32_t kk = 0; kk < NK; kk++) {
                uint32_t t = b * NK + kk;
                // The slot this step overwrites was last read by the dispatch DEPTH back.
                uint32_t w0 = snrt_mcycle();
                if (t >= DEPTH) SPIN_UNTIL(sync[2] >= t - DEPTH + 1, timeouts);
                dma_block += snrt_mcycle() - w0;

                uint32_t a0 = snrt_mcycle() - t_org;
                snrt_dma_start_1d(l1 + a_off[t % DEPTH], A, alen);
                snrt_dma_wait_all();
                uint32_t a1 = snrt_mcycle() - t_org;
                sync[0] = t + 1;

                snrt_dma_start_1d(l1 + b_off[t % DEPTH], B, blen);
                snrt_dma_wait_all();
                uint32_t b1 = snrt_mcycle() - t_org;
                sync[1] = t + 1;

                dma_busy += b1 - a0;
                if (t < 16) { pub[100 + 2 * t] = a0; pub[101 + 2 * t] = b1; }
            }
            // The store sits in this lane, so it delays the next block's loads. That
            // head-of-line cost is real; the model has to show it rather than assume it.
            uint32_t w1 = snrt_mcycle();
            SPIN_UNTIL(sync[3] >= b + 1, timeouts);
            dma_block += snrt_mcycle() - w1;
            uint32_t s0 = snrt_mcycle() - t_org;
            snrt_dma_start_1d((void *)gemm_out, (void *)(l1 + d_off), dlen);
            snrt_dma_wait_all();
            uint32_t s1 = snrt_mcycle() - t_org;
            dma_busy += s1 - s0;
        }
        pub[8] = timeouts;
        pub[0] = dma_busy;
        pub[1] = dma_block;
        pub[2] = snrt_mcycle() - t_org;
    }

    // ---- the dispatch lane ------------------------------------------------------------
    if (snax_is_gemm_core()) {
        const uint32_t l1u = (uint32_t)l1;
        uint32_t gemm_busy = 0, gemm_stall = 0, gemm_cfg = 0, seq = 0;
        uint32_t array_cc = 0, stall_a = 0, stall_b = 0, stall_d = 0;
        // A real retired-task counter reads zero here and one after the first dispatch. A
        // simulator built before the array gained that CSR returns a stuck value from outside
        // its read-only window, and then the first wait passes by luck while every later one
        // spins for ever. Sample it now so the run can SAY so rather than hang.
        counter_before_first = csrr_ss(GEMMX_FINISHED_TASK);
        for (uint32_t b = 0; b < NBLOCK; b++) {
            for (uint32_t kk = 0; kk < NK; kk++) {
                uint32_t t = b * NK + kk;
                uint32_t w0 = snrt_mcycle();
                SPIN_UNTIL(sync[0] >= t + 1, timeouts);   // A(t) landed
                SPIN_UNTIL(sync[1] >= t + 1, timeouts);   // B(t) landed
                gemm_stall += snrt_mcycle() - w0;

                uint32_t c0 = snrt_mcycle();
                gemm_set_bases(l1u + a_off[t % DEPTH], l1u + b_off[t % DEPTH],
                               l1u + d_off, l1u + d_off);
                gemm_accumulate(kk != 0);                 // first k step seeds, rest add
                uint32_t g0 = snrt_mcycle();
                gemm_cfg += g0 - c0;

                uint32_t tid = ++seq;
                gemm_launch();
                gemm_ack();
                timeouts += gemm_wait(tid);
                if (tid == 1) counter_after_first = csrr_ss(GEMMX_FINISHED_TASK);
                // The per-task counters, read once per dispatch because the census wants them.
                // An earlier note here guessed that leaving them UNREAD was why the array
                // refused to retire a second task -- it is not: reading a performance counter
                // cannot gate retirement, and the array is measurably idle the moment its
                // counter rises. The real cause was a simulator without that counter at all.
                array_cc += csrr_ss(GEMMX_PERFORMANCE_COUNTER);
                stall_a += csrr_ss(GEMMX_STALL_A);
                stall_b += csrr_ss(GEMMX_STALL_B);
                stall_d += csrr_ss(GEMMX_STALL_D);
                uint32_t g1 = snrt_mcycle();
                gemm_busy += g1 - g0;
                // A slot is free the instant its dispatch retired -- publish per STEP, not
                // per block, or the loader waits on an edge the machine does not have.
                sync[2] = t + 1;
                if (t < 16) { pub[132 + 2 * t] = g0 - t_org; pub[133 + 2 * t] = g1 - t_org; }
            }
            sync[3] = b + 1;      // the accumulator holds a finished block -- store it
        }
        pub[3] = gemm_busy;
        pub[4] = gemm_stall;
        pub[5] = gemm_cfg;
        pub[6] = snrt_mcycle() - t_org;
        pub[7] = array_cc;
        pub[9] = stall_a; pub[10] = stall_b; pub[11] = stall_d;
    }

    snrt_cluster_hw_barrier();

    // ---- the report -------------------------------------------------------------------
    // The numerical claim is narrow and is stated as such: the FIRST output block is the
    // dispatch data.h generated a golden for, so it is checked. Later blocks reload the
    // same A and B and are a traffic and timing construct, not a product.
    if (snax_is_gemm_core()) {
        // Before any number measured on this machine: can the machine do what the pipeline
        // assumes? If the array has no retired-task counter, every cycle count below is
        // meaningless and the run only got here by luck.
        if (counter_after_first != counter_before_first + 1u) {
            printf("FATAL: this build has no array retired-task counter. GEMMX_FINISHED_TASK "
                   "read %lu before the first dispatch and %lu after; a real counter reads 0 "
                   "then 1. The CSR sits outside the accelerator's read-only window here and "
                   "returns a stuck value, so the first wait passes by luck and the next spins "
                   "for ever. Rebuild the simulator against this checkout's RTL.\n",
                   (unsigned long)counter_before_first, (unsigned long)counter_after_first);
            return 1;
        }
        uint32_t wall = pub[2] > pub[6] ? pub[2] : pub[6];
        uint32_t err = 0;
        if (NK == 1)
            err = check_versacore_result_D32((int8_t *)(l1 + d_off), (int8_t *)D,
                                             d_data_length, false);

        printf("=== GEMM pipeline: %d blocks x %d k-steps, %d operand slots ===\n",
               NBLOCK, NK, DEPTH);
        printf("  dispatch shape   M %d N %d K %d mesh tiles (%dx%dx%d array)\n",
               M, N, K, meshRow, tileSize, meshCol);
        printf("  bytes per step   A %d  B %d   per block   D %d\n", (int)alen, (int)blen,
               (int)dlen);
        printf("  pipeline         %5u cycles, %u per block\n", wall, wall / NBLOCK);
        printf("  GEMM core        busy %5u (%2u%%)  peer-wait %5u  config %5u\n",
               pub[3], pct(pub[3], wall), pub[4], pub[5]);
        printf("  iDMA core        moving %5u (%2u%%)  blocked on a buffer %5u  wall %5u\n",
               pub[0], pct(pub[0], wall), pub[1], pub[2]);
        printf("  array counter    %u cycles for %d passes over %d dispatches\n", pub[7],
               M * N * K * NSTEP, NSTEP);
        printf("  array stalls     A %u  B %u  D %u\n", pub[9], pub[10], pub[11]);
        printf("  TRACE gemm");
        for (uint32_t i = 0; i < 2u * (NSTEP < 16 ? NSTEP : 16); i++) printf(" %u", pub[132 + i]);
        printf("\n  TRACE idma");
        for (uint32_t i = 0; i < 2u * (NSTEP < 16 ? NSTEP : 16); i++) printf(" %u", pub[100 + i]);
        printf("\n");
        if (NK == 1) printf("  first block %s (err %u)\n", err ? "FAIL" : "PASS", err);
        else         printf("  first block not checked (NK > 1 accumulates reloaded operands)\n");
        if (timeouts || pub[8])
            printf("  WARNING: spin timeouts gemm %u idma %u -- the handoff deadlocked\n",
                   timeouts, pub[8]);
        printf("GEMMPIPE done\n");
    }
    return 0;
}
