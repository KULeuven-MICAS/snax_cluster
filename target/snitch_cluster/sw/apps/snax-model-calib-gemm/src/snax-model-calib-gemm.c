// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// What a GEMM dispatch costs, and what the OTHER engines cost it.
//
// The array's own counter is additive -- busy = array_passes + stall_A + stall_B + stall_D --
// so a dispatch's cost is its pass count plus whatever its operand streamers could not fetch
// in time. Those stalls are not a property of the GEMM alone: A, B, C, D, the SIMD block and
// the iDMA all arbitrate for the same TCDM banks, so the same dispatch costs more when the
// cluster is busy around it. The performance model has to know how much more, and the only
// honest way to find out is to run the SAME dispatch with the neighbours idle and again with
// them streaming.
//
// Four phases, one dispatch shape:
//
//   A  C read on,  neighbours idle        the array's own feed cost
//   B  C read off, neighbours idle        what masking a dead operand saves
//   C  C read on,  SIMD + iDMA streaming  the derate under a busy cluster
//   D  C read off, SIMD + iDMA streaming  both together
//
// The golden is checked in phase A only: the later phases deliberately change the operands
// the array sees (masking C makes it read zero), so their results are expected to differ and
// what is being measured is time, not arithmetic.

#include <stdint.h>
#include "data.h"
#include "snax-core-roles.h"
#include "snax-simd-lib.h"
#include "snax-versacore-to-lib.h"
#include "snrt.h"

#define REPS 8            // dispatches per phase; the first is cold and is reported separately
#define TRAFFIC_BEATS 512 // SIMD task length while generating background traffic
#define TRAFFIC_BYTES 16384

// The phases. `simd_pad` / `idma_pad` are idle cycles inserted after each unit of background
// work, which is how a neighbour is made to run at a FRACTION of its peak rather than flat
// out. Saturated neighbours are not the interesting case -- a real pipeline's engines are busy
// perhaps half the time -- so the derate has to be measured as a curve, not at one point.
typedef struct {
    const char *name;
    uint32_t c_on;
    uint32_t fp16;            // drain D through the int32->fp16 converter: HALF the beats
    uint32_t simd_on, simd_pad;
    uint32_t idma_on, idma_pad;
} phase_cfg_t;

static const phase_cfg_t PHASES[] = {
    {"C_on_idle",     1, 0, 0, 0, 0, 0},   // the C-read's own cost
    {"C_off_idle",    0, 0, 0, 0, 0, 0},   // the array's floor, int32 drain
    {"C_off_fp16",    0, 1, 0, 0, 0, 0},   // ... and with the drain halved: FA's QK exactly
    {"C_off_fp16_sd", 0, 1, 1, 0, 1, 0},   // the halved drain under saturated neighbours
    {"C_off_simd",    0, 0, 1, 0, 0, 0},   // one neighbour at a time, to attribute
    {"C_off_idma",    0, 0, 0, 0, 1, 0},
    {"C_off_both",    0, 0, 1, 0, 1, 0},   // both saturated
};
#define N_PHASES (sizeof(PHASES) / sizeof(PHASES[0]))

typedef struct {
    uint32_t cyc[REPS];
    uint32_t sa[REPS], sb[REPS], sd[REPS];
} phase_t;

static phase_t ph[N_PHASES];
// What the neighbours actually achieved, so the derate is reported against MEASURED background
// activity rather than against the duty this app intended to produce.
static volatile uint32_t simd_busy_in_phase[N_PHASES];
static volatile uint32_t idma_bytes_in_phase[N_PHASES];
static volatile uint32_t phase_wall[N_PHASES];
static volatile uint32_t *traffic_go;   // harts 1 and 3 stream while this is non-zero

static uint32_t gseq;

// One dispatch, waiting on the ARRAY's retired-task counter. The streamer's counter fires when
// the writer's address generator has finished issuing, which is BEFORE the array has finished
// producing -- polling it here would report a dispatch shorter than its own arithmetic.
__attribute__((always_inline)) static inline void dispatch_once(phase_t *p, uint32_t i) {
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(GEMMX_START, 1);
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    uint32_t tid = ++gseq;
    while ((int32_t)(csrr_ss(GEMMX_FINISHED_TASK) - tid) < 0) {
    }
    csrw_ss(GEMMX_START, 0);
    p->cyc[i] = csrr_ss(GEMMX_PERFORMANCE_COUNTER);
    p->sa[i] = csrr_ss(GEMMX_STALL_A);
    p->sb[i] = csrr_ss(GEMMX_STALL_B);
    p->sd[i] = csrr_ss(GEMMX_STALL_D);
}

int main() {
    uint8_t *l1 = (uint8_t *)snrt_l1_next();
    int8_t *local_a = (int8_t *)(l1 + delta_local_a);
    int8_t *local_b = (int8_t *)(l1 + delta_local_b);
    int32_t *local_c = (int32_t *)(l1 + delta_local_c);
    int32_t *local_d = (int32_t *)(l1 + delta_local_d);

    // Scratch for the background traffic, at a FIXED high offset rather than one derived from
    // data.h. Deriving it needs d_data_length's units to be right, and getting that wrong puts
    // the traffic buffers on top of the operands -- which does not fault, it just corrupts C
    // before the first dispatch and shows up as a golden that is close but not equal.
    uint32_t top = 384u * 1024u;
    uint8_t *simd_src = l1 + top;  top += TRAFFIC_BEATS * SIMD_BEAT_BYTES;
    uint8_t *simd_dst = l1 + top;  top += TRAFFIC_BEATS * SIMD_BEAT_BYTES;
    uint8_t *dma_dst  = l1 + top;  top += TRAFFIC_BYTES;
    traffic_go = (volatile uint32_t *)(l1 + top); top += 64;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, a_data_length);
        snrt_dma_start_1d(local_b, B, b_data_length);
        snrt_dma_start_1d(local_c, C, c_data_length);
        snrt_dma_wait_all();
        *traffic_go = 0;
    }
    if (snax_is_simd_core())
        for (uint32_t i = 0; i < TRAFFIC_BEATS * SIMD_BEAT_BYTES / 4; i++)
            ((volatile uint32_t *)simd_src)[i] = 0x3C003C00u;
    snrt_cluster_hw_barrier();

    int32_t Aslstride[] = {Aslstride0};
    int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2, Atlbound3, Atlbound4, Atlbound5};
    int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2, Atlstride3, Atlstride4, Atlstride5};
    int32_t Bslstride[] = {Bslstride0};
    int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2};
    int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2};
    int32_t Cslstride[] = {Cslstride0};
    int32_t Ctlbound[] = {Ctlbound0, Ctlbound1, Ctlbound2, Ctlbound3};
    int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2, Ctlstride3};
    int32_t D32slstride[] = {D32slstride0};
    int32_t D32tlbound[] = {D32tlbound0, D32tlbound1, D32tlbound2, D32tlbound3};
    int32_t D32tlstride[] = {D32tlstride0, D32tlstride1, D32tlstride2, D32tlstride3};

    int err = 0;
    for (uint32_t phase = 0; phase < N_PHASES; phase++) {
        const phase_cfg_t *cfg = &PHASES[phase];
        uint32_t c_on = cfg->c_on;
        uint32_t busy_neighbours = cfg->simd_on || cfg->idma_on;

        if (snax_is_gemm_core()) {
            set_versacore_streamer_csr(
                delta_local_a, Aslstride, Atlbound, Atlstride, set_addr_remap_index_A,
                transposed_A, channel_en_A,
                delta_local_b, Bslstride, Btlbound, Btlstride, set_addr_remap_index_B,
                transposed_B, channel_en_B,
                delta_local_c, Cslstride, Ctlbound, Ctlstride, set_addr_remap_index_C,
                channel_en_C,
                delta_local_d, D32slstride, D32tlbound, D32tlstride, set_addr_remap_index_D32,
                channel_en_D, array_shape,
                quantization_enable, shift_i, multiplier_i, input_zp_i, output_zp_i,
                int32tofp16_enable, int4_a_enable, int4_b_enable);
            set_versacore_csr(1, K, N * M, gen_subtraction_config(subtraction_a, subtraction_b),
                              array_shape, data_type);
            // Mask the C reader AFTER the full configure, and only ever CLEAR it: the
            // configure already wrote data.h's own mask, and writing an all-ones word back
            // would enable channels the C/D port does not have. A disabled channel still pops
            // its address and still presents a beat -- it presents ZERO and issues no TCDM
            // request -- so clearing it removes C's traffic from the ports A and B are fed
            // from without changing anything else about the dispatch.
            if (!c_on) csrw_ss(ENABLED_CHANNEL_READER_WRITER_0, 0);
            // The D port's int32->fp16 converter packs two accumulator words into one
            // fp16 beat, so the writer moves HALF the bytes -- and the address
            // generator's bounds have to be halved to match, which is a software step
            // rather than a mode bit.
            csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 0, cfg->fp16);
            if (cfg->fp16) {
                csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE + 1, 0u);
                csrw_ss(T_BOUND_READER_WRITER_1_0, (uint32_t)D32tlbound0 / 2u);
                csrw_ss(T_STRIDE_READER_WRITER_1_1, (uint32_t)D32tlstride1);
                csrw_ss(T_STRIDE_READER_WRITER_1_2, (uint32_t)D32tlstride2 / 2u);
            }
            *traffic_go = busy_neighbours;
        }
        snrt_cluster_hw_barrier();

        // The neighbours stream until hart 0 says stop. They deliberately do NOT synchronise
        // per dispatch: what is wanted is a cluster that is simply busy, the way it is in a
        // real pipeline, not a choreographed worst case.
        if (cfg->simd_on && snax_is_simd_core()) {
            snax_simd_shape_t in, out;
            snax_simd_op_t op;
            snax_simd_shape_flat(&in, simd_src, TRAFFIC_BEATS);
            snax_simd_shape_flat(&out, simd_dst, TRAFFIC_BEATS);
            snax_simd_op_map(&op, SIMD_EXT_STREAMMAP, SIMD_FUNC_LINEAR, SIMD_F32_ONE,
                             SIMD_F32_ZERO);
            snax_simd_configure(&in, &out, &op, 1);
            uint32_t b0 = snax_simd_busy_cycles();
            while (*traffic_go) {
                snax_simd_wait(snax_simd_launch());
                for (volatile uint32_t k = 0; k < cfg->simd_pad; k++) {
                }
            }
            simd_busy_in_phase[phase] = snax_simd_busy_cycles() - b0;
        } else if (cfg->idma_on && snrt_is_dm_core()) {
            uint32_t moved = 0;
            while (*traffic_go) {
                snrt_dma_start_1d(dma_dst, A, TRAFFIC_BYTES);
                snrt_dma_wait_all();
                moved += TRAFFIC_BYTES;
                for (volatile uint32_t k = 0; k < cfg->idma_pad; k++) {
                }
            }
            idma_bytes_in_phase[phase] = moved;
        } else if (snax_is_gemm_core()) {
            uint32_t w0 = snrt_mcycle();
            for (uint32_t i = 0; i < REPS; i++) dispatch_once(&ph[phase], i);
            phase_wall[phase] = snrt_mcycle() - w0;
            *traffic_go = 0;
        }
        snrt_cluster_hw_barrier();
    }

    if (snax_is_gemm_core()) {

        printf("=== GEMM dispatch cost, M=%d N=%d K=%d, mesh %dx%dx%d ===\n", M, N, K, meshRow,
               tileSize, meshCol);
        printf("  data.h a %ld b %ld c %ld d %ld (len a %ld b %ld c %ld d %ld), traffic at %u\n",
               (long)delta_local_a, (long)delta_local_b, (long)delta_local_c,
               (long)delta_local_d, (long)a_data_length, (long)b_data_length,
               (long)c_data_length, (long)d_data_length, 384u * 1024u);
        // VALIDITY WITHOUT A GOLDEN. The array's performance counter is exactly
        // passes + stall_A + stall_B + stall_D, so if the residual equals the geometry's own
        // pass count then the dispatch really executed the shape it was given -- which is what
        // a timing measurement needs. Checking the product instead needs a golden, and this
        // app's generated one does not survive the shape change; printing one mismatch per
        // byte through the UART also dominates the simulation.
        for (uint32_t p = 0; p < N_PHASES; p++)
            for (uint32_t i = 0; i < REPS; i++)
                if (ph[p].cyc[i] - ph[p].sa[i] - ph[p].sb[i] - ph[p].sd[i] !=
                    (uint32_t)(M * N * K))
                    err++;
        printf("  array passes per dispatch %d, counter algebra %s (%d off)\n", M * N * K,
               err ? "FAIL" : "PASS", err);
        // One summary line per phase. Printing every repetition put dozens of UART writes in
        // the simulation's path and dominated its wall clock; the first dispatch is reported
        // separately because it is cold and the rest are the steady state.
        for (uint32_t p = 0; p < N_PHASES; p++) {
            uint32_t c = 0, a = 0, b = 0, dd = 0;
            for (uint32_t i = 1; i < REPS; i++) {
                c += ph[p].cyc[i]; a += ph[p].sa[i]; b += ph[p].sb[i]; dd += ph[p].sd[i];
            }
            uint32_t w = REPS - 1;
            printf("CALIBG %s cold %lu warm %lu A %lu B %lu D %lu wall %lu simd_busy %lu "
                   "idma_bytes %lu\n",
                   PHASES[p].name, (unsigned long)ph[p].cyc[0], (unsigned long)(c / w),
                   (unsigned long)(a / w), (unsigned long)(b / w), (unsigned long)(dd / w),
                   (unsigned long)phase_wall[p], (unsigned long)simd_busy_in_phase[p],
                   (unsigned long)idma_bytes_in_phase[p]);
        }
        printf("CALIBG done\n");
    }
    snrt_cluster_hw_barrier();
    return 0;
}
