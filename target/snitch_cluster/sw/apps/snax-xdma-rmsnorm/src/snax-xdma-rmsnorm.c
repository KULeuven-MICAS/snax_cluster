// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// xDMA FP16 rmsnorm: out = x * rsqrt(mean(x^2)), mean = sum(x^2)/N, eps = 0.
// The scalar rsqrt is NOT an xDMA op: the StreamScalar extension and StreamMap's operandMode were removed
// (a single 1/sqrt only uses one SIMD lane, so it wastes the xDMA datapath). The xDMA does the SIMD-wide
// work (sum of squares, then a plain scale); the HOST computes the one 1/sqrt(mean):
//
//   T1  sum(x^2)  : StreamReduce(SUMSQ)                      x -> ssq (1 beat)   [xDMA]
//   host          : inv_rms = 1/sqrt(ssq/N)                                      [host, ~110 cc]
//   T2  scale     : StreamMap(a=inv_rms, b=0, func=LINEAR)   x -> out            [xDMA]
//
// The DM core (rv32ima, no FPU) cannot do the rsqrt; on the real HeMAiA host the CVA6 computes it. This
// standalone test precomputes inv_rms in data.h (rmsnorm_inv_rms) to stay self-validating on vsim, and
// checks the runtime SUMSQ against the golden scalar (rmsnorm_ssq_golden). mean = sum(x^2)/N is exact
// (N = 2^log2n) and is folded into the precomputed inv_rms.
//
// Cost: the xDMA reduce+scale is unchanged vs the old folded-HW-rsqrt version (the rsqrt added ~0 cc); the
// host now adds ~110 cc per call for 1/sqrt(mean) -- estimated from the bingo op-LUTs (inputs/op_luts/simd)
// at the scalar case n=1: fp16_sqrt ~52 cc composed with fp16_reciprocal ~57 cc (there is no rsqrt LUT, and
// fp16_reciprocal is a flagged placeholder). Net: an area-for-cycles trade (drops 2x FpRecipRsqrt + the
// StreamScalar extension) costing ~110 host cc -- small vs the orchestration-bound xDMA total.
//
// Performance (FP16, measured under vsim, L1<->L1; xDMA part). warm = steady-state (AGU shape persists);
// cold = first call. "+host" adds the estimated host rsqrt (~110 cc). host = a single CVA6+Ara core running
// the full FP32 rmsnorm (the old all-host baseline). Significant outputs match the FP64 golden to <=4 ULP.
//
//   N      beats   xDMA warm   +host est   cold    host(full)   warm speedup(+host)
//   ----   -----   ---------   ---------   -----   ----------   -------------------
//   64     2       661         ~771        1,843   2,962        3.8x
//   256    8       697         ~807        1,883   11,239       13.9x
//   1024   32      855         ~965        2,025   44,463       46.1x
//   4096   128     1,527       ~1,637      2,697   177,381      108.4x

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMREDUCE) || !defined(READER_EXT_STREAMMAP)
#error "Regenerate the XDMA CSR map with StreamReduce and StreamMap."
#endif

#define XDMA_BEAT_BYTES 64
#define OP_SUMSQ 2u
#define ACT_NONE 0u  // StreamMap func CSR bits[1:0]=0: LINEAR (out = a*x + b)

// FP16 bits -> monotonic ordering key (handles signed outputs): adjacent FP16
// values map to adjacent keys, so |key(a)-key(b)| is the FP16-ULP distance.
static inline uint32_t fp16_mono(uint16_t h) {
    uint32_t mag = h & 0x7FFFu;
    return (h & 0x8000u) ? (0x8000u - mag) : (0x8000u + mag);
}

// Minimal re-task launch: reuse the persisted temporal shape, rewrite only addresses + dst bound.
static uint32_t retask_and_run(void* src, void* dst, uint32_t dst_bound0) {
    if (snax_xdma_retask_1d(src, dst, dst_bound0) != 0) return 0xFFFFFFFFu;
    int task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    return snax_xdma_last_task_cycle();
}

int main() {
    int err = 0;
    if (snrt_is_dm_core()) {
        uint32_t base = snrt_cluster_base_addrl();
        uint32_t beats = rmsnorm_beats;
        uint32_t row_bytes = beats * XDMA_BEAT_BYTES;

        uint8_t* x_in    = (uint8_t*)base;
        uint8_t* ssq_buf = x_in + row_bytes;             // 1 beat (sum of squares)
        uint8_t* out_buf = ssq_buf + XDMA_BEAT_BYTES;    // beats

        printf("[Rmsnorm] N=%u beats=%u\n", rmsnorm_n, beats);

        snrt_dma_start_1d(x_in, rmsnorm_input, row_bytes);
        snrt_dma_wait_all();

        // The AGU shape (2D [beats,1] src reading x_in, 1D dst) is identical for both tasks and across
        // rows, so it is programmed ONCE (full _fast); each task then needs only a 5-write retask
        // (dst addr + writer beat count; the src addr x_in is common). xDMA regs reset to 0, so the
        // unused multicast dst pointers need no zeroing.
        uint32_t str_beat[1]    = {XDMA_BEAT_BYTES};
        uint32_t src_str_2d[2]  = {XDMA_BEAT_BYTES, beats * XDMA_BEAT_BYTES};
        uint32_t src_bnd_2d[2]  = {beats, 1};
        uint32_t bnd_1[1]       = {1};

        uint32_t c1 = 0, c2 = 0, lat_cold = 0, lat_warm = 0;
        int ok = 1;
        // Run twice: iter 0 = cold (one-time AGU shape setup + icache warming); iter 1 = warm / steady-state.
        for (int iter = 0; iter < 2; iter++) {
            uint32_t t0 = snrt_mcycle();
            if (iter == 0)
                ok &= (snax_xdma_memcpy_nd_fast(x_in, ssq_buf, 8, 8, 2, src_str_2d,
                                                src_bnd_2d, 1, str_beat, bnd_1,
                                                0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);

            // T1: sum(x^2).
            uint32_t csr_ssq[2] = {beats, OP_SUMSQ};
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_ssq) == 0);
            c1 = retask_and_run(x_in, ssq_buf, 1);
            snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);

            // The host computes inv_rms = 1/sqrt(sum(x^2)/N) from the SUMSQ scalar (on the real HeMAiA host
            // the CVA6 does the rsqrt; here it is precomputed in rmsnorm_inv_rms). Validate T1's reduce.
            if (iter == 1) {
                uint32_t r = fp16_mono(((uint16_t*)ssq_buf)[0]);
                uint32_t g = fp16_mono((uint16_t)rmsnorm_ssq_golden);
                uint32_t u = (r > g) ? (r - g) : (g - r);
                if (u > 2)
                    printf("[Rmsnorm] WARN runtime ssq %04x vs golden %04x (%u ulp)\n",
                           ((uint16_t*)ssq_buf)[0], (uint16_t)rmsnorm_ssq_golden, u);
            }

            // T2: out = x * inv_rms. Plain LINEAR multiply by the host-provided scalar (no operandMode).
            uint32_t csr_norm[3] = {rmsnorm_inv_rms, 0u, ACT_NONE};
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_norm) == 0);
            c2 = retask_and_run(x_in, out_buf, beats);
            snax_xdma_disable_src_ext(READER_EXT_STREAMMAP);

            uint32_t t1 = snrt_mcycle();
            if (iter == 0) lat_cold = t1 - t0; else lat_warm = t1 - t0;
        }

        if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu) {
            printf("[Rmsnorm] xDMA task setup failed\n");
            return 1;
        }
        printf("[Rmsnorm] cycles: sumsq=%u norm=%u xdma_total=%u\n", c1, c2, c1 + c2);
        printf("[Rmsnorm] full latency: cold=%u warm=%u cycles\n", lat_cold, lat_warm);

        // Integer FP16-ULP check on the significant (signed) outputs; skip the negligible subnormal/zero
        // golden (FTZ tail). The DM core has no FPU, so the compare stays integer (monotonic-key ULP).
        uint16_t* out_h = (uint16_t*)out_buf;
        uint32_t mism = 0, worst = 0, checked = 0;
        for (uint32_t i = 0; i < rmsnorm_n; i++) {
            if ((rmsnorm_golden[i] & 0x7C00u) == 0) continue;  // skip subnormal/zero golden
            checked++;
            uint32_t o = fp16_mono(out_h[i]), g = fp16_mono(rmsnorm_golden[i]);
            uint32_t ulp = (o > g) ? (o - g) : (g - o);
            if (ulp > worst) worst = ulp;
            if (ulp > 4) {
                if (mism < 6)
                    printf("[Rmsnorm] mismatch[%u]: got %04x golden %04x (%u ulp)\n",
                           i, out_h[i], rmsnorm_golden[i], ulp);
                mism++;
            }
        }
        printf("[Rmsnorm] significant=%u/%u worst FP16 ULP=%u\n", checked, rmsnorm_n, worst);
        if (mism == 0) {
            printf("[Rmsnorm] PASS (%u significant elements, <=4 ULP)\n", checked);
        } else {
            printf("[Rmsnorm] FAIL (%u/%u significant beyond 4 ULP)\n", mism, checked);
            err++;
        }
    }
    return err != 0;
}
