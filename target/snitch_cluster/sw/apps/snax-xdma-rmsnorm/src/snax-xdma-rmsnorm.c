// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// xDMA FP16 rmsnorm: out = x * rsqrt(mean(x^2)), mean = sum(x^2)/N, eps = 0.
// 2 chained xDMA tasks, L1<->L1. ALL math is offloaded to the xDMA reader SIMD
// extensions; the DM core (rv32ima, no FPU) only forms one scalar via integer
// bit-ops between the tasks:
//
//   T1  sum(x^2) : StreamReduce(SUMSQ)                         x -> ssq (1 beat)
//   T2  scale    : StreamMap(a=mean, operandMode=RSQRT, b=0)   x -> out  (= x*rsqrt(mean))
//
// N is a power of two, so mean = sum(x^2)/N is an integer exponent-field subtraction
// on the FP32 bits (no FPU, no FP multiply). The recip/rsqrt of the scalar `a` is
// folded into the normalize map via StreamMap's operandMode (one shared FpRecipRsqrt).
// Reuses the same generic blocks as snax-xdma-softmax.
//
// Performance (FP16, measured under vsim, L1<->L1). warm = steady-state (the AGU shape persists across
// rows, as in repeated inference); cold = first call, incl. the one-time shape setup + icache warm.
// host = a single CVA6+Ara core, FP32. Significant outputs match the FP64 golden to <=1 FP16 ULP.
//
//   N      beats   warm    cold    host       warm speedup
//   ----   -----   -----   -----   -------    ------------
//   64     2       644     1,894   2,962      4.6x
//   256    8       680     1,934   11,239     16.5x
//   1024   32      841     2,104   44,463     52.9x
//   4096   128     1,513   2,776   177,381    117.2x

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMREDUCE) || !defined(READER_EXT_STREAMMAP)
#error "Regenerate the XDMA CSR map with StreamReduce and StreamMap."
#endif

#define XDMA_BEAT_BYTES 64
#define OP_SUMSQ 2u
#define ACT_NONE 0u
#define MAP_RSQRT_A 0x200u  // StreamMap act CSR bits[9:8]=2: aEff = 1/sqrt(a)

// FP16 bit pattern -> FP32 bit pattern (pure integer, no FPU).
static inline uint32_t fp16_bits_to_fp32_bits(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 0x1;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1F;
    uint32_t mant = (uint32_t)(h & 0x3FF);
    if (exp == 0) {
        if (mant == 0) return sign << 31;
        int shift = 0;
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            shift++;
        }
        mant &= 0x3FF;
        return (sign << 31) | ((127 - 15 - shift) << 23) | (mant << 13);
    } else if (exp == 31) {
        return (sign << 31) | 0x7F800000u | (mant << 13);
    }
    return (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
}

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

            // mean = sum(x^2)/N, N = 2^log2n -> subtract log2n from the FP32 exponent field (integer).
            uint32_t mean_bits =
                fp16_bits_to_fp32_bits(((uint16_t*)ssq_buf)[0]) - (rmsnorm_log2n << 23);

            // T2: out = x * rsqrt(mean). rsqrt folded into the map operand (a := 1/sqrt(a)).
            uint32_t csr_norm[3] = {mean_bits, 0u, ACT_NONE | MAP_RSQRT_A};
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
