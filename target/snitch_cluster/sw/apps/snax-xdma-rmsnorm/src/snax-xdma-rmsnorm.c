// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// xDMA FP16 rmsnorm: out = x * rsqrt(mean(x^2)), mean = sum(x^2)/N, eps = 0.
// The SIMD-wide work runs on the xDMA; the scalar rsqrt runs on the host (a single 1/sqrt only uses one
// lane, so it wastes the datapath -- StreamScalar and StreamMap.operandMode were removed for it):
//
//   T1  sum(x^2)  : StreamReduce(SUMSQ)                      x -> ssq           [xDMA]
//   host          : inv_rms = 1/sqrt(ssq/N)                                     [host, ~110 cc]
//   T2  scale     : StreamMap(a=inv_rms, b=0, func=LINEAR)   x -> out           [xDMA]
//
// LANE DATAFLOW (FP16 transport = 32 lanes per 512-b beat):
//   T1 StreamReduce(SUMSQ)  N beats -> 1 beat (the row's 32 lanes COLLAPSE to a scalar; not 512->512):
//     in beat 0..N-1 [ x0 .. x31 ] -> per-lane FP32 partials accumulate, then a horizontal tree folds
//     them to s = sum(x^2) over the row;  out [ s s ... s ] SPLATTED across all 32 lanes (read lane 0).
//   T2 StreamMap(LINEAR)    1 beat -> 1 beat, all 32 lanes:  out[i] = inv_rms*x[i] + 0   (clean 512->512)
//   Fp16ToInt8 (fused quant): packs 2 FP16 beats -> 1 INT8 beat (writer drains beats/2, even count).
//
// The DM core (rv32ima, no FPU) can't do the rsqrt; a host core does it. This standalone test precomputes
// inv_rms in data.h (rmsnorm_inv_rms; mean = sum(x^2)/N is exact for N=2^log2n) and checks the runtime
// SUMSQ against rmsnorm_ssq_golden.
//
// Performance (FP16, vsim, L1<->L1; xDMA part) on the area/timing-optimized RTL: pipelined FP datapaths +
// time-mux computeLanes (StreamReduce=4, StreamMap=2, Fp16ToInt8=8). "+host" adds the estimated host rsqrt
// (~110 cc, from the fp16_sqrt + fp16_reciprocal op-LUTs at n=1). host = a single host vector core, full
// FP32 rmsnorm. Outputs match the FP64 golden to <=4 ULP (worst FP16 ULP=1). The CSR orchestration is a
// fixed ~0.9k cc (3 task setups); the datapath scales with N (CSR-bound at small N, datapath-bound N>=1k).
//
//   N      beats   xDMA warm   +host est   cold    host(full)   warm speedup(+host)
//   ----   -----   ---------   ---------   -----   ----------   -------------------
//   64     2       1,064       ~1,174      1,732   2,962        2.5x
//   256    8       1,416       ~1,526      2,088   11,239       7.4x
//   1024   32      2,828       ~2,938      3,486   44,463       15.1x
//   4096   128     8,492       ~8,602      9,150   177,381      20.6x

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMREDUCE) || !defined(READER_EXT_STREAMMAP) || \
    !defined(READER_EXT_FP16TOINT8)
#error "Regenerate the XDMA CSR map with StreamReduce, StreamMap and Fp16ToInt8."
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

        uint8_t* x_in       = (uint8_t*)base;
        uint8_t* ssq_buf    = x_in + row_bytes;            // 1 beat (sum of squares)
        uint8_t* out_buf    = ssq_buf + XDMA_BEAT_BYTES;   // beats   (FP16 rmsnorm result)
        uint8_t* out_i8_buf = out_buf + row_bytes;         // beats/2 (INT8 packed, fused quantize)

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

        uint32_t c1 = 0, c2 = 0, cq = 0, lat_cold = 0, lat_warm = 0;
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

            // The host computes inv_rms = 1/sqrt(sum(x^2)/N) from the SUMSQ scalar (a host core does the
            // rsqrt; here it is precomputed in rmsnorm_inv_rms). Validate T1's reduce.
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

            // Tq: SAME scale pass, but Fp16ToInt8 chained after StreamMap quantizes the result to int8
            // in-stream (no re-read). The writer emits beats/2 packed beats; cq - c2 is the marginal cost.
            uint32_t csr_q[1] = {rmsnorm_inv_scale};
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_norm) == 0);
            ok &= (snax_xdma_enable_src_ext(READER_EXT_FP16TOINT8, csr_q) == 0);
            cq = retask_and_run(x_in, out_i8_buf, beats / 2);
            snax_xdma_disable_src_ext(READER_EXT_FP16TOINT8);
            snax_xdma_disable_src_ext(READER_EXT_STREAMMAP);

            uint32_t t1 = snrt_mcycle();
            if (iter == 0) lat_cold = t1 - t0; else lat_warm = t1 - t0;
        }

        if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu || cq == 0xFFFFFFFFu) {
            printf("[Rmsnorm] xDMA task setup failed\n");
            return 1;
        }
        printf("[Rmsnorm] cycles: sumsq=%u norm=%u norm+quant=%u (quant marginal=%u)\n",
               c1, c2, cq, cq - c2);
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

        // Quantize check: fused int8 out vs golden quant(rmsnorm_golden). Tolerance 2 absorbs the upstream
        // FP16 (<=4 ULP) difference between the HW result and the FP16 golden after scaling.
        int8_t* out_i8 = (int8_t*)out_i8_buf;
        uint32_t qmism = 0, qworst = 0;
        for (uint32_t i = 0; i < rmsnorm_n; i++) {
            int d = (int)out_i8[i] - (int)rmsnorm_golden_i8[i];
            uint32_t ad = (d < 0) ? (uint32_t)(-d) : (uint32_t)d;
            if (ad > qworst) qworst = ad;
            if (ad > 2) {
                if (qmism < 6)
                    printf("[Rmsnorm] int8 mismatch[%u]: got %d golden %d\n", i, out_i8[i], rmsnorm_golden_i8[i]);
                qmism++;
            }
        }
        printf("[Rmsnorm] quant worst |delta int8|=%u\n", qworst);
        if (qmism == 0) {
            printf("[Rmsnorm] QUANT PASS (<=2 int8)\n");
        } else {
            printf("[Rmsnorm] QUANT FAIL (%u beyond 2 int8)\n", qmism);
            err++;
        }
    }
    return err != 0;
}
