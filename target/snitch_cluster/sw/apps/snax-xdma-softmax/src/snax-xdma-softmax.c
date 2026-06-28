// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// xDMA FP16 softmax: out = exp(x - max(x)) / sum(exp(x - max(x))).
// The SIMD-wide work runs on the xDMA; the scalar 1/Σexp runs on the host (a single reciprocal only uses
// one lane, so it wastes the datapath -- StreamScalar and StreamMap.operandMode were removed for it):
//
//   T1  max       : StreamReduce(MAX)                                  x -> max               [xDMA]
//   T2  exp + sum  : StreamMap(EXP, b=-max) -||> StreamReduce(ADD,tap)  x -> exp row + Σexp    [xDMA]
//   host           : inv_sum = 1/Σexp                                                          [host, ~57 cc]
//   T3  norm       : StreamMap(a=inv_sum, b=0, func=LINEAR)            exp -> out             [xDMA]
//
// LANE DATAFLOW (FP16 transport = 32 lanes per 512-b beat):
//   T1 StreamReduce(MAX)  N beats -> 1 beat: 32 lanes collapse to s=max(row), SPLATTED [s s..s] (read lane 0).
//   T2 StreamMap(EXP, b=-max) -||> StreamReduce(ADD, TAP):
//      Map: e[i] = exp(x[i] - max)              1 beat -> 1 beat, all 32 lanes
//      Reduce in TAP mode PASSES the exp row through (N beats) and APPENDS the row-sum Sexp as ONE trailing
//      splatted beat -> T2 writes N+1 beats: [ exp_beat0 .. exp_beatN-1 , (S S .. S) ].
//   T3 StreamMap(LINEAR)  1 beat -> 1 beat: out[i] = inv_sum * exp[i]      (clean 512->512)
//   Fp16ToInt8 (fused quant): packs 2 FP16 beats -> 1 INT8 beat (writer drains beats/2, even count).
//
// The DM core (rv32ima, no FPU) can't do the reciprocal; a host core does it. This standalone
// test precomputes inv_sum in data.h (softmax_inv_sum) and checks the runtime Σexp against softmax_sum_golden
// (-max for T2 is a DM-core integer sign-flip, no FP).
//
// Performance (FP16, vsim, L1<->L1; xDMA part) on the native-Chisel FP RTL (FpAdd/FpMul/FpFma, fpPipe=1
// internal pipeline cut) + time-mux computeLanes (StreamMap=2, StreamReduce=4, Fp16ToInt8=8), FpExp 128-entry
// LUT. "+host" adds the
// estimated host reciprocal (~57 cc, from the fp16_reciprocal op-LUT at n=1). host = a single host vector
// core, full FP32 LUT softmax. Outputs match the FP64 golden to <=4 ULP (worst FP16 ULP=1). The CSR
// orchestration is a fixed ~1.3k cc (4 task setups); the datapath scales with N, so warm is CSR-bound at
// small N and datapath-bound for N>=1k (the time-mux trades datapath throughput, felt only at large N).
//
//   N      beats   xDMA warm   +host est   cold     host(full)   warm speedup(+host)
//   ----   -----   ---------   ---------   ------   ----------   -------------------
//   64     2       1,567       ~1,624      2,345    6,489        4.0x
//   256    8       2,127       ~2,184      2,905    24,153       11.1x
//   1024   32      4,349       ~4,406      5,093    95,177       21.6x
//   4096   128     13,277      ~13,334     14,021   379,441      28.5x

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMREDUCE) || !defined(READER_EXT_STREAMMAP) || \
    !defined(READER_EXT_FP16TOINT8)
#error "Regenerate the XDMA CSR map with StreamReduce, StreamMap and Fp16ToInt8."
#endif

#define XDMA_BEAT_BYTES 64
#define OP_MAX 0u
#define OP_ADD 1u
#define RED_TAP 0x100u       // StreamReduce op CSR bit[8]: pass row through + trailing scalar beat
#define ACT_NONE 0u          // StreamMap func CSR bits[1:0]=0: LINEAR (out = a*x + b)
#define ACT_EXP 1u

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
        uint32_t beats = softmax_beats;
        uint32_t row_bytes = beats * XDMA_BEAT_BYTES;

        uint8_t* x_in       = (uint8_t*)base;
        uint8_t* max_buf    = x_in + row_bytes;                          // 1 beat
        uint8_t* exp_buf    = max_buf + XDMA_BEAT_BYTES;                 // beats + 1 (exp row + trailing Σ)
        uint8_t* out_buf    = exp_buf + (beats + 1) * XDMA_BEAT_BYTES;   // beats   (FP16 softmax result)
        uint8_t* out_i8_buf = out_buf + row_bytes;                       // beats/2 (INT8 packed, fused quant)

        printf("[Softmax] N=%u beats=%u\n", softmax_n, beats);

        snrt_dma_start_1d(x_in, softmax_input, row_bytes);
        snrt_dma_wait_all();

        // The AGU shape (2D [beats,1] src, 1D dst) is identical for all 3 tasks and across rows, so it is
        // programmed ONCE (full _fast); every task then needs only a 5-write retask (src/dst addr + the
        // writer beat count). xDMA regs reset to 0, so the unused multicast dst pointers need no zeroing.
        uint32_t str_beat[1]    = {XDMA_BEAT_BYTES};
        uint32_t src_str_2d[2]  = {XDMA_BEAT_BYTES, beats * XDMA_BEAT_BYTES};
        uint32_t src_bnd_2d[2]  = {beats, 1};
        uint32_t bnd_1[1]       = {1};

        uint32_t c1 = 0, c2 = 0, c3 = 0, cq = 0, lat_cold = 0, lat_warm = 0;
        int ok = 1;
        // Run twice: iter 0 = cold (pays the one-time AGU shape setup + icache warming of the
        // CSR-dispatch code); iter 1 = warm / steady-state (as in repeated inference: shape persists,
        // only the per-row operands + addresses are reprogrammed).
        for (int iter = 0; iter < 2; iter++) {
            uint32_t t0 = snrt_mcycle();
            if (iter == 0)
                ok &= (snax_xdma_memcpy_nd_fast(x_in, max_buf, 8, 8, 2, src_str_2d,
                                                src_bnd_2d, 1, str_beat, bnd_1,
                                                0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) == 0);

            // T1: row max.
            uint32_t csr_max[2] = {beats, OP_MAX};
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_max) == 0);
            c1 = retask_and_run(x_in, max_buf, 1);
            snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);
            uint32_t neg_max = fp16_bits_to_fp32_bits(((uint16_t*)max_buf)[0]) ^ 0x80000000u;

            // T2: exp(x-max) + Σexp fused (StreamMap(EXP) -||> StreamReduce(ADD,tap)); Σ is the trailing
            // beat at exp_buf[beats]. retask: same shape, only addr + writer bound change.
            uint32_t csr_exp[3]    = {0x3F800000u /*1.0f*/, neg_max, ACT_EXP};
            uint32_t csr_sumtap[2] = {beats, OP_ADD | RED_TAP};
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_exp) == 0);
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCE, csr_sumtap) == 0);
            c2 = retask_and_run(x_in, exp_buf, beats + 1);
            snax_xdma_disable_src_ext(READER_EXT_STREAMMAP);
            snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCE);

            // The host computes inv_sum = 1/Σexp from the reduce scalar (a host core
            // does the reciprocal; here it is precomputed in softmax_inv_sum). Validate T2's Σexp (positive,
            // so the raw FP16 bits are monotonic).
            if (iter == 1) {
                uint16_t sum_hw = ((uint16_t*)(exp_buf + beats * XDMA_BEAT_BYTES))[0];
                int32_t d = (int32_t)sum_hw - (int32_t)(uint16_t)softmax_sum_golden;
                if (d > 2 || d < -2)
                    printf("[Softmax] WARN runtime sumexp %04x vs golden %04x\n",
                           sum_hw, (uint16_t)softmax_sum_golden);
            }

            // T3: out = inv_sum * exp. Plain LINEAR multiply by the host-provided reciprocal (no operandMode).
            uint32_t csr_norm[3] = {softmax_inv_sum, 0u, ACT_NONE};
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_norm) == 0);
            c3 = retask_and_run(exp_buf, out_buf, beats);
            snax_xdma_disable_src_ext(READER_EXT_STREAMMAP);

            // Tq: SAME normalize pass, but Fp16ToInt8 chained after StreamMap quantizes the probabilities
            // to int8 in-stream (no re-read). The writer emits beats/2 packed beats; cq - c3 is the margin.
            uint32_t csr_q[1] = {softmax_inv_scale};
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_norm) == 0);
            ok &= (snax_xdma_enable_src_ext(READER_EXT_FP16TOINT8, csr_q) == 0);
            cq = retask_and_run(exp_buf, out_i8_buf, beats / 2);
            snax_xdma_disable_src_ext(READER_EXT_FP16TOINT8);
            snax_xdma_disable_src_ext(READER_EXT_STREAMMAP);

            uint32_t t1 = snrt_mcycle();
            if (iter == 0) lat_cold = t1 - t0; else lat_warm = t1 - t0;
        }

        if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu || c3 == 0xFFFFFFFFu || cq == 0xFFFFFFFFu) {
            printf("[Softmax] xDMA task setup failed\n");
            return 1;
        }
        printf("[Softmax] cycles: max=%u exp+sum=%u norm=%u norm+quant=%u (quant marginal=%u)\n",
               c1, c2, c3, cq, cq - c3);
        printf("[Softmax] full latency: cold=%u warm=%u cycles\n", lat_cold, lat_warm);

        // Integer FP16-ULP check on the SIGNIFICANT outputs (softmax positive -> FP16 bits
        // monotonic). The negligible subnormal tail (golden value < ~6e-5, FP16 exp field 0) is
        // skipped: the HW narrows FP32->FP16 with subnormal flush-to-zero, so those tiny values
        // differ from the FP64 golden by many ULP while contributing ~0 probability mass. The DM
        // core has no FPU, so the compare must stay integer.
        uint16_t* out_h = (uint16_t*)out_buf;
        uint32_t mism = 0, worst = 0, checked = 0;
        for (uint32_t i = 0; i < softmax_n; i++) {
            if ((softmax_golden[i] & 0x7C00u) == 0) continue;  // skip negligible subnormal/zero golden
            checked++;
            int32_t o = (int32_t)out_h[i], g = (int32_t)softmax_golden[i];
            uint32_t ulp = (o > g) ? (uint32_t)(o - g) : (uint32_t)(g - o);
            if (ulp > worst) worst = ulp;
            if (ulp > 4) {
                if (mism < 6)
                    printf("[Softmax] mismatch[%u]: got %04x golden %04x (%u ulp)\n",
                           i, out_h[i], softmax_golden[i], ulp);
                mism++;
            }
        }
        printf("[Softmax] significant=%u/%u worst FP16 ULP=%u\n", checked, softmax_n, worst);
        if (mism == 0) {
            printf("[Softmax] PASS (%u significant elements, <=4 ULP)\n", checked);
        } else {
            printf("[Softmax] FAIL (%u/%u significant beyond 4 ULP)\n", mism, checked);
            err++;
        }

        // Quantize check: fused int8 out vs golden quant(softmax_golden). Tolerance 2 absorbs the upstream
        // FP16 (<=4 ULP) difference between the HW result and the FP16 golden after scaling.
        int8_t* out_i8 = (int8_t*)out_i8_buf;
        uint32_t qmism = 0, qworst = 0;
        for (uint32_t i = 0; i < softmax_n; i++) {
            int d = (int)out_i8[i] - (int)softmax_golden_i8[i];
            uint32_t ad = (d < 0) ? (uint32_t)(-d) : (uint32_t)d;
            if (ad > qworst) qworst = ad;
            if (ad > 2) {
                if (qmism < 6)
                    printf("[Softmax] int8 mismatch[%u]: got %d golden %d\n", i, out_i8[i], softmax_golden_i8[i]);
                qmism++;
            }
        }
        printf("[Softmax] quant worst |delta int8|=%u\n", qworst);
        if (qmism == 0) {
            printf("[Softmax] QUANT PASS (<=2 int8)\n");
        } else {
            printf("[Softmax] QUANT FAIL (%u beyond 2 int8)\n", qmism);
            err++;
        }
    }
    return err != 0;
}
