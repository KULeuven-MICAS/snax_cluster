// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// xDMA FP16 softmax: out = exp(x - max(x)) / sum(exp(x - max(x))).
// CHAIN-FUSED into 3 xDMA tasks (was 5), L1<->L1 (localLoopback). ALL math is offloaded to the xDMA
// reader SIMD extensions. The DM core (rv32ima, no FPU) only shuffles two row-scalars via integer
// bit-ops between tasks, so it avoids every floating-point instruction (the toolchain emits hard-FP
// that traps on this core).
//
//   T1  max         : StreamReduce(MAX)                                  x -> max
//   T2  exp + sum    : StreamMap(EXP, b=-max)  -||>  StreamReduce(ADD,tap)  x -> exp row + Σexp(trailing beat)
//   T3  norm (1/Σ)   : StreamMap(a=Σ, operandMode=RECIP)                 exp -> out
//
// Softmax has two hard barriers (need max before exp; need Σexp before normalize), so 3 streaming
// phases is the floor for reusable (non-monolithic) blocks. The exp+sum fusion uses StreamReduce's tap
// mode (pass the row through + emit the scalar as a trailing beat); the recip is folded into the
// normalize map's operandMode. Per-task CSR programming uses the delta (_fast) path after one full
// config.
//
// Performance (FP16, measured under vsim, L1<->L1). warm = steady-state (the AGU shape persists across
// rows, as in repeated inference); cold = first call, incl. the one-time shape setup + icache warm.
// host = a single CVA6+Ara core running the FP32 LUT softmax. Significant outputs match the FP64 golden
// to <=1 FP16 ULP at every N.
//
//   N      beats   warm    cold    host       warm speedup
//   ----   -----   -----   -----   -------    ------------
//   64     2       1,059   2,495   6,489      6.1x
//   256    8       1,135   2,563   24,153     21.3x
//   1024   32      1,449   2,890   95,177     65.7x
//   4096   128     2,697   4,138   379,441    140.7x

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMREDUCE) || !defined(READER_EXT_STREAMMAP)
#error "Regenerate the XDMA CSR map with StreamReduce and StreamMap."
#endif

#define XDMA_BEAT_BYTES 64
#define OP_MAX 0u
#define OP_ADD 1u
#define RED_TAP 0x100u       // StreamReduce op CSR bit[8]: pass row through + trailing scalar beat
#define ACT_NONE 0u
#define ACT_EXP 1u
#define MAP_RECIP_A 0x100u   // StreamMap act CSR bits[9:8]=1: aEff = 1/a (fold recip into the map)

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

        uint8_t* x_in    = (uint8_t*)base;
        uint8_t* max_buf = x_in + row_bytes;                          // 1 beat
        uint8_t* exp_buf = max_buf + XDMA_BEAT_BYTES;                 // beats + 1 (exp row + trailing Σ)
        uint8_t* out_buf = exp_buf + (beats + 1) * XDMA_BEAT_BYTES;   // beats

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

        uint32_t c1 = 0, c2 = 0, c3 = 0, lat_cold = 0, lat_warm = 0;
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
            uint32_t sum_f32 = fp16_bits_to_fp32_bits(
                ((uint16_t*)(exp_buf + beats * XDMA_BEAT_BYTES))[0]);

            // T3: normalize = (1/Σ)*exp, recip folded into the map operand (a:=1/a). retask: same shape.
            uint32_t csr_norm[3] = {sum_f32, 0u, ACT_NONE | MAP_RECIP_A};
            ok &= (snax_xdma_enable_src_ext(READER_EXT_STREAMMAP, csr_norm) == 0);
            c3 = retask_and_run(exp_buf, out_buf, beats);
            snax_xdma_disable_src_ext(READER_EXT_STREAMMAP);

            uint32_t t1 = snrt_mcycle();
            if (iter == 0) lat_cold = t1 - t0; else lat_warm = t1 - t0;
        }

        if (!ok || c1 == 0xFFFFFFFFu || c2 == 0xFFFFFFFFu || c3 == 0xFFFFFFFFu) {
            printf("[Softmax] xDMA task setup failed\n");
            return 1;
        }
        printf("[Softmax] cycles: max=%u exp+sum=%u norm=%u xdma_total=%u\n",
               c1, c2, c3, c1 + c2 + c3);
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
    }
    return err != 0;
}
