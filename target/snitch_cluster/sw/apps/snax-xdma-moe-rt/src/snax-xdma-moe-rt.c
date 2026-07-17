// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// RUNTIME-PRECISION xDMA MoE gating: argmax over a row of E expert logits -> the
// winning expert index, computed IN-TRANSIT by StreamReduceRt(ARGMAX) at FP16
// (32 experts/beat) AND FP8 (64 experts/beat) on the SAME netlist, precision by
// the `fmt` CSR field. The reduce carries the (value,index) pair through the
// horizontal fold (same comparator as MAX -> roofline preserved) and emits the
// winning lane INDEX as a raw 32-bit int splatted across the output beat. The
// token->expert DISPATCH that follows is the DMA's native gather (no new HW).
// Top-k = k masked-argmax passes; here we demo top-1.

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(READER_EXT_STREAMREDUCERT)
#error "This app needs StreamReduceRt with ARGMAX (cfg/snax_xdma_cluster.hjson op: [..., ARGMAX])."
#endif

#define XDMA_BEAT_BYTES 64
#define FMT_FP16 0u
#define FMT_FP8 2u
#define OP_ARGMAX 3u

// argmax over one beat of E experts (E<=64). Returns the winning index; *cyc = task cycles.
static uint32_t run_argmax(uint32_t fmt, uint8_t* x_in, uint8_t* idx_buf, uint32_t* cyc) {
    uint32_t str[1]     = {XDMA_BEAT_BYTES};
    uint32_t bnd1[1]    = {1};
    uint32_t src_str[2] = {XDMA_BEAT_BYTES, XDMA_BEAT_BYTES};
    uint32_t src_bnd[2] = {1, 1}; // one beat = one row (operandCount=1)
    snax_xdma_memcpy_nd_fast(x_in, idx_buf, 8, 8, 2, src_str, src_bnd, 1, str, bnd1,
                             0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    uint32_t csr[2] = {1u | (fmt << 16), OP_ARGMAX}; // operandCount=1, fmt; op=ARGMAX
    snax_xdma_enable_src_ext(READER_EXT_STREAMREDUCERT, csr);
    snax_xdma_retask_1d(x_in, idx_buf, 1);
    int t = snax_xdma_start();
    snax_xdma_local_wait(t);
    *cyc = snax_xdma_last_task_cycle();
    snax_xdma_disable_src_ext(READER_EXT_STREAMREDUCERT);
    return ((uint32_t*)idx_buf)[0]; // low 32 bits = the argmax lane index (splatted)
}

int main() {
    int err = 0;
    if (snrt_is_dm_core()) {
        uint32_t base = snrt_cluster_base_addrl();
        uint8_t* x_in = (uint8_t*)base;
        uint8_t* idx_buf = x_in + XDMA_BEAT_BYTES;
        printf("[MoERt] argmax gating on ONE netlist: FP16 (%u experts) + FP8 (%u experts)\n",
               moe_e_fp16, moe_e_fp8);

        // FP16: 32 experts in one beat
        snrt_dma_start_1d(x_in, moe_logits_fp16, XDMA_BEAT_BYTES);
        snrt_dma_wait_all();
        uint32_t c16 = 0;
        uint32_t hw16 = run_argmax(FMT_FP16, x_in, idx_buf, &c16);
        printf("[MoERt] FP16: argmax expert=%u (gold %u) cycles=%u : %s\n", hw16, moe_argmax_fp16,
               c16, hw16 == moe_argmax_fp16 ? "PASS" : "FAIL");
        if (hw16 != moe_argmax_fp16) err++;

        // FP8: 64 experts in one beat, SAME netlist
        snrt_dma_start_1d(x_in, moe_logits_fp8, XDMA_BEAT_BYTES);
        snrt_dma_wait_all();
        uint32_t c8 = 0;
        uint32_t hw8 = run_argmax(FMT_FP8, x_in, idx_buf, &c8);
        printf("[MoERt] FP8 : argmax expert=%u (gold %u) cycles=%u : %s\n", hw8, moe_argmax_fp8,
               c8, hw8 == moe_argmax_fp8 ? "PASS" : "FAIL");
        if (hw8 != moe_argmax_fp8) err++;

        if (!err)
            printf("[MoERt] ALL PASS — MoE argmax gating at FP16 & FP8 by runtime fmt\n");
    }
    return err != 0;
}
