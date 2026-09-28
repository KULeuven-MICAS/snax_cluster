// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V2-Lite's latent cache and its append (D1). Each token caches one INT8 row
// [c (512) | k_pe (64)], kept twice in DRAM so each attention GEMM reads its operand as is:
//
//   KEY    A-layout of tokens x 576 (blocks of 16 tokens x 4 values): the scores' A operand
//   VALUE  A-layout of V^T = 512 x tokens (blocks of 16 latents x 4 tokens): the weighted
//          sum's A operand. The INT8 transposer works on 8 x 8-byte tiles and cannot turn one
//          layout into the other, so both are written on every append.
//
// An append (snax-dsv2.h dsv2_cache_append) is one 2-D iDMA transfer into the key copy (144
// runs of 4 bytes) and 32 into the value copy (16 runs of one byte each). This app starts
// from the golden pack's L cached rows, appends the rows of positions L, L+1 and L+2, then
// reads back every block an append touched -- the whole 16-token key blocks and the whole
// 4-token value blocks around the new rows -- and compares them byte for byte with the copies
// the datagen built. main returns the number of failed checks. The iDMA's spans over the
// appends and the cluster census over them follow ([SPAN], [UTIL]; snax-dsv2-trace.h).

#include "data.h"
#include "snax-core-roles.h"
#include "snax-dsv2.h"
#include "snax-dsv2-trace.h"

#if !SNAX_HAS_IDMA_CORE
#error "the cache append runs on the iDMA"
#endif

#define TAG "[CACHE]"
#define KBLK ((DSV2_KV_ROW / 4u) * 64u)  // one 16-token key block: 9,216 bytes

int main() {
    if (!snax_is_idma_core()) return 0;

    uint8_t *p = (uint8_t *)(((uint32_t)snrt_l1_next() + 63u) & ~63u);
    const uint32_t tlo = CACHE_L, thi = CACHE_L + N_APPEND - 1u;
    const uint32_t mlo = tlo / 16u, mhi = thi / 16u, klo = tlo / 4u, khi = thi / 4u;
    const uint32_t kbytes = (mhi - mlo + 1u) * KBLK;          // key blocks touched
    const uint32_t vrun = (khi - klo + 1u) * 64u;             // value bytes touched per m
    const uint32_t vbytes = (DSV2_KV_RANK / 16u) * vrun;
    int8_t *rows = (int8_t *)p;                               p += N_APPEND * DSV2_KV_ROW;
    p = (uint8_t *)(((uint32_t)p + 63u) & ~63u);
    int8_t *kgot = (int8_t *)p, *kwant = kgot + kbytes;       p += 2u * kbytes;
    int8_t *vgot = (int8_t *)p, *vwant = vgot + vbytes;       p += 2u * vbytes;
    p = (uint8_t *)(((uint32_t)p + 63u) & ~63u);
    dsv2_trace_t *tr = (dsv2_trace_t *)p;                     p += sizeof(dsv2_trace_t);
    dsv2_print_l1((uint32_t)(p - (uint8_t *)rows), DSV2_L1_TOP - (uint32_t)rows);
    if ((uint32_t)p > DSV2_L1_TOP) {
        printf(TAG " FAIL: the buffers end past the stacks\n");
        return 1;
    }

    for (uint32_t k = 0; k < 16u; k++) tr->n[k] = 0u;  // this arena is not zeroed
    uint32_t t0 = snrt_mcycle();
    snrt_dma_start_1d(rows, new_rows, N_APPEND * DSV2_KV_ROW);
    snrt_dma_wait_all();
    dsv2_span(tr, DSV2_TR_IDMA, "7 the new rows", t0, snrt_mcycle());
    uint32_t cyc[N_APPEND];
    snax_perf_arm();
    for (uint32_t i = 0; i < N_APPEND; i++) {
        t0 = snrt_mcycle();
        dsv2_cache_append(key, val, CACHE_CAP, CACHE_L + i, rows + i * DSV2_KV_ROW);
        cyc[i] = snrt_mcycle() - t0;
        dsv2_span(tr, DSV2_TR_IDMA, "7 cache append", t0, snrt_mcycle());
    }
    snax_perf_snapshot_t census;
    snax_perf_read(&census);

    // Read back what the appends touched, and the same bytes of the expected copies.
    snrt_dma_start_1d(kgot, key + mlo * KBLK, kbytes);
    snrt_dma_start_1d(kwant, g_key + mlo * KBLK, kbytes);
    snrt_dma_start_2d(vgot, val + klo * 64u, vrun, vrun, CACHE_CAP * 16u, DSV2_KV_RANK / 16u);
    snrt_dma_start_2d(vwant, g_val + klo * 64u, vrun, vrun, CACHE_CAP * 16u, DSV2_KV_RANK / 16u);
    snrt_dma_wait_all();

    uint32_t fails = 0, worst;
    printf(TAG " %u rows cached, capacity %u; appended positions %u..%u:", CACHE_L, CACHE_CAP,
           tlo, thi);
    for (uint32_t i = 0; i < N_APPEND; i++) printf(" %u cc", cyc[i]);
    printf("\n");
    uint32_t bad = dsv2_lsb8(kgot, kwant, kbytes, &worst);
    fails += dsv2_check(TAG, bad == 0, "key copy", "the touched 16-token blocks, byte-exact");
    if (bad) printf(TAG "     %u of %u bytes differ\n", bad, dsv2_check_terms(kbytes));
    bad = dsv2_lsb8(vgot, vwant, vbytes, &worst);
    fails += dsv2_check(TAG, bad == 0, "value copy", "the touched 4-token blocks, byte-exact");
    if (bad) printf(TAG "     %u of %u bytes differ\n", bad, dsv2_check_terms(vbytes));
    printf(fails ? TAG " FAIL (%u checks)\n" : TAG " PASS\n", fails);
    dsv2_trace_print(tr);
    dsv2_util_print(&census, 0u);
    return (int)fails;
}
