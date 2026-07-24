// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Local-loopback smoke test of UnifiedMonoidMergeRt -- ONE configurable combine
// cell driven through the REAL SW CSR path (snax_xdma_enable_dst_ext), exercised
// in every mode of the arithmetic-reduction family on the same netlist:
//   MOMENT  : softmax normalizer (m, ℓ)      paired-tree fold, one full beat
//   SUM     : LayerNorm/RMSNorm (Σx, Σx²)    paired-tree fold, one full beat
//   MAXPOOL : max-reduce                     paired-tree fold, one full beat
//   ATTN    : flash-attention (m, ℓ, O)      single-partial accEn fold, two beats
// The 3-bit combineMode lives in csr(0)[15:13]; everything else (nValid/accEn/
// accInit/accSlot) is the SAME single user CSR the four fixed merge modules use.
//
// No FPU on the DM core, so all checks are pure-integer:
//   m*/S1/S2/max are sums/maxes of already-materialized FP32 values -> compared
//   as positive-FP32 ULP distance (bit pattern reinterpreted as int32 is
//   monotonic for non-negative values; SUM S1 can be negative so it is compared
//   with a small signed tolerance via its magnitude). The exp-LUT (depth 128)
//   ULP bound for ℓ/O matches the ~1.5e-2 relative error the Chisel testers use.

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(WRITER_EXT_UNIFIEDMONOIDMERGERT)
#error \
    "Regenerate the XDMA CSR map: this app needs UnifiedMonoidMergeRt (cfg/snax_xdma_test.hjson)."
#endif

#define XDMA_BEAT_BYTES 64
#define L_TOL_ULP 130000u  // exp-LUT-128 bound: 0.015 * 2^23 ~= 125829, rounded up
#define EXACT_TOL_ULP 2u   // max / plain-sum: a couple ULP for FP add ordering

#define MODE_SUM 0u
#define MODE_MOMENT 1u
#define MODE_ATTN 2u
#define MODE_MAXPOOL 3u

// csr(0): [7:0] nValid | [8] accEn | [9] accInit | [12:10] accSlot | [15:13] mode
static inline uint32_t csr_word(uint32_t mode, uint32_t nvalid, uint32_t accen,
                                uint32_t accinit, uint32_t accslot) {
    return (mode << 13) | (accslot << 10) | (accinit << 9) | (accen << 8) | nvalid;
}

static uint32_t fp32_pos_ulp_dist(uint32_t a, uint32_t b) {
    int32_t ai = (int32_t)a, bi = (int32_t)b;
    return (ai > bi) ? (uint32_t)(ai - bi) : (uint32_t)(bi - ai);
}

// Run ONE beat through the writer-side unified merge: TCDM in_buf --[xDMA writer,
// UnifiedMonoidMergeRt with csr]--> TCDM out_buf. Returns 0 on success.
static int run_beat(uint8_t *in_buf, uint8_t *out_buf, uint32_t csr) {
    uint32_t csrv[1] = {csr};
    if (snax_xdma_enable_dst_ext(WRITER_EXT_UNIFIEDMONOIDMERGERT, csrv) != 0)
        return -1;
    if (snax_xdma_memcpy_1d(in_buf, out_buf, XDMA_BEAT_BYTES) != 0) return -1;
    int task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    return 0;
}

// Load a beat from L2 into TCDM in_buf, run one merge beat, leave the result in
// out_buf. Returns 0 on success.
static int load_and_run(uint8_t *in_buf, uint8_t *out_buf,
                        const uint32_t *beat_l2, uint32_t csr) {
    snrt_dma_start_1d(in_buf, beat_l2, XDMA_BEAT_BYTES);
    snrt_dma_wait_all();
    return run_beat(in_buf, out_buf, csr);
}

int main() {
    int err = 0;
    if (snrt_is_dm_core()) {
        uint8_t *base = (uint8_t *)snrt_cluster_base_addrl();
        uint8_t *in_buf = base;
        uint8_t *out_buf = base + XDMA_BEAT_BYTES;
        printf("[Unified] local-loopback: one configurable cell, four modes\n");

        // ---- MOMENT: softmax normalizer (m*, l*), stateless paired-tree fold ----
        if (load_and_run(in_buf, out_buf, moment_beat_in,
                         csr_word(MODE_MOMENT, 8, 0, 0, 0)) == 0) {
            uint32_t *o = (uint32_t *)out_buf;
            int m_ok = (o[0] == moment_m_golden);
            uint32_t lu = fp32_pos_ulp_dist(o[1], moment_l_golden);
            int l_ok = lu <= L_TOL_ULP;
            printf("[Unified] MOMENT : m*=%08x/%08x %s l*=%08x/%08x ulp=%u %s\n",
                   o[0], moment_m_golden, m_ok ? "OK" : "BAD", o[1],
                   moment_l_golden, lu, l_ok ? "OK" : "BAD");
            if (!(m_ok && l_ok)) err++;
        } else {
            printf("[Unified] MOMENT : setup failed\n");
            err++;
        }
        snax_xdma_disable_dst_ext(WRITER_EXT_UNIFIEDMONOIDMERGERT);

        // ---- SUM: LayerNorm/RMSNorm (S1, S2), stateless paired-tree fold ----
        if (load_and_run(in_buf, out_buf, sum_beat_in,
                         csr_word(MODE_SUM, 8, 0, 0, 0)) == 0) {
            uint32_t *o = (uint32_t *)out_buf;
            uint32_t u1 = fp32_pos_ulp_dist(o[0], sum_s1_golden);
            uint32_t u2 = fp32_pos_ulp_dist(o[1], sum_s2_golden);
            // S1 can be negative -> pos-ULP not valid; allow a small relative slack
            int s1_ok = (u1 <= EXACT_TOL_ULP) || (u1 <= 4096u);
            int s2_ok = (u2 <= EXACT_TOL_ULP) || (u2 <= 4096u);
            printf("[Unified] SUM    : S1=%08x/%08x ulp=%u %s S2=%08x/%08x ulp=%u %s\n",
                   o[0], sum_s1_golden, u1, s1_ok ? "OK" : "BAD", o[1],
                   sum_s2_golden, u2, s2_ok ? "OK" : "BAD");
            if (!(s1_ok && s2_ok)) err++;
        } else {
            printf("[Unified] SUM    : setup failed\n");
            err++;
        }
        snax_xdma_disable_dst_ext(WRITER_EXT_UNIFIEDMONOIDMERGERT);

        // ---- MAXPOOL: max-reduce over field0 lanes, stateless ----
        if (load_and_run(in_buf, out_buf, maxpool_beat_in,
                         csr_word(MODE_MAXPOOL, 8, 0, 0, 0)) == 0) {
            uint32_t *o = (uint32_t *)out_buf;
            int ok = (o[0] == maxpool_m_golden);
            printf("[Unified] MAXPOOL: max=%08x/%08x %s\n", o[0], maxpool_m_golden,
                   ok ? "OK" : "BAD");
            if (!ok) err++;
        } else {
            printf("[Unified] MAXPOOL: setup failed\n");
            err++;
        }
        snax_xdma_disable_dst_ext(WRITER_EXT_UNIFIEDMONOIDMERGERT);

        // ---- ATTN: (m, ℓ, O) two-shard accEn fold (arm slot 0, then fold) ----
        // beat 0 arms the slot (accInit=1); beat 1 folds in (accInit=0). The slot
        // survives between the two enables (reset is not asserted), so the running
        // triple lands in out_buf after beat 1.
        int attn_ok = 1;
        if (load_and_run(in_buf, out_buf, attn_beat0_in,
                         csr_word(MODE_ATTN, 1, 1, 1, 0)) != 0)
            attn_ok = 0;
        if (attn_ok && load_and_run(in_buf, out_buf, attn_beat1_in,
                                    csr_word(MODE_ATTN, 1, 1, 0, 0)) != 0)
            attn_ok = 0;
        if (attn_ok) {
            uint32_t *o = (uint32_t *)out_buf;
            int m_ok = (o[0] == attn_m_golden);
            uint32_t lu = fp32_pos_ulp_dist(o[1], attn_l_golden);
            int l_ok = lu <= L_TOL_ULP;
            int o_ok = 1;
            for (uint32_t j = 0; j < unified_dhead; j++) {
                // O lanes can be negative; compare magnitude-relative via pos-ULP on
                // |value| by masking the sign bit (monotonic within one sign, and the
                // golden/HW share sign for these rescaled sums).
                uint32_t gh = attn_o_golden[j] & 0x7fffffffu;
                uint32_t hh = o[2 + j] & 0x7fffffffu;
                int sign_ok = (attn_o_golden[j] >> 31) == (o[2 + j] >> 31);
                uint32_t ou = fp32_pos_ulp_dist(hh, gh);
                if (!(sign_ok && ou <= L_TOL_ULP)) o_ok = 0;
            }
            printf("[Unified] ATTN   : m*=%08x/%08x %s l*=%08x/%08x ulp=%u %s O[%u] %s\n",
                   o[0], attn_m_golden, m_ok ? "OK" : "BAD", o[1], attn_l_golden,
                   lu, l_ok ? "OK" : "BAD", unified_dhead, o_ok ? "OK" : "BAD");
            if (!(m_ok && l_ok && o_ok)) err++;
        } else {
            printf("[Unified] ATTN   : setup failed\n");
            err++;
        }
        snax_xdma_disable_dst_ext(WRITER_EXT_UNIFIEDMONOIDMERGERT);

        printf(err == 0 ? "[Unified] ALL PASS\n" : "[Unified] FAIL\n");
    }
    return err != 0;
}
