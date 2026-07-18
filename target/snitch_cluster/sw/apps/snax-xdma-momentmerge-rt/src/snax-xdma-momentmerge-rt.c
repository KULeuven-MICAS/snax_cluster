// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// F3 S0a -- local-loopback smoke test of StreamMomentMergeRt (the in-transit
// nonlinear online-softmax moment-merge collective) through the REAL SW CSR
// path (snax_xdma_enable_dst_ext), before any cross-cluster plumbing is
// involved. A single cluster folds a pre-packed beat of (m_i, l_i) pairs into
// itself: TCDM -> xDMA reader -> [StreamMomentMergeRt, writer-side] -> TCDM.
// Two cases: a FULL beat (nValid=8, no masking) and a SHORT beat (nValid=3,
// exercises the identity-lane masking for lanes >= nValid).
//
// No FPU on the DM core, so both checks are pure-integer:
//   m* is a max of already-materialized FP32 values -> bit-exact vs golden.
//   l* involves the HW's LUT-based exp -> compare as an unsigned-integer ULP
//   distance (valid since l* is always positive, so its FP32 bit pattern
//   reinterpreted as int32 is monotonic in value). ~1.5e-2 relative error is
//   the same exp-LUT-depth-128 bound the Chisel testers use; in ULP that's
//   0.015 * 2^23 ~= 125829, rounded up to 130000 for margin.

#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#if !defined(WRITER_EXT_STREAMMOMENTMERGERT)
#error \
    "Regenerate the XDMA CSR map: this app needs StreamMomentMergeRt (cfg/snax_xdma_test.hjson)."
#endif

#define XDMA_BEAT_BYTES 64
#define L_TOL_ULP 130000u

// Positive-FP32 ULP distance (integer-only, no FPU): valid because l* is
// always >= 0, so its bit pattern reinterpreted as int32 is monotonic.
static uint32_t fp32_pos_ulp_dist(uint32_t a, uint32_t b) {
    int32_t ai = (int32_t)a, bi = (int32_t)b;
    return (ai > bi) ? (uint32_t)(ai - bi) : (uint32_t)(bi - ai);
}

// Run one merge beat through the local loopback: TCDM in_buf --[xDMA writer,
// StreamMomentMergeRt armed with nvalid]--> TCDM out_buf. Returns HW cycle
// count for the task, or 0xFFFFFFFF on setup failure.
static uint32_t run_merge_beat(uint8_t *in_buf, uint8_t *out_buf,
                               uint32_t nvalid) {
    uint32_t csr[1] = {nvalid};
    if (snax_xdma_enable_dst_ext(WRITER_EXT_STREAMMOMENTMERGERT, csr) != 0)
        return 0xFFFFFFFFu;
    if (snax_xdma_memcpy_1d(in_buf, out_buf, XDMA_BEAT_BYTES) != 0)
        return 0xFFFFFFFFu;
    int task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    uint32_t cycles = snax_xdma_last_task_cycle();
    snax_xdma_disable_dst_ext(WRITER_EXT_STREAMMOMENTMERGERT);
    return cycles;
}

// Load one beat from L2 (data.h array) into TCDM, run the merge, check
// (m*, l*) against the golden. Returns 1 on PASS, 0 on FAIL.
static int check_one_beat(const char *tag, uint8_t *base,
                          const uint32_t *beat_in_l2, uint32_t nvalid,
                          uint32_t m_gold, uint32_t l_gold) {
    uint8_t *in_buf = base;
    uint8_t *out_buf = base + XDMA_BEAT_BYTES;

    snrt_dma_start_1d(in_buf, beat_in_l2, XDMA_BEAT_BYTES);
    snrt_dma_wait_all();

    uint32_t cycles = run_merge_beat(in_buf, out_buf, nvalid);
    if (cycles == 0xFFFFFFFFu) {
        printf("[MomentMergeRt] %s: xDMA task setup failed\n", tag);
        return 0;
    }

    uint32_t *out_u32 = (uint32_t *)out_buf;
    uint32_t m_hw = out_u32[0];  // low 32b of the splatted pair = m*
    uint32_t l_hw = out_u32[1];  // next 32b = l*

    int m_ok = (m_hw == m_gold);
    uint32_t l_ulp = fp32_pos_ulp_dist(l_hw, l_gold);
    int l_ok = (l_ulp <= L_TOL_ULP);

    printf("[MomentMergeRt] %s: nValid=%u cycles=%u | m* got=%08x gold=%08x %s | "
           "l* got=%08x gold=%08x ulp=%u %s\n",
           tag, nvalid, cycles, m_hw, m_gold, m_ok ? "PASS" : "FAIL", l_hw,
           l_gold, l_ulp, l_ok ? "PASS" : "FAIL");
    return m_ok && l_ok;
}

int main() {
    int err = 0;
    if (snrt_is_dm_core()) {
        uint8_t *base = (uint8_t *)snrt_cluster_base_addrl();
        printf("[MomentMergeRt] F3 S0a: local-loopback smoke test\n");

        if (!check_one_beat("full(nValid=8)", base, mmerge_beat8_in,
                            mmerge_beat8_nvalid, mmerge_beat8_m_golden,
                            mmerge_beat8_l_golden))
            err++;

        if (!check_one_beat("short(nValid=3)", base, mmerge_beat3_in,
                            mmerge_beat3_nvalid, mmerge_beat3_m_golden,
                            mmerge_beat3_l_golden))
            err++;

        printf(err == 0 ? "[MomentMergeRt] ALL PASS\n" : "[MomentMergeRt] FAIL\n");
    }
    return err != 0;
}
