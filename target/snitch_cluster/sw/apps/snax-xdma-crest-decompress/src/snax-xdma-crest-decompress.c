// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Throughput of CrestDecompressor, the xDMA writer extension that expands
// CREST-compressed weights losslessly on their way into L1: INT4 / FP4
// nibbles, INT8 / FP8 bytes or BF16, as the stream header says
// (hw/chisel/doc/crest_decompressor.md).
//
// Per case (data/params.hjson), the iDMA stages the compressed words and the
// expected weights in L1. The xDMA hart then times three L1 -> L1 transfers
// with the engine's own task counter:
//
//   decompress  W compressed words in, N beats out through the extension
//   copy N      a plain copy of the N beats (the weights uncompressed)
//   copy W      a plain copy of the W words (only the compressed bytes)
//
// The decompressor keeps up when "decompress" is as fast as "copy N": the
// writer, which takes one 512-bit beat a cycle, is then the limit, and the
// reader needs only W/N of its cycles. "cfg" is the xDMA hart's cycles to
// configure the decompress task, "run" its cycles from start to done. Every
// output word is checked against the weights, by all harts, and the beat after
// the output must still hold its sentinel.

#include <stdint.h>
#include "data.h"
#include "snax-core-roles.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

#define BEAT 64u
#define MAX_CORES 8u
#define SENTINEL 0x5A5A0000u

typedef struct {
    // decompress: the engine's task, reader and writer cycles
    uint32_t dec_task, dec_rd, dec_wr;
    // decompress on the xDMA hart: configuration, then start to done
    uint32_t dec_cfg, dec_run;
    uint32_t copy_n, copy_w;  // the two plain copies, task cycles
    uint32_t cfg_err;  // the helper refused (no extension on this cluster)
    uint32_t bad[MAX_CORES];  // mismatched output words, per checking hart
    uint32_t first_bad;       // lowest mismatching word index + 1, 0 if none
    uint32_t guard_bad;       // the beat past the output was written
} crest_result_t;

static inline uintptr_t align_up(uintptr_t a, uintptr_t to) {
    return (a + to - 1u) & ~(to - 1u);
}

static uint32_t run_task(void) {
    snax_xdma_task_t t = snax_xdma_start_task();
    snax_xdma_wait_task(t);
    return snax_xdma_last_task_cycle();
}

// Thousandths, for printing ratios with integer printf.
static inline uint32_t milli(uint32_t num, uint32_t den) {
    return den ? (num * 1000u) / den : 0u;
}

int main() {
    uint32_t core = snrt_cluster_core_idx();
    uint32_t ncore = snrt_cluster_core_num();
    if (ncore > MAX_CORES) ncore = MAX_CORES;

    uint8_t* l1 = (uint8_t*)snrt_l1_next();
    volatile crest_result_t* res = (volatile crest_result_t*)l1;
    uint32_t* in_buf =
        (uint32_t*)align_up((uintptr_t)l1 + sizeof(crest_result_t), 4096u);
    uint32_t* out_buf = (uint32_t*)align_up(
        (uintptr_t)in_buf + CREST_MAX_IN_WORDS * BEAT, 4096u);
    uint32_t* exp_buf = (uint32_t*)align_up(
        (uintptr_t)out_buf + (CREST_MAX_OUT_BEATS + 1u) * BEAT, 4096u);
    int err = 0;

    if (core == 0) {
        printf(
            "Weight decompressor: L1 in 0x%08lx, out 0x%08lx, expected "
            "0x%08lx\n",
            (unsigned long)(uintptr_t)in_buf, (unsigned long)(uintptr_t)out_buf,
            (unsigned long)(uintptr_t)exp_buf);
        printf(
            "%-17s %4s %5s %6s %6s %6s | %8s %6s %6s %6s %6s %7s | %7s %7s | "
            "%s\n",
            "case", "mode", "beats", "words", "ratio", "esc", "decomp", "read",
            "write", "cfg", "run", "beat/cc", "copy N", "copy W", "check");
    }

    for (uint32_t c = 0; c < CREST_NUM_CASES; c++) {
        const crest_case_t* cs = &crest_cases[c];
        uint32_t out_words = cs->out_beats * (BEAT / 4u);

        // ---- stage the case in L1 ----
        if (snax_is_idma_core()) {
            snrt_dma_start_1d(in_buf, cs->in, cs->in_words * BEAT);
            snrt_dma_start_1d(exp_buf, cs->out, cs->out_beats * BEAT);
            snrt_dma_wait_all();
        }
        if (core == 0) {
            for (uint32_t i = 0; i < sizeof(crest_result_t) / 4u; i++)
                ((volatile uint32_t*)res)[i] = 0u;
            for (uint32_t i = 0; i < BEAT / 4u; i++)
                out_buf[out_words + i] = SENTINEL + i;
        }
        snrt_cluster_hw_barrier();

        // ---- decompress, then the two plain copies for comparison ----
        if (snax_is_xdma_core()) {
            uint32_t t0 = snrt_mcycle();
            if (snax_xdma_crest_decompress(in_buf, cs->in_words, out_buf,
                                           cs->out_beats) != 0) {
                res->cfg_err = 1;
            } else {
                uint32_t t1 = snrt_mcycle();
                res->dec_task = run_task();
                res->dec_run = snrt_mcycle() - t1;
                res->dec_cfg = t1 - t0;
                res->dec_rd = snax_xdma_last_read_cycle();
                res->dec_wr = snax_xdma_last_write_cycle();
            }
#ifdef WRITER_EXT_CRESTDECOMPRESSOR
            snax_xdma_disable_dst_ext(WRITER_EXT_CRESTDECOMPRESSOR);
#endif
        }
        snrt_cluster_hw_barrier();

        // ---- check: every output word, split over the harts, and the guard
        // beat ----
        {
            uint32_t q = (out_words + ncore - 1u) / ncore, lo = core * q;
            uint32_t hi = lo + q < out_words ? lo + q : out_words, bad = 0,
                     first = 0;
            for (uint32_t w = lo; w < hi; w++) {
                if (out_buf[w] != exp_buf[w]) {
                    if (!bad) first = w + 1u;
                    bad++;
                }
            }
            if (core < MAX_CORES) res->bad[core] = bad;
            if (bad && (res->first_bad == 0 || first < res->first_bad))
                res->first_bad = first;
            if (core == 0) {
                for (uint32_t i = 0; i < BEAT / 4u; i++)
                    if (out_buf[out_words + i] != SENTINEL + i)
                        res->guard_bad++;
            }
        }
        snrt_cluster_hw_barrier();

        if (snax_is_xdma_core() && !res->cfg_err) {
            snax_xdma_memcpy_1d(out_buf, exp_buf, cs->out_beats * BEAT);
            res->copy_n = run_task();
            snax_xdma_memcpy_1d(in_buf, exp_buf, cs->in_words * BEAT);
            res->copy_w = run_task();
        }
        snrt_cluster_hw_barrier();

        if (core == 0) {
            uint32_t bad = 0;
            for (uint32_t i = 0; i < ncore; i++) bad += res->bad[i];
            int ok = !res->cfg_err && bad == 0 && res->guard_bad == 0;
            uint32_t r = milli(cs->out_beats, cs->in_words);
            uint32_t bpc = milli(cs->out_beats, res->dec_task);
            printf(
                "%-17s %4s %5lu %6lu %2lu.%03lu %6lu | %8lu %6lu %6lu %6lu "
                "%6lu "
                "%3lu.%03lu | %7lu %7lu | %s\n",
                cs->name, cs->mode, (unsigned long)cs->out_beats,
                (unsigned long)cs->in_words, (unsigned long)(r / 1000u),
                (unsigned long)(r % 1000u),
                (unsigned long)(cs->tier1 + cs->tier2),
                (unsigned long)res->dec_task, (unsigned long)res->dec_rd,
                (unsigned long)res->dec_wr, (unsigned long)res->dec_cfg,
                (unsigned long)res->dec_run, (unsigned long)(bpc / 1000u),
                (unsigned long)(bpc % 1000u), (unsigned long)res->copy_n,
                (unsigned long)res->copy_w, ok ? "ok" : "FAIL");
            if (!ok) {
                if (res->cfg_err)
                    printf("  no weight decompressor on this cluster\n");
                if (bad)
                    printf(
                        "  %lu words differ, the first at word %lu (beat "
                        "%lu)\n",
                        (unsigned long)bad,
                        (unsigned long)(res->first_bad - 1u),
                        (unsigned long)((res->first_bad - 1u) / (BEAT / 4u)));
                if (res->guard_bad)
                    printf("  the beat after the output was overwritten\n");
                err++;
            }
        }
        snrt_cluster_hw_barrier();
    }

    if (core == 0) printf("Weight decompressor: %s\n", err ? "FAIL" : "PASS");
    return err;
}
