// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// How much main-memory bandwidth reaches L1 when the iDMA and the xDMA read at the same time.
//
// The two engines reach main memory by different routes (snax-xdma-remote.h). The iDMA
// issues AXI reads: AR out of the cluster's wide port, the data back on R, into the TCDM
// through the wide DMA port. The xDMA asks the endpoint that sits on main memory, which
// pushes the bytes back as AXI writes: AW/W into the cluster's xDMA data window, into the
// TCDM through the xDMA writer's eight 64-bit ports. Each route moves at most one 512-bit
// beat a cycle. The app times each engine alone and both at once, over the same bytes:
//
//   alone   each engine by itself, at three sizes, for its rate and its fixed cost
//   both    each engine its own 128 KiB at once. The xDMA's destination is shifted 0, 64,
//           128 and 192 B against the iDMA's, one TCDM super bank at a time, because the
//           wide port takes a whole super bank per beat and wins it against narrow ports
//   split   one buffer, the first half on the iDMA and the second on the xDMA: what a
//           kernel's single load gains
//
// Every destination is checked word by word against the source, and the words on either
// side of it must stay zero. A last test runs 300 small reads in a row on the xDMA alone, each
// checked the moment its wait returns: the engine's task counters are 8 bits and wrap, and a
// wait that is not wrap-safe returns before the task that wraps has moved a byte.
//
// The testbench serves the two routes from two memory models that never contend
// (tb_memory_axi for the iDMA, tb_memory_tcdm under the endpoint). What this measures is the
// cluster's side: its wide link, its crossbar and its TCDM. On a real system both readers
// share the memory's banks (hemaia_mem_system arbitrates them per super bank).

#include <stdint.h>
#include "snax-core-roles.h"
#include "snax-perf-census.h"
#include "snax-xdma-lib.h"
#include "snax-xdma-remote.h"
#include "snrt.h"

#define KIB 1024u
#define BIG (128u * KIB)               // bytes per engine in the "both" tests
#define SRC_BYTES (2u * BIG)           // main memory: one half per engine
#define SRC_WORDS (SRC_BYTES / 4u)
#define DST_BYTES (2u * BIG + 256u)    // L1: both destinations and the largest shift
#define SPIN_LIMIT 200000u             // CSR polls, about 5 cycles each

// In main memory, and not in .bss, which the runtime clears word by word at boot. The source
// starts 320 KiB into a buffer that sits on a 1 MiB boundary, so the xDMA's reads cross the
// 512 KiB line at 0x8018_0000: the endpoint's 19-bit TCDM address wraps there, and the
// testbench has to carry it on into the next 512 KiB (tb_memory_tcdm).
static uint32_t dram_buf[(640u * KIB) / 4u]
    __attribute__((section(".dram"), aligned(1024u * KIB)));
#define src_buf (dram_buf + (320u * KIB) / 4u)

static inline uint32_t pattern(uint32_t word) { return 0xA5000000u + word; }

typedef struct {
    const char *what;
    uint32_t i_bytes, i_src, i_dst;  // iDMA: bytes, offset in src_buf, offset in dst
    uint32_t x_bytes, x_src, x_dst;  // xDMA: the same
} bw_test_t;

static const bw_test_t tests[] = {
    {"iDMA alone,  16 KiB", 16u * KIB, 0, 0, 0, 0, 0},
    {"iDMA alone,  64 KiB", 64u * KIB, 0, 0, 0, 0, 0},
    {"iDMA alone, 128 KiB", BIG, 0, 0, 0, 0, 0},
    {"xDMA alone,  16 KiB", 0, 0, 0, 16u * KIB, BIG, BIG},
    {"xDMA alone,  64 KiB", 0, 0, 0, 64u * KIB, BIG, BIG},
    {"xDMA alone, 128 KiB", 0, 0, 0, BIG, BIG, BIG},
    {"both, shift   0 B", BIG, 0, 0, BIG, BIG, BIG},
    {"both, shift  64 B", BIG, 0, 0, BIG, BIG, BIG + 64u},
    {"both, shift 128 B", BIG, 0, 0, BIG, BIG, BIG + 128u},
    {"both, shift 192 B", BIG, 0, 0, BIG, BIG, BIG + 192u},
    {"split  16 KiB", 8u * KIB, 0, 0, 8u * KIB, 8u * KIB, 8u * KIB},
    {"split  64 KiB", 32u * KIB, 0, 0, 32u * KIB, 32u * KIB, 32u * KIB},
    {"split 128 KiB", BIG / 2u, 0, 0, BIG / 2u, BIG / 2u, BIG / 2u},
};
#define NTESTS (sizeof(tests) / sizeof(tests[0]))

// Shared through L1: each engine's hart writes its own fields, hart 0 prints.
typedef struct {
    uint32_t i_t0, i_t1;           // iDMA issue and completion, mcycle
    uint32_t x_t0, x_t1;           // xDMA start and finish, mcycle
    uint32_t x_remote, x_timeout;  // which counter the xDMA used; a wait that ran out
    uint32_t x_task, x_rd, x_wr;   // the xDMA's own cycle counters for its task
    uint32_t bad[4];               // mismatched words, per checking hart
    uint32_t wrap_bad, wrap_to;    // the wrap test: reads not landed when their wait returned
} bw_result_t;

#define WRAP_READS 300u  // past the 8-bit task counters' 256

// Check a quarter of one destination against the source and clear it for the next test.
static uint32_t check_clear(uint32_t *d, uint32_t first_word, uint32_t words,
                            uint32_t part, uint32_t parts) {
    uint32_t bad = 0, q = words / parts;
    for (uint32_t w = part * q; w < (part + 1u) * q; w++) {
        if (d[w] != pattern(first_word + w)) bad++;
        d[w] = 0;
    }
    return bad;
}

// The words just outside a destination: an engine that wrote past its range leaves them
// non-zero.
static uint32_t check_guards(uint8_t *dst, uint32_t off, uint32_t bytes) {
    uint32_t bad = 0;
    if (off >= 4u && *(uint32_t *)(dst + off - 4u) != 0u) bad++;
    if (*(uint32_t *)(dst + off + bytes) != 0u) bad++;
    return bad;
}

// Tenths of a byte per cycle.
static inline uint32_t bpc10(uint32_t bytes, uint32_t cycles) {
    return cycles ? (bytes * 10u) / cycles : 0u;
}

int main() {
    uint32_t core = snrt_cluster_core_idx();
    uint32_t ncore = snrt_cluster_core_num();
    uint8_t *l1 = (uint8_t *)snrt_l1_next();
    volatile bw_result_t *res = (volatile bw_result_t *)l1;
    // 256-byte aligned: a shift of the xDMA's destination is then a shift in super banks.
    uint8_t *dst = (uint8_t *)(((uintptr_t)l1 + sizeof(bw_result_t) + 255u) & ~(uintptr_t)255u);
    int err = 0;

    // ---- the source: staged in L1 by all four harts, copied out by the iDMA -----------
    {
        uint32_t *stage = (uint32_t *)dst, q = SRC_WORDS / ncore;
        for (uint32_t w = core * q; w < (core + 1u) * q; w++) stage[w] = pattern(w);
        snrt_cluster_hw_barrier();
        if (snax_is_idma_core()) {
            snrt_dma_start_1d(src_buf, stage, SRC_BYTES);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
        q = (DST_BYTES / 4u + 1u + ncore - 1u) / ncore;  // the guard word past the end too
        for (uint32_t w = core * q; w < (core + 1u) * q && w < DST_BYTES / 4u + 1u; w++)
            stage[w] = 0u;
        if (core == 0) {
            for (uint32_t i = 0; i < sizeof(bw_result_t) / 4u; i++)
                ((volatile uint32_t *)res)[i] = 0u;
            printf("source 0x%08lx..0x%08lx in main memory, destinations 0x%08lx in L1\n",
                   (unsigned long)(uintptr_t)src_buf,
                   (unsigned long)((uintptr_t)src_buf + SRC_BYTES),
                   (unsigned long)(uintptr_t)dst);
            printf("%-20s %28s %28s %20s  %s\n", "test", "iDMA", "xDMA", "together",
                   "xDMA TCDM denied, wide beats");
        }
        snrt_cluster_hw_barrier();
    }

    // The yardstick: the iDMA alone, per size. A test with both engines is compared with the
    // iDMA moving the same total alone -- measured when that size was, else at its 128 KiB rate.
    uint32_t alone_bytes[3] = {0}, alone_cc[3] = {0}, nalone = 0, ref10 = 0;
    snax_perf_snapshot_t census;

    for (uint32_t k = 0; k < NTESTS; k++) {
        bw_test_t t = tests[k];  // out of main memory before the window opens

        // The xDMA's 33 CSR writes happen before the window: inside it the engine sees only
        // its start, as the iDMA sees only its two dmcpy instructions.
        if (snax_is_xdma_core() && t.x_bytes)
            snax_xdma_read_arm((uint32_t)(uintptr_t)(dst + t.x_dst),
                               (uint32_t)(uintptr_t)src_buf + t.x_src,
                               t.x_bytes / SNAX_XDMA_BEAT);
        if (core == 0) snax_perf_arm();
        snrt_cluster_hw_barrier();

        if (snax_is_idma_core() && t.i_bytes) {
            uint32_t t0 = snrt_mcycle();
            snrt_dma_start_1d(dst + t.i_dst, (uint8_t *)src_buf + t.i_src, t.i_bytes);
            snrt_dma_wait_all();
            uint32_t t1 = snrt_mcycle();
            res->i_t0 = t0;
            res->i_t1 = t1;
        }
        if (snax_is_xdma_core() && t.x_bytes) {
            int to = 0;
            uint32_t t0 = snrt_mcycle();
            snax_xdma_task_t tk = snax_xdma_start_bounded(SPIN_LIMIT, &to);
            if (!to) to = snax_xdma_wait_bounded(tk, SPIN_LIMIT);
            uint32_t t1 = snrt_mcycle();
            res->x_t0 = t0;
            res->x_t1 = t1;
            res->x_remote = tk.remote;
            res->x_timeout = (uint32_t)to;
            res->x_task = snax_read_xdma_cfg_reg(XDMA_PERF_CTR_TASK);
            res->x_rd = snax_read_xdma_cfg_reg(XDMA_PERF_CTR_READER);
            res->x_wr = snax_read_xdma_cfg_reg(XDMA_PERF_CTR_WRITER);
        }
        snrt_cluster_hw_barrier();
        if (core == 0) snax_perf_read(&census);

        // ---- check and clear, a quarter per hart ---------------------------------------
        uint32_t bad = 0;
        if (t.i_bytes)
            bad += check_clear((uint32_t *)(dst + t.i_dst), t.i_src / 4u, t.i_bytes / 4u,
                               core, ncore);
        if (t.x_bytes)
            bad += check_clear((uint32_t *)(dst + t.x_dst), t.x_src / 4u, t.x_bytes / 4u,
                               core, ncore);
        res->bad[core] = bad;
        snrt_cluster_hw_barrier();

        if (core == 0) {
            // Every hart has cleared its quarters, so the words around each range are zero
            // unless an engine wrote past it.
            uint32_t words_bad = res->bad[0] + res->bad[1] + res->bad[2] + res->bad[3];
            if (t.i_bytes) words_bad += check_guards(dst, t.i_dst, t.i_bytes);
            if (t.x_bytes) words_bad += check_guards(dst, t.x_dst, t.x_bytes);
            uint32_t ic = t.i_bytes ? res->i_t1 - res->i_t0 : 0u;
            uint32_t xc = t.x_bytes ? res->x_t1 - res->x_t0 : 0u;
            uint32_t t0 = t.i_bytes ? res->i_t0 : res->x_t0;
            uint32_t t1 = t.i_bytes ? res->i_t1 : res->x_t1;
            if (t.x_bytes && t.i_bytes) {
                if (res->x_t0 < t0) t0 = res->x_t0;
                if (res->x_t1 > t1) t1 = res->x_t1;
            }
            uint32_t wall = t1 - t0, total = t.i_bytes + t.x_bytes;
            uint32_t xreq = census.v[SNAX_PERF_C_XDMA_REQ];
            uint32_t xden = census.v[SNAX_PERF_C_XDMA_STALL];
            uint32_t xd = snax_perf_permille(xden, xreq);
            if (t.i_bytes && !t.x_bytes && nalone < 3u) {
                alone_bytes[nalone] = t.i_bytes;
                alone_cc[nalone++] = ic;
                if (t.i_bytes == BIG) ref10 = bpc10(t.i_bytes, ic);
            }
            printf("%-20s %6lu B %5lu cc %3lu.%lu B/cc %6lu B %5lu cc %3lu.%lu B/cc"
                   " %5lu cc %3lu.%lu B/cc  %2lu.%lu%% %6lu",
                   t.what, (unsigned long)t.i_bytes, (unsigned long)ic,
                   (unsigned long)(bpc10(t.i_bytes, ic) / 10u),
                   (unsigned long)(bpc10(t.i_bytes, ic) % 10u), (unsigned long)t.x_bytes,
                   (unsigned long)xc, (unsigned long)(bpc10(t.x_bytes, xc) / 10u),
                   (unsigned long)(bpc10(t.x_bytes, xc) % 10u), (unsigned long)wall,
                   (unsigned long)(bpc10(total, wall) / 10u),
                   (unsigned long)(bpc10(total, wall) % 10u), (unsigned long)(xd / 10u),
                   (unsigned long)(xd % 10u),
                   (unsigned long)census.v[SNAX_PERF_C_WIDE_REQ]);
            if (ref10 && t.i_bytes && t.x_bytes && wall) {
                uint32_t ref = (total * 10u) / ref10;
                for (uint32_t a = 0; a < nalone; a++)
                    if (alone_bytes[a] == total) ref = alone_cc[a];
                uint32_t x100 = ref * 100u / wall;
                printf("  %lu.%02lux", (unsigned long)(x100 / 100u),
                       (unsigned long)(x100 % 100u));
            }
            if (t.x_bytes && !res->x_remote) {
                printf("  LOCAL?");
                err++;
            }
            if (t.x_bytes && res->x_timeout) {
                printf("  TIMEOUT");
                err++;
            }
            if (words_bad) {
                printf("  %lu words WRONG", (unsigned long)words_bad);
                err++;
            }
            printf("\n");
            // Machine-readable: test, iDMA bytes and cycles, xDMA bytes and cycles, wall,
            // the xDMA's task/reader/writer counters, its TCDM requests and denials, wide
            // beats, wide pre-emptions, window cycles, wrong words.
            printf("BW %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu\n",
                   (unsigned long)k, (unsigned long)t.i_bytes, (unsigned long)ic,
                   (unsigned long)t.x_bytes, (unsigned long)xc, (unsigned long)wall,
                   (unsigned long)(t.x_bytes ? res->x_task : 0u),
                   (unsigned long)(t.x_bytes ? res->x_rd : 0u),
                   (unsigned long)(t.x_bytes ? res->x_wr : 0u), (unsigned long)xreq,
                   (unsigned long)xden, (unsigned long)census.v[SNAX_PERF_C_WIDE_REQ],
                   (unsigned long)census.v[SNAX_PERF_C_WIDE_PREEMPT],
                   (unsigned long)census.v[SNAX_PERF_C_CYCLE], (unsigned long)words_bad);
            res->x_remote = 0;
            res->x_timeout = 0;
        }
        snrt_cluster_hw_barrier();
    }

    // ---- the task counters wrap: WRAP_READS reads of 1 KiB, each checked at once ---------
    if (snax_is_xdma_core()) {
        volatile uint32_t *d = (volatile uint32_t *)dst;
        uint32_t wbad = 0, wto = 0;
        snax_xdma_read_arm((uint32_t)(uintptr_t)dst, (uint32_t)(uintptr_t)src_buf, 1024u / 64u);
        for (uint32_t i = 0; i < WRAP_READS; i++) {
            const uint32_t off = (i % 64u) * 1024u;  // a different source every time
            d[255] = 0u;                              // the read's last word
            snax_xdma_read_retask((uint32_t)(uintptr_t)dst, (uint32_t)(uintptr_t)src_buf + off,
                                  1024u / 64u);
            int to = 0;
            const snax_xdma_task_t tk = snax_xdma_start_bounded(SPIN_LIMIT, &to);
            if (!to) to = snax_xdma_wait_bounded(tk, SPIN_LIMIT);
            wto += (uint32_t)to;
            if (d[255] != pattern(off / 4u + 255u)) wbad++;
        }
        res->wrap_bad = wbad;
        res->wrap_to = wto;
    }
    snrt_cluster_hw_barrier();
    if (core == 0) {
        printf("%u reads of 1 KiB on the xDMA: %lu not landed when their wait returned, "
               "%lu timeouts\n", WRAP_READS, (unsigned long)res->wrap_bad,
               (unsigned long)res->wrap_to);
        if (res->wrap_bad || res->wrap_to) err++;
        printf("%s\n", err ? "FAIL" : "PASS");
    }
    return core == 0 ? err : 0;
}
