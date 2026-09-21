// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// A cluster contention census, to print next to a kernel's own timing.
//
// A kernel's engine counters say how long each engine was busy. They cannot say
// WHY an engine was not: an idle GEMM array looks the same whether its operands
// were late because the SIMD core was hammering the same banks, because the iDMA
// pre-empted a super bank, or because the kernel simply had nothing for it. The
// cluster performance counters answer that, and this is a fixed reading of them
// so two kernels, or one kernel before and after a change, can be compared line
// by line.
//
// THE COUNTERS ARE CLUSTER-GLOBAL and there are only 16, so the census is split
// into passes selected at compile time with -DSNAX_PERF_PASS=n. Pass 0 is the
// system census and is the one to read first. Passes 1 and up sweep the TCDM
// interconnect port by port, SNAX_PERF_PORTS_PER_PASS ports at a time, for when
// the system census says one engine is stalling and the question becomes which
// of its channels. Every pass prints its own cycle count, so a pass can be
// normalised against itself and cross-pass comparison never depends on two runs
// taking the same time.
//
// The window is armed and read by ONE hart around a barrier the whole cluster
// passes, so it covers every engine, including the ones whose cores are asleep.

#pragma once

#include "perf_cnt.h"
#include "snax-tcdm-ports-defs.h"

#ifndef SNAX_PERF_PASS
#define SNAX_PERF_PASS 0
#endif

// Eight ports per sweep pass: each port takes two counters, offered and denied,
// because a stall count alone cannot distinguish a channel that is contended
// from one that is merely busy.
#define SNAX_PERF_PORTS_PER_PASS 8
#define SNAX_PERF_SWEEP_PASSES                                     \
    ((SNAX_TCDM_NUM_PORTS + SNAX_PERF_PORTS_PER_PASS - 1) /        \
     SNAX_PERF_PORTS_PER_PASS)

typedef struct {
    uint32_t v[SNRT_PERF_N_CNT];
} snax_perf_snapshot_t;

// Counter assignment for pass 0. Named so the report cannot drift from the arm.
enum {
    SNAX_PERF_C_CYCLE = 0,
    SNAX_PERF_C_ALL_REQ,
    SNAX_PERF_C_ALL_STALL,
    SNAX_PERF_C_GEMM_REQ,
    SNAX_PERF_C_GEMM_STALL,
    SNAX_PERF_C_SIMD_REQ,
    SNAX_PERF_C_SIMD_STALL,
    SNAX_PERF_C_XDMA_REQ,
    SNAX_PERF_C_XDMA_STALL,
    SNAX_PERF_C_SNITCH_REQ,
    SNAX_PERF_C_SNITCH_STALL,
    SNAX_PERF_C_BANK_SERVED,
    SNAX_PERF_C_WIDE_PREEMPT,
    SNAX_PERF_C_WIDE_REQ,
    SNAX_PERF_C_L1_STALL,
    SNAX_PERF_C_L1_MISS,
};

static inline uint32_t snax_perf_first_port(void) {
    return (uint32_t)(SNAX_PERF_PASS - 1) * SNAX_PERF_PORTS_PER_PASS;
}

// Arm the census. Call from ONE hart immediately before the barrier that starts
// the measured window.
static inline void snax_perf_arm(void) {
    for (uint32_t i = 0; i < SNRT_PERF_N_CNT; i++)
        snrt_reset_perf_counter((enum snrt_perf_cnt)i);
#if SNAX_PERF_PASS == 0
    snrt_start_perf_counter(SNAX_PERF_C_CYCLE, SNRT_PERF_CNT_CYCLES, 0);
    snrt_start_perf_counter_tcdm_group(SNAX_PERF_C_ALL_REQ,
                                       SNRT_PERF_CNT_TCDM_GRP_REQ,
                                       SNAX_TCDM_GRP_ALL);
    snrt_start_perf_counter_tcdm_group(SNAX_PERF_C_ALL_STALL,
                                       SNRT_PERF_CNT_TCDM_GRP_STALL,
                                       SNAX_TCDM_GRP_ALL);
    snrt_start_perf_counter_tcdm_group(SNAX_PERF_C_GEMM_REQ,
                                       SNRT_PERF_CNT_TCDM_GRP_REQ,
                                       SNAX_TCDM_GRP_GEMM);
    snrt_start_perf_counter_tcdm_group(SNAX_PERF_C_GEMM_STALL,
                                       SNRT_PERF_CNT_TCDM_GRP_STALL,
                                       SNAX_TCDM_GRP_GEMM);
    snrt_start_perf_counter_tcdm_group(SNAX_PERF_C_SIMD_REQ,
                                       SNRT_PERF_CNT_TCDM_GRP_REQ,
                                       SNAX_TCDM_GRP_SIMD);
    snrt_start_perf_counter_tcdm_group(SNAX_PERF_C_SIMD_STALL,
                                       SNRT_PERF_CNT_TCDM_GRP_STALL,
                                       SNAX_TCDM_GRP_SIMD);
    snrt_start_perf_counter_tcdm_group(SNAX_PERF_C_XDMA_REQ,
                                       SNRT_PERF_CNT_TCDM_GRP_REQ,
                                       SNAX_TCDM_GRP_XDMA);
    snrt_start_perf_counter_tcdm_group(SNAX_PERF_C_XDMA_STALL,
                                       SNRT_PERF_CNT_TCDM_GRP_STALL,
                                       SNAX_TCDM_GRP_XDMA);
    snrt_start_perf_counter_tcdm_group(SNAX_PERF_C_SNITCH_REQ,
                                       SNRT_PERF_CNT_TCDM_GRP_REQ,
                                       SNAX_TCDM_GRP_SNITCH);
    snrt_start_perf_counter_tcdm_group(SNAX_PERF_C_SNITCH_STALL,
                                       SNRT_PERF_CNT_TCDM_GRP_STALL,
                                       SNAX_TCDM_GRP_SNITCH);
    snrt_start_perf_counter(SNAX_PERF_C_BANK_SERVED,
                            SNRT_PERF_CNT_TCDM_BANK_SERVED, 0);
    snrt_start_perf_counter(SNAX_PERF_C_WIDE_PREEMPT,
                            SNRT_PERF_CNT_TCDM_WIDE_PREEMPT, 0);
    snrt_start_perf_counter(SNAX_PERF_C_WIDE_REQ, SNRT_PERF_CNT_TCDM_WIDE_REQ, 0);
    snrt_start_perf_counter(SNAX_PERF_C_L1_STALL,
                            SNRT_PERF_CNT_ICACHE_L1_STALL, 0);
    snrt_start_perf_counter(SNAX_PERF_C_L1_MISS, SNRT_PERF_CNT_ICACHE_L1_MISS, 0);
#else
    // Sweep: counter 2i observes port `first+i`'s offered requests, 2i+1 its
    // denied ones. Ports past the end of the interconnect are left disabled
    // rather than wrapped, so the last pass is short instead of double-counting.
    uint32_t first = snax_perf_first_port();
    for (uint32_t i = 0; i < SNAX_PERF_PORTS_PER_PASS; i++) {
        if (first + i >= SNAX_TCDM_NUM_PORTS) break;
        snrt_start_perf_counter_tcdm_port((enum snrt_perf_cnt)(2 * i),
                                          SNRT_PERF_CNT_TCDM_PORT_REQ, first + i);
        snrt_start_perf_counter_tcdm_port((enum snrt_perf_cnt)(2 * i + 1),
                                          SNRT_PERF_CNT_TCDM_PORT_STALL,
                                          first + i);
    }
#endif
}

// Read the census. Call from ONE hart immediately after the barrier that ends
// the measured window, before anything that prints.
static inline void snax_perf_read(snax_perf_snapshot_t *s) {
    for (uint32_t i = 0; i < SNRT_PERF_N_CNT; i++) {
        s->v[i] = snrt_get_perf_counter((enum snrt_perf_cnt)i);
        snrt_stop_perf_counter((enum snrt_perf_cnt)i);
    }
}

// Tenths of a percent of `part` in `whole`, saturating rather than dividing by
// zero. Printed as u.u because the runtime has no float formatting.
//
// Kept strictly 32-bit: this is RV32 and the kernels link no libgcc, so a
// 64-bit divide is an undefined `__udivdi3` at link time rather than slow code.
// Scaling the numerator is exact but overflows past ~4.29M, which a port census
// over a long kernel reaches, so past that the denominator is scaled instead.
static inline uint32_t snax_perf_permille(uint32_t part, uint32_t whole) {
    if (!whole) return 0u;
    if (part < 4294967u) return (part * 1000u) / whole;
    uint32_t w = whole / 1000u;
    return w ? part / w : 1000u;
}

static inline const char* snax_perf_port_owner(uint32_t port) {
    static const snax_tcdm_channel_group_t g[] = SNAX_TCDM_CHANNEL_GROUPS_INIT;
    for (uint32_t i = 0; i < SNAX_TCDM_NUM_CHANNEL_GROUPS; i++)
        if (port >= g[i].base && port < (uint32_t)(g[i].base + g[i].count))
            return g[i].name;
    if (port >= SNAX_TCDM_SNITCH_BASE && port < SNAX_TCDM_SOC_PORT) return "snitch";
    if (port == SNAX_TCDM_SOC_PORT) return "soc";
    return "?";
}

#if SNAX_PERF_PASS == 0
// One engine's row of the census. `req` is what the engine's TCDM ports offered,
// `stall` what the interconnect refused; the difference, times the narrow port
// width, is the bandwidth it actually got.
static void snax_perf_engine_row(const char *name, uint32_t req, uint32_t stall,
                                 uint32_t cycles) {
    uint32_t grant = req - stall;
    uint32_t bpc10 = cycles ? (grant * SNAX_TCDM_NARROW_BYTES * 10u) / cycles : 0u;
    uint32_t sp = snax_perf_permille(stall, req);
    printf("    %-10s %9lu %9lu   %2lu.%lu%%   %3lu.%lu B/cc\n", name,
           (unsigned long)req, (unsigned long)stall, (unsigned long)(sp / 10u),
           (unsigned long)(sp % 10u), (unsigned long)(bpc10 / 10u),
           (unsigned long)(bpc10 % 10u));
}
#endif

// Print the census. `pipeline` is the kernel's own idea of the window, printed
// alongside the counter's own cycle count so a mismatch is visible rather than
// assumed away.
static void snax_perf_report(const snax_perf_snapshot_t *s, uint32_t pipeline) {
#if SNAX_PERF_PASS == 0
    uint32_t cyc = s->v[SNAX_PERF_C_CYCLE];
    uint32_t all_req = s->v[SNAX_PERF_C_ALL_REQ];
    uint32_t all_stall = s->v[SNAX_PERF_C_ALL_STALL];
    uint32_t served = s->v[SNAX_PERF_C_BANK_SERVED];
    uint32_t preempt = s->v[SNAX_PERF_C_WIDE_PREEMPT];
    uint32_t wide = s->v[SNAX_PERF_C_WIDE_REQ];
    uint32_t bank_cap = cyc * SNAX_TCDM_NUM_BANKS;

    printf("\n=== cluster contention census ===\n");
    printf("  window           %lu cc counted, kernel pipeline %lu cc\n",
           (unsigned long)cyc, (unsigned long)pipeline);
    printf("    engine          offered    denied   denied%%    granted\n");
    snax_perf_engine_row("gemm", s->v[SNAX_PERF_C_GEMM_REQ],
                         s->v[SNAX_PERF_C_GEMM_STALL], cyc);
    snax_perf_engine_row("simd", s->v[SNAX_PERF_C_SIMD_REQ],
                         s->v[SNAX_PERF_C_SIMD_STALL], cyc);
    snax_perf_engine_row("xdma", s->v[SNAX_PERF_C_XDMA_REQ],
                         s->v[SNAX_PERF_C_XDMA_STALL], cyc);
    snax_perf_engine_row("snitch", s->v[SNAX_PERF_C_SNITCH_REQ],
                         s->v[SNAX_PERF_C_SNITCH_STALL], cyc);
    snax_perf_engine_row("ALL", all_req, all_stall, cyc);

    // The banks are the shared resource everything above is competing for. A low
    // utilisation with a high denied% means the traffic is colliding on a few
    // banks rather than saturating them, which is a layout problem, not a
    // bandwidth one.
    printf("  banks            %lu of %lu bank-cycles served (%lu.%lu%% of %d banks)\n",
           (unsigned long)served, (unsigned long)bank_cap,
           (unsigned long)(snax_perf_permille(served, bank_cap) / 10u),
           (unsigned long)(snax_perf_permille(served, bank_cap) % 10u),
           SNAX_TCDM_NUM_BANKS);
    // The wide DMA port wins its super bank unconditionally, so this is the one
    // interference term that no amount of narrow-side arbitration can avoid.
    printf("  wide DMA         %lu beats = %lu B (%lu.%lu B/cc), preempting %lu "
           "bank-cycles (%lu.%lu%% of bank capacity)\n",
           (unsigned long)wide, (unsigned long)(wide * SNAX_TCDM_WIDE_BYTES),
           (unsigned long)(cyc ? (wide * SNAX_TCDM_WIDE_BYTES * 10u) / cyc / 10u : 0),
           (unsigned long)(cyc ? (wide * SNAX_TCDM_WIDE_BYTES * 10u) / cyc % 10u : 0),
           (unsigned long)preempt,
           (unsigned long)(snax_perf_permille(preempt, bank_cap) / 10u),
           (unsigned long)(snax_perf_permille(preempt, bank_cap) % 10u));
    printf("  icache L1        %lu misses, %lu stall cycles  (shared by all harts)\n",
           (unsigned long)s->v[SNAX_PERF_C_L1_MISS],
           (unsigned long)s->v[SNAX_PERF_C_L1_STALL]);
    printf("  CENSUS pass 0 %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu "
           "%lu %lu %lu\n",
           (unsigned long)s->v[0], (unsigned long)s->v[1], (unsigned long)s->v[2],
           (unsigned long)s->v[3], (unsigned long)s->v[4], (unsigned long)s->v[5],
           (unsigned long)s->v[6], (unsigned long)s->v[7], (unsigned long)s->v[8],
           (unsigned long)s->v[9], (unsigned long)s->v[10],
           (unsigned long)s->v[11], (unsigned long)s->v[12],
           (unsigned long)s->v[13], (unsigned long)s->v[14],
           (unsigned long)s->v[15]);
#else
    uint32_t first = snax_perf_first_port();
    printf("\n=== TCDM port census, ports %lu..%lu (pass %d of %d) ===\n",
           (unsigned long)first,
           (unsigned long)(first + SNAX_PERF_PORTS_PER_PASS - 1),
           SNAX_PERF_PASS, SNAX_PERF_SWEEP_PASSES);
    printf("  kernel pipeline %lu cc\n", (unsigned long)pipeline);
    printf("    port  owner            offered     denied   denied%%\n");
    for (uint32_t i = 0; i < SNAX_PERF_PORTS_PER_PASS; i++) {
        uint32_t port = first + i;
        if (port >= SNAX_TCDM_NUM_PORTS) break;
        uint32_t req = s->v[2 * i], stall = s->v[2 * i + 1];
        uint32_t sp = snax_perf_permille(stall, req);
        printf("    %4lu  %-14s %9lu %10lu   %2lu.%lu%%\n", (unsigned long)port,
               snax_perf_port_owner(port), (unsigned long)req,
               (unsigned long)stall, (unsigned long)(sp / 10u),
               (unsigned long)(sp % 10u));
        printf("  PORTCENSUS %lu %lu %lu\n", (unsigned long)port,
               (unsigned long)req, (unsigned long)stall);
    }
#endif
}
