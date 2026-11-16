// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// What every snitch-forge program shares. A program runs four sections, each
// opened by a barrier of every core (sf_section):
//
//   1 dma_in   the DMA core moves the inputs from L2 to L1
//   2 run      the workers compute; a fold adds a barrier and a combine
//   3 dma_out  the DMA core moves the outputs from L1 to L2
//   4 check    core 0 counts the outputs that differ from their golden
//
// The workers are the compute cores, or the one core of a one-core cluster,
// which then moves the data too. Of P workers, worker c takes the elements in
// the 64-bit words c, c + P, c + 2P, ... of the inputs (SF_PER_WORD each), and
// the outputs at the same indices, so the workers start in different banks.
// Only core 0 drives obs_o, so the QUIET build has core 0 write each
// section's mark to CSR 0x7c5 (obs_marks.json) right after the barrier that
// opens it, also for the DMA sections it does not run, and prints nothing.
// The default build prints the cycles of each section and of each worker's
// compute instead. data.h gives in_t, acc_t, the SF_ sizes and the arrays.

#pragma once

#include "snrt.h"

#include "data.h"

// The sections' marks, as obs_marks.json lists them
enum { SF_DMA_IN = 1, SF_RUN = 2, SF_DMA_OUT = 3, SF_CHECK = 4 };

// in_t elements per 64-bit L1 word: the unit the workers share out
#define SF_PER_WORD (8 / sizeof(in_t))

#ifdef QUIET
#define SF_PRINTF(...)              \
    do {                            \
        if (0) printf(__VA_ARGS__); \
    } while (0)
#else
#define SF_PRINTF(...) printf(__VA_ARGS__)
#endif

typedef struct {
    uint32_t workers;  // P: the compute cores, or 1 on a one-core cluster
    uint32_t core;     // this core's index in the cluster
    uintptr_t next;    // the next free L1 byte, on a 64-byte line
    uint32_t *stamp;   // L1: core 0's mcycle as sections 1 to 4 open
    uint32_t *cycles;  // L1: each worker's compute cycles
} sf_t;

// The next `bytes` of L1, on a 64-byte line. Every core takes the same blocks
// in the same order, so every core gets the same addresses.
static inline void *sf_l1(sf_t *sf, size_t bytes) {
    void *p = (void *)sf->next;
    sf->next = (sf->next + bytes + 63) & ~(uintptr_t)63;
    return p;
}

static inline sf_t sf_init(void) {
    sf_t sf;
    uint32_t p = snrt_cluster_compute_core_num();
    sf.workers = p ? p : 1;
    sf.core = snrt_cluster_core_idx();
    sf.next = ((uintptr_t)snrt_l1_next() + 63) & ~(uintptr_t)63;
    sf.stamp = (uint32_t *)sf_l1(&sf, SF_CHECK * sizeof(uint32_t));
    sf.cycles = (uint32_t *)sf_l1(&sf, sf.workers * sizeof(uint32_t));
    return sf;
}

static inline int sf_is_worker(const sf_t *sf) {
    return sf->core < sf->workers;
}

// Open a section: a barrier of every core, then core 0's mark (QUIET) or
// mcycle (default)
static inline void sf_section(sf_t *sf, uint32_t mark) {
    snrt_cluster_hw_barrier();
    if (sf->core == 0) {
#ifdef QUIET
        write_csr_obs(mark);
#else
        sf->stamp[mark - 1] = snrt_mcycle();
#endif
    }
}

// A worker's compute cycles, from sf_tic to sf_toc (default build only)
static inline uint32_t sf_tic(void) {
#ifdef QUIET
    return 0;
#else
    return snrt_mcycle();
#endif
}

static inline void sf_toc(sf_t *sf, uint32_t t0) {
#ifdef QUIET
    (void)sf;
    (void)t0;
#else
    sf->cycles[sf->core] = snrt_mcycle() - t0;
#endif
}

// The outputs in L2 (written by the DMA) that differ from their golden
static inline int sf_check(const acc_t *got, const acc_t *golden, uint32_t n) {
    int errors = 0;
    uint32_t first = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (((volatile const acc_t *)got)[i] != golden[i]) {
            if (errors == 0) first = i;
            errors++;
        }
    }
    if (errors) SF_PRINTF("snitch-forge first error at %u\n", first);
    return errors;
}

// The default build's lines: the point, each section's and each worker's
// cycles, the errors. Core 0 only, in the check section.
static inline void sf_report(const sf_t *sf, const char *kernel, int errors) {
#ifdef QUIET
    (void)sf;
    (void)kernel;
    (void)errors;
#else
    printf("snitch-forge %s %s %s workers %u\n", kernel, SF_TYPE, SF_SIZE,
           sf->workers);
    printf("snitch-forge cycles dma_in: %u\n", sf->stamp[1] - sf->stamp[0]);
    printf("snitch-forge cycles run: %u\n", sf->stamp[2] - sf->stamp[1]);
    printf("snitch-forge cycles dma_out: %u\n", sf->stamp[3] - sf->stamp[2]);
    for (uint32_t c = 0; c < sf->workers; c++)
        printf("snitch-forge core %u compute: %u\n", c, sf->cycles[c]);
    printf("snitch-forge errors: %d\n", errors);
#endif
}
