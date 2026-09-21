// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Does the cluster's xDMA reach MAIN MEMORY?
//
// On a single cluster it never could. An xDMA transfer is a conversation between
// two xDMA endpoints, and this platform's testbench had only one of them -- so the
// engine could memset its own L1 and transpose within it, and every byte that came
// from outside had to come on the iDMA. HeMAiA does not have that limitation
// because `hemaia_mem_system` carries a second endpoint sitting on the memory, and
// that is what lets its FlashAttention load K on the iDMA and V on the xDMA at the
// same time.
//
// The testbench now carries the same second endpoint, and this is the smallest
// program that proves it.
//
// EVERY WAIT HERE IS BOUNDED, and the control runs first. An xDMA that never
// answers presents as a slow simulation, not as a failure, so an unbounded wait
// turns a one-line bug report into an overnight run that says nothing. The four
// task counters are printed on timeout because they localise the break:
//
//   commit_local  did the local half accept the descriptor at all
//   commit_remote did the local half decide this transfer needs a peer
//   finish_local  did the local half's data movement complete
//   finish_remote did the peer report back
//
// A remote transfer that commits remotely and never finishes locally is a link
// that carries cfg but not data; one that never commits remotely is a local
// address decode that did not recognise the source as somebody else's.

#include <stdint.h>
#include "snax-core-roles.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

// 8 KiB, which is 128 wide beats -- enough for the reader to reach steady state
// rather than measuring only its fill.
#define NBYTES 65536u
#define NWORDS (NBYTES / 4u)
#define SPIN_LIMIT 400000u

// A non-const global lands in .data, which the linker places in main memory. That
// is the point: this array is NOT reachable from the cluster's TCDM, so if it
// arrives, it arrived over the link.
uint32_t src_pattern[NWORDS];

static void dump_counters(const char *what) {
    printf("  %s: commit_local %lu commit_remote %lu finish_local %lu "
           "finish_remote %lu\n",
           what,
           (unsigned long)snax_read_xdma_cfg_reg(XDMA_COMMIT_LOCAL_TASK_PTR),
           (unsigned long)snax_read_xdma_cfg_reg(XDMA_COMMIT_REMOTE_TASK_PTR),
           (unsigned long)snax_read_xdma_cfg_reg(XDMA_FINISH_LOCAL_TASK_PTR),
           (unsigned long)snax_read_xdma_cfg_reg(XDMA_FINISH_REMOTE_TASK_PTR));
}

// snax_xdma_start() spins forever waiting for a commit counter to move, so it is
// not usable when the question is whether the engine responds at all.
static int bounded_start(uint32_t *task, int *remote) {
    uint32_t l0 = snax_read_xdma_cfg_reg(XDMA_COMMIT_LOCAL_TASK_PTR);
    uint32_t r0 = snax_read_xdma_cfg_reg(XDMA_COMMIT_REMOTE_TASK_PTR);
    snax_write_xdma_cfg_reg(XDMA_START_PTR, 1);
    for (uint32_t s = 0; s < SPIN_LIMIT; s++) {
        uint32_t l = snax_read_xdma_cfg_reg(XDMA_COMMIT_LOCAL_TASK_PTR);
        uint32_t r = snax_read_xdma_cfg_reg(XDMA_COMMIT_REMOTE_TASK_PTR);
        if (l != l0) { *task = l; *remote = 0; return 0; }
        if (r != r0) { *task = r; *remote = 1; return 0; }
    }
    return 1;
}

// Poll the counter the hardware actually used. Waiting on LOCAL for a transfer the
// hardware ran REMOTE spins for ever -- and worse, it can PASS BY ACCIDENT when a
// preceding local transfer has already pushed the local counter past this id, so the
// wait returns without ever having checked the transfer it names.
static int bounded_wait(uint32_t task_id, int remote) {
    uint32_t ptr = remote ? XDMA_FINISH_REMOTE_TASK_PTR : XDMA_FINISH_LOCAL_TASK_PTR;
    for (uint32_t s = 0; s < SPIN_LIMIT; s++)
        if (snax_read_xdma_cfg_reg(ptr) >= task_id) return 0;
    return 1;
}

int main() {
    int err = 0;

    // Hart 0 fills the pattern through the ordinary narrow path, so the compare
    // below is against something this program wrote rather than against whatever
    // the loader happened to leave.
    if (snrt_cluster_core_idx() == 0) {
        for (uint32_t i = 0; i < NWORDS; i++) src_pattern[i] = 0xA5000000u + i;
    }
    snrt_cluster_hw_barrier();

    uint32_t *dst = (uint32_t *)snrt_l1_next();
    uint32_t *scratch = dst + NWORDS;

    if (snax_is_xdma_core()) {
        uint32_t stride[1] = {64u};
        uint32_t bound[1] = {NBYTES / 64u};
        uint32_t task_l; int task_is_remote;

        // ---- CONTROL: a purely local move of the same shape ------------------
        // If this fails, nothing about the new endpoint is implicated -- the
        // descriptor itself is wrong.
        for (uint32_t i = 0; i < NWORDS; i++) scratch[i] = 0xC0DE0000u + i;
        for (uint32_t i = 0; i < NWORDS; i++) dst[i] = 0u;
        if (snax_xdma_memcpy_nd_full_addr(
                (uint64_t)(uintptr_t)scratch, (uint64_t)(uintptr_t)dst, 8, 8, 1,
                stride, bound, 1, stride, bound, 0xff, 0xff, 0xffffffff) != 0) {
            printf("local control: agu configuration failed\n");
            return 1;
        }
        if (bounded_start(&task_l, &task_is_remote)) {
            printf("local control: xdma never committed the task\n");
            dump_counters("local");
            return 1;
        }
        if (bounded_wait(task_l, task_is_remote)) {
            printf("local control: xdma never finished\n");
            dump_counters("local");
            return 1;
        }
        uint32_t bad = 0;
        for (uint32_t i = 0; i < NWORDS; i++)
            if (dst[i] != 0xC0DE0000u + i) bad++;
        printf("local control   L1->L1 %lu B: %lu mismatches\n",
               (unsigned long)NBYTES, (unsigned long)bad);
        if (bad) err++;

        // ---- THE TEST: pull from main memory over the link --------------------
        for (uint32_t i = 0; i < NWORDS; i++) dst[i] = 0u;
        uint32_t t0 = snrt_mcycle();
        // The source is a full 48-bit main-memory address. The local xDMA sees it
        // is not local and forwards a cfg frame to the endpoint that owns it.
        if (snax_xdma_memcpy_nd_full_addr(
                (uint64_t)(uintptr_t)src_pattern, (uint64_t)(uintptr_t)dst, 8, 8,
                1, stride, bound, 1, stride, bound, 0xff, 0xff,
                0xffffffff) != 0) {
            printf("remote read: agu configuration failed\n");
            return 1;
        }
        if (bounded_start(&task_l, &task_is_remote)) {
            printf("remote read: xdma never committed the task\n");
            dump_counters("remote");
            return 1;
        }
        printf("remote read     committed: task %lu on the %s counter\n",
               (unsigned long)task_l, task_is_remote ? "REMOTE" : "local");
        if (!task_is_remote) {
            printf("  WARNING: a main-memory read committed LOCALLY -- the peer was "
                   "not engaged, so this is not testing the link\n");
            err++;
        }
        // The LOCAL half is the writer here: it finishes when the bytes land in L1.
        if (bounded_wait(task_l, task_is_remote)) {
            printf("remote read: TIMEOUT -- the peer never delivered\n");
            dump_counters("remote");
            return 1;
        }
        uint32_t t1 = snrt_mcycle();

        bad = 0;
        uint32_t first_bad = 0xffffffffu;
        for (uint32_t i = 0; i < NWORDS; i++) {
            if (dst[i] != 0xA5000000u + i) {
                if (!bad) first_bad = i;
                bad++;
            }
        }
        printf("remote read     %lu B in %lu cc (%lu.%lu B/cc)\n",
               (unsigned long)NBYTES, (unsigned long)(t1 - t0),
               (unsigned long)(NBYTES / (t1 - t0)),
               (unsigned long)((NBYTES * 10u / (t1 - t0)) % 10u));
        if (bad) {
            printf("  MISMATCH: %lu of %lu words, first at %lu: got %08lx "
                   "want %08lx\n",
                   (unsigned long)bad, (unsigned long)NWORDS,
                   (unsigned long)first_bad, (unsigned long)dst[first_bad],
                   (unsigned long)(0xA5000000u + first_bad));
            err++;
        } else {
            printf("  all %lu words match\n", (unsigned long)NWORDS);
        }
        dump_counters("final");
        printf("%s\n", err ? "FAIL" : "PASS");
    }

    snrt_cluster_hw_barrier();
    return err;
}
