// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Guarded because this is no longer reached only through snrt.h: a kernel that
// programs the cluster contention counters includes it directly as well.
#pragma once

/// Different perf counters
// Must match with `snitch_cluster_peripheral`
enum snrt_perf_cnt {
    SNRT_PERF_CNT0,
    SNRT_PERF_CNT1,
    SNRT_PERF_CNT2,
    SNRT_PERF_CNT3,
    SNRT_PERF_CNT4,
    SNRT_PERF_CNT5,
    SNRT_PERF_CNT6,
    SNRT_PERF_CNT7,
    SNRT_PERF_CNT8,
    SNRT_PERF_CNT9,
    SNRT_PERF_CNT10,
    SNRT_PERF_CNT11,
    SNRT_PERF_CNT12,
    SNRT_PERF_CNT13,
    SNRT_PERF_CNT14,
    SNRT_PERF_CNT15,
    SNRT_PERF_N_CNT,
};

/// Different types of performance counters
enum snrt_perf_cnt_type {
    SNRT_PERF_CNT_CYCLES,
    SNRT_PERF_CNT_TCDM_ACCESSED,
    SNRT_PERF_CNT_TCDM_CONGESTED,
    SNRT_PERF_CNT_ISSUE_FPU,
    SNRT_PERF_CNT_ISSUE_FPU_SEQ,
    SNRT_PERF_CNT_ISSUE_CORE_TO_FPU,
    SNRT_PERF_CNT_RETIRED_INSTR,
    SNRT_PERF_CNT_RETIRED_LOAD,
    SNRT_PERF_CNT_RETIRED_I,
    SNRT_PERF_CNT_RETIRED_ACC,
    SNRT_PERF_CNT_DMA_AW_STALL,
    SNRT_PERF_CNT_DMA_AR_STALL,
    SNRT_PERF_CNT_DMA_R_STALL,
    SNRT_PERF_CNT_DMA_W_STALL,
    SNRT_PERF_CNT_DMA_BUF_W_STALL,
    SNRT_PERF_CNT_DMA_BUF_R_STALL,
    SNRT_PERF_CNT_DMA_AW_DONE,
    SNRT_PERF_CNT_DMA_AW_BW,
    SNRT_PERF_CNT_DMA_AR_DONE,
    SNRT_PERF_CNT_DMA_AR_BW,
    SNRT_PERF_CNT_DMA_R_DONE,
    SNRT_PERF_CNT_DMA_R_BW,
    SNRT_PERF_CNT_DMA_W_DONE,
    SNRT_PERF_CNT_DMA_W_BW,
    SNRT_PERF_CNT_DMA_B_DONE,
    SNRT_PERF_CNT_DMA_BUSY,
    SNRT_PERF_CNT_ICACHE_MISS,
    SNRT_PERF_CNT_ICACHE_HIT,
    SNRT_PERF_CNT_ICACHE_PREFETCH,
    SNRT_PERF_CNT_ICACHE_DOUBLE_HIT,
    SNRT_PERF_CNT_ICACHE_STALL,
    /// Events from here on live in `PERF_COUNTER_ENABLE_EXT`, not in
    /// `PERF_COUNTER_ENABLE`. `snrt_start_perf_counter` picks the right register;
    /// nothing else in this API needs to know which side an event is on.
    SNRT_PERF_CNT_EXT_BASE = 32,
    /// TCDM requests offered by the port group in `PORT_GROUP`, granted or not.
    SNRT_PERF_CNT_TCDM_GRP_REQ = SNRT_PERF_CNT_EXT_BASE + 0,
    /// ... of which were refused. Granted = REQ - STALL; at 8 byte per narrow port
    /// that is the group's achieved TCDM bandwidth.
    SNRT_PERF_CNT_TCDM_GRP_STALL = SNRT_PERF_CNT_EXT_BASE + 1,
    /// The same two, for the single interconnect port in `PORT_INDEX`.
    SNRT_PERF_CNT_TCDM_PORT_REQ = SNRT_PERF_CNT_EXT_BASE + 2,
    SNRT_PERF_CNT_TCDM_PORT_STALL = SNRT_PERF_CNT_EXT_BASE + 3,
    /// TCDM banks that accepted a narrow request. Over `CYCLE * NrBanks` this is
    /// the bank utilisation every engine is competing for.
    SNRT_PERF_CNT_TCDM_BANK_SERVED = SNRT_PERF_CNT_EXT_BASE + 4,
    /// TCDM banks that had a narrow request pending while the wide DMA port took
    /// their super bank. Bank-cycles the DMA took from the compute engines.
    SNRT_PERF_CNT_TCDM_WIDE_PREEMPT = SNRT_PERF_CNT_EXT_BASE + 5,
    /// Wide DMA port TCDM accesses, and accesses it offered but lost.
    SNRT_PERF_CNT_TCDM_WIDE_REQ = SNRT_PERF_CNT_EXT_BASE + 6,
    SNRT_PERF_CNT_TCDM_WIDE_STALL = SNRT_PERF_CNT_EXT_BASE + 7,
    /// The L1 instruction cache shared by the harts of a hive. The plain
    /// `SNRT_PERF_CNT_ICACHE_*` events above are the hart-private L0 buffer.
    SNRT_PERF_CNT_ICACHE_L1_MISS = SNRT_PERF_CNT_EXT_BASE + 8,
    SNRT_PERF_CNT_ICACHE_L1_HIT = SNRT_PERF_CNT_EXT_BASE + 9,
    SNRT_PERF_CNT_ICACHE_L1_STALL = SNRT_PERF_CNT_EXT_BASE + 10,
    SNRT_PERF_CNT_ICACHE_L1_HANDLER_STALL = SNRT_PERF_CNT_EXT_BASE + 11,
};

/// Port groups `SNRT_PERF_CNT_TCDM_GRP_*` can aggregate over. Group 0 is every
/// input of the narrow TCDM interconnect; groups 1..NrCores are the SNAX ports
/// owned by core 0..NrCores-1; then all Snitch core data ports, then the narrow
/// SoC port. The per-core groups are what make this useful: on a split cluster
/// they are exactly the GEMM, SIMD and xDMA engines.
#define SNRT_PERF_TCDM_GRP_ALL 0u
#define SNRT_PERF_TCDM_GRP_SNAX_CORE(core) (1u + (core))
#define SNRT_PERF_TCDM_GRP_SNITCH_CORES(nr_cores) ((nr_cores) + 1u)
#define SNRT_PERF_TCDM_GRP_SOC(nr_cores) ((nr_cores) + 2u)

typedef union {
    uint32_t value __attribute__((aligned(8)));
} perf_reg32_t;

typedef struct {
    volatile perf_reg32_t enable[SNRT_PERF_N_CNT];
    volatile perf_reg32_t hart_select[SNRT_PERF_N_CNT];
    volatile perf_reg32_t perf_counter[SNRT_PERF_N_CNT];
    volatile perf_reg32_t cl_clint_set;
    volatile perf_reg32_t cl_clint_clear;
    volatile perf_reg32_t hw_barrier;
    volatile perf_reg32_t icache_prefetch_enable;
    volatile perf_reg32_t enable_ext[SNRT_PERF_N_CNT];
} perf_regs_t;

/// `PORT_INDEX` and `PORT_GROUP` share the `HART_SELECT` word with the hart index.
#define SNRT_PERF_PORT_INDEX_SHIFT 10u
#define SNRT_PERF_PORT_GROUP_SHIFT 20u

inline perf_regs_t* snrt_perf_counters() {
    return (perf_regs_t*)snrt_cluster_perf_counters_addr();
}

// Enable a specific perf_counter. Events at or above SNRT_PERF_CNT_EXT_BASE live
// in the extended enable register; the hardware scans the base register first, so
// exactly one of the two must be non-zero for a given counter.
inline void snrt_start_perf_counter(enum snrt_perf_cnt perf_cnt,
                                    enum snrt_perf_cnt_type perf_cnt_type,
                                    uint32_t hart_id) {
    snrt_perf_counters()->hart_select[perf_cnt].value |= hart_id;
    if ((uint32_t)perf_cnt_type >= (uint32_t)SNRT_PERF_CNT_EXT_BASE) {
        snrt_perf_counters()->enable[perf_cnt].value = 0x0;
        snrt_perf_counters()->enable_ext[perf_cnt].value =
            (0x1u << ((uint32_t)perf_cnt_type -
                      (uint32_t)SNRT_PERF_CNT_EXT_BASE));
    } else {
        snrt_perf_counters()->enable_ext[perf_cnt].value = 0x0;
        snrt_perf_counters()->enable[perf_cnt].value = (0x1u << perf_cnt_type);
    }
}

// Enable a TCDM port-group event. `group` is one of the SNRT_PERF_TCDM_GRP_*
// selectors.
inline void snrt_start_perf_counter_tcdm_group(
    enum snrt_perf_cnt perf_cnt, enum snrt_perf_cnt_type perf_cnt_type,
    uint32_t group) {
    snrt_perf_counters()->hart_select[perf_cnt].value =
        (group << SNRT_PERF_PORT_GROUP_SHIFT);
    snrt_start_perf_counter(perf_cnt, perf_cnt_type, 0);
}

// Enable a single-TCDM-port event. `port` indexes the interconnect request vector
// {soc, cores, snax}; see the cluster's generated TCDM port map.
inline void snrt_start_perf_counter_tcdm_port(
    enum snrt_perf_cnt perf_cnt, enum snrt_perf_cnt_type perf_cnt_type,
    uint32_t port) {
    snrt_perf_counters()->hart_select[perf_cnt].value =
        (port << SNRT_PERF_PORT_INDEX_SHIFT);
    snrt_start_perf_counter(perf_cnt, perf_cnt_type, 0);
}

// Stops the counter but does not reset it
inline void snrt_stop_perf_counter(enum snrt_perf_cnt perf_cnt) {
    snrt_perf_counters()->enable[perf_cnt].value = 0x0;
    snrt_perf_counters()->enable_ext[perf_cnt].value = 0x0;
}

// Resets the counter completely
inline void snrt_reset_perf_counter(enum snrt_perf_cnt perf_cnt) {
    snrt_perf_counters()->enable[perf_cnt].value = 0x0;
    snrt_perf_counters()->enable_ext[perf_cnt].value = 0x0;
    snrt_perf_counters()->hart_select[perf_cnt].value = 0x0;
    snrt_perf_counters()->perf_counter[perf_cnt].value = 0x0;
}

// Get counter of specified perf_counter
inline uint32_t snrt_get_perf_counter(enum snrt_perf_cnt perf_cnt) {
    return (uint32_t)(snrt_perf_counters()->perf_counter[perf_cnt].value);
}
