# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# The snax-dsv2-* apps' shared build settings; each app includes this after common.mk.
#
# DSV2_STAGE_CHECKS selects what an app checks against the golden:
#   0 (default)  the final results only, after the kernel: the kernel runs without a stop, the
#                goldens load once it is done, and its trace ([SPAN]) is the kernel's alone
#   1            every intermediate stage as well; apps with several shapes check each one as it
#                finishes, between the shapes
#
#   make -C sw/apps/dsv2/snax-dsv2-gemv DSV2_STAGE_CHECKS=1
#
# DSV2_CHECK_TERMS limits how much of each checked tensor a check compares, for regressions:
#   0 (default)  every term
#   N            the first N terms only; the app prints a [CHECK] line saying so, and each
#                check's report counts the terms it compared
# A check runs on one hart with one load in flight, so comparing a large tensor whole can take
# more simulated cycles than its kernel (snax-dsv2-mla-fa: 131,072 bytes, some 340 K cycles).
#
#   make -C sw/apps/dsv2/snax-dsv2-mla-fa DSV2_CHECK_TERMS=128
#
# DSV2_DUAL_LOAD selects how a one-token pass loads its streamed weights: the MLA's W_DKV, W_Q and
# W_O chunks (snax-dsv2-mla.h) and the MoE's (snax-dsv2-moe.h):
#   1 (default)  each chunk in two parts at once: the iDMA reads the head of the chunk with AXI
#                reads, the xDMA its tail (DSV2_DUAL_XBYTES, default half) through the
#                main-memory endpoint, which pushes it into the cluster as AXI writes
#   0            the whole chunk on the iDMA
# Two tokens per pass load on the iDMA either way: their GEMV takes 64 weight bytes a cycle,
# which the iDMA feeds alone.
#
# DSV2_GEMV selects the array shape a one-token GEMV runs (snax-dsv2.h, THE ARRAY SHAPE):
#   1 (default)  (1, 4, 32): one row, 128 weight bytes a pass
#   0            (16, 4, 16): the GEMM shape, row 0 kept
#
# DSV2_TCDM_PRIO selects the TCDM arbitration policy for the whole run (snax-tcdm-priority.h,
# SNAX_TCDM_POLICY_*): 0 (default) the hardware's own, every requester's urgency and the
# starvation guard; 1 one round robin; 2 urgency without the guard; 3 GEMM first; 4 xDMA first;
# 5 SIMD first; 6 xDMA last. DSV2_TCDM_GUARD sets the guard's wait in cycles (0, the default,
# keeps the hardware's). DSV2_TCDM_PRIO_STREAM, _HEAD and _ATTN override the policy per kernel
# class in the MLA: its GEMM hart installs the class's policy as each phase starts.
#
# The ELF rebuilds when a setting changes: it depends on a stamp named after the values.

# The golden pack every app's datagen imports: the Python package util/ next to the apps.
DSV2_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

DSV2_STAGE_CHECKS ?= 0
DSV2_CHECK_TERMS ?= 0
DSV2_DUAL_LOAD ?= 1
DSV2_GEMV ?= 1
DSV2_TCDM_PRIO ?= 0
DSV2_TCDM_GUARD ?= 0
RISCV_CFLAGS += -DDSV2_STAGE_CHECKS=$(DSV2_STAGE_CHECKS)
RISCV_CFLAGS += -DDSV2_CHECK_TERMS=$(DSV2_CHECK_TERMS)
RISCV_CFLAGS += -DDSV2_DUAL_LOAD=$(DSV2_DUAL_LOAD)
RISCV_CFLAGS += -DDSV2_GEMV=$(DSV2_GEMV)
RISCV_CFLAGS += -DDSV2_TCDM_PRIO=$(DSV2_TCDM_PRIO)
RISCV_CFLAGS += -DDSV2_TCDM_GUARD=$(DSV2_TCDM_GUARD)
ifdef DSV2_DUAL_XBYTES
RISCV_CFLAGS += -DDSV2_DUAL_XBYTES=$(DSV2_DUAL_XBYTES)
endif
ifdef DSV2_TCDM_PRIO_STREAM
RISCV_CFLAGS += -DDSV2_TCDM_PRIO_STREAM=$(DSV2_TCDM_PRIO_STREAM)
endif
ifdef DSV2_TCDM_PRIO_HEAD
RISCV_CFLAGS += -DDSV2_TCDM_PRIO_HEAD=$(DSV2_TCDM_PRIO_HEAD)
endif
ifdef DSV2_TCDM_PRIO_ATTN
RISCV_CFLAGS += -DDSV2_TCDM_PRIO_ATTN=$(DSV2_TCDM_PRIO_ATTN)
endif

DSV2_CFG_STAMP = $(BUILDDIR)/dsv2-cfg-$(DSV2_STAGE_CHECKS)-$(DSV2_CHECK_TERMS)-$(DSV2_DUAL_LOAD)-$(DSV2_GEMV)-$(DSV2_TCDM_PRIO)-$(DSV2_TCDM_GUARD)-$(DSV2_TCDM_PRIO_STREAM)-$(DSV2_TCDM_PRIO_HEAD)-$(DSV2_TCDM_PRIO_ATTN)-$(DSV2_DUAL_XBYTES)

$(DSV2_CFG_STAMP): | $(BUILDDIR)
	rm -f $(BUILDDIR)/stage-checks-* $(BUILDDIR)/dsv2-cfg-*
	touch $@

$(DEP) $(ELF): $(DSV2_CFG_STAMP)
