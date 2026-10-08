# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

SUBDIRS  = lto
SUBDIRS += nop
# Tests below don't work with the generic runtime
ifneq ($(filter $(SELECT_RUNTIME),rtl banshee),)
SUBDIRS += blas/axpy
SUBDIRS += blas/gemm
SUBDIRS += dnn/batchnorm
SUBDIRS += dnn/conv2d
SUBDIRS += dnn/fusedconv
SUBDIRS += dnn/gelu
SUBDIRS += dnn/gemm
SUBDIRS += dnn/layernorm
SUBDIRS += dnn/linear
SUBDIRS += dnn/maxpool
SUBDIRS += dnn/softmax
endif
SUBDIRS += montecarlo/pi_estimation
SUBDIRS += snax-mac
SUBDIRS += snax-mac-simple/untiled
SUBDIRS += snax-mac-simple/tiled
ifeq ($(CFG_OVERRIDE), cfg/snax_alu_cluster.hjson)
SUBDIRS += snax-alu
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_multi_alu_cluster.hjson)
SUBDIRS += snax-alu
SUBDIRS += snax-multi-alu
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_KUL_cluster.hjson)
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-xdma-reshape-comparison
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-rdext-elementwise-add
endif

# The four-engine split cluster. The reader-side extensions moved to the SIMD
# block on hart 1, so the SIMD apps live here; the xDMA keeps its writer
# extension and the junctions, so the memset app still applies to it.
ifeq ($(CFG_OVERRIDE), cfg/snax_split_cluster.hjson)
# Bring-up and the concurrency proof
SUBDIRS += snax-simd-test
SUBDIRS += snax-simd-xdma-concurrent
# Per-engine machine constants for the bingo performance model: one ELF, four sweeps.
SUBDIRS += snax-model-calib
SUBDIRS += snax-model-calib-gemm
SUBDIRS += snax-gemm-pipeline
# Reader-extension kernels, which on this cfg run on the SIMD block (hart 1). This list is
# exactly the snax_xdma_cluster set: the split cluster inherits that cfg's reader
# extension list, so kernels written against a different one (maxpool, avgpool,
# rescale-*, which belong to snax_compiler_test_cluster and index their
# extensions positionally) do NOT apply here.
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-simd-softmax
SUBDIRS += snax-simd-rmsnorm
SUBDIRS += snax-simd-silu
SUBDIRS += snax-simd-swiglu
SUBDIRS += snax-simd-rope
SUBDIRS += snax-simd-reduce-multirow
# Kernels that stay on the xDMA core (hart 2): writer extension / plain moves.
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-xdma-reshape-comparison
# Proves the testbench's main-memory xDMA endpoint: the engine can only move bytes
# it did not write itself once there is a second endpoint to talk to.
SUBDIRS += snax-xdma-mem-read
# Main-memory bandwidth into L1 with the iDMA (AXI reads) and the xDMA (the endpoint's AXI
# writes) reading at once.
SUBDIRS += snax-xdma-idma-read-bw
# GEMM on hart 0. FlashAttention is the workload that exercises GEMM and SIMD together;
# its data.h is generated from the cluster cfg, so it follows the array shape.
#
# TWO APPS, ONE KERNEL SHAPE. The difference between the phases of inference is how many
# QUERY tiles amortise one pass over K and V:
#
#   decode   NQ = 1   one query tile of g*k rows against the whole cache, per token
#   prefill  NQ > 1   many query tiles against the same resident K/V
#
# Arithmetic intensity is exactly Br*NQ -- Bc and d cancel -- so NQ is the only knob that
# moves bytes per MAC, and it is what separates a memory-bound decode from a compute-bound
# prefill on this machine.
SUBDIRS += snax-flashattn-decode
SUBDIRS += snax-flashattn-prefill
# A few-row GEMM (an MoE expert's routed tokens) in the two GEMM array shapes, (16, 4, 16) and
# (8, 8, 16), INT8 and INT4 weights: checked against the golden, and the array's efficiency.
SUBDIRS += snax-versacore-few-row
# The layer stretch that FEEDS FlashAttention: RMSNorm -> quantise -> Q/K/V projections
# -> int8 operands in FA's own layouts, all four engines, checked against FA's two matmuls.
SUBDIRS += snax-llm-qkv-to-fa
# DeepSeek-V2-Lite layer 1, kernel by kernel, against the golden in sw/apps/dsv2/util.
# sw/snax-dsv2-run.yaml runs them.
SUBDIRS += dsv2/snax-dsv2-gemv
SUBDIRS += dsv2/snax-dsv2-absorb
SUBDIRS += dsv2/snax-dsv2-simd
SUBDIRS += dsv2/snax-dsv2-cache
SUBDIRS += dsv2/snax-dsv2-mla-fa
SUBDIRS += dsv2/snax-dsv2-mla
SUBDIRS += dsv2/snax-dsv2-router
SUBDIRS += dsv2/snax-dsv2-moe
SUBDIRS += dsv2/snax-dsv2-layer
SUBDIRS += dsv2/snax-dsv2-spec
endif
# The FP16 SIMD apps below need the StreamMap / StreamReduce / StreamElementwise /
# Fp16ToInt8 reader extensions, which only cfg/snax_xdma_cluster.hjson builds --
# they do not compile against the plain KUL cluster above.
ifeq ($(CFG_OVERRIDE), cfg/snax_xdma_cluster.hjson)
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-xdma-reshape-comparison
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-rdext-elementwise-add
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_compiler_test_cluster.hjson)
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-rdext-maxpool
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-xdma-reshape-comparison
SUBDIRS += snax-rdext-elementwise-add
SUBDIRS += snax-rdext-rescale-down
SUBDIRS += snax-rdext-rescale-up
SUBDIRS += snax-rdext-avgpool
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_versacore_to_cluster.hjson)
SUBDIRS += snax-versacore-to-matmul
SUBDIRS += snax-versacore-to-matmul-profile
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-xdma-reshape-comparison
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_versacore_to_simd_cluster.hjson)
SUBDIRS += snax-versacore-to-matmul
SUBDIRS += snax-versacore-to-matmul-profile
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-xdma-reshape-comparison
SUBDIRS += snax-rdext-elementwise-add
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_versacore_to_256KB_cluster.hjson)
SUBDIRS += snax-versacore-to-matmul
SUBDIRS += snax-versacore-to-matmul-profile
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-xdma-reshape-comparison
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_versacore_to_256KB_simd_cluster.hjson)
SUBDIRS += snax-versacore-to-matmul
SUBDIRS += snax-versacore-to-matmul-profile
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-xdma-reshape-comparison
SUBDIRS += snax-rdext-elementwise-add
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_versacore_dse_cluster.hjson)
SUBDIRS += snax-versacore-dse-matmul-profile
endif
SUBDIRS += snax-multi-cluster-playground
ifeq ($(CFG_OVERRIDE), cfg/snax_hypercorex_cluster.hjson)
SUBDIRS += snax-hypercorex/test-csr
SUBDIRS += snax-hypercorex/char-recog
SUBDIRS += snax-hypercorex/high-dim-bind
SUBDIRS += snax-hypercorex/am-search
SUBDIRS += snax-hypercorex/isolet-recog
SUBDIRS += snax-hypercorex/isolet-recog-long
SUBDIRS += snax-hypercorex/ucihar-recog
SUBDIRS += snax-hypercorex/ucihar-recog-long
SUBDIRS += snax-hypercorex/digit-recog
SUBDIRS += snax-hypercorex/digit-recog-long
SUBDIRS += snax-hypercorex/dna-recog
SUBDIRS += snax-hypercorex/dna-recog-long
SUBDIRS += snax-hypercorex/lang-recog
SUBDIRS += snax-hypercorex/lang-recog-long
SUBDIRS += snax-hypercorex/vec-compress
endif


.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)
