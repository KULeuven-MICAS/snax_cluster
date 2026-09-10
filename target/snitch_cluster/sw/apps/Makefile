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
SUBDIRS += snax-simd-elementwise-add
endif

# The four-engine split cluster. The reader-side extensions moved to the SIMD
# block on hart 1, so the SIMD apps live here; the xDMA keeps its writer
# extension and the junctions, so the memset app still applies to it.
ifeq ($(CFG_OVERRIDE), cfg/snax_xdma_split_cluster.hjson)
# Bring-up and the concurrency proof
SUBDIRS += snax-simd-test
SUBDIRS += snax-simd-xdma-concurrent
# Reader-extension kernels, now running on the SIMD block (hart 1). This list is
# exactly the snax_xdma_cluster set: the split cluster inherits that cfg's reader
# extension list, so kernels written against a different one (maxpool, avgpool,
# rescale-*, which belong to snax_compiler_test_cluster and index their
# extensions positionally) do NOT apply here.
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-simd-elementwise-add
SUBDIRS += snax-simd-softmax
SUBDIRS += snax-simd-rmsnorm
SUBDIRS += snax-simd-silu
SUBDIRS += snax-simd-swiglu
SUBDIRS += snax-simd-rope
SUBDIRS += snax-simd-reduce-multirow
SUBDIRS += snax-simd-rmsnorm-multirow
SUBDIRS += snax-simd-rmsnorm-fold
SUBDIRS += snax-simd-softmax-multirow
SUBDIRS += snax-simd-softmax-fold
# Kernels that stay on the xDMA core (hart 2): writer extension / plain moves.
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-xdma-reshape-comparison
# GEMM on hart 0. Its data.h is generated from the cluster cfg, so it follows the
# mesh shape automatically.
SUBDIRS += snax-gemmx-matmul
# FlashAttention: the workload that exercises GEMM and SIMD together.
SUBDIRS += snax-flashattn
endif
# The FP16 SIMD apps below need the StreamMap / StreamReduce / StreamElementwise /
# Fp16ToInt8 reader extensions, which only cfg/snax_xdma_cluster.hjson builds --
# they do not compile against the plain KUL cluster above.
ifeq ($(CFG_OVERRIDE), cfg/snax_xdma_cluster.hjson)
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-xdma-reshape-comparison
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-simd-elementwise-add
SUBDIRS += snax-simd-softmax
SUBDIRS += snax-simd-rmsnorm
SUBDIRS += snax-simd-silu
SUBDIRS += snax-simd-swiglu
SUBDIRS += snax-simd-rope
SUBDIRS += snax-simd-reduce-multirow
SUBDIRS += snax-simd-rmsnorm-multirow
SUBDIRS += snax-simd-rmsnorm-fold
SUBDIRS += snax-simd-softmax-multirow
SUBDIRS += snax-simd-softmax-fold
endif
ifeq ($(CFG_OVERRIDE), cfg/snax_compiler_test_cluster.hjson)
SUBDIRS += snax-xdma-memset
SUBDIRS += snax-simd-maxpool
SUBDIRS += snax-xdma-transpose
SUBDIRS += snax-xdma-reshape-comparison
SUBDIRS += snax-simd-elementwise-add
SUBDIRS += snax-simd-rescale-down
SUBDIRS += snax-simd-rescale-up
SUBDIRS += snax-simd-avgpool
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
SUBDIRS += snax-simd-elementwise-add
SUBDIRS += snax-simd-reduce-multirow
SUBDIRS += snax-simd-rmsnorm-multirow
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
SUBDIRS += snax-simd-elementwise-add
SUBDIRS += snax-simd-reduce-multirow
SUBDIRS += snax-simd-rmsnorm-multirow
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
