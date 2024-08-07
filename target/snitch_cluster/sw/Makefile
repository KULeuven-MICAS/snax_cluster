# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

TARGET ?= all
SELECT_RUNTIME ?= rtl

ifeq ($(SELECT_RUNTIME), banshee)
RUNTIME = runtime/banshee
else ifeq ($(SELECT_RUNTIME), rtl-generic)
RUNTIME = runtime/rtl-generic
else
RUNTIME = runtime/rtl
endif

SUBDIRS += math $(RUNTIME) snax/mac snax/gemm snax/streamer-gemm snax/streamer-simd snax/data-reshuffler snax/streamer-gemm-conv snax/streamer-gemm-conv-simd snax/xdma apps tests

.PHONY: all $(SUBDIRS)

all: $(SUBDIRS)

# Explicit dependency of apps on runtime
apps: math $(RUNTIME) snax/mac snax/gemm snax/streamer-gemm snax/streamer-simd snax/data-reshuffler snax/streamer-gemm-conv snax/streamer-gemm-conv-simd snax/xdma
	$(MAKE) -C $@ TARGET=$(TARGET)

# Explicit dependency of tests on runtime
tests: math $(RUNTIME) snax/mac snax/gemm snax/streamer-gemm snax/streamer-simd snax/data-reshuffler snax/streamer-gemm-conv snax/streamer-gemm-conv-simd snax/xdma
	$(MAKE) -C $@ $(TARGET)

runtime/rtl runtime/banshee runtime/rtl-generic math snax/mac snax/gemm snax/streamer-gemm snax/streamer-simd snax/data-reshuffler snax/streamer-gemm-conv snax/streamer-gemm-conv-simd snax/xdma:
	$(MAKE) -C $@ $(TARGET)
