// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

#include <stdbool.h>
#include "snax-cgra-params.h"
#include "snrt.h"
#include "stdint.h"
#include "streamer_csr_addr_map.h"

void cgra_hw_barrier(int t_stage, int threshold, bool verbose, bool quiet);
void cgra_hw_barrier_fast(int t_stage, int threshold);
void cgra_hw_profiler();
