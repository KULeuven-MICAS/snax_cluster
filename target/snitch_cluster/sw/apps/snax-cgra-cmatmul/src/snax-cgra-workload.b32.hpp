#include <stdbool.h>
#include "snax-cgra-lib.h"
#include "snax-cgra-params.h"
#include "snrt.h"
#include "stdint.h"
#include "streamer_csr_addr_map.h"


void launch_cgra_0(int32_t delta_config_data, int32_t delta_comp_data, int32_t delta_store_data, uint32_t *mcycle_timestamps) {
		mcycle_timestamps[0] = snrt_mcycle();
		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 1275069473);
		csrw_ss(CGRA_CSR_ADDR_BASE +  1,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  2,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  3,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  4,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  5,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  6,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  7,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  8,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  9,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 10,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 11,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 12,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 13,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 14,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 15,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 16,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 17,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 18,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 19,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 20,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 21,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 22,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 23,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 24,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 25,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 26,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 27,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 28,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 29,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 30,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 31,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 32,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 33,    2127873);
		csrw_ss(CGRA_CSR_ADDR_BASE + 34,          1);
		csrw_ss(CGRA_CSR_ADDR_BASE + 35,          1);
		csrw_ss(CGRA_CSR_ADDR_BASE + 36,          1);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		mcycle_timestamps[1] = snrt_mcycle();
		// DS_R handles begin
		// READER_0
		csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +    0));
		csrw_ss(S_STRIDE_READER_0_0, 8);
		csrw_ss(T_BOUND_READER_0_0,    1);
		csrw_ss(T_STRIDE_READER_0_0,   64);
		csrw_ss(T_BOUND_READER_0_1,    1);
		csrw_ss(T_STRIDE_READER_0_1,    0);
		csrw_ss(T_BOUND_READER_0_2,    1);
		csrw_ss(T_STRIDE_READER_0_2,    0);
		csrw_ss(T_BOUND_READER_0_3,    1);
		csrw_ss(T_STRIDE_READER_0_3,    0);
		csrw_ss(T_BOUND_READER_0_4,    1);
		csrw_ss(T_STRIDE_READER_0_4,    0);
		csrw_ss(T_BOUND_READER_0_5,    1);
		csrw_ss(T_STRIDE_READER_0_5,    0);
		csrw_ss(T_BOUND_READER_0_6,    1);
		csrw_ss(T_STRIDE_READER_0_6,    0);
		csrw_ss(T_BOUND_READER_0_7,    1);
		csrw_ss(T_STRIDE_READER_0_7,    0);
		// READER_1
		csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +    8));
		csrw_ss(S_STRIDE_READER_1_0, 8);
		csrw_ss(T_BOUND_READER_1_0,    1);
		csrw_ss(T_STRIDE_READER_1_0,   64);
		csrw_ss(T_BOUND_READER_1_1,    1);
		csrw_ss(T_STRIDE_READER_1_1,    0);
		csrw_ss(T_BOUND_READER_1_2,    1);
		csrw_ss(T_STRIDE_READER_1_2,    0);
		csrw_ss(T_BOUND_READER_1_3,    1);
		csrw_ss(T_STRIDE_READER_1_3,    0);
		csrw_ss(T_BOUND_READER_1_4,    1);
		csrw_ss(T_STRIDE_READER_1_4,    0);
		csrw_ss(T_BOUND_READER_1_5,    1);
		csrw_ss(T_STRIDE_READER_1_5,    0);
		csrw_ss(T_BOUND_READER_1_6,    1);
		csrw_ss(T_STRIDE_READER_1_6,    0);
		csrw_ss(T_BOUND_READER_1_7,    1);
		csrw_ss(T_STRIDE_READER_1_7,    0);
		// READER_2
		csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   16));
		csrw_ss(S_STRIDE_READER_2_0, 8);
		csrw_ss(T_BOUND_READER_2_0,    1);
		csrw_ss(T_STRIDE_READER_2_0,   64);
		csrw_ss(T_BOUND_READER_2_1,    1);
		csrw_ss(T_STRIDE_READER_2_1,    0);
		csrw_ss(T_BOUND_READER_2_2,    1);
		csrw_ss(T_STRIDE_READER_2_2,    0);
		csrw_ss(T_BOUND_READER_2_3,    1);
		csrw_ss(T_STRIDE_READER_2_3,    0);
		csrw_ss(T_BOUND_READER_2_4,    1);
		csrw_ss(T_STRIDE_READER_2_4,    0);
		csrw_ss(T_BOUND_READER_2_5,    1);
		csrw_ss(T_STRIDE_READER_2_5,    0);
		csrw_ss(T_BOUND_READER_2_6,    1);
		csrw_ss(T_STRIDE_READER_2_6,    0);
		csrw_ss(T_BOUND_READER_2_7,    1);
		csrw_ss(T_STRIDE_READER_2_7,    0);
		// READER_3
		csrw_ss(BASE_PTR_READER_3_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   24));
		csrw_ss(S_STRIDE_READER_3_0, 8);
		csrw_ss(T_BOUND_READER_3_0,    1);
		csrw_ss(T_STRIDE_READER_3_0,   64);
		csrw_ss(T_BOUND_READER_3_1,    1);
		csrw_ss(T_STRIDE_READER_3_1,    0);
		csrw_ss(T_BOUND_READER_3_2,    1);
		csrw_ss(T_STRIDE_READER_3_2,    0);
		csrw_ss(T_BOUND_READER_3_3,    1);
		csrw_ss(T_STRIDE_READER_3_3,    0);
		csrw_ss(T_BOUND_READER_3_4,    1);
		csrw_ss(T_STRIDE_READER_3_4,    0);
		csrw_ss(T_BOUND_READER_3_5,    1);
		csrw_ss(T_STRIDE_READER_3_5,    0);
		csrw_ss(T_BOUND_READER_3_6,    1);
		csrw_ss(T_STRIDE_READER_3_6,    0);
		csrw_ss(T_BOUND_READER_3_7,    1);
		csrw_ss(T_STRIDE_READER_3_7,    0);
		// READER_4
		csrw_ss(BASE_PTR_READER_4_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   32));
		csrw_ss(S_STRIDE_READER_4_0, 8);
		csrw_ss(T_BOUND_READER_4_0,    1);
		csrw_ss(T_STRIDE_READER_4_0,   64);
		csrw_ss(T_BOUND_READER_4_1,    1);
		csrw_ss(T_STRIDE_READER_4_1,    0);
		csrw_ss(T_BOUND_READER_4_2,    1);
		csrw_ss(T_STRIDE_READER_4_2,    0);
		csrw_ss(T_BOUND_READER_4_3,    1);
		csrw_ss(T_STRIDE_READER_4_3,    0);
		csrw_ss(T_BOUND_READER_4_4,    1);
		csrw_ss(T_STRIDE_READER_4_4,    0);
		csrw_ss(T_BOUND_READER_4_5,    1);
		csrw_ss(T_STRIDE_READER_4_5,    0);
		csrw_ss(T_BOUND_READER_4_6,    1);
		csrw_ss(T_STRIDE_READER_4_6,    0);
		csrw_ss(T_BOUND_READER_4_7,    1);
		csrw_ss(T_STRIDE_READER_4_7,    0);
		// READER_5
		csrw_ss(BASE_PTR_READER_5_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   40));
		csrw_ss(S_STRIDE_READER_5_0, 8);
		csrw_ss(T_BOUND_READER_5_0,    1);
		csrw_ss(T_STRIDE_READER_5_0,   64);
		csrw_ss(T_BOUND_READER_5_1,    1);
		csrw_ss(T_STRIDE_READER_5_1,    0);
		csrw_ss(T_BOUND_READER_5_2,    1);
		csrw_ss(T_STRIDE_READER_5_2,    0);
		csrw_ss(T_BOUND_READER_5_3,    1);
		csrw_ss(T_STRIDE_READER_5_3,    0);
		csrw_ss(T_BOUND_READER_5_4,    1);
		csrw_ss(T_STRIDE_READER_5_4,    0);
		csrw_ss(T_BOUND_READER_5_5,    1);
		csrw_ss(T_STRIDE_READER_5_5,    0);
		csrw_ss(T_BOUND_READER_5_6,    1);
		csrw_ss(T_STRIDE_READER_5_6,    0);
		csrw_ss(T_BOUND_READER_5_7,    1);
		csrw_ss(T_STRIDE_READER_5_7,    0);
		// READER_6
		csrw_ss(BASE_PTR_READER_6_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   48));
		csrw_ss(S_STRIDE_READER_6_0, 8);
		csrw_ss(T_BOUND_READER_6_0,    1);
		csrw_ss(T_STRIDE_READER_6_0,   64);
		csrw_ss(T_BOUND_READER_6_1,    1);
		csrw_ss(T_STRIDE_READER_6_1,    0);
		csrw_ss(T_BOUND_READER_6_2,    1);
		csrw_ss(T_STRIDE_READER_6_2,    0);
		csrw_ss(T_BOUND_READER_6_3,    1);
		csrw_ss(T_STRIDE_READER_6_3,    0);
		csrw_ss(T_BOUND_READER_6_4,    1);
		csrw_ss(T_STRIDE_READER_6_4,    0);
		csrw_ss(T_BOUND_READER_6_5,    1);
		csrw_ss(T_STRIDE_READER_6_5,    0);
		csrw_ss(T_BOUND_READER_6_6,    1);
		csrw_ss(T_STRIDE_READER_6_6,    0);
		csrw_ss(T_BOUND_READER_6_7,    1);
		csrw_ss(T_STRIDE_READER_6_7,    0);
		// READER_7
		csrw_ss(BASE_PTR_READER_7_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   56));
		csrw_ss(S_STRIDE_READER_7_0, 8);
		csrw_ss(T_BOUND_READER_7_0,    1);
		csrw_ss(T_STRIDE_READER_7_0,   64);
		csrw_ss(T_BOUND_READER_7_1,    1);
		csrw_ss(T_STRIDE_READER_7_1,    0);
		csrw_ss(T_BOUND_READER_7_2,    1);
		csrw_ss(T_STRIDE_READER_7_2,    0);
		csrw_ss(T_BOUND_READER_7_3,    1);
		csrw_ss(T_STRIDE_READER_7_3,    0);
		csrw_ss(T_BOUND_READER_7_4,    1);
		csrw_ss(T_STRIDE_READER_7_4,    0);
		csrw_ss(T_BOUND_READER_7_5,    1);
		csrw_ss(T_STRIDE_READER_7_5,    0);
		csrw_ss(T_BOUND_READER_7_6,    1);
		csrw_ss(T_STRIDE_READER_7_6,    0);
		csrw_ss(T_BOUND_READER_7_7,    1);
		csrw_ss(T_STRIDE_READER_7_7,    0);
		// DS_R handles end
		csrw_ss(STREAMER_START_CSR, 1);


		// DS_R handles begin
		// READER_0
		csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +    0));
		csrw_ss(S_STRIDE_READER_0_0, 8);
		csrw_ss(T_BOUND_READER_0_0,    2);
		csrw_ss(T_STRIDE_READER_0_0,   64);
		csrw_ss(T_BOUND_READER_0_1,    1);
		csrw_ss(T_STRIDE_READER_0_1,    0);
		csrw_ss(T_BOUND_READER_0_2,    1);
		csrw_ss(T_STRIDE_READER_0_2,    0);
		csrw_ss(T_BOUND_READER_0_3,    1);
		csrw_ss(T_STRIDE_READER_0_3,    0);
		csrw_ss(T_BOUND_READER_0_4,    1);
		csrw_ss(T_STRIDE_READER_0_4,    0);
		csrw_ss(T_BOUND_READER_0_5,    1);
		csrw_ss(T_STRIDE_READER_0_5,    0);
		csrw_ss(T_BOUND_READER_0_6,    1);
		csrw_ss(T_STRIDE_READER_0_6,    0);
		csrw_ss(T_BOUND_READER_0_7,    1);
		csrw_ss(T_STRIDE_READER_0_7,    0);
		// READER_1
		csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +    8));
		csrw_ss(S_STRIDE_READER_1_0, 8);
		csrw_ss(T_BOUND_READER_1_0,    2);
		csrw_ss(T_STRIDE_READER_1_0,   64);
		csrw_ss(T_BOUND_READER_1_1,    1);
		csrw_ss(T_STRIDE_READER_1_1,    0);
		csrw_ss(T_BOUND_READER_1_2,    1);
		csrw_ss(T_STRIDE_READER_1_2,    0);
		csrw_ss(T_BOUND_READER_1_3,    1);
		csrw_ss(T_STRIDE_READER_1_3,    0);
		csrw_ss(T_BOUND_READER_1_4,    1);
		csrw_ss(T_STRIDE_READER_1_4,    0);
		csrw_ss(T_BOUND_READER_1_5,    1);
		csrw_ss(T_STRIDE_READER_1_5,    0);
		csrw_ss(T_BOUND_READER_1_6,    1);
		csrw_ss(T_STRIDE_READER_1_6,    0);
		csrw_ss(T_BOUND_READER_1_7,    1);
		csrw_ss(T_STRIDE_READER_1_7,    0);
		// READER_2
		csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   16));
		csrw_ss(S_STRIDE_READER_2_0, 8);
		csrw_ss(T_BOUND_READER_2_0,    2);
		csrw_ss(T_STRIDE_READER_2_0,   64);
		csrw_ss(T_BOUND_READER_2_1,    1);
		csrw_ss(T_STRIDE_READER_2_1,    0);
		csrw_ss(T_BOUND_READER_2_2,    1);
		csrw_ss(T_STRIDE_READER_2_2,    0);
		csrw_ss(T_BOUND_READER_2_3,    1);
		csrw_ss(T_STRIDE_READER_2_3,    0);
		csrw_ss(T_BOUND_READER_2_4,    1);
		csrw_ss(T_STRIDE_READER_2_4,    0);
		csrw_ss(T_BOUND_READER_2_5,    1);
		csrw_ss(T_STRIDE_READER_2_5,    0);
		csrw_ss(T_BOUND_READER_2_6,    1);
		csrw_ss(T_STRIDE_READER_2_6,    0);
		csrw_ss(T_BOUND_READER_2_7,    1);
		csrw_ss(T_STRIDE_READER_2_7,    0);
		// READER_3
		csrw_ss(BASE_PTR_READER_3_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   24));
		csrw_ss(S_STRIDE_READER_3_0, 8);
		csrw_ss(T_BOUND_READER_3_0,    2);
		csrw_ss(T_STRIDE_READER_3_0,   64);
		csrw_ss(T_BOUND_READER_3_1,    1);
		csrw_ss(T_STRIDE_READER_3_1,    0);
		csrw_ss(T_BOUND_READER_3_2,    1);
		csrw_ss(T_STRIDE_READER_3_2,    0);
		csrw_ss(T_BOUND_READER_3_3,    1);
		csrw_ss(T_STRIDE_READER_3_3,    0);
		csrw_ss(T_BOUND_READER_3_4,    1);
		csrw_ss(T_STRIDE_READER_3_4,    0);
		csrw_ss(T_BOUND_READER_3_5,    1);
		csrw_ss(T_STRIDE_READER_3_5,    0);
		csrw_ss(T_BOUND_READER_3_6,    1);
		csrw_ss(T_STRIDE_READER_3_6,    0);
		csrw_ss(T_BOUND_READER_3_7,    1);
		csrw_ss(T_STRIDE_READER_3_7,    0);
		// READER_4
		csrw_ss(BASE_PTR_READER_4_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   32));
		csrw_ss(S_STRIDE_READER_4_0, 8);
		csrw_ss(T_BOUND_READER_4_0,    2);
		csrw_ss(T_STRIDE_READER_4_0,   64);
		csrw_ss(T_BOUND_READER_4_1,    1);
		csrw_ss(T_STRIDE_READER_4_1,    0);
		csrw_ss(T_BOUND_READER_4_2,    1);
		csrw_ss(T_STRIDE_READER_4_2,    0);
		csrw_ss(T_BOUND_READER_4_3,    1);
		csrw_ss(T_STRIDE_READER_4_3,    0);
		csrw_ss(T_BOUND_READER_4_4,    1);
		csrw_ss(T_STRIDE_READER_4_4,    0);
		csrw_ss(T_BOUND_READER_4_5,    1);
		csrw_ss(T_STRIDE_READER_4_5,    0);
		csrw_ss(T_BOUND_READER_4_6,    1);
		csrw_ss(T_STRIDE_READER_4_6,    0);
		csrw_ss(T_BOUND_READER_4_7,    1);
		csrw_ss(T_STRIDE_READER_4_7,    0);
		// READER_5
		csrw_ss(BASE_PTR_READER_5_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   40));
		csrw_ss(S_STRIDE_READER_5_0, 8);
		csrw_ss(T_BOUND_READER_5_0,    2);
		csrw_ss(T_STRIDE_READER_5_0,   64);
		csrw_ss(T_BOUND_READER_5_1,    1);
		csrw_ss(T_STRIDE_READER_5_1,    0);
		csrw_ss(T_BOUND_READER_5_2,    1);
		csrw_ss(T_STRIDE_READER_5_2,    0);
		csrw_ss(T_BOUND_READER_5_3,    1);
		csrw_ss(T_STRIDE_READER_5_3,    0);
		csrw_ss(T_BOUND_READER_5_4,    1);
		csrw_ss(T_STRIDE_READER_5_4,    0);
		csrw_ss(T_BOUND_READER_5_5,    1);
		csrw_ss(T_STRIDE_READER_5_5,    0);
		csrw_ss(T_BOUND_READER_5_6,    1);
		csrw_ss(T_STRIDE_READER_5_6,    0);
		csrw_ss(T_BOUND_READER_5_7,    1);
		csrw_ss(T_STRIDE_READER_5_7,    0);
		// READER_6
		csrw_ss(BASE_PTR_READER_6_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   48));
		csrw_ss(S_STRIDE_READER_6_0, 8);
		csrw_ss(T_BOUND_READER_6_0,    2);
		csrw_ss(T_STRIDE_READER_6_0,   64);
		csrw_ss(T_BOUND_READER_6_1,    1);
		csrw_ss(T_STRIDE_READER_6_1,    0);
		csrw_ss(T_BOUND_READER_6_2,    1);
		csrw_ss(T_STRIDE_READER_6_2,    0);
		csrw_ss(T_BOUND_READER_6_3,    1);
		csrw_ss(T_STRIDE_READER_6_3,    0);
		csrw_ss(T_BOUND_READER_6_4,    1);
		csrw_ss(T_STRIDE_READER_6_4,    0);
		csrw_ss(T_BOUND_READER_6_5,    1);
		csrw_ss(T_STRIDE_READER_6_5,    0);
		csrw_ss(T_BOUND_READER_6_6,    1);
		csrw_ss(T_STRIDE_READER_6_6,    0);
		csrw_ss(T_BOUND_READER_6_7,    1);
		csrw_ss(T_STRIDE_READER_6_7,    0);
		// READER_7
		csrw_ss(BASE_PTR_READER_7_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   56));
		csrw_ss(S_STRIDE_READER_7_0, 8);
		csrw_ss(T_BOUND_READER_7_0,    2);
		csrw_ss(T_STRIDE_READER_7_0,   64);
		csrw_ss(T_BOUND_READER_7_1,    1);
		csrw_ss(T_STRIDE_READER_7_1,    0);
		csrw_ss(T_BOUND_READER_7_2,    1);
		csrw_ss(T_STRIDE_READER_7_2,    0);
		csrw_ss(T_BOUND_READER_7_3,    1);
		csrw_ss(T_STRIDE_READER_7_3,    0);
		csrw_ss(T_BOUND_READER_7_4,    1);
		csrw_ss(T_STRIDE_READER_7_4,    0);
		csrw_ss(T_BOUND_READER_7_5,    1);
		csrw_ss(T_STRIDE_READER_7_5,    0);
		csrw_ss(T_BOUND_READER_7_6,    1);
		csrw_ss(T_STRIDE_READER_7_6,    0);
		csrw_ss(T_BOUND_READER_7_7,    1);
		csrw_ss(T_STRIDE_READER_7_7,    0);
		// DS_R handles end
		csrw_ss(STREAMER_START_CSR, 1);


		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 1308623905);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 3825239073);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		// DS_R handles begin
		// READER_0
		csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +    0));
		csrw_ss(S_STRIDE_READER_0_0, 8);
		csrw_ss(T_BOUND_READER_0_0,    4);
		csrw_ss(T_STRIDE_READER_0_0,   64);
		csrw_ss(T_BOUND_READER_0_1,    1);
		csrw_ss(T_STRIDE_READER_0_1,    0);
		csrw_ss(T_BOUND_READER_0_2,    1);
		csrw_ss(T_STRIDE_READER_0_2,    0);
		csrw_ss(T_BOUND_READER_0_3,    1);
		csrw_ss(T_STRIDE_READER_0_3,    0);
		csrw_ss(T_BOUND_READER_0_4,    1);
		csrw_ss(T_STRIDE_READER_0_4,    0);
		csrw_ss(T_BOUND_READER_0_5,    1);
		csrw_ss(T_STRIDE_READER_0_5,    0);
		csrw_ss(T_BOUND_READER_0_6,    1);
		csrw_ss(T_STRIDE_READER_0_6,    0);
		csrw_ss(T_BOUND_READER_0_7,    1);
		csrw_ss(T_STRIDE_READER_0_7,    0);
		// READER_1
		csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +    8));
		csrw_ss(S_STRIDE_READER_1_0, 8);
		csrw_ss(T_BOUND_READER_1_0,    4);
		csrw_ss(T_STRIDE_READER_1_0,   64);
		csrw_ss(T_BOUND_READER_1_1,    1);
		csrw_ss(T_STRIDE_READER_1_1,    0);
		csrw_ss(T_BOUND_READER_1_2,    1);
		csrw_ss(T_STRIDE_READER_1_2,    0);
		csrw_ss(T_BOUND_READER_1_3,    1);
		csrw_ss(T_STRIDE_READER_1_3,    0);
		csrw_ss(T_BOUND_READER_1_4,    1);
		csrw_ss(T_STRIDE_READER_1_4,    0);
		csrw_ss(T_BOUND_READER_1_5,    1);
		csrw_ss(T_STRIDE_READER_1_5,    0);
		csrw_ss(T_BOUND_READER_1_6,    1);
		csrw_ss(T_STRIDE_READER_1_6,    0);
		csrw_ss(T_BOUND_READER_1_7,    1);
		csrw_ss(T_STRIDE_READER_1_7,    0);
		// READER_2
		csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   16));
		csrw_ss(S_STRIDE_READER_2_0, 8);
		csrw_ss(T_BOUND_READER_2_0,    4);
		csrw_ss(T_STRIDE_READER_2_0,   64);
		csrw_ss(T_BOUND_READER_2_1,    1);
		csrw_ss(T_STRIDE_READER_2_1,    0);
		csrw_ss(T_BOUND_READER_2_2,    1);
		csrw_ss(T_STRIDE_READER_2_2,    0);
		csrw_ss(T_BOUND_READER_2_3,    1);
		csrw_ss(T_STRIDE_READER_2_3,    0);
		csrw_ss(T_BOUND_READER_2_4,    1);
		csrw_ss(T_STRIDE_READER_2_4,    0);
		csrw_ss(T_BOUND_READER_2_5,    1);
		csrw_ss(T_STRIDE_READER_2_5,    0);
		csrw_ss(T_BOUND_READER_2_6,    1);
		csrw_ss(T_STRIDE_READER_2_6,    0);
		csrw_ss(T_BOUND_READER_2_7,    1);
		csrw_ss(T_STRIDE_READER_2_7,    0);
		// READER_3
		csrw_ss(BASE_PTR_READER_3_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   24));
		csrw_ss(S_STRIDE_READER_3_0, 8);
		csrw_ss(T_BOUND_READER_3_0,    4);
		csrw_ss(T_STRIDE_READER_3_0,   64);
		csrw_ss(T_BOUND_READER_3_1,    1);
		csrw_ss(T_STRIDE_READER_3_1,    0);
		csrw_ss(T_BOUND_READER_3_2,    1);
		csrw_ss(T_STRIDE_READER_3_2,    0);
		csrw_ss(T_BOUND_READER_3_3,    1);
		csrw_ss(T_STRIDE_READER_3_3,    0);
		csrw_ss(T_BOUND_READER_3_4,    1);
		csrw_ss(T_STRIDE_READER_3_4,    0);
		csrw_ss(T_BOUND_READER_3_5,    1);
		csrw_ss(T_STRIDE_READER_3_5,    0);
		csrw_ss(T_BOUND_READER_3_6,    1);
		csrw_ss(T_STRIDE_READER_3_6,    0);
		csrw_ss(T_BOUND_READER_3_7,    1);
		csrw_ss(T_STRIDE_READER_3_7,    0);
		// READER_4
		csrw_ss(BASE_PTR_READER_4_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   32));
		csrw_ss(S_STRIDE_READER_4_0, 8);
		csrw_ss(T_BOUND_READER_4_0,    4);
		csrw_ss(T_STRIDE_READER_4_0,   64);
		csrw_ss(T_BOUND_READER_4_1,    1);
		csrw_ss(T_STRIDE_READER_4_1,    0);
		csrw_ss(T_BOUND_READER_4_2,    1);
		csrw_ss(T_STRIDE_READER_4_2,    0);
		csrw_ss(T_BOUND_READER_4_3,    1);
		csrw_ss(T_STRIDE_READER_4_3,    0);
		csrw_ss(T_BOUND_READER_4_4,    1);
		csrw_ss(T_STRIDE_READER_4_4,    0);
		csrw_ss(T_BOUND_READER_4_5,    1);
		csrw_ss(T_STRIDE_READER_4_5,    0);
		csrw_ss(T_BOUND_READER_4_6,    1);
		csrw_ss(T_STRIDE_READER_4_6,    0);
		csrw_ss(T_BOUND_READER_4_7,    1);
		csrw_ss(T_STRIDE_READER_4_7,    0);
		// READER_5
		csrw_ss(BASE_PTR_READER_5_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   40));
		csrw_ss(S_STRIDE_READER_5_0, 8);
		csrw_ss(T_BOUND_READER_5_0,    4);
		csrw_ss(T_STRIDE_READER_5_0,   64);
		csrw_ss(T_BOUND_READER_5_1,    1);
		csrw_ss(T_STRIDE_READER_5_1,    0);
		csrw_ss(T_BOUND_READER_5_2,    1);
		csrw_ss(T_STRIDE_READER_5_2,    0);
		csrw_ss(T_BOUND_READER_5_3,    1);
		csrw_ss(T_STRIDE_READER_5_3,    0);
		csrw_ss(T_BOUND_READER_5_4,    1);
		csrw_ss(T_STRIDE_READER_5_4,    0);
		csrw_ss(T_BOUND_READER_5_5,    1);
		csrw_ss(T_STRIDE_READER_5_5,    0);
		csrw_ss(T_BOUND_READER_5_6,    1);
		csrw_ss(T_STRIDE_READER_5_6,    0);
		csrw_ss(T_BOUND_READER_5_7,    1);
		csrw_ss(T_STRIDE_READER_5_7,    0);
		// READER_6
		csrw_ss(BASE_PTR_READER_6_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   48));
		csrw_ss(S_STRIDE_READER_6_0, 8);
		csrw_ss(T_BOUND_READER_6_0,    4);
		csrw_ss(T_STRIDE_READER_6_0,   64);
		csrw_ss(T_BOUND_READER_6_1,    1);
		csrw_ss(T_STRIDE_READER_6_1,    0);
		csrw_ss(T_BOUND_READER_6_2,    1);
		csrw_ss(T_STRIDE_READER_6_2,    0);
		csrw_ss(T_BOUND_READER_6_3,    1);
		csrw_ss(T_STRIDE_READER_6_3,    0);
		csrw_ss(T_BOUND_READER_6_4,    1);
		csrw_ss(T_STRIDE_READER_6_4,    0);
		csrw_ss(T_BOUND_READER_6_5,    1);
		csrw_ss(T_STRIDE_READER_6_5,    0);
		csrw_ss(T_BOUND_READER_6_6,    1);
		csrw_ss(T_STRIDE_READER_6_6,    0);
		csrw_ss(T_BOUND_READER_6_7,    1);
		csrw_ss(T_STRIDE_READER_6_7,    0);
		// READER_7
		csrw_ss(BASE_PTR_READER_7_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   56));
		csrw_ss(S_STRIDE_READER_7_0, 8);
		csrw_ss(T_BOUND_READER_7_0,    4);
		csrw_ss(T_STRIDE_READER_7_0,   64);
		csrw_ss(T_BOUND_READER_7_1,    1);
		csrw_ss(T_STRIDE_READER_7_1,    0);
		csrw_ss(T_BOUND_READER_7_2,    1);
		csrw_ss(T_STRIDE_READER_7_2,    0);
		csrw_ss(T_BOUND_READER_7_3,    1);
		csrw_ss(T_STRIDE_READER_7_3,    0);
		csrw_ss(T_BOUND_READER_7_4,    1);
		csrw_ss(T_STRIDE_READER_7_4,    0);
		csrw_ss(T_BOUND_READER_7_5,    1);
		csrw_ss(T_STRIDE_READER_7_5,    0);
		csrw_ss(T_BOUND_READER_7_6,    1);
		csrw_ss(T_STRIDE_READER_7_6,    0);
		csrw_ss(T_BOUND_READER_7_7,    1);
		csrw_ss(T_STRIDE_READER_7_7,    0);
		// DS_R handles end
		csrw_ss(STREAMER_START_CSR, 1);


		mcycle_timestamps[2] = snrt_mcycle();
		// DS_R handles begin
		// READER_0
		csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data + 16624));
		csrw_ss(S_STRIDE_READER_0_0, 8);
		csrw_ss(T_BOUND_READER_0_0, 1039);
		csrw_ss(T_STRIDE_READER_0_0,    8);
		csrw_ss(T_BOUND_READER_0_1,   64);
		csrw_ss(T_STRIDE_READER_0_1,    0);
		csrw_ss(T_BOUND_READER_0_2,    1);
		csrw_ss(T_STRIDE_READER_0_2,    0);
		csrw_ss(T_BOUND_READER_0_3,    1);
		csrw_ss(T_STRIDE_READER_0_3,    0);
		csrw_ss(T_BOUND_READER_0_4,    1);
		csrw_ss(T_STRIDE_READER_0_4,    0);
		csrw_ss(T_BOUND_READER_0_5,    1);
		csrw_ss(T_STRIDE_READER_0_5,    0);
		csrw_ss(T_BOUND_READER_0_6,    1);
		csrw_ss(T_STRIDE_READER_0_6,    0);
		csrw_ss(T_BOUND_READER_0_7,    1);
		csrw_ss(T_STRIDE_READER_0_7,    0);
		// READER_1
		csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_1_0, 8);
		csrw_ss(T_BOUND_READER_1_0,    0);
		csrw_ss(T_STRIDE_READER_1_0,    0);
		csrw_ss(T_BOUND_READER_1_1,    0);
		csrw_ss(T_STRIDE_READER_1_1,    0);
		csrw_ss(T_BOUND_READER_1_2,    0);
		csrw_ss(T_STRIDE_READER_1_2,    0);
		csrw_ss(T_BOUND_READER_1_3,    0);
		csrw_ss(T_STRIDE_READER_1_3,    0);
		csrw_ss(T_BOUND_READER_1_4,    0);
		csrw_ss(T_STRIDE_READER_1_4,    0);
		csrw_ss(T_BOUND_READER_1_5,    0);
		csrw_ss(T_STRIDE_READER_1_5,    0);
		csrw_ss(T_BOUND_READER_1_6,    0);
		csrw_ss(T_STRIDE_READER_1_6,    0);
		csrw_ss(T_BOUND_READER_1_7,    0);
		csrw_ss(T_STRIDE_READER_1_7,    0);
		// READER_2
		csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_2_0, 8);
		csrw_ss(T_BOUND_READER_2_0,    0);
		csrw_ss(T_STRIDE_READER_2_0,    0);
		csrw_ss(T_BOUND_READER_2_1,    0);
		csrw_ss(T_STRIDE_READER_2_1,    0);
		csrw_ss(T_BOUND_READER_2_2,    0);
		csrw_ss(T_STRIDE_READER_2_2,    0);
		csrw_ss(T_BOUND_READER_2_3,    0);
		csrw_ss(T_STRIDE_READER_2_3,    0);
		csrw_ss(T_BOUND_READER_2_4,    0);
		csrw_ss(T_STRIDE_READER_2_4,    0);
		csrw_ss(T_BOUND_READER_2_5,    0);
		csrw_ss(T_STRIDE_READER_2_5,    0);
		csrw_ss(T_BOUND_READER_2_6,    0);
		csrw_ss(T_STRIDE_READER_2_6,    0);
		csrw_ss(T_BOUND_READER_2_7,    0);
		csrw_ss(T_STRIDE_READER_2_7,    0);
		// READER_3
		csrw_ss(BASE_PTR_READER_3_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_3_0, 8);
		csrw_ss(T_BOUND_READER_3_0, 2078);
		csrw_ss(T_STRIDE_READER_3_0,    8);
		csrw_ss(T_BOUND_READER_3_1,   32);
		csrw_ss(T_STRIDE_READER_3_1,    0);
		csrw_ss(T_BOUND_READER_3_2,    1);
		csrw_ss(T_STRIDE_READER_3_2,    0);
		csrw_ss(T_BOUND_READER_3_3,    1);
		csrw_ss(T_STRIDE_READER_3_3,    0);
		csrw_ss(T_BOUND_READER_3_4,    1);
		csrw_ss(T_STRIDE_READER_3_4,    0);
		csrw_ss(T_BOUND_READER_3_5,    1);
		csrw_ss(T_STRIDE_READER_3_5,    0);
		csrw_ss(T_BOUND_READER_3_6,    1);
		csrw_ss(T_STRIDE_READER_3_6,    0);
		csrw_ss(T_BOUND_READER_3_7,    1);
		csrw_ss(T_STRIDE_READER_3_7,    0);
		// READER_4
		csrw_ss(BASE_PTR_READER_4_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_4_0, 8);
		csrw_ss(T_BOUND_READER_4_0,    0);
		csrw_ss(T_STRIDE_READER_4_0,    0);
		csrw_ss(T_BOUND_READER_4_1,    0);
		csrw_ss(T_STRIDE_READER_4_1,    0);
		csrw_ss(T_BOUND_READER_4_2,    0);
		csrw_ss(T_STRIDE_READER_4_2,    0);
		csrw_ss(T_BOUND_READER_4_3,    0);
		csrw_ss(T_STRIDE_READER_4_3,    0);
		csrw_ss(T_BOUND_READER_4_4,    0);
		csrw_ss(T_STRIDE_READER_4_4,    0);
		csrw_ss(T_BOUND_READER_4_5,    0);
		csrw_ss(T_STRIDE_READER_4_5,    0);
		csrw_ss(T_BOUND_READER_4_6,    0);
		csrw_ss(T_STRIDE_READER_4_6,    0);
		csrw_ss(T_BOUND_READER_4_7,    0);
		csrw_ss(T_STRIDE_READER_4_7,    0);
		// READER_5
		csrw_ss(BASE_PTR_READER_5_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_5_0, 8);
		csrw_ss(T_BOUND_READER_5_0,    0);
		csrw_ss(T_STRIDE_READER_5_0,    0);
		csrw_ss(T_BOUND_READER_5_1,    0);
		csrw_ss(T_STRIDE_READER_5_1,    0);
		csrw_ss(T_BOUND_READER_5_2,    0);
		csrw_ss(T_STRIDE_READER_5_2,    0);
		csrw_ss(T_BOUND_READER_5_3,    0);
		csrw_ss(T_STRIDE_READER_5_3,    0);
		csrw_ss(T_BOUND_READER_5_4,    0);
		csrw_ss(T_STRIDE_READER_5_4,    0);
		csrw_ss(T_BOUND_READER_5_5,    0);
		csrw_ss(T_STRIDE_READER_5_5,    0);
		csrw_ss(T_BOUND_READER_5_6,    0);
		csrw_ss(T_STRIDE_READER_5_6,    0);
		csrw_ss(T_BOUND_READER_5_7,    0);
		csrw_ss(T_STRIDE_READER_5_7,    0);
		// READER_6
		csrw_ss(BASE_PTR_READER_6_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_6_0, 8);
		csrw_ss(T_BOUND_READER_6_0,    0);
		csrw_ss(T_STRIDE_READER_6_0,    0);
		csrw_ss(T_BOUND_READER_6_1,    0);
		csrw_ss(T_STRIDE_READER_6_1,    0);
		csrw_ss(T_BOUND_READER_6_2,    0);
		csrw_ss(T_STRIDE_READER_6_2,    0);
		csrw_ss(T_BOUND_READER_6_3,    0);
		csrw_ss(T_STRIDE_READER_6_3,    0);
		csrw_ss(T_BOUND_READER_6_4,    0);
		csrw_ss(T_STRIDE_READER_6_4,    0);
		csrw_ss(T_BOUND_READER_6_5,    0);
		csrw_ss(T_STRIDE_READER_6_5,    0);
		csrw_ss(T_BOUND_READER_6_6,    0);
		csrw_ss(T_STRIDE_READER_6_6,    0);
		csrw_ss(T_BOUND_READER_6_7,    0);
		csrw_ss(T_STRIDE_READER_6_7,    0);
		// READER_7
		csrw_ss(BASE_PTR_READER_7_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_7_0, 8);
		csrw_ss(T_BOUND_READER_7_0,    0);
		csrw_ss(T_STRIDE_READER_7_0,    0);
		csrw_ss(T_BOUND_READER_7_1,    0);
		csrw_ss(T_STRIDE_READER_7_1,    0);
		csrw_ss(T_BOUND_READER_7_2,    0);
		csrw_ss(T_STRIDE_READER_7_2,    0);
		csrw_ss(T_BOUND_READER_7_3,    0);
		csrw_ss(T_STRIDE_READER_7_3,    0);
		csrw_ss(T_BOUND_READER_7_4,    0);
		csrw_ss(T_STRIDE_READER_7_4,    0);
		csrw_ss(T_BOUND_READER_7_5,    0);
		csrw_ss(T_STRIDE_READER_7_5,    0);
		csrw_ss(T_BOUND_READER_7_6,    0);
		csrw_ss(T_STRIDE_READER_7_6,    0);
		csrw_ss(T_BOUND_READER_7_7,    0);
		csrw_ss(T_STRIDE_READER_7_7,    0);
		// DS_R handles end
		// DS_W handles begin
		// WRITER_0
		csrw_ss(BASE_PTR_WRITER_0_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_0_0, 8);
		csrw_ss(T_BOUND_WRITER_0_0,    0);
		csrw_ss(T_STRIDE_WRITER_0_0,    0);
		csrw_ss(T_BOUND_WRITER_0_1,    0);
		csrw_ss(T_STRIDE_WRITER_0_1,    0);
		csrw_ss(T_BOUND_WRITER_0_2,    0);
		csrw_ss(T_STRIDE_WRITER_0_2,    0);
		csrw_ss(T_BOUND_WRITER_0_3,    0);
		csrw_ss(T_STRIDE_WRITER_0_3,    0);
		csrw_ss(T_BOUND_WRITER_0_4,    0);
		csrw_ss(T_STRIDE_WRITER_0_4,    0);
		csrw_ss(T_BOUND_WRITER_0_5,    0);
		csrw_ss(T_STRIDE_WRITER_0_5,    0);
		csrw_ss(T_BOUND_WRITER_0_6,    0);
		csrw_ss(T_STRIDE_WRITER_0_6,    0);
		csrw_ss(T_BOUND_WRITER_0_7,    0);
		csrw_ss(T_STRIDE_WRITER_0_7,    0);
		// WRITER_1
		csrw_ss(BASE_PTR_WRITER_1_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_1_0, 8);
		csrw_ss(T_BOUND_WRITER_1_0,    0);
		csrw_ss(T_STRIDE_WRITER_1_0,    0);
		csrw_ss(T_BOUND_WRITER_1_1,    0);
		csrw_ss(T_STRIDE_WRITER_1_1,    0);
		csrw_ss(T_BOUND_WRITER_1_2,    0);
		csrw_ss(T_STRIDE_WRITER_1_2,    0);
		csrw_ss(T_BOUND_WRITER_1_3,    0);
		csrw_ss(T_STRIDE_WRITER_1_3,    0);
		csrw_ss(T_BOUND_WRITER_1_4,    0);
		csrw_ss(T_STRIDE_WRITER_1_4,    0);
		csrw_ss(T_BOUND_WRITER_1_5,    0);
		csrw_ss(T_STRIDE_WRITER_1_5,    0);
		csrw_ss(T_BOUND_WRITER_1_6,    0);
		csrw_ss(T_STRIDE_WRITER_1_6,    0);
		csrw_ss(T_BOUND_WRITER_1_7,    0);
		csrw_ss(T_STRIDE_WRITER_1_7,    0);
		// WRITER_2
		csrw_ss(BASE_PTR_WRITER_2_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_2_0, 8);
		csrw_ss(T_BOUND_WRITER_2_0,    0);
		csrw_ss(T_STRIDE_WRITER_2_0,    0);
		csrw_ss(T_BOUND_WRITER_2_1,    0);
		csrw_ss(T_STRIDE_WRITER_2_1,    0);
		csrw_ss(T_BOUND_WRITER_2_2,    0);
		csrw_ss(T_STRIDE_WRITER_2_2,    0);
		csrw_ss(T_BOUND_WRITER_2_3,    0);
		csrw_ss(T_STRIDE_WRITER_2_3,    0);
		csrw_ss(T_BOUND_WRITER_2_4,    0);
		csrw_ss(T_STRIDE_WRITER_2_4,    0);
		csrw_ss(T_BOUND_WRITER_2_5,    0);
		csrw_ss(T_STRIDE_WRITER_2_5,    0);
		csrw_ss(T_BOUND_WRITER_2_6,    0);
		csrw_ss(T_STRIDE_WRITER_2_6,    0);
		csrw_ss(T_BOUND_WRITER_2_7,    0);
		csrw_ss(T_STRIDE_WRITER_2_7,    0);
		// WRITER_3
		csrw_ss(BASE_PTR_WRITER_3_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_3_0, 8);
		csrw_ss(T_BOUND_WRITER_3_0,    0);
		csrw_ss(T_STRIDE_WRITER_3_0,    0);
		csrw_ss(T_BOUND_WRITER_3_1,    0);
		csrw_ss(T_STRIDE_WRITER_3_1,    0);
		csrw_ss(T_BOUND_WRITER_3_2,    0);
		csrw_ss(T_STRIDE_WRITER_3_2,    0);
		csrw_ss(T_BOUND_WRITER_3_3,    0);
		csrw_ss(T_STRIDE_WRITER_3_3,    0);
		csrw_ss(T_BOUND_WRITER_3_4,    0);
		csrw_ss(T_STRIDE_WRITER_3_4,    0);
		csrw_ss(T_BOUND_WRITER_3_5,    0);
		csrw_ss(T_STRIDE_WRITER_3_5,    0);
		csrw_ss(T_BOUND_WRITER_3_6,    0);
		csrw_ss(T_STRIDE_WRITER_3_6,    0);
		csrw_ss(T_BOUND_WRITER_3_7,    0);
		csrw_ss(T_STRIDE_WRITER_3_7,    0);
		// WRITER_4
		csrw_ss(BASE_PTR_WRITER_4_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_4_0, 8);
		csrw_ss(T_BOUND_WRITER_4_0, 2078);
		csrw_ss(T_STRIDE_WRITER_4_0,    8);
		csrw_ss(T_BOUND_WRITER_4_1,   32);
		csrw_ss(T_STRIDE_WRITER_4_1,    0);
		csrw_ss(T_BOUND_WRITER_4_2,    1);
		csrw_ss(T_STRIDE_WRITER_4_2,    0);
		csrw_ss(T_BOUND_WRITER_4_3,    1);
		csrw_ss(T_STRIDE_WRITER_4_3,    0);
		csrw_ss(T_BOUND_WRITER_4_4,    1);
		csrw_ss(T_STRIDE_WRITER_4_4,    0);
		csrw_ss(T_BOUND_WRITER_4_5,    1);
		csrw_ss(T_STRIDE_WRITER_4_5,    0);
		csrw_ss(T_BOUND_WRITER_4_6,    1);
		csrw_ss(T_STRIDE_WRITER_4_6,    0);
		csrw_ss(T_BOUND_WRITER_4_7,    1);
		csrw_ss(T_STRIDE_WRITER_4_7,    0);
		// WRITER_5
		csrw_ss(BASE_PTR_WRITER_5_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_5_0, 8);
		csrw_ss(T_BOUND_WRITER_5_0,    0);
		csrw_ss(T_STRIDE_WRITER_5_0,    0);
		csrw_ss(T_BOUND_WRITER_5_1,    0);
		csrw_ss(T_STRIDE_WRITER_5_1,    0);
		csrw_ss(T_BOUND_WRITER_5_2,    0);
		csrw_ss(T_STRIDE_WRITER_5_2,    0);
		csrw_ss(T_BOUND_WRITER_5_3,    0);
		csrw_ss(T_STRIDE_WRITER_5_3,    0);
		csrw_ss(T_BOUND_WRITER_5_4,    0);
		csrw_ss(T_STRIDE_WRITER_5_4,    0);
		csrw_ss(T_BOUND_WRITER_5_5,    0);
		csrw_ss(T_STRIDE_WRITER_5_5,    0);
		csrw_ss(T_BOUND_WRITER_5_6,    0);
		csrw_ss(T_STRIDE_WRITER_5_6,    0);
		csrw_ss(T_BOUND_WRITER_5_7,    0);
		csrw_ss(T_STRIDE_WRITER_5_7,    0);
		// WRITER_6
		csrw_ss(BASE_PTR_WRITER_6_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_6_0, 8);
		csrw_ss(T_BOUND_WRITER_6_0,    0);
		csrw_ss(T_STRIDE_WRITER_6_0,    0);
		csrw_ss(T_BOUND_WRITER_6_1,    0);
		csrw_ss(T_STRIDE_WRITER_6_1,    0);
		csrw_ss(T_BOUND_WRITER_6_2,    0);
		csrw_ss(T_STRIDE_WRITER_6_2,    0);
		csrw_ss(T_BOUND_WRITER_6_3,    0);
		csrw_ss(T_STRIDE_WRITER_6_3,    0);
		csrw_ss(T_BOUND_WRITER_6_4,    0);
		csrw_ss(T_STRIDE_WRITER_6_4,    0);
		csrw_ss(T_BOUND_WRITER_6_5,    0);
		csrw_ss(T_STRIDE_WRITER_6_5,    0);
		csrw_ss(T_BOUND_WRITER_6_6,    0);
		csrw_ss(T_STRIDE_WRITER_6_6,    0);
		csrw_ss(T_BOUND_WRITER_6_7,    0);
		csrw_ss(T_STRIDE_WRITER_6_7,    0);
		// WRITER_7
		csrw_ss(BASE_PTR_WRITER_7_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_7_0, 8);
		csrw_ss(T_BOUND_WRITER_7_0,    0);
		csrw_ss(T_STRIDE_WRITER_7_0,    0);
		csrw_ss(T_BOUND_WRITER_7_1,    0);
		csrw_ss(T_STRIDE_WRITER_7_1,    0);
		csrw_ss(T_BOUND_WRITER_7_2,    0);
		csrw_ss(T_STRIDE_WRITER_7_2,    0);
		csrw_ss(T_BOUND_WRITER_7_3,    0);
		csrw_ss(T_STRIDE_WRITER_7_3,    0);
		csrw_ss(T_BOUND_WRITER_7_4,    0);
		csrw_ss(T_STRIDE_WRITER_7_4,    0);
		csrw_ss(T_BOUND_WRITER_7_5,    0);
		csrw_ss(T_STRIDE_WRITER_7_5,    0);
		csrw_ss(T_BOUND_WRITER_7_6,    0);
		csrw_ss(T_STRIDE_WRITER_7_6,    0);
		csrw_ss(T_BOUND_WRITER_7_7,    0);
		csrw_ss(T_STRIDE_WRITER_7_7,    0);
		// DS_W handles end
		csrw_ss(STREAMER_START_CSR, 1);


		mcycle_timestamps[3] = snrt_mcycle();
		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 1308656673);
		csrw_ss(CGRA_CSR_ADDR_BASE +  1,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  2,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  3,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  4,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  5,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  6,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  7,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  8,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  9,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 10,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 11,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 12,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 13,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 14,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 15,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 16,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 17,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 18,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 19,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 20,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 21,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 22,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 23,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 24,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 25,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 26,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 27,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 28,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 29,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 30,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 31,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 32,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 33,    2127873);
		csrw_ss(CGRA_CSR_ADDR_BASE + 34,          1);
		csrw_ss(CGRA_CSR_ADDR_BASE + 35,          1);
		csrw_ss(CGRA_CSR_ADDR_BASE + 36,          1);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		mcycle_timestamps[4] = snrt_mcycle();
}

void launch_cgra_0_compact(int32_t delta_config_data, int32_t delta_comp_data, int32_t delta_store_data, uint32_t *mcycle_timestamps) {
		mcycle_timestamps[0] = snrt_mcycle();
		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 1275069473);
		csrw_ss(CGRA_CSR_ADDR_BASE +  1,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  2,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  3,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  4,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  5,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  6,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  7,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  8,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  9,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 10,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 11,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 12,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 13,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 14,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 15,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 16,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 17,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 18,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 19,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 20,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 21,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 22,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 23,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 24,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 25,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 26,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 27,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 28,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 29,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 30,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 31,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 32,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 33,    2127873);
		csrw_ss(CGRA_CSR_ADDR_BASE + 34,          1);
		csrw_ss(CGRA_CSR_ADDR_BASE + 35,          1);
		csrw_ss(CGRA_CSR_ADDR_BASE + 36,          1);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		mcycle_timestamps[1] = snrt_mcycle();
		// DS_R handles begin
		// READER_0
		csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +    0));
		csrw_ss(S_STRIDE_READER_0_0, 8);
		csrw_ss(T_BOUND_READER_0_0,    1);
		csrw_ss(T_STRIDE_READER_0_0,   64);
		csrw_ss(T_BOUND_READER_0_1,    1);
		csrw_ss(T_STRIDE_READER_0_1,    0);
		csrw_ss(T_BOUND_READER_0_2,    1);
		csrw_ss(T_STRIDE_READER_0_2,    0);
		csrw_ss(T_BOUND_READER_0_3,    1);
		csrw_ss(T_STRIDE_READER_0_3,    0);
		csrw_ss(T_BOUND_READER_0_4,    1);
		csrw_ss(T_STRIDE_READER_0_4,    0);
		csrw_ss(T_BOUND_READER_0_5,    1);
		csrw_ss(T_STRIDE_READER_0_5,    0);
		csrw_ss(T_BOUND_READER_0_6,    1);
		csrw_ss(T_STRIDE_READER_0_6,    0);
		csrw_ss(T_BOUND_READER_0_7,    1);
		csrw_ss(T_STRIDE_READER_0_7,    0);
		// READER_1
		csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +    8));
		csrw_ss(S_STRIDE_READER_1_0, 8);
		csrw_ss(T_BOUND_READER_1_0,    1);
		csrw_ss(T_STRIDE_READER_1_0,   64);
		csrw_ss(T_BOUND_READER_1_1,    1);
		csrw_ss(T_STRIDE_READER_1_1,    0);
		csrw_ss(T_BOUND_READER_1_2,    1);
		csrw_ss(T_STRIDE_READER_1_2,    0);
		csrw_ss(T_BOUND_READER_1_3,    1);
		csrw_ss(T_STRIDE_READER_1_3,    0);
		csrw_ss(T_BOUND_READER_1_4,    1);
		csrw_ss(T_STRIDE_READER_1_4,    0);
		csrw_ss(T_BOUND_READER_1_5,    1);
		csrw_ss(T_STRIDE_READER_1_5,    0);
		csrw_ss(T_BOUND_READER_1_6,    1);
		csrw_ss(T_STRIDE_READER_1_6,    0);
		csrw_ss(T_BOUND_READER_1_7,    1);
		csrw_ss(T_STRIDE_READER_1_7,    0);
		// READER_2
		csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   16));
		csrw_ss(S_STRIDE_READER_2_0, 8);
		csrw_ss(T_BOUND_READER_2_0,    1);
		csrw_ss(T_STRIDE_READER_2_0,   64);
		csrw_ss(T_BOUND_READER_2_1,    1);
		csrw_ss(T_STRIDE_READER_2_1,    0);
		csrw_ss(T_BOUND_READER_2_2,    1);
		csrw_ss(T_STRIDE_READER_2_2,    0);
		csrw_ss(T_BOUND_READER_2_3,    1);
		csrw_ss(T_STRIDE_READER_2_3,    0);
		csrw_ss(T_BOUND_READER_2_4,    1);
		csrw_ss(T_STRIDE_READER_2_4,    0);
		csrw_ss(T_BOUND_READER_2_5,    1);
		csrw_ss(T_STRIDE_READER_2_5,    0);
		csrw_ss(T_BOUND_READER_2_6,    1);
		csrw_ss(T_STRIDE_READER_2_6,    0);
		csrw_ss(T_BOUND_READER_2_7,    1);
		csrw_ss(T_STRIDE_READER_2_7,    0);
		// READER_3
		csrw_ss(BASE_PTR_READER_3_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   24));
		csrw_ss(S_STRIDE_READER_3_0, 8);
		csrw_ss(T_BOUND_READER_3_0,    1);
		csrw_ss(T_STRIDE_READER_3_0,   64);
		csrw_ss(T_BOUND_READER_3_1,    1);
		csrw_ss(T_STRIDE_READER_3_1,    0);
		csrw_ss(T_BOUND_READER_3_2,    1);
		csrw_ss(T_STRIDE_READER_3_2,    0);
		csrw_ss(T_BOUND_READER_3_3,    1);
		csrw_ss(T_STRIDE_READER_3_3,    0);
		csrw_ss(T_BOUND_READER_3_4,    1);
		csrw_ss(T_STRIDE_READER_3_4,    0);
		csrw_ss(T_BOUND_READER_3_5,    1);
		csrw_ss(T_STRIDE_READER_3_5,    0);
		csrw_ss(T_BOUND_READER_3_6,    1);
		csrw_ss(T_STRIDE_READER_3_6,    0);
		csrw_ss(T_BOUND_READER_3_7,    1);
		csrw_ss(T_STRIDE_READER_3_7,    0);
		// READER_4
		csrw_ss(BASE_PTR_READER_4_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   32));
		csrw_ss(S_STRIDE_READER_4_0, 8);
		csrw_ss(T_BOUND_READER_4_0,    1);
		csrw_ss(T_STRIDE_READER_4_0,   64);
		csrw_ss(T_BOUND_READER_4_1,    1);
		csrw_ss(T_STRIDE_READER_4_1,    0);
		csrw_ss(T_BOUND_READER_4_2,    1);
		csrw_ss(T_STRIDE_READER_4_2,    0);
		csrw_ss(T_BOUND_READER_4_3,    1);
		csrw_ss(T_STRIDE_READER_4_3,    0);
		csrw_ss(T_BOUND_READER_4_4,    1);
		csrw_ss(T_STRIDE_READER_4_4,    0);
		csrw_ss(T_BOUND_READER_4_5,    1);
		csrw_ss(T_STRIDE_READER_4_5,    0);
		csrw_ss(T_BOUND_READER_4_6,    1);
		csrw_ss(T_STRIDE_READER_4_6,    0);
		csrw_ss(T_BOUND_READER_4_7,    1);
		csrw_ss(T_STRIDE_READER_4_7,    0);
		// READER_5
		csrw_ss(BASE_PTR_READER_5_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   40));
		csrw_ss(S_STRIDE_READER_5_0, 8);
		csrw_ss(T_BOUND_READER_5_0,    1);
		csrw_ss(T_STRIDE_READER_5_0,   64);
		csrw_ss(T_BOUND_READER_5_1,    1);
		csrw_ss(T_STRIDE_READER_5_1,    0);
		csrw_ss(T_BOUND_READER_5_2,    1);
		csrw_ss(T_STRIDE_READER_5_2,    0);
		csrw_ss(T_BOUND_READER_5_3,    1);
		csrw_ss(T_STRIDE_READER_5_3,    0);
		csrw_ss(T_BOUND_READER_5_4,    1);
		csrw_ss(T_STRIDE_READER_5_4,    0);
		csrw_ss(T_BOUND_READER_5_5,    1);
		csrw_ss(T_STRIDE_READER_5_5,    0);
		csrw_ss(T_BOUND_READER_5_6,    1);
		csrw_ss(T_STRIDE_READER_5_6,    0);
		csrw_ss(T_BOUND_READER_5_7,    1);
		csrw_ss(T_STRIDE_READER_5_7,    0);
		// READER_6
		csrw_ss(BASE_PTR_READER_6_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   48));
		csrw_ss(S_STRIDE_READER_6_0, 8);
		csrw_ss(T_BOUND_READER_6_0,    1);
		csrw_ss(T_STRIDE_READER_6_0,   64);
		csrw_ss(T_BOUND_READER_6_1,    1);
		csrw_ss(T_STRIDE_READER_6_1,    0);
		csrw_ss(T_BOUND_READER_6_2,    1);
		csrw_ss(T_STRIDE_READER_6_2,    0);
		csrw_ss(T_BOUND_READER_6_3,    1);
		csrw_ss(T_STRIDE_READER_6_3,    0);
		csrw_ss(T_BOUND_READER_6_4,    1);
		csrw_ss(T_STRIDE_READER_6_4,    0);
		csrw_ss(T_BOUND_READER_6_5,    1);
		csrw_ss(T_STRIDE_READER_6_5,    0);
		csrw_ss(T_BOUND_READER_6_6,    1);
		csrw_ss(T_STRIDE_READER_6_6,    0);
		csrw_ss(T_BOUND_READER_6_7,    1);
		csrw_ss(T_STRIDE_READER_6_7,    0);
		// READER_7
		csrw_ss(BASE_PTR_READER_7_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   56));
		csrw_ss(S_STRIDE_READER_7_0, 8);
		csrw_ss(T_BOUND_READER_7_0,    1);
		csrw_ss(T_STRIDE_READER_7_0,   64);
		csrw_ss(T_BOUND_READER_7_1,    1);
		csrw_ss(T_STRIDE_READER_7_1,    0);
		csrw_ss(T_BOUND_READER_7_2,    1);
		csrw_ss(T_STRIDE_READER_7_2,    0);
		csrw_ss(T_BOUND_READER_7_3,    1);
		csrw_ss(T_STRIDE_READER_7_3,    0);
		csrw_ss(T_BOUND_READER_7_4,    1);
		csrw_ss(T_STRIDE_READER_7_4,    0);
		csrw_ss(T_BOUND_READER_7_5,    1);
		csrw_ss(T_STRIDE_READER_7_5,    0);
		csrw_ss(T_BOUND_READER_7_6,    1);
		csrw_ss(T_STRIDE_READER_7_6,    0);
		csrw_ss(T_BOUND_READER_7_7,    1);
		csrw_ss(T_STRIDE_READER_7_7,    0);
		// DS_R handles end
		csrw_ss(STREAMER_START_CSR, 1);


		// DS_R handles begin
		// READER_0
		csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +    0));
		csrw_ss(S_STRIDE_READER_0_0, 8);
		csrw_ss(T_BOUND_READER_0_0,    2);
		csrw_ss(T_STRIDE_READER_0_0,   64);
		// READER_1
		csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +    8));
		csrw_ss(S_STRIDE_READER_1_0, 8);
		csrw_ss(T_BOUND_READER_1_0,    2);
		csrw_ss(T_STRIDE_READER_1_0,   64);
		// READER_2
		csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   16));
		csrw_ss(S_STRIDE_READER_2_0, 8);
		csrw_ss(T_BOUND_READER_2_0,    2);
		csrw_ss(T_STRIDE_READER_2_0,   64);
		// READER_3
		csrw_ss(BASE_PTR_READER_3_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   24));
		csrw_ss(S_STRIDE_READER_3_0, 8);
		csrw_ss(T_BOUND_READER_3_0,    2);
		csrw_ss(T_STRIDE_READER_3_0,   64);
		// READER_4
		csrw_ss(BASE_PTR_READER_4_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   32));
		csrw_ss(S_STRIDE_READER_4_0, 8);
		csrw_ss(T_BOUND_READER_4_0,    2);
		csrw_ss(T_STRIDE_READER_4_0,   64);
		// READER_5
		csrw_ss(BASE_PTR_READER_5_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   40));
		csrw_ss(S_STRIDE_READER_5_0, 8);
		csrw_ss(T_BOUND_READER_5_0,    2);
		csrw_ss(T_STRIDE_READER_5_0,   64);
		// READER_6
		csrw_ss(BASE_PTR_READER_6_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   48));
		csrw_ss(S_STRIDE_READER_6_0, 8);
		csrw_ss(T_BOUND_READER_6_0,    2);
		csrw_ss(T_STRIDE_READER_6_0,   64);
		// READER_7
		csrw_ss(BASE_PTR_READER_7_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   56));
		csrw_ss(S_STRIDE_READER_7_0, 8);
		csrw_ss(T_BOUND_READER_7_0,    2);
		csrw_ss(T_STRIDE_READER_7_0,   64);
		// DS_R handles end
		csrw_ss(STREAMER_START_CSR, 1);


		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 1308623905);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 3825239073);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		// DS_R handles begin
		// READER_0
		csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +    0));
		csrw_ss(S_STRIDE_READER_0_0, 8);
		csrw_ss(T_BOUND_READER_0_0,    4);
		csrw_ss(T_STRIDE_READER_0_0,   64);
		// READER_1
		csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +    8));
		csrw_ss(S_STRIDE_READER_1_0, 8);
		csrw_ss(T_BOUND_READER_1_0,    4);
		csrw_ss(T_STRIDE_READER_1_0,   64);
		// READER_2
		csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   16));
		csrw_ss(S_STRIDE_READER_2_0, 8);
		csrw_ss(T_BOUND_READER_2_0,    4);
		csrw_ss(T_STRIDE_READER_2_0,   64);
		// READER_3
		csrw_ss(BASE_PTR_READER_3_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   24));
		csrw_ss(S_STRIDE_READER_3_0, 8);
		csrw_ss(T_BOUND_READER_3_0,    4);
		csrw_ss(T_STRIDE_READER_3_0,   64);
		// READER_4
		csrw_ss(BASE_PTR_READER_4_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   32));
		csrw_ss(S_STRIDE_READER_4_0, 8);
		csrw_ss(T_BOUND_READER_4_0,    4);
		csrw_ss(T_STRIDE_READER_4_0,   64);
		// READER_5
		csrw_ss(BASE_PTR_READER_5_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   40));
		csrw_ss(S_STRIDE_READER_5_0, 8);
		csrw_ss(T_BOUND_READER_5_0,    4);
		csrw_ss(T_STRIDE_READER_5_0,   64);
		// READER_6
		csrw_ss(BASE_PTR_READER_6_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   48));
		csrw_ss(S_STRIDE_READER_6_0, 8);
		csrw_ss(T_BOUND_READER_6_0,    4);
		csrw_ss(T_STRIDE_READER_6_0,   64);
		// READER_7
		csrw_ss(BASE_PTR_READER_7_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   56));
		csrw_ss(S_STRIDE_READER_7_0, 8);
		csrw_ss(T_BOUND_READER_7_0,    4);
		csrw_ss(T_STRIDE_READER_7_0,   64);
		// DS_R handles end
		csrw_ss(STREAMER_START_CSR, 1);


		mcycle_timestamps[2] = snrt_mcycle();
		// DS_R handles begin
		// READER_0
		csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data + 16624));
		csrw_ss(S_STRIDE_READER_0_0, 8);
		csrw_ss(T_BOUND_READER_0_0, 1039);
		csrw_ss(T_STRIDE_READER_0_0,    8);
		csrw_ss(T_BOUND_READER_0_1,   64);
		csrw_ss(T_STRIDE_READER_0_1,    0);
		csrw_ss(T_BOUND_READER_0_2,    1);
		csrw_ss(T_STRIDE_READER_0_2,    0);
		csrw_ss(T_BOUND_READER_0_3,    1);
		csrw_ss(T_STRIDE_READER_0_3,    0);
		csrw_ss(T_BOUND_READER_0_4,    1);
		csrw_ss(T_STRIDE_READER_0_4,    0);
		csrw_ss(T_BOUND_READER_0_5,    1);
		csrw_ss(T_STRIDE_READER_0_5,    0);
		csrw_ss(T_BOUND_READER_0_6,    1);
		csrw_ss(T_STRIDE_READER_0_6,    0);
		csrw_ss(T_BOUND_READER_0_7,    1);
		csrw_ss(T_STRIDE_READER_0_7,    0);
		// READER_1
		csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_1_0, 8);
		csrw_ss(T_BOUND_READER_1_0,    0);
		csrw_ss(T_STRIDE_READER_1_0,    0);
		csrw_ss(T_BOUND_READER_1_1,    0);
		csrw_ss(T_STRIDE_READER_1_1,    0);
		csrw_ss(T_BOUND_READER_1_2,    0);
		csrw_ss(T_STRIDE_READER_1_2,    0);
		csrw_ss(T_BOUND_READER_1_3,    0);
		csrw_ss(T_STRIDE_READER_1_3,    0);
		csrw_ss(T_BOUND_READER_1_4,    0);
		csrw_ss(T_STRIDE_READER_1_4,    0);
		csrw_ss(T_BOUND_READER_1_5,    0);
		csrw_ss(T_STRIDE_READER_1_5,    0);
		csrw_ss(T_BOUND_READER_1_6,    0);
		csrw_ss(T_STRIDE_READER_1_6,    0);
		csrw_ss(T_BOUND_READER_1_7,    0);
		csrw_ss(T_STRIDE_READER_1_7,    0);
		// READER_2
		csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_2_0, 8);
		csrw_ss(T_BOUND_READER_2_0,    0);
		csrw_ss(T_STRIDE_READER_2_0,    0);
		csrw_ss(T_BOUND_READER_2_1,    0);
		csrw_ss(T_STRIDE_READER_2_1,    0);
		csrw_ss(T_BOUND_READER_2_2,    0);
		csrw_ss(T_STRIDE_READER_2_2,    0);
		csrw_ss(T_BOUND_READER_2_3,    0);
		csrw_ss(T_STRIDE_READER_2_3,    0);
		csrw_ss(T_BOUND_READER_2_4,    0);
		csrw_ss(T_STRIDE_READER_2_4,    0);
		csrw_ss(T_BOUND_READER_2_5,    0);
		csrw_ss(T_STRIDE_READER_2_5,    0);
		csrw_ss(T_BOUND_READER_2_6,    0);
		csrw_ss(T_STRIDE_READER_2_6,    0);
		csrw_ss(T_BOUND_READER_2_7,    0);
		csrw_ss(T_STRIDE_READER_2_7,    0);
		// READER_3
		csrw_ss(BASE_PTR_READER_3_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_3_0, 8);
		csrw_ss(T_BOUND_READER_3_0, 2078);
		csrw_ss(T_STRIDE_READER_3_0,    8);
		csrw_ss(T_BOUND_READER_3_1,   32);
		csrw_ss(T_STRIDE_READER_3_1,    0);
		csrw_ss(T_BOUND_READER_3_2,    1);
		csrw_ss(T_STRIDE_READER_3_2,    0);
		csrw_ss(T_BOUND_READER_3_3,    1);
		csrw_ss(T_STRIDE_READER_3_3,    0);
		csrw_ss(T_BOUND_READER_3_4,    1);
		csrw_ss(T_STRIDE_READER_3_4,    0);
		csrw_ss(T_BOUND_READER_3_5,    1);
		csrw_ss(T_STRIDE_READER_3_5,    0);
		csrw_ss(T_BOUND_READER_3_6,    1);
		csrw_ss(T_STRIDE_READER_3_6,    0);
		csrw_ss(T_BOUND_READER_3_7,    1);
		csrw_ss(T_STRIDE_READER_3_7,    0);
		// READER_4
		csrw_ss(BASE_PTR_READER_4_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_4_0, 8);
		csrw_ss(T_BOUND_READER_4_0,    0);
		csrw_ss(T_STRIDE_READER_4_0,    0);
		csrw_ss(T_BOUND_READER_4_1,    0);
		csrw_ss(T_STRIDE_READER_4_1,    0);
		csrw_ss(T_BOUND_READER_4_2,    0);
		csrw_ss(T_STRIDE_READER_4_2,    0);
		csrw_ss(T_BOUND_READER_4_3,    0);
		csrw_ss(T_STRIDE_READER_4_3,    0);
		csrw_ss(T_BOUND_READER_4_4,    0);
		csrw_ss(T_STRIDE_READER_4_4,    0);
		csrw_ss(T_BOUND_READER_4_5,    0);
		csrw_ss(T_STRIDE_READER_4_5,    0);
		csrw_ss(T_BOUND_READER_4_6,    0);
		csrw_ss(T_STRIDE_READER_4_6,    0);
		csrw_ss(T_BOUND_READER_4_7,    0);
		csrw_ss(T_STRIDE_READER_4_7,    0);
		// READER_5
		csrw_ss(BASE_PTR_READER_5_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_5_0, 8);
		csrw_ss(T_BOUND_READER_5_0,    0);
		csrw_ss(T_STRIDE_READER_5_0,    0);
		csrw_ss(T_BOUND_READER_5_1,    0);
		csrw_ss(T_STRIDE_READER_5_1,    0);
		csrw_ss(T_BOUND_READER_5_2,    0);
		csrw_ss(T_STRIDE_READER_5_2,    0);
		csrw_ss(T_BOUND_READER_5_3,    0);
		csrw_ss(T_STRIDE_READER_5_3,    0);
		csrw_ss(T_BOUND_READER_5_4,    0);
		csrw_ss(T_STRIDE_READER_5_4,    0);
		csrw_ss(T_BOUND_READER_5_5,    0);
		csrw_ss(T_STRIDE_READER_5_5,    0);
		csrw_ss(T_BOUND_READER_5_6,    0);
		csrw_ss(T_STRIDE_READER_5_6,    0);
		csrw_ss(T_BOUND_READER_5_7,    0);
		csrw_ss(T_STRIDE_READER_5_7,    0);
		// READER_6
		csrw_ss(BASE_PTR_READER_6_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_6_0, 8);
		csrw_ss(T_BOUND_READER_6_0,    0);
		csrw_ss(T_STRIDE_READER_6_0,    0);
		csrw_ss(T_BOUND_READER_6_1,    0);
		csrw_ss(T_STRIDE_READER_6_1,    0);
		csrw_ss(T_BOUND_READER_6_2,    0);
		csrw_ss(T_STRIDE_READER_6_2,    0);
		csrw_ss(T_BOUND_READER_6_3,    0);
		csrw_ss(T_STRIDE_READER_6_3,    0);
		csrw_ss(T_BOUND_READER_6_4,    0);
		csrw_ss(T_STRIDE_READER_6_4,    0);
		csrw_ss(T_BOUND_READER_6_5,    0);
		csrw_ss(T_STRIDE_READER_6_5,    0);
		csrw_ss(T_BOUND_READER_6_6,    0);
		csrw_ss(T_STRIDE_READER_6_6,    0);
		csrw_ss(T_BOUND_READER_6_7,    0);
		csrw_ss(T_STRIDE_READER_6_7,    0);
		// READER_7
		csrw_ss(BASE_PTR_READER_7_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_7_0, 8);
		csrw_ss(T_BOUND_READER_7_0,    0);
		csrw_ss(T_STRIDE_READER_7_0,    0);
		csrw_ss(T_BOUND_READER_7_1,    0);
		csrw_ss(T_STRIDE_READER_7_1,    0);
		csrw_ss(T_BOUND_READER_7_2,    0);
		csrw_ss(T_STRIDE_READER_7_2,    0);
		csrw_ss(T_BOUND_READER_7_3,    0);
		csrw_ss(T_STRIDE_READER_7_3,    0);
		csrw_ss(T_BOUND_READER_7_4,    0);
		csrw_ss(T_STRIDE_READER_7_4,    0);
		csrw_ss(T_BOUND_READER_7_5,    0);
		csrw_ss(T_STRIDE_READER_7_5,    0);
		csrw_ss(T_BOUND_READER_7_6,    0);
		csrw_ss(T_STRIDE_READER_7_6,    0);
		csrw_ss(T_BOUND_READER_7_7,    0);
		csrw_ss(T_STRIDE_READER_7_7,    0);
		// DS_R handles end
		// DS_W handles begin
		// WRITER_0
		csrw_ss(BASE_PTR_WRITER_0_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_0_0, 8);
		csrw_ss(T_BOUND_WRITER_0_0,    0);
		csrw_ss(T_STRIDE_WRITER_0_0,    0);
		csrw_ss(T_BOUND_WRITER_0_1,    0);
		csrw_ss(T_STRIDE_WRITER_0_1,    0);
		csrw_ss(T_BOUND_WRITER_0_2,    0);
		csrw_ss(T_STRIDE_WRITER_0_2,    0);
		csrw_ss(T_BOUND_WRITER_0_3,    0);
		csrw_ss(T_STRIDE_WRITER_0_3,    0);
		csrw_ss(T_BOUND_WRITER_0_4,    0);
		csrw_ss(T_STRIDE_WRITER_0_4,    0);
		csrw_ss(T_BOUND_WRITER_0_5,    0);
		csrw_ss(T_STRIDE_WRITER_0_5,    0);
		csrw_ss(T_BOUND_WRITER_0_6,    0);
		csrw_ss(T_STRIDE_WRITER_0_6,    0);
		csrw_ss(T_BOUND_WRITER_0_7,    0);
		csrw_ss(T_STRIDE_WRITER_0_7,    0);
		// WRITER_1
		csrw_ss(BASE_PTR_WRITER_1_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_1_0, 8);
		csrw_ss(T_BOUND_WRITER_1_0,    0);
		csrw_ss(T_STRIDE_WRITER_1_0,    0);
		csrw_ss(T_BOUND_WRITER_1_1,    0);
		csrw_ss(T_STRIDE_WRITER_1_1,    0);
		csrw_ss(T_BOUND_WRITER_1_2,    0);
		csrw_ss(T_STRIDE_WRITER_1_2,    0);
		csrw_ss(T_BOUND_WRITER_1_3,    0);
		csrw_ss(T_STRIDE_WRITER_1_3,    0);
		csrw_ss(T_BOUND_WRITER_1_4,    0);
		csrw_ss(T_STRIDE_WRITER_1_4,    0);
		csrw_ss(T_BOUND_WRITER_1_5,    0);
		csrw_ss(T_STRIDE_WRITER_1_5,    0);
		csrw_ss(T_BOUND_WRITER_1_6,    0);
		csrw_ss(T_STRIDE_WRITER_1_6,    0);
		csrw_ss(T_BOUND_WRITER_1_7,    0);
		csrw_ss(T_STRIDE_WRITER_1_7,    0);
		// WRITER_2
		csrw_ss(BASE_PTR_WRITER_2_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_2_0, 8);
		csrw_ss(T_BOUND_WRITER_2_0,    0);
		csrw_ss(T_STRIDE_WRITER_2_0,    0);
		csrw_ss(T_BOUND_WRITER_2_1,    0);
		csrw_ss(T_STRIDE_WRITER_2_1,    0);
		csrw_ss(T_BOUND_WRITER_2_2,    0);
		csrw_ss(T_STRIDE_WRITER_2_2,    0);
		csrw_ss(T_BOUND_WRITER_2_3,    0);
		csrw_ss(T_STRIDE_WRITER_2_3,    0);
		csrw_ss(T_BOUND_WRITER_2_4,    0);
		csrw_ss(T_STRIDE_WRITER_2_4,    0);
		csrw_ss(T_BOUND_WRITER_2_5,    0);
		csrw_ss(T_STRIDE_WRITER_2_5,    0);
		csrw_ss(T_BOUND_WRITER_2_6,    0);
		csrw_ss(T_STRIDE_WRITER_2_6,    0);
		csrw_ss(T_BOUND_WRITER_2_7,    0);
		csrw_ss(T_STRIDE_WRITER_2_7,    0);
		// WRITER_3
		csrw_ss(BASE_PTR_WRITER_3_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_3_0, 8);
		csrw_ss(T_BOUND_WRITER_3_0,    0);
		csrw_ss(T_STRIDE_WRITER_3_0,    0);
		csrw_ss(T_BOUND_WRITER_3_1,    0);
		csrw_ss(T_STRIDE_WRITER_3_1,    0);
		csrw_ss(T_BOUND_WRITER_3_2,    0);
		csrw_ss(T_STRIDE_WRITER_3_2,    0);
		csrw_ss(T_BOUND_WRITER_3_3,    0);
		csrw_ss(T_STRIDE_WRITER_3_3,    0);
		csrw_ss(T_BOUND_WRITER_3_4,    0);
		csrw_ss(T_STRIDE_WRITER_3_4,    0);
		csrw_ss(T_BOUND_WRITER_3_5,    0);
		csrw_ss(T_STRIDE_WRITER_3_5,    0);
		csrw_ss(T_BOUND_WRITER_3_6,    0);
		csrw_ss(T_STRIDE_WRITER_3_6,    0);
		csrw_ss(T_BOUND_WRITER_3_7,    0);
		csrw_ss(T_STRIDE_WRITER_3_7,    0);
		// WRITER_4
		csrw_ss(BASE_PTR_WRITER_4_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_4_0, 8);
		csrw_ss(T_BOUND_WRITER_4_0, 2078);
		csrw_ss(T_STRIDE_WRITER_4_0,    8);
		csrw_ss(T_BOUND_WRITER_4_1,   32);
		csrw_ss(T_STRIDE_WRITER_4_1,    0);
		csrw_ss(T_BOUND_WRITER_4_2,    1);
		csrw_ss(T_STRIDE_WRITER_4_2,    0);
		csrw_ss(T_BOUND_WRITER_4_3,    1);
		csrw_ss(T_STRIDE_WRITER_4_3,    0);
		csrw_ss(T_BOUND_WRITER_4_4,    1);
		csrw_ss(T_STRIDE_WRITER_4_4,    0);
		csrw_ss(T_BOUND_WRITER_4_5,    1);
		csrw_ss(T_STRIDE_WRITER_4_5,    0);
		csrw_ss(T_BOUND_WRITER_4_6,    1);
		csrw_ss(T_STRIDE_WRITER_4_6,    0);
		csrw_ss(T_BOUND_WRITER_4_7,    1);
		csrw_ss(T_STRIDE_WRITER_4_7,    0);
		// WRITER_5
		csrw_ss(BASE_PTR_WRITER_5_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_5_0, 8);
		csrw_ss(T_BOUND_WRITER_5_0,    0);
		csrw_ss(T_STRIDE_WRITER_5_0,    0);
		csrw_ss(T_BOUND_WRITER_5_1,    0);
		csrw_ss(T_STRIDE_WRITER_5_1,    0);
		csrw_ss(T_BOUND_WRITER_5_2,    0);
		csrw_ss(T_STRIDE_WRITER_5_2,    0);
		csrw_ss(T_BOUND_WRITER_5_3,    0);
		csrw_ss(T_STRIDE_WRITER_5_3,    0);
		csrw_ss(T_BOUND_WRITER_5_4,    0);
		csrw_ss(T_STRIDE_WRITER_5_4,    0);
		csrw_ss(T_BOUND_WRITER_5_5,    0);
		csrw_ss(T_STRIDE_WRITER_5_5,    0);
		csrw_ss(T_BOUND_WRITER_5_6,    0);
		csrw_ss(T_STRIDE_WRITER_5_6,    0);
		csrw_ss(T_BOUND_WRITER_5_7,    0);
		csrw_ss(T_STRIDE_WRITER_5_7,    0);
		// WRITER_6
		csrw_ss(BASE_PTR_WRITER_6_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_6_0, 8);
		csrw_ss(T_BOUND_WRITER_6_0,    0);
		csrw_ss(T_STRIDE_WRITER_6_0,    0);
		csrw_ss(T_BOUND_WRITER_6_1,    0);
		csrw_ss(T_STRIDE_WRITER_6_1,    0);
		csrw_ss(T_BOUND_WRITER_6_2,    0);
		csrw_ss(T_STRIDE_WRITER_6_2,    0);
		csrw_ss(T_BOUND_WRITER_6_3,    0);
		csrw_ss(T_STRIDE_WRITER_6_3,    0);
		csrw_ss(T_BOUND_WRITER_6_4,    0);
		csrw_ss(T_STRIDE_WRITER_6_4,    0);
		csrw_ss(T_BOUND_WRITER_6_5,    0);
		csrw_ss(T_STRIDE_WRITER_6_5,    0);
		csrw_ss(T_BOUND_WRITER_6_6,    0);
		csrw_ss(T_STRIDE_WRITER_6_6,    0);
		csrw_ss(T_BOUND_WRITER_6_7,    0);
		csrw_ss(T_STRIDE_WRITER_6_7,    0);
		// WRITER_7
		csrw_ss(BASE_PTR_WRITER_7_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_7_0, 8);
		csrw_ss(T_BOUND_WRITER_7_0,    0);
		csrw_ss(T_STRIDE_WRITER_7_0,    0);
		csrw_ss(T_BOUND_WRITER_7_1,    0);
		csrw_ss(T_STRIDE_WRITER_7_1,    0);
		csrw_ss(T_BOUND_WRITER_7_2,    0);
		csrw_ss(T_STRIDE_WRITER_7_2,    0);
		csrw_ss(T_BOUND_WRITER_7_3,    0);
		csrw_ss(T_STRIDE_WRITER_7_3,    0);
		csrw_ss(T_BOUND_WRITER_7_4,    0);
		csrw_ss(T_STRIDE_WRITER_7_4,    0);
		csrw_ss(T_BOUND_WRITER_7_5,    0);
		csrw_ss(T_STRIDE_WRITER_7_5,    0);
		csrw_ss(T_BOUND_WRITER_7_6,    0);
		csrw_ss(T_STRIDE_WRITER_7_6,    0);
		csrw_ss(T_BOUND_WRITER_7_7,    0);
		csrw_ss(T_STRIDE_WRITER_7_7,    0);
		// DS_W handles end
		csrw_ss(STREAMER_START_CSR, 1);


		mcycle_timestamps[3] = snrt_mcycle();
		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 1308656673);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		mcycle_timestamps[4] = snrt_mcycle();
}


void launch_cgra_0_config(int32_t delta_config_data, int32_t delta_comp_data, int32_t delta_store_data, uint32_t *mcycle_timestamps) {
		mcycle_timestamps[0] = snrt_mcycle();
		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 1275069473);
		csrw_ss(CGRA_CSR_ADDR_BASE +  1,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  2,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  3,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  4,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  5,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  6,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  7,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  8,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE +  9,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 10,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 11,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 12,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 13,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 14,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 15,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 16,        256);
		csrw_ss(CGRA_CSR_ADDR_BASE + 17,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 18,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 19,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 20,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 21,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 22,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 23,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 24,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 25,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 26,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 27,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 28,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 29,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 30,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 31,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 32,   50462976);
		csrw_ss(CGRA_CSR_ADDR_BASE + 33,    2127873);
		csrw_ss(CGRA_CSR_ADDR_BASE + 34,          1);
		csrw_ss(CGRA_CSR_ADDR_BASE + 35,          1);
		csrw_ss(CGRA_CSR_ADDR_BASE + 36,          1);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		mcycle_timestamps[1] = snrt_mcycle();
		// DS_R handles begin
		// READER_0
		csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +    0));
		csrw_ss(S_STRIDE_READER_0_0, 8);
		csrw_ss(T_BOUND_READER_0_0,    1);
		csrw_ss(T_STRIDE_READER_0_0,   64);
		csrw_ss(T_BOUND_READER_0_1,    1);
		csrw_ss(T_STRIDE_READER_0_1,    0);
		csrw_ss(T_BOUND_READER_0_2,    1);
		csrw_ss(T_STRIDE_READER_0_2,    0);
		csrw_ss(T_BOUND_READER_0_3,    1);
		csrw_ss(T_STRIDE_READER_0_3,    0);
		csrw_ss(T_BOUND_READER_0_4,    1);
		csrw_ss(T_STRIDE_READER_0_4,    0);
		csrw_ss(T_BOUND_READER_0_5,    1);
		csrw_ss(T_STRIDE_READER_0_5,    0);
		csrw_ss(T_BOUND_READER_0_6,    1);
		csrw_ss(T_STRIDE_READER_0_6,    0);
		csrw_ss(T_BOUND_READER_0_7,    1);
		csrw_ss(T_STRIDE_READER_0_7,    0);
		// READER_1
		csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +    8));
		csrw_ss(S_STRIDE_READER_1_0, 8);
		csrw_ss(T_BOUND_READER_1_0,    1);
		csrw_ss(T_STRIDE_READER_1_0,   64);
		csrw_ss(T_BOUND_READER_1_1,    1);
		csrw_ss(T_STRIDE_READER_1_1,    0);
		csrw_ss(T_BOUND_READER_1_2,    1);
		csrw_ss(T_STRIDE_READER_1_2,    0);
		csrw_ss(T_BOUND_READER_1_3,    1);
		csrw_ss(T_STRIDE_READER_1_3,    0);
		csrw_ss(T_BOUND_READER_1_4,    1);
		csrw_ss(T_STRIDE_READER_1_4,    0);
		csrw_ss(T_BOUND_READER_1_5,    1);
		csrw_ss(T_STRIDE_READER_1_5,    0);
		csrw_ss(T_BOUND_READER_1_6,    1);
		csrw_ss(T_STRIDE_READER_1_6,    0);
		csrw_ss(T_BOUND_READER_1_7,    1);
		csrw_ss(T_STRIDE_READER_1_7,    0);
		// READER_2
		csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   16));
		csrw_ss(S_STRIDE_READER_2_0, 8);
		csrw_ss(T_BOUND_READER_2_0,    1);
		csrw_ss(T_STRIDE_READER_2_0,   64);
		csrw_ss(T_BOUND_READER_2_1,    1);
		csrw_ss(T_STRIDE_READER_2_1,    0);
		csrw_ss(T_BOUND_READER_2_2,    1);
		csrw_ss(T_STRIDE_READER_2_2,    0);
		csrw_ss(T_BOUND_READER_2_3,    1);
		csrw_ss(T_STRIDE_READER_2_3,    0);
		csrw_ss(T_BOUND_READER_2_4,    1);
		csrw_ss(T_STRIDE_READER_2_4,    0);
		csrw_ss(T_BOUND_READER_2_5,    1);
		csrw_ss(T_STRIDE_READER_2_5,    0);
		csrw_ss(T_BOUND_READER_2_6,    1);
		csrw_ss(T_STRIDE_READER_2_6,    0);
		csrw_ss(T_BOUND_READER_2_7,    1);
		csrw_ss(T_STRIDE_READER_2_7,    0);
		// READER_3
		csrw_ss(BASE_PTR_READER_3_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   24));
		csrw_ss(S_STRIDE_READER_3_0, 8);
		csrw_ss(T_BOUND_READER_3_0,    1);
		csrw_ss(T_STRIDE_READER_3_0,   64);
		csrw_ss(T_BOUND_READER_3_1,    1);
		csrw_ss(T_STRIDE_READER_3_1,    0);
		csrw_ss(T_BOUND_READER_3_2,    1);
		csrw_ss(T_STRIDE_READER_3_2,    0);
		csrw_ss(T_BOUND_READER_3_3,    1);
		csrw_ss(T_STRIDE_READER_3_3,    0);
		csrw_ss(T_BOUND_READER_3_4,    1);
		csrw_ss(T_STRIDE_READER_3_4,    0);
		csrw_ss(T_BOUND_READER_3_5,    1);
		csrw_ss(T_STRIDE_READER_3_5,    0);
		csrw_ss(T_BOUND_READER_3_6,    1);
		csrw_ss(T_STRIDE_READER_3_6,    0);
		csrw_ss(T_BOUND_READER_3_7,    1);
		csrw_ss(T_STRIDE_READER_3_7,    0);
		// READER_4
		csrw_ss(BASE_PTR_READER_4_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   32));
		csrw_ss(S_STRIDE_READER_4_0, 8);
		csrw_ss(T_BOUND_READER_4_0,    1);
		csrw_ss(T_STRIDE_READER_4_0,   64);
		csrw_ss(T_BOUND_READER_4_1,    1);
		csrw_ss(T_STRIDE_READER_4_1,    0);
		csrw_ss(T_BOUND_READER_4_2,    1);
		csrw_ss(T_STRIDE_READER_4_2,    0);
		csrw_ss(T_BOUND_READER_4_3,    1);
		csrw_ss(T_STRIDE_READER_4_3,    0);
		csrw_ss(T_BOUND_READER_4_4,    1);
		csrw_ss(T_STRIDE_READER_4_4,    0);
		csrw_ss(T_BOUND_READER_4_5,    1);
		csrw_ss(T_STRIDE_READER_4_5,    0);
		csrw_ss(T_BOUND_READER_4_6,    1);
		csrw_ss(T_STRIDE_READER_4_6,    0);
		csrw_ss(T_BOUND_READER_4_7,    1);
		csrw_ss(T_STRIDE_READER_4_7,    0);
		// READER_5
		csrw_ss(BASE_PTR_READER_5_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   40));
		csrw_ss(S_STRIDE_READER_5_0, 8);
		csrw_ss(T_BOUND_READER_5_0,    1);
		csrw_ss(T_STRIDE_READER_5_0,   64);
		csrw_ss(T_BOUND_READER_5_1,    1);
		csrw_ss(T_STRIDE_READER_5_1,    0);
		csrw_ss(T_BOUND_READER_5_2,    1);
		csrw_ss(T_STRIDE_READER_5_2,    0);
		csrw_ss(T_BOUND_READER_5_3,    1);
		csrw_ss(T_STRIDE_READER_5_3,    0);
		csrw_ss(T_BOUND_READER_5_4,    1);
		csrw_ss(T_STRIDE_READER_5_4,    0);
		csrw_ss(T_BOUND_READER_5_5,    1);
		csrw_ss(T_STRIDE_READER_5_5,    0);
		csrw_ss(T_BOUND_READER_5_6,    1);
		csrw_ss(T_STRIDE_READER_5_6,    0);
		csrw_ss(T_BOUND_READER_5_7,    1);
		csrw_ss(T_STRIDE_READER_5_7,    0);
		// READER_6
		csrw_ss(BASE_PTR_READER_6_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   48));
		csrw_ss(S_STRIDE_READER_6_0, 8);
		csrw_ss(T_BOUND_READER_6_0,    1);
		csrw_ss(T_STRIDE_READER_6_0,   64);
		csrw_ss(T_BOUND_READER_6_1,    1);
		csrw_ss(T_STRIDE_READER_6_1,    0);
		csrw_ss(T_BOUND_READER_6_2,    1);
		csrw_ss(T_STRIDE_READER_6_2,    0);
		csrw_ss(T_BOUND_READER_6_3,    1);
		csrw_ss(T_STRIDE_READER_6_3,    0);
		csrw_ss(T_BOUND_READER_6_4,    1);
		csrw_ss(T_STRIDE_READER_6_4,    0);
		csrw_ss(T_BOUND_READER_6_5,    1);
		csrw_ss(T_STRIDE_READER_6_5,    0);
		csrw_ss(T_BOUND_READER_6_6,    1);
		csrw_ss(T_STRIDE_READER_6_6,    0);
		csrw_ss(T_BOUND_READER_6_7,    1);
		csrw_ss(T_STRIDE_READER_6_7,    0);
		// READER_7
		csrw_ss(BASE_PTR_READER_7_LOW, (uint32_t)(snrt_l1_next() + delta_config_data +   56));
		csrw_ss(S_STRIDE_READER_7_0, 8);
		csrw_ss(T_BOUND_READER_7_0,    1);
		csrw_ss(T_STRIDE_READER_7_0,   64);
		csrw_ss(T_BOUND_READER_7_1,    1);
		csrw_ss(T_STRIDE_READER_7_1,    0);
		csrw_ss(T_BOUND_READER_7_2,    1);
		csrw_ss(T_STRIDE_READER_7_2,    0);
		csrw_ss(T_BOUND_READER_7_3,    1);
		csrw_ss(T_STRIDE_READER_7_3,    0);
		csrw_ss(T_BOUND_READER_7_4,    1);
		csrw_ss(T_STRIDE_READER_7_4,    0);
		csrw_ss(T_BOUND_READER_7_5,    1);
		csrw_ss(T_STRIDE_READER_7_5,    0);
		csrw_ss(T_BOUND_READER_7_6,    1);
		csrw_ss(T_STRIDE_READER_7_6,    0);
		csrw_ss(T_BOUND_READER_7_7,    1);
		csrw_ss(T_STRIDE_READER_7_7,    0);
		// DS_R handles end
		csrw_ss(STREAMER_START_CSR, 1);


		// DS_R handles begin
		// READER_0
		csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +    0));
		csrw_ss(S_STRIDE_READER_0_0, 8);
		csrw_ss(T_BOUND_READER_0_0,    2);
		csrw_ss(T_STRIDE_READER_0_0,   64);
		csrw_ss(T_BOUND_READER_0_1,    1);
		csrw_ss(T_STRIDE_READER_0_1,    0);
		csrw_ss(T_BOUND_READER_0_2,    1);
		csrw_ss(T_STRIDE_READER_0_2,    0);
		csrw_ss(T_BOUND_READER_0_3,    1);
		csrw_ss(T_STRIDE_READER_0_3,    0);
		csrw_ss(T_BOUND_READER_0_4,    1);
		csrw_ss(T_STRIDE_READER_0_4,    0);
		csrw_ss(T_BOUND_READER_0_5,    1);
		csrw_ss(T_STRIDE_READER_0_5,    0);
		csrw_ss(T_BOUND_READER_0_6,    1);
		csrw_ss(T_STRIDE_READER_0_6,    0);
		csrw_ss(T_BOUND_READER_0_7,    1);
		csrw_ss(T_STRIDE_READER_0_7,    0);
		// READER_1
		csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +    8));
		csrw_ss(S_STRIDE_READER_1_0, 8);
		csrw_ss(T_BOUND_READER_1_0,    2);
		csrw_ss(T_STRIDE_READER_1_0,   64);
		csrw_ss(T_BOUND_READER_1_1,    1);
		csrw_ss(T_STRIDE_READER_1_1,    0);
		csrw_ss(T_BOUND_READER_1_2,    1);
		csrw_ss(T_STRIDE_READER_1_2,    0);
		csrw_ss(T_BOUND_READER_1_3,    1);
		csrw_ss(T_STRIDE_READER_1_3,    0);
		csrw_ss(T_BOUND_READER_1_4,    1);
		csrw_ss(T_STRIDE_READER_1_4,    0);
		csrw_ss(T_BOUND_READER_1_5,    1);
		csrw_ss(T_STRIDE_READER_1_5,    0);
		csrw_ss(T_BOUND_READER_1_6,    1);
		csrw_ss(T_STRIDE_READER_1_6,    0);
		csrw_ss(T_BOUND_READER_1_7,    1);
		csrw_ss(T_STRIDE_READER_1_7,    0);
		// READER_2
		csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   16));
		csrw_ss(S_STRIDE_READER_2_0, 8);
		csrw_ss(T_BOUND_READER_2_0,    2);
		csrw_ss(T_STRIDE_READER_2_0,   64);
		csrw_ss(T_BOUND_READER_2_1,    1);
		csrw_ss(T_STRIDE_READER_2_1,    0);
		csrw_ss(T_BOUND_READER_2_2,    1);
		csrw_ss(T_STRIDE_READER_2_2,    0);
		csrw_ss(T_BOUND_READER_2_3,    1);
		csrw_ss(T_STRIDE_READER_2_3,    0);
		csrw_ss(T_BOUND_READER_2_4,    1);
		csrw_ss(T_STRIDE_READER_2_4,    0);
		csrw_ss(T_BOUND_READER_2_5,    1);
		csrw_ss(T_STRIDE_READER_2_5,    0);
		csrw_ss(T_BOUND_READER_2_6,    1);
		csrw_ss(T_STRIDE_READER_2_6,    0);
		csrw_ss(T_BOUND_READER_2_7,    1);
		csrw_ss(T_STRIDE_READER_2_7,    0);
		// READER_3
		csrw_ss(BASE_PTR_READER_3_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   24));
		csrw_ss(S_STRIDE_READER_3_0, 8);
		csrw_ss(T_BOUND_READER_3_0,    2);
		csrw_ss(T_STRIDE_READER_3_0,   64);
		csrw_ss(T_BOUND_READER_3_1,    1);
		csrw_ss(T_STRIDE_READER_3_1,    0);
		csrw_ss(T_BOUND_READER_3_2,    1);
		csrw_ss(T_STRIDE_READER_3_2,    0);
		csrw_ss(T_BOUND_READER_3_3,    1);
		csrw_ss(T_STRIDE_READER_3_3,    0);
		csrw_ss(T_BOUND_READER_3_4,    1);
		csrw_ss(T_STRIDE_READER_3_4,    0);
		csrw_ss(T_BOUND_READER_3_5,    1);
		csrw_ss(T_STRIDE_READER_3_5,    0);
		csrw_ss(T_BOUND_READER_3_6,    1);
		csrw_ss(T_STRIDE_READER_3_6,    0);
		csrw_ss(T_BOUND_READER_3_7,    1);
		csrw_ss(T_STRIDE_READER_3_7,    0);
		// READER_4
		csrw_ss(BASE_PTR_READER_4_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   32));
		csrw_ss(S_STRIDE_READER_4_0, 8);
		csrw_ss(T_BOUND_READER_4_0,    2);
		csrw_ss(T_STRIDE_READER_4_0,   64);
		csrw_ss(T_BOUND_READER_4_1,    1);
		csrw_ss(T_STRIDE_READER_4_1,    0);
		csrw_ss(T_BOUND_READER_4_2,    1);
		csrw_ss(T_STRIDE_READER_4_2,    0);
		csrw_ss(T_BOUND_READER_4_3,    1);
		csrw_ss(T_STRIDE_READER_4_3,    0);
		csrw_ss(T_BOUND_READER_4_4,    1);
		csrw_ss(T_STRIDE_READER_4_4,    0);
		csrw_ss(T_BOUND_READER_4_5,    1);
		csrw_ss(T_STRIDE_READER_4_5,    0);
		csrw_ss(T_BOUND_READER_4_6,    1);
		csrw_ss(T_STRIDE_READER_4_6,    0);
		csrw_ss(T_BOUND_READER_4_7,    1);
		csrw_ss(T_STRIDE_READER_4_7,    0);
		// READER_5
		csrw_ss(BASE_PTR_READER_5_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   40));
		csrw_ss(S_STRIDE_READER_5_0, 8);
		csrw_ss(T_BOUND_READER_5_0,    2);
		csrw_ss(T_STRIDE_READER_5_0,   64);
		csrw_ss(T_BOUND_READER_5_1,    1);
		csrw_ss(T_STRIDE_READER_5_1,    0);
		csrw_ss(T_BOUND_READER_5_2,    1);
		csrw_ss(T_STRIDE_READER_5_2,    0);
		csrw_ss(T_BOUND_READER_5_3,    1);
		csrw_ss(T_STRIDE_READER_5_3,    0);
		csrw_ss(T_BOUND_READER_5_4,    1);
		csrw_ss(T_STRIDE_READER_5_4,    0);
		csrw_ss(T_BOUND_READER_5_5,    1);
		csrw_ss(T_STRIDE_READER_5_5,    0);
		csrw_ss(T_BOUND_READER_5_6,    1);
		csrw_ss(T_STRIDE_READER_5_6,    0);
		csrw_ss(T_BOUND_READER_5_7,    1);
		csrw_ss(T_STRIDE_READER_5_7,    0);
		// READER_6
		csrw_ss(BASE_PTR_READER_6_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   48));
		csrw_ss(S_STRIDE_READER_6_0, 8);
		csrw_ss(T_BOUND_READER_6_0,    2);
		csrw_ss(T_STRIDE_READER_6_0,   64);
		csrw_ss(T_BOUND_READER_6_1,    1);
		csrw_ss(T_STRIDE_READER_6_1,    0);
		csrw_ss(T_BOUND_READER_6_2,    1);
		csrw_ss(T_STRIDE_READER_6_2,    0);
		csrw_ss(T_BOUND_READER_6_3,    1);
		csrw_ss(T_STRIDE_READER_6_3,    0);
		csrw_ss(T_BOUND_READER_6_4,    1);
		csrw_ss(T_STRIDE_READER_6_4,    0);
		csrw_ss(T_BOUND_READER_6_5,    1);
		csrw_ss(T_STRIDE_READER_6_5,    0);
		csrw_ss(T_BOUND_READER_6_6,    1);
		csrw_ss(T_STRIDE_READER_6_6,    0);
		csrw_ss(T_BOUND_READER_6_7,    1);
		csrw_ss(T_STRIDE_READER_6_7,    0);
		// READER_7
		csrw_ss(BASE_PTR_READER_7_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_0 +   56));
		csrw_ss(S_STRIDE_READER_7_0, 8);
		csrw_ss(T_BOUND_READER_7_0,    2);
		csrw_ss(T_STRIDE_READER_7_0,   64);
		csrw_ss(T_BOUND_READER_7_1,    1);
		csrw_ss(T_STRIDE_READER_7_1,    0);
		csrw_ss(T_BOUND_READER_7_2,    1);
		csrw_ss(T_STRIDE_READER_7_2,    0);
		csrw_ss(T_BOUND_READER_7_3,    1);
		csrw_ss(T_STRIDE_READER_7_3,    0);
		csrw_ss(T_BOUND_READER_7_4,    1);
		csrw_ss(T_STRIDE_READER_7_4,    0);
		csrw_ss(T_BOUND_READER_7_5,    1);
		csrw_ss(T_STRIDE_READER_7_5,    0);
		csrw_ss(T_BOUND_READER_7_6,    1);
		csrw_ss(T_STRIDE_READER_7_6,    0);
		csrw_ss(T_BOUND_READER_7_7,    1);
		csrw_ss(T_STRIDE_READER_7_7,    0);
		// DS_R handles end
		csrw_ss(STREAMER_START_CSR, 1);


		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 1308623905);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 3825239073);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		// DS_R handles begin
		// READER_0
		csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +    0));
		csrw_ss(S_STRIDE_READER_0_0, 8);
		csrw_ss(T_BOUND_READER_0_0,    4);
		csrw_ss(T_STRIDE_READER_0_0,   64);
		csrw_ss(T_BOUND_READER_0_1,    1);
		csrw_ss(T_STRIDE_READER_0_1,    0);
		csrw_ss(T_BOUND_READER_0_2,    1);
		csrw_ss(T_STRIDE_READER_0_2,    0);
		csrw_ss(T_BOUND_READER_0_3,    1);
		csrw_ss(T_STRIDE_READER_0_3,    0);
		csrw_ss(T_BOUND_READER_0_4,    1);
		csrw_ss(T_STRIDE_READER_0_4,    0);
		csrw_ss(T_BOUND_READER_0_5,    1);
		csrw_ss(T_STRIDE_READER_0_5,    0);
		csrw_ss(T_BOUND_READER_0_6,    1);
		csrw_ss(T_STRIDE_READER_0_6,    0);
		csrw_ss(T_BOUND_READER_0_7,    1);
		csrw_ss(T_STRIDE_READER_0_7,    0);
		// READER_1
		csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +    8));
		csrw_ss(S_STRIDE_READER_1_0, 8);
		csrw_ss(T_BOUND_READER_1_0,    4);
		csrw_ss(T_STRIDE_READER_1_0,   64);
		csrw_ss(T_BOUND_READER_1_1,    1);
		csrw_ss(T_STRIDE_READER_1_1,    0);
		csrw_ss(T_BOUND_READER_1_2,    1);
		csrw_ss(T_STRIDE_READER_1_2,    0);
		csrw_ss(T_BOUND_READER_1_3,    1);
		csrw_ss(T_STRIDE_READER_1_3,    0);
		csrw_ss(T_BOUND_READER_1_4,    1);
		csrw_ss(T_STRIDE_READER_1_4,    0);
		csrw_ss(T_BOUND_READER_1_5,    1);
		csrw_ss(T_STRIDE_READER_1_5,    0);
		csrw_ss(T_BOUND_READER_1_6,    1);
		csrw_ss(T_STRIDE_READER_1_6,    0);
		csrw_ss(T_BOUND_READER_1_7,    1);
		csrw_ss(T_STRIDE_READER_1_7,    0);
		// READER_2
		csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   16));
		csrw_ss(S_STRIDE_READER_2_0, 8);
		csrw_ss(T_BOUND_READER_2_0,    4);
		csrw_ss(T_STRIDE_READER_2_0,   64);
		csrw_ss(T_BOUND_READER_2_1,    1);
		csrw_ss(T_STRIDE_READER_2_1,    0);
		csrw_ss(T_BOUND_READER_2_2,    1);
		csrw_ss(T_STRIDE_READER_2_2,    0);
		csrw_ss(T_BOUND_READER_2_3,    1);
		csrw_ss(T_STRIDE_READER_2_3,    0);
		csrw_ss(T_BOUND_READER_2_4,    1);
		csrw_ss(T_STRIDE_READER_2_4,    0);
		csrw_ss(T_BOUND_READER_2_5,    1);
		csrw_ss(T_STRIDE_READER_2_5,    0);
		csrw_ss(T_BOUND_READER_2_6,    1);
		csrw_ss(T_STRIDE_READER_2_6,    0);
		csrw_ss(T_BOUND_READER_2_7,    1);
		csrw_ss(T_STRIDE_READER_2_7,    0);
		// READER_3
		csrw_ss(BASE_PTR_READER_3_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   24));
		csrw_ss(S_STRIDE_READER_3_0, 8);
		csrw_ss(T_BOUND_READER_3_0,    4);
		csrw_ss(T_STRIDE_READER_3_0,   64);
		csrw_ss(T_BOUND_READER_3_1,    1);
		csrw_ss(T_STRIDE_READER_3_1,    0);
		csrw_ss(T_BOUND_READER_3_2,    1);
		csrw_ss(T_STRIDE_READER_3_2,    0);
		csrw_ss(T_BOUND_READER_3_3,    1);
		csrw_ss(T_STRIDE_READER_3_3,    0);
		csrw_ss(T_BOUND_READER_3_4,    1);
		csrw_ss(T_STRIDE_READER_3_4,    0);
		csrw_ss(T_BOUND_READER_3_5,    1);
		csrw_ss(T_STRIDE_READER_3_5,    0);
		csrw_ss(T_BOUND_READER_3_6,    1);
		csrw_ss(T_STRIDE_READER_3_6,    0);
		csrw_ss(T_BOUND_READER_3_7,    1);
		csrw_ss(T_STRIDE_READER_3_7,    0);
		// READER_4
		csrw_ss(BASE_PTR_READER_4_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   32));
		csrw_ss(S_STRIDE_READER_4_0, 8);
		csrw_ss(T_BOUND_READER_4_0,    4);
		csrw_ss(T_STRIDE_READER_4_0,   64);
		csrw_ss(T_BOUND_READER_4_1,    1);
		csrw_ss(T_STRIDE_READER_4_1,    0);
		csrw_ss(T_BOUND_READER_4_2,    1);
		csrw_ss(T_STRIDE_READER_4_2,    0);
		csrw_ss(T_BOUND_READER_4_3,    1);
		csrw_ss(T_STRIDE_READER_4_3,    0);
		csrw_ss(T_BOUND_READER_4_4,    1);
		csrw_ss(T_STRIDE_READER_4_4,    0);
		csrw_ss(T_BOUND_READER_4_5,    1);
		csrw_ss(T_STRIDE_READER_4_5,    0);
		csrw_ss(T_BOUND_READER_4_6,    1);
		csrw_ss(T_STRIDE_READER_4_6,    0);
		csrw_ss(T_BOUND_READER_4_7,    1);
		csrw_ss(T_STRIDE_READER_4_7,    0);
		// READER_5
		csrw_ss(BASE_PTR_READER_5_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   40));
		csrw_ss(S_STRIDE_READER_5_0, 8);
		csrw_ss(T_BOUND_READER_5_0,    4);
		csrw_ss(T_STRIDE_READER_5_0,   64);
		csrw_ss(T_BOUND_READER_5_1,    1);
		csrw_ss(T_STRIDE_READER_5_1,    0);
		csrw_ss(T_BOUND_READER_5_2,    1);
		csrw_ss(T_STRIDE_READER_5_2,    0);
		csrw_ss(T_BOUND_READER_5_3,    1);
		csrw_ss(T_STRIDE_READER_5_3,    0);
		csrw_ss(T_BOUND_READER_5_4,    1);
		csrw_ss(T_STRIDE_READER_5_4,    0);
		csrw_ss(T_BOUND_READER_5_5,    1);
		csrw_ss(T_STRIDE_READER_5_5,    0);
		csrw_ss(T_BOUND_READER_5_6,    1);
		csrw_ss(T_STRIDE_READER_5_6,    0);
		csrw_ss(T_BOUND_READER_5_7,    1);
		csrw_ss(T_STRIDE_READER_5_7,    0);
		// READER_6
		csrw_ss(BASE_PTR_READER_6_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   48));
		csrw_ss(S_STRIDE_READER_6_0, 8);
		csrw_ss(T_BOUND_READER_6_0,    4);
		csrw_ss(T_STRIDE_READER_6_0,   64);
		csrw_ss(T_BOUND_READER_6_1,    1);
		csrw_ss(T_STRIDE_READER_6_1,    0);
		csrw_ss(T_BOUND_READER_6_2,    1);
		csrw_ss(T_STRIDE_READER_6_2,    0);
		csrw_ss(T_BOUND_READER_6_3,    1);
		csrw_ss(T_STRIDE_READER_6_3,    0);
		csrw_ss(T_BOUND_READER_6_4,    1);
		csrw_ss(T_STRIDE_READER_6_4,    0);
		csrw_ss(T_BOUND_READER_6_5,    1);
		csrw_ss(T_STRIDE_READER_6_5,    0);
		csrw_ss(T_BOUND_READER_6_6,    1);
		csrw_ss(T_STRIDE_READER_6_6,    0);
		csrw_ss(T_BOUND_READER_6_7,    1);
		csrw_ss(T_STRIDE_READER_6_7,    0);
		// READER_7
		csrw_ss(BASE_PTR_READER_7_LOW, (uint32_t)(snrt_l1_next() + delta_config_cmd_ss +   56));
		csrw_ss(S_STRIDE_READER_7_0, 8);
		csrw_ss(T_BOUND_READER_7_0,    4);
		csrw_ss(T_STRIDE_READER_7_0,   64);
		csrw_ss(T_BOUND_READER_7_1,    1);
		csrw_ss(T_STRIDE_READER_7_1,    0);
		csrw_ss(T_BOUND_READER_7_2,    1);
		csrw_ss(T_STRIDE_READER_7_2,    0);
		csrw_ss(T_BOUND_READER_7_3,    1);
		csrw_ss(T_STRIDE_READER_7_3,    0);
		csrw_ss(T_BOUND_READER_7_4,    1);
		csrw_ss(T_STRIDE_READER_7_4,    0);
		csrw_ss(T_BOUND_READER_7_5,    1);
		csrw_ss(T_STRIDE_READER_7_5,    0);
		csrw_ss(T_BOUND_READER_7_6,    1);
		csrw_ss(T_STRIDE_READER_7_6,    0);
		csrw_ss(T_BOUND_READER_7_7,    1);
		csrw_ss(T_STRIDE_READER_7_7,    0);
		// DS_R handles end
		csrw_ss(STREAMER_START_CSR, 1);


		mcycle_timestamps[2] = snrt_mcycle();
		// DS_R handles begin
		// READER_0
		csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data + 16624));
		csrw_ss(S_STRIDE_READER_0_0, 8);
		csrw_ss(T_BOUND_READER_0_0, 1039);
		csrw_ss(T_STRIDE_READER_0_0,    8);
		csrw_ss(T_BOUND_READER_0_1,   64);
		csrw_ss(T_STRIDE_READER_0_1,    0);
		csrw_ss(T_BOUND_READER_0_2,    1);
		csrw_ss(T_STRIDE_READER_0_2,    0);
		csrw_ss(T_BOUND_READER_0_3,    1);
		csrw_ss(T_STRIDE_READER_0_3,    0);
		csrw_ss(T_BOUND_READER_0_4,    1);
		csrw_ss(T_STRIDE_READER_0_4,    0);
		csrw_ss(T_BOUND_READER_0_5,    1);
		csrw_ss(T_STRIDE_READER_0_5,    0);
		csrw_ss(T_BOUND_READER_0_6,    1);
		csrw_ss(T_STRIDE_READER_0_6,    0);
		csrw_ss(T_BOUND_READER_0_7,    1);
		csrw_ss(T_STRIDE_READER_0_7,    0);
		// READER_1
		csrw_ss(BASE_PTR_READER_1_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_1_0, 8);
		csrw_ss(T_BOUND_READER_1_0,    0);
		csrw_ss(T_STRIDE_READER_1_0,    0);
		csrw_ss(T_BOUND_READER_1_1,    0);
		csrw_ss(T_STRIDE_READER_1_1,    0);
		csrw_ss(T_BOUND_READER_1_2,    0);
		csrw_ss(T_STRIDE_READER_1_2,    0);
		csrw_ss(T_BOUND_READER_1_3,    0);
		csrw_ss(T_STRIDE_READER_1_3,    0);
		csrw_ss(T_BOUND_READER_1_4,    0);
		csrw_ss(T_STRIDE_READER_1_4,    0);
		csrw_ss(T_BOUND_READER_1_5,    0);
		csrw_ss(T_STRIDE_READER_1_5,    0);
		csrw_ss(T_BOUND_READER_1_6,    0);
		csrw_ss(T_STRIDE_READER_1_6,    0);
		csrw_ss(T_BOUND_READER_1_7,    0);
		csrw_ss(T_STRIDE_READER_1_7,    0);
		// READER_2
		csrw_ss(BASE_PTR_READER_2_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_2_0, 8);
		csrw_ss(T_BOUND_READER_2_0,    0);
		csrw_ss(T_STRIDE_READER_2_0,    0);
		csrw_ss(T_BOUND_READER_2_1,    0);
		csrw_ss(T_STRIDE_READER_2_1,    0);
		csrw_ss(T_BOUND_READER_2_2,    0);
		csrw_ss(T_STRIDE_READER_2_2,    0);
		csrw_ss(T_BOUND_READER_2_3,    0);
		csrw_ss(T_STRIDE_READER_2_3,    0);
		csrw_ss(T_BOUND_READER_2_4,    0);
		csrw_ss(T_STRIDE_READER_2_4,    0);
		csrw_ss(T_BOUND_READER_2_5,    0);
		csrw_ss(T_STRIDE_READER_2_5,    0);
		csrw_ss(T_BOUND_READER_2_6,    0);
		csrw_ss(T_STRIDE_READER_2_6,    0);
		csrw_ss(T_BOUND_READER_2_7,    0);
		csrw_ss(T_STRIDE_READER_2_7,    0);
		// READER_3
		csrw_ss(BASE_PTR_READER_3_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_3_0, 8);
		csrw_ss(T_BOUND_READER_3_0, 2078);
		csrw_ss(T_STRIDE_READER_3_0,    8);
		csrw_ss(T_BOUND_READER_3_1,   32);
		csrw_ss(T_STRIDE_READER_3_1,    0);
		csrw_ss(T_BOUND_READER_3_2,    1);
		csrw_ss(T_STRIDE_READER_3_2,    0);
		csrw_ss(T_BOUND_READER_3_3,    1);
		csrw_ss(T_STRIDE_READER_3_3,    0);
		csrw_ss(T_BOUND_READER_3_4,    1);
		csrw_ss(T_STRIDE_READER_3_4,    0);
		csrw_ss(T_BOUND_READER_3_5,    1);
		csrw_ss(T_STRIDE_READER_3_5,    0);
		csrw_ss(T_BOUND_READER_3_6,    1);
		csrw_ss(T_STRIDE_READER_3_6,    0);
		csrw_ss(T_BOUND_READER_3_7,    1);
		csrw_ss(T_STRIDE_READER_3_7,    0);
		// READER_4
		csrw_ss(BASE_PTR_READER_4_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_4_0, 8);
		csrw_ss(T_BOUND_READER_4_0,    0);
		csrw_ss(T_STRIDE_READER_4_0,    0);
		csrw_ss(T_BOUND_READER_4_1,    0);
		csrw_ss(T_STRIDE_READER_4_1,    0);
		csrw_ss(T_BOUND_READER_4_2,    0);
		csrw_ss(T_STRIDE_READER_4_2,    0);
		csrw_ss(T_BOUND_READER_4_3,    0);
		csrw_ss(T_STRIDE_READER_4_3,    0);
		csrw_ss(T_BOUND_READER_4_4,    0);
		csrw_ss(T_STRIDE_READER_4_4,    0);
		csrw_ss(T_BOUND_READER_4_5,    0);
		csrw_ss(T_STRIDE_READER_4_5,    0);
		csrw_ss(T_BOUND_READER_4_6,    0);
		csrw_ss(T_STRIDE_READER_4_6,    0);
		csrw_ss(T_BOUND_READER_4_7,    0);
		csrw_ss(T_STRIDE_READER_4_7,    0);
		// READER_5
		csrw_ss(BASE_PTR_READER_5_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_5_0, 8);
		csrw_ss(T_BOUND_READER_5_0,    0);
		csrw_ss(T_STRIDE_READER_5_0,    0);
		csrw_ss(T_BOUND_READER_5_1,    0);
		csrw_ss(T_STRIDE_READER_5_1,    0);
		csrw_ss(T_BOUND_READER_5_2,    0);
		csrw_ss(T_STRIDE_READER_5_2,    0);
		csrw_ss(T_BOUND_READER_5_3,    0);
		csrw_ss(T_STRIDE_READER_5_3,    0);
		csrw_ss(T_BOUND_READER_5_4,    0);
		csrw_ss(T_STRIDE_READER_5_4,    0);
		csrw_ss(T_BOUND_READER_5_5,    0);
		csrw_ss(T_STRIDE_READER_5_5,    0);
		csrw_ss(T_BOUND_READER_5_6,    0);
		csrw_ss(T_STRIDE_READER_5_6,    0);
		csrw_ss(T_BOUND_READER_5_7,    0);
		csrw_ss(T_STRIDE_READER_5_7,    0);
		// READER_6
		csrw_ss(BASE_PTR_READER_6_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_6_0, 8);
		csrw_ss(T_BOUND_READER_6_0,    0);
		csrw_ss(T_STRIDE_READER_6_0,    0);
		csrw_ss(T_BOUND_READER_6_1,    0);
		csrw_ss(T_STRIDE_READER_6_1,    0);
		csrw_ss(T_BOUND_READER_6_2,    0);
		csrw_ss(T_STRIDE_READER_6_2,    0);
		csrw_ss(T_BOUND_READER_6_3,    0);
		csrw_ss(T_STRIDE_READER_6_3,    0);
		csrw_ss(T_BOUND_READER_6_4,    0);
		csrw_ss(T_STRIDE_READER_6_4,    0);
		csrw_ss(T_BOUND_READER_6_5,    0);
		csrw_ss(T_STRIDE_READER_6_5,    0);
		csrw_ss(T_BOUND_READER_6_6,    0);
		csrw_ss(T_STRIDE_READER_6_6,    0);
		csrw_ss(T_BOUND_READER_6_7,    0);
		csrw_ss(T_STRIDE_READER_6_7,    0);
		// READER_7
		csrw_ss(BASE_PTR_READER_7_LOW, (uint32_t)(snrt_l1_next() + delta_comp_data +    0));
		csrw_ss(S_STRIDE_READER_7_0, 8);
		csrw_ss(T_BOUND_READER_7_0,    0);
		csrw_ss(T_STRIDE_READER_7_0,    0);
		csrw_ss(T_BOUND_READER_7_1,    0);
		csrw_ss(T_STRIDE_READER_7_1,    0);
		csrw_ss(T_BOUND_READER_7_2,    0);
		csrw_ss(T_STRIDE_READER_7_2,    0);
		csrw_ss(T_BOUND_READER_7_3,    0);
		csrw_ss(T_STRIDE_READER_7_3,    0);
		csrw_ss(T_BOUND_READER_7_4,    0);
		csrw_ss(T_STRIDE_READER_7_4,    0);
		csrw_ss(T_BOUND_READER_7_5,    0);
		csrw_ss(T_STRIDE_READER_7_5,    0);
		csrw_ss(T_BOUND_READER_7_6,    0);
		csrw_ss(T_STRIDE_READER_7_6,    0);
		csrw_ss(T_BOUND_READER_7_7,    0);
		csrw_ss(T_STRIDE_READER_7_7,    0);
		// DS_R handles end
		// DS_W handles begin
		// WRITER_0
		csrw_ss(BASE_PTR_WRITER_0_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_0_0, 8);
		csrw_ss(T_BOUND_WRITER_0_0,    0);
		csrw_ss(T_STRIDE_WRITER_0_0,    0);
		csrw_ss(T_BOUND_WRITER_0_1,    0);
		csrw_ss(T_STRIDE_WRITER_0_1,    0);
		csrw_ss(T_BOUND_WRITER_0_2,    0);
		csrw_ss(T_STRIDE_WRITER_0_2,    0);
		csrw_ss(T_BOUND_WRITER_0_3,    0);
		csrw_ss(T_STRIDE_WRITER_0_3,    0);
		csrw_ss(T_BOUND_WRITER_0_4,    0);
		csrw_ss(T_STRIDE_WRITER_0_4,    0);
		csrw_ss(T_BOUND_WRITER_0_5,    0);
		csrw_ss(T_STRIDE_WRITER_0_5,    0);
		csrw_ss(T_BOUND_WRITER_0_6,    0);
		csrw_ss(T_STRIDE_WRITER_0_6,    0);
		csrw_ss(T_BOUND_WRITER_0_7,    0);
		csrw_ss(T_STRIDE_WRITER_0_7,    0);
		// WRITER_1
		csrw_ss(BASE_PTR_WRITER_1_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_1_0, 8);
		csrw_ss(T_BOUND_WRITER_1_0,    0);
		csrw_ss(T_STRIDE_WRITER_1_0,    0);
		csrw_ss(T_BOUND_WRITER_1_1,    0);
		csrw_ss(T_STRIDE_WRITER_1_1,    0);
		csrw_ss(T_BOUND_WRITER_1_2,    0);
		csrw_ss(T_STRIDE_WRITER_1_2,    0);
		csrw_ss(T_BOUND_WRITER_1_3,    0);
		csrw_ss(T_STRIDE_WRITER_1_3,    0);
		csrw_ss(T_BOUND_WRITER_1_4,    0);
		csrw_ss(T_STRIDE_WRITER_1_4,    0);
		csrw_ss(T_BOUND_WRITER_1_5,    0);
		csrw_ss(T_STRIDE_WRITER_1_5,    0);
		csrw_ss(T_BOUND_WRITER_1_6,    0);
		csrw_ss(T_STRIDE_WRITER_1_6,    0);
		csrw_ss(T_BOUND_WRITER_1_7,    0);
		csrw_ss(T_STRIDE_WRITER_1_7,    0);
		// WRITER_2
		csrw_ss(BASE_PTR_WRITER_2_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_2_0, 8);
		csrw_ss(T_BOUND_WRITER_2_0,    0);
		csrw_ss(T_STRIDE_WRITER_2_0,    0);
		csrw_ss(T_BOUND_WRITER_2_1,    0);
		csrw_ss(T_STRIDE_WRITER_2_1,    0);
		csrw_ss(T_BOUND_WRITER_2_2,    0);
		csrw_ss(T_STRIDE_WRITER_2_2,    0);
		csrw_ss(T_BOUND_WRITER_2_3,    0);
		csrw_ss(T_STRIDE_WRITER_2_3,    0);
		csrw_ss(T_BOUND_WRITER_2_4,    0);
		csrw_ss(T_STRIDE_WRITER_2_4,    0);
		csrw_ss(T_BOUND_WRITER_2_5,    0);
		csrw_ss(T_STRIDE_WRITER_2_5,    0);
		csrw_ss(T_BOUND_WRITER_2_6,    0);
		csrw_ss(T_STRIDE_WRITER_2_6,    0);
		csrw_ss(T_BOUND_WRITER_2_7,    0);
		csrw_ss(T_STRIDE_WRITER_2_7,    0);
		// WRITER_3
		csrw_ss(BASE_PTR_WRITER_3_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_3_0, 8);
		csrw_ss(T_BOUND_WRITER_3_0,    0);
		csrw_ss(T_STRIDE_WRITER_3_0,    0);
		csrw_ss(T_BOUND_WRITER_3_1,    0);
		csrw_ss(T_STRIDE_WRITER_3_1,    0);
		csrw_ss(T_BOUND_WRITER_3_2,    0);
		csrw_ss(T_STRIDE_WRITER_3_2,    0);
		csrw_ss(T_BOUND_WRITER_3_3,    0);
		csrw_ss(T_STRIDE_WRITER_3_3,    0);
		csrw_ss(T_BOUND_WRITER_3_4,    0);
		csrw_ss(T_STRIDE_WRITER_3_4,    0);
		csrw_ss(T_BOUND_WRITER_3_5,    0);
		csrw_ss(T_STRIDE_WRITER_3_5,    0);
		csrw_ss(T_BOUND_WRITER_3_6,    0);
		csrw_ss(T_STRIDE_WRITER_3_6,    0);
		csrw_ss(T_BOUND_WRITER_3_7,    0);
		csrw_ss(T_STRIDE_WRITER_3_7,    0);
		// WRITER_4
		csrw_ss(BASE_PTR_WRITER_4_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_4_0, 8);
		csrw_ss(T_BOUND_WRITER_4_0, 2078);
		csrw_ss(T_STRIDE_WRITER_4_0,    8);
		csrw_ss(T_BOUND_WRITER_4_1,   32);
		csrw_ss(T_STRIDE_WRITER_4_1,    0);
		csrw_ss(T_BOUND_WRITER_4_2,    1);
		csrw_ss(T_STRIDE_WRITER_4_2,    0);
		csrw_ss(T_BOUND_WRITER_4_3,    1);
		csrw_ss(T_STRIDE_WRITER_4_3,    0);
		csrw_ss(T_BOUND_WRITER_4_4,    1);
		csrw_ss(T_STRIDE_WRITER_4_4,    0);
		csrw_ss(T_BOUND_WRITER_4_5,    1);
		csrw_ss(T_STRIDE_WRITER_4_5,    0);
		csrw_ss(T_BOUND_WRITER_4_6,    1);
		csrw_ss(T_STRIDE_WRITER_4_6,    0);
		csrw_ss(T_BOUND_WRITER_4_7,    1);
		csrw_ss(T_STRIDE_WRITER_4_7,    0);
		// WRITER_5
		csrw_ss(BASE_PTR_WRITER_5_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_5_0, 8);
		csrw_ss(T_BOUND_WRITER_5_0,    0);
		csrw_ss(T_STRIDE_WRITER_5_0,    0);
		csrw_ss(T_BOUND_WRITER_5_1,    0);
		csrw_ss(T_STRIDE_WRITER_5_1,    0);
		csrw_ss(T_BOUND_WRITER_5_2,    0);
		csrw_ss(T_STRIDE_WRITER_5_2,    0);
		csrw_ss(T_BOUND_WRITER_5_3,    0);
		csrw_ss(T_STRIDE_WRITER_5_3,    0);
		csrw_ss(T_BOUND_WRITER_5_4,    0);
		csrw_ss(T_STRIDE_WRITER_5_4,    0);
		csrw_ss(T_BOUND_WRITER_5_5,    0);
		csrw_ss(T_STRIDE_WRITER_5_5,    0);
		csrw_ss(T_BOUND_WRITER_5_6,    0);
		csrw_ss(T_STRIDE_WRITER_5_6,    0);
		csrw_ss(T_BOUND_WRITER_5_7,    0);
		csrw_ss(T_STRIDE_WRITER_5_7,    0);
		// WRITER_6
		csrw_ss(BASE_PTR_WRITER_6_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_6_0, 8);
		csrw_ss(T_BOUND_WRITER_6_0,    0);
		csrw_ss(T_STRIDE_WRITER_6_0,    0);
		csrw_ss(T_BOUND_WRITER_6_1,    0);
		csrw_ss(T_STRIDE_WRITER_6_1,    0);
		csrw_ss(T_BOUND_WRITER_6_2,    0);
		csrw_ss(T_STRIDE_WRITER_6_2,    0);
		csrw_ss(T_BOUND_WRITER_6_3,    0);
		csrw_ss(T_STRIDE_WRITER_6_3,    0);
		csrw_ss(T_BOUND_WRITER_6_4,    0);
		csrw_ss(T_STRIDE_WRITER_6_4,    0);
		csrw_ss(T_BOUND_WRITER_6_5,    0);
		csrw_ss(T_STRIDE_WRITER_6_5,    0);
		csrw_ss(T_BOUND_WRITER_6_6,    0);
		csrw_ss(T_STRIDE_WRITER_6_6,    0);
		csrw_ss(T_BOUND_WRITER_6_7,    0);
		csrw_ss(T_STRIDE_WRITER_6_7,    0);
		// WRITER_7
		csrw_ss(BASE_PTR_WRITER_7_LOW, (uint32_t)(snrt_l1_next() + delta_store_data +    0));
		csrw_ss(S_STRIDE_WRITER_7_0, 8);
		csrw_ss(T_BOUND_WRITER_7_0,    0);
		csrw_ss(T_STRIDE_WRITER_7_0,    0);
		csrw_ss(T_BOUND_WRITER_7_1,    0);
		csrw_ss(T_STRIDE_WRITER_7_1,    0);
		csrw_ss(T_BOUND_WRITER_7_2,    0);
		csrw_ss(T_STRIDE_WRITER_7_2,    0);
		csrw_ss(T_BOUND_WRITER_7_3,    0);
		csrw_ss(T_STRIDE_WRITER_7_3,    0);
		csrw_ss(T_BOUND_WRITER_7_4,    0);
		csrw_ss(T_STRIDE_WRITER_7_4,    0);
		csrw_ss(T_BOUND_WRITER_7_5,    0);
		csrw_ss(T_STRIDE_WRITER_7_5,    0);
		csrw_ss(T_BOUND_WRITER_7_6,    0);
		csrw_ss(T_STRIDE_WRITER_7_6,    0);
		csrw_ss(T_BOUND_WRITER_7_7,    0);
		csrw_ss(T_STRIDE_WRITER_7_7,    0);
		// DS_W handles end
		csrw_ss(STREAMER_START_CSR, 1);


		mcycle_timestamps[3] = snrt_mcycle();
}

void launch_cgra_0_go(int32_t delta_config_data, int32_t delta_comp_data, int32_t delta_store_data, uint32_t *mcycle_timestamps) {
		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 1308656673);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		mcycle_timestamps[4] = snrt_mcycle();
}

void launch_cgra_0_relaunch(uint32_t *mcycle_timestamps) {
		mcycle_timestamps[5] = snrt_mcycle();
		csrw_ss(STREAMER_START_CSR, 1);

		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 1610646561);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		// CSR handles begin
		csrw_ss(CGRA_CSR_ADDR_BASE +  0, 1107330081);
		// CSR handles end
		csrw_ss(CGRA_START_CSR, 1);


		mcycle_timestamps[6] = snrt_mcycle();
}
