#include <stdbool.h>
#include "snax-cgra-lib.h"
#include "snax-cgra-params.h"
#include "snrt.h"
#include "stdint.h"
#include "streamer_csr_addr_map.h"

const char* stage_helper[7] = {
	"CTRL_REG",
	"CONFIG",
	"EXE_INI",
	"COMP_S1",
	"COMP_S2",
	"COMP_S3",
	"COMP_S4"
};

void cgra_hw_barrier(int t_stage, int threshold, bool verbose, bool quiet){
	int fsafe_counter, ref_stage;
	fsafe_counter = 0;
	ref_stage = (t_stage << 16) + t_stage;
	while(csrr_ss(STREAMER_BUSY_CSR) != 0 || csrr_ss(CGRA_CSR_RO_ADDR_BASE+8) < ref_stage){
		if (verbose && !quiet) {
			printf ("system busy [cgra_cc=%d, stage=%d: ss3_%d, ss2_%d, ss1_%d, ss0_%d] ...\r\n", csrr_ss(CGRA_CSR_RO_ADDR_BASE), csrr_ss(CGRA_CSR_RO_ADDR_BASE+8) >> 16, csrr_ss(CGRA_CSR_RO_ADDR_BASE+10) >> 16, csrr_ss(CGRA_CSR_RO_ADDR_BASE+10) << 16 >> 16, csrr_ss(CGRA_CSR_RO_ADDR_BASE+9) >> 16, csrr_ss(CGRA_CSR_RO_ADDR_BASE+9) << 16 >> 16); 
		}
		else {
			if (!quiet) printf ("system busy ...\r\n"); 
		}	
		if (csrr_ss(CGRA_CSR_RO_ADDR_BASE+8) >= ref_stage && fsafe_counter >= 22) {
			printf ("--> Break because CGRA is done [%d]. Please check the Streamer IO amount settings...\r\n", csrr_ss(CGRA_CSR_RO_ADDR_BASE+8)); 
			break;
		}
		else if (fsafe_counter >= threshold) {
			printf ("--> Break because fsafe_counter exceeds threshold [%d]...\r\n", fsafe_counter); 
			break;
		}
		fsafe_counter++;
	}
	printf ("ckpt hit!\r\n\r\n"); 
}

void cgra_hw_barrier_fast(int t_stage, int threshold){
	int fsafe_counter, ref_stage;
	fsafe_counter = 0;
	ref_stage = (t_stage << 16) + t_stage;
	while(csrr_ss(CGRA_CSR_RO_ADDR_BASE+8) < ref_stage){
		if (fsafe_counter >= threshold) {
			printf ("--> fsafe\r\n", fsafe_counter); 
			break;
		}
		fsafe_counter++;
	}
	printf ("ckpt[%0d]!\r\n\r\n", (csrr_ss(STREAMER_BUSY_CSR) != 0)); 
}

void cgra_hw_profiler(){
	int perf_counter[2];
	int perf_counter_tmp;
	printf("ckpt @ CGRA stage [%2d]\r\n", csrr_ss(CGRA_CSR_RO_ADDR_BASE+8) >> 16);
	printf("\tCGRA cc timestamp = %d\r\n", csrr_ss(CGRA_CSR_RO_ADDR_BASE));
	for(int i = 1; i < 7; i++) {
		perf_counter[0] = csrr_ss(CGRA_CSR_RO_ADDR_BASE+i);
		perf_counter[1] = csrr_ss(CGRA_CSR_RO_ADDR_BASE+i+1);
		
		perf_counter_tmp = perf_counter[1] < perf_counter[0] ? 0 : perf_counter[1] - perf_counter[0];
		printf("\tElapsed cc @ %s = %d (%d -> %d)\r\n", stage_helper[i], perf_counter_tmp, perf_counter[0], perf_counter[1]);
	}
	printf("\tCGRA cc timestamp = %d\r\n\r\n", csrr_ss(CGRA_CSR_RO_ADDR_BASE));
}
