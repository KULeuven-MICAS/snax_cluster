// #include "conv_res_qkt.h"
// #include "conv_res_qkv.h"
#include "kernel_stimulus.h"
#include "mha_golden_result.h"
#include "conv_golden_result.h"
#include "Q1K1T_G.h"

uint64_t Q1K1T_LENGTH = 512;

uint64_t Q_LENGTH = 4096;

// here Q, K V and WQ, WK, WV should have the form of 1-D array
// with a length of 4096, and elements of uint64_t
#include "Q_content.h"
// #include "K_content.h"
// #include "V_content.h"
// #include "WQ_content.h"
// #include "WK_content.h"
// #include "WV_content.h"


// DMA test

uint64_t DATA_LEN = 8;
uint64_t A[8] = {1, 2, 3, 4, 5, 6, 7, 8};
uint64_t B[8] = {8, 7, 6, 5, 4, 3, 2, 1};

