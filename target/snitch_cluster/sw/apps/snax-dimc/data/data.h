// this file contains the golden model for MHA and CONV mode

// for testing simple TCDM & DMA functionality
uint32_t DATA_LEN = 20;
uint32_t A[] = {99, 67, 39, 26, 62, 14, 17, 18, 54, 16,
                44, 9,  26, 85, 72, 66, 95, 65, 43, 84};
uint32_t B[] = {86, 10, 14, 11, 38, 41, 94, 82, 97, 25,
                96, 71, 44, 59, 93, 38, 57, 21, 84, 29};
uint32_t OUT[] = {8514, 670, 546,  286,  2356, 574,  1598, 1476, 5238, 400,
                  4224, 639, 1144, 5015, 6696, 2508, 5415, 1365, 3612, 2436};

// for testing streamer functionality
// matrix Q has a dimension of 64*512, each element is 8 bits
uint64_t Q_LENGTH = 64*512*8/64;

uint64_t matrix_Q[4096] = {1};
uint64_t matrix_WQ[4096] = {2};

uint64_t matrix_K[4096] = {3};
uint64_t matrix_WK[4096] = {4};

uint64_t matrix_WV[4096] = {3};
uint64_t matrix_V[4096] = {4};

// 64x64x8/64 = 512
uint64_t matrix_Q1K1T[512] = {5};

uint64_t test_array[5] = {0, 1, 2, 3, 4};


void delay_cycles(uint32_t cycle) {
    uint32_t target_cycle, current_cycle;
 
 
    __asm__ volatile(
                    "csrr %0, mcycle;"
                    : "=r"(target_cycle)
                    );
    target_cycle = target_cycle + cycle;
    while (current_cycle < target_cycle) {
    __asm__ volatile(
                    "csrr %0, mcycle;"
                    : "=r"(current_cycle)
                    );
    }
}
