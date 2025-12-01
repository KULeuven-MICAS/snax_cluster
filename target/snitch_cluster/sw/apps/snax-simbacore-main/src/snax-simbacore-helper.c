// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "data.h"
#include "snax-simbacore-lib.h"

void set_streamer_phase1(uint32_t ptr_oscore_in, uint32_t ptr_oscore_weight, uint32_t ptr_conv_weight,
                         uint32_t ptr_conv_bias, uint32_t ptr_iscore_weight, uint32_t ptr_iscore_out,
                         uint32_t ptr_conv_out) {
    printf("Setting up Streamer and SimbaCore for Phase1...\n");

    set_streamer_csr(

        (uint32_t)ptr_oscore_in, M1_R0_ss, M1_R0_tb, M1_R0_ts, M1_R0_en,          //
        (uint32_t)ptr_oscore_weight, M1_R1_ss, M1_R1_tb, M1_R1_ts, M1_R1_en,      //
        (uint32_t)0, 0, 0, 0, M1_R2_en,                                           // disable
        (uint32_t)ptr_conv_weight, M1_R3_ss, M1_R3_tb, M1_R3_ts, M1_R3_en,        //
        (uint32_t)ptr_conv_bias, M1_R4_ss, M1_R4_tb, M1_R4_ts, M1_R4_en,          //
        (uint32_t)0, 0, 0, 0, M1_R5_en,                                           // disable
        (uint32_t)0, 0, 0, 0, M1_R6_en,                                           // disable
        (uint32_t)0, 0, 0, 0, M1_R7_en,                                           // disable
        (uint32_t)0, 0, 0, 0, M1_R8_en,                                           // disable
        (uint32_t)0, 0, 0, 0, M1_R9_en,                                           // disable
        (uint32_t)0, 0, 0, 0, M1_R10_en,                                          // disable
        (uint32_t)0, 0, 0, 0, M1_R11_en,                                          // disable
        (uint32_t)ptr_iscore_weight, M1_R12_ss, M1_R12_tb, M1_R12_ts, M1_R12_en,  //
        (uint32_t)ptr_iscore_out, M1_R13_ss, M1_R13_tb, M1_R13_ts, M1_R13_en,     // psums
        (uint32_t)0, 0, 0, 0, M1_W0_en,                                           // disable
        (uint32_t)ptr_conv_out, M1_W1_ss, M1_W1_tb, M1_W1_ts, M1_W1_en,           //
        (uint32_t)0, 0, 0, 0, M1_W2_en,                                           // disable
        (uint32_t)ptr_iscore_out, M1_W3_ss, M1_W3_tb, M1_W3_ts, M1_W3_en          //
    );
}

void set_streamer_phase2(uint32_t ptr_oscore_in, uint32_t ptr_oscore_weight, uint32_t ptr_z, uint32_t ptr_dt_in,
                         uint32_t ptr_dt_weight_1, uint32_t ptr_dt_weight_2, uint32_t ptr_dt_bias, uint32_t ptr_x,
                         uint32_t ptr_A, uint32_t ptr_BC, uint32_t ptr_D, uint32_t ptr_y, uint32_t ptr_iscore_weight,
                         uint32_t ptr_iscore_out) {
    printf("Setting up Streamer and SimbaCore for Phase2...\n");

    set_streamer_csr(

        (uint32_t)ptr_oscore_in, M2_R0_ss, M2_R0_tb, M2_R0_ts, M2_R0_en,          // osCore in
        (uint32_t)ptr_oscore_weight, M2_R1_ss, M2_R1_tb, M2_R1_ts, M2_R1_en,      // oscore weight
        (uint32_t)ptr_dt_in, M2_R2_ss, M2_R2_tb, M2_R2_ts, M2_R2_en,              // switchCore in
        (uint32_t)ptr_dt_weight_1, M2_R3_ss, M2_R3_tb, M2_R3_ts, M2_R3_en,        // switchCore weight
        (uint32_t)ptr_dt_bias, M2_R4_ss, M2_R4_tb, M2_R4_ts, M2_R4_en,            // switchCore bias
        (uint32_t)ptr_dt_weight_2, M2_R5_ss, M2_R5_tb, M2_R5_ts, M2_R5_en,        // switchCore  matmul weight
        (uint32_t)ptr_A, M2_R6_ss, M2_R6_tb, M2_R6_ts, M2_R6_en,                  //  SUC A
        (uint32_t)ptr_BC, M2_R7_ss, M2_R7_tb, M2_R7_ts, M2_R7_en,                 // SUC BC
        (uint32_t)ptr_D, M2_R8_ss, M2_R8_tb, M2_R8_ts, M2_R8_en,                  // SUC  D
        (uint32_t)ptr_x, M2_R9_ss, M2_R9_tb, M2_R9_ts, M2_R9_en,                  // SUC x
        (uint32_t)ptr_z, M2_R10_ss, M2_R10_tb, M2_R10_ts, M2_R10_en,              // SUC z = osCore out
        (uint32_t)ptr_y, M2_R11_ss, M2_R11_tb, M2_R11_ts, M2_R11_en,              // iscore in = SUC y
        (uint32_t)ptr_iscore_weight, M2_R12_ss, M2_R12_tb, M2_R12_ts, M2_R12_en,  // isCore weight
        (uint32_t)ptr_iscore_out, M2_R13_ss, M2_R13_tb, M2_R13_ts, M2_R13_en,     // isCore psum

        (uint32_t)ptr_z, M2_W0_ss, M2_W0_tb, M2_W0_ts, M2_W0_en,          // osCore out = z
        (uint32_t)0, 0, 0, 0, M2_W1_en,                                   // disable
        (uint32_t)ptr_y, M2_W2_ss, M2_W2_tb, M2_W2_ts, M2_W2_en,          // SUC y
        (uint32_t)ptr_iscore_out, M2_W3_ss, M2_W3_tb, M2_W3_ts, M2_W3_en  // isCore out

    );
}
