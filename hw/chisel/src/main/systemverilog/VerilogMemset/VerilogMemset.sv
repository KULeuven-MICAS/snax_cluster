// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Yunhao Deng <yunhao.deng@kuleuven.be>

module VerilogMemset #(
    parameter int UserCsrNum = 1,
    parameter int DataWidth = 512
) (
    input  logic clk_i,
    input  logic rst_ni,
    output logic ext_data_i_ready,
    input  logic ext_data_i_valid,
    input  logic [DataWidth-1:0] ext_data_i_bits,
    input  logic ext_data_o_ready,
    output logic ext_data_o_valid,
    output logic [DataWidth-1:0] ext_data_o_bits,
    input  logic [31:0]ext_csr_i_0,
    input  logic ext_start_i,
    output logic ext_busy_o
);

    assign ext_data_o_valid = ext_data_i_valid;
    assign ext_data_i_ready = ext_data_o_ready;

    // The CSR is a 32-bit PATTERN tiled across the beat, not a byte. One
    // mechanism covers every constant the datapath can carry -- INT8, FP16,
    // BF16, FP32, INT32 -- without the extension knowing any of those formats.
    // A byte fill is the pattern with all four lanes equal, so callers that
    // write 0xFFFFFFFF or 0x00000000 are unaffected.
    logic [31:0] memset_data;
    assign memset_data = ext_csr_i_0;

    genvar i;
    generate
        for(i = 0; i < DataWidth/32; i = i + 1) begin: g_memset
            assign ext_data_o_bits[i*32 +: 32] = memset_data;
        end
    endgenerate
    assign ext_busy_o = 0;
endmodule
