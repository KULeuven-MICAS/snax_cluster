// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Fanchen Kong <fanchen.kong@kuleuven.be>

// The actual data memory. This memory is made into a module
// to support multiple power domain needed by the floor plan tool

(* no_ungroup *)
(* no_boundary_optimization *)
module spm_1p_adv_mem_wrapper #(
    parameter int unsigned NumWords   = 32'd1024, // Number of Words in data array
    parameter int unsigned DataWidth  = 32'd128,  // Data signal width
    parameter int unsigned ByteWidth  = 32'd8,    // Width of a data byte
    parameter              SimInit      = "none",   // Simulation 
    parameter bit          PrintSimCfg  = 1'b0,     // Print configuration
    parameter type         impl_in_t  = logic,    // Type for implementation inputs
    parameter type         impl_out_t = logic,
    // DEPENDENT PARAMETERS, DO NOT OVERWRITE!
    parameter int unsigned AddrWidth  = (NumWords > 32'd1) ? $clog2(NumWords) : 32'd1,
    parameter int unsigned BeWidth    = (DataWidth + ByteWidth - 32'd1) / ByteWidth, // ceil_div
    parameter type         addr_t     = logic [AddrWidth-1:0],
    parameter type         data_t     = logic [DataWidth-1:0],
    parameter type         be_t       = logic [BeWidth-1:0]

) (
    input  logic                 clk_i,
    input  logic                 rst_ni,
    // implementation-related IO
    input  impl_in_t             impl_i,
    output impl_out_t            impl_o,
    // input ports
    input  logic                 req_i,      // request
    input  logic                 we_i,       // write enable
    input  addr_t                addr_i,     // request address
    input  data_t                wdata_i,    // write data
    input  be_t                  be_i,       // write byte enable
    // output ports
    output data_t                rdata_o     // read data
);

    tc_sram_impl #(
        .NumWords (NumWords),
        .DataWidth(DataWidth),
        .ByteWidth(ByteWidth),
        .NumPorts (1),
        .SimInit (SimInit),
        .PrintSimCfg (PrintSimCfg),
        .Latency  (1),
        .impl_in_t (impl_in_t),
        .impl_out_t (impl_out_t)
    ) i_mem (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .impl_i (impl_i),
        .impl_o (impl_o),
        .req_i(req_i),
        .we_i(we_i),
        .addr_i(addr_i),
        .wdata_i(wdata_i),
        .be_i(be_i),
        .rdata_o(rdata_o)
    );

endmodule
