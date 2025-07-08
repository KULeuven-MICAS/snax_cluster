// Copyright 2019 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Stefan Mach <smach@iis.ee.ethz.ch>

module fpnew_classifier #(
    parameter fpnew_pkg_versacore::fp_format_e FpFormat    = fpnew_pkg_versacore::fp_format_e'(0),
    parameter int unsigned           NumOperands = 1,
    // Do not change
    //localparam int unsigned WIDTH = fpnew_pkg_versacore::fp_width(FpFormat)
    parameter int unsigned           WIDTH       = fpnew_pkg_versacore::fp_width(FpFormat)
) (
    input logic [NumOperands-1:0][WIDTH-1:0] operands_i,
    input logic [NumOperands-1:0] is_boxed_i,  //used when more than one inputs
    output fpnew_pkg_versacore::fp_info_t [NumOperands-1:0] info_o
);

  localparam int unsigned EXP_BITS = fpnew_pkg_versacore::exp_bits(FpFormat);
  localparam int unsigned MAN_BITS = fpnew_pkg_versacore::man_bits(FpFormat);

  // Type definition
  typedef struct packed {
    logic                sign;
    logic [EXP_BITS-1:0] exponent;
    logic [MAN_BITS-1:0] mantissa;
  } fp_t;

  // Iterate through all operands
  for (genvar op = 0; op < int'(NumOperands); op++) begin : gen_num_values

    fp_t  value;
    logic is_boxed;
    logic is_normal;
    logic is_inf;
    logic is_nan;
    logic is_signalling;
    logic is_quiet;
    logic is_zero;
    logic is_subnormal;

    // ---------------
    // Classify Input
    // ---------------
    always_comb begin : classify_input
      value                    = operands_i[op];
      is_boxed                 = is_boxed_i[op];
      is_normal                = is_boxed && (value.exponent != '0) && (value.exponent != '1);
      is_zero                  = is_boxed && (value.exponent == '0) && (value.mantissa == '0);
      is_subnormal             = is_boxed && (value.exponent == '0) && !is_zero;
      is_inf                   = is_boxed && ((value.exponent == '1) && (value.mantissa == '0));
      is_nan                   = !is_boxed || ((value.exponent == '1) && (value.mantissa != '0));
      is_signalling            = is_boxed && is_nan && (value.mantissa[MAN_BITS-1] == 1'b0);
      is_quiet                 = is_nan && !is_signalling;
      // Assign output for current input
      info_o[op].is_normal     = is_normal;
      info_o[op].is_subnormal  = is_subnormal;
      info_o[op].is_zero       = is_zero;
      info_o[op].is_inf        = is_inf;
      info_o[op].is_nan        = is_nan;
      info_o[op].is_signalling = is_signalling;
      info_o[op].is_quiet      = is_quiet;
      info_o[op].is_boxed      = is_boxed;
    end
  end
endmodule
