// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Floating-Point Adder
// Based on fpnew_fma modified for fp addition
// Author: Man Shi <man.shi@kuleuven.be>

module fp_mul_int #(
  parameter fpnew_pkg::fp_format_e   FpFormat_a         = fpnew_pkg::fp_format_e'(2), //FP16 
  parameter fpnew_pkg::int_format_e  IntFormat_b         = fpnew_pkg::int_format_e'(4), //int4
  parameter fpnew_pkg::fp_format_e   FpFormat_b         = fpnew_pkg::fp_format_e'(2), //FP16 
  parameter fpnew_pkg::fp_format_e   FpFormat_out     = fpnew_pkg::fp_format_e'(0), //FP32
  parameter fpnew_pkg::roundmode_e   RndMode          = fpnew_pkg::roundmode_e'(0), 

parameter int unsigned WIDTH_a = fpnew_pkg::fp_width(FpFormat_a),    // do not change
parameter int unsigned WIDTH_b = fpnew_pkg::fp_width(FpFormat_b),    // do not change
parameter int unsigned WIDTH_B = fpnew_pkg::fp_width(IntFormat_b),    // do not change
parameter int unsigned WIDTH_out = fpnew_pkg::fp_width(FpFormat_out),    // do not change
parameter int unsigned rnd_mode    = RndMode  // do not change, we fix the round mode in rtl
) (
  // Input Handshake
  //input  logic                         in_valid_i,
  // Input signals
  input logic [WIDTH_a-1:0]         operand_a_i, // 2 operands
  input logic [WIDTH_B-1:0]         operand_b_i, // 2 operands
  // Output signals
  output logic [WIDTH_out-1:0]         result_o
  // Output Handshake
  //output logic                         out_valid_o
);

  // ----------
  // Constants
  // ----------
  // for operand A
    logic [WIDTH_b-1:0] operand_b_i_fp;

    intN_to_fp16  #(
     . INT_WIDTH (WIDTH_B)
      )
    Int2fp 
    (
    .intN_in (operand_b_i),
    .fp16_out (operand_b_i_fp)
    );

localparam int unsigned EXP_BITS_A = fpnew_pkg::exp_bits(FpFormat_a);
    localparam int unsigned MAN_BITS_A = fpnew_pkg::man_bits(FpFormat_a);
    localparam int unsigned BIAS_A     = fpnew_pkg::bias(FpFormat_a);
    // for operand B
    localparam int unsigned EXP_BITS_B = fpnew_pkg::exp_bits(FpFormat_b);
    localparam int unsigned MAN_BITS_B = fpnew_pkg::man_bits(FpFormat_b);
    localparam int unsigned BIAS_B     = fpnew_pkg::bias(FpFormat_b);
    // for operand C and result
    localparam int unsigned EXP_BITS_C = fpnew_pkg::exp_bits(FpFormat_out);
    localparam int unsigned MAN_BITS_C = fpnew_pkg::man_bits(FpFormat_out);
    localparam int unsigned BIAS_C     = fpnew_pkg::bias(FpFormat_out);

    localparam int unsigned PRECISION_BITS_A = MAN_BITS_A + 1;
    localparam int unsigned PRECISION_BITS_B = MAN_BITS_B + 1;
    localparam int unsigned PRECISION_BITS_C = MAN_BITS_C + 1;

    // localparam int unsigned LOWER_SUM_WIDTH  = (PRECISION_BITS_A + PRECISION_BITS_B) + 3;
    localparam int unsigned LOWER_SUM_WIDTH = (PRECISION_BITS_A + PRECISION_BITS_B > PRECISION_BITS_C) 
                                                ? (PRECISION_BITS_A + PRECISION_BITS_B) 
                                                : (PRECISION_BITS_C );
    localparam int unsigned LZC_RESULT_WIDTH = $clog2(LOWER_SUM_WIDTH);
    localparam int unsigned EXP_WIDTH = unsigned'(fpnew_pkg::maximum(EXP_BITS_C + 2, LZC_RESULT_WIDTH));
    // (PRECISION_BITS_C + 2) + (PRECISION_BITS_A + PRECISION_BITS_B) + 2 - 1
    // localparam int unsigned SHIFT_AMOUNT_WIDTH = $clog2(PRECISION_BITS_C + PRECISION_BITS_B + PRECISION_BITS_A + 3);
    localparam int unsigned SHIFT_AMOUNT_WIDTH = $clog2(LOWER_SUM_WIDTH + PRECISION_BITS_C);

  
  // ----------------
  // Type definition
  // ----------------
   typedef struct packed {
        logic                   sign;
        logic [EXP_BITS_A-1:0]  exponent;
        logic [MAN_BITS_A-1:0]  mantissa;
    } fp_a_t;

    typedef struct packed {
        logic                   sign;
        logic [EXP_BITS_B-1:0]  exponent;
        logic [MAN_BITS_B-1:0]  mantissa;
    } fp_b_t;

    typedef struct packed {
        logic                   sign;
        logic [EXP_BITS_C-1:0]  exponent;
        logic [MAN_BITS_C-1:0]  mantissa;
    } fp_t_out;

  
  // -----------------
  // Input processing
  // -----------------
  fpnew_pkg::fp_info_t [1:0] info_q;
  fp_a_t                operand_a;
  fp_b_t                operand_b;
  fpnew_pkg::fp_info_t info_a,    info_b;


  // Classify input
  fpnew_classifier #(
    .FpFormat    ( FpFormat_a ),
    .NumOperands ( 1       )
    ) i_class_inputs_a (
    .operands_i ( operand_a_i ),
    .is_boxed_i ( 1'b1 ),
    .info_o     ( info_q[0]  )
  );

  fpnew_classifier #(
    .FpFormat    ( FpFormat_b ),
    .NumOperands ( 1        )
    ) i_class_inputs_b (
    .operands_i ( operand_b_i_fp ),
    .is_boxed_i ( 1'b1 ),
    .info_o     ( info_q[1] )
  );

  // Default assignments - packing-order-agnostic
  assign operand_a = operand_a_i;
  assign operand_b = operand_b_i_fp;
  assign info_a    = info_q[0];
  assign info_b    = info_q[1];


    // Input classification
    // ---------------------
    logic any_operand_inf;
    logic any_operand_nan;
    logic signalling_nan;
    logic effective_subtraction;
    logic tentative_sign;

    // Reduction for special case handling
    assign any_operand_inf = (| {info_a.is_inf,        info_b.is_inf});
    assign any_operand_nan = (| {info_a.is_nan,        info_b.is_nan});
    assign signalling_nan  = (| {info_a.is_signalling, info_b.is_signalling});
    // Effective subtraction in FMA occurs when product and addend signs differ
    logic result_sign;

    assign result_sign = operand_a.sign ^ operand_b.sign;
    // ----------------------
    // Special case handling
    // ----------------------
    fp_t_out            special_result;
    logic               result_is_special;

    always_comb begin : special_cases
        // Default assignments
        special_result    = '{sign: 1'b0, exponent: '1, mantissa: 2**(MAN_BITS_C-1)}; // canonical qNaN
        result_is_special = 1'b0;

        if ((info_a.is_inf && info_b.is_zero) || (info_a.is_zero && info_b.is_inf)) begin
            result_is_special = 1'b1;
            special_result    = '{sign: operand_a.sign ^ operand_b.sign, exponent: '1, mantissa: 2**(MAN_BITS_C-1)};
        end 
        else if (any_operand_nan) begin
            result_is_special = 1'b1;
        end 
        else if (any_operand_inf) begin
            result_is_special = 1'b1;
            if ((info_a.is_inf || info_b.is_inf) && effective_subtraction)
                special_result    = '{sign: 1'b0, exponent: '1, mantissa: 2**(MAN_BITS_C-1)};
            else if (info_a.is_inf || info_b.is_inf) begin
                special_result = '{sign: operand_a.sign ^ operand_b.sign, exponent: '1, mantissa: '0};
            end 
        end
    end


  // ---------------------------
  // Initial exponent data path
  // ---------------------------
    logic signed [EXP_BITS_A-1:0] exponent_a;
    logic signed [EXP_BITS_B-1:0] exponent_b;
    logic signed [EXP_WIDTH-1:0] exponent_product;
  

    assign exponent_a = signed'({1'b0, operand_a.exponent});
    assign exponent_b = signed'({1'b0, operand_b.exponent});

    assign exponent_product = (info_a.is_zero || info_b.is_zero)
                                ? 2 - signed'(BIAS_C)   // [NOTE]
                                : signed'(exponent_a + info_a.is_subnormal - signed'(BIAS_A)
                                        + exponent_b + info_b.is_subnormal - signed'(BIAS_B)
                                        + signed'(BIAS_C));



  // ------------------
  // Product data path
  // ------------------
    localparam int unsigned MUL_WIDTH = PRECISION_BITS_A + PRECISION_BITS_B;
    localparam int unsigned PRODUCT_WIDTH = (MUL_WIDTH > PRECISION_BITS_C) ? MUL_WIDTH : PRECISION_BITS_C;
    localparam int unsigned SUM_WIDTH = PRODUCT_WIDTH + PRECISION_BITS_C + 4;
    logic [PRECISION_BITS_A-1:0] mantissa_a;
    logic [PRECISION_BITS_B-1:0] mantissa_b;
    logic [(MUL_WIDTH-1):0] product;
    logic [(PRECISION_BITS_C):0] product_shifted;

    // Add implicit bits to mantissae
    assign mantissa_a = {info_a.is_normal, operand_a.mantissa};
    assign mantissa_b = {info_b.is_normal, operand_b.mantissa};

    // Mantissa multiplier (a*b)
    assign product = mantissa_a * mantissa_b;
    // assign product_shifted = product << 2;
    assign product_shifted = product << (2); 
    
 

  // --------------
  // Normalization
  // --------------
  logic        [LOWER_SUM_WIDTH-1:0]  sum_lower;              // lower 2p+3 bits of sum are searched
  logic        [LZC_RESULT_WIDTH-1:0] leading_zero_count;     // the number of leading zeroes
  logic signed [LZC_RESULT_WIDTH:0]   leading_zero_count_sgn; // signed leading-zero count
  logic                               lzc_zeroes;             // in case only zeroes found

  logic        [SHIFT_AMOUNT_WIDTH-1:0] norm_shamt; // Normalization shift amount
  logic signed [EXP_WIDTH-1:0]      normalized_exponent;
  


  logic [PRECISION_BITS_C:0]         sum_shifted;
  logic [PRECISION_BITS_C:0]         final_mantissa;
  logic sum_sticky_bits;
  logic                       sticky_after_norm;

  logic signed [EXP_WIDTH-1:0] final_exponent;

  assign sum_lower = product_shifted[LOWER_SUM_WIDTH-1:0];

  // Leading zero counter for cancellations
  lzc #(
    .WIDTH ( LOWER_SUM_WIDTH ),
    .MODE  ( 1               ) // MODE = 1 counts leading zeroes
  ) i_lzc (
    .in_i    ( sum_lower          ),
    .cnt_o   ( leading_zero_count ),
    .empty_o ( lzc_zeroes         )
  );

  assign leading_zero_count_sgn = signed'({1'b0, leading_zero_count});
  
  always_comb begin : norm_shift_amount
  
            if ((exponent_product - leading_zero_count_sgn + 1 >= 0) && !lzc_zeroes) begin
                norm_shamt          = PRECISION_BITS_C + 2 + leading_zero_count;
                normalized_exponent = exponent_product - leading_zero_count_sgn + 1; // account for shift
            // Subnormal result
            end else begin
                norm_shamt          = unsigned'(signed'(PRECISION_BITS_C) + 2 + exponent_product);
                normalized_exponent = 0; // subnormals encoded as 0
            end
        // Addend-anchored case

  end

  assign sum_shifted  = product_shifted << (leading_zero_count + 1);

  always_comb begin : small_norm
    // Default assignment, discarding carry bit
    {final_mantissa} = sum_shifted;
    final_exponent                    = normalized_exponent;
  end

  // Update the sticky bit with the shifted-out bits
  assign sticky_after_norm = (| sum_sticky_bits);
  //assign final_exponent_out = final_exponent + (BIAS_OUT - BIAS_a - BIAS_b);
  
  // ----------------------------
  // Rounding and classification
  // ----------------------------
  logic                             pre_round_sign;
  logic [EXP_BITS_C-1:0]          pre_round_exponent;
  logic [MAN_BITS_C-1:0]          pre_round_mantissa;
  logic [EXP_BITS_C+MAN_BITS_C-1:0] pre_round_abs; // absolute value of result before rounding
  logic [1:0]                   round_sticky_bits;

  logic of_before_round, of_after_round; // overflow
  logic uf_before_round, uf_after_round; // underflow
  logic result_zero;

  logic                         rounded_sign;
  logic [EXP_BITS_C+MAN_BITS_C-1:0] rounded_abs; // absolute value of result after rounding

 
  assign of_before_round = final_exponent >= 2**(EXP_BITS_C)-1; // infinity exponent is all ones
  assign uf_before_round = final_exponent == 0;               // exponent for subnormals capped to 0
  
  // Assemble result before rounding. In case of overflow, the largest normal value is set.
  assign pre_round_sign     = result_sign;
  assign pre_round_exponent = (of_before_round) ? 2**EXP_BITS_C-2 : unsigned'(final_exponent[EXP_BITS_C-1:0]);
  assign pre_round_mantissa = (of_before_round) ? '1 : final_mantissa[MAN_BITS_C:1]; // bit 0 is R bit
  assign pre_round_abs      = {pre_round_exponent, pre_round_mantissa};

  // In case of overflow, the round and sticky bits are set for proper rounding
  assign round_sticky_bits  = (of_before_round) ? 2'b11 : {final_mantissa[0], sticky_after_norm};

  // Perform rounding
fpnew_rounding #(
  .AbsWidth ( EXP_BITS_C + MAN_BITS_C )
) i_fpnew_rounding (
  .abs_value_i             ( pre_round_abs ),
  .sign_i                  ( pre_round_sign ),
  .round_sticky_bits_i     ( round_sticky_bits ),
  .rnd_mode_i              ( fpnew_pkg::RNE ),
  .effective_subtraction_i ( 1'b0 ), // pure mul, no subtraction
  .abs_rounded_o           ( rounded_abs ),
  .sign_o                  ( rounded_sign ),
  .exact_zero_o            ( result_zero )
);

    // -----------------
    // Result selection
    // -----------------
    logic [WIDTH_out-1:0]     regular_result;

    assign regular_result    = {rounded_sign, rounded_abs};
    assign result_o = result_is_special ? special_result : regular_result;


endmodule
