module intN_to_fp16 #(
    parameter INT_WIDTH = 4  // Set to 1, 2, 3, or 4
)(
    input  wire signed [INT_WIDTH-1:0] intN_in,
    output reg  [15:0]                 fp16_out
);

    // Internal signals
    reg        sign;
    reg [INT_WIDTH-1:0] abs_val;
    reg [4:0]  exponent;
    reg [9:0]  mantissa;

    reg [3:0]  val_ext;  // Only 4 bits now
    integer    shift;

    always @(*) begin
        // Handle zero
        if (intN_in == 0) begin
            fp16_out = 16'b0;
        end else begin
            // Sign and absolute value
            
            if (INT_WIDTH!=1) begin
               abs_val = (sign==1) ? (~(intN_in[0]-1)) : intN_in;
               sign    = intN_in[INT_WIDTH-1];
            end else begin
               abs_val = intN_in;
               sign    = 0;
            end 
            // Set val_ext to abs_val (4 bits or fewer)
            val_ext = abs_val;

            // Normalize: find position of MSB
            shift = 0;
            casez (val_ext)
                4'b1???: shift = 3;
                4'b01??: shift = 2;
                4'b001?: shift = 1;
                4'b0001: shift = 0;
                default: shift = 0;
            endcase

            // Compute exponent: exponent bias is 15
            exponent = 15 + shift;

            // Compute mantissa: shift remaining bits after leading 1 to MSB
            // Drop the leading 1, align the remaining bits to 10-bit mantissa
            mantissa = (val_ext << (10 - shift)) & 10'b1111111111;
            
            // Final fp16 assembly
            fp16_out = {sign, exponent, mantissa};
        end
    end

endmodule
