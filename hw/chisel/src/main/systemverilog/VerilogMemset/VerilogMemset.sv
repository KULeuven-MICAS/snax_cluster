module VerilogMemset #(
    userCsrNum = 1,
    dataWidth = 512
) (
    input logic clk, 
    input logic rst_n, 
    output logic ext_data_i_ready, 
    input logic ext_data_i_valid,
    input logic [dataWidth-1:0] ext_data_i_bits,
    input logic ext_data_o_ready, 
    output logic ext_data_o_valid,
    output logic [dataWidth-1:0] ext_data_o_bits,
    input logic [31:0]ext_csr_i_0, 
    input logic ext_start_i, 
    output logic ext_busy_o
);

    assign ext_data_o_valid = ext_data_i_valid;
    assign ext_data_i_ready = ext_data_o_ready;
    logic [7:0] memset_data;
    assign memset_data = ext_csr_i_0[7:0];

    genvar i;
    generate
        for(i = 0; i < dataWidth/8; i = i + 1) begin
            assign ext_data_o_bits[i*8 +: 8] = memset_data;
        end
    endgenerate

    assign ext_busy_o = 0;
    
endmodule