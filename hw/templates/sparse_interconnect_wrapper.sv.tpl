// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

//-----------------------------
// Streamer wrapper
//-----------------------------
module sparse_interconnect_wrapper #(
  // Payload type of the data request ports.
  parameter type         tcdm_req_t            = logic,
  // Payload type of the data response ports.
  parameter type         tcdm_rsp_t            = logic,
  // Payload type of the data request ports.
  parameter type         mem_req_t             = logic,
  // Payload type of the data response ports.
  parameter type         mem_rsp_t             = logic,
  parameter int unsigned NumInp = ${cfg["sparse_interconnect_cfg"]["NumInp"]},
  parameter int unsigned NumOut = ${cfg["sparse_interconnect_cfg"]["NumOut"]}
)(
  //-----------------------------
  // Clocks and reset
  //-----------------------------
  input  logic clk_i,
  input  logic rst_ni,
  //-----------------------------
  // TCDM ports
  //-----------------------------
  // Request port.
  input  tcdm_req_t           [NumInp-1:0] req_i,
  // Resposne port.
  output tcdm_rsp_t           [NumInp-1:0] rsp_o,
  // Memory Side
  // Request.
  output mem_req_t            [NumOut-1:0] mem_req_o,
  // Response.
  input  mem_rsp_t            [NumOut-1:0] mem_rsp_i
);

import reqrsp_pkg::*;

logic [${cfg["sparse_interconnect_cfg"]["NumOut"]}-1:0][3:0] mem_req_o_amo;

% for idx in range(cfg["sparse_interconnect_cfg"]["NumOut"]):
    assign mem_req_o[${idx}].q.amo = amo_op_e'(mem_req_o_amo[${idx}]);
    assign mem_req_o[${idx}].q.user = '0;
% endfor


//-----------------------------
// Instantiate generated interconnect
//-----------------------------

SparseInterconnect sparse_interconnect_i (
    .clock ( clk_i   ),
    .reset ( ~rst_ni ),


    // Tcdm Requests
    // Catch: the tcdm request ready signal is in the response struct
    % for idx in range(cfg["sparse_interconnect_cfg"]["NumInp"]):
        .io_tcdmReqs_${idx}_ready ( rsp_o[${idx}].q_ready ),
        .io_tcdmReqs_${idx}_valid ( req_i[${idx}].q_valid ),
        .io_tcdmReqs_${idx}_bits_addr  ( req_i[${idx}].q.addr  ),
        .io_tcdmReqs_${idx}_bits_write ( req_i[${idx}].q.write ),
        .io_tcdmReqs_${idx}_bits_amo   ( req_i[${idx}].q.amo   ),
        .io_tcdmReqs_${idx}_bits_data  ( req_i[${idx}].q.data  ),
        .io_tcdmReqs_${idx}_bits_strb  ( req_i[${idx}].q.strb  ),
        .io_tcdmReqs_${idx}_bits_priority  ( req_i[${idx}].q.user.tcdm_priority   ),
    % endfor

    // Tcdm Responses, the ready signal is set to 1, it is ignored by the tcdm
    % for idx in range(cfg["sparse_interconnect_cfg"]["NumInp"]):
        .io_tcdmRsps_${idx}_ready ( 1'b1 ),
        .io_tcdmRsps_${idx}_valid ( rsp_o[${idx}].p_valid ),
        .io_tcdmRsps_${idx}_bits_data  ( rsp_o[${idx}].p.data  ),
    % endfor

    // Mem Requests, the ready signal comes fro the memrsp
    % for idx in range(cfg["sparse_interconnect_cfg"]["NumOut"]):
        .io_memReqs_${idx}_ready ( mem_rsp_i[${idx}].q_ready ),
        .io_memReqs_${idx}_valid ( mem_req_o[${idx}].q_valid ),
        .io_memReqs_${idx}_bits_addr  ( mem_req_o[${idx}].q.addr  ),
        .io_memReqs_${idx}_bits_write ( mem_req_o[${idx}].q.write ),
        .io_memReqs_${idx}_bits_amo   ( mem_req_o_amo[${idx}] ),
        .io_memReqs_${idx}_bits_data  ( mem_req_o[${idx}].q.data  ),
        .io_memReqs_${idx}_bits_strb  ( mem_req_o[${idx}].q.strb  ),
    % endfor
    // user signal not used
    // .io_memReqs_${idx}_bits_user  ( mem_req_o[${idx}].q.user  ),

    // Mem Responses
    // the valid signal is set to 1, it should be ignored by the mem interface)
    // the ready signal is ignored so it is left unconnected
    % for idx in range(cfg["sparse_interconnect_cfg"]["NumOut"]):
        .io_memRsps_${idx}_ready (  ),
        .io_memRsps_${idx}_valid ( 1'b1 ),
        .io_memRsps_${idx}_bits_data  ( mem_rsp_i[${idx}].p.data  ) \
        % if idx < cfg["sparse_interconnect_cfg"]["NumOut"] - 1:
            ,
        % endif
    % endfor

);

endmodule
