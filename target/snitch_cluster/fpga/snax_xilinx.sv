// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yunhao Deng <nwistoff@iis.ee.ethz.ch>

// The simple wrapper with very basic AXI infrastructure to avoid Vivado to optimize everything away
// The wrapper, after more development, is possible to make snitch work.

`include "axi/typedef.svh"
`include "axi_flat.sv"

module snax_xilinx (
    // Clock and reset
    input clk_i,
    input rst_ni,

    // CSR Observation Bits for external inspection
    output [7:0] obs_o,

    // AXI Slave interface for external inspection
    input s_axi_xilinx_awvalid,
    input [3:0] s_axi_xilinx_awid,
    input [47:0] s_axi_xilinx_awaddr,
    input [7:0] s_axi_xilinx_awlen,
    input [2:0] s_axi_xilinx_awsize,
    input [1:0] s_axi_xilinx_awburst,
    input s_axi_xilinx_awlock,
    input [3:0] s_axi_xilinx_awcache,
    input [2:0] s_axi_xilinx_awprot,
    input [3:0] s_axi_xilinx_awqos,
    input [3:0] s_axi_xilinx_awregion,
    input [1:0] s_axi_xilinx_awuser,

    input s_axi_xilinx_wvalid,
    input [63:0] s_axi_xilinx_wdata,
    input [7:0] s_axi_xilinx_wstrb,
    input s_axi_xilinx_wlast,
    input [1:0] s_axi_xilinx_wuser,

    input s_axi_xilinx_bready,

    input s_axi_xilinx_arvalid,
    input [3:0] s_axi_xilinx_arid,
    input [47:0] s_axi_xilinx_araddr,
    input [7:0] s_axi_xilinx_arlen,
    input [2:0] s_axi_xilinx_arsize,
    input [1:0] s_axi_xilinx_arburst,
    input s_axi_xilinx_arlock,
    input [3:0] s_axi_xilinx_arcache,
    input [2:0] s_axi_xilinx_arprot,
    input [3:0] s_axi_xilinx_arqos,
    input [3:0] s_axi_xilinx_arregion,
    input [1:0] s_axi_xilinx_aruser,

    input s_axi_xilinx_rready,

    output s_axi_xilinx_awready,
    output s_axi_xilinx_arready,
    output s_axi_xilinx_wready,

    output s_axi_xilinx_bvalid,
    output [3:0] s_axi_xilinx_bid,
    output [1:0] s_axi_xilinx_bresp,
    output [1:0] s_axi_xilinx_buser,

    output s_axi_xilinx_rvalid,
    output [3:0] s_axi_xilinx_rid,
    output [63:0] s_axi_xilinx_rdata,
    output [1:0] s_axi_xilinx_rresp,
    output s_axi_xilinx_rlast,
    output [1:0] s_axi_xilinx_ruser
);

  // From SNAX to XBar: narrow_out, wide ID
  // From XBar to SNAX: narrow_in, narrow ID
  // From Xilinx to XBar: narrow_out, wide ID
  // From XBar to Memory: narrow_in, narrow ID

  // XBar's Slave interface is wide
  // XBar's Master interface is narrow

  // The original AXI bus width used by the Snitch cluster
  `AXI_TYPEDEF_ALL(narrow_in, logic [47:0], logic [1:0], logic [63:0], logic [7:0], logic [1:0])
  `AXI_TYPEDEF_ALL(narrow_out, logic [47:0], logic [3:0], logic [63:0], logic [7:0], logic [1:0])

  // Snitch cluster AXI interfaces
  narrow_in_req_t narrow_snax_req_i;
  narrow_out_req_t narrow_snax_req_o, xilinx_s_req_i;
  narrow_in_resp_t narrow_snax_rsp_o;
  narrow_out_resp_t narrow_snax_rsp_i, xilinx_s_rsp_o;


  `AXI_FLATTEN_SLAVE(xilinx, xilinx_s_req_i, xilinx_s_rsp_o)


  snax_KUL_xdma_cluster_wrapper snax_cluster (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .obs_o(obs_o),
      .meip_i('0),
      .mtip_i('0),
      .msip_i('0),
      .hart_base_id_i('0),
      .cluster_base_addr_i(48'h10000000),
      .boot_addr_i(32'h80000000),
      .sram_cfgs_i('0),
      .narrow_in_req_i(narrow_snax_req_i),
      .narrow_in_resp_o(narrow_snax_rsp_o),
      .narrow_out_req_o(narrow_snax_req_o),
      .narrow_out_resp_i(narrow_snax_rsp_i),
      // Wide connect is not cared about
      .wide_out_req_o(),
      .wide_out_resp_i('0),
      .wide_in_req_i('0),
      .wide_in_resp_o()
  );

  narrow_in_req_t   xbar_to_snax_req;
  narrow_in_resp_t  xbar_to_snax_rsp;
  narrow_out_req_t  snax_to_xbar_req;
  narrow_out_resp_t snax_to_xbar_rsp;

  axi_multicut #(
      .NoCuts(32'd5),
      // AXI channel structs
      .aw_chan_t(narrow_out_aw_chan_t),
      .w_chan_t(narrow_out_w_chan_t),
      .b_chan_t(narrow_out_b_chan_t),
      .ar_chan_t(narrow_out_ar_chan_t),
      .r_chan_t(narrow_out_r_chan_t),
      // AXI request & response structs
      .axi_req_t(narrow_out_req_t),
      .axi_resp_t(narrow_out_resp_t)
  ) i_snax_out_cut (
      .clk_i,
      .rst_ni,
      // salve port
      .slv_req_i (narrow_snax_req_o),
      .slv_resp_o(narrow_snax_rsp_i),
      // master port
      .mst_req_o (snax_to_xbar_req),
      .mst_resp_i(snax_to_xbar_rsp)
  );

  narrow_in_req_t  snax_to_xbar_wc_req;
  narrow_in_resp_t snax_to_xbar_wc_rsp;

  axi_id_remap #(
      /// ID width of the AXI4+ATOP slave port.
      .AxiSlvPortIdWidth(4),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(16),
      .AxiMstPortIdWidth(2),
      .slv_req_t(narrow_out_req_t),
      .slv_resp_t(narrow_out_resp_t),
      .mst_req_t(narrow_in_req_t),
      .mst_resp_t(narrow_in_resp_t)
  ) i_snax_to_xbar_remap (
      .clk_i,
      .rst_ni,
      .slv_req_i (snax_to_xbar_req),
      .slv_resp_o(snax_to_xbar_rsp),
      .mst_req_o (snax_to_xbar_wc_req),
      .mst_resp_i(snax_to_xbar_wc_rsp)
  );

  narrow_out_req_t  xbar_to_snax_wc_req;
  narrow_out_resp_t xbar_to_snax_wc_rsp;

  axi_id_remap #(
      /// ID width of the AXI4+ATOP slave port.
      .AxiSlvPortIdWidth(4),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(16),
      .AxiMstPortIdWidth(2),
      .slv_req_t(narrow_out_req_t),
      .slv_resp_t(narrow_out_resp_t),
      .mst_req_t(narrow_in_req_t),
      .mst_resp_t(narrow_in_resp_t)
  ) i_xbar_to_snax_remap (
      .clk_i,
      .rst_ni,
      .slv_req_i (xbar_to_snax_wc_req),
      .slv_resp_o(xbar_to_snax_wc_rsp),
      .mst_req_o (xbar_to_snax_req),
      .mst_resp_i(xbar_to_snax_rsp)
  );

  axi_multicut #(
      .NoCuts(32'd5),
      // AXI channel structs
      .aw_chan_t(narrow_in_aw_chan_t),
      .w_chan_t(narrow_in_w_chan_t),
      .b_chan_t(narrow_in_b_chan_t),
      .ar_chan_t(narrow_in_ar_chan_t),
      .r_chan_t(narrow_in_r_chan_t),
      // AXI request & response structs
      .axi_req_t(narrow_in_req_t),
      .axi_resp_t(narrow_in_resp_t)
  ) i_snax_in_cut (
      .clk_i,
      .rst_ni,
      // salve port
      .slv_req_i (xbar_to_snax_req),
      .slv_resp_o(xbar_to_snax_rsp),
      // master port
      .mst_req_o (narrow_snax_req_i),
      .mst_resp_i(narrow_snax_rsp_o)
  );

  narrow_in_req_t xbar_to_mem_req, axi_mem_req;
  narrow_in_resp_t xbar_to_mem_rsp, axi_mem_rsp;

  narrow_out_req_t  xbar_to_mem_wc_req;
  narrow_out_resp_t xbar_to_mem_wc_rsp;

  axi_id_remap #(
      /// ID width of the AXI4+ATOP slave port.
      .AxiSlvPortIdWidth(4),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(16),
      .AxiMstPortIdWidth(2),
      .slv_req_t(narrow_out_req_t),
      .slv_resp_t(narrow_out_resp_t),
      .mst_req_t(narrow_in_req_t),
      .mst_resp_t(narrow_in_resp_t)
  ) i_xbar_to_mem_remap (
      .clk_i,
      .rst_ni,
      .slv_req_i (xbar_to_mem_wc_req),
      .slv_resp_o(xbar_to_mem_wc_rsp),
      .mst_req_o (xbar_to_mem_req),
      .mst_resp_i(xbar_to_mem_rsp)
  );

  axi_multicut #(
      .NoCuts(32'd1),
      // AXI channel structs
      .aw_chan_t(narrow_in_aw_chan_t),
      .w_chan_t(narrow_in_w_chan_t),
      .b_chan_t(narrow_in_b_chan_t),
      .ar_chan_t(narrow_in_ar_chan_t),
      .r_chan_t(narrow_in_r_chan_t),
      // AXI request & response structs
      .axi_req_t(narrow_in_req_t),
      .axi_resp_t(narrow_in_resp_t)
  ) i_mem_cut (
      .clk_i,
      .rst_ni,
      // salve port
      .slv_req_i (xbar_to_mem_req),
      .slv_resp_o(xbar_to_mem_rsp),
      // master port
      .mst_req_o (axi_mem_req),
      .mst_resp_i(axi_mem_rsp)
  );

  logic mem_req;
  logic mem_gnt;
  logic [47:0] mem_addr;
  logic [63:0] mem_wdata;
  logic [7:0] mem_strb;
  logic mem_we;
  logic mem_rvalid;
  logic [63:0] mem_rdata;

  axi_to_mem #(
      .axi_req_t(narrow_in_req_t),
      .axi_resp_t(narrow_in_resp_t),
      .AddrWidth(48),
      .DataWidth(64),
      .IdWidth(2),
      .NumBanks(1),
      .BufDepth(1),
      .HideStrb(0),
      .OutFifoDepth(1)
  ) i_axi_to_mem (
      .clk_i,
      .rst_ni,
      .busy_o(),
      .axi_req_i(axi_mem_req),
      .axi_resp_o(axi_mem_rsp),
      .mem_req_o(mem_req),
      .mem_gnt_i(mem_gnt),
      .mem_addr_o(mem_addr),
      .mem_wdata_o(mem_wdata),
      .mem_strb_o(mem_strb),
      .mem_atop_o(),
      .mem_we_o(mem_we),
      .mem_rvalid_i(mem_rvalid),
      .mem_rdata_i(mem_rdata)
  );

  // Memory instatiation

  spm_1p_adv #(
      .NumWords(65536),
      .DataWidth(64),
      .ByteWidth(8),
      .EnableInputPipeline(1'b1),
      .EnableOutputPipeline(1'b1)
  ) i_sram (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .valid_i(mem_req),
      .ready_o(mem_gnt),
      .we_i(mem_we),
      .addr_i(mem_addr[18:3]),
      .wdata_i(mem_wdata),
      .be_i(mem_strb),
      .rdata_o(mem_rdata),
      .rvalid_o(mem_rvalid),
      .rerror_o(),
      .sram_cfg_i('0)
  );

  narrow_out_req_t  xilinx_to_xbar_req;
  narrow_out_resp_t xilinx_to_xbar_rsp;

  narrow_in_req_t   xilinx_to_xbar_wc_req;
  narrow_in_resp_t  xilinx_to_xbar_wc_rsp;

  axi_id_remap #(
      /// ID width of the AXI4+ATOP slave port.
      .AxiSlvPortIdWidth(4),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(16),
      .AxiMstPortIdWidth(2),
      .slv_req_t(narrow_out_req_t),
      .slv_resp_t(narrow_out_resp_t),
      .mst_req_t(narrow_in_req_t),
      .mst_resp_t(narrow_in_resp_t)
  ) i_xilinx_to_xbar_remap (
      .clk_i,
      .rst_ni,
      .slv_req_i (xilinx_to_xbar_req),
      .slv_resp_o(xilinx_to_xbar_rsp),
      .mst_req_o (xilinx_to_xbar_wc_req),
      .mst_resp_i(xilinx_to_xbar_wc_rsp)
  );

  axi_multicut #(
      .NoCuts(32'd5),
      // AXI channel structs
      .aw_chan_t(narrow_out_aw_chan_t),
      .w_chan_t(narrow_out_w_chan_t),
      .b_chan_t(narrow_out_b_chan_t),
      .ar_chan_t(narrow_out_ar_chan_t),
      .r_chan_t(narrow_out_r_chan_t),
      // AXI request & response structs
      .axi_req_t(narrow_out_req_t),
      .axi_resp_t(narrow_out_resp_t)
  ) i_xilinx_xbar_cut (
      .clk_i,
      .rst_ni,
      // salve port
      .slv_req_i (xilinx_s_req_i),
      .slv_resp_o(xilinx_s_rsp_o),
      // master port
      .mst_req_o (xilinx_to_xbar_req),
      .mst_resp_i(xilinx_to_xbar_rsp)
  );

  // The xbar to connect everything

  localparam axi_pkg::xbar_cfg_t XbarCfg = '{
      NoSlvPorts: 2,
      NoMstPorts: 2,
      MaxSlvTrans: 64,
      MaxMstTrans: 64,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      PipelineStages: 0,
      AxiIdWidthSlvPorts: 2,
      AxiIdUsedSlvPorts: 2,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 64,
      NoAddrRules: 2
  };

  typedef struct packed {
    logic [31:0] idx;
    logic [47:0] start_addr;
    logic [47:0] end_addr;
  } xbar_rule_48_t;

  xbar_rule_48_t [1:0] SocWideXbarAddrmap;
  assign SocWideXbarAddrmap = '{
          '{idx: 0, start_addr: 48'h10000000, end_addr: 48'h11000000},
          '{idx: 1, start_addr: 48'h80000000, end_addr: 48'h80080000}
      };

  axi_xbar #(
      .Cfg          (XbarCfg),
      .Connectivity (4'b1111),
      .ATOPs        (0),
      .slv_aw_chan_t(narrow_in_aw_chan_t),
      .mst_aw_chan_t(narrow_out_aw_chan_t),
      .w_chan_t     (narrow_out_w_chan_t),
      .slv_b_chan_t (narrow_in_b_chan_t),
      .mst_b_chan_t (narrow_out_b_chan_t),
      .slv_ar_chan_t(narrow_in_ar_chan_t),
      .mst_ar_chan_t(narrow_out_ar_chan_t),
      .slv_r_chan_t (narrow_in_r_chan_t),
      .mst_r_chan_t (narrow_out_r_chan_t),
      .slv_req_t    (narrow_in_req_t),
      .slv_resp_t   (narrow_in_resp_t),
      .mst_req_t    (narrow_out_req_t),
      .mst_resp_t   (narrow_out_resp_t),
      .rule_t       (xbar_rule_48_t)
  ) i_soc_wide_xbar (
      .clk_i                (clk_i),
      .rst_ni               (rst_ni),
      .test_i               (test_mode_i),
      .slv_ports_req_i      ({xbar_to_mem_wc_req, xbar_to_snax_wc_req}),
      .slv_ports_resp_o     ({xbar_to_mem_wc_rsp, xbar_to_snax_wc_rsp}),
      .mst_ports_req_o      ({snax_to_xbar_wc_req, xilinx_to_xbar_wc_req}),
      .mst_ports_resp_i     ({snax_to_xbar_wc_rsp, xilinx_to_xbar_wc_rsp}),
      .addr_map_i           (XbarAddrmap),
      .en_default_mst_port_i('1),
      .default_mst_port_i   (1)
  );

endmodule
