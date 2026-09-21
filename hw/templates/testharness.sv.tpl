// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "axi/typedef.svh"

<%
  # An xDMA transfer is a conversation between TWO xDMA endpoints, never between an
  # engine and a bus. A cluster reading main memory therefore needs a second xDMA
  # sitting on that memory to answer it. On HeMAiA that endpoint is inside
  # `hemaia_mem_system`; here it has to be built, or the cluster's xDMA can only
  # reach itself and the kernel is forced to move every byte with the iDMA.
  cluster = cfg["cluster"]
  has_xdma = any(
      "snax_xdma_cfg" in core
      for hive in cluster["hives"]
      for core in hive["cores"]
  )
  dram = cfg.get("dram") or {}
  main_mem_base = int(dram.get("address", 0x80000000))
  main_mem_end = main_mem_base + int(dram.get("length", 0x80000000))
  # Must match the cluster's own xDMA wrapper, which the cluster wrapper fixes at 16.
  xdma_mmio_size_kib = 16
%>
module testharness import ${cluster["name"]}_pkg::*; (
  input logic clk_i,
  input logic rst_ni
);

  import "DPI-C" function void clint_tick(
    output byte msip[]
  );

  narrow_in_req_t   narrow_in_req;
  narrow_in_resp_t  narrow_in_resp;
  narrow_out_req_t  narrow_out_req;
  narrow_out_resp_t narrow_out_resp;
  wide_out_req_t    wide_out_req;
  wide_out_resp_t   wide_out_resp;
  wide_in_req_t     wide_in_req;
  wide_in_resp_t    wide_in_resp;

  logic [${cluster["name"]}_pkg::NrCores-1:0] msip;

  ${cluster["name"]}_wrapper i_${cluster["name"]} (
    .clk_i                ( clk_i           ),
    .rst_ni               ( rst_ni          ),
    .hart_base_id_i       ( HartBaseID      ),
    .cluster_base_addr_i  ( ClusterBaseAddr ),
    .boot_addr_i          ( BootAddr        ),
% if cluster["timing"]["iso_crossings"]:
    .clk_d2_bypass_i      ( '0              ),
% endif
% if cluster["enable_debug"]:
    .debug_req_i          ( '0              ),
% endif
% if cluster["sram_cfg_expose"]:
    .sram_cfgs_i          ( '0             ),
% endif
    .meip_i               ( '0              ),
    .mtip_i               ( '0              ),
    .msip_i               ( msip            ),
    .narrow_in_req_i      ( narrow_in_req   ),
    .narrow_in_resp_o     ( narrow_in_resp  ),
    .narrow_out_req_o     ( narrow_out_req  ),
    .narrow_out_resp_i    ( narrow_out_resp ),
    .wide_out_req_o       ( wide_out_req    ),
    .wide_out_resp_i      ( wide_out_resp   ),
    .wide_in_req_i        ( wide_in_req     ),
    .wide_in_resp_o       ( wide_in_resp    )
  );

% if has_xdma:
  //---------------------------------------------------------------------------
  // Main-memory xDMA endpoint
  //---------------------------------------------------------------------------
  // The cluster's xDMA addresses a peer by that peer's address space, and reserves
  // the top 16 KiB of it for the four MMIO windows the peers talk over (data, cfg,
  // grant, finish). An endpoint whose OWN base lies in main memory takes its window
  // from the top of main memory rather than from a cluster-sized slot --
  // `local_end_addr` in xdma_axi_adapter_top.sv -- so both ends land on the same
  // band, ${"0x%x" % (main_mem_end - xdma_mmio_size_kib * 1024)}..${"0x%x" % main_mem_end},
  // with no address invented here to make them meet.
  //
  // This is the same module the cluster instantiates, elaborated once and used
  // twice. Generating a second one would emit a second copy of every library module
  // the two share, under the same names, which does not compile -- the same reason
  // snaxgen elaborates the xDMA and SIMD blocks together.
  localparam logic [AddrWidth-1:0] MainMemBase    = ${"48'h%x" % main_mem_base};
  localparam logic [AddrWidth-1:0] MainMemEnd     = ${"48'h%x" % main_mem_end};
  localparam int unsigned          XdmaMMIOSizeKiB = ${xdma_mmio_size_kib};
  localparam logic [AddrWidth-1:0] XdmaMMIOBase   = MainMemEnd - XdmaMMIOSizeKiB * 1024;

  // Everything below the MMIO band is ordinary memory and keeps going to the DPI
  // image; only the four windows are the endpoint's.
  wide_out_req_t   [1:0] wide_demux_req;
  wide_out_resp_t  [1:0] wide_demux_rsp;
  narrow_out_req_t [1:0] narrow_demux_req;
  narrow_out_resp_t[1:0] narrow_demux_rsp;

  function automatic logic addr_is_xdma_mmio(logic [AddrWidth-1:0] addr);
    return (addr >= XdmaMMIOBase) && (addr < MainMemEnd);
  endfunction

  // ATOMICS MUST PASS. snRuntime's global barrier and its allocator use AMOs on main
  // memory, and an axi_demux built with AtopSupport 0 does not carry the atomic's
  // response back -- the core issues one amoadd during boot and then waits on the
  // next load for ever. It presents as a simulation that is merely slow.
  axi_demux #(
    .AxiIdWidth  ( WideIdWidthOut     ),
    .AtopSupport ( 1'b1               ),
    .aw_chan_t   ( wide_out_aw_chan_t ),
    .w_chan_t    ( wide_out_w_chan_t  ),
    .b_chan_t    ( wide_out_b_chan_t  ),
    .ar_chan_t   ( wide_out_ar_chan_t ),
    .r_chan_t    ( wide_out_r_chan_t  ),
    .axi_req_t   ( wide_out_req_t     ),
    .axi_resp_t  ( wide_out_resp_t    ),
    .NoMstPorts  ( 2                  ),
    .MaxTrans    ( 32'd8              ),
    .AxiLookBits ( WideIdWidthOut     ),
    .SpillAw     ( 1'b1               ),
    .SpillAr     ( 1'b1               )
  ) i_wide_demux (
    .clk_i           ( clk_i  ),
    .rst_ni          ( rst_ni ),
    .test_i          ( 1'b0   ),
    .slv_req_i       ( wide_out_req  ),
    .slv_aw_select_i ( addr_is_xdma_mmio(wide_out_req.aw.addr) ),
    .slv_ar_select_i ( addr_is_xdma_mmio(wide_out_req.ar.addr) ),
    .slv_resp_o      ( wide_out_resp ),
    .mst_reqs_o      ( wide_demux_req ),
    .mst_resps_i     ( wide_demux_rsp )
  );

  axi_demux #(
    .AxiIdWidth  ( NarrowIdWidthOut     ),
    .AtopSupport ( 1'b1                 ),
    .aw_chan_t   ( narrow_out_aw_chan_t ),
    .w_chan_t    ( narrow_out_w_chan_t  ),
    .b_chan_t    ( narrow_out_b_chan_t  ),
    .ar_chan_t   ( narrow_out_ar_chan_t ),
    .r_chan_t    ( narrow_out_r_chan_t  ),
    .axi_req_t   ( narrow_out_req_t     ),
    .axi_resp_t  ( narrow_out_resp_t    ),
    .NoMstPorts  ( 2                    ),
    .MaxTrans    ( 32'd8                ),
    .AxiLookBits ( NarrowIdWidthOut     ),
    .SpillAw     ( 1'b1                 ),
    .SpillAr     ( 1'b1                 )
  ) i_narrow_demux (
    .clk_i           ( clk_i  ),
    .rst_ni          ( rst_ni ),
    .test_i          ( 1'b0   ),
    .slv_req_i       ( narrow_out_req  ),
    .slv_aw_select_i ( addr_is_xdma_mmio(narrow_out_req.aw.addr) ),
    .slv_ar_select_i ( addr_is_xdma_mmio(narrow_out_req.ar.addr) ),
    .slv_resp_o      ( narrow_out_resp ),
    .mst_reqs_o      ( narrow_demux_req ),
    .mst_resps_i     ( narrow_demux_rsp )
  );

  // The endpoint's own TCDM ports. It emits offsets into the memory it sits on, so
  // the bridge below adds the main-memory base to reach the DPI image.
  localparam int unsigned XdmaTcdmPorts = ${round(cluster["dma_data_width"] / cluster["data_width"] * 2)};
  tcdm_req_t [XdmaTcdmPorts-1:0] xdma_mem_tcdm_req;
  tcdm_rsp_t [XdmaTcdmPorts-1:0] xdma_mem_tcdm_rsp;

  ${cluster["name"]}_xdma_wrapper #(
    .tcdm_req_t         ( tcdm_req_t        ),
    .tcdm_rsp_t         ( tcdm_rsp_t        ),
    // Crossed exactly as the cluster crosses them: what this endpoint drives OUT is
    // what the cluster takes IN, and the other way round.
    .wide_slv_id_t      ( wide_out_id_t     ),
    .wide_out_req_t     ( wide_in_req_t     ),
    .wide_out_resp_t    ( wide_in_resp_t    ),
    .wide_in_req_t      ( wide_out_req_t    ),
    .wide_in_resp_t     ( wide_out_resp_t   ),
    .narrow_slv_id_t    ( narrow_out_id_t   ),
    .narrow_out_req_t   ( narrow_in_req_t   ),
    .narrow_out_resp_t  ( narrow_in_resp_t  ),
    .narrow_in_req_t    ( narrow_out_req_t  ),
    .narrow_in_resp_t   ( narrow_out_resp_t ),
    .ClusterBaseAddr    ( MainMemBase       ),
    .ClusterAddressSpace( 48'h400000        ),
    .MMIOSize           ( XdmaMMIOSizeKiB   )
  ) i_tb_xdma_endpoint (
    .clk_i  ( clk_i  ),
    .rst_ni ( rst_ni ),
    // Being in main memory is what puts this endpoint's MMIO band at the top of
    // main memory instead of at the top of a cluster slot.
    .cluster_base_addr_i ( MainMemBase ),
    // No software configures this side. A remote transfer arrives as a cfg frame
    // over the narrow link, which is the whole point of the endpoint.
    .csr_req_bits_data_i  ( '0    ),
    .csr_req_bits_strb_i  ( '0    ),
    .csr_req_bits_addr_i  ( '0    ),
    .csr_req_bits_write_i ( 1'b0  ),
    .csr_req_valid_i      ( 1'b0  ),
    .csr_req_ready_o      (       ),
    .csr_rsp_bits_data_o  (       ),
    .csr_rsp_valid_o      (       ),
    .csr_rsp_ready_i      ( 1'b1  ),
    .xdma_wide_out_req_o    ( ep_wide_out_req   ),
    .xdma_wide_out_resp_i   ( ep_wide_out_resp  ),
    .xdma_wide_in_req_i     ( wide_demux_req[1] ),
    .xdma_wide_in_resp_o    ( wide_demux_rsp[1] ),
    .xdma_narrow_out_req_o  ( ep_narrow_out_req   ),
    .xdma_narrow_out_resp_i ( ep_narrow_out_resp  ),
    .xdma_narrow_in_req_i   ( narrow_demux_req[1] ),
    .xdma_narrow_in_resp_o  ( narrow_demux_rsp[1] ),
    .tcdm_req_o ( xdma_mem_tcdm_req ),
    .tcdm_rsp_i ( xdma_mem_tcdm_rsp )
  );

  // Register the endpoint's master ports before they re-enter the cluster. This is
  // what breaks the ring: without it the cluster's outbound ready depends on the
  // endpoint's logic which depends on the cluster's outbound valid, and a real
  // inter-cluster link would carry these registers in any case.
  wide_in_req_t    ep_wide_out_req;
  wide_in_resp_t   ep_wide_out_resp;
  narrow_in_req_t  ep_narrow_out_req;
  narrow_in_resp_t ep_narrow_out_resp;

  axi_cut #(
    .Bypass     ( 1'b0              ),
    .aw_chan_t  ( wide_in_aw_chan_t ),
    .w_chan_t   ( wide_in_w_chan_t  ),
    .b_chan_t   ( wide_in_b_chan_t  ),
    .ar_chan_t  ( wide_in_ar_chan_t ),
    .r_chan_t   ( wide_in_r_chan_t  ),
    .axi_req_t  ( wide_in_req_t     ),
    .axi_resp_t ( wide_in_resp_t    )
  ) i_ep_wide_cut (
    .clk_i      ( clk_i            ),
    .rst_ni     ( rst_ni           ),
    .slv_req_i  ( ep_wide_out_req  ),
    .slv_resp_o ( ep_wide_out_resp ),
    .mst_req_o  ( wide_in_req      ),
    .mst_resp_i ( wide_in_resp     )
  );

  axi_cut #(
    .Bypass     ( 1'b0                ),
    .aw_chan_t  ( narrow_in_aw_chan_t ),
    .w_chan_t   ( narrow_in_w_chan_t  ),
    .b_chan_t   ( narrow_in_b_chan_t  ),
    .ar_chan_t  ( narrow_in_ar_chan_t ),
    .r_chan_t   ( narrow_in_r_chan_t  ),
    .axi_req_t  ( narrow_in_req_t     ),
    .axi_resp_t ( narrow_in_resp_t    )
  ) i_ep_narrow_cut (
    .clk_i      ( clk_i              ),
    .rst_ni     ( rst_ni             ),
    .slv_req_i  ( ep_narrow_out_req  ),
    .slv_resp_o ( ep_narrow_out_resp ),
    .mst_req_o  ( narrow_in_req      ),
    .mst_resp_i ( narrow_in_resp     )
  );

  tb_memory_tcdm #(
    .NumPorts   ( XdmaTcdmPorts  ),
    .DataWidth  ( NarrowDataWidth),
    .BaseAddr   ( ${"64'h%x" % main_mem_base} ),
    .tcdm_req_t ( tcdm_req_t     ),
    .tcdm_rsp_t ( tcdm_rsp_t     )
  ) i_tb_xdma_endpoint_mem (
    .clk_i  ( clk_i  ),
    .rst_ni ( rst_ni ),
    .req_i  ( xdma_mem_tcdm_req ),
    .rsp_o  ( xdma_mem_tcdm_rsp )
  );
% else:
  // No xDMA in this cluster, so nothing drives the inbound ports.
  assign narrow_in_req = '0;
  assign wide_in_req   = '0;
% endif

  // Narrow port into simulation memory.
  tb_memory_axi #(
    .AxiAddrWidth ( AddrWidth         ),
    .AxiDataWidth ( NarrowDataWidth   ),
    .AxiIdWidth   ( NarrowIdWidthOut  ),
    .AxiUserWidth ( NarrowUserWidth   ),
    .req_t        ( narrow_out_req_t  ),
    .rsp_t        ( narrow_out_resp_t )
  ) i_mem (
    .clk_i        ( clk_i             ),
    .rst_ni       ( rst_ni            ),
% if has_xdma:
    .req_i        ( narrow_demux_req[0] ),
    .rsp_o        ( narrow_demux_rsp[0] )
% else:
    .req_i        ( narrow_out_req    ),
    .rsp_o        ( narrow_out_resp   )
% endif
  );

  // Wide port into simulation memory.
  tb_memory_axi #(
    .AxiAddrWidth ( AddrWidth       ),
    .AxiDataWidth ( WideDataWidth   ),
    .AxiIdWidth   ( WideIdWidthOut  ),
    .AxiUserWidth ( WideUserWidth   ),
    .req_t        ( wide_out_req_t  ),
    .rsp_t        ( wide_out_resp_t )
  ) i_dma (
    .clk_i        ( clk_i           ),
    .rst_ni       ( rst_ni          ),
% if has_xdma:
    .req_i        ( wide_demux_req[0] ),
    .rsp_o        ( wide_demux_rsp[0] )
% else:
    .req_i        ( wide_out_req    ),
    .rsp_o        ( wide_out_resp   )
% endif
  );

  // CLINT
  // verilog_lint: waive-start always-ff-non-blocking
  localparam int NumCores = ${cluster["name"]}_pkg::NrCores;
  always_ff @(posedge clk_i) begin
    automatic byte msip_ret[NumCores];
    if (rst_ni) begin
      clint_tick(msip_ret);
      for (int i = 0; i < NumCores; i++) begin
        msip[i] = msip_ret[i];
      end
    end
  end
  // verilog_lint: waive-stop always-ff-non-blocking

endmodule
