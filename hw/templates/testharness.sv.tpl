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
  # The endpoint is the cluster's own xDMA, so its TCDM address is as wide as the cluster's.
  xdma_tcdm_addr_width = (int(cluster["tcdm"]["size"]) * 1024 - 1).bit_length()
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

  // The endpoint's own TCDM ports, served from the DPI image by tb_memory_tcdm below.
  localparam int unsigned XdmaTcdmPorts = ${round(cluster["dma_data_width"] / cluster["data_width"] * 2)};
  tcdm_req_t [XdmaTcdmPorts-1:0] xdma_mem_tcdm_req;
  tcdm_rsp_t [XdmaTcdmPorts-1:0] xdma_mem_tcdm_rsp;

  // The endpoint's master ports, registered below before they re-enter the cluster.
  wide_in_req_t    ep_wide_out_req;
  wide_in_resp_t   ep_wide_out_resp;
  narrow_in_req_t  ep_narrow_out_req;
  narrow_in_resp_t ep_narrow_out_resp;

  // WHERE THE ENDPOINT READS AND WRITES. It is the cluster's own xDMA: its TCDM address is
  // ${xdma_tcdm_addr_width} bits, and it takes a task as its own only when the task's pointer lies in the same
  // ${"%d" % (2 ** xdma_tcdm_addr_width // 1024)} KiB as its base address (SrcConfigRouter and DstConfigRouter, XDMACtrl.scala). It
  // stands for the xDMA that sits on ALL of main memory, as the one in hemaia_mem_system sits
  // on all of L3, so the testbench moves it to each task. A task's first cfg frame carries both
  // of its full 48-bit pointers: from the next cycle, before the task is routed, the endpoint's
  // base is the ${"%d" % (2 ** xdma_tcdm_addr_width // 1024)} KiB window holding the pointer that is in main memory, and the memory
  // below places the endpoint's addresses relative to the pointers themselves (tb_memory_tcdm).
  // The adapter's MMIO band follows any base in main memory, and a finish is matched by its
  // transfer id alone. Exact while the endpoint runs one task at a time and a task spans less
  // than ${"%d" % (2 ** xdma_tcdm_addr_width // 1024)} KiB: a hart starts its next main-memory transfer after the last one finished.
  localparam logic [AddrWidth-1:0] EpWindowMask = ~((48'd1 << ${xdma_tcdm_addr_width}) - 48'd1);

  logic [AddrWidth-1:0] ep_base_q, ep_cfg_rd, ep_cfg_wr;
  logic [63:0]          ep_rd_base_q, ep_wr_base_q;
  logic [3:0]           ep_frames_left_q;

  assign ep_cfg_rd = AddrWidth'(i_tb_xdma_endpoint.xdma_from_remote_cfg.reader_addr);
  assign ep_cfg_wr = AddrWidth'(i_tb_xdma_endpoint.xdma_from_remote_cfg.writer_addr);

  function automatic logic in_main_mem(logic [AddrWidth-1:0] addr);
    return (addr >= MainMemBase) && (addr < XdmaMMIOBase);
  endfunction

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ep_base_q        <= MainMemBase;
      ep_rd_base_q     <= 64'(MainMemBase);
      ep_wr_base_q     <= 64'(MainMemBase);
      ep_frames_left_q <= '0;
    end else if (i_tb_xdma_endpoint.xdma_from_remote_cfg_valid &&
                 i_tb_xdma_endpoint.xdma_from_remote_cfg_ready) begin
      if (ep_frames_left_q == '0) begin
        ep_rd_base_q <= 64'(ep_cfg_rd);
        ep_wr_base_q <= 64'(ep_cfg_wr);
        if (in_main_mem(ep_cfg_rd))      ep_base_q <= ep_cfg_rd & EpWindowMask;
        else if (in_main_mem(ep_cfg_wr)) ep_base_q <= ep_cfg_wr & EpWindowMask;
        ep_frames_left_q <= (i_tb_xdma_endpoint.xdma_from_remote_cfg.frame_length > 4'd1) ?
                            i_tb_xdma_endpoint.xdma_from_remote_cfg.frame_length - 4'd1 : '0;
      end else begin
        ep_frames_left_q <= ep_frames_left_q - 4'd1;
      end
    end
  end

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
    // main memory instead of at the top of a cluster slot. Which window of main memory
    // follows the task it runs (below).
    .cluster_base_addr_i ( ep_base_q ),
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
    .NumPorts    ( XdmaTcdmPorts  ),
    .DataWidth   ( NarrowDataWidth),
    .WindowWidth ( ${xdma_tcdm_addr_width} ),
    .tcdm_req_t  ( tcdm_req_t     ),
    .tcdm_rsp_t  ( tcdm_rsp_t     )
  ) i_tb_xdma_endpoint_mem (
    .clk_i     ( clk_i  ),
    .rst_ni    ( rst_ni ),
    .rd_base_i ( ep_rd_base_q ),
    .wr_base_i ( ep_wr_base_q ),
    .req_i     ( xdma_mem_tcdm_req ),
    .rsp_o     ( xdma_mem_tcdm_rsp )
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
