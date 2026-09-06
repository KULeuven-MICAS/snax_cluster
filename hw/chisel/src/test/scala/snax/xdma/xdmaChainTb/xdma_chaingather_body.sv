// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Authors:
// - Fanchen Kong <fanchen.kong@kuleuven.be>
//
// ============================================================================================
// An N-endpoint ChainGather testbench, at snax level.
// ============================================================================================
//
// This replaces the HeMAiA `snax-xdma-chain-gather-sweep` loop for everything that does not
// need a real D2D hop. It keeps exactly the stack a ChainGather bug has ever lived in --
// the Chisel xDMA frontend AND `xdma_axi_adapter_top`, both inside
// `snax_xdma_cluster_xdma_wrapper` -- and drops the host, the boot flow, the routers and the
// mesh. Stimulus enters where software does: the CSR port.
//
// The deliberate scope cut is that EVERY ENDPOINT CAN REACH EVERY OTHER IN ONE HOP. Chain
// adjacency is meaningless here, which removes the topology as a variable. A bug that needs a
// real D2D hop will not reproduce here.
//
// ---- What it exercises that nothing else in this repo does -------------------------------
//
// `XDMADataSwitchTester` covers the switch alone, `ChainUnrollTester` the cfg plane alone, and
// `XDMADataPathGatherTester` one datapath in isolation. None of them has a SECOND endpoint, so
// none can form a chain: the grant that must cascade tail -> middle -> head before any data
// moves, the finish that must walk back, and the whole adapter, are all absent. This is the
// smallest thing that has all of it.
//
// ---- The chain -----------------------------------------------------------------------------
//
//   endpoint 0  ->  endpoint 1  ->  ...  ->  endpoint P-2  ->  endpoint P-1
//     HEAD           MIDDLE                    MIDDLE           TAIL / collector
//
// The collector issues the task. `P=2` has NO middle hop and is the control -- it has always
// passed. Every failure so far has needed at least one middle.
//
// Each participating endpoint contributes one 64 B partial from its own TCDM; the answer lands
// in the collector's TCDM at a DIFFERENT offset, so the fold cannot be confused with an operand.
//
// ---- Rounds --------------------------------------------------------------------------------
//
// `NumRounds` is the whole point. The bug class is "only the first gather through a chain with
// middle nodes works": round 1 byte-exact, round 2 returns having moved nothing. A sweep that
// runs each configuration ONCE cannot see any of it.

`timescale 1ns / 1ps
`include "axi/typedef.svh"
`include "tcdm_interface/typedef.svh"

module xdma_chaingather_body #(
    /// How many endpoints to instantiate.
    parameter int unsigned NumEndpoints = 3,
    /// Chain width P for a single-configuration run: endpoints 0..P-1 take part and endpoint 0
    /// is always the collector. Needs P <= NumEndpoints. Ignored when Sweep is set.
    parameter int unsigned ChainWidth   = 3,
    /// How many times to run the identical gather. MUST be >= 2 to gate the known bug.
    parameter int unsigned NumRounds    = 3,
    /// 0 = ElementwiseJunction (linear, FP32 ADD), 1 = MonoidJunction (nonlinear (m,l) merge).
    /// Ignored when Sweep is set, which runs both.
    parameter int unsigned JunctionId   = 0,
    /// 1 = run the whole scaling sweep in ONE simulation: every width in SweepP that fits, times
    /// both folds, and print the latency table. This is the snax-level twin of the HeMAiA app
    /// snax-xdma-chain-gather-sweep.
    parameter bit          Sweep        = 1'b0,
    /// Beats per transfer. 1 beat = 64 B = 16 FP32 lanes.
    parameter int unsigned NumBeats     = 1,
    /// 0 = per-round summary only. 1 = + commit/finish counters and every completion pulse.
    /// 2 = + every edge of the composed busy level with its components (very noisy at P=16, but
    /// it is what localises a spurious completion to a cycle).
    parameter int unsigned Verbose      = 1
) ();

  // ==========================================================================================
  // System constants
  // ==========================================================================================
  localparam int unsigned AxiAddrWidth       = 32'd48;
  localparam int unsigned AxiWideDataWidth   = 32'd512;
  localparam int unsigned AxiNarrowDataWidth = 32'd64;

  typedef logic [AxiAddrWidth-1:0] tb_addr_t;

  // The endpoint stride MUST be at least the xDMA's cluster-tag granularity, which is
  // `log2Ceil(tcdmSize KiB) + 10` = 17 bits for this config (tcdm.size: 128). It is set to the
  // real system's `cluster_base_offset` (4 MiB) so the address arithmetic here is identical to
  // HeMAiA's. A smaller stride would give two endpoints the SAME cluster tag, and the config
  // routers -- which decode nothing but that tag -- would treat a remote pointer as local and
  // the chain would silently short-circuit.
  localparam tb_addr_t    ClusterBaseAddr     = 48'h1000_0000;
  localparam tb_addr_t    ClusterAddressSpace = 48'h0040_0000;
  localparam tb_addr_t    MainMemBaseAddr     = 48'h8000_0000;
  localparam tb_addr_t    MainMemEndAddr      = 48'h1_0000_0000;
  localparam int          MMIOSize            = 16;

  localparam time CyclTime   = 10ns;
  localparam time SimTimeout = 20ms;

  // TCDM map, deliberately the same one snax-xdma-chain-gather-sweep.c uses, so a number
  // measured here and a number measured on HeMAiA refer to the same layout. Each endpoint keeps
  // BOTH partials; the collector's two results land clear of them, so a fold that accidentally
  // writes over an operand shows up as a wrong answer rather than a pass.
  localparam int unsigned LinSrcOffset = 32'h0000_0000;
  localparam int unsigned LinDstOffset = 32'h0000_0040;
  localparam int unsigned MomSrcOffset = 32'h0000_0080;
  localparam int unsigned MomDstOffset = 32'h0000_00C0;

  localparam int unsigned ModeLin = 0;
  localparam int unsigned ModeMom = 1;

  // The chain widths to sweep, exactly the HeMAiA app's SWEEP_P. A width larger than
  // NumEndpoints is skipped rather than failing the build, so one sweep body serves 4 and 16
  // endpoint builds alike.
  localparam int unsigned NumSweepP = 4;
  localparam int unsigned SweepP[NumSweepP] = '{2, 4, 8, 16};

  // MonoidJunction lane map for the online-softmax (m, l) partial: geometry sigma=3 -> S=8, so
  // field-major lane = field*S + slot puts the key m at lane 0 and the value l at lane 8.
  localparam int unsigned MomentMLane = 0;
  localparam int unsigned MomentLLane = 8;
  // l* goes through the writer's exp LUT, so it is checked to a ULP bound rather than exactly;
  // m* is a max and must be exact.
  //
  // The HeMAiA app carries 0x20000 (~1.5%), sized before anyone had measured the LUT. MEASURED
  // here against a `real`-arithmetic golden, over the whole sweep: 0, 2, 9 and 8 ULP for
  // P = 2, 4, 8, 16. A bound four orders of magnitude above the observed error is not a gate, so
  // this one is set to 256 -- still ~28x headroom on the worst case, but tight enough that a
  // genuine regression in the exp path fails instead of passing. Widen it deliberately (and say
  // why) if the key spread in mom_m() is ever pushed further into the LUT's tails.
  localparam logic [31:0] LTolUlps = 32'h0000_0100;

  localparam int unsigned BeatBytes = 64;
  localparam int unsigned XferBytes = BeatBytes * NumBeats;
  localparam int unsigned LanesPerBeat = 16;  // 64 B / 4 B

  function automatic tb_addr_t cluster_base(input int unsigned i);
    return ClusterBaseAddr + i * ClusterAddressSpace;
  endfunction

  // ==========================================================================================
  // CSR map. Indices are PLAIN 0-BASED REGISTER INDICES.
  //
  // `SNAX_XDMA_CFG_ADDR = 960` in snax_xdma_lib.h is the RISC-V CSR *number* base, stripped by
  // the snitch cluster's CSR decode -- which this testbench bypasses entirely. Drive the
  // XDMA_*_PTR value, never 960 + it.
  //
  // Cross-checked against ReqRspManager's own bounds assertions in the generated netlist:
  //   "csr write address overflow! Address: %d, Max: 77"
  //   "csr read address overflow! Max allowed address is 85"
  // i.e. 78 read-write registers (0..77) then 8 read-only ones (78..85). If a regenerated
  // netlist prints different bounds, the config changed and this map is stale.
  // ==========================================================================================
  localparam int unsigned CSR_SRC_ADDR        = 0;   // +0 LSB, +1 MSB
  localparam int unsigned CSR_DST_ADDR        = 2;   // 16 slots x 2 CSRs; slot k at 2 + 2*k
  localparam int unsigned CSR_SRC_SPAT_STRIDE = 34;
  localparam int unsigned CSR_SRC_TEMP_BOUND  = 35;  // +0..4
  localparam int unsigned CSR_SRC_TEMP_STRIDE = 40;  // +0..4
  localparam int unsigned CSR_SRC_ENAB_CHAN   = 45;
  localparam int unsigned CSR_SRC_ENABLE      = 46;
  localparam int unsigned CSR_SRC_EXT         = 47;  // +0..9
  localparam int unsigned CSR_DST_SPAT_STRIDE = 57;
  localparam int unsigned CSR_DST_TEMP_BOUND  = 58;  // +0..4
  localparam int unsigned CSR_DST_TEMP_STRIDE = 63;  // +0..4
  localparam int unsigned CSR_DST_ENAB_CHAN   = 68;
  localparam int unsigned CSR_DST_ENAB_BYTE   = 69;
  localparam int unsigned CSR_DST_ENABLE      = 70;
  localparam int unsigned CSR_DST_EXT         = 71;
  // The junction bank puts the ENABLE BITMASK FIRST, then the per-junction user CSRs.
  // Reversing them arms no junction, the transfer degrades to a plain chained write, and a test
  // that only checks "it completed" passes vacuously.
  localparam int unsigned CSR_DST_JCT_ENABLE  = 72;
  localparam int unsigned CSR_DST_JCT_CSR     = 73;  // +0..3, one per junction
  localparam int unsigned CSR_START           = 77;
  localparam int unsigned CSR_COMMIT_LOCAL    = 78;
  localparam int unsigned CSR_COMMIT_REMOTE   = 79;
  localparam int unsigned CSR_FINISH_LOCAL    = 80;
  localparam int unsigned CSR_FINISH_REMOTE   = 81;
  localparam int unsigned CSR_PERF_TASK       = 82;
  localparam int unsigned CSR_JCT_STATUS      = 85;

  // Junction ids, in elaboration order (snax_xdma_cluster.hjson writer_junctions).
  localparam int unsigned JCT_ELEMENTWISE = 0;
  localparam int unsigned JCT_MONOID      = 1;

  // ElementwiseJunction CSR(0): [3:0] op (0=ADD), [6:4] fmt (3=FP32).
  localparam logic [31:0] JCT_CSR_LINEAR = 32'h0000_0030;
  // MonoidJunction CSR(0) is a GEOMETRY word, not a flag word:
  //   [7:0] nValid | [11:8] n | [21:18] nExp | [25:22] nAdd | [27:26] sigma | [28] keyPol
  // For the online-softmax (m,l) merge the value is 0x0C040101. The plausible-looking
  // (1<<13)|1 decodes to n=0, sigma=0 -- key-only, S=1 -- so `l` at lane 8 is never read and
  // the fold silently returns garbage.
  localparam logic [31:0] JCT_CSR_MONOID = 32'h0C04_0101;

  localparam int unsigned ActiveJunction = (JunctionId == 0) ? JCT_ELEMENTWISE : JCT_MONOID;
  localparam logic [31:0] ActiveJctCsr   = (JunctionId == 0) ? JCT_CSR_LINEAR : JCT_CSR_MONOID;
  localparam string       FoldName       = (JunctionId == 0) ? "linear" : "monoid";

  // ==========================================================================================
  // TCDM types -- defined locally rather than importing snax_xdma_cluster_pkg, which drags in
  // the whole cluster. `CoreIDWidth = idx_width(NrCores=2) = 1`.
  // ==========================================================================================
  localparam int unsigned TcdmNumPorts  = 16;  // 8 reader + 8 writer
  localparam int unsigned TcdmDataWidth = 64;
  localparam int unsigned TcdmAddrWidth = 17;  // 128 KiB
  localparam int unsigned CoreIDWidth   = 1;
  localparam int unsigned TcdmWords     = (1 << TcdmAddrWidth) / (TcdmDataWidth / 8);

  typedef logic [  TcdmAddrWidth-1:0] tb_tcdm_addr_t;
  typedef logic [ TcdmDataWidth-1:0] tb_tcdm_data_t;
  typedef logic [TcdmDataWidth/8-1:0] tb_tcdm_strb_t;

  typedef struct packed {
    logic [CoreIDWidth-1:0] core_id;
    bit                     is_core;
    logic                   tcdm_priority;
  } tb_tcdm_user_t;

  `TCDM_TYPEDEF_ALL(tb_tcdm, tb_tcdm_addr_t, tb_tcdm_data_t, tb_tcdm_strb_t, tb_tcdm_user_t)

  // ==========================================================================================
  // AXI types and the two crossbars. Both buses are required: data rides the wide bus, and
  // cfg, grant and finish ALL ride the narrow bus, so a chain cannot form without it.
  // ==========================================================================================
  localparam int unsigned TbAxiUserWidth = 32'd1;
  localparam int unsigned TbIdWidthIn    = 32'd4;
  localparam int unsigned TbIdWidthOut   = $clog2(NumEndpoints) + TbIdWidthIn;
  localparam int unsigned TbPipeline     = 32'd1;

  typedef logic [               TbIdWidthIn-1:0] id_mst_t;
  typedef logic [              TbIdWidthOut-1:0] id_slv_t;
  typedef logic [            TbAxiUserWidth-1:0] user_t;
  typedef logic [        AxiWideDataWidth-1:0] data_wide_t;
  typedef logic [      AxiWideDataWidth/8-1:0] strb_wide_t;
  typedef logic [      AxiNarrowDataWidth-1:0] data_narrow_t;
  typedef logic [    AxiNarrowDataWidth/8-1:0] strb_narrow_t;

  `AXI_TYPEDEF_ALL(axi_wide_mst, tb_addr_t, id_mst_t, data_wide_t, strb_wide_t, user_t)
  `AXI_TYPEDEF_ALL(axi_wide_slv, tb_addr_t, id_slv_t, data_wide_t, strb_wide_t, user_t)
  `AXI_TYPEDEF_ALL(axi_narrow_mst, tb_addr_t, id_mst_t, data_narrow_t, strb_narrow_t, user_t)
  `AXI_TYPEDEF_ALL(axi_narrow_slv, tb_addr_t, id_slv_t, data_narrow_t, strb_narrow_t, user_t)

  typedef struct packed {
    int unsigned idx;
    tb_addr_t    start_addr;
    tb_addr_t    end_addr;
  } tb_rule_t;

  function automatic tb_rule_t [NumEndpoints-1:0] addr_map_gen();
    for (int unsigned i = 0; i < NumEndpoints; i++) begin
      addr_map_gen[i] = tb_rule_t'{
          idx: i,
          start_addr: ClusterBaseAddr + i * ClusterAddressSpace,
          end_addr: ClusterBaseAddr + (i + 1) * ClusterAddressSpace
      };
    end
  endfunction

  localparam tb_rule_t [NumEndpoints-1:0] XbarRule = addr_map_gen();

  localparam axi_pkg::xbar_cfg_t WideXbarCfg = '{
      NoSlvPorts: NumEndpoints,
      NoMstPorts: NumEndpoints,
      MaxMstTrans: 10,
      MaxSlvTrans: 6,
      FallThrough: 1'b0,
      LatencyMode: axi_pkg::CUT_ALL_AX,
      PipelineStages: TbPipeline,
      AxiIdWidthSlvPorts: TbIdWidthIn,
      AxiIdUsedSlvPorts: TbIdWidthIn,
      UniqueIds: 1'b0,
      AxiAddrWidth: AxiAddrWidth,
      AxiDataWidth: AxiWideDataWidth,
      NoAddrRules: NumEndpoints
  };

  localparam axi_pkg::xbar_cfg_t NarrowXbarCfg = '{
      NoSlvPorts: NumEndpoints,
      NoMstPorts: NumEndpoints,
      MaxMstTrans: 10,
      MaxSlvTrans: 6,
      FallThrough: 1'b0,
      LatencyMode: axi_pkg::CUT_ALL_AX,
      PipelineStages: TbPipeline,
      AxiIdWidthSlvPorts: TbIdWidthIn,
      AxiIdUsedSlvPorts: TbIdWidthIn,
      UniqueIds: 1'b0,
      AxiAddrWidth: AxiAddrWidth,
      AxiDataWidth: AxiNarrowDataWidth,
      NoAddrRules: NumEndpoints
  };

  logic clk;
  logic rst_n;

  axi_wide_mst_req_t    [NumEndpoints-1:0] wide_mst_req;
  axi_wide_mst_resp_t   [NumEndpoints-1:0] wide_mst_rsp;
  axi_wide_slv_req_t    [NumEndpoints-1:0] wide_slv_req;
  axi_wide_slv_resp_t   [NumEndpoints-1:0] wide_slv_rsp;
  axi_narrow_mst_req_t  [NumEndpoints-1:0] narrow_mst_req;
  axi_narrow_mst_resp_t [NumEndpoints-1:0] narrow_mst_rsp;
  axi_narrow_slv_req_t  [NumEndpoints-1:0] narrow_slv_req;
  axi_narrow_slv_resp_t [NumEndpoints-1:0] narrow_slv_rsp;

  axi_xbar #(
      .Cfg          (WideXbarCfg),
      .ATOPs        (0),
      .slv_aw_chan_t(axi_wide_mst_aw_chan_t),
      .mst_aw_chan_t(axi_wide_slv_aw_chan_t),
      .w_chan_t     (axi_wide_mst_w_chan_t),
      .slv_b_chan_t (axi_wide_mst_b_chan_t),
      .mst_b_chan_t (axi_wide_slv_b_chan_t),
      .slv_ar_chan_t(axi_wide_mst_ar_chan_t),
      .mst_ar_chan_t(axi_wide_slv_ar_chan_t),
      .slv_r_chan_t (axi_wide_mst_r_chan_t),
      .mst_r_chan_t (axi_wide_slv_r_chan_t),
      .slv_req_t    (axi_wide_mst_req_t),
      .slv_resp_t   (axi_wide_mst_resp_t),
      .mst_req_t    (axi_wide_slv_req_t),
      .mst_resp_t   (axi_wide_slv_resp_t),
      .rule_t       (tb_rule_t)
  ) i_wide_xbar (
      .clk_i                (clk),
      .rst_ni               (rst_n),
      .test_i               (1'b0),
      .slv_ports_req_i      (wide_mst_req),
      .slv_ports_resp_o     (wide_mst_rsp),
      .mst_ports_req_o      (wide_slv_req),
      .mst_ports_resp_i     (wide_slv_rsp),
      .addr_map_i           (XbarRule),
      .en_default_mst_port_i('0),
      .default_mst_port_i   ('0)
  );

  axi_xbar #(
      .Cfg          (NarrowXbarCfg),
      .ATOPs        (0),
      .slv_aw_chan_t(axi_narrow_mst_aw_chan_t),
      .mst_aw_chan_t(axi_narrow_slv_aw_chan_t),
      .w_chan_t     (axi_narrow_mst_w_chan_t),
      .slv_b_chan_t (axi_narrow_mst_b_chan_t),
      .mst_b_chan_t (axi_narrow_slv_b_chan_t),
      .slv_ar_chan_t(axi_narrow_mst_ar_chan_t),
      .mst_ar_chan_t(axi_narrow_slv_ar_chan_t),
      .slv_r_chan_t (axi_narrow_mst_r_chan_t),
      .mst_r_chan_t (axi_narrow_slv_r_chan_t),
      .slv_req_t    (axi_narrow_mst_req_t),
      .slv_resp_t   (axi_narrow_mst_resp_t),
      .mst_req_t    (axi_narrow_slv_req_t),
      .mst_resp_t   (axi_narrow_slv_resp_t),
      .rule_t       (tb_rule_t)
  ) i_narrow_xbar (
      .clk_i                (clk),
      .rst_ni               (rst_n),
      .test_i               (1'b0),
      .slv_ports_req_i      (narrow_mst_req),
      .slv_ports_resp_o     (narrow_mst_rsp),
      .mst_ports_req_o      (narrow_slv_req),
      .mst_ports_resp_i     (narrow_slv_rsp),
      .addr_map_i           (XbarRule),
      .en_default_mst_port_i('0),
      .default_mst_port_i   ('0)
  );

  clk_rst_gen #(
      .ClkPeriod   (CyclTime),
      .RstClkCycles(5)
  ) i_clk_gen (
      .clk_o (clk),
      .rst_no(rst_n)
  );

  // ==========================================================================================
  // Endpoints
  // ==========================================================================================
  tb_tcdm_req_t [NumEndpoints-1:0][TcdmNumPorts-1:0] tcdm_req;
  tb_tcdm_rsp_t [NumEndpoints-1:0][TcdmNumPorts-1:0] tcdm_rsp;

  logic [31:0] csr_addr      [NumEndpoints];
  logic [31:0] csr_wdata     [NumEndpoints];
  logic        csr_write     [NumEndpoints];
  logic        csr_valid     [NumEndpoints];
  logic        csr_req_ready [NumEndpoints];
  logic [31:0] csr_rdata     [NumEndpoints];
  logic        csr_rsp_valid [NumEndpoints];
  logic        csr_rsp_ready [NumEndpoints];

  for (genvar i = 0; i < NumEndpoints; i++) begin : gen_ep
    snax_xdma_cluster_xdma_wrapper #(
        .tcdm_req_t         (tb_tcdm_req_t),
        .tcdm_rsp_t         (tb_tcdm_rsp_t),
        .wide_slv_id_t      (id_slv_t),
        .wide_out_req_t     (axi_wide_mst_req_t),
        .wide_out_resp_t    (axi_wide_mst_resp_t),
        .wide_in_req_t      (axi_wide_slv_req_t),
        .wide_in_resp_t     (axi_wide_slv_resp_t),
        .narrow_slv_id_t    (id_slv_t),
        .narrow_out_req_t   (axi_narrow_mst_req_t),
        .narrow_out_resp_t  (axi_narrow_mst_resp_t),
        .narrow_in_req_t    (axi_narrow_slv_req_t),
        .narrow_in_resp_t   (axi_narrow_slv_resp_t),
        .TCDMDataWidth      (TcdmDataWidth),
        .TCDMNumPorts       (TcdmNumPorts),
        .TCDMAddrWidth      (TcdmAddrWidth),
        .ClusterBaseAddr    (ClusterBaseAddr),
        .ClusterAddressSpace(ClusterAddressSpace),
        .MainMemBaseAddr    (MainMemBaseAddr),
        .MainMemEndAddr     (MainMemEndAddr),
        .MMIOSize           (MMIOSize)
    ) i_ep (
        .clk_i                 (clk),
        .rst_ni                (rst_n),
        .cluster_base_addr_i   (cluster_base(i)),
        .tcdm_req_o            (tcdm_req[i]),
        .tcdm_rsp_i            (tcdm_rsp[i]),
        .csr_req_bits_addr_i   (csr_addr[i]),
        .csr_req_bits_write_i  (csr_write[i]),
        .csr_req_bits_data_i   (csr_wdata[i]),
        .csr_req_bits_strb_i   ('1),
        .csr_req_valid_i       (csr_valid[i]),
        .csr_req_ready_o       (csr_req_ready[i]),
        .csr_rsp_bits_data_o   (csr_rdata[i]),
        .csr_rsp_valid_o       (csr_rsp_valid[i]),
        .csr_rsp_ready_i       (csr_rsp_ready[i]),
        .xdma_wide_out_req_o   (wide_mst_req[i]),
        .xdma_wide_out_resp_i  (wide_mst_rsp[i]),
        .xdma_wide_in_req_i    (wide_slv_req[i]),
        .xdma_wide_in_resp_o   (wide_slv_rsp[i]),
        .xdma_narrow_out_req_o (narrow_mst_req[i]),
        .xdma_narrow_out_resp_i(narrow_mst_rsp[i]),
        .xdma_narrow_in_req_i  (narrow_slv_req[i]),
        .xdma_narrow_in_resp_o (narrow_slv_rsp[i])
    );
  end

  // ==========================================================================================
  // TCDM model: one behavioural array per endpoint, `q_ready` always high, one-cycle read
  // latency, no `p_ready` (a TCDM response cannot be backpressured).
  //
  // `q.addr` is a BYTE address (TCDMAddrWidth = clog2(TCDMSize in bytes)); the low 3 bits are
  // the byte offset within a 64-bit word. All 16 ports of one endpoint are served from a single
  // always_ff so two ports touching the same word in the same cycle is ordered, not a race.
  // ==========================================================================================
  logic [63:0] mem [NumEndpoints][TcdmWords];

  logic          [NumEndpoints-1:0][TcdmNumPorts-1:0] rsp_valid_q;
  tb_tcdm_data_t [NumEndpoints-1:0][TcdmNumPorts-1:0] rsp_data_q;

  for (genvar e = 0; e < NumEndpoints; e++) begin : gen_tcdm_rsp
    for (genvar p = 0; p < TcdmNumPorts; p++) begin : gen_port
      assign tcdm_rsp[e][p].q_ready = 1'b1;
      assign tcdm_rsp[e][p].p_valid = rsp_valid_q[e][p];
      assign tcdm_rsp[e][p].p.data  = rsp_data_q[e][p];
    end
  end

  for (genvar e = 0; e < NumEndpoints; e++) begin : gen_tcdm_mem
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        rsp_valid_q[e] <= '0;
      end else begin
        for (int unsigned p = 0; p < TcdmNumPorts; p++) begin
          rsp_valid_q[e][p] <= 1'b0;
          if (tcdm_req[e][p].q_valid) begin
            if (tcdm_req[e][p].q.write) begin
              for (int unsigned b = 0; b < TcdmDataWidth / 8; b++) begin
                if (tcdm_req[e][p].q.strb[b]) begin
                  mem[e][tcdm_req[e][p].q.addr>>3][b*8+:8] <= tcdm_req[e][p].q.data[b*8+:8];
                end
              end
            end else begin
              rsp_data_q[e][p]  <= mem[e][tcdm_req[e][p].q.addr>>3];
              rsp_valid_q[e][p] <= 1'b1;
            end
          end
        end
      end
    end
  end

  // ==========================================================================================
  // The "software": two CSR tasks. `csr_req_bits_addr_i` is a plain 0-based register index.
  // ==========================================================================================
  localparam int unsigned SpinLimit = 200000;

  int unsigned errors;
  string       fail_reason;

  task automatic fail(input string msg);
    begin
      errors++;
      fail_reason = msg;
      $error("[FAIL] %s", msg);
    end
  endtask

  task automatic csr_wr(input int unsigned ep, input int unsigned idx, input logic [31:0] d);
    int unsigned guard;
    begin
      @(negedge clk);
      csr_addr[ep]  = idx;
      csr_wdata[ep] = d;
      csr_write[ep] = 1'b1;
      csr_valid[ep] = 1'b1;
      guard         = 0;
      forever begin
        @(posedge clk);
        if (csr_req_ready[ep]) break;
        guard++;
        if (guard > SpinLimit) begin
          fail($sformatf("csr_wr(ep=%0d, idx=%0d) never accepted", ep, idx));
          break;
        end
      end
      @(negedge clk);
      csr_valid[ep] = 1'b0;
    end
  endtask

  // The read response is COMBINATIONAL: ReqRspManager drives `rsp.valid` and `rsp.bits.data`
  // in the very cycle the read request fires (`when(readReg) { ... }`), and only buffers them if
  // the receiver is not ready. With `rsp_ready` held high the beat is consumed immediately, so a
  // task that goes looking for the response on a LATER cycle waits forever.
  task automatic csr_rd(input int unsigned ep, input int unsigned idx, output logic [31:0] d);
    int unsigned guard;
    begin
      @(negedge clk);
      csr_addr[ep]  = idx;
      csr_wdata[ep] = '0;
      csr_write[ep] = 1'b0;
      csr_valid[ep] = 1'b1;
      d             = '0;
      guard         = 0;
      forever begin
        @(posedge clk);
        if (csr_req_ready[ep]) begin
          // same cycle as the request fires
          if (csr_rsp_valid[ep]) d = csr_rdata[ep];
          else fail($sformatf("csr_rd(ep=%0d, idx=%0d): request fired with no response", ep, idx));
          break;
        end
        guard++;
        if (guard > SpinLimit) begin
          fail($sformatf("csr_rd(ep=%0d, idx=%0d) request never accepted", ep, idx));
          break;
        end
      end
      @(negedge clk);
      csr_valid[ep] = 1'b0;
    end
  endtask

  task automatic csr_wr64(input int unsigned ep, input int unsigned idx, input tb_addr_t a);
    begin
      csr_wr(ep, idx, a[31:0]);
      csr_wr(ep, idx + 1, {16'h0, a[47:32]});
    end
  endtask

  // ==========================================================================================
  // Payload
  //
  // The collector is ALWAYS endpoint 0 and a width-P round folds endpoints 0..P-1, with data
  // flowing from the farthest source inward:
  //
  //     ep(P-1) -> ep(P-2) -> ... -> ep1 -> ep0
  //      HEAD       MIDDLE           MIDDLE  collector
  //
  // That mirrors the HeMAiA app, whose SNAKE[0] is the collector and whose chain is built from
  // snake index P-1 down to 1. Keeping the collector fixed is what makes a sweep a real stress:
  // every width in the sweep re-arms the SAME node, so a node that fails to retire poisons the
  // next width rather than hiding on an endpoint nobody revisits.
  // ==========================================================================================
  function automatic logic [31:0] f32(input real v);
    return $shortrealtobits(shortreal'(v));
  endfunction

  function automatic real r32(input logic [31:0] b);
    return real'($bitstoshortreal(b));
  endfunction

  // Linear operand: integer-valued and distinct per endpoint, so the FP32 sum is EXACT and the
  // check can be byte-exact. A repeating seed would let a stale-data bug match by accident.
  function automatic logic [31:0] lin_lane(input int unsigned ep, input int unsigned lane);
    return f32(real'((ep + 1) * 1000 + lane));
  endfunction

  // Moment operand: the online-softmax partial (m, l). The key spread is kept modest so every
  // exp argument stays well inside the writer LUT's useful range -- the point of this arm is the
  // TRANSPORT of a nonlinear fold, not a study of the LUT's tails.
  function automatic real mom_m(input int unsigned ep);
    return 1.0 + 0.125 * real'(ep);
  endfunction

  function automatic real mom_l(input int unsigned ep);
    return 1.0 + 0.125 * real'(ep);
  endfunction

  task automatic seed_partials();
    logic [63:0] w;
    int unsigned lane;
    begin
      for (int unsigned e = 0; e < NumEndpoints; e++) begin
        // linear: 16 lanes of a distinct ramp
        for (int unsigned bt = 0; bt < NumBeats; bt++) begin
          for (int unsigned wi = 0; wi < BeatBytes / 8; wi++) begin
            lane = bt * LanesPerBeat + wi * 2;
            w    = {lin_lane(e, lane + 1), lin_lane(e, lane)};
            mem[e][(LinSrcOffset + bt * BeatBytes + wi * 8)>>3] = w;
          end
        end
        // moment: only slot 0 is live (nValid = 1), so m at lane 0 and l at lane 8 carry the
        // partial and every other lane is fed its field's identity by the junction itself.
        for (int unsigned wi = 0; wi < BeatBytes / 8; wi++) begin
          mem[e][(MomSrcOffset + wi * 8)>>3] = 64'h0;
        end
        mem[e][(MomSrcOffset + (MomentMLane / 2) * 8)>>3][(MomentMLane % 2)*32+:32] =
            f32(mom_m(e));
        mem[e][(MomSrcOffset + (MomentLLane / 2) * 8)>>3][(MomentLLane % 2)*32+:32] =
            f32(mom_l(e));
      end
    end
  endtask

  task automatic fill_sentinel(input int unsigned mode);
    int unsigned off;
    begin
      off = (mode == ModeLin) ? LinDstOffset : MomDstOffset;
      for (int unsigned wi = 0; wi < XferBytes / 8; wi++) begin
        mem[0][(off + wi * 8)>>3] = {32'hDEAD_BEEF, 32'hDEAD_BEEF};
      end
    end
  endtask

  // Golden for the linear fold: the exact FP32 sum over the participating prefix.
  function automatic logic [31:0] golden_lin(input int unsigned width, input int unsigned lane);
    logic [31:0] acc;
    begin
      acc = lin_lane(0, lane);
      for (int unsigned e = 1; e < width; e++) begin
        acc = $shortrealtobits($bitstoshortreal(acc) + $bitstoshortreal(lin_lane(e, lane)));
      end
      return acc;
    end
  endfunction

  // Golden for the moment fold: (m1,l1) (+) (m2,l2) = (max, l1*exp(m1-m*) + l2*exp(m2-m*)),
  // which over a prefix collapses to m* = max(m_e) and l* = sum(l_e * exp(m_e - m*)).
  task automatic golden_mom(input int unsigned width, output logic [31:0] m_bits,
                            output logic [31:0] l_bits);
    real ms, ls;
    begin
      ms = mom_m(0);
      for (int unsigned e = 1; e < width; e++) if (mom_m(e) > ms) ms = mom_m(e);
      ls = 0.0;
      for (int unsigned e = 0; e < width; e++) ls = ls + mom_l(e) * $exp(mom_m(e) - ms);
      m_bits = f32(ms);
      l_bits = f32(ls);
    end
  endtask

  // ==========================================================================================
  // Issue one ChainGather from the collector, mirroring xdma_chain_gather_1d_full_address().
  // ==========================================================================================
  task automatic issue_gather(input int unsigned width, input int unsigned mode);
    tb_addr_t    dst_ptr;
    int unsigned src_off, dst_off;
    logic [31:0] jct_csr;
    int unsigned junction;
    begin
      src_off = (mode == ModeLin) ? LinSrcOffset : MomSrcOffset;
      dst_off = (mode == ModeLin) ? LinDstOffset : MomDstOffset;

      // 1. reader ptr: the collector's own partial
      csr_wr64(0, CSR_SRC_ADDR, cluster_base(0) + src_off);

      // 2. writer ptrs, in DATA order: the farthest source first, down to the nearest, ending at
      //    the collector's own destination. Slot k is that hop's own partial.
      for (int unsigned k = 0; k < width; k++) begin
        dst_ptr = (k == width - 1) ? (cluster_base(0) + dst_off)
                                   : (cluster_base(width - 1 - k) + src_off);
        csr_wr64(0, CSR_DST_ADDR + 2 * k, dst_ptr);
      end
      // Zero-terminate the remaining multicast slots.
      for (int unsigned k = width; k < 16; k++) csr_wr64(0, CSR_DST_ADDR + 2 * k, '0);

      // 3. strides and bounds. Unused temporal dims must be 1, NOT 0.
      csr_wr(0, CSR_SRC_SPAT_STRIDE, 32'd8);
      csr_wr(0, CSR_SRC_TEMP_BOUND + 0, NumBeats);
      for (int unsigned d = 1; d < 5; d++) csr_wr(0, CSR_SRC_TEMP_BOUND + d, 32'd1);
      csr_wr(0, CSR_SRC_TEMP_STRIDE + 0, 32'd64);
      for (int unsigned d = 1; d < 5; d++) csr_wr(0, CSR_SRC_TEMP_STRIDE + d, 32'd0);
      csr_wr(0, CSR_SRC_ENAB_CHAN, 32'hFFFF_FFFF);

      csr_wr(0, CSR_DST_SPAT_STRIDE, 32'd8);
      csr_wr(0, CSR_DST_TEMP_BOUND + 0, NumBeats);
      for (int unsigned d = 1; d < 5; d++) csr_wr(0, CSR_DST_TEMP_BOUND + d, 32'd1);
      csr_wr(0, CSR_DST_TEMP_STRIDE + 0, 32'd64);
      for (int unsigned d = 1; d < 5; d++) csr_wr(0, CSR_DST_TEMP_STRIDE + d, 32'd0);
      csr_wr(0, CSR_DST_ENAB_CHAN, 32'hFFFF_FFFF);
      // enabledByte is tcdmParam.dataWidth/8 = 8 bits wide, not the 64 B beat.
      csr_wr(0, CSR_DST_ENAB_BYTE, 32'h0000_00FF);

      // 4. a CLEAN plugin config, exactly as the app does before every round: no reader or
      //    writer extensions, and no junction left armed from the previous round.
      csr_wr(0, CSR_SRC_ENABLE, 32'd0);
      csr_wr(0, CSR_DST_ENABLE, 32'd0);
      csr_wr(0, CSR_DST_EXT, 32'd0);
      csr_wr(0, CSR_DST_JCT_ENABLE, 32'd0);

      // 5. arm this round's junction: ENABLE BITMASK FIRST, then its user CSR.
      if (mode == ModeLin) begin
        junction = JCT_ELEMENTWISE;
        jct_csr  = JCT_CSR_LINEAR;
      end else begin
        junction = JCT_MONOID;
        jct_csr  = JCT_CSR_MONOID;
      end
      csr_wr(0, CSR_DST_JCT_ENABLE, 32'd1 << junction);
      csr_wr(0, CSR_DST_JCT_CSR + junction, jct_csr);
    end
  endtask

  // ==========================================================================================
  // Start + wait, reproducing xdma_start(): read both commit counters, launch, see which one
  // moved, then poll the MATCHING finish counter. Bounded, always.
  // ==========================================================================================
  logic [31:0] cyc_cnt;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) cyc_cnt <= '0;
    else cyc_cnt <= cyc_cnt + 1;
  end

  task automatic start_and_wait(input int unsigned round, output logic [31:0] task_cycles,
                                output logic [31:0] wall_cycles, output bit ok);
    logic [31:0] cl0, cr0, cl1, cr1, fl0, fr0, fin, jct, t0;
    bit          is_local;
    logic [31:0] want;
    int unsigned guard;
    begin
      ok = 1'b1;
      csr_rd(0, CSR_COMMIT_LOCAL, cl0);
      csr_rd(0, CSR_COMMIT_REMOTE, cr0);
      csr_rd(0, CSR_FINISH_LOCAL, fl0);
      csr_rd(0, CSR_FINISH_REMOTE, fr0);
      if (Verbose >= 1 || fr0 !== cr0 || fl0 !== cl0)
        $display("   [ctr] before round %0d: commit(l=%0d,r=%0d) finish(l=%0d,r=%0d)%s", round,
                 cl0, cr0, fl0, fr0, (fr0 !== cr0 || fl0 !== cl0) ? "   <-- MISMATCH" : "");
      // A finish counter ahead of its commit counter is a BOGUS COMPLETION already banked. It is
      // visible a whole round before the data check fails, so make it an error in its own right.
      if (fr0 !== cr0 || fl0 !== cl0) begin
        fail($sformatf(
             "round %0d: a completion was banked for a transfer that never happened -- commit(l=%0d,r=%0d) but finish(l=%0d,r=%0d)",
             round, cl0, cr0, fl0, fr0));
        ok = 1'b0;
      end

      t0 = cyc_cnt;
      csr_wr(0, CSR_START, 32'd1);

      guard = 0;
      forever begin
        csr_rd(0, CSR_COMMIT_LOCAL, cl1);
        csr_rd(0, CSR_COMMIT_REMOTE, cr1);
        if (cl1 !== cl0 || cr1 !== cr0) break;
        guard++;
        if (guard > 2000) begin
          fail($sformatf("round %0d: task never committed (local %0d, remote %0d)", round, cl1,
                         cr1));
          task_cycles = '0;
          wall_cycles = cyc_cnt - t0;
          ok          = 1'b0;
          return;
        end
      end
      is_local = (cl1 !== cl0);
      want     = is_local ? cl1 : cr1;

      // If the counter is ALREADY at the target when the round starts, this round's wait is
      // vacuous: it returns before any data has moved. Say so loudly rather than passing it off
      // as a fast completion.
      csr_rd(0, is_local ? CSR_FINISH_LOCAL : CSR_FINISH_REMOTE, fin);
      if (fin === want) begin
        $display(
            "   [ctr] round %0d: %s finish counter was ALREADY %0d at the start -- the wait is vacuous",
            round, is_local ? "local" : "remote", want);
      end

      guard = 0;
      forever begin
        csr_rd(0, is_local ? CSR_FINISH_LOCAL : CSR_FINISH_REMOTE, fin);
        if (fin === want) break;
        guard++;
        if (guard > 20000) begin
          csr_rd(0, CSR_JCT_STATUS, jct);
          fail($sformatf(
               "round %0d: %s finish counter stuck at %0d, want %0d (JCT_STATUS=0x%08x)", round,
               is_local ? "local" : "remote", fin, want, jct));
          ok = 1'b0;
          break;
        end
      end

      wall_cycles = cyc_cnt - t0;
      csr_rd(0, CSR_PERF_TASK, task_cycles);
    end
  endtask

  // ==========================================================================================
  // Checks
  // ==========================================================================================
  task automatic check_result(input int unsigned round, input int unsigned width,
                              input int unsigned mode, output bit ok);
    logic [63:0] w;
    logic [31:0] got, exp, m_got, m_exp, l_got, l_exp, l_ulp;
    int unsigned lane, bad, off;
    bit          sentinel_left;
    begin
      ok            = 1'b1;
      bad           = 0;
      sentinel_left = 1'b0;
      off           = (mode == ModeLin) ? LinDstOffset : MomDstOffset;

      for (int unsigned wi = 0; wi < XferBytes / 8; wi++) begin
        if (mem[0][(off + wi * 8)>>3] === {32'hDEAD_BEEF, 32'hDEAD_BEEF}) sentinel_left = 1'b1;
      end
      if (sentinel_left) begin
        fail($sformatf(
             "round %0d (P=%0d %s): the 0xDEADBEEF sentinel is still in the collector's buffer -- the gather moved NOTHING",
             round, width, (mode == ModeLin) ? "lin" : "mom"));
        ok = 1'b0;
        return;
      end

      if (mode == ModeLin) begin
        // Small integer-valued FP32 operands -> the sum is exact -> byte-exact compare.
        for (int unsigned wi = 0; wi < XferBytes / 8; wi++) begin
          w = mem[0][(off + wi * 8)>>3];
          for (int unsigned h = 0; h < 2; h++) begin
            lane = wi * 2 + h;
            got  = w[h*32+:32];
            exp  = golden_lin(width, lane % LanesPerBeat);
            if (got !== exp) begin
              if (bad < 4) begin
                $display("      lane %0d: got 0x%08x (%f) want 0x%08x (%f)", lane, got, r32(got),
                         exp, r32(exp));
              end
              bad++;
            end
          end
        end
        if (bad != 0) begin
          fail($sformatf("round %0d (P=%0d lin): %0d/%0d lanes wrong", round, width, bad,
                         XferBytes / 4));
          ok = 1'b0;
        end
      end else begin
        // m* is a max, so exact; l* goes through the writer's exp LUT, so ULP-bounded.
        m_got = mem[0][(off + (MomentMLane / 2) * 8)>>3][(MomentMLane % 2)*32+:32];
        l_got = mem[0][(off + (MomentLLane / 2) * 8)>>3][(MomentLLane % 2)*32+:32];
        golden_mom(width, m_exp, l_exp);
        l_ulp = (l_got > l_exp) ? (l_got - l_exp) : (l_exp - l_got);
        $display("   [mom] P=%0d m*: got 0x%08x (%f) exp 0x%08x (%f) | l*: got 0x%08x (%f) exp 0x%08x (%f) ulp=0x%0x tol=0x%0x",
                 width, m_got, r32(m_got), m_exp, r32(m_exp), l_got, r32(l_got), l_exp,
                 r32(l_exp), l_ulp, LTolUlps);
        if (m_got !== m_exp) begin
          fail($sformatf("round %0d (P=%0d mom): m* mismatch, got 0x%08x want 0x%08x", round,
                         width, m_got, m_exp));
          ok = 1'b0;
        end
        if (l_ulp > LTolUlps) begin
          fail($sformatf("round %0d (P=%0d mom): l* out of tolerance, %0d ULP > %0d", round, width,
                         l_ulp, LTolUlps));
          ok = 1'b0;
        end
      end
    end
  endtask

  // The busy levels are `snax_xdma_cluster_xdma`'s `io_status_*` ports and the adapter's
  // watchdog is `unused_xdma_stall_error`; the wrapper leaves both unconnected, but they still
  // exist on the instances. Probe them into flat arrays through a generate loop, because a
  // hierarchical path cannot be indexed by a runtime variable inside a task.
  logic [NumEndpoints-1:0] ep_writer_busy;
  logic [NumEndpoints-1:0] ep_reader_busy;
  logic [NumEndpoints-1:0] ep_stall_error;

  // `xdma_finish` is the adapter's completion pulse, and the Chisel side counts it straight into
  // XDMA_FINISH_REMOTE. Counting the pulses per endpoint is what separates "the task never
  // finished" from "the task finished TWICE and banked a credit the next round then spends".
  logic [31:0] ep_finish_count[NumEndpoints];

  for (genvar e = 0; e < NumEndpoints; e++) begin : gen_probe
    assign ep_writer_busy[e] = gen_ep[e].i_ep.i_snax_xdma_cluster_xdma.io_status_writerBusy;
    assign ep_reader_busy[e] = gen_ep[e].i_ep.i_snax_xdma_cluster_xdma.io_status_readerBusy;
    assign ep_stall_error[e] = gen_ep[e].i_ep.unused_xdma_stall_error;

    // Trace every EDGE of the composed busy level and its three components. The finish manager's
    // tail FSM arms on `ready_to_transfer` rising and completes on it falling, so a level that
    // pulses TWICE inside one task produces TWO tail finishes -- which is exactly the credit a
    // later round then spends without moving any data.
    logic bw_q;
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) bw_q <= 1'b0;
      else begin
        bw_q <= gen_ep[e].i_ep.i_snax_xdma_cluster_xdma.xdmaDatapath.dataSwitch.io_writerBusy_0;
        if (Verbose >= 2 &&
            gen_ep[e].i_ep.i_snax_xdma_cluster_xdma.xdmaDatapath.dataSwitch.io_writerBusy_0
            !== bw_q) begin
          $display(
              "   [bsy] t=%0t ep%0d writerBusy %0b->%0b   raw=%0b chained=%0b gather=%0b pending=%0b jctActive=%0b rdBusy=%0b",
              $time, e, bw_q,
              gen_ep[e].i_ep.i_snax_xdma_cluster_xdma.xdmaDatapath.dataSwitch.io_writerBusy_0,
              gen_ep[e].i_ep.i_snax_xdma_cluster_xdma.xdmaDatapath.dataSwitch.io_writerBusyRaw,
              gen_ep[e].i_ep.i_snax_xdma_cluster_xdma.xdmaDatapath.dataSwitch.isChainedWrite,
              gen_ep[e].i_ep.i_snax_xdma_cluster_xdma.xdmaDatapath.dataSwitch.isGather,
              gen_ep[e].i_ep.i_snax_xdma_cluster_xdma.xdmaDatapath.dataSwitch.gatherPending,
              gen_ep[e].i_ep.i_snax_xdma_cluster_xdma.xdmaDatapath.dataSwitch._junctionHost_io_active,
              gen_ep[e].i_ep.i_snax_xdma_cluster_xdma.xdmaDatapath.dataSwitch.io_readerBusy);
        end
      end
    end

    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) ep_finish_count[e] <= '0;
      else if (gen_ep[e].i_ep.xdma_finish) begin
        ep_finish_count[e] <= ep_finish_count[e] + 1;
        if (Verbose >= 1)
          $display("   [fin] t=%0t endpoint %0d raised xdma_finish (count %0d)", $time, e,
                   ep_finish_count[e] + 1);
      end
    end
  end

  // Every endpoint must be back at rest before the next round. A parked FSM IS the failure
  // signature, and it is invisible until the round after -- so check it explicitly rather than
  // inferring it from a passing round.
  task automatic check_idle(input int unsigned round, input int unsigned width, output bit ok);
    logic [31:0] jct;
    begin
      ok = 1'b1;
      for (int unsigned e = 0; e < width; e++) begin
        if (ep_writer_busy[e] !== 1'b0) begin
          fail($sformatf(
               "round %0d: endpoint %0d still reports writerBusy after the round -- this hop can never be armed again",
               round, e));
          ok = 1'b0;
        end
        if (ep_reader_busy[e] !== 1'b0) begin
          fail($sformatf("round %0d: endpoint %0d still reports readerBusy after the round", round,
                         e));
          ok = 1'b0;
        end
        if (ep_stall_error[e] !== 1'b0) begin
          fail($sformatf("round %0d: endpoint %0d raised xdma_stall_error", round, e));
          ok = 1'b0;
        end
      end
      csr_rd(0, CSR_JCT_STATUS, jct);
      // [0] cfgerr, [1] starved. A zero here is NOT evidence of health on a failing round: both
      // sticky bits are cleared by the writer-start pulse even a dead round issues.
      if (jct[0]) begin
        fail($sformatf("round %0d: junction reports a configuration error", round));
        ok = 1'b0;
      end
      if (jct[1]) begin
        fail($sformatf("round %0d: junction reports starvation", round));
        ok = 1'b0;
      end
    end
  endtask

  // ==========================================================================================
  // One measured configuration: NumRounds identical gathers at (width, mode).
  // ==========================================================================================
  task automatic run_config(input int unsigned width, input int unsigned mode,
                            output logic [31:0] task_cycles, output logic [31:0] wall_cycles,
                            output int unsigned cfg_err);
    logic [31:0] tc, wc, tc_first;
    bit          ok_start, ok_data, ok_idle;
    begin
      cfg_err     = 0;
      task_cycles = '0;
      wall_cycles = '0;
      tc_first    = '0;

      for (int unsigned r = 1; r <= NumRounds; r++) begin
        fill_sentinel(mode);
        repeat (5) @(posedge clk);

        if (Verbose >= 1)
          $display("   [rnd] t=%0t ---- P=%0d %s round %0d ----", $time, width,
                   (mode == ModeLin) ? "lin" : "mom", r);

        issue_gather(width, mode);
        start_and_wait(r, tc, wc, ok_start);
        // The finish counter is not a data-visibility barrier; let the last beats land.
        repeat (50) @(posedge clk);

        check_result(r, width, mode, ok_data);
        check_idle(r, width, ok_idle);

        if (r == 1) begin
          tc_first    = tc;
          task_cycles = tc;
          wall_cycles = wc;
        end else if (tc !== tc_first) begin
          // Not fatal, but worth seeing: a re-armed chain that takes a different number of
          // cycles is doing something different on the second pass.
          $display("   [rnd] P=%0d %s round %0d took %0d task cycles, round 1 took %0d", width,
                   (mode == ModeLin) ? "lin" : "mom", r, tc, tc_first);
        end

        if (!ok_start || !ok_data || !ok_idle) begin
          cfg_err++;
          break;  // a broken configuration poisons its own later rounds; move on
        end
        repeat (100) @(posedge clk);
      end
    end
  endtask

  // ==========================================================================================
  // Stimulus
  // ==========================================================================================
  logic [31:0] res_task[2][NumSweepP];
  logic [31:0] res_wall[2][NumSweepP];
  int unsigned res_err [2][NumSweepP];
  bit          res_run [2][NumSweepP];

  initial begin : p_stim
    logic [31:0] tc, wc;
    int unsigned ce;
    int unsigned width;

    errors      = 0;
    fail_reason = "";
    for (int unsigned i = 0; i < NumEndpoints; i++) begin
      csr_addr[i]      = '0;
      csr_wdata[i]     = '0;
      csr_write[i]     = 1'b0;
      csr_valid[i]     = 1'b0;
      csr_rsp_ready[i] = 1'b1;
    end
    for (int unsigned e = 0; e < NumEndpoints; e++) begin
      for (int unsigned w = 0; w < TcdmWords; w++) mem[e][w] = '0;
    end
    for (int unsigned m = 0; m < 2; m++) begin
      for (int unsigned pi = 0; pi < NumSweepP; pi++) begin
        res_task[m][pi] = '0;
        res_wall[m][pi] = '0;
        res_err[m][pi]  = 0;
        res_run[m][pi]  = 1'b0;
      end
    end

    @(posedge rst_n);
    repeat (20) @(posedge clk);
    seed_partials();

    $display("");
    $display("================================================================");
    if (Sweep) begin
      $display(" ChainGather SCALING SWEEP: %0d endpoints, %0d rounds per configuration",
               NumEndpoints, NumRounds);
      $display("   collector = ep0; a width-P round folds ep0..ep%s, %s middle hop(s)",
               "(P-1)", "P-2");
    end else begin
      $display(" ChainGather: %0d endpoints, P=%0d, fold=%s, %0d rounds", NumEndpoints,
               ChainWidth, (JunctionId == 0) ? "linear" : "monoid", NumRounds);
      $display("   chain: ep%0d (HEAD) -> ... -> ep0 (collector), %0d middle hop(s)",
               ChainWidth - 1, (ChainWidth >= 2) ? ChainWidth - 2 : 0);
    end
    $display("================================================================");

    if (Sweep) begin
      for (int unsigned mode = 0; mode < 2; mode++) begin
        for (int unsigned pi = 0; pi < NumSweepP; pi++) begin
          width = SweepP[pi];
          if (width > NumEndpoints) begin
            $display(" P=%0d %s: SKIPPED (needs %0d endpoints, this build has %0d)", width,
                     (mode == ModeLin) ? "lin" : "mom", width, NumEndpoints);
            continue;
          end
          run_config(width, mode, tc, wc, ce);
          res_task[mode][pi] = tc;
          res_wall[mode][pi] = wc;
          res_err[mode][pi]  = ce;
          res_run[mode][pi]  = 1'b1;
          $display(" P=%2d %s: %0d task cycles, %0d wall cycles  %s", width,
                   (mode == ModeLin) ? "lin" : "mom", tc, wc, (ce == 0) ? "PASS" : "FAIL");
        end
      end
    end else begin
      run_config(ChainWidth, JunctionId, tc, wc, ce);
      $display(" P=%0d %s: %0d task cycles, %0d wall cycles  %s", ChainWidth,
               (JunctionId == 0) ? "lin" : "mom", tc, wc, (ce == 0) ? "PASS" : "FAIL");
    end

    if (Sweep) begin
      $display("");
      $display("[Sweep] ChainGather scaling on %0d endpoints -- xDMA task cycles", NumEndpoints);
      $display("[Sweep]   P  fold  task_cc  wall_cc  result");
      for (int unsigned mode = 0; mode < 2; mode++) begin
        for (int unsigned pi = 0; pi < NumSweepP; pi++) begin
          if (!res_run[mode][pi]) continue;
          $display("[Sweep] %3d  %s  %7d  %7d  %s", SweepP[pi], (mode == ModeLin) ? "lin" : "mom",
                   res_task[mode][pi], res_wall[mode][pi],
                   (res_err[mode][pi] == 0) ? "PASS" : "FAIL");
        end
      end
    end

    $display("================================================================");
    if (errors == 0) $display(" RESULT: PASS");
    else $display(" RESULT: FAIL  (%0d error(s))", errors);
    $display("================================================================");
    $display("");
    $finish;
  end

  initial begin : p_timeout
    #(SimTimeout);
    $display("");
    $error("[FAIL] global simulation timeout -- something wedged and no bounded spin caught it");
    $finish;
  end

endmodule
