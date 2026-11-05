// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

<%
  # An entry with snax_streamer_sets: one streamer per set, the shell's CSRs and the sets'
  # streamers behind snax_csr_range_demux (see the sets branch below)
  sets = "snax_streamer_sets" in cfg

  if not sets:
      num_tcdm_reader = 0
      num_tcdm_writer = 0
      num_tcdm_reader_writer = 0

      num_reader_offset = 0
      num_writer_offset = 0

      if("data_reader_params" in cfg["snax_streamer_cfg"]):
          num_tcdm_reader = sum(cfg["snax_streamer_cfg"]["data_reader_params"]["num_channel"])
          num_reader_offset = len(cfg["snax_streamer_cfg"]["data_reader_params"]["num_channel"])

      if("data_writer_params" in cfg["snax_streamer_cfg"]):
          num_tcdm_writer = sum(cfg["snax_streamer_cfg"]["data_writer_params"]["num_channel"])
          num_writer_offset = len(cfg["snax_streamer_cfg"]["data_writer_params"]["num_channel"])

      if("data_reader_writer_params" in cfg["snax_streamer_cfg"]):
          num_tcdm_reader_writer = sum(cfg["snax_streamer_cfg"]["data_reader_writer_params"]["num_channel"])

      num_tcdm_ports = num_tcdm_reader + num_tcdm_writer + int(num_tcdm_reader_writer / 2)

  else:
    # Per set: its readers and writers (local index in its streamer wrapper, global index on
    # the shell, width in TCDM words), its TCDM slice, and its CSR ranges. Global stream
    # indices run across the sets in set order, readers and writers each on their own.
    set_units = []
    tcdm_offset = 0
    reader_offset = 0
    writer_offset = 0
    for k, streamer_set in enumerate(cfg["snax_streamer_sets"]):
      scfg = streamer_set["snax_streamer_cfg"]
      rd = list(scfg.get("data_reader_params", {}).get("num_channel", []))
      wr = list(scfg.get("data_writer_params", {}).get("num_channel", []))
      rw = list(scfg.get("data_reader_writer_params", {}).get("num_channel", []))
      # A reader-writer pair (even index) is one reader and one writer, after the others
      rw_pairs = [rw[i] for i in range(0, len(rw), 2)]
      readers = [(i, reader_offset + i, dw) for i, dw in enumerate(rd + rw_pairs)]
      writers = [(i, writer_offset + i, dw) for i, dw in enumerate(wr + rw_pairs)]
      tcdm_num = sum(rd) + sum(wr) + int(sum(rw) / 2)
      set_units.append({
        "name": streamer_set["name"],
        "tag": cfg["tag_name"] + "_" + streamer_set["name"],
        "readers": readers,
        "writers": writers,
        "tcdm_offset": tcdm_offset,
        "tcdm_num": tcdm_num,
      })
      tcdm_offset += tcdm_num
      reader_offset += len(readers)
      writer_offset += len(writers)
    num_tcdm_ports = tcdm_offset

    # CSR ranges from csr_range_table: set k's streamer -> target k, set k's own CSRs -> the
    # shell (last target) at its shell offset. The table counts from CSR 960; the wrapper's
    # addresses count from 0 (the translator has subtracted 0x3C0).
    num_sets = len(set_units)
    csr_ranges = []
    for k, row in enumerate(cfg["csr_range_table"]):
      csr_ranges.append((row["streamer_base"] - 960, row["streamer_csr_num"], k, 0))
      if row["csr_num"] > 0:
        csr_ranges.append((row["csr_base"] - 960, row["csr_num"], num_sets, row["shell_offset"]))

    def packed(values):
      return "{" + ", ".join("32'd" + str(v) for v in reversed(values)) + "}"
%>

//-------------------------------
// Streamer-MUL wrapper
// This is the entire accelerator
// That connecst to the TCDM subsystem
//-------------------------------
module ${cfg["tag_name"]}_wrapper # (
  // TCDM typedefs
  parameter type         tcdm_req_t         = logic,
  parameter type         tcdm_rsp_t         = logic,
  // Reconfigurable parameters
  parameter int unsigned DataWidth          = ${cfg["tcdm_data_width"]},
  parameter int unsigned SnaxTcdmPorts      = ${num_tcdm_ports},
  // Addr width is pre-computed in the generator
  // TCDMAddrWidth = log2(TCDMBankNum * TCDMDepth * (TCDMDataWidth/8))
  parameter int unsigned TCDMAddrWidth      = ${cfg["tcdm_addr_width"]},
  // Don't touch parameters (or modify at your own risk)
  parameter int unsigned RegDataWidth       = 32,
  parameter int unsigned RegAddrWidth       = 32
)(
  //-----------------------------
  // Clocks and reset
  //-----------------------------
  input  logic clk_i,
  input  logic rst_ni,
  //-----------------------------
  // CSR control ports
  //-----------------------------
  // Request
  input  logic [31:0] snax_req_data_i,
  input  logic [31:0] snax_req_addr_i,
  input  logic        snax_req_write_i,
  input  logic        snax_req_valid_i,
  output logic        snax_req_ready_o,
  // Response
  input  logic        snax_rsp_ready_i,
  output logic        snax_rsp_valid_o,
  output logic [31:0] snax_rsp_data_o,
  //-----------------------------
  // Barrier
  //-----------------------------
  output logic        snax_barrier_o,
  //-----------------------------
  // TCDM ports
  //-----------------------------
  output tcdm_req_t [SnaxTcdmPorts-1:0] snax_tcdm_req_o,
  input  tcdm_rsp_t [SnaxTcdmPorts-1:0] snax_tcdm_rsp_i
);

% if sets:
  //-----------------------------
  // Accelerator ports, numbered across the streamer sets
  //-----------------------------
% for u in set_units:
  // Set ${u["name"]}
% for (li, gi, dw) in u["writers"]:
  logic [${dw*cfg["tcdm_data_width"]-1}:0] acc2stream_${gi}_data;
  logic acc2stream_${gi}_valid;
  logic acc2stream_${gi}_ready;

% endfor
% for (li, gi, dw) in u["readers"]:
  logic [${dw*cfg["tcdm_data_width"]-1}:0] stream2acc_${gi}_data;
  logic stream2acc_${gi}_valid;
  logic stream2acc_${gi}_ready;

% endfor
% endfor
  //-----------------------------
  // CSR routing: targets 0..${num_sets-1} are the sets' streamers, ${num_sets} the shell
  //-----------------------------
  localparam int unsigned NumCsrTargets = ${num_sets + 1};

  logic [NumCsrTargets-1:0][RegAddrWidth-1:0] acc_csr_req_addr;
  logic [NumCsrTargets-1:0][RegDataWidth-1:0] acc_csr_req_data;
  logic [NumCsrTargets-1:0]                   acc_csr_req_wen;
  logic [NumCsrTargets-1:0]                   acc_csr_req_valid;
  logic [NumCsrTargets-1:0]                   acc_csr_req_ready;
  logic [NumCsrTargets-1:0][RegDataWidth-1:0] acc_csr_rsp_data;
  logic [NumCsrTargets-1:0]                   acc_csr_rsp_valid;
  logic [NumCsrTargets-1:0]                   acc_csr_rsp_ready;

  snax_csr_range_demux #(
    .NumTargets           ( NumCsrTargets ),
    .NumRanges            ( ${len(csr_ranges)} ),
    .RegDataWidth         ( RegDataWidth  ),
    .RegAddrWidth         ( RegAddrWidth  ),
    // csr_range_table's bases minus 960: the translator has already subtracted 0x3C0
    .RangeBase            ( ${packed([r[0] for r in csr_ranges])} ),
    .RangeSize            ( ${packed([r[1] for r in csr_ranges])} ),
    .RangeTarget          ( ${packed([r[2] for r in csr_ranges])} ),
    .RangeTargetOffset    ( ${packed([r[3] for r in csr_ranges])} )
  ) i_snax_csr_range_demux (
    //-------------------------------
    // Clocks and reset
    //-------------------------------
    .clk_i                ( clk_i             ),
    .rst_ni               ( rst_ni            ),
    //-------------------------------
    // Input Core
    //-------------------------------
    .csr_req_addr_i       ( snax_req_addr_i   ),
    .csr_req_data_i       ( snax_req_data_i   ),
    .csr_req_wen_i        ( snax_req_write_i  ),
    .csr_req_valid_i      ( snax_req_valid_i  ),
    .csr_req_ready_o      ( snax_req_ready_o  ),
    .csr_rsp_data_o       ( snax_rsp_data_o   ),
    .csr_rsp_valid_o      ( snax_rsp_valid_o  ),
    .csr_rsp_ready_i      ( snax_rsp_ready_i  ),
    //-------------------------------
    // Output Port
    //-------------------------------
    .acc_csr_req_addr_o   ( acc_csr_req_addr  ),
    .acc_csr_req_data_o   ( acc_csr_req_data  ),
    .acc_csr_req_wen_o    ( acc_csr_req_wen   ),
    .acc_csr_req_valid_o  ( acc_csr_req_valid ),
    .acc_csr_req_ready_i  ( acc_csr_req_ready ),
    .acc_csr_rsp_data_i   ( acc_csr_rsp_data  ),
    .acc_csr_rsp_valid_i  ( acc_csr_rsp_valid ),
    .acc_csr_rsp_ready_o  ( acc_csr_rsp_ready )
  );

  //-----------------------------
  // Accelerator: one shell for all sets, its CSRs as the last target
  //-----------------------------
  ${cfg["tag_name"]}_shell_wrapper i_${cfg["tag_name"]}_shell_wrapper (
    //-------------------------------
    // Clocks and reset
    //-------------------------------
    .clk_i                ( clk_i                 ),
    .rst_ni               ( rst_ni                ),

    //-----------------------------
    // Accelerator ports, set after set
    //-----------------------------
% for u in set_units:
    // Set ${u["name"]}
% for (li, gi, dw) in u["writers"]:
    .acc2stream_${gi}_data_o  ( acc2stream_${gi}_data  ),
    .acc2stream_${gi}_valid_o ( acc2stream_${gi}_valid ),
    .acc2stream_${gi}_ready_i ( acc2stream_${gi}_ready ),
% endfor
% for (li, gi, dw) in u["readers"]:
    .stream2acc_${gi}_data_i  ( stream2acc_${gi}_data  ),
    .stream2acc_${gi}_valid_i ( stream2acc_${gi}_valid ),
    .stream2acc_${gi}_ready_o ( stream2acc_${gi}_ready ),
% endfor
% endfor

    //-----------------------------
    // CSR control ports
    //-----------------------------
    // Request
    .csr_req_addr_i       ( acc_csr_req_addr [${num_sets}]  ),
    .csr_req_data_i       ( acc_csr_req_data [${num_sets}]  ),
    .csr_req_write_i      ( acc_csr_req_wen  [${num_sets}]  ),
    .csr_req_valid_i      ( acc_csr_req_valid[${num_sets}]  ),
    .csr_req_ready_o      ( acc_csr_req_ready[${num_sets}]  ),
    // Response
    .csr_rsp_data_o       ( acc_csr_rsp_data [${num_sets}]  ),
    .csr_rsp_valid_o      ( acc_csr_rsp_valid[${num_sets}]  ),
    .csr_rsp_ready_i      ( acc_csr_rsp_ready[${num_sets}]  )
  );

% for k, u in enumerate(set_units):
  //-----------------------------
  // Streamer wrapper of set ${u["name"]}: CSR target ${k},
  // TCDM ports ${u["tcdm_offset"]}..${u["tcdm_offset"] + u["tcdm_num"] - 1}
  //-----------------------------
  ${u["tag"]}_streamer_wrapper #(
    .tcdm_req_t               ( tcdm_req_t    ),
    .tcdm_rsp_t               ( tcdm_rsp_t    ),
    .TCDMDataWidth            ( DataWidth     ),
    .TCDMNumPorts             ( ${u["tcdm_num"]} ),
    .TCDMAddrWidth            ( TCDMAddrWidth )
  ) i_${u["tag"]}_streamer_wrapper (
    //-----------------------------
    // Clocks and reset
    //-----------------------------
    .clk_i                    ( clk_i  ),
    .rst_ni                   ( rst_ni ),
    //-----------------------------
    // Accelerator ports (the set's own numbering on the left)
    //-----------------------------
% for (li, gi, dw) in u["writers"]:
    .acc2stream_${li}_data_i  ( acc2stream_${gi}_data  ),
    .acc2stream_${li}_valid_i ( acc2stream_${gi}_valid ),
    .acc2stream_${li}_ready_o ( acc2stream_${gi}_ready ),
% endfor
% for (li, gi, dw) in u["readers"]:
    .stream2acc_${li}_data_o  ( stream2acc_${gi}_data  ),
    .stream2acc_${li}_valid_o ( stream2acc_${gi}_valid ),
    .stream2acc_${li}_ready_i ( stream2acc_${gi}_ready ),
% endfor
    //-----------------------------
    // TCDM ports
    //-----------------------------
    .tcdm_req_o ( snax_tcdm_req_o[${u["tcdm_offset"] + u["tcdm_num"] - 1}:${u["tcdm_offset"]}] ),
    .tcdm_rsp_i ( snax_tcdm_rsp_i[${u["tcdm_offset"] + u["tcdm_num"] - 1}:${u["tcdm_offset"]}] ),

    //-----------------------------
    // CSR control ports
    //-----------------------------
    // Request
    .csr_req_bits_data_i   ( acc_csr_req_data [${k}] ),
    .csr_req_bits_addr_i   ( acc_csr_req_addr [${k}] ),
    .csr_req_bits_write_i  ( acc_csr_req_wen  [${k}] ),
    .csr_req_valid_i       ( acc_csr_req_valid[${k}] ),
    .csr_req_ready_o       ( acc_csr_req_ready[${k}] ),
    // Response
    .csr_rsp_bits_data_o   ( acc_csr_rsp_data [${k}] ),
    .csr_rsp_valid_o       ( acc_csr_rsp_valid[${k}] ),
    .csr_rsp_ready_i       ( acc_csr_rsp_ready[${k}] )
  );

% endfor
  //-----------------------------
  // Barrier control
  // TODO: Fix me later
  //-----------------------------
  assign snax_barrier_o = '0;

% else:
% if not cfg.get("snax_disable_csr_manager", False):
  //-----------------------------
  // Internal local parameters
  //-----------------------------
  localparam int unsigned NumRwCsr = ${cfg["snax_num_rw_csr"]};
  localparam int unsigned NumRoCsr = ${cfg["snax_num_ro_csr"]};
% endif

  //-----------------------------
  // Accelerator ports
  //-----------------------------
  // Note that these have very specific data widths
  // found from the configuration file

% if "data_writer_params" in cfg["snax_streamer_cfg"]:
  // Ports from accelerator to streamer by writer data movers
% for idx, dw in enumerate(cfg["snax_streamer_cfg"]["data_writer_params"]["num_channel"]):
  logic [${dw*cfg["tcdm_data_width"]-1}:0] acc2stream_${idx}_data;
  logic acc2stream_${idx}_valid;
  logic acc2stream_${idx}_ready;

% endfor
% endif
% if "data_reader_writer_params" in cfg["snax_streamer_cfg"]:
  // Ports from accelerator to streamer by reader-writer data movers
% for idx, dw in enumerate(cfg["snax_streamer_cfg"]["data_reader_writer_params"]["num_channel"]):
% if idx % 2 == 0:
  logic [${dw*cfg["tcdm_data_width"]-1}:0] acc2stream_${int(idx/2+num_writer_offset)}_data;
  logic acc2stream_${int(idx/2+num_writer_offset)}_valid;
  logic acc2stream_${int(idx/2+num_writer_offset)}_ready;

% endif
% endfor
% endif
% if "data_reader_params" in cfg["snax_streamer_cfg"]:
  // Ports from streamer to accelerator by reader data movers
% for idx, dw in enumerate(cfg["snax_streamer_cfg"]["data_reader_params"]["num_channel"]):
  logic [${dw*cfg["tcdm_data_width"]-1}:0] stream2acc_${idx}_data;
  logic stream2acc_${idx}_valid;
  logic stream2acc_${idx}_ready;

% endfor
% endif
% if "data_reader_writer_params" in cfg["snax_streamer_cfg"]:
  // Ports from streamer to accelerator by reader-writer data movers
% for idx, dw in enumerate(cfg["snax_streamer_cfg"]["data_reader_writer_params"]["num_channel"]):
% if idx % 2 == 0:
  logic [${dw*cfg["tcdm_data_width"]-1}:0] stream2acc_${int(idx/2+num_reader_offset)}_data;
  logic stream2acc_${int(idx/2+num_reader_offset)}_valid;
  logic stream2acc_${int(idx/2+num_reader_offset)}_ready;

% endif
% endfor
% endif
  // CSR MUXing
  logic [1:0][RegAddrWidth-1:0] acc_csr_req_addr;
  logic [1:0][RegDataWidth-1:0] acc_csr_req_data;
  logic [1:0]                   acc_csr_req_wen;
  logic [1:0]                   acc_csr_req_valid;
  logic [1:0]                   acc_csr_req_ready;
  logic [1:0][RegDataWidth-1:0] acc_csr_rsp_data;
  logic [1:0]                   acc_csr_rsp_valid;
  logic [1:0]                   acc_csr_rsp_ready;

% if not cfg.get("snax_disable_csr_manager", False):
  // Register set signals
  logic [NumRwCsr-1:0][31:0]    acc_csr_reg_rw_set;
  logic                         acc_csr_reg_set_valid;
  logic                         acc_csr_reg_set_ready;
  logic [NumRoCsr-1:0][31:0]    acc_csr_reg_ro_set;
% endif

  //-------------------------------
  // MUX and DEMUX for control signals
  // That separate between streamer CSRs
  // and accelerator CSRs
  //-------------------------------
  snax_csr_mux_demux #(
    // The AddrSelOffset indicates when the MUX switches
    // To streamer or the accelerator CSRs
    // If snax_req_addr_i <  AddrSelOffset, it points
    // to the accelerator CSR manager
    .AddrSelOffSet        ( ${cfg["streamer_csr_num"]}     ), // Streamer CSR number
    .RegDataWidth         ( RegDataWidth      ),
    .RegAddrWidth         ( RegAddrWidth      )
  ) i_snax_csr_mux_demux (
    //-------------------------------
    // Input Core
    //-------------------------------
    .csr_req_addr_i       ( snax_req_addr_i   ),
    .csr_req_data_i       ( snax_req_data_i   ),
    .csr_req_wen_i        ( snax_req_write_i  ),
    .csr_req_valid_i      ( snax_req_valid_i  ),
    .csr_req_ready_o      ( snax_req_ready_o  ),
    .csr_rsp_data_o       ( snax_rsp_data_o   ),
    .csr_rsp_valid_o      ( snax_rsp_valid_o  ),
    .csr_rsp_ready_i      ( snax_rsp_ready_i  ),

    //-------------------------------
    // Output Port
    //-------------------------------
    .acc_csr_req_addr_o   ( acc_csr_req_addr  ),
    .acc_csr_req_data_o   ( acc_csr_req_data  ),
    .acc_csr_req_wen_o    ( acc_csr_req_wen   ),
    .acc_csr_req_valid_o  ( acc_csr_req_valid ),
    .acc_csr_req_ready_i  ( acc_csr_req_ready ),
    .acc_csr_rsp_data_i   ( acc_csr_rsp_data  ),
    .acc_csr_rsp_valid_i  ( acc_csr_rsp_valid ),
    .acc_csr_rsp_ready_o  ( acc_csr_rsp_ready )
  );

% if not cfg.get("snax_disable_csr_manager", False):
  //-----------------------------
  // CSR Manager to control the accelerator
  //-----------------------------
  ${cfg["tag_name"]}_csrman_wrapper #(
    .NumRwCsr             ( NumRwCsr              ),
    .NumRoCsr             ( NumRoCsr              )
  ) i_${cfg["tag_name"]}_csrman_wrapper (
    //-------------------------------
    // Clocks and reset
    //-------------------------------
    .clk_i                ( clk_i                 ),
    .rst_ni               ( rst_ni                ),
    //-----------------------------
    // CSR control ports
    //-----------------------------
    // Request
    .csr_req_addr_i       ( acc_csr_req_addr [1]  ),
    .csr_req_data_i       ( acc_csr_req_data [1]  ),
    .csr_req_write_i      ( acc_csr_req_wen  [1]  ),
    .csr_req_valid_i      ( acc_csr_req_valid[1]  ),
    .csr_req_ready_o      ( acc_csr_req_ready[1]  ),

    // Response
    .csr_rsp_data_o       ( acc_csr_rsp_data [1]  ),
    .csr_rsp_valid_o      ( acc_csr_rsp_valid[1]  ),
    .csr_rsp_ready_i      ( acc_csr_rsp_ready[1]  ),

    //-----------------------------
    // Packed CSR register signals
    //-----------------------------
    // Read-write CSRs
    .csr_reg_rw_set_o     ( acc_csr_reg_rw_set    ),
    .csr_reg_set_valid_o  ( acc_csr_reg_set_valid ),
    .csr_reg_set_ready_i  ( acc_csr_reg_set_ready ),
    // Read-only CSRs
    .csr_reg_ro_set_i     ( acc_csr_reg_ro_set    )  
  );
% endif

  //-----------------------------
  // Accelerator
  //-----------------------------
  // Note: This is the part that needs to be consistent
  // It needs to have the correct connections to the control and data ports!
  // Parameter declarations are custom inside the shell
  // Do not pass it from the top towards here to achieve
  // Uniform shell wrapping :)
  
  ${cfg["tag_name"]}_shell_wrapper i_${cfg["tag_name"]}_shell_wrapper (
    //-------------------------------
    // Clocks and reset
    //-------------------------------
    .clk_i                ( clk_i                 ),
    .rst_ni               ( rst_ni                ),

    //-----------------------------
    // Accelerator ports
    //-----------------------------
    // Note that these have very specific data widths
    // found from the configuration file
% if "data_writer_params" in cfg["snax_streamer_cfg"]:
    // Ports from accelerator to streamer by writer data movers
% for idx, dw in enumerate(cfg["snax_streamer_cfg"]["data_writer_params"]["num_channel"]):
    .acc2stream_${idx}_data_o  ( acc2stream_${idx}_data  ),
    .acc2stream_${idx}_valid_o ( acc2stream_${idx}_valid ),
    .acc2stream_${idx}_ready_i ( acc2stream_${idx}_ready ),

% endfor
% endif
% if "data_reader_writer_params" in cfg["snax_streamer_cfg"]:
    // Ports from accelerator to streamer by reader-writer data movers
% for idx, dw in enumerate(cfg["snax_streamer_cfg"]["data_reader_writer_params"]["num_channel"]):
% if idx % 2 == 0:
    .acc2stream_${int(idx/2+num_writer_offset)}_data_o  ( acc2stream_${int(idx/2+num_writer_offset)}_data  ),
    .acc2stream_${int(idx/2+num_writer_offset)}_valid_o ( acc2stream_${int(idx/2+num_writer_offset)}_valid ),
    .acc2stream_${int(idx/2+num_writer_offset)}_ready_i ( acc2stream_${int(idx/2+num_writer_offset)}_ready ),
% endif

% endfor
% endif
% if "data_reader_params" in cfg["snax_streamer_cfg"]:
    // Ports from streamer to accelerator by reader data movers
% for idx, dw in enumerate(cfg["snax_streamer_cfg"]["data_reader_params"]["num_channel"]):
    .stream2acc_${idx}_data_i  ( stream2acc_${idx}_data  ),
    .stream2acc_${idx}_valid_i ( stream2acc_${idx}_valid ),
    .stream2acc_${idx}_ready_o ( stream2acc_${idx}_ready ),

% endfor
% endif
% if "data_reader_writer_params" in cfg["snax_streamer_cfg"]:
    // Ports from streamer to accelerator by reader-writer data movers
% for idx, dw in enumerate(cfg["snax_streamer_cfg"]["data_reader_writer_params"]["num_channel"]):
% if idx % 2 == 0:
    .stream2acc_${int(idx/2+num_reader_offset)}_data_i  ( stream2acc_${int(idx/2+num_reader_offset)}_data  ),
    .stream2acc_${int(idx/2+num_reader_offset)}_valid_i ( stream2acc_${int(idx/2+num_reader_offset)}_valid ),
    .stream2acc_${int(idx/2+num_reader_offset)}_ready_o ( stream2acc_${int(idx/2+num_reader_offset)}_ready ),
% endif

% endfor
% endif

% if not cfg.get("snax_disable_csr_manager", False):
    //-----------------------------
    // Packed CSR register signals
    //-----------------------------
    .csr_reg_set_i        ( acc_csr_reg_rw_set    ),
    .csr_reg_set_valid_i  ( acc_csr_reg_set_valid ),
    .csr_reg_set_ready_o  ( acc_csr_reg_set_ready ),
    .csr_reg_ro_set_o     ( acc_csr_reg_ro_set    )
% else:
    //-----------------------------
    // CSR control ports
    //-----------------------------
    // Request
    .csr_req_addr_i       ( acc_csr_req_addr [1]  ),
    .csr_req_data_i       ( acc_csr_req_data [1]  ),
    .csr_req_write_i      ( acc_csr_req_wen  [1]  ),
    .csr_req_valid_i      ( acc_csr_req_valid[1]  ),
    .csr_req_ready_o      ( acc_csr_req_ready[1]  ),
    // Response
    .csr_rsp_data_o       ( acc_csr_rsp_data [1]  ),
    .csr_rsp_valid_o      ( acc_csr_rsp_valid[1]  ),
    .csr_rsp_ready_i      ( acc_csr_rsp_ready[1]  )
% endif
  );

  //-----------------------------
  // Streamer Wrapper
  //-----------------------------
  ${cfg["tag_name"]}_streamer_wrapper #(
    .tcdm_req_t               ( tcdm_req_t    ),
    .tcdm_rsp_t               ( tcdm_rsp_t    ),
    .TCDMDataWidth            ( DataWidth     ),
    .TCDMNumPorts             ( SnaxTcdmPorts ),
    .TCDMAddrWidth            ( TCDMAddrWidth )
  ) i_${cfg["tag_name"]}_streamer_wrapper (
    //-----------------------------
    // Clocks and reset
    //-----------------------------
    .clk_i                    ( clk_i  ),
    .rst_ni                   ( rst_ni ),
    //-----------------------------
    // Accelerator ports
    //-----------------------------
    // Note that these have very specific data widths
    // found from the configuration file

% if "data_writer_params" in cfg["snax_streamer_cfg"]:
    // Ports from accelerator to streamer by writer data movers
% for idx, dw in enumerate(cfg["snax_streamer_cfg"]["data_writer_params"]["num_channel"]):
    .acc2stream_${idx}_data_i  ( acc2stream_${idx}_data  ),
    .acc2stream_${idx}_valid_i ( acc2stream_${idx}_valid ),
    .acc2stream_${idx}_ready_o ( acc2stream_${idx}_ready ),

% endfor
% endif
% if "data_reader_writer_params" in cfg["snax_streamer_cfg"]:
    // Ports from accelerator to streamer by reader-writer data movers
% for idx, dw in enumerate(cfg["snax_streamer_cfg"]["data_reader_writer_params"]["num_channel"]):
% if idx % 2 == 0:
    .acc2stream_${int(idx/2+num_writer_offset)}_data_i  ( acc2stream_${int(idx/2+num_writer_offset)}_data  ),
    .acc2stream_${int(idx/2+num_writer_offset)}_valid_i ( acc2stream_${int(idx/2+num_writer_offset)}_valid ),
    .acc2stream_${int(idx/2+num_writer_offset)}_ready_o ( acc2stream_${int(idx/2+num_writer_offset)}_ready ),
% endif

% endfor
% endif
% if "data_reader_params" in cfg["snax_streamer_cfg"]:
    // Ports from streamer to accelerator by reader data movers
% for idx, dw in enumerate(cfg["snax_streamer_cfg"]["data_reader_params"]["num_channel"]):
    .stream2acc_${idx}_data_o  ( stream2acc_${idx}_data  ),
    .stream2acc_${idx}_valid_o ( stream2acc_${idx}_valid ),
    .stream2acc_${idx}_ready_i ( stream2acc_${idx}_ready ),

% endfor
% endif
% if "data_reader_writer_params" in cfg["snax_streamer_cfg"]:
    // Ports from streamer to accelerator by reader-writer data movers
% for idx, dw in enumerate(cfg["snax_streamer_cfg"]["data_reader_writer_params"]["num_channel"]):
% if idx % 2 == 0:
    .stream2acc_${int(idx/2+num_reader_offset)}_data_o  ( stream2acc_${int(idx/2+num_reader_offset)}_data  ),
    .stream2acc_${int(idx/2+num_reader_offset)}_valid_o ( stream2acc_${int(idx/2+num_reader_offset)}_valid ),
    .stream2acc_${int(idx/2+num_reader_offset)}_ready_i ( stream2acc_${int(idx/2+num_reader_offset)}_ready ),
% endif

% endfor
% endif
    //-----------------------------
    // TCDM ports
    //-----------------------------
    .tcdm_req_o ( snax_tcdm_req_o ),
    .tcdm_rsp_i ( snax_tcdm_rsp_i ),

    //-----------------------------
    // CSR control ports
    //-----------------------------
    // Request
    .csr_req_bits_data_i   ( acc_csr_req_data [0] ),
    .csr_req_bits_addr_i   ( acc_csr_req_addr [0] ),
    .csr_req_bits_write_i  ( acc_csr_req_wen  [0] ),
    .csr_req_valid_i       ( acc_csr_req_valid[0] ),
    .csr_req_ready_o       ( acc_csr_req_ready[0] ),
    // Response
    .csr_rsp_bits_data_o   ( acc_csr_rsp_data [0] ),
    .csr_rsp_valid_o       ( acc_csr_rsp_valid[0] ),
    .csr_rsp_ready_i       ( acc_csr_rsp_ready[0] )
  );

  //-----------------------------
  // Barrier control
  // TODO: Fix me later
  //-----------------------------
  assign snax_barrier_o = '0;

% endif
endmodule
