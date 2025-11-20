# This script was generated automatically by bender.
set ROOT "/esat/micas-lapserv11/users/rgeens/snax_cluster"
set search_path_initial $search_path

set search_path $search_path_initial

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/tech_cells_generic-1282165f7b690985/src/rtl/tc_sram.sv" \
        "$ROOT/.bender/git/checkouts/tech_cells_generic-1282165f7b690985/src/rtl/tc_sram_impl.sv" \
    ]
]} {return 1}

set search_path $search_path_initial

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/tech_cells_generic-1282165f7b690985/src/rtl/tc_clk.sv" \
    ]
]} {return 1}

set search_path $search_path_initial

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/tech_cells_generic-1282165f7b690985/src/deprecated/pulp_clock_gating_async.sv" \
        "$ROOT/.bender/git/checkouts/tech_cells_generic-1282165f7b690985/src/deprecated/cluster_clk_cells.sv" \
        "$ROOT/.bender/git/checkouts/tech_cells_generic-1282165f7b690985/src/deprecated/pulp_clk_cells.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/binary_to_gray.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/cb_filter_pkg.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/cc_onehot.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/cdc_reset_ctrlr_pkg.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/cf_math_pkg.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/clk_int_div.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/credit_counter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/delta_counter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/ecc_pkg.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/edge_propagator_tx.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/exp_backoff.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/fifo_v3.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/gray_to_binary.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/isochronous_4phase_handshake.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/isochronous_spill_register.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/lfsr.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/lfsr_16bit.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/lfsr_8bit.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/lossy_valid_to_stream.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/mv_filter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/onehot_to_bin.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/plru_tree.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/passthrough_stream_fifo.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/popcount.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/rr_arb_tree.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/rstgen_bypass.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/serial_deglitch.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/shift_reg.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/shift_reg_gated.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/spill_register_flushable.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_demux.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_filter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_fork.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_intf.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_join_dynamic.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_mux.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_throttle.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/sub_per_hash.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/sync.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/sync_wedge.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/unread.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/read.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/addr_decode_dync.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/cdc_2phase.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/cdc_4phase.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/clk_int_div_static.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/addr_decode.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/addr_decode_napot.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/multiaddr_decode.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/cb_filter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/cdc_fifo_2phase.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/clk_mux_glitch_free.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/counter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/ecc_decode.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/ecc_encode.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/edge_detect.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/lzc.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/max_counter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/rstgen.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/spill_register.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_delay.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_fifo.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_fork_dynamic.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_join.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/cdc_reset_ctrlr.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/cdc_fifo_gray.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/fall_through_register.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/id_queue.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_to_mem.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_arbiter_flushable.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_fifo_optimal_wrap.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_register.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_xbar.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/cdc_fifo_gray_clearable.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/cdc_2phase_clearable.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/mem_to_banks_detailed.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_arbiter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/stream_omega_net.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/mem_to_banks.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/deprecated/clock_divider_counter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/deprecated/clk_div.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/deprecated/find_first_one.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/deprecated/generic_LFSR_8bit.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/deprecated/generic_fifo.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/deprecated/prioarbiter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/deprecated/pulp_sync.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/deprecated/pulp_sync_wedge.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/deprecated/rrarbiter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/deprecated/clock_divider.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/deprecated/fifo_v2.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/deprecated/fifo_v1.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/edge_propagator_ack.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/edge_propagator.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/src/edge_propagator_rx.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/apb-0b8df27abad700ff/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/apb-0b8df27abad700ff/src/apb_pkg.sv" \
        "$ROOT/.bender/git/checkouts/apb-0b8df27abad700ff/src/apb_intf.sv" \
        "$ROOT/.bender/git/checkouts/apb-0b8df27abad700ff/src/apb_err_slv.sv" \
        "$ROOT/.bender/git/checkouts/apb-0b8df27abad700ff/src/apb_regs.sv" \
        "$ROOT/.bender/git/checkouts/apb-0b8df27abad700ff/src/apb_cdc.sv" \
        "$ROOT/.bender/git/checkouts/apb-0b8df27abad700ff/src/apb_demux.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_pkg.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_intf.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_atop_filter.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_burst_splitter.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_bus_compare.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_cdc_dst.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_cdc_src.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_cut.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_delayer.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_demux_simple.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_dw_downsizer.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_dw_upsizer.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_fifo.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_id_remap.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_id_prepend.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_isolate.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_join.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_lite_demux.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_lite_dw_converter.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_lite_from_mem.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_lite_join.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_lite_lfsr.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_lite_mailbox.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_lite_mux.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_lite_regs.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_lite_to_apb.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_lite_to_axi.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_modify_address.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_mux.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_rw_join.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_rw_split.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_serializer.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_slave_compare.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_throttle.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_to_detailed_mem.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_cdc.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_demux.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_err_slv.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_dw_converter.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_from_mem.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_id_serialize.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_lfsr.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_multicut.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_to_axi_lite.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_to_mem.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_iw_converter.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_lite_xbar.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_xbar.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_to_mem_banked.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_to_mem_interleaved.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_to_mem_split.sv" \
        "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/src/axi_xp.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-ca8b2349afb6712f/hdl/defs_div_sqrt_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-ca8b2349afb6712f/hdl/iteration_div_sqrt_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-ca8b2349afb6712f/hdl/control_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-ca8b2349afb6712f/hdl/norm_div_sqrt_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-ca8b2349afb6712f/hdl/preprocess_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-ca8b2349afb6712f/hdl/nrbd_nrsc_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-ca8b2349afb6712f/hdl/div_sqrt_top_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-ca8b2349afb6712f/hdl/div_sqrt_mvp_wrapper.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/hwpe-ctrl-078b13d4b656e469/rtl"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-078b13d4b656e469/rtl/hwpe_ctrl_interfaces.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-078b13d4b656e469/rtl/hwpe_ctrl_package.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-078b13d4b656e469/rtl/hwpe_ctrl_regfile_latch.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-078b13d4b656e469/rtl/hwpe_ctrl_partial_mult.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-078b13d4b656e469/rtl/hwpe_ctrl_seq_mult.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-078b13d4b656e469/rtl/hwpe_ctrl_uloop.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-078b13d4b656e469/rtl/hwpe_ctrl_regfile_latch_test_wrap.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-078b13d4b656e469/rtl/hwpe_ctrl_regfile.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-078b13d4b656e469/rtl/hwpe_ctrl_slave.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/hwpe_stream_package.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/hwpe_stream_interfaces.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/basic/hwpe_stream_assign.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/basic/hwpe_stream_buffer.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/basic/hwpe_stream_demux_static.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/basic/hwpe_stream_deserialize.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/basic/hwpe_stream_fence.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/basic/hwpe_stream_merge.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/basic/hwpe_stream_mux_static.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/basic/hwpe_stream_serialize.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/basic/hwpe_stream_split.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/fifo/hwpe_stream_fifo_ctrl.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/fifo/hwpe_stream_fifo_scm.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/streamer/hwpe_stream_addressgen.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/streamer/hwpe_stream_addressgen_v2.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/streamer/hwpe_stream_addressgen_v3.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/streamer/hwpe_stream_sink_realign.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/streamer/hwpe_stream_source_realign.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/streamer/hwpe_stream_strbgen.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/streamer/hwpe_stream_streamer_queue.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/tcdm/hwpe_stream_tcdm_assign.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/tcdm/hwpe_stream_tcdm_mux.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/tcdm/hwpe_stream_tcdm_mux_static.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/tcdm/hwpe_stream_tcdm_reorder.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/tcdm/hwpe_stream_tcdm_reorder_static.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/fifo/hwpe_stream_fifo_earlystall.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/fifo/hwpe_stream_fifo_earlystall_sidech.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/fifo/hwpe_stream_fifo_scm_test_wrap.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/fifo/hwpe_stream_fifo_sidech.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/fifo/hwpe_stream_fifo.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/tcdm/hwpe_stream_tcdm_fifo_load_sidech.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/fifo/hwpe_stream_fifo_passthrough.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/streamer/hwpe_stream_source.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/tcdm/hwpe_stream_tcdm_fifo.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/tcdm/hwpe_stream_tcdm_fifo_load.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/tcdm/hwpe_stream_tcdm_fifo_store.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-b9a93a56b69d7039/rtl/streamer/hwpe_stream_sink.sv" \
    ]
]} {return 1}

set search_path $search_path_initial

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1r_1w_test_wrap.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1w_64b_multi_port_read_32b_1row.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1w_multi_port_read_1row.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1r_1w_all.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1r_1w_all_test_wrap.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1r_1w_be.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1r_1w.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1r_1w_1row.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1w_128b_multi_port_read_32b.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1w_64b_multi_port_read_32b.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1w_64b_1r_32b.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1w_multi_port_read_be.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_1w_multi_port_read.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_2r_1w_asymm.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_2r_1w_asymm_test_wrap.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_2r_2w.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_3r_2w.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_3r_2w_be.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_multi_way_1w_64b_multi_port_read_32b.sv" \
        "$ROOT/.bender/git/checkouts/scm-e009308f0dd3005a/latch_scm/register_file_multi_way_1w_multi_port_read.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-360637610c0ab464/src/axi_res_tbl.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-360637610c0ab464/src/axi_riscv_amos_alu.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-360637610c0ab464/src/axi_riscv_amos.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-360637610c0ab464/src/axi_riscv_lrsc.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-360637610c0ab464/src/axi_riscv_atomics.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-360637610c0ab464/src/axi_riscv_lrsc_wrap.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-360637610c0ab464/src/axi_riscv_amos_wrap.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-360637610c0ab464/src/axi_riscv_atomics_wrap.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-360637610c0ab464/src/axi_riscv_atomics_structs.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/snitch_icache_pkg.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/riscv_instr_branch.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/multi_accept_rr_arb.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/snitch_axi_to_cache.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/snitch_icache_l0.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/snitch_icache_refill.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/snitch_icache_lfsr.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/snitch_icache_tag.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/snitch_icache_data.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/snitch_icache_lookup_parallel.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/snitch_icache_handler.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/snitch_icache.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-0cf2c89bb71ebd3b/src/snitch_read_only_cache.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_pkg.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_cast_multi.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_classifier.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/vendor/opene906/E906_RTL_FACTORY/gen_rtl/clk/rtl/gated_clk_cell.v" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ctrl.v" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ff1.v" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_pack_single.v" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_prepare.v" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_round_single.v" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_special.v" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_srt_single.v" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_top.v" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_dp.v" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_frbus.v" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_src_type.v" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_divsqrt_th_32.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_divsqrt_multi.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_fma.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_fma_multi.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_sdotp_multi.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_sdotp_multi_wrapper.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_noncomp.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_opgroup_block.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_opgroup_fmt_slice.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_opgroup_multifmt_slice.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_rounding.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/lfsr_sr.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-4a953b8b22f6345f/src/fpnew_top.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/hwpe-mac-engine-2e73b3bfb9f7ed43/rtl"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-2e73b3bfb9f7ed43/rtl/mac_package.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-2e73b3bfb9f7ed43/rtl/mac_engine.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-2e73b3bfb9f7ed43/rtl/mac_fsm.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-2e73b3bfb9f7ed43/rtl/mac_streamer.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-2e73b3bfb9f7ed43/rtl/mac_ctrl.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-2e73b3bfb9f7ed43/rtl/mac_top.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-2e73b3bfb9f7ed43/wrap/mac_top_wrap.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/apb-0b8df27abad700ff/include"
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/reg_intf.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/vendor/lowrisc_opentitan/src/prim_subreg_arb.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/vendor/lowrisc_opentitan/src/prim_subreg_ext.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/apb_to_reg.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/axi_lite_to_reg.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/axi_to_reg_v2.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/periph_to_reg.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/reg_cdc.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/reg_cut.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/reg_demux.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/reg_err_slv.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/reg_filter_empty_writes.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/reg_mux.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/reg_to_apb.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/reg_to_mem.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/reg_to_tlul.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/reg_to_axi.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/reg_uniform.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/vendor/lowrisc_opentitan/src/prim_subreg_shadow.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/vendor/lowrisc_opentitan/src/prim_subreg.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/src/deprecated/axi_to_reg.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/riscv-dbg-4d2e83f8d49c3adc/src/dm_pkg.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-4d2e83f8d49c3adc/debug_rom/debug_rom.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-4d2e83f8d49c3adc/debug_rom/debug_rom_one_scratch.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-4d2e83f8d49c3adc/src/dm_csrs.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-4d2e83f8d49c3adc/src/dm_mem.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-4d2e83f8d49c3adc/src/dmi_cdc.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/riscv-dbg-4d2e83f8d49c3adc/src/dmi_jtag_tap.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/riscv-dbg-4d2e83f8d49c3adc/src/dm_sba.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-4d2e83f8d49c3adc/src/dm_top.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-4d2e83f8d49c3adc/src/dmi_jtag.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-4d2e83f8d49c3adc/src/dm_obi_top.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/xdma_pkg.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/find_first_one_idx.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/dw_up_converter.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/dw_down_converter.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/dw_converter.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/xdma_req_manager.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/xdma_grant_manager.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/xdma_data_path.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/xdma_burst_reshaper.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/xdma_meta_manager.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/xdma_axi_to_write.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/xdma_write_demux.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/xdma_finish_manager.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/xdma_req_backend.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-95e82763dbd85cfa/src/xdma_axi_adapter_top.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/hw/future/src/mem_to_axi_lite.sv" \
        "$ROOT/hw/future/src/idma_reg64_frontend_reg_pkg.sv" \
        "$ROOT/hw/future/src/idma_tf_id_gen.sv" \
        "$ROOT/hw/future/src/dma/axi_dma_data_path.sv" \
        "$ROOT/hw/future/src/axi_interleaved_xbar.sv" \
        "$ROOT/hw/future/src/axi_zero_mem.sv" \
        "$ROOT/hw/future/src/idma_reg64_frontend_reg_top.sv" \
        "$ROOT/hw/future/src/idma_reg64_frontend.sv" \
        "$ROOT/hw/future/src/dma/axi_dma_data_mover.sv" \
        "$ROOT/hw/future/src/dma/axi_dma_burst_reshaper.sv" \
        "$ROOT/hw/future/src/dma/axi_dma_backend.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/hw/reqrsp_interface/src/reqrsp_pkg.sv" \
        "$ROOT/hw/reqrsp_interface/src/reqrsp_intf.sv" \
        "$ROOT/hw/reqrsp_interface/src/axi_to_reqrsp.sv" \
        "$ROOT/hw/reqrsp_interface/src/reqrsp_cut.sv" \
        "$ROOT/hw/reqrsp_interface/src/reqrsp_demux.sv" \
        "$ROOT/hw/reqrsp_interface/src/reqrsp_iso.sv" \
        "$ROOT/hw/reqrsp_interface/src/reqrsp_mux.sv" \
        "$ROOT/hw/reqrsp_interface/src/reqrsp_to_axi.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/hw/mem_interface/src/mem_wide_narrow_mux.sv" \
        "$ROOT/hw/mem_interface/src/mem_interface.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/hw/tcdm_interface/src/tcdm_interface.sv" \
        "$ROOT/hw/tcdm_interface/src/axi_to_tcdm.sv" \
        "$ROOT/hw/tcdm_interface/src/reqrsp_to_tcdm.sv" \
        "$ROOT/hw/tcdm_interface/src/tcdm_mux.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/hw/snitch/src/snitch_pma_pkg.sv" \
        "$ROOT/hw/snitch/src/riscv_instr.sv" \
        "$ROOT/hw/snitch/src/csr_snax_def.sv" \
        "$ROOT/hw/snitch/src/snitch_pkg.sv" \
        "$ROOT/hw/snitch/src/snitch_regfile_ff.sv" \
        "$ROOT/hw/snitch/src/snitch_lsu.sv" \
        "$ROOT/hw/snitch/src/snitch_l0_tlb.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        SNITCH_ENABLE_PERF \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/hw/snitch/src/snitch.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/hw/snitch_vm/src/snitch_ptw.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/hw/snitch_dma/src/axi_dma_pkg.sv" \
        "$ROOT/hw/snitch_dma/src/axi_dma_error_handler.sv" \
        "$ROOT/hw/snitch_dma/src/axi_dma_perf_counters.sv" \
        "$ROOT/hw/snitch_dma/src/axi_dma_twod_ext.sv" \
        "$ROOT/hw/snitch_dma/src/axi_dma_tc_snitch_fe.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/hw/snitch_ipu/src/snitch_ipu_pkg.sv" \
        "$ROOT/hw/snitch_ipu/src/snitch_ipu_alu.sv" \
        "$ROOT/hw/snitch_ipu/src/snitch_int_ss.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/hw/snitch_ssr/src/snitch_ssr_pkg.sv" \
        "$ROOT/hw/snitch_ssr/src/snitch_ssr_switch.sv" \
        "$ROOT/hw/snitch_ssr/src/snitch_ssr_credit_counter.sv" \
        "$ROOT/hw/snitch_ssr/src/snitch_ssr_indirector.sv" \
        "$ROOT/hw/snitch_ssr/src/snitch_ssr_intersector.sv" \
        "$ROOT/hw/snitch_ssr/src/snitch_ssr_addr_gen.sv" \
        "$ROOT/hw/snitch_ssr/src/snitch_ssr.sv" \
        "$ROOT/hw/snitch_ssr/src/snitch_ssr_streamer.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/hw/snitch_cluster/src/snitch_data_mem.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_amo_shim.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg_pkg.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg_top.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_fpu.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_sequencer.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_tcdm_interconnect.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_barrier.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_fp_ss.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_shared_muldiv.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_cc.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_clkdiv2.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_hive.sv" \
        "$ROOT/hw/snitch_cluster/src/snitch_cluster.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/hw/snax_util/src/snax_csr_mux_demux.sv" \
        "$ROOT/hw/snax_util/src/snax_acc_mux_demux.sv" \
        "$ROOT/hw/snax_util/src/snax_intf_translator.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/../chisel-ssm/chisel-float/src/main/resources/common_block/fpnew_pkg_snax.sv" \
        "$ROOT/../chisel-ssm/chisel-float/src/main/resources/common_block/lzc.sv" \
        "$ROOT/../chisel-ssm/chisel-float/src/main/resources/common_block/fpnew_classifier.sv" \
        "$ROOT/../chisel-ssm/chisel-float/src/main/resources/common_block/fpnew_rounding.sv" \
        "$ROOT/../chisel-ssm/chisel-float/src/main/resources/fp_convert.sv" \
        "$ROOT/../chisel-ssm/chisel-float/src/main/resources/fp_add.sv" \
        "$ROOT/../chisel-ssm/chisel-float/src/main/resources/fp_mul.sv" \
        "$ROOT/../chisel-ssm/chisel-float/src/main/resources/fp_fma.sv" \
        "$ROOT/../chisel-ssm/generated/SimbaCore/SimbaCore.sv" \
        "$ROOT/hw/snax_simbacore/snax_simbacore_shell_wrapper.sv" \
        "$ROOT/target/snitch_cluster/generated/snax_simbacore/snax_simbacore_reqrspman_ReqRspManager.sv" \
        "$ROOT/target/snitch_cluster/generated/snax_simbacore/snax_simbacore_Streamer.sv" \
        "$ROOT/target/snitch_cluster/generated/snax_simbacore/snax_simbacore_csrman_wrapper.sv" \
        "$ROOT/target/snitch_cluster/generated/snax_simbacore/snax_simbacore_streamer_wrapper.sv" \
        "$ROOT/target/snitch_cluster/generated/snax_simbacore/snax_simbacore_wrapper.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi-10c18867bc585e38/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-02aa01ee4a3b2e52/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-f4ec7adf92a180e1/include"
lappend search_path "$ROOT/hw/mem_interface/include"
lappend search_path "$ROOT/hw/reqrsp_interface/include"
lappend search_path "$ROOT/hw/snitch/include"
lappend search_path "$ROOT/hw/snitch_ssr/include"
lappend search_path "$ROOT/hw/tcdm_interface/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_SNAX_SIMBACORE \
        TARGET_SNAX_SIMBACORE_CLUSTER \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "$ROOT/target/snitch_cluster/generated/snax_simbacore_cluster_wrapper.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
