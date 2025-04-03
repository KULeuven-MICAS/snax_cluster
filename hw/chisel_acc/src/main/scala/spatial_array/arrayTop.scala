package snax_acc.spatial_array

import chisel3._
import chisel3.util._

import snax_acc.utils._ 
import snax_acc.utils.DecoupledCut._

// arrayTop with the fsm controller and array
class ArrayTopCfg(params: SpatialArrayParam) extends Bundle {
  val fsmCfg = new Bundle {
    val K_i                    = UInt(params.configWidth.W)
    val N_i                    = UInt(params.configWidth.W)
    val M_i                    = UInt(params.configWidth.W)
    val subtraction_constant_i = UInt(params.configWidth.W)
  }

  val arrayCfg = new Bundle {
    val spatialArrayCfg = UInt(params.configWidth.W)
    val dataTypeCfg     = UInt(params.configWidth.W)
  }
}

class ArrayTopIO(params: SpatialArrayParam) extends Bundle {
  val data = new Bundle {
    val in_a  = Flipped(DecoupledIO(UInt(params.inputAWidth.W)))
    val in_b  = Flipped(DecoupledIO(UInt(params.inputBWidth.W)))
    val in_c  = Flipped(DecoupledIO(UInt(params.inputCSerialDataWidth.W)))
    val out_d = DecoupledIO(UInt(params.outputDSerialDataWidth.W))
  }

  val ctrl = Flipped(DecoupledIO(new ArrayTopCfg(params)))

  val busy_o              = Output(Bool())
  val performance_counter = Output(UInt(params.configWidth.W))
}

class ArrayTop(params: SpatialArrayParam) extends Module with RequireAsyncReset {

  val io = IO(new ArrayTopIO(params))

  val csrReg = RegInit(0.U.asTypeOf(new ArrayTopCfg(params)))

  // -----------------------------------
  // state machine starts
  // -----------------------------------

  // State declaration
  val sIDLE :: sBUSY :: Nil = Enum(2)
  val cstate                = RegInit(sIDLE)
  val nstate                = WireInit(sIDLE)

  // signals for state transition
  val config_valid       = WireInit(0.B)
  val computation_finish = WireInit(0.B)

  val zeroLoopBoundCase =
    io.ctrl.bits.fsmCfg.M_i === 0.U || io.ctrl.bits.fsmCfg.K_i === 0.U || io.ctrl.bits.fsmCfg.K_i === 0.U

  // Changing states
  cstate := nstate

  chisel3.dontTouch(cstate)
  switch(cstate) {
    is(sIDLE) {
      when(config_valid) {
        nstate := sBUSY
      }.otherwise {
        nstate := sIDLE
      }

    }
    is(sBUSY) {
      when(computation_finish) {
        nstate := sIDLE
      }.otherwise {
        nstate := sBUSY
      }
    }
  }

  config_valid := io.ctrl.fire && !zeroLoopBoundCase
  io.ctrl.ready := cstate === sIDLE

  // Store the configurations when config valid
  when(config_valid && cstate === sIDLE) {
    when(!zeroLoopBoundCase) {
      csrReg.fsmCfg.M_i := io.ctrl.bits.fsmCfg.M_i
      csrReg.fsmCfg.N_i := io.ctrl.bits.fsmCfg.N_i
      csrReg.fsmCfg.K_i := io.ctrl.bits.fsmCfg.K_i
    }.otherwise {
      assert(
        io.ctrl.bits.fsmCfg.M_i =/= 0.U || io.ctrl.bits.fsmCfg.K_i =/= 0.U || io.ctrl.bits.fsmCfg.K_i =/= 0.U,
        " M == 0 or K ==0 or N == 0, invalid configuration!"
      )
    }
    csrReg.fsmCfg.subtraction_constant_i := io.ctrl.bits.fsmCfg.subtraction_constant_i
  }

  // hardware loop


  // -----------------------------------
  // state machine ends
  // -----------------------------------

  // -----------------------------------
  // insert resgisters for data cut starts
  // -----------------------------------
  val cut_combined_decoupled_a_b_sub_in  = Wire(
    Decoupled(UInt((params.inputAWidth + params.inputBWidth + params.configWidth).W))
  )
  val cut_combined_decoupled_a_b_sub_out = Wire(
    Decoupled(UInt((params.inputAWidth + params.inputBWidth + params.configWidth).W))
  )

  val a_b_sub_cut = Module(
    new DecoupledCatNto1(
      Seq(
        params.inputAWidth,
        params.inputBWidth,
        params.configWidth
      )
    )
  )

  a_b_sub_cut.io.in(0) <> io.data.in_a
  a_b_sub_cut.io.in(1) <> io.data.in_b

  val decoupled_sub = Wire(Decoupled(UInt(params.configWidth.W)))
  decoupled_sub.bits  := io.ctrl.bits.fsmCfg.subtraction_constant_i
  decoupled_sub.valid := cstate === sBUSY
  a_b_sub_cut.io.in(2) <> decoupled_sub

  a_b_sub_cut.io.out <> cut_combined_decoupled_a_b_sub_in

  cut_combined_decoupled_a_b_sub_in -\> cut_combined_decoupled_a_b_sub_out

  val a_after_cut   = Wire(Decoupled(UInt(params.inputAWidth.W)))
  val b_after_cut   = Wire(Decoupled(UInt(params.inputBWidth.W)))
  val sub_after_cut = Wire(Decoupled(UInt(params.configWidth.W)))

  a_after_cut.bits  := cut_combined_decoupled_a_b_sub_out.bits(
    params.inputAWidth + params.inputBWidth + params.configWidth - 1,
    params.inputAWidth + params.inputBWidth
  )
  a_after_cut.valid := cut_combined_decoupled_a_b_sub_out.valid

  b_after_cut.bits  := cut_combined_decoupled_a_b_sub_out.bits(
    params.inputAWidth + params.inputBWidth - 1,
    params.configWidth
  )
  b_after_cut.valid := cut_combined_decoupled_a_b_sub_out.valid

  sub_after_cut.bits  := cut_combined_decoupled_a_b_sub_out.bits(
    params.configWidth - 1,
    0
  )
  sub_after_cut.valid := cut_combined_decoupled_a_b_sub_out.valid

  cut_combined_decoupled_a_b_sub_out.ready := a_after_cut.fire && b_after_cut.fire && sub_after_cut.fire

  // -----------------------------------
  // insert resgisters for data cut ends
  // -----------------------------------

  // -----------------------------------
  // serial_parallel data converters starts
  // ---------------------------------

  // C32 serial to parallel converter
  val C_s2p = Module(
    new SerialToParallel(
      SerialToParallelParams(
        parallelWidth = params.arrayInputCWidth,
        serialWidth   = params.inputCSerialDataWidth
      )
    )
  )

  // D32 parallel to serial converter
  val D_p2s = Module(
    new ParallelToSerial(
      ParallelToSerialParams(
        parallelWidth = params.arrayOutputDWidth,
        serialWidth   = params.outputDSerialDataWidth
      )
    )
  )
  require(params.inputCSerialDataWidth == params.outputDSerialDataWidth)

  io.data.in_c <> C_s2p.io.in
  io.data.out_d <> D_p2s.io.out

  // ------------------------------------
  // serial_parallel data converters ends
  // ------------------------------------

  // ------------------------------------
  // array instance and data handshake signal connections starts
  // ------------------------------------

  val array = Module(new SpatialArray(params))

  // ctrl signals
  array.io.ctrl.spatialArrayCfg := csrReg.arrayCfg.spatialArrayCfg
  array.io.ctrl.dataTypeCfg     := csrReg.arrayCfg.dataTypeCfg

  // data signals
  array.io.data.in_a <> a_after_cut
  array.io.data.in_b <> b_after_cut
  array.io.data.in_c <> C_s2p.io.out
  array.io.data.out_d <> D_p2s.io.in
  array.io.data.in_substraction <> sub_after_cut

  // ------------------------------------
  // array instance and data handshake signal connections ends
  // ------------------------------------

  val performance_counter = RegInit(0.U(params.configWidth.W))

  when(cstate === sBUSY) {
    performance_counter := performance_counter + 1.U
  }.elsewhen(config_valid) {
    performance_counter := 0.U
  }

  // output control signals for read-only csrs
  io.performance_counter := performance_counter

  io.busy_o := cstate =/= sIDLE
}

object ArrayTopEmitter extends App {
  emitVerilog(
    new ArrayTop(SpatialArrayParam()),
    Array("--target-dir", "generated/SpatialArray")
  )
}
