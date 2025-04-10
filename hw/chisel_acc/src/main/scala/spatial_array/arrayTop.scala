package snax_acc.spatial_array

import chisel3._
import chisel3.util._

import snax_acc.utils.DecoupledCut._
import snax_acc.utils._

// arrayTop with the fsm controller and array
class ArrayTopCfg(params: SpatialArrayParam) extends Bundle {
  val fsmCfg = new Bundle {
    val K_i                    = UInt(params.configWidth.W)
    val N_i                    = UInt(params.configWidth.W)
    val M_i                    = UInt(params.configWidth.W)
    val subtraction_constant_i = UInt(params.configWidth.W)
  }

  val arrayCfg = new Bundle {
    val arrayShapeCfg = UInt(params.configWidth.W)
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

  config_valid  := io.ctrl.fire && !zeroLoopBoundCase && cstate === sIDLE
  io.ctrl.ready := cstate === sIDLE

  val dOutputCounter = Module(new BasicCounter(params.configWidth))
  val output_serial_factor = 
    if (params.arrayOutputDWidth <= params.outputDSerialDataWidth) 1
    else params.arrayOutputDWidth / params.outputDSerialDataWidth

  dOutputCounter.io.ceil  := csrReg.fsmCfg.M_i * csrReg.fsmCfg.N_i * output_serial_factor.U

  dOutputCounter.io.tick  := io.data.out_d.fire && cstate === sBUSY
  dOutputCounter.io.reset := computation_finish
  computation_finish      := dOutputCounter.io.lastVal

  // Store the configurations when config valid
  when(config_valid) {
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
    csrReg.arrayCfg.arrayShapeCfg := io.ctrl.bits.arrayCfg.arrayShapeCfg
    csrReg.arrayCfg.dataTypeCfg     := io.ctrl.bits.arrayCfg.dataTypeCfg
  }

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

  val combined_decoupled_a_b_sub = Module(
    new DecoupledCatNto1(
      Seq(
        params.inputAWidth,
        params.inputBWidth,
        params.configWidth
      )
    )
  )

  combined_decoupled_a_b_sub.io.in(0) <> io.data.in_a
  combined_decoupled_a_b_sub.io.in(1) <> io.data.in_b

  val decoupled_sub = Wire(Decoupled(UInt(params.configWidth.W)))
  decoupled_sub.bits  := io.ctrl.bits.fsmCfg.subtraction_constant_i
  decoupled_sub.valid := cstate === sBUSY
  combined_decoupled_a_b_sub.io.in(2) <> decoupled_sub

  combined_decoupled_a_b_sub.io.out <> cut_combined_decoupled_a_b_sub_in

  cut_combined_decoupled_a_b_sub_in -\> cut_combined_decoupled_a_b_sub_out

  val a_after_cut   = Wire(Decoupled(UInt(params.inputAWidth.W)))
  val b_after_cut   = Wire(Decoupled(UInt(params.inputBWidth.W)))
  val sub_after_cut = Wire(Decoupled(UInt(params.configWidth.W)))

  a_after_cut.bits  := cut_combined_decoupled_a_b_sub_out.bits(
    params.inputAWidth + params.inputBWidth + params.configWidth - 1,
    params.inputBWidth + params.configWidth
  )
  a_after_cut.valid := cut_combined_decoupled_a_b_sub_out.valid

  b_after_cut.bits  := cut_combined_decoupled_a_b_sub_out.bits(
    params.inputBWidth + params.configWidth - 1,
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

  // array accAddExtIn control signal

  val array = Module(new SpatialArray(params))

  val accAddExtIn        = WireInit(0.B)
  val computeFireCounter = Module(new BasicCounter(params.configWidth))
  computeFireCounter.io.ceil := csrReg.fsmCfg.K_i
  val addCFire =
    (a_after_cut.fire && b_after_cut.fire && array.io.data.in_c.fire && computeFireCounter.io.value === 0.U)
  val mulABFire = (a_after_cut.fire && b_after_cut.fire && computeFireCounter.io.value =/= 0.U)
  computeFireCounter.io.tick  := (addCFire || mulABFire) && cstate === sBUSY
  computeFireCounter.io.reset := computation_finish

  accAddExtIn := computeFireCounter.io.value === 0.U && cstate === sBUSY

  // ctrl signals
  array.io.ctrl.arrayShapeCfg := csrReg.arrayCfg.arrayShapeCfg
  array.io.ctrl.dataTypeCfg     := csrReg.arrayCfg.dataTypeCfg
  array.io.ctrl.accAddExtIn     := accAddExtIn
  array.io.ctrl.accClear        := computation_finish

  // data signals
  array.io.data.in_a <> a_after_cut
  array.io.data.in_b <> b_after_cut

  array.io.data.in_c.bits  := C_s2p.io.out.bits
  array.io.data.in_c.valid := C_s2p.io.out.valid && cstate === sBUSY
  // array c_ready  considering output stationary
  C_s2p.io.out.ready       := addCFire           && cstate === sBUSY

  array.io.data.in_substraction <> sub_after_cut

  // array d_ready considering output stationary
  val dOutputValidCounter = Module(new BasicCounter(params.configWidth))
  dOutputValidCounter.io.ceil  := csrReg.fsmCfg.K_i
  dOutputValidCounter.io.tick  := array.io.data.out_d.fire && cstate === sBUSY
  dOutputValidCounter.io.reset := computation_finish

  D_p2s.io.in.bits := array.io.data.out_d.bits
  D_p2s.io.in.valid := array.io.data.out_d.valid && cstate === sBUSY && dOutputValidCounter.io.value === (csrReg.fsmCfg.K_i - 1.U)
  array.io.data.out_d.ready := Mux(D_p2s.io.in.valid, D_p2s.io.in.ready, true.B) && cstate === sBUSY

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
