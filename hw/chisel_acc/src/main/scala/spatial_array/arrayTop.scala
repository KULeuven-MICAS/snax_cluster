package snax_acc.spatial_array

import chisel3._
import chisel3.util._

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
    val dataTypeCfg   = UInt(params.configWidth.W)
  }
}

class ArrayTopIO(params: SpatialArrayParam) extends Bundle {
  val data = new Bundle {
    val in_a  = Flipped(DecoupledIO(UInt(params.arrayInputAWidth.W)))
    val in_b  = Flipped(DecoupledIO(UInt(params.arrayInputBWidth.W)))
    val in_c  = Flipped(DecoupledIO(UInt(params.serialInputCDataWidth.W)))
    val out_d = DecoupledIO(UInt(params.serialOutputDDataWidth.W))
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

  val dimRom = VecInit(params.arrayDim.map { twoD =>
    VecInit(twoD.map { oneD =>
      VecInit(oneD.map(_.U(params.configWidth.W)))
    })
  })

  val inputCElemWidthRom = VecInit(params.inputCElemWidth.map(_.U(params.configWidth.W)))
  val outPutDWidthRom    = VecInit(params.outputDElemWidth.map(_.U(params.configWidth.W)))

  def realBandWidth(
    dataTypeIdx:  UInt,
    dimIdx:       UInt,
    elemWidthSeq: Vec[UInt]
  ) = {
    val dim = dimRom(dataTypeIdx)(dimIdx)
    dim(0) * dim(2) * elemWidthSeq(dataTypeIdx)
  }

  val dOutputCounter = Module(new BasicCounter(params.configWidth))

  val runTimeOutputBandWidthFactor = (realBandWidth(
    csrReg.arrayCfg.dataTypeCfg,
    csrReg.arrayCfg.arrayShapeCfg,
    outPutDWidthRom
  ) / params.serialOutputDDataWidth.U)

  val output_serial_factor =
    Mux(
      params.arrayOutputDWidth.U <= params.serialOutputDDataWidth.U,
      1.U,
      Mux(
        runTimeOutputBandWidthFactor
          === 0.U,
        1.U,
        runTimeOutputBandWidthFactor
      )
    )

  dOutputCounter.io.ceil := csrReg.fsmCfg.M_i * csrReg.fsmCfg.N_i * output_serial_factor

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
    csrReg.arrayCfg.arrayShapeCfg        := io.ctrl.bits.arrayCfg.arrayShapeCfg
    csrReg.arrayCfg.dataTypeCfg          := io.ctrl.bits.arrayCfg.dataTypeCfg
  }

  // -----------------------------------
  // state machine ends
  // -----------------------------------

  // -----------------------------------
  // insert resgisters for data cut starts
  // -----------------------------------
  val cut_combined_decoupled_a_b_sub_in  = Wire(
    Decoupled(UInt((params.arrayInputAWidth + params.arrayInputBWidth + params.configWidth).W))
  )
  val cut_combined_decoupled_a_b_sub_out = Wire(
    Decoupled(UInt((params.arrayInputAWidth + params.arrayInputBWidth + params.configWidth).W))
  )

  val combined_decoupled_a_b_sub = Module(
    new DecoupledCatNto1(
      Seq(
        params.arrayInputAWidth,
        params.arrayInputBWidth,
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

  val cut_buffer = Module(
    new DataCut(chiselTypeOf(cut_combined_decoupled_a_b_sub_in.bits), delay = params.adderTreeDelay) {
      override val desiredName =
        s"DataCut${params.adderTreeDelay}_W_" + cut_combined_decoupled_a_b_sub_in.bits.getWidth.toString + "_T_" + cut_combined_decoupled_a_b_sub_in.bits.getClass.getSimpleName
    }
  )
  cut_buffer.suggestName(cut_combined_decoupled_a_b_sub_in.circuitName + s"_dataCut${params.adderTreeDelay}")
  cut_combined_decoupled_a_b_sub_in <> cut_buffer.io.in
  cut_buffer.io.out <> cut_combined_decoupled_a_b_sub_out

  val a_after_cut   = Wire(Decoupled(UInt(params.arrayInputAWidth.W)))
  val b_after_cut   = Wire(Decoupled(UInt(params.arrayInputBWidth.W)))
  val sub_after_cut = Wire(Decoupled(UInt(params.configWidth.W)))

  a_after_cut.bits  := cut_combined_decoupled_a_b_sub_out.bits(
    params.arrayInputAWidth + params.arrayInputBWidth + params.configWidth - 1,
    params.arrayInputBWidth + params.configWidth
  )
  a_after_cut.valid := cut_combined_decoupled_a_b_sub_out.valid

  b_after_cut.bits  := cut_combined_decoupled_a_b_sub_out.bits(
    params.arrayInputBWidth + params.configWidth - 1,
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
        parallelWidth  = params.arrayInputCWidth,
        serialWidth    = params.serialInputCDataWidth,
        earlyTerminate = true
      )
    )
  )

  // D32 parallel to serial converter
  val D_p2s = Module(
    new ParallelToSerial(
      ParallelToSerialParams(
        parallelWidth  = params.arrayOutputDWidth,
        serialWidth    = params.serialOutputDDataWidth,
        earlyTerminate = true
      )
    )
  )
  require(params.serialInputCDataWidth == params.serialOutputDDataWidth)
  require(params.arrayInputCWidth == params.arrayOutputDWidth)

  // Design-time check to ensure real bandwidth is divisible by serialization width
  params.arrayDim.zipWithIndex.foreach { case (shapes, dataTypeIdx) =>
    shapes.zipWithIndex.foreach { case (dim, dimIdx) =>
      val outputDElemWidth = params.outputDElemWidth(dataTypeIdx)
      val realBandwidth    = dim(0) * dim(2) * outputDElemWidth
      require(
        if (realBandwidth > params.serialOutputDDataWidth) realBandwidth % params.serialOutputDDataWidth == 0 else true,
        s"Invalid config: real bandwidth ($realBandwidth) not divisible by serialOutputDDataWidth (${params.serialOutputDDataWidth}) " +
          s"at dataTypeIdx=$dataTypeIdx, dimIdx=$dimIdx"
      )
    }
  }

  val runTimeInputCBandWidthFactor = (realBandWidth(
    csrReg.arrayCfg.dataTypeCfg,
    csrReg.arrayCfg.arrayShapeCfg,
    inputCElemWidthRom
  ) / params.serialInputCDataWidth.U)

  val input_serial_factor =
    Mux(
      params.arrayInputCWidth.U <= params.serialInputCDataWidth.U,
      1.U,
      Mux(
        runTimeInputCBandWidthFactor === 0.U,
        1.U,
        runTimeInputCBandWidthFactor
      )
    )

  C_s2p.io.terminate_factor.get := input_serial_factor

  D_p2s.io.terminate_factor.get := output_serial_factor

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
  array.io.ctrl.dataTypeCfg   := csrReg.arrayCfg.dataTypeCfg
  array.io.ctrl.accAddExtIn   := accAddExtIn
  array.io.ctrl.accClear      := computation_finish

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
