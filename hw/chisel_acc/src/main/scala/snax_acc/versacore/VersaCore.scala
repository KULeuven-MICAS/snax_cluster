// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi <xiaoling.yi@kuleuven.be>
// arrayTop with the fsm controller and the spatial array

package snax_acc.versacore

import chisel3._
import chisel3.util._

import fp_unit._
import snax_acc.utils._

/** VersaCoreCfg is a configuration bundle for the VersaCore module. */
class VersaCoreCfg(params: SpatialArrayParam) extends Bundle {
  val fsmCfg = new Bundle {
    // signal to decide whether to take in new C data for the first computation or not
    // if yes, use the new C, if not, reuse the C stored in the array for accumulation
    val take_in_new_c               = UInt(params.configWidth.W)
    // decide the computation count for one output, K in the GEMM case
    val temporal_accumulation_times = UInt(params.configWidth.W)
    // output_times == 0 means no output, only used for accumulation or reduction scenarios
    // otherwise, output one data after temporal_accumulation_times computations, output count = M * N
    val output_times                = UInt(params.configWidth.W)
    val subtraction_constant_i      = UInt(params.configWidth.W)
  }

  val arrayCfg = new Bundle {
    // runtime array selection
    val arrayShapeCfg = UInt(params.configWidth.W)
    // runtime data type selection
    val dataTypeCfg   = UInt(params.configWidth.W)
  }
}

/** VersaCoreIO defines the input and output interfaces for the VersaCore module. */
class VersaCoreIO(params: SpatialArrayParam) extends Bundle {
  // data interface
  val versacore_data = new Bundle {
    val in_a  = Flipped(DecoupledIO(UInt(params.arrayInputAWidth.W)))
    val in_b  = Flipped(DecoupledIO(UInt(params.arrayInputBWidth.W)))
    val in_c  = Flipped(DecoupledIO(UInt(params.serialInputCDataWidth.W)))
    val out_d = DecoupledIO(UInt(params.serialOutputDDataWidth.W))
  }

  // control interface
  val ctrl = Flipped(DecoupledIO(new VersaCoreCfg(params)))

  // profiling and status signals
  val busy_o              = Output(Bool())
  val performance_counter = Output(UInt(params.configWidth.W))
}

/** VersaCore is the top-level module for VersaCore. */
class VersaCore(params: SpatialArrayParam) extends Module with RequireAsyncReset {

  val io = IO(new VersaCoreIO(params))

  // -----------------------------------
  // design constraints check
  // -----------------------------------

  if (params.dataflow.length > 1) {
    require(
      params.arrayInputAWidth == params.serialInputADataWidth && params.arrayInputBWidth == params.serialInputBDataWidth && params.arrayInputCWidth == params.serialInputCDataWidth &&
        params.arrayOutputDWidth == params.serialOutputDDataWidth,
      "For multi-temporal-dataflow, the array input/output widths must match the serial input/output data widths."
    )
  }

  // constraints, regardless of the computation bound or bandwidth bound array
  params.arrayDim.zipWithIndex.foreach { case (dims, dataTypeIdx) =>
    dims.foreach { dim =>
      {
        require(dim.length == 3)
        // mac number should be enough to support the computation bound
        require(dim(0) * dim(1) * dim(2) <= params.multiplierNum(dataTypeIdx))
        // arrayInputAWidth should be enough to support the bandwidth bound
        require(
          params.arrayInputAWidth        >= dim(0) * dim(1) * params.inputTypeA(dataTypeIdx).width
        )
        // arrayInputBWidth should be enough to support the bandwidth bound
        require(
          params.arrayInputBWidth        >= dim(1) * dim(2) * params.inputTypeB(dataTypeIdx).width
        )
        // arrayInputCWidth should be enough to support the bandwidth bound
        require(
          params.arrayInputCWidth        >= dim(0) * dim(2) * params.inputTypeC(dataTypeIdx).width
        )
        // arrayOutputDWidth should be enough to support the bandwidth bound
        require(params.arrayOutputDWidth >= dim(0) * dim(2) * params.outputTypeD(dataTypeIdx).width)

        // adder tree should be power of 2
        require(isPow2(dim(1)))

      }
    }
  }

  // constraints for the number of spatial array dimensions
  require(
    params.arrayDim.map(_.length).sum < 32 && params.arrayDim
      .map(_.length)
      .sum                                 >= 1
  )

  require(
    params.inputTypeA.length == params.multiplierNum.length    &&
      params.inputTypeB.length == params.multiplierNum.length  &&
      params.inputTypeC.length == params.multiplierNum.length  &&
      params.inputTypeC.length == params.multiplierNum.length  &&
      params.outputTypeD.length == params.multiplierNum.length &&
      params.arrayDim.length == params.multiplierNum.length,
    "All data type related parameters should have the same length"
  )

  val dimRom = VecInit(params.arrayDim.map { twoD =>
    VecInit(twoD.map { oneD =>
      VecInit(oneD.map(_.U(params.configWidth.W)))
    })
  })

  // -----------------------------------
  // state machine starts
  // -----------------------------------

  // State declaration
  val sIDLE :: sBUSY :: Nil = Enum(2)
  val cstate                = RegInit(sIDLE)
  val nstate                = WireInit(sIDLE)

  // signals for state transition
  val config_fire      = WireInit(0.B)
  val versacore_finish = WireInit(0.B)

  val zeroLoopBoundCase = io.ctrl.bits.fsmCfg.temporal_accumulation_times === 0.U

  // Changing states
  cstate := nstate

  chisel3.dontTouch(cstate)
  switch(cstate) {
    is(sIDLE) {
      when(config_fire) {
        nstate := sBUSY
      }.otherwise {
        nstate := sIDLE
      }
    }
    is(sBUSY) {
      when(versacore_finish) {
        nstate := sIDLE
      }.otherwise {
        nstate := sBUSY
      }
    }
  }

  config_fire   := io.ctrl.fire && cstate === sIDLE
  io.ctrl.ready := cstate === sIDLE

  val csrReg = RegInit(0.U.asTypeOf(new VersaCoreCfg(params)))

  // Store the configurations when config valid
  when(config_fire) {
    csrReg.fsmCfg.take_in_new_c               := io.ctrl.bits.fsmCfg.take_in_new_c
    csrReg.fsmCfg.temporal_accumulation_times := io.ctrl.bits.fsmCfg.temporal_accumulation_times
    csrReg.fsmCfg.output_times                := io.ctrl.bits.fsmCfg.output_times
    when(!zeroLoopBoundCase) {}.otherwise {
      assert(
        io.ctrl.bits.fsmCfg.temporal_accumulation_times =/= 0.U,
        " temporal_accumulation_times == 0, invalid configuration!"
      )
    }
    csrReg.fsmCfg.subtraction_constant_i      := io.ctrl.bits.fsmCfg.subtraction_constant_i
    csrReg.arrayCfg.arrayShapeCfg             := io.ctrl.bits.arrayCfg.arrayShapeCfg
    csrReg.arrayCfg.dataTypeCfg               := io.ctrl.bits.arrayCfg.dataTypeCfg
  }

  // -----------------------------------
  // state machine ends
  // -----------------------------------

  // -----------------------------------
  // runtime data width decoding
  // -----------------------------------

  // data serial to parallel converters for input A and B
  // only used with a single input or weight stationary
  // the serial factor also dynamically calculated based on the run-time configuration
  val A_s2p = Module(
    new SerialToParallel(
      ParallelAndSerialConverterParams(
        parallelWidth           = params.arrayInputAWidth,
        serialWidth             = params.serialInputADataWidth,
        earlyTerminate          = true,
        allowedTerminateFactors = Seq(1)
      )
    )
  )

  val B_s2p = Module(
    new SerialToParallel(
      ParallelAndSerialConverterParams(
        parallelWidth           = params.arrayInputBWidth,
        serialWidth             = params.serialInputBDataWidth,
        earlyTerminate          = true,
        allowedTerminateFactors = Seq(1)
      )
    )
  )

  // TODO: a single input or weight stationary are not tested, but should be valid
  // so for now we require the input widths to be equal, e.g., no serialization
  require(params.serialInputADataWidth == params.arrayInputAWidth)
  require(params.serialInputBDataWidth == params.arrayInputBWidth)

  A_s2p.io.in <> io.versacore_data.in_a
  A_s2p.io.counter_value_reset := versacore_finish
  A_s2p.io.is_busy_cstate      := cstate === sBUSY

  B_s2p.io.in <> io.versacore_data.in_b
  B_s2p.io.counter_value_reset := versacore_finish
  B_s2p.io.is_busy_cstate      := cstate === sBUSY

  // dynamically calculate the serial factor for input A and B
  // based on the run-time configuration
  def real_A_BandWidth(
    dataTypeIdx:  UInt,
    dimIdx:       UInt,
    elemWidthSeq: Vec[UInt]
  ) = {
    val dim = dimRom(dataTypeIdx)(dimIdx)
    dim(0) * dim(1) * elemWidthSeq(dataTypeIdx)
  }

  val inputAElemWidthRom = VecInit(params.inputTypeA.map(_.width.U(params.configWidth.W)))

  val runTimeInputABandWidthFactor = (real_A_BandWidth(
    csrReg.arrayCfg.dataTypeCfg,
    csrReg.arrayCfg.arrayShapeCfg,
    inputAElemWidthRom
  ) / params.serialInputADataWidth.U)

  val input_a_serial_factor =
    Mux(
      params.arrayInputAWidth.U <= params.serialInputADataWidth.U,
      1.U,
      Mux(
        runTimeInputABandWidthFactor === 0.U,
        1.U,
        runTimeInputABandWidthFactor
      )
    )
  A_s2p.io.terminate_factor.get := input_a_serial_factor

  def real_B_BandWidth(
    dataTypeIdx:  UInt,
    dimIdx:       UInt,
    elemWidthSeq: Vec[UInt]
  ) = {
    val dim = dimRom(dataTypeIdx)(dimIdx)
    dim(1) * dim(2) * elemWidthSeq(dataTypeIdx)
  }

  val inputBElemWidthRom = VecInit(params.inputTypeB.map(_.width.U(params.configWidth.W)))

  val runTimeInputBBandWidthFactor = (real_B_BandWidth(
    csrReg.arrayCfg.dataTypeCfg,
    csrReg.arrayCfg.arrayShapeCfg,
    inputBElemWidthRom
  ) / params.serialInputBDataWidth.U)

  val input_b_serial_factor =
    Mux(
      params.arrayInputBWidth.U <= params.serialInputBDataWidth.U,
      1.U,
      Mux(
        runTimeInputBBandWidthFactor === 0.U,
        1.U,
        runTimeInputBBandWidthFactor
      )
    )
  B_s2p.io.terminate_factor.get := input_b_serial_factor

  // -----------------------------------
  // serial_parallel C/D data converters starts
  // ---------------------------------
  // Max ratios for the converters
  val ratioC = params.arrayInputCWidth / params.serialInputCDataWidth
  val ratioD = params.arrayOutputDWidth / params.serialOutputDDataWidth

  // Allowed terminate factors for C (SerialToParallel)
  // Adjust `inputTypeC` to the actual type array you have for C.
  val allowedTerminateFactorsC: Seq[Int] = {
    val perShapeFactors =
      params.arrayDim.zipWithIndex.flatMap { case (shapes, dataTypeIdx) =>
        val inputTypeC = params.inputTypeC(dataTypeIdx) // or reuse outputTypeD if appropriate
        shapes.map { dim =>
          val realBandwidth = dim(0) * dim(2) * inputTypeC.width
          val words         = math.max(1, realBandwidth / params.serialInputCDataWidth)

          require(
            words <= ratioC,
            s"Computed terminate factor $words exceeds max ratio $ratioC " +
              s"for C at dataTypeIdx=$dataTypeIdx, dim=$dim"
          )

          words
        }
      }

    (perShapeFactors :+ ratioC).distinct.sorted
  }

  // Allowed terminate factors for D (ParallelToSerial)
  val allowedTerminateFactorsD: Seq[Int] = {
    val perShapeFactors =
      params.arrayDim.zipWithIndex.flatMap { case (shapes, dataTypeIdx) =>
        val outputTypeD = params.outputTypeD(dataTypeIdx)
        shapes.map { dim =>
          val realBandwidth = dim(0) * dim(2) * outputTypeD.width
          // you already ensured divisibility when > serialOutputDDataWidth
          val words         = math.max(1, realBandwidth / params.serialOutputDDataWidth)

          require(
            words <= ratioD,
            s"Computed terminate factor $words exceeds max ratio $ratioD " +
              s"for D at dataTypeIdx=$dataTypeIdx, dim=$dim"
          )

          words
        }
      }

    // Include the full ratio as well, and deduplicate/sort for sanity
    (perShapeFactors :+ ratioD).distinct.sorted
  }

  // C32 serial to parallel converter
  val C_s2p = Module(
    new SerialToParallel(
      ParallelAndSerialConverterParams(
        parallelWidth           = params.arrayInputCWidth,
        serialWidth             = params.serialInputCDataWidth,
        earlyTerminate          = true,
        allowedTerminateFactors = allowedTerminateFactorsC
      )
    )
  )

  // D32 parallel to serial converter
  val D_p2s = Module(
    new ParallelToSerial(
      ParallelAndSerialConverterParams(
        parallelWidth           = params.arrayOutputDWidth,
        serialWidth             = params.serialOutputDDataWidth,
        earlyTerminate          = true,
        allowedTerminateFactors = allowedTerminateFactorsD
      )
    )
  )
  require(params.serialInputCDataWidth == params.serialOutputDDataWidth)
  require(params.arrayInputCWidth == params.arrayOutputDWidth)

  // Design-time check to ensure real C bandwidth is divisible by serialization width
  params.arrayDim.zipWithIndex.foreach { case (shapes, dataTypeIdx) =>
    shapes.zipWithIndex.foreach { case (dim, dimIdx) =>
      val inputTypeC    = params.inputTypeC(dataTypeIdx)
      val realBandwidth = dim(0) * dim(2) * inputTypeC.width
      require(
        if (realBandwidth > params.serialInputCDataWidth) realBandwidth % params.serialInputCDataWidth == 0 else true,
        s"Invalid config: real C bandwidth ($realBandwidth) not divisible by serialInputCDataWidth (${params.serialInputCDataWidth}) " +
          s"at dataTypeIdx=$dataTypeIdx, dimIdx=$dimIdx"
      )
    }
  }

  // Design-time check to ensure real D bandwidth is divisible by serialization width
  params.arrayDim.zipWithIndex.foreach { case (shapes, dataTypeIdx) =>
    shapes.zipWithIndex.foreach { case (dim, dimIdx) =>
      val outputTypeD   = params.outputTypeD(dataTypeIdx)
      val realBandwidth = dim(0) * dim(2) * outputTypeD.width
      require(
        if (realBandwidth > params.serialOutputDDataWidth) realBandwidth % params.serialOutputDDataWidth == 0 else true,
        s"Invalid config: real bandwidth ($realBandwidth) not divisible by serialOutputDDataWidth (${params.serialOutputDDataWidth}) " +
          s"at dataTypeIdx=$dataTypeIdx, dimIdx=$dimIdx"
      )
    }
  }

  val inputCElemWidthRom = VecInit(params.inputTypeC.map(_.width.U(params.configWidth.W)))

  val runTimeInputCBandWidthFactor = (realCDBandWidth(
    csrReg.arrayCfg.dataTypeCfg,
    csrReg.arrayCfg.arrayShapeCfg,
    inputCElemWidthRom
  ) / params.serialInputCDataWidth.U)

  val input_c_serial_factor =
    Mux(
      params.arrayInputCWidth.U <= params.serialInputCDataWidth.U,
      1.U,
      Mux(
        runTimeInputCBandWidthFactor === 0.U,
        1.U,
        runTimeInputCBandWidthFactor
      )
    )

  // Calculate the run-time output serial factor based on the configuration
  // (how many cycles it to output one data)
  val outPutDWidthRom = VecInit(params.outputTypeD.map(_.width.U(params.configWidth.W)))

  val runTimeOutputBandWidthFactor = (realCDBandWidth(
    csrReg.arrayCfg.dataTypeCfg,
    csrReg.arrayCfg.arrayShapeCfg,
    outPutDWidthRom
  ) / params.serialOutputDDataWidth.U)

  val output_d_serial_factor =
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

  C_s2p.io.terminate_factor.get := input_c_serial_factor
  C_s2p.io.counter_value_reset  := versacore_finish
  C_s2p.io.is_busy_cstate       := cstate === sBUSY

  D_p2s.io.terminate_factor.get := output_d_serial_factor
  D_p2s.io.counter_value_reset  := versacore_finish
  D_p2s.io.is_busy_cstate       := cstate === sBUSY

  io.versacore_data.in_c <> C_s2p.io.in
  io.versacore_data.out_d <> D_p2s.io.out

  // ------------------------------------
  // serial_parallel data converters ends
  // ------------------------------------

  // ------------------------------------
  // array instance and data handshake signal connections starts
  // ------------------------------------
  val array = Module(new SpatialArray(params))

  // array accAddExtIn control signal
  val accAddExtIn        = WireInit(0.B)
  val computeFireCounter = Module(new BasicCounter(params.configWidth, hasCeil = true, nameTag = "computeFireCounter"))
  computeFireCounter.io.ceilOpt.get := csrReg.fsmCfg.temporal_accumulation_times

  computeFireCounter.io.tick  := array.io.ctrl.computeFire && cstate === sBUSY
  computeFireCounter.io.reset := versacore_finish

  accAddExtIn := computeFireCounter.io.value === 0.U && csrReg.fsmCfg.take_in_new_c === 1.U && cstate === sBUSY

  // array ctrl signals
  array.io.ctrl.arrayShapeCfg  := csrReg.arrayCfg.arrayShapeCfg
  array.io.ctrl.dataTypeCfg    := csrReg.arrayCfg.dataTypeCfg
  array.io.ctrl.accAddExtIn    := accAddExtIn
  array.io.ctrl.cstate_is_busy := cstate === sBUSY

  // array data signals
  array.io.array_data.in_a <> A_s2p.io.out
  array.io.array_data.in_b <> B_s2p.io.out

  array.io.array_data.in_c <> C_s2p.io.out

  array.io.array_data.in_subtraction := csrReg.fsmCfg.subtraction_constant_i

  // array d_ready considering output stationary
  val dOutputValidCounter = Module(
    new BasicCounter(params.configWidth, hasCeil = true, nameTag = "dOutputValidCounter")
  )
  dOutputValidCounter.io.ceilOpt.get := csrReg.fsmCfg.temporal_accumulation_times
  dOutputValidCounter.io.tick  := array.io.array_data.out_d.fire && cstate === sBUSY
  dOutputValidCounter.io.reset := versacore_finish

  // array output data to the D_p2s converter
  D_p2s.io.in.bits                := array.io.array_data.out_d.bits
  // output_times == 0 means no output
  // If output_times is 0, we need to ensure that the valid signal is not asserted
  // othwerwise, output one valid signal after temporal_accumulation_times computations
  when(csrReg.fsmCfg.output_times === 0.U) {
    D_p2s.io.in.valid := false.B
  }.otherwise {
    D_p2s.io.in.valid := array.io.array_data.out_d.valid && cstate === sBUSY && dOutputValidCounter.io.value === (csrReg.fsmCfg.temporal_accumulation_times - 1.U)
  }
  array.io.array_data.out_d.ready := Mux(D_p2s.io.in.valid, D_p2s.io.in.ready, true.B)

  // ------------------------------------
  // array instance and data handshake signal connections ends
  // ------------------------------------

  // profiling and status signals
  val performance_counter = RegInit(0.U(params.configWidth.W))

  when(cstate === sBUSY) {
    performance_counter := performance_counter + 1.U
  }.elsewhen(config_fire) {
    performance_counter := 0.U
  }

  // output control signals for read-only csrs
  io.performance_counter := performance_counter

  def realCDBandWidth(
    dataTypeIdx:  UInt,
    dimIdx:       UInt,
    elemWidthSeq: Vec[UInt]
  ) = {
    val dim = dimRom(dataTypeIdx)(dimIdx)
    dim(0) * dim(2) * elemWidthSeq(dataTypeIdx)
  }
  // counter for output data count
  val dOutputCounter = Module(new BasicCounter(params.configWidth, hasCeil = false, nameTag = "dOutputCounter"))

  // all number of counts that the data needs to be outputted
  dOutputCounter.io.tick  := io.versacore_data.out_d.fire && cstate === sBUSY
  dOutputCounter.io.reset := versacore_finish

  val output_finish =
    (dOutputCounter.io.value === csrReg.fsmCfg.output_times * output_d_serial_factor) && cstate === sBUSY
  val computation_finish = WireInit(0.B)
  // if no output, computation finish depends on the computeFireCounter only
  when(csrReg.fsmCfg.output_times === 0.U && cstate === sBUSY) {
    computation_finish := (computeFireCounter.io.lastVal) && cstate === sBUSY
  }.otherwise {
    computation_finish := output_finish
  }

  // all the data is outputted means the computation is finished
  versacore_finish := computation_finish && output_finish

  io.busy_o := cstate =/= sIDLE
}

object VersaCoreEmitter extends App {
  emitVerilog(
    new VersaCore(SpatialArrayParam()),
    Array("--target-dir", "generated/versacore")
  )
}

object VersaCoreEmitterFloat16Int4 extends App {
  val FP16Int4Array_Param = SpatialArrayParam(
    multiplierNum          = Seq(8),
    inputTypeA             = Seq(FP16),
    inputTypeB             = Seq(Int4),
    inputTypeC             = Seq(FP32),
    outputTypeD            = Seq(FP32),
    arrayInputAWidth       = 64,
    arrayInputBWidth       = 16,
    arrayInputCWidth       = 128,
    arrayOutputDWidth      = 128,
    serialInputADataWidth  = 64,
    serialInputBDataWidth  = 16,
    serialInputCDataWidth  = 128,
    serialOutputDDataWidth = 128,
    arrayDim               = Seq(Seq(Seq(2, 2, 2)))
  )
  emitVerilog(
    new VersaCore(FP16Int4Array_Param),
    Array("--target-dir", "generated/versacore")
  )
}

object VersaCoreEmitterFloat16Float16 extends App {
  val FP16Float16Array_Param = SpatialArrayParam(
    multiplierNum          = Seq(8),
    inputTypeA             = Seq(FP16),
    inputTypeB             = Seq(FP16),
    inputTypeC             = Seq(FP32),
    outputTypeD            = Seq(FP32),
    arrayInputAWidth       = 64,
    arrayInputBWidth       = 64,
    arrayInputCWidth       = 128,
    arrayOutputDWidth      = 128,
    serialInputADataWidth  = 64,
    serialInputBDataWidth  = 64,
    serialInputCDataWidth  = 128,
    serialOutputDDataWidth = 128,
    arrayDim               = Seq(Seq(Seq(2, 2, 2)))
  )
  emitVerilog(
    new VersaCore(FP16Float16Array_Param),
    Array("--target-dir", "generated/versacore")
  )
}
