// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Xiaoling Yi <xiaoling.yi@kuleuven.be>

package snax_acc.versacore

import chisel3._
import chisel3.util._
import snax_acc.utils.DecoupledCut._

import fp_unit._

// data io
class SpatialArrayDataIO(params: SpatialArrayParam) extends Bundle {
  val in_a           = Flipped(DecoupledIO(UInt(params.arrayInputAWidth.W)))
  val in_b           = Flipped(DecoupledIO(UInt(params.arrayInputBWidth.W)))
  val in_c           = Flipped(DecoupledIO(UInt(params.arrayInputCWidth.W)))
  val out_d          = DecoupledIO(UInt(params.arrayOutputDWidth.W))
  val in_subtraction = Input(UInt(params.configWidth.W))
}

// control io
class SpatialArrayCtrlIO(params: SpatialArrayParam) extends Bundle {
  val arrayShapeCfg  = Input(UInt(params.configWidth.W))
  val dataTypeCfg    = Input(UInt(params.configWidth.W))
  // Two stages need to know about the external C, and they need to know at different times.
  // accAddExtInInput gates the input side: it says the pass now being ACCEPTED is the first
  // of an output block, so one C word is taken from the port and buffered. accAddExtIn gates
  // the accumulator: it says the pass now RETIRING is that same first pass, so C is added
  // instead of the running accumulator. They are generated from separate counters because a
  // pass is accepted cycles before it retires; driving both from the retire-side counter
  // holds the input gate open across several accepted passes and swallows extra C words.
  val accAddExtInInput = Input(Bool())
  val accAddExtIn      = Input(Bool())
  // The third case: no C at all. Then nothing resets the accumulator between output
  // blocks, so the first pass of a block has to be told to start from zero.
  val accClear         = Input(Bool())
  val cstate_is_busy = Input(Bool())
  val computeFire    = Output(Bool())
}

class SpatialArrayIO(params: SpatialArrayParam) extends Bundle {
  val array_data = new SpatialArrayDataIO(params)
  val ctrl       = new SpatialArrayCtrlIO(params)
}

/** SpatialArray is a module that implements a spatial array for parallel computation.
  */
class SpatialArray(params: SpatialArrayParam) extends Module with RequireAsyncReset {

  // io instantiation
  val io = IO(new SpatialArrayIO(params))

  // N-D data feeding network, spatial loop bounds are specified by `dims` and data reuse strides by `strides`, idx 0 is the outermost dimension
  // e.g., for 3D data, dims = Seq(Mu, Nu, Ku) and strides = Seq(stride_Ku, stride_Nu, stride_Mu)
  def dataForwardN(
    multiplierNum: Int,
    elemBits:      Int,
    dims:          Seq[Int],
    strides:       Seq[Int],
    input:         UInt
  ): Vec[UInt] = {
    require(dims.length == strides.length)
    dims.length

    val reshapedData = Wire(Vec(multiplierNum, UInt(elemBits.W)))

    for (i <- 0 until multiplierNum) {
      // Compute multi-dimensional index: idx = [d0, d1, ..., dn]
      def computeMultiIndex(flatIdx: Int, dims: Seq[Int]): Seq[Int] = {
        var remainder = flatIdx
        dims.reverse.map { dim =>
          val idx = remainder % dim
          remainder = remainder / dim
          idx
        }.reverse
      }

      if (i < dims.product) {
        val indices = computeMultiIndex(i, dims) // e.g., [m, n, k]

        // Calculate 1D input index using strides
        val indexExpr = indices
          .zip(strides)
          .map { case (idx, stride) =>
            idx * stride
          }
          .reduce(_ + _) // index = Σ (idx_i * stride_i)

        reshapedData(i) := input(indexExpr * elemBits + elemBits - 1, indexExpr * elemBits)
      } else {
        reshapedData(i) := 0.U
      }
    }

    reshapedData
  }

  val inputA = params.arrayDim.zipWithIndex.map { case (dims, dataTypeIdx) =>
    dims.map(dim => {
      dataForwardN(
        params.multiplierNum(dataTypeIdx),
        params.inputTypeA(dataTypeIdx).width,
        // Mu, Nu, Ku
        Seq(dim(0), dim(2), dim(1)),
        // stride_Mu, stride_Nu, stride_Ku
        Seq(dim(1), 0, 1),
        io.array_data.in_a.bits
      )
    })
  }

  val inputB = params.arrayDim.zipWithIndex.map { case (dims, dataTypeIdx) =>
    dims.map(dim => {
      dataForwardN(
        params.multiplierNum(dataTypeIdx),
        params.inputTypeB(dataTypeIdx).width,
        // Mu, Nu, Ku
        Seq(dim(0), dim(2), dim(1)),
        // stride_Mu, stride_Nu, stride_Ku
        Seq(0, dim(1), 1),
        io.array_data.in_b.bits
      )
    })
  }

  val in_c_before_pipe = Wire(Decoupled(chiselTypeOf(io.array_data.in_c.bits)))
  in_c_before_pipe.bits := io.array_data.in_c.bits
  // The actual valid/ready logic for in_c_before_pipe is handled in the handshake section later

  // Buffer between the input stage and the accumulator holding the C of the output block in
  // flight: pushed once when the block's first pass is accepted, popped once when that pass
  // retires. Exactly one word is in flight, so the depth and the re-arm behaviour of the cut
  // do not affect which C reaches the accumulator. Pushing a copy of the port on every pass
  // instead would make this a fixed-depth delay line of the C port, and then only a cut that
  // holds exactly one item at all times delivers the right word.
  val in_c_after_pipe = Wire(Decoupled(chiselTypeOf(io.array_data.in_c.bits)))
  in_c_before_pipe -||> in_c_after_pipe

  val inputC = params.arrayDim.zipWithIndex.map { case (dims, dataTypeIdx) =>
    dims.map(dim => {
      dataForwardN(
        params.multiplierNum(dataTypeIdx),
        params.inputTypeC(dataTypeIdx).width,
        // Mu, Nu, 1
        Seq(dim(0), dim(2), 1),
        // stride_Mu, stride_Nu, stride_Ku
        Seq(dim(2), 1, 0),
        in_c_after_pipe.bits
      )
    })
  }

  val dimRom = VecInit(params.arrayDim.map { twoD =>
    VecInit(twoD.map { oneD =>
      VecInit(oneD.map(_.U(params.configWidth.W)))
    })
  })

  def multiplierCount(
    dataTypeIdx: UInt,
    dimIdx:      UInt
  ) = {
    val dim = dimRom(dataTypeIdx)(dimIdx)
    dim(0) * dim(1) * dim(2) // Mu * Nu * Ku
  }

  val runTimeMultiplierCount = multiplierCount(io.ctrl.dataTypeCfg, io.ctrl.arrayShapeCfg)

  // instantiate a bunch of multipliers with different data type
  val multipliers = (0 until params.inputTypeA.length).map(dataTypeIdx =>
    Seq.fill(params.multiplierNum(dataTypeIdx))(
      Module(
        new Multiplier(
          params.inputTypeA(dataTypeIdx),
          params.inputTypeB(dataTypeIdx),
          params.inputTypeC(dataTypeIdx)
        )
      )
    )
  )

  // multipliers connection with the output from data feeding network
  (0 until params.inputTypeA.length).foreach(dataTypeIdx =>
    multipliers(dataTypeIdx).zipWithIndex.foreach { case (mul, mulIdx) =>
      mul.io.in.bits.in_a := MuxLookup(
        io.ctrl.arrayShapeCfg,
        inputA(dataTypeIdx)(0)(mulIdx)
      )(
        (0 until params.arrayDim(dataTypeIdx).length).map(j => j.U -> inputA(dataTypeIdx)(j)(mulIdx))
      )
      mul.io.in.bits.in_b := MuxLookup(
        io.ctrl.arrayShapeCfg,
        inputB(dataTypeIdx)(0)(mulIdx)
      )(
        (0 until params.arrayDim(dataTypeIdx).length).map(j => j.U -> inputB(dataTypeIdx)(j)(mulIdx))
      )
    }
  )

  // instantiate adder tree
  val adderTree = (0 until params.inputTypeA.length).map(dataTypeIdx =>
    Module(
      new AdderTree(
        params.inputTypeC(dataTypeIdx),
        params.outputTypeD(dataTypeIdx),
        params.multiplierNum(dataTypeIdx),
        // adderGroupSizes = params.arrayDim(dataTypeIdx).map(_(1)), which describes the spatial reduction dimension
        params.arrayDim(dataTypeIdx).map(_(1))
      )
    )
  )

  // connect output of the multipliers to adder tree
  // insert a register to pipeline the output of the multipliers
  (0 until params.inputTypeA.length).foreach { dataTypeIdx =>
    // a shortcut to get the multipliers and adder tree for the current data type
    val muls = multipliers(dataTypeIdx)
    val tree = adderTree(dataTypeIdx)

    // collect the output bits and valid signals from all multipliers
    val output_bits  = VecInit(muls.map(_.io.out.bits))
    val output_valid = muls.map(_.io.out.valid).reduce(_ && _)

    // create a Decoupled output for the multipliers' results, which will be connected to the adder tree input through a pipeline register (-\>)
    val muls_out_data =
      Wire(Decoupled(Vec(params.multiplierNum(dataTypeIdx), UInt(params.inputTypeC(dataTypeIdx).width.W))))
    muls_out_data.bits  := output_bits
    muls_out_data.valid := output_valid
    // The multipliers' ready signal comes from the pipeline register (-\>).
    muls.foreach(_.io.out.ready := muls_out_data.ready)

    // -\> (DataCut, delay = 1), NOT -|>. This register sits on the per-array-pass datapath,
    // so its acceptance rate IS the array's throughput. -|> is Queue(entries = 1, pipe =
    // false) -- the operator names itself FullCutHalfBandwidth -- and with no pipe bypass a
    // one-entry queue drives enq.ready = !full, accepting on alternate cycles however ready
    // the consumer is. That halves the array: measured 2.00 cycles per pass on a 16x4x16
    // INT8 unrolling where the arithmetic floor is 1.
    //
    // DataCut holds the same single register, so the data path is cut exactly as before and
    // the area is unchanged, but it re-arms whenever the consumer takes the value, which
    // gives full rate. The cost is that ready is combinational through the cut, so the
    // accumulator's backpressure reaches the streamer in one cycle. -||> cuts ready as well,
    // at two entries -- twice this register stage, 1024 lanes x 32 b of it.
    muls_out_data -\> tree.io.in
  }

  // adder tree runtime configuration
  adderTree.foreach(_.io.cfg := io.ctrl.arrayShapeCfg)

  // instantiate accumulators for each data type
  // the number of accumulators is the same as the number of multipliers for that data type
  // and each accumulator can be enabled or disabled based on the runtime multiplier count
  val accumulators = (0 until params.inputTypeA.length).map(dataTypeIdx =>
    Module(
      new Accumulator(
        params.outputTypeD(dataTypeIdx),
        params.outputTypeD(dataTypeIdx),
        params.multiplierNum(dataTypeIdx)
      )
    )
  )

  val accumulatorIn2Ready  = Wire(Vec(params.inputTypeA.length, Bool()))
  val accumulatorAccUpdate = Wire(Vec(params.inputTypeA.length, Bool()))

  // connect adder tree output to accumulators
  // and inputC to accumulators
  accumulators.zipWithIndex.foreach { case (acc, dataTypeIdx) =>
    // ------------------------------------------
    // the accumulator input1 is from the adder tree
    // ------------------------------------------
    acc.io.in1.bits                     := adderTree(dataTypeIdx).io.out.bits
    acc.io.in1.valid                    := adderTree(dataTypeIdx).io.out.valid
    // The adder tree's ready signal comes from the accumulator's inputReady, considering both the input1 and input2 are ready in different cases
    adderTree(dataTypeIdx).io.out.ready := acc.io.inputReady

    // ------------------------------------------
    // the accumulator input2 is from the inputC
    // ------------------------------------------
    acc.io.in2.bits := MuxLookup(
      io.ctrl.arrayShapeCfg,
      inputC(dataTypeIdx)(0)
    )(
      // one dim data
      (0 until params.arrayDim(dataTypeIdx).length).map(j => j.U -> inputC(dataTypeIdx)(j))
    )

    // The in2 valid should come from the pipelined in_c
    acc.io.in2.valid := in_c_after_pipe.valid

    accumulatorIn2Ready(dataTypeIdx)  := acc.io.in2.ready
    accumulatorAccUpdate(dataTypeIdx) := acc.io.accUpdate
  }

  // handle the control signals for accumulators
  accumulators.foreach(_.io.accAddExtIn := io.ctrl.accAddExtIn)
  accumulators.foreach(_.io.accClear := io.ctrl.accClear)
  accumulators.foreach(_.io.out.ready := io.array_data.out_d.ready)

  // enable the accumulators based on the runtime multiplier count
  (0 until params.inputTypeA.length).foreach { dataTypeIdx =>
    (0 until params.multiplierNum(dataTypeIdx)).foreach { mulIdx =>
      accumulators(dataTypeIdx).io.enable(
        mulIdx
      ) := (io.ctrl.dataTypeCfg === dataTypeIdx.U && mulIdx.U < runTimeMultiplierCount && io.ctrl.cstate_is_busy) // Only enable the accumulators corresponding to the active multipliers and when the state is busy
    }
  }

  // The multipliers' ready signal comes from its pipeline register (-\>)
  val muls_ready = MuxLookup(
    io.ctrl.dataTypeCfg,
    multipliers(0)(0).io.in.ready
  )(
    (0 until params.arrayDim.length).map(dataTypeIdx => dataTypeIdx.U -> multipliers(dataTypeIdx)(0).io.in.ready)
  )

  val selectedIn2Ready = MuxLookup(
    io.ctrl.dataTypeCfg,
    accumulatorIn2Ready(0)
  )(
    (0 until params.arrayDim.length).map(dataTypeIdx => dataTypeIdx.U -> accumulatorIn2Ready(dataTypeIdx))
  )

  io.ctrl.computeFire := MuxLookup(
    io.ctrl.dataTypeCfg,
    accumulatorAccUpdate(0)
  )(
    (0 until params.arrayDim.length).map(dataTypeIdx => dataTypeIdx.U -> accumulatorAccUpdate(dataTypeIdx))
  )

  // ---------------------------------------------------
  // Top-level input synchronization
  // ---------------------------------------------------
  // A, B and C (if enabled) must fire together to ensure the input wave enters the pipeline correctly.
  val in_c_active  = io.ctrl.accAddExtInInput
  val common_valid =
    io.array_data.in_a.valid && io.array_data.in_b.valid && (io.array_data.in_c.valid || !in_c_active) && io.ctrl.cstate_is_busy
  val common_accept_data_ready = muls_ready && (in_c_before_pipe.ready || !in_c_active) && io.ctrl.cstate_is_busy

  // sync a and b ready signals for the three inputs based on the common valid and the pipeline ready
  io.array_data.in_a.ready := io.array_data.in_b.valid && (io.array_data.in_c.valid || !in_c_active) && common_accept_data_ready && io.ctrl.cstate_is_busy
  io.array_data.in_b.ready := io.array_data.in_a.valid && (io.array_data.in_c.valid || !in_c_active) && common_accept_data_ready && io.ctrl.cstate_is_busy
  // only takes in c when accAddExtIn is true to accept new c input data
  io.array_data.in_c.ready := io.array_data.in_a.valid && io.array_data.in_b.valid && common_accept_data_ready && io.ctrl.cstate_is_busy && in_c_active

  // Drive the valid signals for the first stage
  multipliers.foreach(_.foreach(_.io.in.valid := common_valid))
  in_c_before_pipe.valid := common_valid && in_c_active
  in_c_after_pipe.ready  := selectedIn2Ready && io.ctrl.accAddExtIn

  // output data and valid signals
  io.array_data.out_d.bits := MuxLookup(
    io.ctrl.dataTypeCfg,
    accumulators(0).io.out.asUInt
  )(
    (0 until params.arrayDim.length).map(dataTypeIdx => dataTypeIdx.U -> accumulators(dataTypeIdx).io.out.bits.asUInt)
  )

  io.array_data.out_d.valid := MuxLookup(
    io.ctrl.dataTypeCfg,
    accumulators(0).io.out.valid
  )(
    (0 until params.arrayDim.length).map(dataTypeIdx => dataTypeIdx.U -> accumulators(dataTypeIdx).io.out.valid)
  )
}

object SpatialArrayEmitter extends App {
  emitVerilog(
    new SpatialArray(SpatialArrayParam()),
    Array("--target-dir", "generated/versacore")
  )

  val params = SpatialArrayParam(
    multiplierNum          = Seq(1024),
    inputTypeA             = Seq(Int8),
    inputTypeB             = Seq(Int8),
    inputTypeC             = Seq(Int8),
    outputTypeD            = Seq(Int32),
    arrayInputAWidth       = 1024,
    arrayInputBWidth       = 8192,
    arrayInputCWidth       = 4096,
    arrayOutputDWidth      = 4096,
    serialInputADataWidth  = 1024,
    serialInputBDataWidth  = 8192,
    serialInputCDataWidth  = 512,
    serialOutputDDataWidth = 512,
    // Mu, Ku, Nu
    arrayDim               = Seq(Seq(Seq(16, 8, 8), Seq(1, 32, 32)))
  )
  emitVerilog(
    new SpatialArray(params),
    Array("--target-dir", "generated/versacore")
  )

}
