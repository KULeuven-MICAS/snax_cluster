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
  val in_subtraction = Flipped(DecoupledIO(UInt(params.configWidth.W)))
}

// control io
class SpatialArrayCtrlIO(params: SpatialArrayParam) extends Bundle {
  val arrayShapeCfg = Input(UInt(params.configWidth.W))
  val dataTypeCfg   = Input(UInt(params.configWidth.W))
  val accAddExtIn   = Input(Bool())
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

  // Synchronize and pipeline in_c to match the multiplier + register stage latency
  val in_c_sync = Wire(Decoupled(chiselTypeOf(io.array_data.in_c.bits)))
  in_c_sync.bits := io.array_data.in_c.bits
  // The actual valid/ready logic for in_c_sync is handled in the handshake section later

  val in_c_pipe = Wire(Decoupled(chiselTypeOf(io.array_data.in_c.bits)))
  in_c_sync -|> in_c_pipe

  val inputC = params.arrayDim.zipWithIndex.map { case (dims, dataTypeIdx) =>
    dims.map(dim => {
      dataForwardN(
        params.multiplierNum(dataTypeIdx),
        params.inputTypeC(dataTypeIdx).width,
        // Mu, Nu, 1
        Seq(dim(0), dim(2), 1),
        // stride_Mu, stride_Nu, stride_Ku
        Seq(dim(2), 1, 0),
        in_c_pipe.bits
      )
    })
  }

  val dimRom = VecInit(params.arrayDim.map { twoD =>
    VecInit(twoD.map { oneD =>
      VecInit(oneD.map(_.U(params.configWidth.W)))
    })
  })

  def realUnrollFactorProd(
    dataTypeIdx: UInt,
    dimIdx:      UInt
  ) = {
    val dim = dimRom(dataTypeIdx)(dimIdx)
    dim(0) * dim(1) * dim(2) // Mu * Nu * Ku
  }

  val runTimeUnrollFactorProd = realUnrollFactorProd(io.ctrl.dataTypeCfg, io.ctrl.arrayShapeCfg)

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
    val muls         = multipliers(dataTypeIdx)
    val tree         = adderTree(dataTypeIdx)
    val output_bits  = VecInit(muls.map(_.io.out.bits))
    val output_valid = muls.map(_.io.out.valid).reduce(_ && _)

    val muls_out_data =
      Wire(Decoupled(Vec(params.multiplierNum(dataTypeIdx), UInt(params.inputTypeC(dataTypeIdx).width.W))))
    muls_out_data.bits  := output_bits
    muls_out_data.valid := output_valid
    // The multipliers' ready signal comes from the pipeline register (-|>).
    muls.foreach(_.io.out.ready := muls_out_data.ready)

    // Use the -|> operator to insert a pipeline register
    muls_out_data -|> tree.io.in
  }

  // adder tree runtime configuration
  adderTree.foreach(_.io.cfg := io.ctrl.arrayShapeCfg)

  val accumulators = (0 until params.inputTypeA.length).map(dataTypeIdx =>
    Module(
      new Accumulator(
        params.outputTypeD(dataTypeIdx),
        params.outputTypeD(dataTypeIdx),
        params.multiplierNum(dataTypeIdx)
      )
    )
  )

  // connect adder tree output to accumulators
  // and inputC to accumulators
  accumulators.zipWithIndex.foreach { case (acc, dataTypeIdx) =>
    // the accumulator input1 is from the adder tree
    acc.io.in1.bits                     := adderTree(dataTypeIdx).io.out.bits
    acc.io.in1.valid                    := adderTree(dataTypeIdx).io.out.valid
    // The adder tree's ready signal comes from the accumulator's inputReady, considering both the input1 and input2 are ready in different cases
    adderTree(dataTypeIdx).io.out.ready := acc.io.inputReady

    // the accumulator input2 is from the inputC
    acc.io.in2.bits := MuxLookup(
      io.ctrl.arrayShapeCfg,
      inputC(dataTypeIdx)(0)
    )(
      (0 until params.arrayDim(dataTypeIdx).length).map(j => j.U -> inputC(dataTypeIdx)(j))
    )
  }

  // handle the control signals for accumulators
  // The in2 valid should come from the pipelined in_c
  accumulators.foreach(_.io.in2.valid := in_c_pipe.valid)
  accumulators.foreach(_.io.accAddExtIn := io.ctrl.accAddExtIn)
  accumulators.foreach(_.io.out.ready := io.array_data.out_d.ready)

  // Connect the ready signal for the in_c pipeline to the accumulators' in2 ready
  in_c_pipe.ready := MuxLookup(
    io.ctrl.dataTypeCfg,
    accumulators(0).io.in2.ready
  )(
    (0 until params.arrayDim.length).map(dataTypeIdx => dataTypeIdx.U -> accumulators(dataTypeIdx).io.in2.ready)
  )

  (0 until params.inputTypeA.length).foreach { dataTypeIdx =>
    (0 until params.multiplierNum(dataTypeIdx)).foreach { accIdx =>
      accumulators(dataTypeIdx).io.enable(
        accIdx
      ) := (io.ctrl.dataTypeCfg === dataTypeIdx.U && accIdx.U < runTimeUnrollFactorProd)
    }
  }

  // The multipliers' ready signal comes from its pipeline register (-|>)
  val muls_ready = MuxLookup(
    io.ctrl.dataTypeCfg,
    multipliers(0)(0).io.in.ready
  )(
    (0 until params.arrayDim.length).map(dataTypeIdx => dataTypeIdx.U -> multipliers(dataTypeIdx)(0).io.in.ready)
  )

  // Top-level input synchronization
  // A, B and C (if enabled) must fire together to ensure the input wave enters the pipeline correctly.
  val in_c_active  = io.ctrl.accAddExtIn
  val common_valid = io.array_data.in_a.valid && io.array_data.in_b.valid && (io.array_data.in_c.valid || !in_c_active)
  val common_ready = muls_ready               && (in_c_sync.ready || !in_c_active)

  io.array_data.in_a.ready := io.array_data.in_b.valid && (io.array_data.in_c.valid || !in_c_active) && common_ready
  io.array_data.in_b.ready := io.array_data.in_a.valid && (io.array_data.in_c.valid || !in_c_active) && common_ready
  io.array_data.in_c.ready := io.array_data.in_a.valid && io.array_data.in_b.valid                   && common_ready

  // Drive the valid signals for the first stage
  multipliers.foreach(_.foreach(_.io.in.valid := common_valid))
  in_c_sync.valid := common_valid

  io.array_data.in_subtraction.ready := io.array_data.in_a.ready && io.array_data.in_b.ready

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
