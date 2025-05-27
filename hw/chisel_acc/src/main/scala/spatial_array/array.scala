package snax_acc.spatial_array

import chisel3._
import chisel3.util._

class SpatialArrayDataIO(params: SpatialArrayParam) extends Bundle {
  val in_a            = Flipped(DecoupledIO(UInt(params.arrayInputAWidth.W)))
  val in_b            = Flipped(DecoupledIO(UInt(params.arrayInputBWidth.W)))
  val in_c            = Flipped(DecoupledIO(UInt(params.arrayInputCWidth.W)))
  val out_d           = DecoupledIO(UInt(params.arrayOutputDWidth.W))
  val in_substraction = Flipped(DecoupledIO(UInt(params.configWidth.W)))
}

class SpatialArrayCtrlIO(params: SpatialArrayParam) extends Bundle {
  val arrayShapeCfg = Input(UInt(params.configWidth.W))
  val dataTypeCfg   = Input(UInt(params.configWidth.W))
  val accAddExtIn   = Input(Bool())
  val accClear      = Input(Bool())
}

class SpatialArrayIO(params: SpatialArrayParam) extends Bundle {
  val data = new SpatialArrayDataIO(params)
  val ctrl = new SpatialArrayCtrlIO(params)
}

class SpatialArray(params: SpatialArrayParam) extends Module with RequireAsyncReset {

  // io instantiation
  val io = IO(new SpatialArrayIO(params))

  // different type of 1D multipliers
  val multipliers = (0 until params.opType.length).map(dataTypeIdx =>
    Seq.fill(params.macNum(dataTypeIdx))(
      Module(
        new Multiplier(
          params.opType(dataTypeIdx),
          params.inputAElemWidth(dataTypeIdx),
          params.inputBElemWidth(dataTypeIdx),
          params.mulElemWidth(dataTypeIdx)
        )
      )
    )
  )

  // constraints, regardless of the computation bound or bandwidth bound array
  params.arrayDim.zipWithIndex.foreach { case (dims, dataTypeIdx) =>
    dims.foreach { dim =>
      {
        require(dim.length == 3)
        // mac number should be enough to support the computation bound
        require(dim(0) * dim(1) * dim(2) <= params.macNum(dataTypeIdx))
        // arrayInputAWidth should be enough to support the bandwidth bound
        require(
          params.arrayInputAWidth        >= dim(0) * dim(1) * params.inputAElemWidth(dataTypeIdx)
        )
        // arrayInputBWidth should be enough to support the bandwidth bound
        require(
          params.arrayInputBWidth        >= dim(1) * dim(2) * params.inputBElemWidth(dataTypeIdx)
        )
        // arrayInputCWidth should be enough to support the bandwidth bound
        require(
          params.arrayInputCWidth        >= dim(0) * dim(2) * params.inputCElemWidth(dataTypeIdx)
        )
        // arrayOutputDWidth should be enough to support the bandwidth bound
        require(params.arrayOutputDWidth >= dim(0) * dim(2) * params.outputDElemWidth(dataTypeIdx))

        // adder tree should be power of 2
        require(isPow2(dim(1)))

      }
    }
  }

  require(
    params.arrayDim.map(_.length).sum < 32 && params.arrayDim
      .map(_.length)
      .sum                                 >= 1
  )

  require(
    params.opType.length == params.macNum.length             &&
      params.inputAElemWidth.length == params.macNum.length  &&
      params.inputBElemWidth.length == params.macNum.length  &&
      params.inputCElemWidth.length == params.macNum.length  &&
      params.mulElemWidth.length == params.macNum.length     &&
      params.outputDElemWidth.length == params.macNum.length &&
      params.arrayDim.length == params.macNum.length,
    "All data type related parameters should have the same length"
  )

  // data feeding network
  def dataForward3(
    macNum:    Int,
    elemBits:  Int,
    Mu:        Int,
    Ku:        Int,
    Nu:        Int,
    stride_Ku: Int,
    stride_Nu: Int,
    stride_Mu: Int,
    input:     UInt
  ) = {
    val reshapedData = Wire(Vec(macNum, UInt(elemBits.W)))
    for (i <- 0 until macNum) {
      if (i < Mu * Nu * Ku) {
        val m     = i / (Nu * Ku)
        val n     = (i % (Nu * Ku)) / Ku
        val k     = (i % (Nu * Ku)) % Ku
        val index = k * stride_Ku + n * stride_Nu + m * stride_Mu
        reshapedData(i) := input(index * elemBits + elemBits - 1, index * elemBits)
      } else {
        reshapedData(i) := 0.U
      }
    }
    reshapedData
  }

  def dataForwardN(
    macNum:   Int,
    elemBits: Int,
    dims:     Seq[Int], // e.g., Seq(Mu, Nu, Ku)
    strides:  Seq[Int], // e.g., Seq(stride_Mu, stride_Nu, stride_Ku)
    input:    UInt
  ): Vec[UInt] = {
    require(dims.length == strides.length)
    dims.length

    val reshapedData = Wire(Vec(macNum, UInt(elemBits.W)))

    for (i <- 0 until macNum) {
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
      dataForward3(
        params.macNum(dataTypeIdx),
        params.inputAElemWidth(dataTypeIdx),
        // Mu, Ku, Nu
        dim(0),
        dim(1),
        dim(2),
        // stride_Ku, stride_Nu, stride_Mu
        1,
        0,
        dim(1),
        io.data.in_a.bits
      )
    })
  }

  val inputB = params.arrayDim.zipWithIndex.map { case (dims, dataTypeIdx) =>
    dims.map(dim => {
      dataForward3(
        params.macNum(dataTypeIdx),
        params.inputBElemWidth(dataTypeIdx),
        // Mu, Ku, Nu
        dim(0),
        dim(1),
        dim(2),
        // stride_Ku, stride_Nu, stride_Mu
        1,
        dim(1),
        0,
        io.data.in_b.bits
      )
    })
  }

  val inputC = params.arrayDim.zipWithIndex.map { case (dims, dataTypeIdx) =>
    dims.map(dim => {
      dataForward3(
        params.macNum(dataTypeIdx),
        params.inputCElemWidth(dataTypeIdx),
        // Mu, Ku = 1, Nu, only two dimensions
        dim(0),
        1,
        dim(2),
        // stride_Ku, stride_Nu, stride_Mu
        0,
        1,
        dim(2),
        io.data.in_c.bits
      )
    })
  }

  // multipliers
  (0 until params.opType.length).foreach(dataTypeIdx =>
    multipliers(dataTypeIdx).zipWithIndex.foreach { case (mul, mulIdx) =>
      mul.io.in_a := MuxLookup(
        io.ctrl.arrayShapeCfg,
        inputA(dataTypeIdx)(0)(mulIdx)
      )(
        (0 until params.arrayDim(dataTypeIdx).length).map(j => j.U -> inputA(dataTypeIdx)(j)(mulIdx))
      )
      mul.io.in_b := MuxLookup(
        io.ctrl.arrayShapeCfg,
        inputB(dataTypeIdx)(0)(mulIdx)
      )(
        (0 until params.arrayDim(dataTypeIdx).length).map(j => j.U -> inputB(dataTypeIdx)(j)(mulIdx))
      )
    }
  )

  // adder tree
  val adderTree = (0 until params.opType.length).map(dataTypeIdx =>
    Module(
      new AdderTree(
        params.opType(dataTypeIdx),
        params.mulElemWidth(dataTypeIdx),
        params.outputDElemWidth(dataTypeIdx),
        params.macNum(dataTypeIdx),
        params.arrayDim(dataTypeIdx).map(_(1))
      )
    )
  )

  // connect multipliers to adder tree
  multipliers.zipWithIndex.foreach { case (muls, dataTypeIdx) =>
    muls.zipWithIndex.foreach { case (mul, mulIdx) =>
      adderTree(dataTypeIdx).io.in(mulIdx) := mul.io.out_c
    }
  }

  adderTree.foreach(_.io.cfg := io.ctrl.arrayShapeCfg)

  // accumulator
  val accumulators = (0 until params.opType.length).map(dataTypeIdx =>
    Module(
      new Accumulator(
        params.opType(dataTypeIdx),
        params.outputDElemWidth(dataTypeIdx),
        params.outputDElemWidth(dataTypeIdx),
        params.macNum(dataTypeIdx)
      )
    )
  )
  accumulators.zipWithIndex.foreach { case (acc, dataTypeIdx) =>
    acc.io.in1.bits := adderTree(dataTypeIdx).io.out
    acc.io.in2.bits := MuxLookup(
      io.ctrl.arrayShapeCfg,
      inputC(dataTypeIdx)(0)
    )(
      (0 until params.arrayDim(dataTypeIdx).length).map(j => j.U -> inputC(dataTypeIdx)(j))
    )
  }
  accumulators.foreach(_.io.in1.valid := io.data.in_a.valid && io.data.in_b.valid)
  accumulators.foreach(_.io.in2.valid := io.data.in_c.valid)
  accumulators.foreach(_.io.accAddExtIn := io.ctrl.accAddExtIn)
  accumulators.foreach(_.io.accClear := io.ctrl.accClear)
  accumulators.foreach(_.io.out.ready := io.data.out_d.ready)
  (0 until params.opType.length).foreach { dataTypeIdx =>
    accumulators(dataTypeIdx).io.enable := io.ctrl.dataTypeCfg === dataTypeIdx.U
  }

  // input fire signals
  val acc_in1_fire = MuxLookup(
    io.ctrl.dataTypeCfg,
    accumulators(0).io.in1.fire
  )(
    (0 until params.arrayDim.length).map(dataTypeIdx => dataTypeIdx.U -> accumulators(dataTypeIdx).io.in1.fire)
  )
  val acc_in2_fire = MuxLookup(
    io.ctrl.dataTypeCfg,
    accumulators(0).io.in2.fire
  )(
    (0 until params.arrayDim.length).map(dataTypeIdx => dataTypeIdx.U -> accumulators(dataTypeIdx).io.in2.fire)
  )
  io.data.in_a.ready := Mux(io.ctrl.accAddExtIn, acc_in1_fire && acc_in2_fire, acc_in1_fire)
  io.data.in_b.ready := Mux(io.ctrl.accAddExtIn, acc_in1_fire && acc_in2_fire, acc_in1_fire)
  io.data.in_c.ready := Mux(io.ctrl.accAddExtIn, acc_in1_fire && acc_in2_fire, false.B)

  io.data.in_substraction.ready := io.data.in_a.ready && io.data.in_b.ready

  // output data and valid signals
  io.data.out_d.bits := MuxLookup(
    io.ctrl.dataTypeCfg,
    accumulators(0).io.out.asUInt
  )(
    (0 until params.arrayDim.length).map(dataTypeIdx => dataTypeIdx.U -> accumulators(dataTypeIdx).io.out.bits.asUInt)
  )

  io.data.out_d.valid := MuxLookup(
    io.ctrl.dataTypeCfg,
    accumulators(0).io.out.valid
  )(
    (0 until params.arrayDim.length).map(dataTypeIdx => dataTypeIdx.U -> accumulators(dataTypeIdx).io.out.valid)
  )
}

object SpatialArrayEmitter extends App {
  emitVerilog(
    new SpatialArray(SpatialArrayParam()),
    Array("--target-dir", "generated/SpatialArray")
  )

  val params = SpatialArrayParam(
    opType                 = Seq(OpType.UIntUIntOp),
    macNum                 = Seq(1024),
    inputAElemWidth        = Seq(8),
    inputBElemWidth        = Seq(8),
    inputCElemWidth        = Seq(8),
    mulElemWidth           = Seq(16),
    outputDElemWidth       = Seq(32),
    arrayInputAWidth       = 1024,
    arrayInputBWidth       = 8192,
    arrayInputCWidth       = 4096,
    arrayOutputDWidth      = 4096,
    serialInputCDataWidth  = 512,
    serialOutputDDataWidth = 512,
    // Mu, Ku, Nu
    arrayDim               = Seq(Seq(Seq(16, 8, 8), Seq(1, 32, 32)))
  )
  emitVerilog(
    new SpatialArray(params),
    Array("--target-dir", "generated/SpatialArray")
  )

}
