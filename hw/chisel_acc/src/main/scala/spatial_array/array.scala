package snax_acc.spatial_array

import chisel3._
import chisel3.util._

class SpatialArrayDataIO(params: SpatialArrayParam) extends Bundle {
  val in_a = Flipped(DecoupledIO(UInt(params.inputAWidth.W)))
  val in_b = Flipped(DecoupledIO(UInt(params.inputBWidth.W)))
  val in_c = Flipped(DecoupledIO(UInt(params.inputCWidth.W)))
  val out_d = DecoupledIO(UInt(params.outputDWidth.W))
}

class SpatialArrayCtrlIO(params: SpatialArrayParam) extends Bundle {
  val spatialArrayCfg = Input(
    UInt(params.configWidth.W)
  )
  val dataTypeCfg = Input(UInt(params.configWidth.W))
}

class SpatialArrayIO(params: SpatialArrayParam) extends Bundle {
  val data = new SpatialArrayDataIO(params)
  val ctrl = new SpatialArrayCtrlIO(params)
}

class SpatialArray(params: SpatialArrayParam)
    extends Module
    with RequireAsyncReset {

  // io instantiation
  val io = IO(new SpatialArrayIO(params))

  // different type of 1D multipliers
  val multipliers = (0 until params.opType.length).map(i =>
    Seq.fill(params.macNum(i))(
      Module(
        new Multiplier(
          params.opType(i),
          params.inputAElemWidth(i),
          params.inputBElemWidth(i),
          params.mulElemWidth(i)
        )
      )
    )
  )

  // constraints, regardless of the computation bound or bandwidth bound array
  params.arrayDim.zipWithIndex.foreach { case (dims, i) =>
    dims.foreach { dim =>
      {
        require(dim.length == 3)
        // mac number should be enough to support the computation bound
        require(dim(0) * dim(1) * dim(2) <= params.macNum(i))
        // inputAWidth should be enough to support the bandwidth bound
        require(
          params.inputAWidth >= dim(0) * dim(1) * params.inputAElemWidth(i)
        )
        // inputBWidth should be enough to support the bandwidth bound
        require(
          params.inputBWidth >= dim(1) * dim(2) * params.inputBElemWidth(i)
        )
        // inputCWidth should be enough to support the bandwidth bound
        require(
          params.inputCWidth >= dim(0) * dim(2) * params.inputCElemWidth(i)
        )
        // outputDWidth should be enough to support the bandwidth bound
        require(params.outputDWidth >= dim(0) * dim(2) * params.outElemWidth(i))

        // adder tree should be power of 2
        require(isPow2(dim(1)))

      }
    }
  }

  require(
    params.arrayDim.map(_.length).sum < 32 && params.arrayDim
      .map(_.length)
      .sum >= 1
  )

  require(
    params.opType.length == params.macNum.length &&
      params.inputAElemWidth.length == params.macNum.length &&
      params.inputBElemWidth.length == params.macNum.length &&
      params.inputCElemWidth.length == params.macNum.length &&
      params.mulElemWidth.length == params.macNum.length &&
      params.outElemWidth.length == params.macNum.length &&
      params.arrayDim.length == params.macNum.length,
    "All data type related parameters should have the same length"
  )

  // data feeding network
  def dataForward(
      macNum: Int,
      elemBits: Int,
      Mu: Int,
      Ku: Int,
      Nu: Int,
      stride_Ku: Int,
      stride_Nu: Int,
      stride_Mu: Int,
      input: UInt
  ) = {
    val data = Wire(Vec(macNum, UInt(elemBits.W)))
    for (i <- 0 until macNum) {
      if (i < Mu * Nu * Ku) {
        val m = i / (Nu * Ku)
        val n = (i % (Nu * Ku)) / Ku
        val k = (i % (Nu * Ku)) % Ku
        val index = k * stride_Ku + n * stride_Nu + m * stride_Mu
        data(i) := input(index * elemBits + elemBits - 1, index * elemBits)
      } else {
        data(i) := 0.U
      }
    }
    data
  }

  val inputA = params.arrayDim.zipWithIndex.map { case (dims, i) =>
    dims.map(dim => {
      dataForward(
        params.macNum(i),
        params.inputAElemWidth(i),
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

  val inputB = params.arrayDim.zipWithIndex.map { case (dims, i) =>
    dims.map(dim => {
      dataForward(
        params.macNum(i),
        params.inputBElemWidth(i),
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

  val inputC = params.arrayDim.zipWithIndex.map { case (dims, i) =>
    dims.map(dim => {
      dataForward(
        params.macNum(i),
        params.inputCElemWidth(i),
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
  (0 until params.opType.length).foreach(i =>
    multipliers(i).zipWithIndex.foreach { case (mul, mulIdx) =>
      mul.io.in_a := MuxLookup(
        io.ctrl.spatialArrayCfg,
        inputA(i)(0)(mulIdx)
      )(
        (0 until params.arrayDim(i).length).map(j =>
          j.U -> inputA(i)(j)(mulIdx)
        )
      )
      mul.io.in_b := MuxLookup(
        io.ctrl.spatialArrayCfg,
        inputB(i)(0)(mulIdx)
      )(
        (0 until params.arrayDim(i).length).map(j =>
          j.U -> inputB(i)(j)(mulIdx)
        )
      )
    }
  )

  // adder tree
  val adderTree = (0 until params.opType.length).map(i =>
    Module(
      new AdderTree(
        params.opType(i),
        params.mulElemWidth(i),
        params.outElemWidth(i),
        params.macNum(i),
        params.arrayDim(i).map(_(1))
      )
    )
  )

  // connect multipliers to adder tree
  multipliers.zipWithIndex.foreach { case (muls, dataTypeIdx) =>
    muls.zipWithIndex.foreach { case (mul, mulIdx) =>
      adderTree(dataTypeIdx).io.in(mulIdx) := mul.io.out_c
    }
  }

  adderTree.foreach(_.io.cfg := io.ctrl.spatialArrayCfg)

  // accumulator

  // output data
  io.data.out_d.bits := MuxLookup(
    io.ctrl.dataTypeCfg,
    adderTree(0).io.out.asUInt
  )(
    (0 until params.arrayDim.length).map(i => i.U -> adderTree(i).io.out.asUInt)
  )

  // ready/valid signals
  io.data.in_a.ready := false.B
  io.data.in_b.ready := false.B
  io.data.in_c.ready := false.B
  io.data.out_d.valid := false.B
}

object SpatialArrayEmitter extends App {
  emitVerilog(
    new SpatialArray(SpatialArrayParam()),
    Array("--target-dir", "generated/SpatialArray")
  )

  // val params = SpatialArrayParam(
  //   opType = Seq(OpType.UIntUIntOp),
  //   macNum = Seq(1024),
  //   inputAElemWidth = Seq(8),
  //   inputBElemWidth = Seq(8),
  //   inputCElemWidth = Seq(8),
  //   mulElemWidth = Seq(16),
  //   outElemWidth = Seq(32),
  //   inputAWidth = 1024,
  //   inputBWidth = 8192,
  //   inputCWidth = 4096,
  //   outputDWidth = 4096,
  //   // Mu, Ku, Nu
  //   arrayDim = Seq(Seq(Seq(16, 8, 8), Seq(1, 32, 32)))
  // )
  // emitVerilog(
  //   new SpatialArray(params),
  //   Array("--target-dir", "generated/SpatialArray")
  // )
}
