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
  val spatialArrayCfg = Input(UInt(log2Ceil(params.arrayDim.length + 1).W))
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

  // 1D multipliers
  val multipliers = Seq.fill(params.macNum)(
    Module(
      new Multiplier(
        params.opType,
        params.inputAElemWidth,
        params.inputBElemWidth,
        params.mulElemWidth
      )
    )
  )

  // constraints, regardless of the computation bound or bandwidth bound array
  params.arrayDim.map(dim => {
    require(dim.length == 3)
    // mac number should be enough to support the computation bound
    require(dim(0) * dim(1) * dim(2) <= params.macNum)
    // inputAWidth should be enough to support the bandwidth bound
    require(params.inputAWidth >= dim(0) * dim(1) * params.inputAElemWidth)
    // inputBWidth should be enough to support the bandwidth bound
    require(params.inputBWidth >= dim(1) * dim(2) * params.inputBElemWidth)
    // inputCWidth should be enough to support the bandwidth bound
    require(params.inputCWidth >= dim(0) * dim(2) * params.inputCElemWidth)
    // outputDWidth should be enough to support the bandwidth bound
    require(params.outputDWidth >= dim(0) * dim(2) * params.outElemWidth)

    // adder tree should be power of 2
    require(isPow2(dim(1)))

  })

  require(params.arrayDim.length < 32 && params.arrayDim.length >= 1)

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

  val inputA = params.arrayDim.map(dim => {
    dataForward(
      params.macNum,
      params.inputAElemWidth,
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
  val inputB = params.arrayDim.map(dim => {
    dataForward(
      params.macNum,
      params.inputBElemWidth,
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
  val inputC = params.arrayDim.map(dim => {
    dataForward(
      params.macNum,
      params.inputCElemWidth,
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

  // mac tree
  multipliers.zipWithIndex.foreach { case (mac, i) =>
    mac.io.in_a := MuxLookup(
      io.ctrl.spatialArrayCfg,
      inputA(0)(i)
    )(
      (0 until params.arrayDim.length).map(j => j.U -> inputA(j)(i))
    )

    mac.io.in_b := MuxLookup(
      io.ctrl.spatialArrayCfg,
      inputB(0)(i)
    )(
      (0 until params.arrayDim.length).map(j => j.U -> inputB(j)(i))
    )
  }

  // adder tree
  val adderTree = Module(
    new AdderTree(
      params.opType,
      params.mulElemWidth,
      params.outElemWidth,
      params.macNum,
      params.arrayDim.map(dim => dim(1))
    )
  )

  adderTree.io.in := multipliers.map(_.io.out_c)
  adderTree.io.cfg := io.ctrl.spatialArrayCfg

  // accumulator

  // output data
  io.data.out_d.bits := adderTree.io.out.asUInt

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
  val params = SpatialArrayParam(
    opType = OpType.UIntUIntOp,
    macNum = 1024,
    inputAElemWidth = 8,
    inputBElemWidth = 8,
    inputCElemWidth = 8,
    mulElemWidth = 16,
    outElemWidth = 32,
    inputAWidth = 1024,
    inputBWidth = 8192,
    inputCWidth = 4096,
    outputDWidth = 4096,
    // Mu, Ku, Nu
    arrayDim = Seq(Seq(16, 8, 8), Seq(1, 32, 32))
  )
  emitVerilog(
    new SpatialArray(params),
    Array("--target-dir", "generated/SpatialArray")
  )
}
