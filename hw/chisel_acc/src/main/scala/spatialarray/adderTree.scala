package snax_acc.spatialarray

import chisel3._
import chisel3.util._

class AdderTree[T <: Data with Num[T]](
    val inputType: T,
    val outputType: T,
    val numElements: Int,
    val groupSizes: Seq[Int]
) extends Module {
  require(
    isPow2(numElements),
    "numElements must be a power of 2"
  ) // Ensure valid size
  groupSizes.foreach(size =>
    require(isPow2(size), "groupSizes must be a power of 2")
  ) // Ensure valid size
  require(
    groupSizes.length < 32 && groupSizes.length >= 1,
    "groupSizes must be less than 32 and greater than 0"
  )

  val io = IO(new Bundle {
    val in = Input(Vec(numElements, inputType))
    val out = Output(Vec(numElements, outputType))
    val cfg = Input(UInt(log2Ceil(groupSizes.length + 1).W))
  })

  // adder tree initialization
  val maxGroupSize = groupSizes.max
  val treeDepth = log2Ceil(maxGroupSize)
  val layers = Wire(Vec(treeDepth + 1, Vec(numElements, outputType)))

  layers.map(_.map(_ := 0.U.asTypeOf(outputType)))
  layers(0) := io.in.map(_.asTypeOf(outputType))

  // Generate adder tree layers
  for (d <- 0 until treeDepth) {
    val step = 1
    for (i <- 0 until numElements by (2 * step)) {
      layers(d + 1)(i / 2) := layers(d)(i) + layers(d)(i + step)
    }
  }

  // Generate multiple adder tree outputs based on groupSizes
  val adderResults = groupSizes.map(size => (layers(log2Ceil(size))))

  // Mux output based on cfg
  io.out := MuxLookup(io.cfg, adderResults(0))(
    (0 until groupSizes.length).map(i => (i).U -> adderResults(i))
  )
}

object AdderTreeEmitterUInt extends App {
  emitVerilog(
    new AdderTree(UInt(8.W), UInt(9.W), 8, Seq(1, 2, 4)),
    Array("--target-dir", "generated/SpatialArray")
  )
}

object AdderTreeEmitterSInt extends App {
  emitVerilog(
    new AdderTree(SInt(16.W), SInt(32.W), 1024, Seq(1, 2, 8)),
    Array("--target-dir", "generated/SpatialArray")
  )
}
