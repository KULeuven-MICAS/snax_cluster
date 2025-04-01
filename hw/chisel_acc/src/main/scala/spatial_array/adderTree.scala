package snax_acc.spatial_array

import chisel3._
import chisel3.util._

class AdderTree(
    val opType: Int,
    val inputElemWidth: Int,
    val outputElemWidth: Int,
    val numElements: Int,
    val groupSizes: Seq[Int]
) extends Module
    with RequireAsyncReset {
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
  require(
    opType == OpType.UIntUIntOp || opType == OpType.SIntSIntOp,
    "Currenty we only support UIntUIntOp or SIntSIntOp"
  )

  val io = IO(new Bundle {
    val in = Input(Vec(numElements, UInt(inputElemWidth.W)))
    val out = Output(Vec(numElements, UInt(outputElemWidth.W)))
    val cfg = Input(UInt(log2Ceil(groupSizes.length + 1).W))
  })

  // adder tree initialization
  val maxGroupSize = groupSizes.max
  val treeDepth = log2Ceil(maxGroupSize)

  val layers = Wire(
    Vec(treeDepth + 1, Vec(numElements, UInt(outputElemWidth.W)))
  )

  // Initialize the output type based on the operation type
  // For SIntSIntOp, we need to use SInt for the output
  // For UIntUIntOp, we can use UInt for the output
  // TODO: pay attention to other types
  val outputType = if (opType == OpType.SIntSIntOp) {
    SInt(outputElemWidth.W)
  } else {
    UInt(outputElemWidth.W)
  }

  // Initialize all layers to zero
  layers.map(_.map(_ := 0.U.asTypeOf(UInt(outputElemWidth.W))))
  // Initialize the first layer with input values
  layers(0) := VecInit(
    io.in.map(_.asTypeOf(outputType).asTypeOf(UInt(outputElemWidth.W)))
  )

  // Generate adder tree layers
  for (d <- 0 until treeDepth) {
    val step = 1
    for (i <- 0 until numElements by (2 * step)) {
      val adder = Module(
        new Adder(
          opType,
          outputElemWidth,
          outputElemWidth,
          outputElemWidth
        )
      )
      adder.io.in_a := layers(d)(i)
      adder.io.in_b := layers(d)(i + step)
      layers(d + 1)(i / 2) := adder.io.out_c
    }
  }

  // Generate multiple adder tree outputs based on groupSizes
  val adderResults = groupSizes.map(size => layers(log2Ceil(size)))

  // Mux output based on cfg
  io.out := MuxLookup(io.cfg, adderResults(0))(
    (0 until groupSizes.length).map(i => (i).U -> adderResults(i))
  )
}

object AdderTreeEmitterUInt extends App {
  emitVerilog(
    new AdderTree(OpType.UIntUIntOp, 8, 9, 8, Seq(1, 2, 4)),
    Array("--target-dir", "generated/SpatialArray")
  )
}

object AdderTreeEmitterSInt extends App {
  emitVerilog(
    new AdderTree(OpType.SIntSIntOp, 16, 32, 1024, Seq(1, 2, 8)),
    Array("--target-dir", "generated/SpatialArray")
  )
}
