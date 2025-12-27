package snax_acc.utils

import chisel3._
import chisel3.util._

/** Parameter class for ParallelToSerial.
  *
  * @param parallelWidth
  *   The total width of the parallel input data.
  * @param serialWidth
  *   The width of each output serial chunk, must divide parallelWidth evenly.
  * @param earlyTerminate
  *   Whether to support early termination of the serialization process.
  * @param allowedTerminateFactors
  *   A sequence of allowed termination factors (not enforced in this implementation).
  */
case class ParallelAndSerialConverterParams(
  parallelWidth:           Int,
  serialWidth:             Int,
  earlyTerminate:          Boolean  = false,
  allowedTerminateFactors: Seq[Int] = Seq()
) {
  if (parallelWidth > serialWidth) {
    require(
      parallelWidth % serialWidth == 0,
      "parallelWidth must be an integer multiple of serialWidth."
    )
  }

  if (earlyTerminate) {
    require(
      allowedTerminateFactors.nonEmpty,
      "allowedTerminateFactors must be non-empty when earlyTerminate = true."
    )
    // Ensure all allowed factors are valid
    val maxFactor = parallelWidth / serialWidth
    allowedTerminateFactors.foreach { f =>
      require(
        f >= 1 && f <= maxFactor,
        s"Each allowed termination factor must be between 1 and $maxFactor."
      )
    }
  }

}

/** A module that sends a parallel input (via Decoupled I/O) out as multiple serial chunks (also Decoupled I/O).
  */
class ParallelToSerial(val p: ParallelAndSerialConverterParams) extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in               = Flipped(Decoupled(UInt(p.parallelWidth.W)))
    val terminate_factor =
      if (p.earlyTerminate) Some(Input(UInt(log2Ceil(p.parallelWidth / p.serialWidth + 1).W))) else None
    val out              = Decoupled(UInt(p.serialWidth.W))
    val start            = Input(Bool())
  })

  val ratio: Int = p.parallelWidth / p.serialWidth
  assert(ratio >= 1, "The ratio of parallelWidth to serialWidth must be at least 1.")
  val inBitsSeq = Wire(Vec(ratio, UInt(p.serialWidth.W)))
  inBitsSeq := VecInit(
    Seq.tabulate(ratio) { i =>
      io.in.bits((i + 1) * p.serialWidth - 1, i * p.serialWidth)
    }
  )

  val storedData = Seq(inBitsSeq(0)) ++ inBitsSeq.drop(1).map { i =>
    RegEnable(i, io.in.fire)
  }

  // Validate terminate_factor if early termination is enabled at runtime
  if (p.earlyTerminate) {
    val tf        = io.terminate_factor.get
    val isAllowed = p.allowedTerminateFactors
      .map(f => tf === f.U)
      .reduce(_ || _) // since allowedFactors non-empty if earlyTerminate
    assert(isAllowed, s"terminate_factor must be one of ${p.allowedTerminateFactors.mkString(", ")}")
  }

  val counter = Module(new BasicCounter(width = log2Ceil(ratio), hasCeil = true))
  if (p.earlyTerminate) {
    counter.io.ceilOpt.get := io.terminate_factor.get
  } else {
    counter.io.ceilOpt.get := ratio.U
  }
  counter.io.reset := io.start
  counter.io.tick := io.out.fire

  io.out.bits := MuxLookup(counter.io.value, 0.U.asTypeOf(UInt(p.serialWidth.W)))(storedData.zipWithIndex.map {
    case (i, j) =>
      j.U -> i
  })

  when(counter.io.value === 0.U) {
    io.out.valid := io.in.valid
    io.in.ready  := io.out.ready
  } otherwise {
    io.out.valid := true.B
    io.in.ready  := false.B
  }

}

/** A module that collects multiple serial inputs (via Decoupled I/O) and outputs them as a single parallel word (also
  * Decoupled I/O).
  */
class SerialToParallel(val p: ParallelAndSerialConverterParams) extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in               = Flipped(Decoupled(UInt(p.serialWidth.W)))
    val terminate_factor =
      if (p.earlyTerminate) Some(Input(UInt(log2Ceil(p.parallelWidth / p.serialWidth + 1).W))) else None
    val out              = Decoupled(UInt(p.parallelWidth.W))
    val start            = Input(Bool())
  })

  val ratio: Int = p.parallelWidth / p.serialWidth
  assert(ratio >= 1, "The ratio of parallelWidth to serialWidth must be at least 1.")
  val storeData = Wire(Vec(ratio - 1, Bool()))

  val outBitsSeq = Wire(Vec(ratio, UInt(p.serialWidth.W)))
  io.out.bits     := outBitsSeq.asTypeOf(io.out.bits)
  outBitsSeq.dropRight(1).zip(storeData).foreach { case (out, enable) =>
    out := RegEnable(io.in.bits, enable)
  }
  outBitsSeq.last := io.in.bits

  // Validate terminate_factor if early termination is enabled at runtime
  if (p.earlyTerminate) {
    val tf        = io.terminate_factor.get
    val isAllowed = p.allowedTerminateFactors
      .map(f => tf === f.U)
      .reduce(_ || _) // since allowedFactors non-empty if earlyTerminate
    assert(isAllowed, s"terminate_factor must be one of ${p.allowedTerminateFactors.mkString(", ")}")
  }

  val counter = Module(new BasicCounter(width = log2Ceil(ratio), hasCeil = true))
  if (p.earlyTerminate) {
    counter.io.ceilOpt.get := io.terminate_factor.get
  } else {
    counter.io.ceilOpt.get := ratio.U
  }
  counter.io.reset := io.start
  counter.io.tick := io.in.fire

  storeData.zipWithIndex.foreach({ case (a, b) => a := counter.io.value === b.U })

  when(counter.io.value === (ratio - 1).U) {
    io.out.valid := io.in.valid
    io.in.ready  := io.out.ready
  } otherwise {
    io.out.valid := false.B
    io.in.ready  := true.B
  }
}
