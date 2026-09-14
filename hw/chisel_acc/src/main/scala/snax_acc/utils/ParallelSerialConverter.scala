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
  *   Allowed runtime termination factors, checked by an assertion when early termination is enabled.
  * @param p2sChunksPerGroup
  *   Maximum number of serial chunks per ParallelToSerial shift group, including the bypassed first chunk in group 0.
  *   Must be a power of two and at least 2. SerialToParallel does not use this parameter.
  */
case class ParallelAndSerialConverterParams(
  parallelWidth:           Int,
  serialWidth:             Int,
  earlyTerminate:          Boolean  = false,
  allowedTerminateFactors: Seq[Int] = Seq(),
  p2sChunksPerGroup:       Int      = ParallelAndSerialConverterParams.DefaultP2sChunksPerGroup
) {
  require(
    p2sChunksPerGroup >= 2 && isPow2(p2sChunksPerGroup),
    "p2sChunksPerGroup must be a power of two and at least 2."
  )

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

object ParallelAndSerialConverterParams {
  val DefaultP2sChunksPerGroup: Int = 4
}

/** Storage for one group of a ParallelToSerial converter.
  *
  * A separate instance exposes each group's load/shift/hold muxes and registers to physical implementation. Placement
  * and control buffering must still be checked after synthesis; this does not force physical locality.
  */
class ParallelToSerialGroup(serialWidth: Int, storedChunks: Int) extends Module {
  require(serialWidth > 0 && storedChunks > 0)

  override def desiredName: String = s"ParallelToSerialGroup_${serialWidth}_${storedChunks}"

  val io = IO(new Bundle {
    val in    = Input(UInt((serialWidth * storedChunks).W))
    val load  = Input(Bool())
    val shift = Input(Bool())
    val out   = Output(UInt(serialWidth.W))
  })

  // Payload has no reset: the first transfer loads every group before its data is used.
  val shiftReg = Reg(UInt((serialWidth * storedChunks).W))
  when(io.load) {
    shiftReg := io.in
  }.elsewhen(io.shift) {
    shiftReg := shiftReg >> serialWidth
  }
  io.out := shiftReg(serialWidth - 1, 0)
}

/** A module that sends a parallel input (via Decoupled I/O) out as multiple serial chunks (also Decoupled I/O).
  *
  * The first chunk bypasses storage. All other chunks load on that transfer, then only the selected group shifts. For
  * example, 32 chunks and a group size of 8 partition storage into 7/8/8/8 chunks. The total payload storage is
  * unchanged; the additional group-output selection trades mux logic for shorter shift chains and smaller shift-enable
  * domains. A group size >= the ratio retains a single shift group for comparison.
  *
  * For ratios greater than one, is_busy_cstate gates input ready only. Callers must prevent a first-chunk output
  * transfer while not busy; VersaCore does this by keeping input valid low outside its busy state. counter_value_reset
  * discards the remaining word at the next clock edge; it does not suppress an output transfer on that edge. A
  * subsequent first-chunk transfer reloads all payload storage.
  */
class ParallelToSerial(val p: ParallelAndSerialConverterParams) extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in                  = Flipped(Decoupled(UInt(p.parallelWidth.W)))
    val terminate_factor    =
      if (p.earlyTerminate)
        Some(Input(UInt(log2Ceil(p.parallelWidth / p.serialWidth + 1).W)))
      else None
    val out                 = Decoupled(UInt(p.serialWidth.W))
    val counter_value_reset = Input(Bool())
    val is_busy_cstate      = Input(Bool())
  })

  val ratio: Int = p.parallelWidth / p.serialWidth
  assert(
    ratio >= 1,
    "The ratio of parallelWidth to serialWidth must be at least 1."
  )

  // Validate terminate_factor if early termination is enabled at runtime
  if (p.earlyTerminate) {
    val tf        = io.terminate_factor.get
    val isAllowed = p.allowedTerminateFactors
      .map(f => tf === f.U)
      .reduce(_ || _) // since allowedFactors non-empty if earlyTerminate
    assert(
      isAllowed,
      s"terminate_factor must be one of ${p.allowedTerminateFactors.mkString(", ")}"
    )
  }

  if (ratio == 1) {
    io.out.valid := io.in.valid
    io.out.bits  := io.in.bits
    io.in.ready  := io.out.ready
  } else {

    val counter = Module(
      new BasicCounter(width = log2Ceil(ratio), hasCeil = true)
    )
    if (p.earlyTerminate) {
      counter.io.ceilOpt.get := io.terminate_factor.get
    } else {
      counter.io.ceilOpt.get := ratio.U
    }
    counter.io.reset := io.counter_value_reset
    counter.io.tick := io.out.fire

    val firstChunk = counter.io.value === 0.U
    // The power-of-two group size makes group selection a constant bit slice, with no divider.
    val groupIndex = if (ratio > p.p2sChunksPerGroup) {
      counter.io.value >> log2Ceil(p.p2sChunksPerGroup)
    } else {
      0.U
    }
    // Preserve the original transfer semantics, including the caller's is_busy_cstate contract.
    val loadGroups = firstChunk  && io.out.fire
    val shiftGroup = !firstChunk && io.out.fire

    val groups = (0 until ratio by p.p2sChunksPerGroup).zipWithIndex.map { case (first, index) =>
      val storedFirst = math.max(1, first) // Chunk 0 bypasses the registers.
      val last        = math.min(first + p.p2sChunksPerGroup, ratio)
      val group       = Module(new ParallelToSerialGroup(p.serialWidth, last - storedFirst))
      group.suggestName(s"group_$index")
      group.io.in    := io.in.bits(last * p.serialWidth - 1, storedFirst * p.serialWidth)
      group.io.load  := loadGroups
      group.io.shift := shiftGroup && groupIndex === index.U
      group
    }

    val storedOut = if (groups.size == 1) {
      groups.head.io.out
    } else {
      VecInit(groups.map(_.io.out))(groupIndex)
    }

    when(firstChunk) {
      // first chunk comes directly from input
      io.out.valid := io.in.valid
      io.out.bits  := io.in.bits(p.serialWidth - 1, 0)
      io.in.ready  := io.out.ready && io.is_busy_cstate
    } otherwise {
      // Subsequent chunks come from the selected group's head.
      io.out.valid := true.B
      io.out.bits  := storedOut
      io.in.ready  := false.B && io.is_busy_cstate
    }
  }

}

/** A module that collects multiple serial inputs (via Decoupled I/O) and outputs them as a single parallel word (also
  * Decoupled I/O).
  */
class SerialToParallel(val p: ParallelAndSerialConverterParams) extends Module with RequireAsyncReset {
  val io = IO(new Bundle {
    val in                  = Flipped(Decoupled(UInt(p.serialWidth.W)))
    val terminate_factor    =
      if (p.earlyTerminate)
        Some(Input(UInt(log2Ceil(p.parallelWidth / p.serialWidth + 1).W)))
      else None
    val out                 = Decoupled(UInt(p.parallelWidth.W))
    val counter_value_reset = Input(Bool())
    val is_busy_cstate      = Input(Bool())
  })

  val ratio: Int = p.parallelWidth / p.serialWidth
  assert(
    ratio >= 1,
    "The ratio of parallelWidth to serialWidth must be at least 1."
  )

  // Validate terminate_factor if early termination is enabled at runtime
  if (p.earlyTerminate) {
    val tf        = io.terminate_factor.get
    val isAllowed = p.allowedTerminateFactors
      .map(f => tf === f.U)
      .reduce(_ || _) // since allowedFactors non-empty if earlyTerminate
    assert(
      isAllowed,
      s"terminate_factor must be one of ${p.allowedTerminateFactors.mkString(", ")}"
    )
  }

  if (ratio == 1) {
    io.out.valid := io.in.valid
    io.out.bits  := io.in.bits
    io.in.ready  := io.out.ready && io.is_busy_cstate
  } else {
    val storeData = Wire(Vec(ratio, Bool()))

    val outBitsSeq = Wire(Vec(ratio, UInt(p.serialWidth.W)))
    io.out.bits := outBitsSeq.asTypeOf(io.out.bits)
    outBitsSeq.zip(storeData).foreach { case (out, enable) =>
      out := RegEnable(io.in.bits, enable)
    }

    val counter = Module(
      new BasicCounter(width = log2Ceil(ratio) + 1, hasCeil = true)
    )
    if (p.earlyTerminate) {
      counter.io.ceilOpt.get := io.terminate_factor.get
    } else {
      counter.io.ceilOpt.get := ratio.U
    }
    counter.io.reset := io.counter_value_reset
    counter.io.tick := io.in.fire

    storeData.zipWithIndex.foreach({ case (a, b) =>
      a := counter.io.value === b.U && io.in.fire
    })

    val runtime_ratio = WireDefault(ratio.U)
    if (p.earlyTerminate) {
      runtime_ratio := io.terminate_factor.get
    } else {
      runtime_ratio := ratio.U
    }

    val last_data_write_fire =
      RegNext(counter.io.value === (runtime_ratio - 1.U) && io.in.fire, false.B)
    val output_stall         = io.out.valid && ~io.out.ready

    io.out.valid := last_data_write_fire || RegNext(output_stall, false.B)

    io.in.ready := ~output_stall && io.is_busy_cstate

  }

}
