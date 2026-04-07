package snax.readerWriter

import chisel3._
import chisel3.util._

import snax.utils._

/** AGU is the module to automatically generate the address for all ports.
  * @input
  *   cfg The description of the Address Generation Task. It is normally configured by CSR manager
  * @input
  *   start The signal to start a address generation task
  * @output
  *   busy The signal to indicate whether all address generation is finished. Only when busy == 0 the next address
  *   generation task can be launched
  * @output
  *   addresses The Vec[Decoupled[UInt]] signal to give tcdm_requestors the address
  * @param AddressGenUnitParam
  *   The parameter used for generation of the module
  */
class AddressGenUnitCfgIO(param: AddressGenUnitParam) extends Bundle {
  val ptr               = UInt(param.addressWidth.W)
  val enableFixedCache  = Bool() // bit 0 from csr
  val spatialStrides    = Vec(param.spatialBounds.length, UInt(param.addressWidth.W))  
  val temporalBounds    = Vec(param.temporalDimension, UInt(param.addressWidth.W))
  val temporalStrides   = Vec(param.temporalDimension, UInt(param.addressWidth.W))
  val addressRemapIndex = UInt(log2Ceil(param.tcdmLogicWordSize.length).W)

  def connectWithList(csrList: IndexedSeq[UInt]): IndexedSeq[UInt] = {
    var remainingCSR = csrList
    // Connect the ptr
    ptr := Cat(remainingCSR(1), remainingCSR(0))
    remainingCSR = remainingCSR.drop(2)
    // Connect the spatial strides
    for (i <- 0 until spatialStrides.length) {
      spatialStrides(i) := remainingCSR.head
      remainingCSR = remainingCSR.tail
    }
    // Connect the temporal bounds
    for (i <- 0 until temporalBounds.length) {
      temporalBounds(i) := remainingCSR.head
      remainingCSR = remainingCSR.tail
    }
    // Connect the temporal strides
    for (i <- 0 until temporalStrides.length) {
      temporalStrides(i) := remainingCSR.head
      remainingCSR = remainingCSR.tail
    }

    // Connect the address remap index
    if (param.tcdmLogicWordSize.length > 1) {
      addressRemapIndex := remainingCSR.head
      remainingCSR = remainingCSR.tail
    } else {
      addressRemapIndex := 0.U
    }

    remainingCSR
  }
}

// Instruction for writing into the fixed cache (from TCDM data)
class WriteFixedCacheInstructionIO(fixedCacheDepth: Int) extends Bundle {
  val index = UInt(log2Ceil(fixedCacheDepth).W)
}

// Instruction for reading from the fixed cache
class ReadFixedCacheInstructionIO(fixedCacheDepth: Int) extends Bundle {
  val index = UInt(log2Ceil(fixedCacheDepth).W)
}

// Writer-side instruction (kept for backward compatibility with writer AGU)
class FixedCacheInstructionIO(fixedCacheDepth: Int) extends Bundle {
  val index = UInt(log2Ceil(fixedCacheDepth).W)
  // True when all zero-stride counters within the cache scope are at their last value (ceil-1).
  // Used by the Writer in ReaderWriter mode: only write to TCDM (dataBuffer) on this last access;
  // all other accesses go to the reader's FixedLevelCache write port instead.
  val lastAccess = Bool()
}

// ============================================================================
// Common address computation logic shared between Reader and Writer AGUs
// ============================================================================
object AddressGenUnitCommon {
  def AffineAddressMapping(
    inputAddress:    UInt,
    physWordSize:    Int,
    logicalWordSize: Int
  ): UInt = {
    import snax.utils.BitsConcat._
    require(logicalWordSize <= physWordSize)
    require(physWordSize     % logicalWordSize == 0)
    require(isPow2(logicalWordSize))
    require(isPow2(physWordSize))
    if (logicalWordSize == physWordSize) {
      return inputAddress
    } else {
      return inputAddress(
        inputAddress.getWidth - (log2Ceil(physWordSize) - log2Ceil(
          logicalWordSize
        )) - 1,
        log2Ceil(logicalWordSize)
      ) ++ inputAddress(
        inputAddress.getWidth - 1,
        inputAddress.getWidth - (log2Ceil(physWordSize) - log2Ceil(
          logicalWordSize
        ))
      ) ++ inputAddress(log2Ceil(logicalWordSize) - 1, 0)
    }
  }
}

// ============================================================================
// Reader AGU: separate WriteFixedCacheBuffer and ReadFixedCacheBuffer
// ============================================================================
class AddressGenUnitReader(param: AddressGenUnitParam, moduleNamePrefix: String = "unnamed_cluster")
    extends Module
    with RequireAsyncReset {
  val io = IO(new Bundle {
    val cfg         = Input(new AddressGenUnitCfgIO(param))
    val start       = Input(Bool())
    val busy        = Output(Bool())
    val bufferEmpty = Output(Bool())
    val addr        = Vec(param.numChannel, Decoupled(UInt(param.addressWidth.W)))
    // Standalone useCache signal for the FixedLevelCache
    val useFixedCache = Output(Bool())
    // Write instructions: emitted when all invariant loops = 0 (same as outputBuffer)
    val writeFixedCacheInstruction = Decoupled(new WriteFixedCacheInstructionIO(param.fixedCacheDepth))
    // Read instructions: emitted every tick, valid only when enough writes have been serviced
    val readFixedCacheInstruction  = Decoupled(new ReadFixedCacheInstructionIO(param.fixedCacheDepth))
    // Any Critical Loop to cache found
    val anyLoopFound = Output(Bool())
  })

  require(param.spatialBounds.reduce(_ * _) <= param.numChannel)
  if (param.spatialBounds.reduce(_ * _) > param.numChannel) {
    Console.err.print(
      s"The multiplication of temporal bounds (${param.spatialBounds
          .reduce(_ * _)}) is larger than the number of channels(${param.numChannel}). Check the design parameter if you do not design it intentionally."
    )
  }

  override val desiredName = s"${moduleNamePrefix}_AddressGenUnit"

  // Create counters for each dimension
  val counters = for (i <- 0 until param.temporalDimension) yield {
    val counter = Module(
      new ProgrammableCounter(
        param.addressWidth,
        hasCeil = true,
        s"${moduleNamePrefix}_AddressGenUnitCounter"
      )
    )
    counter.io.reset := io.start
    counter.io.ceil := io.cfg.temporalBounds(i)
    counter.io.step := io.cfg.temporalStrides(i)
    counter
  }

  // Create the outputBuffer to store the generated address
  val outputBuffer = Module(
    new ComplexQueueConcat(
      inputWidth  = io.addr.head.bits.getWidth * param.numChannel,
      outputWidth = io.addr.head.bits.getWidth,
      depth       = param.outputBufferDepth,
      pipe        = true
    ) {
      override val desiredName = s"${moduleNamePrefix}_AddressBufferFIFO"
    }
  )

  // Use Fixed Cache when critical loop is not zero, update when necessary
  val criticalLoopFinder = Module(
    new CriticalLoopFinder(
      param.temporalDimension,
      param.addressWidth,
      param.fixedCacheDepth
    )
  )
  criticalLoopFinder.io.temporalBounds := io.cfg.temporalBounds
  criticalLoopFinder.io.temporalStrides := io.cfg.temporalStrides

  val countersIsZero = VecInit(counters.map(_.io.isZero))

  val newUseCache = io.cfg.enableFixedCache && criticalLoopFinder.io.anyLoopFound
  io.useFixedCache := newUseCache
  io.anyLoopFound := criticalLoopFinder.io.anyLoopFound

  val totalBounds = criticalLoopFinder.io.totalBounds

  val writeFixedCacheCounter_tick = Wire(Bool())
  val readFixedCacheCounter_tick  = Wire(Bool())

  val fixedCachePeriod = criticalLoopFinder.io.fixedCachePeriod

  // ── Fixed cache counters for the WRITE index ──
  // These are decoupled from the main counters and tick independently.
  // They are gated by writeNotTooFarAhead so they can never be more than one
  // critical_loop period ahead of the read counter.
  // Cascading is internal: each counter's tick depends on the previous write counter's lastVal.
  val writeFixedCacheCounterModules = scala.collection.mutable.ArrayBuffer[ProgrammableCounter]()
  val writeFixedCacheCounters = for (i <- 0 until param.temporalDimension) yield {
    val fixed_cache_counter = Module(
      new ProgrammableCounter(
        param.addressWidth,
        hasCeil = true,
        s"${moduleNamePrefix}_WriteFixedCacheCounter_${i}"
      )
    )
    if (i == 0) {
      fixed_cache_counter.io.reset := io.start
      fixed_cache_counter.io.tick := writeFixedCacheCounter_tick
      fixed_cache_counter.io.ceil := io.cfg.temporalBounds(i)
      fixed_cache_counter.io.step := 1.U
    } else {
      fixed_cache_counter.io.reset := io.start
      fixed_cache_counter.io.tick := writeFixedCacheCounter_tick && writeFixedCacheCounterModules(i - 1).io.lastVal
      fixed_cache_counter.io.ceil := io.cfg.temporalBounds(i)
      fixed_cache_counter.io.step := totalBounds(i - 1)
    }
    writeFixedCacheCounterModules += fixed_cache_counter
    val counter_out = Wire(UInt(param.addressWidth.W))
    when(io.cfg.temporalStrides(i) === 0.U || i.U > criticalLoopFinder.io.criticalLoop) {
      counter_out := 0.U
    }.otherwise {
      counter_out := fixed_cache_counter.io.value
    }
    (counter_out, fixed_cache_counter)
  }

  val writeIndex = VecInit(writeFixedCacheCounters.map(_._1)).reduceTree(_ + _)

  // ── Fixed cache counters for the READ index ──
  // These are fully decoupled from the write counters.
  // They tick independently when:
  //   1) There is space in the readFixedCacheBuffer
  //   2) The corresponding write instruction has been serviced
  // Cascading is internal: each counter's tick depends on the previous read counter's lastVal.
  val readFixedCacheCounterModules = scala.collection.mutable.ArrayBuffer[ProgrammableCounter]()
  val readFixedCacheCounters = for (i <- 0 until param.temporalDimension) yield {
    val fixed_cache_counter = Module(
      new ProgrammableCounter(
        param.addressWidth,
        hasCeil = true,
        s"${moduleNamePrefix}_ReadFixedCacheCounter_${i}"
      )
    )
    if (i == 0) {
      fixed_cache_counter.io.reset := io.start
      fixed_cache_counter.io.tick := readFixedCacheCounter_tick
      fixed_cache_counter.io.ceil := io.cfg.temporalBounds(i)
      fixed_cache_counter.io.step := 1.U
    } else {
      fixed_cache_counter.io.reset := io.start
      fixed_cache_counter.io.tick := readFixedCacheCounter_tick && readFixedCacheCounterModules(i - 1).io.lastVal
      fixed_cache_counter.io.ceil := io.cfg.temporalBounds(i)
      fixed_cache_counter.io.step := totalBounds(i - 1)
    }
    readFixedCacheCounterModules += fixed_cache_counter
    val counter_out = Wire(UInt(param.addressWidth.W))
    when(io.cfg.temporalStrides(i) === 0.U || i.U > criticalLoopFinder.io.criticalLoop) {
      counter_out := 0.U
    }.otherwise {
      counter_out := fixed_cache_counter.io.value
    }
    (counter_out, fixed_cache_counter)
  }

  val readIndex = VecInit(readFixedCacheCounters.map(_._1)).reduceTree(_ + _)
  // Period transition counters: track how many period boundaries each side has crossed.
  // The write side must never be more than 1 period ahead of the read side.
  // writePeriodCount: increments when a write instruction is *created* (counter tick).
  val writePeriodCount = RegInit(0.U(param.addressWidth.W))

  when(io.start) {
    writePeriodCount := 0.U
  }.elsewhen(writeFixedCacheCounter_tick) {
    writePeriodCount := writePeriodCount + 1.U
  }

  // readServicedPeriodCount: counts how many complete periods of read instructions have been
  // *actually serviced* (consumed by the downstream cache via io.readFixedCacheInstruction.fire).
  // Unlike readPeriodCount which tracked creation, this tracks consumption.
  val readServicedPeriodCount = RegInit(0.U(param.addressWidth.W))

  when(io.start) {
    readServicedPeriodCount := 0.U
  }.elsewhen(io.readFixedCacheInstruction.fire) {
    readServicedPeriodCount := readServicedPeriodCount + 1.U
  }

  // ── Serviced counter: counts how many cache writes have been accepted ──
  // Incremented when the downstream FixedLevelCache consumes a write instruction
  // (writeFixedCacheInstruction.fire = valid && ready).
  val servicedCounter = RegInit(0.U(param.addressWidth.W))
  when(io.start) {
    servicedCounter := 0.U
  }.elsewhen(io.writeFixedCacheInstruction.fire) {
    servicedCounter := servicedCounter + 1.U
  }

  // updateCache condition (write side): all invariant (zero-stride) loops within cache scope are at zero
  val updateCacheConditions = (0 until param.temporalDimension).map { i =>
    !((io.cfg.temporalStrides(i) === 0.U) && (i.U <= criticalLoopFinder.io.criticalLoop)) || countersIsZero(i)
  }
  val newUpdateCache = updateCacheConditions.reduce(_ && _)

  // updateCache condition (write cache side): same logic but using write cache counters' isZero
  val writeCountersIsZero = VecInit(writeFixedCacheCounterModules.map(_.io.isZero).toSeq)
  val writeUpdateCacheConditions = (0 until param.temporalDimension).map { i =>
    !((io.cfg.temporalStrides(i) === 0.U) && (i.U <= criticalLoopFinder.io.criticalLoop)) || writeCountersIsZero(i)
  }
  val writeUpdateCache = writeUpdateCacheConditions.reduce(_ && _)

  // updateCache condition (read side): same logic but using read counters' isZero
  // When true, the read position corresponds to a cache slot that needed a fresh write.
  // When false, we're re-reading already-cached data (invariant loop iteration), no write needed.
  val readCountersIsZero = VecInit(readFixedCacheCounterModules.map(_.io.isZero).toSeq)
  val readUpdateCacheConditions = (0 until param.temporalDimension).map { i =>
    !((io.cfg.temporalStrides(i) === 0.U) && (i.U <= criticalLoopFinder.io.criticalLoop)) || readCountersIsZero(i)
  }
  val readUpdateCache = readUpdateCacheConditions.reduce(_ && _)

  // ── Read issued counter: counts how many read instructions have been issued ──
  // Can never exceed servicedCounter.
  // Only incremented when the read counters are at a position that requires a new cache write
  // (i.e., the read-side updateCache condition is true).
  val readIssuedCounter = RegInit(0.U(param.addressWidth.W))
  when(io.start) {
    readIssuedCounter := 0.U
  }.elsewhen(readFixedCacheCounter_tick && readUpdateCache) {
    readIssuedCounter := readIssuedCounter + 1.U
  }

  // Calculate the current base address
  val temporalOffset = VecInit(counters.map(_.io.value)).reduceTree(_ + _)

  val spatialOffsetTable = for (i <- 0 until param.spatialBounds.length) yield {
    (0 until param.spatialBounds(i)).map(io.cfg.spatialStrides(i) * _.U)
  }

  val spatialOffsets = for (i <- 0 until param.numChannel) yield {
    var remainder     = i
    var spatialOffset = temporalOffset
    for (j <- 0 until param.spatialBounds.length) {
      spatialOffset = spatialOffset + spatialOffsetTable(j)(
        remainder % param.spatialBounds(j)
      )
      remainder     = remainder / param.spatialBounds(j)
    }
    spatialOffset
  }

  val currentAddress = Wire(Vec(io.addr.length, UInt(param.addressWidth.W)))
  currentAddress.zipWithIndex.foreach { case (address, index) =>
    address := io.cfg.ptr + spatialOffsets(index)
  }

  // Address remapping
  val remappedAddress = param.tcdmLogicWordSize.map { logicalWordSize =>
    currentAddress
      .map(i =>
        AddressGenUnitCommon.AffineAddressMapping(
          i,
          param.tcdmPhysWordSize,
          logicalWordSize
        )
      )
      .reduce((a, b) => Cat(b, a))
  }

  outputBuffer.io.in.head.bits := MuxLookup(
    io.cfg.addressRemapIndex,
    remappedAddress.head
  )(
    (0 until param.tcdmLogicWordSize.length).map(i => i.U -> remappedAddress(i))
  )

  outputBuffer.io.out.zip(io.addr).foreach { case (a, b) => a <> b }

  // ── WriteFixedCacheBuffer: appended in same conditions as outputBuffer (all invariant loops = 0) ──
  val writeFixedCacheBuffer = Module(new Queue(new WriteFixedCacheInstructionIO(param.fixedCacheDepth), param.outputBufferDepth, pipe = true) {
    override val desiredName = s"${moduleNamePrefix}_WriteFixedCacheBufferFIFO"
  })
  writeFixedCacheBuffer.io.enq.bits.index := writeIndex
  io.writeFixedCacheInstruction <> writeFixedCacheBuffer.io.deq

  // ── ReadFixedCacheBuffer: appended when readFixedCacheCounter ticks ──
  val readFixedCacheBuffer = Module(new Queue(new ReadFixedCacheInstructionIO(param.fixedCacheDepth), param.outputBufferDepth, pipe = true) {
    override val desiredName = s"${moduleNamePrefix}_ReadFixedCacheBufferFIFO"
  })
  readFixedCacheBuffer.io.enq.bits.index := readIndex

  // Read instructions pass through directly — the gating is done at the read counter tick level.
  io.readFixedCacheInstruction <> readFixedCacheBuffer.io.deq

  // bufferEmpty signal
  io.bufferEmpty := ~(outputBuffer.io.out.map(i => i.valid).reduce(_ | _)) &&
    (!newUseCache || (!writeFixedCacheBuffer.io.deq.valid && !readFixedCacheBuffer.io.deq.valid))

  // FSM
  val sIDLE :: sBUSY :: Nil = Enum(2)
  val currentState          = RegInit(sIDLE)

  // ── Buffer readiness signals (decoupled) ──
  // The write cache counters are now decoupled from the main counters.
  // TCDM requests (outputBuffer) fire as soon as space is available.
  // Write cache instructions wait for writeNotTooFarAhead.
  // Bypass: include current-cycle read fire to eliminate 1-cycle Reg latency on readServicedPeriodCount
  val effectiveReadServicedPeriodCount = readServicedPeriodCount + io.readFixedCacheInstruction.fire.asUInt
  val writeNotTooFarAhead = writePeriodCount < effectiveReadServicedPeriodCount + fixedCachePeriod

  // Output buffer (TCDM requests): can accept as soon as there is space
  val outputBufferCanAccept = outputBuffer.io.in.head.ready

  // Write cache buffer: gated by writeNotTooFarAhead to prevent overwriting unread cache slots
  val writeBufferCanAccept = Wire(Bool())
  when(newUseCache) {
    writeBufferCanAccept := Mux(writeUpdateCache,
      writeFixedCacheBuffer.io.enq.ready && writeNotTooFarAhead,
      writeNotTooFarAhead
    )
  }.otherwise {
    writeBufferCanAccept := true.B
  }

  // ── Read-side counter can tick condition ──
  // A read instruction can be issued when:
  //   1) There is space in the readFixedCacheBuffer
  //   2) If this read position needs a cache write (readUpdateCache), the write must have been
  //      serviced (readIssuedCounter < servicedCounter). Otherwise, we're re-reading cached data
  //      and can proceed freely.
  val effectiveServicedCounter = servicedCounter + io.writeFixedCacheInstruction.fire.asUInt
  val readCanTick = readFixedCacheBuffer.io.enq.ready && (!readUpdateCache || (readIssuedCounter < effectiveServicedCounter))

  // All counters done signals
  val mainCountersDone = counters.map(_.io.lastVal).reduce(_ & _)
  val writeCountersDone = writeFixedCacheCounters.map(_._2.io.lastVal).reduce(_ & _)
  val readCountersDone  = readFixedCacheCounters.map(_._2.io.lastVal).reduce(_ & _)

  when(io.start && io.cfg.temporalBounds.map(_ =/= 0.U).reduce(_ && _)) {
    currentState := sBUSY
  }.elsewhen(currentState === sBUSY && Mux(newUseCache,
    mainCountersDone && Mux(newUpdateCache, outputBufferCanAccept, true.B) &&
    writeCountersDone && writeBufferCanAccept &&
    readCountersDone && readCanTick,
    mainCountersDone && outputBufferCanAccept
  )) {
    currentState := sIDLE
  }.otherwise {
    currentState := currentState
  }

  // Valid signals for the buffers
  // OutputBuffer: written when updateCache (cache miss, first access) or when cache not used
  outputBuffer.io.in.head.valid := currentState === sBUSY && (newUpdateCache || !newUseCache) && outputBufferCanAccept
  // WriteFixedCacheBuffer: written when write cache counters are at a cache-miss position, gated by writeNotTooFarAhead
  writeFixedCacheBuffer.io.enq.valid := currentState === sBUSY && newUseCache && writeUpdateCache && writeBufferCanAccept
  // ReadFixedCacheBuffer: written when the read counter ticks (decoupled from write)
  readFixedCacheBuffer.io.enq.valid := currentState === sBUSY && newUseCache && readCanTick

  // Busy depends on all counters (main, write cache, and read cache)
  io.busy := currentState === sBUSY

  // ── Tick logic ──
  // Main counters: tick based on outputBuffer readiness (cache misses) or freely (cache hits)
  // This allows TCDM requests to be issued as soon as the outputBuffer has space,
  // without waiting for the write cache buffer or writeNotTooFarAhead.
  val counters_tick = currentState === sBUSY && Mux(newUseCache,
    Mux(newUpdateCache, outputBufferCanAccept, true.B),
    outputBufferCanAccept
  )
  counters.head.io.tick := counters_tick
  if (counters.length > 1) {
    counters.tail.zip(counters).foreach { case (a, b) =>
      a.io.tick := b.io.lastVal && counters_tick
    }
  }

  // Write fixed cache counters: tick independently, gated by writeBufferCanAccept
  writeFixedCacheCounter_tick := currentState === sBUSY && newUseCache && writeBufferCanAccept

  // Read fixed cache counters: tick independently when readCanTick
  readFixedCacheCounter_tick := currentState === sBUSY && newUseCache && readCanTick
}

// ============================================================================
// Writer AGU: keeps lastAccess instruction for ReaderWriter routing
// ============================================================================
class AddressGenUnitWriter(param: AddressGenUnitParam, moduleNamePrefix: String = "unnamed_cluster")
    extends Module
    with RequireAsyncReset {
  val io = IO(new Bundle {
    val cfg         = Input(new AddressGenUnitCfgIO(param))
    val start       = Input(Bool())
    val busy        = Output(Bool())
    val bufferEmpty = Output(Bool())
    val addr        = Vec(param.numChannel, Decoupled(UInt(param.addressWidth.W)))
    // Standalone useCache signal
    val useFixedCache = Output(Bool())
    // Writer instruction with lastAccess
    val fixedCacheInstruction = Decoupled(new FixedCacheInstructionIO(param.fixedCacheDepth))
    val anyLoopFound = Output(Bool())
  })

  require(param.spatialBounds.reduce(_ * _) <= param.numChannel)
  if (param.spatialBounds.reduce(_ * _) > param.numChannel) {
    Console.err.print(
      s"The multiplication of temporal bounds (${param.spatialBounds
          .reduce(_ * _)}) is larger than the number of channels(${param.numChannel}). Check the design parameter if you do not design it intentionally."
    )
  }

  override val desiredName = s"${moduleNamePrefix}_AddressGenUnit"

  // Create counters for each dimension
  val counters = for (i <- 0 until param.temporalDimension) yield {
    val counter = Module(
      new ProgrammableCounter(
        param.addressWidth,
        hasCeil = true,
        s"${moduleNamePrefix}_AddressGenUnitCounter"
      )
    )
    counter.io.reset := io.start
    counter.io.ceil := io.cfg.temporalBounds(i)
    counter.io.step := io.cfg.temporalStrides(i)
    counter
  }

  val outputBuffer = Module(
    new ComplexQueueConcat(
      inputWidth  = io.addr.head.bits.getWidth * param.numChannel,
      outputWidth = io.addr.head.bits.getWidth,
      depth       = param.outputBufferDepth,
      pipe        = true
    ) {
      override val desiredName = s"${moduleNamePrefix}_AddressBufferFIFO"
    }
  )

  val criticalLoopFinder = Module(
    new CriticalLoopFinder(
      param.temporalDimension,
      param.addressWidth,
      param.fixedCacheDepth
    )
  )
  criticalLoopFinder.io.temporalBounds := io.cfg.temporalBounds
  criticalLoopFinder.io.temporalStrides := io.cfg.temporalStrides

  val countersIsZero = VecInit(counters.map(_.io.isZero))

  val newUseCache = io.cfg.enableFixedCache && criticalLoopFinder.io.anyLoopFound
  io.useFixedCache := newUseCache
  io.anyLoopFound := criticalLoopFinder.io.anyLoopFound

  val totalBounds = criticalLoopFinder.io.totalBounds

  val fixedCacheCounter_tick = Wire(Bool())
  val fixedCacheCounters = for (i <- 0 until param.temporalDimension) yield {
    val fixed_cache_counter = Module(
      new ProgrammableCounter(
        param.addressWidth,
        hasCeil = true,
        s"${moduleNamePrefix}_FixedCacheCounter_${i}"
      )
    )
    if (i == 0) {
      fixed_cache_counter.io.reset := io.start
      fixed_cache_counter.io.tick := fixedCacheCounter_tick
      fixed_cache_counter.io.ceil := io.cfg.temporalBounds(i)
      fixed_cache_counter.io.step := 1.U
    } else {
      fixed_cache_counter.io.reset := io.start
      fixed_cache_counter.io.tick := fixedCacheCounter_tick && counters(i - 1).io.lastVal
      fixed_cache_counter.io.ceil := io.cfg.temporalBounds(i)
      fixed_cache_counter.io.step := totalBounds(i - 1)
    }
    val counter_out = Wire(UInt(param.addressWidth.W))
    when(io.cfg.temporalStrides(i) === 0.U || i.U > criticalLoopFinder.io.criticalLoop) {
      counter_out := 0.U
    }.otherwise {
      counter_out := fixed_cache_counter.io.value
    }
    counter_out
  }

  val newIndex = VecInit(fixedCacheCounters).reduceTree(_ + _)

  // lastAccess condition (same as before)
  val lastAccessConditions = (0 until param.temporalDimension).map { i =>
    !((io.cfg.temporalStrides(i) === 0.U) && (i.U <= criticalLoopFinder.io.criticalLoop)) || counters(i).io.isLastVal
  }
  val newLastAccess = lastAccessConditions.reduce(_ && _)

  // updateCache condition (for writer: outputBuffer fills on lastAccess)
  val lastAccessForFill = newLastAccess

  // Address computation
  val temporalOffset = VecInit(counters.map(_.io.value)).reduceTree(_ + _)

  val spatialOffsetTable = for (i <- 0 until param.spatialBounds.length) yield {
    (0 until param.spatialBounds(i)).map(io.cfg.spatialStrides(i) * _.U)
  }

  val spatialOffsets = for (i <- 0 until param.numChannel) yield {
    var remainder     = i
    var spatialOffset = temporalOffset
    for (j <- 0 until param.spatialBounds.length) {
      spatialOffset = spatialOffset + spatialOffsetTable(j)(
        remainder % param.spatialBounds(j)
      )
      remainder     = remainder / param.spatialBounds(j)
    }
    spatialOffset
  }

  val currentAddress = Wire(Vec(io.addr.length, UInt(param.addressWidth.W)))
  currentAddress.zipWithIndex.foreach { case (address, index) =>
    address := io.cfg.ptr + spatialOffsets(index)
  }

  val remappedAddress = param.tcdmLogicWordSize.map { logicalWordSize =>
    currentAddress
      .map(i =>
        AddressGenUnitCommon.AffineAddressMapping(
          i,
          param.tcdmPhysWordSize,
          logicalWordSize
        )
      )
      .reduce((a, b) => Cat(b, a))
  }

  outputBuffer.io.in.head.bits := MuxLookup(
    io.cfg.addressRemapIndex,
    remappedAddress.head
  )(
    (0 until param.tcdmLogicWordSize.length).map(i => i.U -> remappedAddress(i))
  )

  outputBuffer.io.out.zip(io.addr).foreach { case (a, b) => a <> b }

  // Fixed cache instruction buffer for writer (with lastAccess)
  val fixedCacheInstructionBuffer = Module(new Queue(new FixedCacheInstructionIO(param.fixedCacheDepth), param.outputBufferDepth) {
    override val desiredName = s"${moduleNamePrefix}_FixedCacheIndexBufferFIFO"
  })
  fixedCacheInstructionBuffer.io.enq.bits.index      := newIndex
  fixedCacheInstructionBuffer.io.enq.bits.lastAccess := newLastAccess
  io.fixedCacheInstruction <> fixedCacheInstructionBuffer.io.deq

  io.bufferEmpty := ~(outputBuffer.io.out.map(i => i.valid).reduce(_ | _)) &&
    (fixedCacheInstructionBuffer.io.deq.valid === false.B || ~newUseCache)

  // FSM
  val sIDLE :: sBUSY :: Nil = Enum(2)
  val currentState          = RegInit(sIDLE)
  when(io.start && io.cfg.temporalBounds.map(_ =/= 0.U).reduce(_ && _)) {
    currentState := sBUSY
  }.elsewhen(
    counters.map(_.io.lastVal).reduce(_ & _) && ((!newUseCache && outputBuffer.io.in.head.fire) || (newUseCache && fixedCacheInstructionBuffer.io.enq.fire))
  ) {
    currentState := sIDLE
  }.otherwise {
    currentState := currentState
  }

  // Writer: fill outputBuffer on last access (newLastAccess) so TCDM is written once.
  val outputBufferFillCondition = newLastAccess
  outputBuffer.io.in.head.valid := currentState === sBUSY && (outputBufferFillCondition || !newUseCache) && (outputBuffer.io.in.head.ready && fixedCacheInstructionBuffer.io.enq.ready)
  fixedCacheInstructionBuffer.io.enq.valid := currentState === sBUSY && newUseCache && (outputBuffer.io.in.head.ready && fixedCacheInstructionBuffer.io.enq.ready)
  io.busy := currentState === sBUSY

  val counters_tick =
    currentState === sBUSY && ((outputBufferFillCondition && outputBuffer.io.in.head.fire) || (fixedCacheInstructionBuffer.io.enq.fire) || (!newUseCache && outputBuffer.io.in.head.fire))
  counters.head.io.tick := counters_tick
  if (counters.length > 1) {
    counters.tail.zip(counters).foreach { case (a, b) =>
      a.io.tick := b.io.lastVal && counters_tick
    }
  }
  fixedCacheCounter_tick := counters_tick
}