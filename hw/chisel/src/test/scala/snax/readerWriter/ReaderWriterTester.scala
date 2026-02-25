package snax.readerWriter

import chisel3._
import chisel3.util._

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

class ReaderWriterTester extends AnyFlatSpec with ChiselScalatestTester {

  // Trivial accelerator transform: add 1 to each 64-bit word in the wide data
  def transformData(data: BigInt, numChannels: Int, dataWidth: Int): BigInt = {
    val mask = (BigInt(1) << dataWidth) - 1
    var result: BigInt = 0
    for (i <- 0 until numChannels) {
      val word = (data >> (i * dataWidth)) & mask
      val transformed = (word + 1) & mask
      result |= (transformed << (i * dataWidth))
    }
    result
  }

  // Apply strobe-masked write: only update bytes where strb bit is set
  def applyStrbWrite(
      oldData: BigInt,
      newData: BigInt,
      strb: BigInt,
      dataWidthBytes: Int
  ): BigInt = {
    var result = oldData
    for (byteIdx <- 0 until dataWidthBytes) {
      if (((strb >> byteIdx) & 1) != 0) {
        val byteMask = BigInt(0xff) << (byteIdx * 8)
        result = (result & ~byteMask) | (newData & byteMask)
      }
    }
    result
  }

  // Helper: configure an AGU with 5 temporal dimensions
  def configureAgu(
      aguCfg: AddressGenUnitCfgIO,
      basePtr: BigInt,
      spatialStride: Int,
      temporalStrides: Seq[Int],
      temporalBounds: Seq[Int]
  ): Unit = {
    require(temporalStrides.length == 5 && temporalBounds.length == 5,
      "Must supply exactly 5 temporal strides and bounds")
    aguCfg.ptr.poke(basePtr.U)
    aguCfg.spatialStrides(0).poke(spatialStride)
    for (i <- 0 until 5) {
      aguCfg.temporalStrides(i).poke(temporalStrides(i))
      aguCfg.temporalBounds(i).poke(temporalBounds(i))
    }
  }

  // Shared TCDM model: handles reads (1-cycle response latency) and writes
  // (fire-and-forget, immediate memory update). Returns (newReadReqs, newWriteReqs).
  def tcdmStep(
      dut: ReaderWriter,
      numChannels: Int,
      dataWidth: Int,
      tcdmMemory: mutable.HashMap[BigInt, BigInt],
      pendingReadAddr: Array[Option[BigInt]]
  ): (Int, Int) = {
    var readReqs  = 0
    var writeReqs = 0
    for (ch <- 0 until numChannels) {
      pendingReadAddr(ch) match {
        case Some(addr) =>
          val data = tcdmMemory.getOrElse(addr, BigInt(0))
          dut.io.readerInterface.tcdmReq(ch).ready.poke(false.B)
          dut.io.readerInterface.tcdmRsp(ch).valid.poke(true.B)
          dut.io.readerInterface.tcdmRsp(ch).bits.data.poke(data.U)
          pendingReadAddr(ch) = None

        case None =>
          dut.io.readerInterface.tcdmReq(ch).ready.poke(true.B)
          dut.io.readerInterface.tcdmRsp(ch).valid.poke(false.B)
          if (dut.io.readerInterface.tcdmReq(ch).valid.peek().litToBoolean) {
            val addr  = dut.io.readerInterface.tcdmReq(ch).bits.addr.peek().litValue
            val write = dut.io.readerInterface.tcdmReq(ch).bits.write.peek().litToBoolean
            val data  = dut.io.readerInterface.tcdmReq(ch).bits.data.peek().litValue
            val strb  = dut.io.readerInterface.tcdmReq(ch).bits.strb.peek().litValue
            if (write) {
              val oldData = tcdmMemory.getOrElse(addr, BigInt(0))
              tcdmMemory(addr) = applyStrbWrite(oldData, data, strb, dataWidth / 8)
              writeReqs += 1
            } else {
              pendingReadAddr(ch) = Some(addr)
              readReqs += 1
            }
          }
      }
    }
    (readReqs, writeReqs)
  }

  // ---- Test 1: Basic read-transform-write (5 temporal dims, 1 active) ----
  // Reader reads data from TCDM, testbench transforms it (+1 per word),
  // writer writes transformed data back to the same addresses.
  // Verify final memory state.
  "ReaderWriter: basic read-transform-write" should "pass" in test(
    new ReaderWriter(
      new ReaderWriterParam(
        tcdmDataWidth      = 64,
        numChannel         = 8,
        spatialBounds      = List(8),
        temporalDimension  = 5,
        addressBufferDepth = 2,
        dataBufferDepth    = 2,
        tcdmSize           = 128
      ),
      new ReaderWriterParam(
        tcdmDataWidth      = 64,
        numChannel         = 8,
        spatialBounds      = List(8),
        temporalDimension  = 5,
        addressBufferDepth = 2,
        dataBufferDepth    = 2,
        tcdmSize           = 128
      )
    )
  ).withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) {
    dut =>
      dut.clock.setTimeout(0)

      val numChannels    = 8
      val dataWidth      = 64
      val mask64         = (BigInt(1) << 64) - 1
      val basePtr: BigInt = 0x1000
      val spatialStride  = 8   // 8 bytes per 64-bit word
      // 1 active temporal dim: stride=64 (8ch*8B), bound=4
      val tBounds = Seq(2, 2, 8, 4, 4)
      val tStrides = Seq(256, 2048, 0, 512, 4096)
      val totalIterations = tBounds.product // 2*2*8*4*4 = 512

      // ---- Compute temporal address offsets (matches AGU nested counters) ----
      // The AGU iterates dim 0 (innermost) to dim 4 (outermost).
      // stride=0 dims produce repeated visits to the same addresses.
      val temporalOffsets = mutable.HashMap[BigInt, Int]().withDefaultValue(0)
      def countVisits(dim: Int, offset: BigInt): Unit = {
        if (dim < 0) temporalOffsets(offset) += 1
        else for (i <- 0 until tBounds(dim)) countVisits(dim - 1, offset + i * tStrides(dim))
      }
      countVisits(tBounds.length - 1, 0)

      // ---- Simulated TCDM memory ----
      val tcdmMemory = mutable.HashMap[BigInt, BigInt]()

      // Pre-fill memory at all unique addresses the AGU will visit: data = address
      for ((tOffset, _) <- temporalOffsets; ch <- 0 until numChannels) {
        val addr = basePtr + tOffset + ch * spatialStride
        tcdmMemory(addr) = addr & mask64
      }

      println("=== Test 1: Basic read-transform-write (5 temporal dims) ===")
      println(s"Total temporal iterations: $totalIterations")
      println(s"Unique temporal offsets: ${temporalOffsets.size}")

      // ---- Configure reader & writer AGU (same access pattern) ----
      configureAgu(dut.io.readerInterface.aguCfg, basePtr, spatialStride, tStrides, tBounds)
      configureAgu(dut.io.writerInterface.aguCfg, basePtr, spatialStride, tStrides, tBounds)

      val enableMask = (BigInt(1) << numChannels) - 1
      if (dut.io.readerInterface.readerwriterCfg.enabledChannel.getWidth > 0)
        dut.io.readerInterface.readerwriterCfg.enabledChannel.poke(enableMask.U)
      if (dut.io.writerInterface.readerwriterCfg.enabledChannel.getWidth > 0)
        dut.io.writerInterface.readerwriterCfg.enabledChannel.poke(enableMask.U)

      // ---- Initialize ports ----
      dut.io.readerInterface.tcdmRsp.foreach { rsp =>
        rsp.valid.poke(false.B)
        rsp.bits.data.poke(0.U)
      }
      dut.io.readerInterface.data.ready.poke(false.B)
      dut.io.writerInterface.data.valid.poke(false.B)
      dut.io.writerInterface.data.bits.poke(0.U)
      dut.io.readerInterface.aguCfg.enableFixedCache.poke(true.B)
      dut.io.writerInterface.aguCfg.enableFixedCache.poke(true.B)


      // ---- Start both ----
      dut.io.readerInterface.start.poke(true.B)
      dut.io.writerInterface.start.poke(true.B)
      dut.clock.step()
      dut.io.readerInterface.start.poke(false.B)
      dut.io.writerInterface.start.poke(false.B)

      val pendingReadAddr = Array.fill(numChannels)(Option.empty[BigInt])
      val dataQueue       = mutable.Queue[BigInt]()

      var cycleCount   = 0
      val maxCycles    = 5000
      var readRequests  = 0
      var writeRequests = 0
      var dataRead      = 0
      var dataWritten   = 0

      while (
        cycleCount < maxCycles &&
        (dut.io.readerInterface.busy.peek().litToBoolean ||
          dut.io.writerInterface.busy.peek().litToBoolean ||
          !dut.io.readerInterface.bufferEmpty.peek().litToBoolean ||
          !dut.io.writerInterface.bufferEmpty.peek().litToBoolean ||
          dataQueue.nonEmpty)
      ) {
        val (rr, wr) = tcdmStep(dut, numChannels, dataWidth, tcdmMemory, pendingReadAddr)
        readRequests += rr
        writeRequests += wr

        // reader output -> transform -> writer input
        if (dut.io.readerInterface.data.valid.peek().litToBoolean) {
          val readData = dut.io.readerInterface.data.bits.peek().litValue
          dataQueue.enqueue(transformData(readData, numChannels, dataWidth))
          dataRead += 1
        }
        dut.io.readerInterface.data.ready.poke(true.B)

        if (dataQueue.nonEmpty) {
          dut.io.writerInterface.data.valid.poke(true.B)
          dut.io.writerInterface.data.bits.poke(dataQueue.head.U)
          if (dut.io.writerInterface.data.ready.peek().litToBoolean) {
            dataQueue.dequeue()
            dataWritten += 1
          }
        } else {
          dut.io.writerInterface.data.valid.poke(false.B)
          dut.io.writerInterface.data.bits.poke(0.U)
        }

        dut.clock.step()
        cycleCount += 1
      }

      // ---- Summary & verification ----
      println(s"\n=== Test 1 Summary ===")
      println(s"  Total cycles: $cycleCount")
      println(s"  Read requests:  $readRequests")
      println(s"  Write requests: $writeRequests")
      println(s"  Data read:   $dataRead, Data written: $dataWritten")

      assert(cycleCount < maxCycles, s"Test timed out after $maxCycles cycles")

      // val expectedAccesses = totalIterations * numChannels
      // assert(readRequests == expectedAccesses,
      //   s"Expected $expectedAccesses read requests, got $readRequests")
      // assert(writeRequests == expectedAccesses,
      //   s"Expected $expectedAccesses write requests, got $writeRequests")
      // assert(dataRead == totalIterations,
      //   s"Expected $totalIterations reader outputs, got $dataRead")
      // assert(dataWritten == totalIterations,
      //   s"Expected $totalIterations writer inputs, got $dataWritten")

      // Verify final memory state.
      // Each unique address is visited visitCount times (due to stride=0 dims).
      // Each visit reads current value, adds 1 per word, writes back.
      // So final value at each address = original + visitCount.
      var errors = 0
      val uniqueAddressCount = temporalOffsets.size * numChannels
      for ((tOffset, visitCount) <- temporalOffsets; ch <- 0 until numChannels) {
        val addr     = basePtr + tOffset + ch * spatialStride
        val original = addr & mask64
        val expected = (original + visitCount) & mask64
        val actual   = tcdmMemory.getOrElse(addr, BigInt(-1))
        if (actual != expected) {
          if (errors < 20)
            println(f"  MISMATCH at 0x$addr%04x: expected=0x$expected%016x actual=0x$actual%016x (visits=$visitCount)")
          errors += 1
        }
      }
      if (errors == 0)
        println(s"  All $uniqueAddressCount memory locations verified correctly!")
      assert(errors == 0, s"$errors memory locations had incorrect values")
  }

  // // ---- Test 2: Pipelined read-write with 2 active temporal dimensions ----
  // // Tests nested temporal loops and reader/writer overlap on the shared TCDM.
  // "ReaderWriter: pipelined multi-dimension" should "pass" in test(
  //   new ReaderWriter(
  //     new ReaderWriterParam(
  //       tcdmDataWidth      = 64,
  //       numChannel         = 32,
  //       spatialBounds      = List(32),
  //       temporalDimension  = 5,
  //       addressBufferDepth = 4,
  //       dataBufferDepth    = 4,
  //       tcdmSize           = 128
  //     ),
  //     new ReaderWriterParam(
  //       tcdmDataWidth      = 64,
  //       numChannel         = 32,
  //       spatialBounds      = List(32),
  //       temporalDimension  = 5,
  //       addressBufferDepth = 4,
  //       dataBufferDepth    = 4,
  //       tcdmSize           = 128
  //     )
  //   )
  // ).withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) {
  //   dut =>
  //     dut.clock.setTimeout(0)

  //     val numChannels    = 32
  //     val dataWidth      = 64
  //     val mask64         = (BigInt(1) << 64) - 1
  //     val basePtr: BigInt = 0x2000
  //     val spatialStride  = 32
  //     // 2 active temporal dims: dim0 stride=64 bound=4, dim1 stride=512 bound=2
  //     // Total iterations: 4 * 2 = 8
  //     val tBounds = Seq(2, 2, 8, 4, 4)
  //     val tStrides = Seq(256, 2048, 0, 512, 4096)
  //     val totalIterations = tBounds.product // 8

  //     val tcdmMemory = mutable.HashMap[BigInt, BigInt]()
  //     // Pre-fill: enumerate all (i0, i1) combinations
  //     for (i1 <- 0 until tBounds(1); i0 <- 0 until tBounds(0); ch <- 0 until numChannels) {
  //       val addr = basePtr + i0 * tStrides(0) + i1 * tStrides(1) + ch * spatialStride
  //       tcdmMemory(addr) = addr & mask64
  //     }

  //     println("\n=== Test 2: Pipelined multi-dimension (5 temporal dims, 2 active) ===")
  //     println(s"Total temporal iterations: $totalIterations")

  //     configureAgu(dut.io.readerInterface.aguCfg, basePtr, spatialStride, tStrides, tBounds)
  //     configureAgu(dut.io.writerInterface.aguCfg, basePtr, spatialStride, tStrides, tBounds)

  //     val enableMask = (BigInt(1) << numChannels) - 1
  //     if (dut.io.readerInterface.readerwriterCfg.enabledChannel.getWidth > 0)
  //       dut.io.readerInterface.readerwriterCfg.enabledChannel.poke(enableMask.U)
  //     if (dut.io.writerInterface.readerwriterCfg.enabledChannel.getWidth > 0)
  //       dut.io.writerInterface.readerwriterCfg.enabledChannel.poke(enableMask.U)

  //     // Initialize ports
  //     dut.io.readerInterface.tcdmRsp.foreach { rsp =>
  //       rsp.valid.poke(false.B)
  //       rsp.bits.data.poke(0.U)
  //     }
  //     dut.io.readerInterface.data.ready.poke(false.B)
  //     dut.io.writerInterface.data.valid.poke(false.B)
  //     dut.io.writerInterface.data.bits.poke(0.U)

  //     // Start both
  //     dut.io.readerInterface.start.poke(true.B)
  //     dut.io.writerInterface.start.poke(true.B)
  //     dut.clock.step()
  //     dut.io.readerInterface.start.poke(false.B)
  //     dut.io.writerInterface.start.poke(false.B)

  //     val pendingReadAddr = Array.fill(numChannels)(Option.empty[BigInt])
  //     val dataQueue       = mutable.Queue[BigInt]()

  //     var cycleCount   = 0
  //     val maxCycles    = 3000
  //     var readRequests  = 0
  //     var writeRequests = 0
  //     var dataRead      = 0
  //     var dataWritten   = 0

  //     while (
  //       cycleCount < maxCycles &&
  //       (dut.io.readerInterface.busy.peek().litToBoolean ||
  //         dut.io.writerInterface.busy.peek().litToBoolean ||
  //         !dut.io.readerInterface.bufferEmpty.peek().litToBoolean ||
  //         !dut.io.writerInterface.bufferEmpty.peek().litToBoolean ||
  //         dataQueue.nonEmpty)
  //     ) {
  //       val (rr, wr) = tcdmStep(dut, numChannels, dataWidth, tcdmMemory, pendingReadAddr)
  //       readRequests += rr
  //       writeRequests += wr

  //       if (dut.io.readerInterface.data.valid.peek().litToBoolean) {
  //         val readData = dut.io.readerInterface.data.bits.peek().litValue
  //         dataQueue.enqueue(transformData(readData, numChannels, dataWidth))
  //         dataRead += 1
  //       }
  //       dut.io.readerInterface.data.ready.poke(true.B)

  //       if (dataQueue.nonEmpty) {
  //         dut.io.writerInterface.data.valid.poke(true.B)
  //         dut.io.writerInterface.data.bits.poke(dataQueue.head.U)
  //         if (dut.io.writerInterface.data.ready.peek().litToBoolean) {
  //           dataQueue.dequeue()
  //           dataWritten += 1
  //         }
  //       } else {
  //         dut.io.writerInterface.data.valid.poke(false.B)
  //         dut.io.writerInterface.data.bits.poke(0.U)
  //       }

  //       dut.clock.step()
  //       cycleCount += 1
  //     }

  //     println(s"\n=== Test 2 Summary ===")
  //     println(s"  Total cycles: $cycleCount")
  //     println(s"  Read requests:  $readRequests")
  //     println(s"  Write requests: $writeRequests")
  //     println(s"  Data read:   $dataRead, Data written: $dataWritten")

  //     assert(cycleCount < maxCycles, s"Test timed out after $maxCycles cycles")

  //     val expectedAccesses = totalIterations * numChannels
  //     assert(readRequests == expectedAccesses,
  //       s"Expected $expectedAccesses read requests, got $readRequests")
  //     assert(writeRequests == expectedAccesses,
  //       s"Expected $expectedAccesses write requests, got $writeRequests")
  //     assert(dataRead == totalIterations,
  //       s"Expected $totalIterations reader outputs, got $dataRead")
  //     assert(dataWritten == totalIterations,
  //       s"Expected $totalIterations writer inputs, got $dataWritten")

  //     // Verify final memory state: enumerate all (i0, i1)
  //     var errors = 0
  //     for (i1 <- 0 until tBounds(1); i0 <- 0 until tBounds(0); ch <- 0 until numChannels) {
  //       val addr     = basePtr + i0 * tStrides(0) + i1 * tStrides(1) + ch * spatialStride
  //       val original = addr & mask64
  //       val expected = (original + 1) & mask64
  //       val actual   = tcdmMemory.getOrElse(addr, BigInt(-1))
  //       if (actual != expected) {
  //         println(f"  MISMATCH at 0x$addr%04x: expected=0x$expected%016x actual=0x$actual%016x")
  //         errors += 1
  //       }
  //     }
  //     if (errors == 0)
  //       println(s"  All ${totalIterations * numChannels} memory locations verified!")
  //     assert(errors == 0, s"$errors memory locations had incorrect values")
  // }

  // // ---- Test 3: Writer backpressure (5 temporal dims, 1 active) ----
  // // Deliberately delay feeding data to the writer to test that the system
  // // handles backpressure without deadlock.
  // "ReaderWriter: writer backpressure handling" should "pass" in test(
  //   new ReaderWriter(
  //     new ReaderWriterParam(
  //       tcdmDataWidth      = 64,
  //       numChannel         = 32,
  //       spatialBounds      = List(32),
  //       temporalDimension  = 5,
  //       addressBufferDepth = 2,
  //       dataBufferDepth    = 2,
  //       tcdmSize           = 128
  //     ),
  //     new ReaderWriterParam(
  //       tcdmDataWidth      = 64,
  //       numChannel         = 32,
  //       spatialBounds      = List(32),
  //       temporalDimension  = 5,
  //       addressBufferDepth = 2,
  //       dataBufferDepth    = 2,
  //       tcdmSize           = 128
  //     )
  //   )
  // ).withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) {
  //   dut =>
  //     dut.clock.setTimeout(0)

  //     val numChannels    = 32
  //     val dataWidth      = 64
  //     val mask64         = (BigInt(1) << 64) - 1
  //     val basePtr: BigInt = 0x3000
  //     val spatialStride  = 32
  //     // 1 active temporal dim: stride=64, bound=4
  //     val tBounds = Seq(2, 2, 8, 4, 4)
  //     val tStrides = Seq(256, 2048, 0, 512, 4096)
  //     val totalIterations = tBounds.product // 4

  //     val tcdmMemory = mutable.HashMap[BigInt, BigInt]()
  //     for (iter <- 0 until totalIterations) {
  //       for (ch <- 0 until numChannels) {
  //         val addr = basePtr + iter * tStrides(0) + ch * spatialStride
  //         tcdmMemory(addr) = addr & mask64
  //       }
  //     }

  //     println("\n=== Test 3: Writer backpressure (5 temporal dims) ===")

  //     configureAgu(dut.io.readerInterface.aguCfg, basePtr, spatialStride, tStrides, tBounds)
  //     configureAgu(dut.io.writerInterface.aguCfg, basePtr, spatialStride, tStrides, tBounds)

  //     val enableMask = (BigInt(1) << numChannels) - 1
  //     if (dut.io.readerInterface.readerwriterCfg.enabledChannel.getWidth > 0)
  //       dut.io.readerInterface.readerwriterCfg.enabledChannel.poke(enableMask.U)
  //     if (dut.io.writerInterface.readerwriterCfg.enabledChannel.getWidth > 0)
  //       dut.io.writerInterface.readerwriterCfg.enabledChannel.poke(enableMask.U)

  //     // Initialize ports
  //     dut.io.readerInterface.tcdmRsp.foreach { rsp =>
  //       rsp.valid.poke(false.B)
  //       rsp.bits.data.poke(0.U)
  //     }
  //     dut.io.readerInterface.data.ready.poke(false.B)
  //     dut.io.writerInterface.data.valid.poke(false.B)
  //     dut.io.writerInterface.data.bits.poke(0.U)

  //     // Start both
  //     dut.io.readerInterface.start.poke(true.B)
  //     dut.io.writerInterface.start.poke(true.B)
  //     dut.clock.step()
  //     dut.io.readerInterface.start.poke(false.B)
  //     dut.io.writerInterface.start.poke(false.B)

  //     val pendingReadAddr = Array.fill(numChannels)(Option.empty[BigInt])
  //     val dataQueue       = mutable.Queue[BigInt]()

  //     var cycleCount   = 0
  //     val maxCycles    = 3000
  //     var readRequests  = 0
  //     var writeRequests = 0
  //     var dataRead      = 0
  //     var dataWritten   = 0

  //     while (
  //       cycleCount < maxCycles &&
  //       (dut.io.readerInterface.busy.peek().litToBoolean ||
  //         dut.io.writerInterface.busy.peek().litToBoolean ||
  //         !dut.io.readerInterface.bufferEmpty.peek().litToBoolean ||
  //         !dut.io.writerInterface.bufferEmpty.peek().litToBoolean ||
  //         dataQueue.nonEmpty)
  //     ) {
  //       val (rr, wr) = tcdmStep(dut, numChannels, dataWidth, tcdmMemory, pendingReadAddr)
  //       readRequests += rr
  //       writeRequests += wr

  //       // reader output -> transform -> enqueue
  //       if (dut.io.readerInterface.data.valid.peek().litToBoolean) {
  //         val readData = dut.io.readerInterface.data.bits.peek().litValue
  //         dataQueue.enqueue(transformData(readData, numChannels, dataWidth))
  //         dataRead += 1
  //       }
  //       dut.io.readerInterface.data.ready.poke(true.B)

  //       // Backpressure: only feed data to writer every 5 cycles
  //       val allowWriterFeed = (cycleCount % 5 == 0) && dataQueue.nonEmpty
  //       if (allowWriterFeed) {
  //         dut.io.writerInterface.data.valid.poke(true.B)
  //         dut.io.writerInterface.data.bits.poke(dataQueue.head.U)
  //         if (dut.io.writerInterface.data.ready.peek().litToBoolean) {
  //           dataQueue.dequeue()
  //           dataWritten += 1
  //         }
  //       } else {
  //         dut.io.writerInterface.data.valid.poke(false.B)
  //         dut.io.writerInterface.data.bits.poke(0.U)
  //       }

  //       dut.clock.step()
  //       cycleCount += 1
  //     }

  //     println(s"\n=== Test 3 Summary ===")
  //     println(s"  Total cycles: $cycleCount")
  //     println(s"  Read requests:  $readRequests")
  //     println(s"  Write requests: $writeRequests")
  //     println(s"  Data read:   $dataRead, Data written: $dataWritten")

  //     assert(cycleCount < maxCycles, s"Test timed out - possible deadlock")

  //     val expectedAccesses = totalIterations * numChannels
  //     assert(readRequests == expectedAccesses,
  //       s"Expected $expectedAccesses read requests, got $readRequests")
  //     assert(writeRequests == expectedAccesses,
  //       s"Expected $expectedAccesses write requests, got $writeRequests")

  //     // Verify final memory state
  //     var errors = 0
  //     for (iter <- 0 until totalIterations) {
  //       for (ch <- 0 until numChannels) {
  //         val addr     = basePtr + iter * tStrides(0) + ch * spatialStride
  //         val original = addr & mask64
  //         val expected = (original + 1) & mask64
  //         val actual   = tcdmMemory.getOrElse(addr, BigInt(-1))
  //         if (actual != expected) {
  //           println(f"  MISMATCH at 0x$addr%04x: expected=0x$expected%016x actual=0x$actual%016x")
  //           errors += 1
  //         }
  //       }
  //     }
  //     if (errors == 0)
  //       println(s"  All ${totalIterations * numChannels} memory locations verified!")
  //     assert(errors == 0, s"$errors memory locations had incorrect values")
  // }
}
