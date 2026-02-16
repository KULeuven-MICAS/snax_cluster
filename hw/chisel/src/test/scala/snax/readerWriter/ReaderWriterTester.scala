package snax.readerWriter

import chisel3._
import chisel3.util._

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

class ReaderWriterTester extends AnyFlatSpec with ChiselScalatestTester {

  // Generate expected addresses based on AGU configuration
  def generateExpectedAddresses(
    ptr: BigInt,
    spatialStrides: Seq[Int],
    temporalStrides: Seq[Int],
    spatialBounds: Seq[Int],
    temporalBounds: Seq[Int]
  ): Seq[Seq[BigInt]] = {
    val addresses = mutable.ArrayBuffer[Seq[BigInt]]()
    
    val tBound0 = if (temporalBounds.length > 0) temporalBounds(0) else 1
    val tBound1 = if (temporalBounds.length > 1) temporalBounds(1) else 1
    
    val tStride0 = if (temporalStrides.length > 0) temporalStrides(0) else 0
    val tStride1 = if (temporalStrides.length > 1) temporalStrides(1) else 0
    
    val sBound = if (spatialBounds.length > 0) spatialBounds(0) else 1
    val sStride = if (spatialStrides.length > 0) spatialStrides(0) else 0

    for (t1 <- 0 until tBound1) {
      for (t0 <- 0 until tBound0) {
        val temporalOffset = t1 * tStride1 + t0 * tStride0
        val cycleAddresses = (0 until sBound).map { s0 =>
           ptr + temporalOffset + s0 * sStride
        }
        addresses += cycleAddresses
      }
    }
    addresses.toSeq
  }

  "ReaderWriter: sequential write then read test" should "pass" in test(
    new ReaderWriter(
      readerParam = new ReaderWriterParam(
        tcdmDataWidth      = 64,
        numChannel         = 8,
        spatialBounds      = List(8),
        temporalDimension  = 2,
        addressBufferDepth = 8,
        dataBufferDepth    = 8,
        tcdmSize           = 128
      ),
      writerParam = new ReaderWriterParam(
        tcdmDataWidth      = 64,
        numChannel         = 8,
        spatialBounds      = List(8),
        temporalDimension  = 2,
        addressBufferDepth = 8,
        dataBufferDepth    = 8,
        tcdmSize           = 128
      )
    )
  ).withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
      
      val random = new scala.util.Random(42)
      
      // ==========================================
      // PART 1: WRITE OPERATION
      // ==========================================
      println("Starting Write Operation...")
      
      val writePtr = 0x1000
      val spatialStride = 8
      val temporalStride0 = 64
      val temporalStride1 = 512 
      val temporalBound0 = 8
      val temporalBound1 = 2

      // Configure Writer
      dut.io.writerInterface.aguCfg.ptr.poke(writePtr.U)
      dut.io.writerInterface.aguCfg.spatialStrides(0).poke(spatialStride.U)
      dut.io.writerInterface.aguCfg.temporalStrides(0).poke(temporalStride0.U)
      dut.io.writerInterface.aguCfg.temporalStrides(1).poke(temporalStride1.U)
      dut.io.writerInterface.aguCfg.temporalBounds(0).poke(temporalBound0.U)
      dut.io.writerInterface.aguCfg.temporalBounds(1).poke(temporalBound1.U)

      // Enable all channels for writer
      if (dut.io.writerInterface.readerwriterCfg.enabledChannel.getWidth > 0) {
        dut.io.writerInterface.readerwriterCfg.enabledChannel.poke(0xFF.U)
      }

      // Initialize ports
      dut.io.readerInterface.tcdmReq.foreach { req =>
        req.ready.poke(false.B)
      }
      dut.io.writerInterface.data.valid.poke(false.B)
      dut.io.writerInterface.data.bits.poke(0.U)
      
      // Reader should be idle
      dut.io.readerInterface.start.poke(false.B)
      dut.io.readerInterface.data.ready.poke(false.B)

      // Start Writer
      dut.io.writerInterface.start.poke(true.B)
      dut.clock.step()
      dut.io.writerInterface.start.poke(false.B)

      var cycleCount = 0
      val maxCycles = 2000
      
      val numWrites = temporalBound0 * temporalBound1
      val inputData = (0 until numWrites).map { i => 
        BigInt(512, random) 
      }
      
      val expectedWriteAddresses = generateExpectedAddresses(
        writePtr, List(spatialStride), List(temporalStride0, temporalStride1), List(8), List(temporalBound0, temporalBound1)
      )

      var inputIdx = 0
      var writesReceived = Array.fill(8)(0)
      
      while (cycleCount < maxCycles && (dut.io.writerInterface.busy.peek().litToBoolean || !dut.io.writerInterface.bufferEmpty.peek().litToBoolean || inputIdx < numWrites)) {
         // Feed Data to Writer
         val dataValid = random.nextBoolean()
         if (inputIdx < numWrites && dataValid) {
             dut.io.writerInterface.data.valid.poke(true.B)
             dut.io.writerInterface.data.bits.poke(inputData(inputIdx).U)
         } else {
             dut.io.writerInterface.data.valid.poke(false.B)
         }
         
         if (inputIdx < numWrites && dataValid && dut.io.writerInterface.data.ready.peek().litToBoolean) {
             inputIdx += 1
         }

         // Accept TCDM Requests (Shared Port)
         dut.io.readerInterface.tcdmReq.zipWithIndex.foreach { case (req, idx) =>
             val ready = random.nextBoolean()
             req.ready.poke(ready.B)
             
             if (req.valid.peek().litToBoolean && ready) {
                 val addr = req.bits.addr.peek().litValue
                 val data = req.bits.data.peek().litValue
                 val write = req.bits.write.peek().litToBoolean
                 
                 assert(write, s"Channel $idx should be writing during write phase")
                 
                 val writeIdx = writesReceived(idx)
                 if (writeIdx < expectedWriteAddresses.length) {
                     val expectedAddr = expectedWriteAddresses(writeIdx)(idx)
                     assert(addr == expectedAddr, s"Channel $idx write $writeIdx: expected addr 0x${expectedAddr.toString(16)}, got 0x${addr.toString(16)}")
                     
                     val mask = (BigInt(1) << 64) - 1
                     val expectedData = (inputData(writeIdx) >> (idx * 64)) & mask
                     assert(data == expectedData, s"Channel $idx write $writeIdx: expected data 0x${expectedData.toString(16)}, got 0x${data.toString(16)}")
                     
                     writesReceived(idx) += 1
                 }
             }
         }
         
         dut.clock.step()
         cycleCount += 1
      }
      
      assert(inputIdx == numWrites, "Did not finish sending all input data")
      assert(writesReceived.forall(_ == numWrites), "Did not receive all write requests")
      println("Write Operation Finished Successfully.")

      // ==========================================
      // PART 2: READ OPERATION
      // ==========================================
      println("Starting Read Operation...")
      
      val readPtr = 0x2000
      // Same config for reader
      dut.io.readerInterface.aguCfg.ptr.poke(readPtr.U)
      dut.io.readerInterface.aguCfg.spatialStrides(0).poke(spatialStride.U)
      dut.io.readerInterface.aguCfg.temporalStrides(0).poke(temporalStride0.U)
      dut.io.readerInterface.aguCfg.temporalStrides(1).poke(temporalStride1.U)
      dut.io.readerInterface.aguCfg.temporalBounds(0).poke(temporalBound0.U)
      dut.io.readerInterface.aguCfg.temporalBounds(1).poke(temporalBound1.U)

      if (dut.io.readerInterface.readerwriterCfg.enabledChannel.getWidth > 0) {
        dut.io.readerInterface.readerwriterCfg.enabledChannel.poke(0xFF.U)
      }

      // Start Reader
      dut.io.readerInterface.start.poke(true.B)
      dut.clock.step()
      dut.io.readerInterface.start.poke(false.B)
      
      val expectedReadAddresses = generateExpectedAddresses(
        readPtr, List(spatialStride), List(temporalStride0, temporalStride1), List(8), List(temporalBound0, temporalBound1)
      )
      
      // We will simulate TCDM memory by returning address as data
      // TCDM Port State for latency simulation
      case class TcdmPortState(addr: BigInt, cyclesRemaining: Int)
      val tcdmPortStates = Array.fill(8)(Option.empty[TcdmPortState])
      
      var readsSent = Array.fill(8)(0)
      var dataReceived = 0
      val numReads = numWrites // Same dimensions
      
      cycleCount = 0
      
      while (cycleCount < maxCycles && (dut.io.readerInterface.busy.peek().litToBoolean || !dut.io.readerInterface.bufferEmpty.peek().litToBoolean || dataReceived < numReads)) {
        
        // Handle TCDM Requests and Responses
        for (channelIdx <- 0 until 8) {
          tcdmPortStates(channelIdx) match {
            case None =>
              // Idle, accept new request
              dut.io.readerInterface.tcdmReq(channelIdx).ready.poke(true.B)
              dut.io.readerInterface.tcdmRsp(channelIdx).valid.poke(false.B)
              
              if (dut.io.readerInterface.tcdmReq(channelIdx).valid.peek().litToBoolean) {
                 val addr = dut.io.readerInterface.tcdmReq(channelIdx).bits.addr.peek().litValue
                 val write = dut.io.readerInterface.tcdmReq(channelIdx).bits.write.peek().litToBoolean
                 assert(!write, s"Channel $channelIdx should be reading during read phase")
                 
                 val readIdx = readsSent(channelIdx)
                 if (readIdx < expectedReadAddresses.length) {
                     val expectedAddr = expectedReadAddresses(readIdx)(channelIdx)
                     assert(addr == expectedAddr, s"Channel $channelIdx read $readIdx: expected addr 0x${expectedAddr.toString(16)}, got 0x${addr.toString(16)}")
                     readsSent(channelIdx) += 1
                     
                     // Schedule response
                     val latency = random.nextInt(4)
                     tcdmPortStates(channelIdx) = Some(TcdmPortState(addr, latency))
                 }
              }
              
            case Some(state) =>
              dut.io.readerInterface.tcdmReq(channelIdx).ready.poke(false.B)
              
              if (state.cyclesRemaining == 0) {
                  dut.io.readerInterface.tcdmRsp(channelIdx).valid.poke(true.B)
                  dut.io.readerInterface.tcdmRsp(channelIdx).bits.data.poke(state.addr.U) // Return addr as data
                  tcdmPortStates(channelIdx) = None
              } else {
                  dut.io.readerInterface.tcdmRsp(channelIdx).valid.poke(false.B)
                  tcdmPortStates(channelIdx) = Some(state.copy(cyclesRemaining = state.cyclesRemaining - 1))
              }
          }
        }
        
        // Accept Output Data
        dut.io.readerInterface.data.ready.poke(true.B)
        if (dut.io.readerInterface.data.valid.peek().litToBoolean) {
            val outData = dut.io.readerInterface.data.bits.peek().litValue
            // Verify data
            // We expect data to be the address.
            // Reconstruct expected data for this transaction
            
            if (dataReceived < expectedReadAddresses.length) {
                var expectedCombinedData = BigInt(0)
                for (c <- 0 until 8) {
                    val addr = expectedReadAddresses(dataReceived)(c)
                    expectedCombinedData = expectedCombinedData | (addr << (c * 64))
                }
                assert(outData == expectedCombinedData, s"Output $dataReceived: expected 0x${expectedCombinedData.toString(16)}, got 0x${outData.toString(16)}")
            }
            
            dataReceived += 1
        }
        
        dut.clock.step()
        cycleCount += 1
      }
      
      assert(dataReceived == numReads, s"Expected $numReads data outputs, got $dataReceived")
      println("Read Operation Finished Successfully.")
  }
}
