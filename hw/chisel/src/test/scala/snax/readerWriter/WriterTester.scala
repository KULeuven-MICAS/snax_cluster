package snax.readerWriter

import chisel3._
import chisel3.util._

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

class WriterTester extends AnyFlatSpec with ChiselScalatestTester {

  // Generate expected addresses based on AGU configuration
  // Returns a sequence of (sequence of addresses for each channel)
  // Outer Seq: Temporal steps
  // Inner Seq: Channel addresses
  def generateExpectedAddresses(
    ptr: BigInt,
    spatialStrides: Seq[Int],
    temporalStrides: Seq[Int],
    spatialBounds: Seq[Int],
    temporalBounds: Seq[Int]
  ): Seq[Seq[BigInt]] = {
    val addresses = mutable.ArrayBuffer[Seq[BigInt]]()
    
    // Assuming 2D temporal loop for this test helper, as per common usage
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

  "Writer: basic sequential write test" should "pass" in test(
    new Writer(
      new ReaderWriterParam(
        tcdmDataWidth      = 64,
        numChannel         = 8,
        spatialBounds      = List(8),
        temporalDimension  = 3,
        addressBufferDepth = 8,
        dataBufferDepth    = 8,
        tcdmSize           = 128
      )
    )
  ).withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
      // Configuration
      val ptr = 0x1000
      val spatialStride = 8
      val temporalStride0 = 64
      val temporalStride1 = 512
      val temporalBound0 = 8
      val temporalBound1 = 2

      dut.io.aguCfg.ptr.poke(ptr.U)
      dut.io.aguCfg.spatialStrides(0).poke(spatialStride.U)
      dut.io.aguCfg.temporalStrides(0).poke(temporalStride0.U)
      dut.io.aguCfg.temporalStrides(1).poke(temporalStride1.U)
      dut.io.aguCfg.temporalBounds(0).poke(temporalBound0.U)
      dut.io.aguCfg.temporalBounds(1).poke(temporalBound1.U)

      // Enable all channels
      if (dut.io.readerwriterCfg.enabledChannel.getWidth > 0) {
        dut.io.readerwriterCfg.enabledChannel.poke(0xFF.U)
      }
      
      // Initialize ports
      dut.io.tcdmReq.foreach { req =>
        req.ready.poke(false.B)
      }
      dut.io.data.valid.poke(false.B)
      dut.io.data.bits.poke(0.U)

      // Start
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      val random = new scala.util.Random(42)
      var cycleCount = 0
      val maxCycles = 1000
      
      // Generate input data
      val numWrites = temporalBound0 * temporalBound1
      val inputData = (0 until numWrites).map { i => 
        BigInt(512, random) // 8 channels * 64 bits = 512 bits
      }
      
      // Expected addresses
      val expectedAddresses = generateExpectedAddresses(
        ptr, List(spatialStride), List(temporalStride0, temporalStride1), List(8), List(temporalBound0, temporalBound1)
      )

      var inputIdx = 0
      var writesReceived = Array.fill(8)(0) // Per channel
      
      println(s"Starting Writer test...")
      println(s"Total expected writes per channel: $numWrites")

      while (cycleCount < maxCycles && (dut.io.busy.peek().litToBoolean || !dut.io.bufferEmpty.peek().litToBoolean || inputIdx < numWrites)) {
        
        // 1. Feed Data
        // Randomly toggle valid to stress test
        val dataValid = random.nextBoolean()
        if (inputIdx < numWrites && dataValid) {
            dut.io.data.valid.poke(true.B)
            dut.io.data.bits.poke(inputData(inputIdx).U)
        } else {
            dut.io.data.valid.poke(false.B)
        }
        
        // Check if data was accepted
        if (inputIdx < numWrites && dataValid && dut.io.data.ready.peek().litToBoolean) {
            inputIdx += 1
        }

        // 2. Accept TCDM Requests
        // Randomly ready to simulate contention
        dut.io.tcdmReq.zipWithIndex.foreach { case (req, idx) =>
            val ready = random.nextBoolean()
            req.ready.poke(ready.B)
            
            if (req.valid.peek().litToBoolean && ready) {
                val addr = req.bits.addr.peek().litValue
                val data = req.bits.data.peek().litValue
                val write = req.bits.write.peek().litToBoolean
                
                assert(write, s"Channel $idx should be writing")
                
                val writeIdx = writesReceived(idx)
                if (writeIdx < expectedAddresses.length) {
                    val expectedAddr = expectedAddresses(writeIdx)(idx)
                    assert(addr == expectedAddr, s"Channel $idx write $writeIdx: expected addr 0x${expectedAddr.toString(16)}, got 0x${addr.toString(16)}")
                    
                    // Verify Data
                    // Channel i corresponds to bits (i+1)*64-1 downto i*64
                    // Mask is 64 bits of 1s
                    val mask = (BigInt(1) << 64) - 1
                    val expectedData = (inputData(writeIdx) >> (idx * 64)) & mask
                    assert(data == expectedData, s"Channel $idx write $writeIdx: expected data 0x${expectedData.toString(16)}, got 0x${data.toString(16)}")
                    
                    writesReceived(idx) += 1
                } else {
                    fail(s"Channel $idx received unexpected write request (idx $writeIdx >= ${expectedAddresses.length})")
                }
            }
        }

        dut.clock.step()
        cycleCount += 1
      }
      
      println(s"Test finished at cycle $cycleCount")
      println(s"Input index: $inputIdx / $numWrites")
      println(s"Writes received: ${writesReceived.mkString(", ")}")

      assert(inputIdx == numWrites, "Did not finish sending all input data")
      assert(writesReceived.forall(_ == numWrites), "Did not receive all write requests on all channels")
  }
}
