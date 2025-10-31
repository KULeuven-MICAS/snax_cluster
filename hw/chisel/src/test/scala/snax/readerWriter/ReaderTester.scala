package snax.readerWriter

import chisel3._
import chisel3.util._

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable

class ReaderTester extends AnyFlatSpec with ChiselScalatestTester {

  println(
    getVerilogString(
      new Reader(
        new ReaderWriterParam(
          tcdmDataWidth = 64,
          numChannel    = 8,
          spatialBounds = List(8)
        )
      )
    )
  )

  "Reader: basic sequential read test" should "pass" in test(
    new Reader(
      new ReaderWriterParam(
        tcdmDataWidth      = 64,
        numChannel         = 8,
        spatialBounds      = List(8),
        temporalDimension  = 2,
        addressBufferDepth = 2,
        dataBufferDepth    = 2,
        tcdmSize           = 128
      )
    )
  )
    .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
      // Configure the address generator
      dut.io.aguCfg.ptr.poke(0x1000.U)
      dut.io.aguCfg.spatialStrides(0).poke(8)
      dut.io.aguCfg.temporalStrides(0).poke(64)
      dut.io.aguCfg.temporalStrides(1).poke(0)
      dut.io.aguCfg.temporalBounds(0).poke(4)
      dut.io.aguCfg.temporalBounds(1).poke(1)

      // Configure reader/writer settings
      if (dut.io.readerwriterCfg.enabledChannel.getWidth > 0) {
        dut.io.readerwriterCfg.enabledChannel.poke(0xFF.U) // All channels enabled
      }

      // Initialize TCDM response ports
      dut.io.tcdmRsp.foreach { rsp =>
        rsp.valid.poke(false.B)
        rsp.bits.data.poke(0.U)
      }

      // Initialize output data ready signal
      dut.io.data.ready.poke(false.B)

      // Start the reader
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      // Queue to track expected addresses
      val expectedAddresses = mutable.Queue[BigInt]()
      for (t <- 0 until 4) {
        for (c <- 0 until 8) {
          expectedAddresses.enqueue(0x1000 + t * 64 + c * 8)
        }
      }

      // Queue to store request-response pairs
      val pendingRequests = mutable.Queue[(Int, BigInt)]() // (channel, address)

      // Simulate TCDM memory with dummy data
      // Data pattern: address value itself for easy verification
      def getTcdmData(addr: BigInt): BigInt = {
        addr & 0xFFFFFFFFFFFFFFFFl
      }

      var cycleCount    = 0
      val maxCycles     = 200
      var requestsSeen  = 0
      var responsesGiven = 0
      var dataReceived  = 0

      println(s"Starting simulation...")

      // Main test loop
      while (cycleCount < maxCycles && (dut.io.busy.peek().litToBoolean || !dut.io.bufferEmpty.peek().litToBoolean)) {
        
        // Process TCDM requests
        for (channelIdx <- 0 until 8) {
          val tcdmReq = dut.io.tcdmReq(channelIdx)
          
          if (tcdmReq.valid.peek().litToBoolean && tcdmReq.ready.peek().litToBoolean) {
            val addr  = tcdmReq.bits.addr.peek().litValue
            val write = tcdmReq.bits.write.peek().litToBoolean
            val strb  = tcdmReq.bits.strb.peek().litValue
            
            println(f"Cycle $cycleCount%3d: Channel $channelIdx - TCDM Request: addr=0x$addr%04x, write=$write, strb=0x$strb%02x")
            
            // Verify it's a read request
            assert(!write, s"Reader should only send read requests, got write at channel $channelIdx")
            
            // Verify address is expected
            if (expectedAddresses.nonEmpty) {
              val expected = expectedAddresses.dequeue()
              assert(addr == expected, s"Address mismatch at channel $channelIdx: expected 0x${expected.toString(16)}, got 0x${addr.toString(16)}")
            }
            
            requestsSeen += 1
            pendingRequests.enqueue((channelIdx, addr))
          }
        }

        // Provide TCDM responses for pending requests
        // Simulate 1 cycle latency
        if (pendingRequests.nonEmpty) {
          val (channelIdx, addr) = pendingRequests.dequeue()
          val data = getTcdmData(addr)
          
          dut.io.tcdmRsp(channelIdx).valid.poke(true.B)
          dut.io.tcdmRsp(channelIdx).bits.data.poke(data.U)
          
          println(f"Cycle $cycleCount%3d: Channel $channelIdx - TCDM Response: data=0x$data%016x")
          responsesGiven += 1
        } else {
          // No response this cycle
          dut.io.tcdmRsp.foreach { rsp =>
            rsp.valid.poke(false.B)
          }
        }

        // Accept output data when available
        if (dut.io.data.valid.peek().litToBoolean) {
          dut.io.data.ready.poke(true.B)
          val outputData = dut.io.data.bits.peek().litValue
          println(f"Cycle $cycleCount%3d: Output Data: 0x$outputData%0128x")
          dataReceived += 1
        } else {
          dut.io.data.ready.poke(false.B)
        }

        dut.clock.step()
        cycleCount += 1
      }

      println(s"\nTest Summary:")
      println(s"  Total cycles: $cycleCount")
      println(s"  Requests seen: $requestsSeen")
      println(s"  Responses given: $responsesGiven")
      println(s"  Data received: $dataReceived")
      println(s"  Expected requests: 32 (4 temporal * 8 channels)")

      // Verify we saw all expected requests
      assert(requestsSeen == 32, s"Expected 32 requests, got $requestsSeen")
      assert(responsesGiven == 32, s"Expected 32 responses, got $responsesGiven")
      assert(dataReceived >= 4, s"Expected at least 4 data outputs, got $dataReceived")
    }

  "Reader: continuous fetch with first temporal loop disabled" should "pass" in test(
    new Reader(
      new ReaderWriterParam(
        tcdmDataWidth      = 64,
        numChannel         = 8,
        spatialBounds      = List(8),
        temporalDimension  = 2,
        addressBufferDepth = 2,
        dataBufferDepth    = 2,
        tcdmSize           = 128
      )
    )
  )
    .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
      // Configure the address generator - first temporal loop disabled
      dut.io.aguCfg.ptr.poke(0x2000.U)
      dut.io.aguCfg.spatialStrides(0).poke(8)
      dut.io.aguCfg.temporalStrides(0).poke(64)
      dut.io.aguCfg.temporalStrides(1).poke(64)
      dut.io.aguCfg.temporalBounds(0).poke(0) // Disabled
      dut.io.aguCfg.temporalBounds(1).poke(8)

      // Configure reader/writer settings
      if (dut.io.readerwriterCfg.enabledChannel.getWidth > 0) {
        dut.io.readerwriterCfg.enabledChannel.poke(0xFF.U)
      }

      // Initialize ports
      dut.io.tcdmRsp.foreach { rsp =>
        rsp.valid.poke(false.B)
        rsp.bits.data.poke(0.U)
      }
      dut.io.data.ready.poke(false.B)

      // Start the reader
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      val pendingRequests = mutable.Queue[(Int, BigInt)]()
      var cycleCount = 0
      val maxCycles = 300

      def getTcdmData(addr: BigInt): BigInt = addr & 0xFFFFFFFFFFFFFFFFl

      while (cycleCount < maxCycles && (dut.io.busy.peek().litToBoolean || !dut.io.bufferEmpty.peek().litToBoolean)) {
        
        // Process requests
        for (channelIdx <- 0 until 8) {
          val tcdmReq = dut.io.tcdmReq(channelIdx)
          if (tcdmReq.valid.peek().litToBoolean && tcdmReq.ready.peek().litToBoolean) {
            val addr = tcdmReq.bits.addr.peek().litValue
            pendingRequests.enqueue((channelIdx, addr))
          }
        }

        // Provide responses
        if (pendingRequests.nonEmpty) {
          val (channelIdx, addr) = pendingRequests.dequeue()
          val data = getTcdmData(addr)
          dut.io.tcdmRsp(channelIdx).valid.poke(true.B)
          dut.io.tcdmRsp(channelIdx).bits.data.poke(data.U)
        } else {
          dut.io.tcdmRsp.foreach(_.valid.poke(false.B))
        }

        // Accept output data
        dut.io.data.ready.poke(true.B)

        dut.clock.step()
        cycleCount += 1
      }

      println(s"Test completed in $cycleCount cycles")
    }

  "Reader: test with channel masking" should "pass" in test(
    new Reader(
      new ReaderWriterParam(
        tcdmDataWidth        = 64,
        numChannel           = 8,
        spatialBounds        = List(8),
        temporalDimension    = 2,
        addressBufferDepth   = 2,
        dataBufferDepth      = 2,
        tcdmSize             = 128,
        configurableChannel  = true
      )
    )
  )
    .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
      // Configure the address generator
      dut.io.aguCfg.ptr.poke(0x3000.U)
      dut.io.aguCfg.spatialStrides(0).poke(8)
      dut.io.aguCfg.temporalStrides(0).poke(64)
      dut.io.aguCfg.temporalStrides(1).poke(0)
      dut.io.aguCfg.temporalBounds(0).poke(2)
      dut.io.aguCfg.temporalBounds(1).poke(1)

      // Enable only channels 0, 1, 2, 3 (lower 4 channels)
      dut.io.readerwriterCfg.enabledChannel.poke(0x0F.U)

      // Initialize ports
      dut.io.tcdmRsp.foreach { rsp =>
        rsp.valid.poke(false.B)
        rsp.bits.data.poke(0.U)
      }
      dut.io.data.ready.poke(false.B)

      // Start the reader
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      val pendingRequests = mutable.Queue[(Int, BigInt)]()
      var cycleCount = 0
      val maxCycles = 200
      var requestsPerChannel = Array.fill(8)(0)

      def getTcdmData(addr: BigInt): BigInt = addr & 0xFFFFFFFFFFFFFFFFl

      while (cycleCount < maxCycles && (dut.io.busy.peek().litToBoolean || !dut.io.bufferEmpty.peek().litToBoolean)) {
        
        // Process requests and track per channel
        for (channelIdx <- 0 until 8) {
          val tcdmReq = dut.io.tcdmReq(channelIdx)
          if (tcdmReq.valid.peek().litToBoolean && tcdmReq.ready.peek().litToBoolean) {
            val addr = tcdmReq.bits.addr.peek().litValue
            requestsPerChannel(channelIdx) += 1
            pendingRequests.enqueue((channelIdx, addr))
          }
        }

        // Provide responses
        if (pendingRequests.nonEmpty) {
          val (channelIdx, addr) = pendingRequests.dequeue()
          val data = getTcdmData(addr)
          dut.io.tcdmRsp(channelIdx).valid.poke(true.B)
          dut.io.tcdmRsp(channelIdx).bits.data.poke(data.U)
        } else {
          dut.io.tcdmRsp.foreach(_.valid.poke(false.B))
        }

        // Accept output data
        dut.io.data.ready.poke(true.B)

        dut.clock.step()
        cycleCount += 1
      }

      println(s"\nChannel masking test summary:")
      for (i <- 0 until 8) {
        println(s"  Channel $i: ${requestsPerChannel(i)} requests")
      }

      // Verify only enabled channels got requests
      for (i <- 0 until 4) {
        assert(requestsPerChannel(i) > 0, s"Enabled channel $i should have requests")
      }
      for (i <- 4 until 8) {
        assert(requestsPerChannel(i) == 0, s"Disabled channel $i should have no requests")
      }
    }

  "Reader: backpressure handling test" should "pass" in test(
    new Reader(
      new ReaderWriterParam(
        tcdmDataWidth      = 64,
        numChannel         = 8,
        spatialBounds      = List(8),
        temporalDimension  = 2,
        addressBufferDepth = 2,
        dataBufferDepth    = 2,
        tcdmSize           = 128
      )
    )
  )
    .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
      // Configure the address generator
      dut.io.aguCfg.ptr.poke(0x4000.U)
      dut.io.aguCfg.spatialStrides(0).poke(8)
      dut.io.aguCfg.temporalStrides(0).poke(64)
      dut.io.aguCfg.temporalStrides(1).poke(0)
      dut.io.aguCfg.temporalBounds(0).poke(3)
      dut.io.aguCfg.temporalBounds(1).poke(1)

      if (dut.io.readerwriterCfg.enabledChannel.getWidth > 0) {
        dut.io.readerwriterCfg.enabledChannel.poke(0xFF.U)
      }

      // Initialize ports
      dut.io.tcdmRsp.foreach { rsp =>
        rsp.valid.poke(false.B)
        rsp.bits.data.poke(0.U)
      }
      dut.io.data.ready.poke(false.B)

      // Start the reader
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      val pendingRequests = mutable.Queue[(Int, BigInt)]()
      var cycleCount = 0
      val maxCycles = 400

      def getTcdmData(addr: BigInt): BigInt = addr & 0xFFFFFFFFFFFFFFFFl

      while (cycleCount < maxCycles && (dut.io.busy.peek().litToBoolean || !dut.io.bufferEmpty.peek().litToBoolean)) {
        
        // Process requests
        for (channelIdx <- 0 until 8) {
          val tcdmReq = dut.io.tcdmReq(channelIdx)
          if (tcdmReq.valid.peek().litToBoolean && tcdmReq.ready.peek().litToBoolean) {
            val addr = tcdmReq.bits.addr.peek().litValue
            pendingRequests.enqueue((channelIdx, addr))
          }
        }

        // Provide responses
        if (pendingRequests.nonEmpty) {
          val (channelIdx, addr) = pendingRequests.dequeue()
          val data = getTcdmData(addr)
          dut.io.tcdmRsp(channelIdx).valid.poke(true.B)
          dut.io.tcdmRsp(channelIdx).bits.data.poke(data.U)
        } else {
          dut.io.tcdmRsp.foreach(_.valid.poke(false.B))
        }

        // Intermittent backpressure - only accept data every 3 cycles
        if (cycleCount % 3 == 0) {
          dut.io.data.ready.poke(true.B)
        } else {
          dut.io.data.ready.poke(false.B)
        }

        dut.clock.step()
        cycleCount += 1
      }

      println(s"Backpressure test completed in $cycleCount cycles")
      assert(cycleCount < maxCycles, "Test should complete before timeout")
    }
}

object ReaderEmitterTest extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Reader(
      new ReaderWriterParam(
        tcdmDataWidth      = 64,
        numChannel         = 8,
        spatialBounds      = List(8),
        temporalDimension  = 2,
        addressBufferDepth = 8,
        dataBufferDepth    = 8,
        tcdmSize           = 128
      )
    ),
    args = Array("--target-dir", "generated/reader")
  )
}
