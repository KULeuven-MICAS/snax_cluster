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
        temporalDimension  = 3,
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
      dut.io.aguCfg.temporalStrides(2).poke(256)
      dut.io.aguCfg.temporalBounds(0).poke(4)
      dut.io.aguCfg.temporalBounds(1).poke(4)
      dut.io.aguCfg.temporalBounds(2).poke(2)

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

      // TCDM port state: per-port pending request tracking
      // Each port can have one pending request with remaining cycles until response
      case class TcdmPortState(addr: BigInt, cyclesRemaining: Int)
      val tcdmPortStates = Array.fill(8)(Option.empty[TcdmPortState])

      // Random number generator for latency (0-3 cycles)
      val random = new scala.util.Random(42) // Fixed seed for reproducibility

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

      // Main test loop - use proper synchronization pattern:
      // 1. Set control signals based on PREVIOUS cycle's observations
      // 2. Clock step (handshake occurs at posedge)
      // 3. Observe NEW DUT state
      // 4. Update internal state for NEXT cycle
      while (cycleCount < maxCycles && (dut.io.busy.peek().litToBoolean || !dut.io.bufferEmpty.peek().litToBoolean)) {
        
        // PHASE 1: Set control signals based on CURRENT state (before clock edge)
        // Set ready signals and response valids
        for (channelIdx <- 0 until 8) {
          tcdmPortStates(channelIdx) match {
            case None =>
              // Port is idle, ready to accept new request
              dut.io.tcdmReq(channelIdx).ready.poke(true.B)
              dut.io.tcdmRsp(channelIdx).valid.poke(false.B)
              
            case Some(state) =>
              // Port is busy processing a request
              dut.io.tcdmReq(channelIdx).ready.poke(false.B)
              
              if (state.cyclesRemaining == 0) {
                // Response is ready this cycle
                val data = getTcdmData(state.addr)
                dut.io.tcdmRsp(channelIdx).valid.poke(true.B)
                dut.io.tcdmRsp(channelIdx).bits.data.poke(data.U)
                
                println(f"Cycle $cycleCount%3d: Channel $channelIdx - TCDM Response: data=0x$data%016x")
                responsesGiven += 1
              } else {
                // Still waiting for response
                dut.io.tcdmRsp(channelIdx).valid.poke(false.B)
              }
          }
        }
        
        // Check for output data before setting ready
        val outputDataValid = dut.io.data.valid.peek().litToBoolean
        if (outputDataValid) {
          val outputData = dut.io.data.bits.peek().litValue
          println(f"Cycle $cycleCount%3d: Output Data handshake will occur: 0x$outputData%0128x")
          dataReceived += 1
        }
        
        // Set output data ready signal
        dut.io.data.ready.poke(true.B)
        
        // PHASE 2: Clock step - DUT now sees our control signals and handshakes occur
        dut.clock.step()
        cycleCount += 1
        
        // PHASE 3: Observe DUT outputs AFTER the clock edge
        // Check for new TCDM requests - but DON'T update state yet
        val newRequestsThisCycle = Array.fill(8)(Option.empty[(BigInt, Boolean, BigInt)])
        for (channelIdx <- 0 until 8) {
          if (tcdmPortStates(channelIdx).isEmpty && dut.io.tcdmReq(channelIdx).valid.peek().litToBoolean) {
            val addr  = dut.io.tcdmReq(channelIdx).bits.addr.peek().litValue
            val write = dut.io.tcdmReq(channelIdx).bits.write.peek().litToBoolean
            val strb  = dut.io.tcdmReq(channelIdx).bits.strb.peek().litValue
            
            println(f"Cycle $cycleCount%3d: Channel $channelIdx - TCDM Request: addr=0x$addr%04x, write=$write, strb=0x$strb%02x")
            
            // Verify it's a read request
            assert(!write, s"Reader should only send read requests, got write at channel $channelIdx")
            
            requestsSeen += 1
            newRequestsThisCycle(channelIdx) = Some((addr, write, strb))
            println(f"Cycle $cycleCount%3d: Channel $channelIdx - Request will be accepted next cycle")
          }
        }
        
        // PHASE 4: Update internal state for NEXT cycle
        for (channelIdx <- 0 until 8) {
          tcdmPortStates(channelIdx) match {
            case None =>
              // If we saw a new request, accept it now for next cycle
              newRequestsThisCycle(channelIdx) match {
                case Some((addr, _, _)) =>
                  val latency = random.nextInt(4)
                  tcdmPortStates(channelIdx) = Some(TcdmPortState(addr, latency))
                case None =>
                  // Stay idle
              }
              
            case Some(state) =>
              // Update state: decrement counter or clear if response was sent
              if (state.cyclesRemaining == 0) {
                // Response was sent this cycle, port becomes idle next cycle
                tcdmPortStates(channelIdx) = None
              } else {
                // Still waiting
                tcdmPortStates(channelIdx) = Some(state.copy(cyclesRemaining = state.cyclesRemaining - 1))
              }
          }
        }
      }

      println(s"\nTest Summary:")
      println(s"  Total cycles: $cycleCount")
      println(s"  Requests seen: $requestsSeen")
      println(s"  Responses given: $responsesGiven")
      println(s"  Data received: $dataReceived")
      println(s"  Expected requests: 256 (4 temporal * 8 channels)")

      // Verify we saw all expected requests
      assert(requestsSeen == 256, s"Expected 256 requests, got $requestsSeen")
      assert(responsesGiven == 256, s"Expected 256 responses, got $responsesGiven")
      assert(dataReceived == 32, s"Expected exactly 32 data outputs, got $dataReceived")
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
      dut.io.aguCfg.temporalBounds(0).poke(1) // Disabled
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

      // TCDM port state: per-port pending request tracking
      case class TcdmPortState(addr: BigInt, cyclesRemaining: Int)
      val tcdmPortStates = Array.fill(8)(Option.empty[TcdmPortState])
      val random = new scala.util.Random(43) // Fixed seed for reproducibility

      var cycleCount = 0
      val maxCycles = 300
      var requestsSeen = 0
      var responsesGiven = 0
      var dataReceived = 0

      def getTcdmData(addr: BigInt): BigInt = addr & 0xFFFFFFFFFFFFFFFFl
      
      println(s"Starting test with first temporal loop disabled...")

      while (cycleCount < maxCycles && (dut.io.busy.peek().litToBoolean || !dut.io.bufferEmpty.peek().litToBoolean)) {
        
        // PHASE 1: Set control signals based on CURRENT state
        for (channelIdx <- 0 until 8) {
          tcdmPortStates(channelIdx) match {
            case None =>
              // Port is idle, ready to accept new request
              dut.io.tcdmReq(channelIdx).ready.poke(true.B)
              dut.io.tcdmRsp(channelIdx).valid.poke(false.B)
              
            case Some(state) =>
              // Port is busy processing a request
              dut.io.tcdmReq(channelIdx).ready.poke(false.B)
              
              if (state.cyclesRemaining == 0) {
                val data = getTcdmData(state.addr)
                dut.io.tcdmRsp(channelIdx).valid.poke(true.B)
                dut.io.tcdmRsp(channelIdx).bits.data.poke(data.U)
                println(f"Cycle $cycleCount%3d: Channel $channelIdx - TCDM Response: data=0x$data%016x")
                responsesGiven += 1
              } else {
                dut.io.tcdmRsp(channelIdx).valid.poke(false.B)
              }
          }
        }
        
        // Check for output data before setting ready
        if (dut.io.data.valid.peek().litToBoolean) {
          val outputData = dut.io.data.bits.peek().litValue
          println(f"Cycle $cycleCount%3d: Output Data handshake will occur: 0x$outputData%0128x")
          dataReceived += 1
        }
        
        // Accept output data
        dut.io.data.ready.poke(true.B)

        // PHASE 2: Clock step
        dut.clock.step()
        cycleCount += 1
        
        // PHASE 3: Observe DUT outputs - collect new requests but don't update state yet
        val newRequestsThisCycle = Array.fill(8)(Option.empty[(BigInt, Boolean, BigInt)])
        for (channelIdx <- 0 until 8) {
          if (tcdmPortStates(channelIdx).isEmpty && dut.io.tcdmReq(channelIdx).valid.peek().litToBoolean) {
            val addr = dut.io.tcdmReq(channelIdx).bits.addr.peek().litValue
            val write = dut.io.tcdmReq(channelIdx).bits.write.peek().litToBoolean
            val strb = dut.io.tcdmReq(channelIdx).bits.strb.peek().litValue
            
            println(f"Cycle $cycleCount%3d: Channel $channelIdx - TCDM Request: addr=0x$addr%04x, write=$write, strb=0x$strb%02x")
            
            // Verify it's a read request
            assert(!write, s"Reader should only send read requests, got write at channel $channelIdx")
            
            requestsSeen += 1
            newRequestsThisCycle(channelIdx) = Some((addr, write, strb))
            println(f"Cycle $cycleCount%3d: Channel $channelIdx - Request will be accepted next cycle")
          }
        }
        
        // PHASE 4: Update internal state for NEXT cycle
        for (channelIdx <- 0 until 8) {
          tcdmPortStates(channelIdx) match {
            case None =>
              // If we saw a new request, accept it now for next cycle
              newRequestsThisCycle(channelIdx) match {
                case Some((addr, _, _)) =>
                  val latency = random.nextInt(4)
                  tcdmPortStates(channelIdx) = Some(TcdmPortState(addr, latency))
                case None =>
                  // Stay idle
              }
              
            case Some(state) =>
              if (state.cyclesRemaining == 0) {
                tcdmPortStates(channelIdx) = None
              } else {
                tcdmPortStates(channelIdx) = Some(state.copy(cyclesRemaining = state.cyclesRemaining - 1))
              }
          }
        }
      }

      println(s"\nTest Summary:")
      println(s"  Total cycles: $cycleCount")
      println(s"  Requests seen: $requestsSeen")
      println(s"  Responses given: $responsesGiven")
      println(s"  Data received: $dataReceived")
      println(s"  Expected requests: 64 (8 temporal * 8 channels)")
      
      // Verify we saw all expected requests
      assert(requestsSeen == 64, s"Expected 64 requests (8 temporal * 8 channels), got $requestsSeen")
      assert(responsesGiven == 64, s"Expected 64 responses, got $responsesGiven")
      assert(dataReceived == 8, s"Expected exactly 8 data outputs, got $dataReceived")
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

      // TCDM port state: per-port pending request tracking
      case class TcdmPortState(addr: BigInt, cyclesRemaining: Int)
      val tcdmPortStates = Array.fill(8)(Option.empty[TcdmPortState])
      val random = new scala.util.Random(44) // Fixed seed for reproducibility
      
      var cycleCount = 0
      val maxCycles = 200
      var requestsPerChannel = Array.fill(8)(0)

      def getTcdmData(addr: BigInt): BigInt = addr & 0xFFFFFFFFFFFFFFFFl

      while (cycleCount < maxCycles && (dut.io.busy.peek().litToBoolean || !dut.io.bufferEmpty.peek().litToBoolean)) {
        
        // PHASE 1: Set control signals based on CURRENT state
        for (channelIdx <- 0 until 8) {
          tcdmPortStates(channelIdx) match {
            case None =>
              // Port is idle, ready to accept new request
              dut.io.tcdmReq(channelIdx).ready.poke(true.B)
              dut.io.tcdmRsp(channelIdx).valid.poke(false.B)
              
            case Some(state) =>
              // Port is busy processing a request
              dut.io.tcdmReq(channelIdx).ready.poke(false.B)
              
              if (state.cyclesRemaining == 0) {
                val data = getTcdmData(state.addr)
                dut.io.tcdmRsp(channelIdx).valid.poke(true.B)
                dut.io.tcdmRsp(channelIdx).bits.data.poke(data.U)
              } else {
                dut.io.tcdmRsp(channelIdx).valid.poke(false.B)
              }
          }
        }
        
        // Accept output data
        dut.io.data.ready.poke(true.B)

        // PHASE 2: Clock step
        dut.clock.step()
        cycleCount += 1
        
        // PHASE 3: Observe DUT outputs - collect new requests but don't update state yet
        val newRequestsThisCycle = Array.fill(8)(Option.empty[BigInt])
        for (channelIdx <- 0 until 8) {
          if (tcdmPortStates(channelIdx).isEmpty && dut.io.tcdmReq(channelIdx).valid.peek().litToBoolean) {
            val addr = dut.io.tcdmReq(channelIdx).bits.addr.peek().litValue
            requestsPerChannel(channelIdx) += 1
            newRequestsThisCycle(channelIdx) = Some(addr)
          }
        }
        
        // PHASE 4: Update internal state for NEXT cycle
        for (channelIdx <- 0 until 8) {
          tcdmPortStates(channelIdx) match {
            case None =>
              // If we saw a new request, accept it now for next cycle
              newRequestsThisCycle(channelIdx) match {
                case Some(addr) =>
                  val latency = random.nextInt(4)
                  tcdmPortStates(channelIdx) = Some(TcdmPortState(addr, latency))
                case None =>
                  // Stay idle
              }
              
            case Some(state) =>
              if (state.cyclesRemaining == 0) {
                tcdmPortStates(channelIdx) = None
              } else {
                tcdmPortStates(channelIdx) = Some(state.copy(cyclesRemaining = state.cyclesRemaining - 1))
              }
          }
        }
      }

      println(s"\nChannel masking test summary:")
      println(s"  Total cycles: $cycleCount")
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
      
      // Expected: 2 temporal bounds * 4 enabled channels = 8 requests
      val totalRequests = requestsPerChannel.sum
      assert(totalRequests == 8, s"Expected 8 total requests (2 temporal * 4 enabled channels), got $totalRequests")
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

      // TCDM port state: per-port pending request tracking
      case class TcdmPortState(addr: BigInt, cyclesRemaining: Int)
      val tcdmPortStates = Array.fill(8)(Option.empty[TcdmPortState])
      val random = new scala.util.Random(45) // Fixed seed for reproducibility
      
      var cycleCount = 0
      val maxCycles = 400
      var requestsSeen = 0
      var responsesGiven = 0
      var dataReceived = 0

      def getTcdmData(addr: BigInt): BigInt = addr & 0xFFFFFFFFFFFFFFFFl
      
      println(s"Starting backpressure test...")

      while (cycleCount < maxCycles && (dut.io.busy.peek().litToBoolean || !dut.io.bufferEmpty.peek().litToBoolean)) {
        
        // PHASE 1: Set control signals based on CURRENT state
        for (channelIdx <- 0 until 8) {
          tcdmPortStates(channelIdx) match {
            case None =>
              // Port is idle, ready to accept new request
              dut.io.tcdmReq(channelIdx).ready.poke(true.B)
              dut.io.tcdmRsp(channelIdx).valid.poke(false.B)
              
            case Some(state) =>
              // Port is busy processing a request
              dut.io.tcdmReq(channelIdx).ready.poke(false.B)
              
              if (state.cyclesRemaining == 0) {
                val data = getTcdmData(state.addr)
                dut.io.tcdmRsp(channelIdx).valid.poke(true.B)
                dut.io.tcdmRsp(channelIdx).bits.data.poke(data.U)
                println(f"Cycle $cycleCount%3d: Channel $channelIdx - TCDM Response: data=0x$data%016x")
                responsesGiven += 1
              } else {
                dut.io.tcdmRsp(channelIdx).valid.poke(false.B)
              }
          }
        }
        
        // Check for output data handshake BEFORE setting ready
        // If valid is already high, and we're about to set ready high, handshake will occur
        val dataWillHandshake = dut.io.data.valid.peek().litToBoolean && ((cycleCount % 3) == 0)
        if (dataWillHandshake) {
          val outputData = dut.io.data.bits.peek().litValue
          println(f"Cycle $cycleCount%3d: Output Data handshake will occur: 0x$outputData%0128x")
          dataReceived += 1
        }
        
        // Intermittent backpressure - only accept data every 3 cycles
        if (cycleCount % 3 == 0) {
          dut.io.data.ready.poke(true.B)
        } else {
          dut.io.data.ready.poke(false.B)
        }

        // PHASE 2: Clock step
        dut.clock.step()
        cycleCount += 1
        
        // PHASE 3: Observe DUT outputs - collect new requests but don't update state yet
        val newRequestsThisCycle = Array.fill(8)(Option.empty[(BigInt, Boolean)])
        for (channelIdx <- 0 until 8) {
          if (tcdmPortStates(channelIdx).isEmpty && dut.io.tcdmReq(channelIdx).valid.peek().litToBoolean) {
            val addr = dut.io.tcdmReq(channelIdx).bits.addr.peek().litValue
            val write = dut.io.tcdmReq(channelIdx).bits.write.peek().litToBoolean
            
            println(f"Cycle $cycleCount%3d: Channel $channelIdx - TCDM Request: addr=0x$addr%04x, write=$write")
            
            // Verify it's a read request
            assert(!write, s"Reader should only send read requests, got write at channel $channelIdx")
            
            requestsSeen += 1
            newRequestsThisCycle(channelIdx) = Some((addr, write))
          }
        }
        
        // PHASE 4: Update internal state for NEXT cycle
        for (channelIdx <- 0 until 8) {
          tcdmPortStates(channelIdx) match {
            case None =>
              // If we saw a new request, accept it now for next cycle
              newRequestsThisCycle(channelIdx) match {
                case Some((addr, _)) =>
                  val latency = random.nextInt(4)
                  tcdmPortStates(channelIdx) = Some(TcdmPortState(addr, latency))
                case None =>
                  // Stay idle
              }
              
            case Some(state) =>
              if (state.cyclesRemaining == 0) {
                tcdmPortStates(channelIdx) = None
              } else {
                tcdmPortStates(channelIdx) = Some(state.copy(cyclesRemaining = state.cyclesRemaining - 1))
              }
          }
        }
      }

      println(s"\nBackpressure Test Summary:")
      println(s"  Total cycles: $cycleCount")
      println(s"  Requests seen: $requestsSeen")
      println(s"  Responses given: $responsesGiven")
      println(s"  Data received: $dataReceived")
      println(s"  Expected requests: 24 (3 temporal * 8 channels)")
      
      assert(cycleCount < maxCycles, "Test should complete before timeout")
      assert(requestsSeen == 24, s"Expected 24 requests (3 temporal * 8 channels), got $requestsSeen")
      assert(responsesGiven == 24, s"Expected 24 responses, got $responsesGiven")
      assert(dataReceived == 3, s"Expected exactly 3 data outputs, got $dataReceived")
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