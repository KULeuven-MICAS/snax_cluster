package snax.xdma.xdmaTop

import scala.util.Random
import scala.util.control.Breaks.break
import scala.util.control.Breaks.breakable

import chisel3._

import chiseltest._
import org.scalatest.freespec.AnyFreeSpec
import snax.DataPathExtension.HasMaxPool
import snax.DataPathExtension.HasTransposer
import snax.DataPathExtension.HasVerilogMemset
import snax.readerWriter.ReaderWriterParam
import snax.xdma.DesignParams._

class XDMATopTester extends AnyFreeSpec with ChiselScalatestTester {

  // ************************ Prepare the simulation data ************************//

  // Prepare the data in the tcdm
  val tcdm_mem       = collection.mutable.Map[Long, BigInt]()
  // We have 128KB of the tcdm data
  // Each element is 64bit(8B) long
  // Hence in total we have 128KB/8B = 16K tcdm memory lines
  // First we generate the first 16KB (2K) of the data ramdomly in a Seq, which can be consumed by golden refernece later
  val input_data     = for (i <- 0 until 2048) yield {
    BigInt(numbits = 64, rnd = Random)
  }
  var testTerminated = false

  for (i <- 0 until input_data.length) {
    tcdm_mem(8 * i) = input_data(i)
  }
  println("[TCDM] TCDM data initialized. ")

  "The XDMATop local loopback test should pass" in test(
    new XDMATop(
      readerParam = new XDMAParam(
        axiParam          = new AXIParam,
        crossClusterParam = new CrossClusterParam,
        rwParam           = new ReaderWriterParam(
          configurableByteMask = false,
          configurableChannel  = true
        )
      ),
      writerParam = new XDMAParam(
        axiParam          = new AXIParam,
        crossClusterParam = new CrossClusterParam,
        rwParam           = new ReaderWriterParam(
          configurableByteMask = true,
          configurableChannel  = true
        ),
        extParam          = Seq(
          new HasVerilogMemset,
          new HasMaxPool,
          new HasTransposer(Seq(8), Seq(8), Seq(8))
        )
      )
    )
  ).withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
    // ************************ Start Simulation **********************************//
    // The thread list for the concurrent elements in a tester
    var concurrent_threads = new chiseltest.internal.TesterThreadList(Seq())

    // The controller for the xdmaTop
    concurrent_threads = concurrent_threads.fork {
      // Test 1: Duplicate the data once with only one channel
      println("[Memcopy test]")
      var readerAGUParam = new AGUParamTest(
        address         = Seq(0),
        spatialStrides  = Array(8),
        temporalStrides = Array(8, 0),
        temporalBounds  = Array(2048, 1)
      )
      var writerAGUParam = new AGUParamTest(
        address         = Seq(16 * 1024, 0, 0, 0),
        spatialStrides  = Array(8),
        temporalStrides = Array(8, 0),
        temporalBounds  = Array(2048, 1)
      )

      var readerRWParam = new RWParamTest(
        enabledChannel = Integer.parseInt("00000001", 2),
        enabledByte    = Integer.parseInt("11111111", 2)
      )
      var writerRWParam = new RWParamTest(
        enabledChannel = Integer.parseInt("00000001", 2),
        enabledByte    = Integer.parseInt("11111111", 2)
      )

      var writerExtParam = new ExtParam(
        bypassMemset     = 1,
        memsetValue      = 0,
        bypassMaxPool    = 1,
        maxPoolPeriod    = 0,
        bypassTransposer = 1
      )

      // Write the configuration
      var currentAddress = 0

      currentAddress = XDMATesterInfrastructure.setXDMA(
        readerAGUParam,
        writerAGUParam,
        readerRWParam,
        writerRWParam,
        writerExtParam,
        dut,
        dut.io.csrIO
      )
      currentAddress += 2

      // Check if the DMA is finished
      while (
        XDMATesterInfrastructure.read_csr(
          dut  = dut,
          port = dut.io.csrIO,
          addr = currentAddress
        ) != 1
      ) {}

      // Check whether the data in memory is coincide with the expectation
      var mem_to_be_checked =
        tcdm_mem.filter(_._1 >= 1024 * 16).map { case (addr, data) =>
          (addr - 1024 * 16, data)
        }
      var mem_expected      = tcdm_mem.filter(_._1 < 1024 * 16)
      if (mem_to_be_checked != mem_expected)
        throw new Exception("[Memcopy test] The test fails. ")
      else println("[Memcopy test] The test passes. ")

      // Test 2: Set the memory to 0x00
      println("[Memset test 0x00]")
      readerAGUParam = new AGUParamTest(
        address         = Seq(0),
        spatialStrides  = Array(8),
        temporalStrides = Array(64, 0),
        temporalBounds  = Array(256, 1)
      )
      writerAGUParam = new AGUParamTest(
        address         = Seq(16 * 1024, 0, 0, 0),
        spatialStrides  = Array(8),
        temporalStrides = Array(64, 0),
        temporalBounds  = Array(256, 1)
      )

      readerRWParam = new RWParamTest(
        enabledChannel = Integer.parseInt("11111111", 2),
        enabledByte    = Integer.parseInt("11111111", 2)
      )
      writerRWParam = new RWParamTest(
        enabledChannel = Integer.parseInt("11111111", 2),
        enabledByte    = Integer.parseInt("11111111", 2)
      )

      writerExtParam = new ExtParam(
        bypassMemset     = 0,
        memsetValue      = 0,
        bypassMaxPool    = 1,
        maxPoolPeriod    = 0,
        bypassTransposer = 1
      )

      // Write the configuration
      currentAddress = XDMATesterInfrastructure.setXDMA(
        readerAGUParam,
        writerAGUParam,
        readerRWParam,
        writerRWParam,
        writerExtParam,
        dut,
        dut.io.csrIO
      )
      currentAddress += 2

      // Check if the DMA is finished
      while (
        XDMATesterInfrastructure.read_csr(
          dut  = dut,
          port = dut.io.csrIO,
          addr = currentAddress
        ) != 2
      ) {}

      // Check whether the data in memory is coincide with the expectation
      mem_to_be_checked = tcdm_mem.filter(_._1 >= 1024 * 16)
      if (mem_to_be_checked.map(_._2 == 0).reduce(_ & _))
        println("[Memset test] The test passes. ")
      else throw new Exception("[Memset test] The test fails. ")

      // Test 3: Set the memory to 0x0000_00ff. This can be done by setting the strb to 0x01 with Memset to 0xff
      println("[Memset test 0xFF]")
      readerAGUParam = new AGUParamTest(
        address         = Seq(0),
        spatialStrides  = Array(8),
        temporalStrides = Array(64, 0),
        temporalBounds  = Array(256, 1)
      )
      writerAGUParam = new AGUParamTest(
        address         = Seq(16 * 1024, 0, 0, 0),
        spatialStrides  = Array(8),
        temporalStrides = Array(64, 0),
        temporalBounds  = Array(256, 1)
      )

      readerRWParam = new RWParamTest(
        enabledChannel = Integer.parseInt("11111111", 2),
        enabledByte    = Integer.parseInt("00000001", 2)
      )
      writerRWParam = new RWParamTest(
        enabledChannel = Integer.parseInt("11111111", 2),
        enabledByte    = Integer.parseInt("00000001", 2)
      )

      writerExtParam = new ExtParam(
        bypassMemset     = 0,
        memsetValue      = 0xff,
        bypassMaxPool    = 1,
        maxPoolPeriod    = 0,
        bypassTransposer = 1
      )

      // Write the configuration
      currentAddress = XDMATesterInfrastructure.setXDMA(
        readerAGUParam,
        writerAGUParam,
        readerRWParam,
        writerRWParam,
        writerExtParam,
        dut,
        dut.io.csrIO
      )
      currentAddress += 2

      // Check if the DMA is finished
      while (
        XDMATesterInfrastructure.read_csr(
          dut  = dut,
          port = dut.io.csrIO,
          addr = currentAddress
        ) != 3
      ) {}

      // Check whether the data in memory is coincide with the expectation
      mem_to_be_checked = tcdm_mem.filter(_._1 >= 1024 * 16)
      if (
        mem_to_be_checked
          .map(_._2 == BigInt("ff", radix = 16))
          .reduce(_ & _)
      )
        println("[Memset test 0xFF] The test passes. ")
      else throw new Exception("[Memset test 0xFF] The test fails. ")

      // Test 4: Do the transpose
      // Remove the useless data
      tcdm_mem.foreach(i => {
        if (i._1 >= 16384) tcdm_mem.remove(i._1)
      })

      val expected_result_4 = collection.mutable.Map[Long, BigInt]()
      input_data.map { element =>
        var temp_element = element
        for (i <- 0 until 8) yield {
          val returned_value = temp_element & 0xff
          temp_element = temp_element >> 8
          returned_value
        }
      }
        .grouped(8)
        .map(_.transpose)
        .map(_.map { i =>
          var returned_value = BigInt(0)
          i.reverse.foreach { j =>
            returned_value = (returned_value << 8) + j
          }
          returned_value
        })
        .reduce(_ ++ _)
        .zipWithIndex
        .foreach { case (a, b) =>
          expected_result_4.addOne((b * 8 + 16 * 1024, a))
        }

      println("[Transposer Test] Data preparation is finished. ")
      println("[Transposer Test] Test Started")
      readerAGUParam = new AGUParamTest(
        address         = Seq(0),
        spatialStrides  = Array(8),
        temporalStrides = Array(64, 0),
        temporalBounds  = Array(256, 1)
      )
      writerAGUParam = new AGUParamTest(
        address         = Seq(16 * 1024, 0, 0, 0),
        spatialStrides  = Array(8),
        temporalStrides = Array(64, 0),
        temporalBounds  = Array(256, 1)
      )

      readerRWParam = new RWParamTest(
        enabledChannel = Integer.parseInt("11111111", 2),
        enabledByte    = Integer.parseInt("11111111", 2)
      )
      writerRWParam = new RWParamTest(
        enabledChannel = Integer.parseInt("11111111", 2),
        enabledByte    = Integer.parseInt("11111111", 2)
      )

      writerExtParam = new ExtParam(
        bypassMemset     = 1,
        memsetValue      = 0xff,
        bypassMaxPool    = 1,
        maxPoolPeriod    = 0,
        bypassTransposer = 0
      )

      // Write the configuration
      currentAddress = XDMATesterInfrastructure.setXDMA(
        readerAGUParam,
        writerAGUParam,
        readerRWParam,
        writerRWParam,
        writerExtParam,
        dut,
        dut.io.csrIO
      )
      currentAddress += 2

      // Check if the DMA is finished
      while (
        XDMATesterInfrastructure.read_csr(
          dut  = dut,
          port = dut.io.csrIO,
          addr = currentAddress
        ) != 4
      ) {}

      // Check the result's correctness
      mem_to_be_checked = tcdm_mem.filter(_._1 >= 1024 * 16)
      if (mem_to_be_checked == expected_result_4)
        println("[Transposer test] The test passes. ")
      else throw new Exception("[Transposer test] The test fails. ")

      println("All tests pass. All test threads are about to be terminated. ")
      testTerminated = true
    }

    // eight threads to mimic the reader req side
    // Each threads will emulate a random delay in the req_ready side
    // ---------                ------------
    // |       |----->addr----->|          |
    // |reader |----->valid---->| tcdm port|
    // |   req |<-----ready----<|          |
    // ---------                ------------

    // Queues to temporarily store the address at request side, which will be consumed by responser
    val queues = Seq.fill(8)(collection.mutable.Queue[Int]())

    for (i <- 0 until 8) {
      concurrent_threads = concurrent_threads.fork {
        breakable(
          while (true) {
            if (testTerminated) break()
            val random_delay    = Random.between(0, 1)
            if (random_delay > 1) {
              dut.io.tcdmReader.req(i).ready.poke(false)
              dut.clock.step(random_delay)
              dut.io.tcdmReader.req(i).ready.poke(true)
            } else dut.io.tcdmReader.req(i).ready.poke(true)
            val reader_req_addr =
              dut.io.tcdmReader.req(i).bits.addr.peekInt().toInt
            if (dut.io.tcdmReader.req(i).valid.peekBoolean()) {
              queues(i).enqueue(reader_req_addr)

              println(
                f"[Reader Req] Read the TCDM with Addr = 0x${reader_req_addr.toHexString}"
              )
            }

            dut.clock.step()
          }
        )
      }
    }
    // eight threads to mimic the reader resp side
    // There are no ready port in the reader side, so we just pop out the data accoring to
    // the addr recored in queues
    // ---------                ------------
    // |       |<-----data-----<|          |
    // |reader |<-----valid----<| tcdm port|
    // |   resp|                |          |
    // ---------                ------------
    for (i <- 0 until 8) {
      concurrent_threads = concurrent_threads.fork {
        breakable(
          while (true) {
            if (testTerminated) break()
            if (queues(i).isEmpty) dut.clock.step()
            else {
              dut.io.tcdmReader.rsp(i).valid.poke(true)
              val reader_addr      = queues(i).dequeue()
              val reader_resp_data = tcdm_mem(reader_addr)
              println(
                f"[Reader Resp] TCDM Response to Reader with Addr = 0x${reader_addr.toHexString} Data = 0x${reader_resp_data
                    .toString(radix = 16)}"
              )
              dut.io.tcdmReader.rsp(i).bits.data.poke(reader_resp_data.U)
              dut.clock.step()
              dut.io.tcdmReader.rsp(i).valid.poke(false)
            }

          }
        )
      }
    }
    // eight threads to mimic the writer req side
    // Like the reader req side, we emulate the random delay by poking to ready signal
    // ---------                ------------
    // |       |>-----addr----->|          |
    // |writer |>-----write---->| tcdm port|
    // |   req |>-----data----->|          |
    // |       |>-----valid---->|          |
    // |       |<-----ready----<|          |
    // ---------                ------------
    for (i <- 0 until 8) {
      concurrent_threads = concurrent_threads.fork {
        breakable(
          while (true) {
            if (testTerminated) break()

            if (dut.io.tcdmWriter.req(i).valid.peekBoolean()) {
              val writer_req_addr =
                dut.io.tcdmWriter.req(i).bits.addr.peekInt().toInt
              val writer_req_data =
                dut.io.tcdmWriter.req(i).bits.data.peekInt()

              val random_delay = Random.between(10, 20)
              if (random_delay > 1) {
                dut.io.tcdmWriter.req(i).ready.poke(false)
                dut.clock.step(random_delay)
                dut.io.tcdmWriter.req(i).ready.poke(true)
              } else dut.io.tcdmWriter.req(i).ready.poke(true)

              val previous_data =
                if (tcdm_mem.contains(writer_req_addr))
                  tcdm_mem(writer_req_addr)
                else BigInt(0)
              val Strb          = dut.io.tcdmWriter.req(i).bits.strb.peekInt().toInt
              var bitStrb       = BigInt(0)
              for (i <- 7 to 0 by -1) {
                val bit   = (Strb >> i) & 1
                val block = (BigInt(255) * bit) << (i * 8)
                bitStrb |= block
              }

              val new_data =
                (previous_data & (~bitStrb)) | (writer_req_data & bitStrb)
              tcdm_mem(writer_req_addr) = new_data
              println(
                f"[Writer Req] Writes to TCDM with Addr: 0x${writer_req_addr.toHexString} and Data = 0x${new_data
                    .toString(radix = 16)}"
              )
              dut.clock.step()
            } else dut.clock.step()

          }
        )
      }
    }

    concurrent_threads.joinAndStep()
  }
}

object xdmaTopEmitter extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new XDMATop(
      clusterName = "test_cluster",
      readerParam = new XDMAParam(
        new AXIParam,
        new CrossClusterParam(4),
        new ReaderWriterParam,
        Seq()
      ),
      writerParam = new XDMAParam(
        new AXIParam,
        new CrossClusterParam(4),
        new ReaderWriterParam,
        Seq(
          new HasVerilogMemset,
          new HasMaxPool,
          new HasTransposer(Seq(8), Seq(8), Seq(8))
        )
      )
    ),
    args = Array("--target-dir", "generated/xdma")
  )
}
