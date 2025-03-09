package snax.xdma.xdmaTop

import chisel3._
import chisel3.util._
// Hardware and its Generation Param
import snax.readerWriter.ReaderWriterParam
import snax.utils.DecoupledCut._

// Import Chiseltest
import chiseltest._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.flatspec.AnyFlatSpec

// Import Random number generator
import scala.util.Random

// Import break support for loops
import scala.util.control.Breaks.{break, breakable}
import snax.xdma.xdmaFrontend._
import snax.csr_manager.SnaxCsrIO
import java.util.HashMap
import snax.xdma.DesignParams._
import snax.DataPathExtension.{HasMaxPool, HasTransposer, HasVerilogMemset}

class DualXDMA(readerParam: XDMAParam, writerParam: XDMAParam)
    extends Module
    with RequireAsyncReset {
  val xdmaOne = Module(
    new XDMATop(
      readerParam = readerParam,
      writerParam = writerParam
    )
  )

  val xdmaTwo = Module(
    new XDMATop(
      readerParam = readerParam,
      writerParam = writerParam
    )
  )

  xdmaOne.io.clusterBaseAddress := 0x10000000.U
  xdmaTwo.io.clusterBaseAddress := (0x10000000 + (1 << 20)).U

  xdmaTwo.io.remoteXDMACfg.toRemote -||> xdmaOne.io.remoteXDMACfg.fromRemote
  xdmaOne.io.remoteXDMACfg.toRemote -||> xdmaTwo.io.remoteXDMACfg.fromRemote

  xdmaTwo.io.remoteXDMAData.toRemote -||> xdmaOne.io.remoteXDMAData.fromRemote
  xdmaOne.io.remoteXDMAData.toRemote -||> xdmaTwo.io.remoteXDMAData.fromRemote

  val io = IO(new Bundle {
    val instanceOne = new Bundle {
      val csrIO = chiselTypeOf(xdmaOne.io.csrIO)
      val tcdmReader = chiselTypeOf(xdmaOne.io.tcdmReader)
      val tcdmWriter = chiselTypeOf(xdmaOne.io.tcdmWriter)
    }
    val instanceTwo = new Bundle {
      val csrIO = chiselTypeOf(xdmaTwo.io.csrIO)
      val tcdmReader = chiselTypeOf(xdmaTwo.io.tcdmReader)
      val tcdmWriter = chiselTypeOf(xdmaTwo.io.tcdmWriter)
    }
  })

  io.instanceOne.csrIO <> xdmaOne.io.csrIO
  io.instanceOne.tcdmReader <> xdmaOne.io.tcdmReader
  io.instanceOne.tcdmWriter <> xdmaOne.io.tcdmWriter
  io.instanceTwo.csrIO <> xdmaTwo.io.csrIO
  io.instanceTwo.tcdmReader <> xdmaTwo.io.tcdmReader
  io.instanceTwo.tcdmWriter <> xdmaTwo.io.tcdmWriter
}

class DualXDMATester extends AnyFreeSpec with ChiselScalatestTester {

  // ************************ Prepare the simulation data ************************//

  // Prepare the data in the tcdm
  val tcdmMem_1 = collection.mutable.Map[Long, BigInt]()
  val tcdmMem_2 = collection.mutable.Map[Long, BigInt]()
  // We have 128KB of the tcdm data
  // Each element is 64bit(8B) long
  // Hence in total we have 128KB/8B = 16K tcdm memory lines
  // First we generate the first 16KB (2K) of the data ramdomly in a Seq, which can be consumed by golden refernece later
  val input_data = for (i <- 0 until 2048) yield {
    BigInt(numbits = 64, rnd = Random)
  }
  var testTerminated = false

  for (i <- 0 until input_data.length) {
    tcdmMem_1(8 * i) = input_data(i)
  }
  println("[TCDM] TCDM 1 data initialized. ")

  def write_csr(dut: Module, port: SnaxCsrIO, addr: Int, data: Int) = {

    // give the data and address to the right ports
    port.req.bits.write.poke(true.B)

    port.req.bits.data.poke(data.U)
    port.req.bits.addr.poke(addr.U)
    port.req.valid.poke(1.B)

    // wait for grant
    while (port.req.ready.peekBoolean() == false) {

      dut.clock.step(1)
    }

    dut.clock.step(1)

    port.req.valid.poke(0.B)
  }

  def read_csr(dut: Module, port: SnaxCsrIO, addr: Int) = {
    dut.clock.step(1)

    // give the data and address to the right ports
    port.req.bits.write.poke(0.B)
    port.req.bits.addr.poke(addr.U)
    port.req.valid.poke(1.B)

    // wait for grant
    while (port.req.ready.peekBoolean() == false) {
      dut.clock.step(1)
    }
    dut.clock.step(1)
    port.req.valid.poke(0.B)

    port.rsp.ready.poke(true)
    // wait for valid signal
    while (port.rsp.valid.peekBoolean() == false) {
      dut.clock.step(1)
    }
    val result = port.rsp.bits.data.peekInt()
    dut.clock.step(1)
    port.rsp.ready.poke(false)

    result
  }

  "The behavior of two XDMA test is as expected" in test(
    new DualXDMA(
      readerParam = new XDMAParam(
        axiParam = new AXIParam,
        crossClusterParam = new CrossClusterParam,
        rwParam = new ReaderWriterParam(
          configurableByteMask = false,
          configurableChannel = true
        )
      ),
      writerParam = new XDMAParam(
        axiParam = new AXIParam,
        crossClusterParam = new CrossClusterParam,
        rwParam = new ReaderWriterParam(
          configurableByteMask = true,
          configurableChannel = true
        ),
        extParam = Seq(
          new HasVerilogMemset,
          new HasMaxPool,
          new HasTransposer(Seq(8), Seq(8), Seq(8))
        )
      )
    )
  ).withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) {
    dut =>
      // ************************ Start Simulation **********************************//
      // The thread list for the concurrent elements in a tester
      var concurrent_threads = new chiseltest.internal.TesterThreadList(Seq())

      // Eight threads to mimic the reader req side for XDMA 1
      // Each threads will emulate a random delay in the req_ready side
      // ---------                ------------
      // |       |----->addr----->|          |
      // |reader |----->valid---->| tcdm port|
      // |   req |<-----ready----<|          |
      // ---------                ------------

      // Queues to temporarily store the address at request side, which will be consumed by responser
      val queues_xdma1 = Seq.fill(8)(collection.mutable.Queue[Int]())

      for (i <- 0 until 8) {
        concurrent_threads = concurrent_threads.fork {
          breakable(while (true) {
            if (testTerminated) break()
            val random_delay = Random.between(0, 1)
            if (random_delay > 1) {
              dut.io.instanceOne.tcdmReader.req(i).ready.poke(false)
              dut.clock.step(random_delay)
              dut.io.instanceOne.tcdmReader.req(i).ready.poke(true)
            } else dut.io.instanceOne.tcdmReader.req(i).ready.poke(true)
            val reader_req_addr =
              dut.io.instanceOne.tcdmReader.req(i).bits.addr.peekInt().toInt
            if (dut.io.instanceOne.tcdmReader.req(i).valid.peekBoolean()) {
              queues_xdma1(i).enqueue(reader_req_addr)

              println(
                f"[XDMA1 Reader Req] Read the TCDM with Addr = 0x${reader_req_addr.toHexString}"
              )
            }

            dut.clock.step()
          })
        }
      }

      // Eight threads to mimic the reader req side for XDMA 2
      // Each threads will emulate a random delay in the req_ready side
      // ---------                ------------
      // |       |----->addr----->|          |
      // |reader |----->valid---->| tcdm port|
      // |   req |<-----ready----<|          |
      // ---------                ------------

      // Queues to temporarily store the address at request side, which will be consumed by responser
      val queues_xdma2 = Seq.fill(8)(collection.mutable.Queue[Int]())

      for (i <- 0 until 8) {
        concurrent_threads = concurrent_threads.fork {
          breakable(
            while (true) {
              if (testTerminated) break()
              val random_delay = Random.between(0, 1)
              if (random_delay > 1) {
                dut.io.instanceTwo.tcdmReader.req(i).ready.poke(false)
                dut.clock.step(random_delay)
                dut.io.instanceTwo.tcdmReader.req(i).ready.poke(true)
              } else dut.io.instanceTwo.tcdmReader.req(i).ready.poke(true)
              val reader_req_addr =
                dut.io.instanceTwo.tcdmReader.req(i).bits.addr.peekInt().toInt
              if (dut.io.instanceTwo.tcdmReader.req(i).valid.peekBoolean()) {
                queues_xdma2(i).enqueue(reader_req_addr)

                println(
                  f"[XDMA2 Reader Req] Read the TCDM with Addr = 0x${reader_req_addr.toHexString}"
                )
              }

              dut.clock.step()
            }
          )
        }
      }

      // eight threads to mimic the reader resp side for XDMA 1
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
              if (queues_xdma1(i).isEmpty) dut.clock.step()
              else {
                dut.io.instanceOne.tcdmReader.rsp(i).valid.poke(true)
                val reader_addr = queues_xdma1(i).dequeue()
                val reader_resp_data = tcdmMem_1(reader_addr)
                println(
                  f"[XDMA 1 Reader Resp] TCDM Response to Reader with Addr = 0x${reader_addr.toHexString} Data = 0x${reader_resp_data
                      .toString(radix = 16)}"
                )
                dut.io.instanceOne.tcdmReader
                  .rsp(i)
                  .bits
                  .data
                  .poke(reader_resp_data.U)
                dut.clock.step()
                dut.io.instanceOne.tcdmReader.rsp(i).valid.poke(false)
              }
            }
          )
        }
      }

      // eight threads to mimic the reader resp side for XDMA 2
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
              if (queues_xdma2(i).isEmpty) dut.clock.step()
              else {
                dut.io.instanceTwo.tcdmReader.rsp(i).valid.poke(true)
                val reader_addr = queues_xdma2(i).dequeue()
                val reader_resp_data = tcdmMem_1(reader_addr)
                println(
                  f"[XDMA 2 Reader Resp] TCDM Response to Reader with Addr = 0x${reader_addr.toHexString} Data = 0x${reader_resp_data
                      .toString(radix = 16)}"
                )
                dut.io.instanceTwo.tcdmReader
                  .rsp(i)
                  .bits
                  .data
                  .poke(reader_resp_data.U)
                dut.clock.step()
                dut.io.instanceTwo.tcdmReader.rsp(i).valid.poke(false)
              }
            }
          )
        }
      }

      // eight threads to mimic the writer req side for XDMA 1
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

              if (dut.io.instanceOne.tcdmWriter.req(i).valid.peekBoolean()) {
                val writer_req_addr =
                  dut.io.instanceOne.tcdmWriter.req(i).bits.addr.peekInt().toInt
                val writer_req_data =
                  dut.io.instanceOne.tcdmWriter.req(i).bits.data.peekInt()

                val random_delay = Random.between(10, 20)
                if (random_delay > 1) {
                  dut.io.instanceOne.tcdmWriter.req(i).ready.poke(false)
                  dut.clock.step(random_delay)
                  dut.io.instanceOne.tcdmWriter.req(i).ready.poke(true)
                } else dut.io.instanceOne.tcdmWriter.req(i).ready.poke(true)

                val previous_data =
                  if (tcdmMem_1.contains(writer_req_addr))
                    tcdmMem_1(writer_req_addr)
                  else BigInt(0)
                val Strb =
                  dut.io.instanceOne.tcdmWriter.req(i).bits.strb.peekInt().toInt
                var bitStrb = BigInt(0)
                for (i <- 7 to 0 by -1) {
                  val bit = (Strb >> i) & 1
                  val block = (BigInt(255) * bit) << (i * 8)
                  bitStrb |= block
                }

                val new_data =
                  (previous_data & (~bitStrb)) | (writer_req_data & bitStrb)
                tcdmMem_1(writer_req_addr) = new_data
                println(
                  f"[XDMA 1 Writer Req] Writes to TCDM with Addr: 0x${writer_req_addr.toHexString} and Data = 0x${new_data
                      .toString(radix = 16)}"
                )
                dut.clock.step()
              } else dut.clock.step()

            }
          )
        }
      }

      // eight threads to mimic the writer req side for XDMA 2
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

              if (dut.io.instanceTwo.tcdmWriter.req(i).valid.peekBoolean()) {
                val writer_req_addr =
                  dut.io.instanceTwo.tcdmWriter.req(i).bits.addr.peekInt().toInt
                val writer_req_data =
                  dut.io.instanceTwo.tcdmWriter.req(i).bits.data.peekInt()

                val random_delay = Random.between(10, 20)
                if (random_delay > 1) {
                  dut.io.instanceTwo.tcdmWriter.req(i).ready.poke(false)
                  dut.clock.step(random_delay)
                  dut.io.instanceTwo.tcdmWriter.req(i).ready.poke(true)
                } else dut.io.instanceTwo.tcdmWriter.req(i).ready.poke(true)

                val previous_data =
                  if (tcdmMem_2.contains(writer_req_addr))
                    tcdmMem_2(writer_req_addr)
                  else BigInt(0)
                val Strb =
                  dut.io.instanceTwo.tcdmWriter.req(i).bits.strb.peekInt().toInt
                var bitStrb = BigInt(0)
                for (i <- 7 to 0 by -1) {
                  val bit = (Strb >> i) & 1
                  val block = (BigInt(255) * bit) << (i * 8)
                  bitStrb |= block
                }

                val new_data =
                  (previous_data & (~bitStrb)) | (writer_req_data & bitStrb)
                tcdmMem_2(writer_req_addr) = new_data
                println(
                  f"[XDMA 2 Writer Req] Writes to TCDM with Addr: 0x${writer_req_addr.toHexString} and Data = 0x${new_data
                      .toString(radix = 16)}"
                )
                dut.clock.step()
              } else dut.clock.step()

            }
          )
        }
      }

      concurrent_threads = concurrent_threads.fork {
        // Test 1: Use XDMA 1 as a host to copy the data from TCDM 1 to TCDM 2
        println(
          "[TEST] Test 1: Use XDMA 1 as a host to copy the data from TCDM 1 to TCDM 2"
        )
        var readerAGUParam = new AGUParamTest(
          address = 0x1000_0000,
          spatialStrides = Array(8),
          temporalStrides = Array(64, 0),
          temporalBounds = Array(256, 1)
        )
        var writerAGUParam = new AGUParamTest(
          address = 0x1000_0000 + (1 << 20),
          spatialStrides = Array(8),
          temporalStrides = Array(64, 0),
          temporalBounds = Array(256, 1)
        )

        var readerRWParam = new RWParamTest(
          enabledChannel = Integer.parseInt("11111111", 2),
          enabledByte = Integer.parseInt("11111111", 2)
        )
        var writerRWParam = new RWParamTest(
          enabledChannel = Integer.parseInt("11111111", 2),
          enabledByte = Integer.parseInt("11111111", 2)
        )

        var writerExtParam = new ExtParam(
          bypassMemset = 1,
          memsetValue = 0,
          bypassMaxPool = 1,
          maxPoolPeriod = 0,
          bypassTransposer = 1
        )

        // Write the configuration
        var currentAddress = 0

        // Pointer Address
        // Reader Side
        write_csr(
          dut = dut,
          port = dut.io.instanceOne.csrIO,
          addr = currentAddress,
          data = (readerAGUParam.address & 0xffff_ffff).toInt
        )
        currentAddress += 1
        write_csr(
          dut = dut,
          port = dut.io.instanceOne.csrIO,
          addr = currentAddress,
          data = ((readerAGUParam.address >> 32) & 0xffff_ffff).toInt
        )
        currentAddress += 1

        // Writer Side
        // Ptr 0
        write_csr(
          dut = dut,
          port = dut.io.instanceOne.csrIO,
          addr = currentAddress,
          data = (writerAGUParam.address & 0xffff_ffff).toInt
        )
        currentAddress += 1
        write_csr(
          dut = dut,
          port = dut.io.instanceOne.csrIO,
          addr = currentAddress,
          data = ((writerAGUParam.address >> 32) & 0xffff_ffff).toInt
        )
        currentAddress += 1

        // Ptr 1-3 is 0
        currentAddress += 6

        // Reader side Strides + Bounds
        readerAGUParam.spatialStrides.foreach { i =>
          write_csr(
            dut = dut,
            port = dut.io.instanceOne.csrIO,
            addr = currentAddress,
            data = i
          )
          currentAddress += 1
        }

        readerAGUParam.temporalBounds.foreach { i =>
          write_csr(
            dut = dut,
            port = dut.io.instanceOne.csrIO,
            addr = currentAddress,
            data = i
          )
          currentAddress += 1
        }

        readerAGUParam.temporalStrides.foreach { i =>
          write_csr(
            dut = dut,
            port = dut.io.instanceOne.csrIO,
            addr = currentAddress,
            data = i
          )
          currentAddress += 1
        }

        // Enabled Channel and Byte
        write_csr(
          dut = dut,
          port = dut.io.instanceOne.csrIO,
          addr = currentAddress,
          data = readerRWParam.enabledChannel
        )
        currentAddress += 1
        // Enabled Byte is not valid for the reader
        // No extension for the reader

        // Writer side Strides + Bounds
        writerAGUParam.spatialStrides.foreach { i =>
          write_csr(
            dut = dut,
            port = dut.io.instanceOne.csrIO,
            addr = currentAddress,
            data = i
          )
          currentAddress += 1
        }

        writerAGUParam.temporalBounds.foreach { i =>
          write_csr(
            dut = dut,
            port = dut.io.instanceOne.csrIO,
            addr = currentAddress,
            data = i
          )
          currentAddress += 1
        }

        writerAGUParam.temporalStrides.foreach { i =>
          write_csr(
            dut = dut,
            port = dut.io.instanceOne.csrIO,
            addr = currentAddress,
            data = i
          )
          currentAddress += 1
        }

        // Enabled Channel and Byte
        write_csr(
          dut = dut,
          port = dut.io.instanceOne.csrIO,
          addr = currentAddress,
          data = writerRWParam.enabledChannel
        )
        currentAddress += 1
        write_csr(
          dut = dut,
          port = dut.io.instanceOne.csrIO,
          addr = currentAddress,
          data = writerRWParam.enabledByte
        )
        currentAddress += 1

        // Data Extension Region
        // Bypass signals
        write_csr(
          dut = dut,
          port = dut.io.instanceOne.csrIO,
          addr = currentAddress,
          data =
            (writerExtParam.bypassMemset << 0) + (writerExtParam.bypassMaxPool << 1) + (writerExtParam.bypassTransposer << 2)
        )
        currentAddress += 1

        write_csr(
          dut = dut,
          port = dut.io.instanceOne.csrIO,
          addr = currentAddress,
          data = writerExtParam.memsetValue
        )
        currentAddress += 1

        write_csr(
          dut = dut,
          port = dut.io.instanceOne.csrIO,
          addr = currentAddress,
          data = writerExtParam.maxPoolPeriod
        )
        currentAddress += 1

        // Start the DMA
        write_csr(
          dut = dut,
          port = dut.io.instanceOne.csrIO,
          addr = currentAddress,
          data = 1
        )
        currentAddress += 1

        // Check if the DMA is finished
        while (
          read_csr(
            dut = dut,
            port = dut.io.instanceOne.csrIO,
            addr = currentAddress + 1
          ) != 1
        ) {}

        // Check the data in the TCDM
        if (tcdmMem_1 != tcdmMem_2)
          throw new Exception(
            "[TEST Failed] Test 1: Use XDMA 1 as a host to copy the data from TCDM 1 to TCDM 2"
          )
        else
          println(
            "[TEST Passed] Test 1: Use XDMA 1 as a host to copy the data from TCDM 1 to TCDM 2"
          )

      }

  }
}
