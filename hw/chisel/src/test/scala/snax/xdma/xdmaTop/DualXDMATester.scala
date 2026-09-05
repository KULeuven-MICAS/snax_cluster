package snax.xdma.xdmaTop

import chisel3._
// Hardware and its Generation Param
import snax.readerWriter.ReaderWriterParam
import snax.utils.DecoupledCut._

// Import Chiseltest
import chiseltest._
import org.scalatest.freespec.AnyFreeSpec

// Import break support for loops
import scala.util.control.Breaks.{break, breakable}
import snax.xdma.DesignParams._
import snax.DataPathExtension.HasVerilogMemset

class DualXDMA(readerParam: XDMAParam, writerParam: XDMAParam) extends Module with RequireAsyncReset {
  val xdma1 = Module(
    new XDMATop(
      clusterName = "xdma1",
      readerParam = readerParam,
      writerParam = writerParam
    )
  )

  val xdma2 = Module(
    new XDMATop(
      clusterName = "xdma2",
      readerParam = readerParam,
      writerParam = writerParam
    )
  )

  xdma1.io.clusterBaseAddress := 0x10000000.U
  xdma2.io.clusterBaseAddress := (0x10000000 + (1 << 20)).U

  xdma2.io.remoteXDMACfg.toRemote -||> xdma1.io.remoteXDMACfg.fromRemote
  xdma1.io.remoteXDMACfg.toRemote -||> xdma2.io.remoteXDMACfg.fromRemote

  xdma2.io.remoteXDMAData.toRemote -||> xdma1.io.remoteXDMAData.fromRemote
  xdma1.io.remoteXDMAData.toRemote -||> xdma2.io.remoteXDMAData.fromRemote

  xdma1.io.remoteTaskFinished := 0.U
  xdma2.io.remoteTaskFinished := 0.U

  val io = IO(new Bundle {
    val instance1 = new Bundle {
      val csrIO      = chiselTypeOf(xdma1.io.csrIO)
      val tcdmReader = chiselTypeOf(xdma1.io.tcdmReader)
      val tcdmWriter = chiselTypeOf(xdma1.io.tcdmWriter)
      val readerBusy = Output(Bool())
      val writerBusy = Output(Bool())
    }
    val instance2 = new Bundle {
      val csrIO      = chiselTypeOf(xdma2.io.csrIO)
      val tcdmReader = chiselTypeOf(xdma2.io.tcdmReader)
      val tcdmWriter = chiselTypeOf(xdma2.io.tcdmWriter)
      val readerBusy = Output(Bool())
      val writerBusy = Output(Bool())
    }
  })

  io.instance1.csrIO <> xdma1.io.csrIO
  io.instance1.tcdmReader <> xdma1.io.tcdmReader
  io.instance1.tcdmWriter <> xdma1.io.tcdmWriter
  io.instance2.csrIO <> xdma2.io.csrIO
  io.instance2.tcdmReader <> xdma2.io.tcdmReader
  io.instance2.tcdmWriter <> xdma2.io.tcdmWriter
  io.instance1.readerBusy := xdma1.io.status.readerBusy
  io.instance1.writerBusy := xdma1.io.status.writerBusy
  io.instance2.readerBusy := xdma2.io.status.readerBusy
  io.instance2.writerBusy := xdma2.io.status.writerBusy

  dontTouch(xdma1.io)
  dontTouch(xdma2.io)
}

// Cross-cluster writer-side extension config, end to end on two REAL XDMATop instances linked exactly as they are
// cross-cluster (toRemote -||> fromRemote both ways), not a single-instance poke. Instance 1 pushes a beat into
// instance 2 with instance 2's writer-side extension armed via `writerExtCfg`, the cross-cluster carry mechanism
// in XDMACfgIO.scala.
//
// The extension under test is VerilogMemset, whose transform is unambiguous: it replaces every byte of the beat
// with csr(0)[7:0]. So the landed word distinguishes the two outcomes exactly -- the memset pattern means
// `writerExtCfg` reached instance 2 and ACTIVATED the extension on the remote-origin frame, whereas the raw
// seeded beat means the config did not arrive and the extension stayed bypassed.
class DualXDMATester extends AnyFreeSpec with ChiselScalatestTester {

  // ************************ Prepare the simulation data ************************//

  // TCDM memory models for both instances -- addresses are cluster-relative (0-based), the same convention the
  // reader/writer TCDM ports present locally regardless of which cluster's base address routed the transfer here.
  val tcdmMem_1      = collection.mutable.Map[Long, BigInt]()
  val tcdmMem_2      = collection.mutable.Map[Long, BigInt]()
  var testTerminated = false

  // One 512-bit beat = 8 TCDM words of 64 bits. The payload is arbitrary but must NOT already look like the
  // memset pattern, so that a bypassed extension is distinguishable from an active one.
  val memsetByte = 0xa5
  val memsetWord = (0 until 8).foldLeft(BigInt(0))((acc, i) => acc | (BigInt(memsetByte) << (8 * i)))

  for (i <- 0 until 8) tcdmMem_1(8L * i) = BigInt("0123456789ABCDEF", 16) + i
  require(tcdmMem_1(0L) != memsetWord, "seed must differ from the memset pattern")
  println(f"[TCDM] TCDM 1 seeded with word0=0x${tcdmMem_1(0L)}%x -- golden (memset) word = 0x$memsetWord%x")

  "Dual_XDMA_CrossCluster_WriterExtCfg" in test(
    new DualXDMA(
      readerParam = new XDMAParam(
        cfgParam          = new XDMAConfigParam(addrWidth = 32, dataWidth = 32),
        axiParam          = new XDMAAXIParam,
        crossClusterParam = new XDMACrossClusterParam,
        rwParam           = new ReaderWriterParam(
          configurableByteMask = false,
          configurableChannel  = true
        )
      ),
      writerParam = new XDMAParam(
        cfgParam          = new XDMAConfigParam(addrWidth = 32, dataWidth = 32),
        axiParam          = new XDMAAXIParam,
        crossClusterParam = new XDMACrossClusterParam,
        rwParam           = new ReaderWriterParam(
          configurableByteMask = true,
          configurableChannel  = true
        ),
        extParam          = Seq(
          new HasVerilogMemset
        )
      )
    )
  ).withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
    // ************************ Start Simulation **********************************//
    var concurrent_threads = new chiseltest.internal.TesterThreadList(Seq())

    // Reader req/resp + writer req TCDM emulation for both instances (extension-agnostic).
    val queues_xdma1 = Seq.fill(8)(collection.mutable.Queue[Int]())
    for (i <- 0 until 8) {
      concurrent_threads = concurrent_threads.fork {
        breakable(while (true) {
          if (testTerminated) break()
          dut.io.instance1.tcdmReader.req(i).ready.poke(true)
          val reader_req_addr = dut.io.instance1.tcdmReader.req(i).bits.addr.peekInt().toInt
          if (dut.io.instance1.tcdmReader.req(i).valid.peekBoolean()) {
            queues_xdma1(i).enqueue(reader_req_addr)
          }
          dut.clock.step()
        })
      }
    }

    val queues_xdma2 = Seq.fill(8)(collection.mutable.Queue[Int]())
    for (i <- 0 until 8) {
      concurrent_threads = concurrent_threads.fork {
        breakable(while (true) {
          if (testTerminated) break()
          dut.io.instance2.tcdmReader.req(i).ready.poke(true)
          val reader_req_addr = dut.io.instance2.tcdmReader.req(i).bits.addr.peekInt().toInt
          if (dut.io.instance2.tcdmReader.req(i).valid.peekBoolean()) {
            queues_xdma2(i).enqueue(reader_req_addr)
          }
          dut.clock.step()
        })
      }
    }

    for (i <- 0 until 8) {
      concurrent_threads = concurrent_threads.fork {
        breakable(while (true) {
          if (testTerminated) break()
          if (queues_xdma1(i).isEmpty) dut.clock.step()
          else {
            dut.io.instance1.tcdmReader.rsp(i).valid.poke(true)
            val reader_addr      = queues_xdma1(i).dequeue()
            val reader_resp_data = tcdmMem_1.getOrElse(reader_addr.toLong, BigInt(0))
            dut.io.instance1.tcdmReader.rsp(i).bits.data.poke(reader_resp_data.U)
            dut.clock.step()
            dut.io.instance1.tcdmReader.rsp(i).valid.poke(false)
          }
        })
      }
    }

    for (i <- 0 until 8) {
      concurrent_threads = concurrent_threads.fork {
        breakable(while (true) {
          if (testTerminated) break()
          if (queues_xdma2(i).isEmpty) dut.clock.step()
          else {
            dut.io.instance2.tcdmReader.rsp(i).valid.poke(true)
            val reader_addr      = queues_xdma2(i).dequeue()
            val reader_resp_data = tcdmMem_2.getOrElse(reader_addr.toLong, BigInt(0))
            dut.io.instance2.tcdmReader.rsp(i).bits.data.poke(reader_resp_data.U)
            dut.clock.step()
            dut.io.instance2.tcdmReader.rsp(i).valid.poke(false)
          }
        })
      }
    }

    for (i <- 0 until 8) {
      concurrent_threads = concurrent_threads.fork {
        breakable(while (true) {
          if (testTerminated) break()
          if (dut.io.instance1.tcdmWriter.req(i).valid.peekBoolean()) {
            val writer_req_addr = dut.io.instance1.tcdmWriter.req(i).bits.addr.peekInt().toInt
            val writer_req_data = dut.io.instance1.tcdmWriter.req(i).bits.data.peekInt()
            dut.io.instance1.tcdmWriter.req(i).ready.poke(true)
            val previous_data = tcdmMem_1.getOrElse(writer_req_addr.toLong, BigInt(0))
            val Strb          = dut.io.instance1.tcdmWriter.req(i).bits.strb.peekInt().toInt
            var bitStrb       = BigInt(0)
            for (b <- 7 to 0 by -1) {
              val bit   = (Strb >> b) & 1
              val block = (BigInt(255) * bit) << (b * 8)
              bitStrb |= block
            }
            tcdmMem_1(writer_req_addr.toLong) = (previous_data & (~bitStrb)) | (writer_req_data & bitStrb)
            dut.clock.step()
          } else dut.clock.step()
        })
      }
    }

    for (i <- 0 until 8) {
      concurrent_threads = concurrent_threads.fork {
        breakable(while (true) {
          if (testTerminated) break()
          if (dut.io.instance2.tcdmWriter.req(i).valid.peekBoolean()) {
            val writer_req_addr = dut.io.instance2.tcdmWriter.req(i).bits.addr.peekInt().toInt
            val writer_req_data = dut.io.instance2.tcdmWriter.req(i).bits.data.peekInt()
            dut.io.instance2.tcdmWriter.req(i).ready.poke(true)
            val previous_data = tcdmMem_2.getOrElse(writer_req_addr.toLong, BigInt(0))
            val Strb          = dut.io.instance2.tcdmWriter.req(i).bits.strb.peekInt().toInt
            var bitStrb       = BigInt(0)
            for (b <- 7 to 0 by -1) {
              val bit   = (Strb >> b) & 1
              val block = (BigInt(255) * bit) << (b * 8)
              bitStrb |= block
            }
            tcdmMem_2(writer_req_addr.toLong) = (previous_data & (~bitStrb)) | (writer_req_data & bitStrb)
            println(
              f"[XDMA 2 Writer Req] landed word@0x${writer_req_addr.toHexString} = 0x${tcdmMem_2(writer_req_addr.toLong).toString(16)}"
            )
            dut.clock.step()
          } else dut.clock.step()
        })
      }
    }

    // Local CSR-sequencing helper (deliberately NOT XDMATesterInfrastructure.setXDMA/ExtParam -- those are
    // hardcoded to the OLD 3-extension {Memset,MaxPool,Transposer} CSR layout and are the one currently-passing
    // test's (XDMATopTester) shared infra; a single-extension writer needs a different ext-region CSR count, so
    // this stays local to avoid touching that shared file).
    def writeCsr(addr: Int, data: Long): Unit = {
      dut.io.instance1.csrIO.req.bits.write.poke(true.B)
      dut.io.instance1.csrIO.req.bits.strb.poke("b1111".U)
      dut.io.instance1.csrIO.req.bits.data.poke((data & 0xffffffffL).U)
      dut.io.instance1.csrIO.req.bits.addr.poke(addr.U)
      dut.io.instance1.csrIO.req.valid.poke(true.B)
      while (!dut.io.instance1.csrIO.req.ready.peekBoolean()) dut.clock.step(1)
      dut.clock.step(1)
      dut.io.instance1.csrIO.req.valid.poke(false.B)
    }

    concurrent_threads = concurrent_threads.fork {
      try {
        println("[TEST] instance1 pushes a beat into instance2 with instance2's writer extension armed cross-cluster")

        var addr = 0
        // Reader (local, instance1) pointer -- 48b split hi/lo.
        val readerBase = 0x1000_0000L
        writeCsr(addr, readerBase & 0xffffffffL); addr += 1
        writeCsr(addr, (readerBase >> 32) & 0xffffffffL); addr += 1
        // Writer (remote, instance2) pointer.
        val writerBase = 0x1000_0000L + (1L << 20)
        writeCsr(addr, writerBase & 0xffffffffL); addr += 1
        writeCsr(addr, (writerBase >> 32) & 0xffffffffL); addr += 1
        // The writer-pointer CSR region is numCSRPerPtr * maxMulticastDest words wide (2 * 4 with the default
        // XDMACrossClusterParam), not just the two words of destination 0: XDMACfgIO.connectPtrWithList reserves a
        // slot for EVERY multicast destination, so the unused ones must still be written or every field below
        // lands at the wrong CSR. Zeroing them also sets writerPtr(1) = 0, i.e. "not a chained write".
        for (_ <- 0 until 6) { writeCsr(addr, 0); addr += 1 }
        // Reader spatial stride (8B/channel, 8 channels = 64B/beat) + temporal bounds/strides padded to 5 dims,
        // one temporal step -> exactly one beat.
        writeCsr(addr, 8); addr += 1
        writeCsr(addr, 1); addr += 1 // temporalBounds(0) = 1 beat
        for (_ <- 0 until 4) { writeCsr(addr, 1); addr += 1 } // pad remaining 4 dims (bound=1)
        writeCsr(addr, 64); addr += 1 // temporalStrides(0)
        for (_ <- 0 until 4) { writeCsr(addr, 0); addr += 1 } // pad remaining 4 dims (stride=0)
        // Reader enabledChannel (writer's enabledByte has no reader-side counterpart).
        writeCsr(addr, Integer.parseInt("11111111", 2)); addr += 1

        // Writer spatial stride + temporal bounds/strides, same one-beat shape.
        writeCsr(addr, 8); addr += 1
        writeCsr(addr, 1); addr += 1
        for (_ <- 0 until 4) { writeCsr(addr, 1); addr += 1 }
        writeCsr(addr, 64); addr += 1
        for (_ <- 0 until 4) { writeCsr(addr, 0); addr += 1 }
        // Writer enabledChannel + enabledByte.
        writeCsr(addr, Integer.parseInt("11111111", 2)); addr += 1
        writeCsr(addr, Integer.parseInt("11111111", 2)); addr += 1

        // Extension region: HasVerilogMemset is writer-ext index 0 (the only writer extension here) ->
        // 1 bypass/enable-bitmask CSR (bit0=1 activates it) + its 1 CSR.
        writeCsr(addr, 1); addr += 1          // enable bitmask: bit0 = VerilogMemset ON
        writeCsr(addr, memsetByte); addr += 1 // csr(0)[7:0] = the byte written into every lane

        // Start.
        writeCsr(addr, 1); addr += 1

        // Wait for instance2's writer to go busy then idle (the remote-origin push landing).
        dut.clock.step(16)
        var w = 0
        while (dut.io.instance2.writerBusy.peekBoolean() && w < 2000) { dut.clock.step(1); w += 1 }
        dut.clock.step(16)

        val landed = tcdmMem_2.getOrElse(0L, BigInt(0))
        println(f"[TEST] instance2 TCDM word0 = 0x$landed%x (memset golden 0x$memsetWord%x, raw seed 0x${tcdmMem_1(0L)}%x)")
        if (landed == tcdmMem_1(0L))
          throw new Exception(
            "[TEST Failed] the raw beat landed: writerExtCfg did not activate instance2's writer extension"
          )
        else if (landed != memsetWord)
          throw new Exception(f"[TEST Failed] landed 0x$landed%x, expected the memset pattern 0x$memsetWord%x")
        else
          println("[TEST Passed] writerExtCfg carried the extension config cross-cluster and the extension was active")
      } finally {
        // The TCDM emulation threads above spin `while (true)` until this flag is set, so it must be set on EVERY
        // exit path. Setting it in a `finally` keeps a failing check from leaving them running, which would stop
        // joinAndStep from ever returning.
        testTerminated = true
        // Then step long enough for every one of them to reach its loop head, observe the flag and return. Without
        // this drain the driver can finish while children are still scheduled, and chiseltest's teardown trips over
        // its own thread bookkeeping (IndexOutOfBoundsException in terminateAllChildThreads).
        dut.clock.step(4)
      }
    }

    concurrent_threads.joinAndStep()
  }
}
