package snax.xdma.xdmaTop

import chisel3._
// Hardware and its Generation Param
import snax.readerWriter.ReaderWriterParam
import snax.utils.DecoupledCut._

// Import Chiseltest
import chiseltest._
import org.scalatest.freespec.AnyFreeSpec

// Import Random number generator
import scala.util.Random

// Import break support for loops
import scala.util.control.Breaks.{break, breakable}
import snax.xdma.DesignParams._
import snax.DataPathExtension.HasStreamMomentMergeRt

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

// KNOWN OPEN ISSUE (2026-07-19): this test currently fails -- the writer-request debug print
// ("[XDMA 2 Writer Req] landed word@...") never fires, meaning instance2's tcdmWriter port never
// sees a request at all; the transfer appears to never reach the writer stage. Not yet root-caused;
// candidates to check next: (1) the 1-beat AGU shape (temporalBounds all-1s, 5 dims padded) may hit
// an edge case a bigger (multi-beat) transfer doesn't -- try temporalBounds=(2,1,1,1,1) as a probe;
// (2) verify the hand-written CSR address sequence in this file actually matches
// snax_xdma_test.hjson's writer_extensions CSR count (1 ext, VERILOGMEMSET/STREAMDECOMPRESSRT are
// NOT in this test's extParam, only HasStreamMomentMergeRt, so ext-region CSR count should be
// 1 bypass-bitmask + 1 nValid = 2 writes -- recount against XDMACtrl.scala's actual CSR layout,
// don't just trust the by-hand derivation below); (3) a VCD was requested (WriteVcdAnnotation) --
// inspect it in test_run_dir/Dual_XDMA_InFabric_MomentMerge/ for where the task actually stalls.
//
// F3 S0b -- the cross-cluster in-fabric collective, end to end on two REAL XDMATop instances linked exactly as
// they are cross-cluster (toRemote -||> fromRemote both ways), not a single-instance poke. Instance 1 pushes a
// pre-packed 2-lane (m,l) beat into instance 2 with the writer-side StreamMomentMergeRt extension armed via
// `writerExtCfg` -- the same cross-cluster carry mechanism XDMACfgIO.scala adds. If `writerExtCfg` fails to reach
// instance 2 (wrong serdes bits, wrong frame layout, extension not actually activated on the remote-origin
// frame), the landed beat is either raw/unprocessed or garbage; if it works, instance 2's TCDM holds the folded
// (m*, l*), decodable and checkable against the same golden model StreamMomentMergeRtTester.scala uses.
class DualXDMATester extends AnyFreeSpec with ChiselScalatestTester {

  // ---- FP32 <-> Double helpers + the moment-merge golden ----
  // NOTE: this is NOT StreamMomentMergeRtTester.scala's global() (that one treats its input as
  // RAW SCORES, implicitly weight-1 each -- global([m0,m1]) = (max, exp(m0-max)+exp(m1-max))).
  // Here (m_i, l_i) are ALREADY-COMPUTED shard moments, so l_i must weight its own exp term:
  // m* = max_i(m_i), l* = sum_i( l_i * exp(m_i - m*) ) -- the actual momentMerge fold
  // (FpHelpers.scala) applied pairwise, generalized to N terms.
  private def f32(f: Double): BigInt = BigInt(java.lang.Float.floatToIntBits(f.toFloat).toLong & 0xffffffffL)
  private def dec(b:  BigInt): Double = java.lang.Float.intBitsToFloat(b.toInt).toDouble
  private def momentMergeGolden(m: Seq[Double], l: Seq[Double]): (Double, Double) = {
    val mStar = m.max
    (mStar, (m zip l).map { case (mi, li) => li * math.exp(mi - mStar) }.sum)
  }

  // ************************ Prepare the simulation data ************************//

  // TCDM memory models for both instances -- addresses are cluster-relative (0-based), the same convention the
  // reader/writer TCDM ports present locally regardless of which cluster's base address routed the transfer here.
  val tcdmMem_1      = collection.mutable.Map[Long, BigInt]()
  val tcdmMem_2      = collection.mutable.Map[Long, BigInt]()
  var testTerminated = false

  // The P=2 in-fabric merge case: pack (m0,l0),(m1,l1) into one 512-bit (8 x 64b word) beat at tcdmMem_1[0..56],
  // lane k = m_k (word k low 32b), lane 8+k = l_k (word (8+k) low 32b) -- matches StreamMomentMergeRt.scala's
  // beat layout. Lanes 2..7 / 10..15 are don't-care (nValid=2 masks them) so left at zero.
  val m0 = 1.0; val l0 = 2.0
  val m1 = 3.0; val l1 = 0.5
  val (mGold, lGold) = momentMergeGolden(Seq(m0, m1), Seq(l0, l1))

  for (i <- 0 until 16) tcdmMem_1(8L * i) = BigInt(0)
  tcdmMem_1(8L * 0) = f32(m0)                  // word 0 low 32b = lane 0 = m0
  tcdmMem_1(8L * 8) = f32(l0)                  // word 8 low 32b = lane 8 = l0
  tcdmMem_1(8L * 1) = f32(m1)                  // word 1 low 32b = lane 1 = m1
  tcdmMem_1(8L * 9) = f32(l1)                  // word 9 low 32b = lane 9 = l1
  println(f"[TCDM] TCDM 1 seeded: (m0,l0)=($m0%.3f,$l0%.3f) (m1,l1)=($m1%.3f,$l1%.3f) -- golden (m*,l*)=($mGold%.6f,$lGold%.6f)")

  "Dual_XDMA_InFabric_MomentMerge" in test(
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
          new HasStreamMomentMergeRt(dataWidth = 512)
        )
      )
    )
  ).withAnnotations(Seq(VerilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>
    // ************************ Start Simulation **********************************//
    var concurrent_threads = new chiseltest.internal.TesterThreadList(Seq())

    // Reader req/resp + writer req TCDM emulation, both instances -- extension-agnostic, unchanged shape from
    // the original dual-instance plain-copy test.
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
      println("[TEST] P=2 in-fabric moment-merge: instance1 pushes a 2-lane beat into instance2, ext armed cross-cluster")

      var addr = 0
      // Reader (local, instance1) pointer -- 48b split hi/lo.
      val readerBase = 0x1000_0000L
      writeCsr(addr, readerBase & 0xffffffffL); addr += 1
      writeCsr(addr, (readerBase >> 32) & 0xffffffffL); addr += 1
      // Writer (remote, instance2) pointer.
      val writerBase = 0x1000_0000L + (1L << 20)
      writeCsr(addr, writerBase & 0xffffffffL); addr += 1
      writeCsr(addr, (writerBase >> 32) & 0xffffffffL); addr += 1
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

      // Extension region: HasStreamMomentMergeRt is writer-ext index 0 (the only writer extension here) ->
      // 1 bypass/enable-bitmask CSR (bit0=1 activates it) + its 1 CSR (nValid=2).
      writeCsr(addr, 1); addr += 1 // enable bitmask: bit0 = StreamMomentMergeRt ON
      writeCsr(addr, 2); addr += 1 // nValid = 2 live (m,l) pairs in the beat

      // Start.
      writeCsr(addr, 1); addr += 1

      // Wait for instance2's writer to go busy then idle (the remote-origin push landing + fold).
      dut.clock.step(16)
      var w = 0
      while (dut.io.instance2.writerBusy.peekBoolean() && w < 2000) { dut.clock.step(1); w += 1 }
      dut.clock.step(16)

      val landed = tcdmMem_2.getOrElse(0L, BigInt(0))
      val mHw    = dec(landed & ((BigInt(1) << 32) - 1))
      val lHw    = dec((landed >> 32) & ((BigInt(1) << 32) - 1))
      val mOk    = math.abs(mHw - mGold) <= 1e-6
      val relL   = math.abs(lHw - lGold) / lGold
      val lOk    = relL <= 1.5e-2

      println(f"[TEST] got (m*,l*)=($mHw%.6f,$lHw%.6f) golden=($mGold%.6f,$lGold%.6f) relErrL=$relL%.4g")
      if (!mOk || !lOk)
        throw new Exception(
          f"[TEST Failed] P=2 in-fabric moment-merge: got (m*,l*)=($mHw%.6f,$lHw%.6f) vs golden ($mGold%.6f,$lGold%.6f)"
        )
      else
        println("[TEST Passed] P=2 in-fabric moment-merge: writerExtCfg carried the ext config cross-cluster and StreamMomentMergeRt folded correctly")

      testTerminated = true
    }

    concurrent_threads.joinAndStep()
  }
}
