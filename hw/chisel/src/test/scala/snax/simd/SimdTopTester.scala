package snax.simd

import scala.collection.mutable

import chisel3._

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import snax.readerWriter.ReaderWriterParam
import snax.xdma.xdmaTop.XDMATesterInfrastructure.read_csr
import snax.xdma.xdmaTop.XDMATesterInfrastructure.write_csr

/** Tier-1 tests for the standalone SIMD block.
  *
  * The point of these is NOT the arithmetic -- every extension already has its own tester under
  * snax/DataPathExtension. The point is the plumbing that is new in SimdTop and therefore untested anywhere else:
  * the CSR decode order, the task FSM, and above all COMPLETION. A block test that only checks the data can pass
  * while the engine never lowers busy, or lowers it early and lets the next task start on top of the previous one
  * -- that is the failure mode that hid the xDMA streaming-FSM hang, so `simd_multitask_busy` below is the test
  * that matters most.
  */
class SimdTopTester extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SimdTop"

  // A small, extension-free instance: plumbing only, and fast to elaborate.
  private val numChannel    = 8
  private val tcdmDataWidth = 64
  private val tcdmSize      = 128 // KiB
  private val beatBytes     = numChannel * tcdmDataWidth / 8

  private def mkParam(temporalDim: Int) = new SimdParam(
    cfgParam    = new SimdConfigParam(addrWidth = 32, dataWidth = 32),
    readerParam = new ReaderWriterParam(
      spatialBounds        = List(numChannel),
      temporalDimension    = temporalDim,
      tcdmDataWidth        = tcdmDataWidth,
      tcdmSize             = tcdmSize,
      numChannel           = numChannel,
      addressBufferDepth   = 8,
      dataBufferDepth      = 8,
      configurableChannel  = true,
      configurableByteMask = false,
      dynamicPriority      = true,
      higherStaticPriority = false
    ),
    writerParam = new ReaderWriterParam(
      spatialBounds        = List(numChannel),
      temporalDimension    = temporalDim,
      tcdmDataWidth        = tcdmDataWidth,
      tcdmSize             = tcdmSize,
      numChannel           = numChannel,
      addressBufferDepth   = 8,
      dataBufferDepth      = 8,
      configurableChannel  = true,
      configurableByteMask = true,
      dynamicPriority      = true,
      higherStaticPriority = false
    ),
    extParam    = Seq()
  )

  /** CSR addresses, derived the same way SimdTopGen derives the SW header. Keeping this arithmetic here rather
    * than hard-coding numbers means a CSR-layout change breaks the test loudly instead of silently.
    */
  private class CsrMap(param: SimdParam, temporalDim: Int) {
    val srcPtrLsb   = 0
    val srcPtrMsb   = 1
    val srcSpatial  = 2
    val srcBound    = srcSpatial + 1
    val srcStride   = srcBound + temporalDim
    val srcChan     = srcStride + temporalDim
    val extEnable   = srcChan + 1
    val dstPtrLsb   = extEnable + param.extCsrNum
    val dstPtrMsb   = dstPtrLsb + 1
    val dstSpatial  = dstPtrMsb + 1
    val dstBound    = dstSpatial + 1
    val dstStride   = dstBound + temporalDim
    val dstChan     = dstStride + temporalDim
    val dstByte     = dstChan + 1
    val start       = dstByte + 1
    val submitted   = start + 1
    val finished    = submitted + 1
    val perfTask    = finished + 1
    val status      = perfTask + 3
    require(start + 1 == param.totalRwCsrNum, s"CSR map drift: start=$start, rw=${param.totalRwCsrNum}")
  }

  /** A one-cycle-latency TCDM behavioural model, driven from the test thread.
    *
    * Reads are answered on the next cycle; writes are recorded. Deliberately always-ready: back-pressure
    * behaviour is the interconnect's business and is covered by the roofline work, not here.
    */
  private class TcdmModel(mem: mutable.Map[Long, BigInt]) {
    private val pending = mutable.Queue[(Int, Option[BigInt])]()

    def step(dut: SimdTop): Unit = {
      // Drive last cycle's read answers.
      val answers = Array.fill(numChannel)(Option.empty[BigInt])
      while (pending.nonEmpty) {
        val (ch, data) = pending.dequeue()
        answers(ch) = data
      }
      for (ch <- 0 until numChannel) {
        dut.io.tcdmReader.rsp(ch).valid.poke(answers(ch).isDefined.B)
        dut.io.tcdmReader.rsp(ch).bits.data.poke(answers(ch).getOrElse(BigInt(0)).U)
      }

      // Accept this cycle's requests.
      for (ch <- 0 until numChannel) {
        dut.io.tcdmReader.req(ch).ready.poke(true.B)
        if (dut.io.tcdmReader.req(ch).valid.peekBoolean()) {
          val addr = dut.io.tcdmReader.req(ch).bits.addr.peekInt().toLong
          pending.enqueue((ch, Some(mem.getOrElse(addr, BigInt(0)))))
        }
        dut.io.tcdmWriter.req(ch).ready.poke(true.B)
        if (dut.io.tcdmWriter.req(ch).valid.peekBoolean()) {
          val addr = dut.io.tcdmWriter.req(ch).bits.addr.peekInt().toLong
          mem(addr) = dut.io.tcdmWriter.req(ch).bits.data.peekInt()
        }
      }
    }
  }

  /** Program one copy task: `beats` beats from `src` to `dst`, contiguous. */
  private def program(
    dut:   SimdTop,
    csr:   CsrMap,
    param: SimdParam,
    tdim:  Int,
    src:   Long,
    dst:   Long,
    beats: Int
  ): Unit = {
    val port = dut.io.csrIO
    write_csr(dut, port, csr.srcPtrLsb, (src & 0xffffffffL).toInt)
    write_csr(dut, port, csr.srcPtrMsb, ((src >> 32) & 0xffffffffL).toInt)
    write_csr(dut, port, csr.srcSpatial, tcdmDataWidth / 8) // one 64-bit word per channel
    write_csr(dut, port, csr.srcBound, beats)
    for (i <- 1 until tdim) write_csr(dut, port, csr.srcBound + i, 1)
    write_csr(dut, port, csr.srcStride, beatBytes)
    for (i <- 1 until tdim) write_csr(dut, port, csr.srcStride + i, 0)
    write_csr(dut, port, csr.srcChan, (1 << numChannel) - 1)

    if (param.extCsrNum > 0) {
      for (i <- 0 until param.extCsrNum) write_csr(dut, port, csr.extEnable + i, 0)
    }

    write_csr(dut, port, csr.dstPtrLsb, (dst & 0xffffffffL).toInt)
    write_csr(dut, port, csr.dstPtrMsb, ((dst >> 32) & 0xffffffffL).toInt)
    write_csr(dut, port, csr.dstSpatial, tcdmDataWidth / 8)
    write_csr(dut, port, csr.dstBound, beats)
    for (i <- 1 until tdim) write_csr(dut, port, csr.dstBound + i, 1)
    write_csr(dut, port, csr.dstStride, beatBytes)
    for (i <- 1 until tdim) write_csr(dut, port, csr.dstStride + i, 0)
    write_csr(dut, port, csr.dstChan, (1 << numChannel) - 1)
    write_csr(dut, port, csr.dstByte, 0xff)
    write_csr(dut, port, csr.start, 1)
  }

  /** Step the model until `cond`, or fail after `limit` cycles. Never spin forever: a hung completion is the
    * exact bug these tests exist to catch, and a test that hangs reports nothing.
    */
  private def runUntil(dut: SimdTop, tcdm: TcdmModel, limit: Int, what: String)(cond: => Boolean): Int = {
    var cycles = 0
    while (!cond) {
      tcdm.step(dut)
      dut.clock.step(1)
      cycles += 1
      assert(cycles < limit, s"timeout after $limit cycles waiting for $what")
    }
    cycles
  }

  it should "copy a contiguous buffer and retire the task" in {
    val tdim  = 3
    val param = mkParam(tdim)
    val csr   = new CsrMap(param, tdim)
    test(new SimdTop(param, clusterName = "test"))
      .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
        dut.clock.setTimeout(0)
        val beats = 4
        val src   = 0x1000L
        val dst   = 0x4000L
        val mem   = mutable.Map[Long, BigInt]()
        for (w <- 0 until beats * numChannel) {
          mem(src + w * (tcdmDataWidth / 8)) = BigInt(0xa0000000L + w)
        }
        val tcdm = new TcdmModel(mem)

        program(dut, csr, param, tdim, src, dst, beats)
        runUntil(dut, tcdm, 200, "busy to rise")(dut.io.status.busy.peekBoolean())
        runUntil(dut, tcdm, 2000, "busy to fall")(!dut.io.status.busy.peekBoolean())
        // A few more cycles so the retire pulse lands in the finished counter.
        for (_ <- 0 until 8) { tcdm.step(dut); dut.clock.step(1) }

        for (w <- 0 until beats * numChannel) {
          val addr = dst + w * (tcdmDataWidth / 8)
          assert(
            mem.get(addr).contains(BigInt(0xa0000000L + w)),
            s"word $w at 0x${addr.toHexString}: expected 0x${(0xa0000000L + w).toHexString}, got ${mem.get(addr)}"
          )
        }
        assert(read_csr(dut, dut.io.csrIO, csr.finished) == 1, "finished-task counter did not reach 1")
      }
  }

  it should "run two back-to-back tasks without retiring early (multitask busy)" in {
    val tdim  = 3
    val param = mkParam(tdim)
    val csr   = new CsrMap(param, tdim)
    test(new SimdTop(param, clusterName = "test"))
      .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
        dut.clock.setTimeout(0)
        val beats = 4
        val srcA  = 0x1000L
        val dstA  = 0x4000L
        val srcB  = 0x8000L
        val dstB  = 0xc000L
        val mem   = mutable.Map[Long, BigInt]()
        for (w <- 0 until beats * numChannel) {
          mem(srcA + w * (tcdmDataWidth / 8)) = BigInt(0xa0000000L + w)
          mem(srcB + w * (tcdmDataWidth / 8)) = BigInt(0xb0000000L + w)
        }
        val tcdm = new TcdmModel(mem)

        // Queue both tasks before either completes -- this is what the 2-deep task queue is for, and it is the
        // case where an early retire silently starts task B on top of task A.
        program(dut, csr, param, tdim, srcA, dstA, beats)
        program(dut, csr, param, tdim, srcB, dstB, beats)

        runUntil(dut, tcdm, 200, "busy to rise")(dut.io.status.busy.peekBoolean())
        // Condition on the model's memory, never on a CSR read: read_csr steps the clock itself, and any cycle
        // the TcdmModel does not step is a cycle where the datapath sees stale responses.
        val lastB = dstB + (beats * numChannel - 1) * (tcdmDataWidth / 8)
        runUntil(dut, tcdm, 4000, "both tasks to finish") {
          mem.contains(lastB) && !dut.io.status.busy.peekBoolean()
        }
        for (_ <- 0 until 8) { tcdm.step(dut); dut.clock.step(1) }
        assert(read_csr(dut, dut.io.csrIO, csr.finished) == 2, "finished-task counter did not reach 2")

        for (w <- 0 until beats * numChannel) {
          val off = w * (tcdmDataWidth / 8)
          assert(mem.get(dstA + off).contains(BigInt(0xa0000000L + w)), s"task A word $w wrong")
          assert(mem.get(dstB + off).contains(BigInt(0xb0000000L + w)), s"task B word $w wrong")
        }
        // The bad-config watchdog must not have fired on a perfectly good pair of tasks.
        assert(
          (read_csr(dut, dut.io.csrIO, csr.status) & 2) == 0,
          "the never-started watchdog fired on a valid task"
        )
      }
  }
}
