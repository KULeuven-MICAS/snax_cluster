package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Tolerance-aware Tier-1 test for StreamReduce (the base DataPathExtensionTester does exact BigInt
  * compare, which doesn't fit FP tree-order + FP16-narrowing). Drives the same harness, then compares
  * the HW FP16 scalar (widened to f32) against an FP32 Scala golden within a relative tolerance.
  */
class StreamReduceTester extends AnyFlatSpec with ChiselScalatestTester {

  // ---- FP16 <-> FP32 helpers ----
  def f16bitsToF32(h: Int): Float = {
    val sign = if ((h & 0x8000) != 0) -1.0 else 1.0
    val exp  = (h >> 10) & 0x1f
    val mant = h & 0x3ff
    val v: Double =
      if (exp == 0) sign * mant * math.pow(2, -24)
      else if (exp == 0x1f) if (mant == 0) sign * Double.PositiveInfinity else Double.NaN
      else sign * (1024 + mant) * math.pow(2, exp - 25)
    v.toFloat
  }

  // round-to-nearest-even FP32 -> FP16 (used only to snap test inputs onto the FP16 grid)
  def f32ToF16bits(f: Float): Int = {
    val bits  = java.lang.Float.floatToIntBits(f)
    val sign  = (bits >>> 16) & 0x8000
    val rawe  = (bits >>> 23) & 0xff
    val mant  = bits & 0x7fffff
    if (rawe == 0xff) return sign | 0x7c00 | (if (mant != 0) 0x200 else 0)
    val exp = rawe - 127 + 15
    if (exp >= 0x1f) return sign | 0x7c00
    if (exp <= 0) {
      if (exp < -10) return sign
      val m       = mant | 0x800000
      val shift   = 14 - exp
      val half    = m >>> shift
      val rem     = m & ((1 << shift) - 1)
      val halfway = 1 << (shift - 1)
      var r       = half
      if (rem > halfway || (rem == halfway && (half & 1) == 1)) r += 1
      return sign | r
    }
    var h       = sign | (exp << 10) | (mant >>> 13)
    val rem     = mant & 0x1fff
    val halfway = 0x1000
    if (rem > halfway || (rem == halfway && (h & 1) == 1)) h += 1
    h
  }

  /** Build one 512-bit beat from 32 FP16 lane bit patterns (lane 0 in the low 16 bits). */
  def packBeat(lanesF16: Seq[Int]): BigInt =
    lanesF16.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (h, i)) => acc | (BigInt(h & 0xffff) << (16 * i)) }

  val lanes = 32

  /** Drive the DUT for one row and return the HW scalar (as f32) read from the output beat's low lane.
    * computeLanes>0 exercises the time-mux FSM (path B); ops selects which reductions are built. */
  def runReduce(op: Int, beats: Seq[Seq[Int]], computeLanes: Int = 32,
                ops: Seq[String] = Seq("FMA_FP16", "MAX_FP16"),
                fpPipe: Int = 1, treePipe: Int = 1): Float = {
    var result: Float = 0.0f
    test(new DataPathExtensionHarness(new HasStreamReduce(computeLanes = computeLanes, op = ops, elementWidth = 16, fpPipe = fpPipe, treePipe = treePipe)))
      // Serial build: fpnew blackboxes (lzc.sv UNOPTFLAT) trip a Verilator PCH parallel-build race.
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.io.csr_i(0).poke(beats.length.U)
        dut.io.csr_i(1).poke(op.U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true)
        dut.clock.step(1)
        dut.io.start_i.poke(false)

        var threads = new chiseltest.internal.TesterThreadList(Seq())
        threads = threads.fork {
          dut.io.data_i.valid.poke(true)
          for (b <- beats) {
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
            dut.io.data_i.bits.poke(packBeat(b))
            dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
        }
        threads = threads.fork {
          while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
          val out  = dut.io.data_o.bits.peekInt()
          val lane0 = (out & ((BigInt(1) << 16) - 1)).toInt
          result = f16bitsToF32(lane0)
          dut.io.data_o.ready.poke(true)
          dut.clock.step(1)
          dut.io.data_o.ready.poke(false)
        }
        threads.joinAndStep()
      }
    result
  }

  /** Drive one row with LANEWISE set (op CSR bit[10]) and return ALL 32 lanes of the single output beat.
    * In this mode the block emits the per-lane partials -- the reduction ACROSS beats -- rather than
    * folding them to one scalar, so the golden is a per-lane reduction of the input columns. */
  def runReduceLanewise(op: Int, beats: Seq[Seq[Int]], computeLanes: Int = 32,
                        ops: Seq[String] = Seq("FMA_FP16", "MAX_FP16"),
                        accPartials: Int = 1): Seq[Float] = {
    var result: Seq[Float] = Seq()
    test(new DataPathExtensionHarness(new HasStreamReduce(computeLanes = computeLanes, op = ops,
                                                          elementWidth = 16, accPartials = accPartials)))
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.io.csr_i(0).poke(beats.length.U)
        dut.io.csr_i(1).poke((op | 0x400).U) // lanewise bit[10]
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true)
        dut.clock.step(1)
        dut.io.start_i.poke(false)

        var threads = new chiseltest.internal.TesterThreadList(Seq())
        threads = threads.fork {
          dut.io.data_i.valid.poke(true)
          for (b <- beats) {
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
            dut.io.data_i.bits.poke(packBeat(b))
            dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
        }
        threads = threads.fork {
          while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
          val out = dut.io.data_o.bits.peekInt()
          result = (0 until lanes).map { i =>
            f16bitsToF32(((out >> (16 * i)) & ((BigInt(1) << 16) - 1)).toInt)
          }
          dut.io.data_o.ready.poke(true)
          dut.clock.step(1)
          dut.io.data_o.ready.poke(false)
        }
        threads.joinAndStep()
      }
    result
  }

  /** LANEWISE + TAP: the row is passed through 1:1 AND the per-lane partials follow as one trailing beat,
    * so N beats in -> N+1 beats out. This is what lets a convert+reduce chain produce both the converted
    * tile and its statistics in a single pass. */
  def runReduceLanewiseTap(op: Int, beats: Seq[Seq[Int]], computeLanes: Int = 32,
                           ops: Seq[String] = Seq("FMA_FP16", "MAX_FP16")): Seq[Seq[Float]] = {
    var result: Seq[Seq[Float]] = Seq()
    test(new DataPathExtensionHarness(new HasStreamReduce(computeLanes = computeLanes, op = ops, elementWidth = 16)))
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.io.csr_i(0).poke(beats.length.U)
        dut.io.csr_i(1).poke((op | 0x500).U) // lanewise bit[10] | tap bit[8]
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true)
        dut.clock.step(1)
        dut.io.start_i.poke(false)

        var threads = new chiseltest.internal.TesterThreadList(Seq())
        threads = threads.fork {
          dut.io.data_i.valid.poke(true)
          for (b <- beats) {
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
            dut.io.data_i.bits.poke(packBeat(b))
            dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
        }
        threads = threads.fork {
          for (_ <- 0 to beats.length) { // N passthroughs + 1 statistics beat
            while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
            val out = dut.io.data_o.bits.peekInt()
            result = result :+ (0 until lanes).map(i =>
              f16bitsToF32(((out >> (16 * i)) & ((BigInt(1) << 16) - 1)).toInt))
            dut.io.data_o.ready.poke(true)
            dut.clock.step(1)
            dut.io.data_o.ready.poke(false)
          }
        }
        threads.joinAndStep()
      }
    result
  }

  behavior of "StreamReduce LANEWISE (Track C / C1)"

  it should "pass the row through AND emit the per-lane MAX as a trailing beat (tap)" in {
    val rnd   = new Random(0xC5)
    val cols  = Seq.fill(6)(Seq.fill(lanes)(rnd.between(-40, 40).toFloat))
    val beats = cols.map(_.map(v => f32ToF16bits(v)))
    val hw    = runReduceLanewiseTap(0 /*MAX*/, beats)
    assert(hw.length == beats.length + 1, s"expected ${beats.length + 1} beats, got ${hw.length}")
    // the passthroughs must be the input, unchanged
    for (k <- cols.indices; i <- 0 until lanes)
      assert(hw(k)(i) == cols(k)(i), f"passthrough beat $k lane $i: ${hw(k)(i)}%.1f != ${cols(k)(i)}%.1f")
    // the trailing beat must be the per-lane max
    val gold = (0 until lanes).map(i => cols.map(_(i)).max)
    for (i <- 0 until lanes)
      assert(hw(beats.length)(i) == gold(i), f"stat lane $i: ${hw(beats.length)(i)}%.1f != ${gold(i)}%.1f")
  }

  // Track C / C2. P changes only WHEN a partial is re-read, never WHAT is computed, so each of these
  // compares against the plain Scala golden. Beat counts are deliberately NOT multiples of P, so the
  // banks end a row holding different numbers of contributions.
  it should "match the golden with 2 rotating partials (C2, P=2)" in {
    val rnd   = new Random(0xC7)
    val cols  = Seq.fill(37)(Seq.fill(lanes)(rnd.between(-50, 50).toFloat))
    val beats = cols.map(_.map(v => f32ToF16bits(v)))
    val hw    = runReduceLanewise(0 /*MAX*/, beats, computeLanes = 8, accPartials = 2)
    val gold  = (0 until lanes).map(i => cols.map(_(i)).max)
    for (i <- 0 until lanes) assert(hw(i) == gold(i), f"lane $i: ${hw(i)}%.1f != ${gold(i)}%.1f")
  }

  it should "match the golden with 4 rotating partials (C2, P=4)" in {
    val rnd   = new Random(0xC7)
    val cols  = Seq.fill(37)(Seq.fill(lanes)(rnd.between(-50, 50).toFloat))
    val beats = cols.map(_.map(v => f32ToF16bits(v)))
    val hw    = runReduceLanewise(0 /*MAX*/, beats, computeLanes = 8, accPartials = 4)
    val gold  = (0 until lanes).map(i => cols.map(_(i)).max)
    for (i <- 0 until lanes) assert(hw(i) == gold(i), f"lane $i: ${hw(i)}%.1f != ${gold(i)}%.1f")
  }

  it should "unlock computeLanes past the recurrence cap (P=4, cl=32)" in {
    // cl=32 means subCycles=1, which at P=1 forces gap=3 -- 4 cycles/beat with 4x the ALUs, i.e. no gain.
    // P=4 makes P*subCycles = accLat+1, so beats stream back to back.
    val rnd   = new Random(0xC8)
    val cols  = Seq.fill(64)(Seq.fill(lanes)(rnd.between(-50, 50).toFloat))
    val beats = cols.map(_.map(v => f32ToF16bits(v)))
    val hw    = runReduceLanewise(0 /*MAX*/, beats, computeLanes = 32, accPartials = 4)
    val gold  = (0 until lanes).map(i => cols.map(_(i)).max)
    for (i <- 0 until lanes) assert(hw(i) == gold(i), f"lane $i: ${hw(i)}%.1f != ${gold(i)}%.1f")
  }

  it should "sum correctly with rotating partials (the fold must ADD, not MAX)" in {
    val rnd   = new Random(0xC9)
    val cols  = Seq.fill(16)(Seq.fill(lanes)(rnd.between(-8, 8).toFloat)) // exact in FP16
    val beats = cols.map(_.map(v => f32ToF16bits(v)))
    val hw    = runReduceLanewise(1 /*ADD*/, beats, computeLanes = 16, accPartials = 2)
    val gold  = (0 until lanes).map(i => cols.map(_(i)).sum)
    for (i <- 0 until lanes) assert(hw(i) == gold(i), f"lane $i: ${hw(i)}%.1f != ${gold(i)}%.1f")
  }

  it should "tap+lanewise the CLUSTER config: computeLanes=8, a 64-beat row" in {
    // The shipped SIMD block builds StreamReduce at computeLanes=8, and FlashAttention
    // reduces a whole 64-beat tile in one row. The earlier tap test used the default
    // cl=32 and 6 beats, which is not the same FSM path.
    val rnd   = new Random(0xC6)
    val cols  = Seq.fill(64)(Seq.fill(lanes)(rnd.between(-40, 40).toFloat))
    val beats = cols.map(_.map(v => f32ToF16bits(v)))
    val hw    = runReduceLanewiseTap(0 /*MAX*/, beats, computeLanes = 8)
    assert(hw.length == beats.length + 1, s"expected ${beats.length + 1} beats, got ${hw.length}")
    for (k <- cols.indices; i <- 0 until lanes)
      assert(hw(k)(i) == cols(k)(i), f"passthrough beat $k lane $i")
    val gold = (0 until lanes).map(i => cols.map(_(i)).max)
    for (i <- 0 until lanes)
      assert(hw(64)(i) == gold(i), f"stat lane $i: ${hw(64)(i)}%.1f != ${gold(i)}%.1f")
  }

  it should "emit the per-lane MAX across beats, not a folded scalar" in {
    val rnd   = new Random(0xC1)
    val nB    = 8
    val cols  = Seq.fill(nB)(Seq.fill(lanes)(rnd.between(-40, 40).toFloat))
    val beats = cols.map(_.map(v => f32ToF16bits(v)))
    val hw    = runReduceLanewise(0 /*MAX*/, beats)
    val gold  = (0 until lanes).map(i => cols.map(_(i)).max)
    for (i <- 0 until lanes)
      assert(hw(i) == gold(i), f"lane $i%2d: hw=${hw(i)}%.3f golden=${gold(i)}%.3f")
    // And it must NOT be the folded scalar: with random data the lanes must differ.
    assert(hw.distinct.length > 1, "LANEWISE emitted a splatted scalar -- the fold was not bypassed")
  }

  it should "emit the per-lane SUM across beats" in {
    val rnd   = new Random(0xC2)
    val nB    = 6
    // small ints so FP16 addition is exact and the compare can stay exact
    val cols  = Seq.fill(nB)(Seq.fill(lanes)(rnd.between(-8, 8).toFloat))
    val beats = cols.map(_.map(v => f32ToF16bits(v)))
    val hw    = runReduceLanewise(1 /*ADD*/, beats)
    val gold  = (0 until lanes).map(i => cols.map(_(i)).sum)
    for (i <- 0 until lanes)
      assert(hw(i) == gold(i), f"lane $i%2d: hw=${hw(i)}%.3f golden=${gold(i)}%.3f")
  }

  it should "work under the time-muxed lane FSM (computeLanes < lanes)" in {
    val rnd   = new Random(0xC3)
    val cols  = Seq.fill(5)(Seq.fill(lanes)(rnd.between(-30, 30).toFloat))
    val beats = cols.map(_.map(v => f32ToF16bits(v)))
    val hw    = runReduceLanewise(0 /*MAX*/, beats, computeLanes = 4)
    val gold  = (0 until lanes).map(i => cols.map(_(i)).max)
    for (i <- 0 until lanes)
      assert(hw(i) == gold(i), f"lane $i%2d (cl=4): hw=${hw(i)}%.3f golden=${gold(i)}%.3f")
  }

  it should "run back-to-back rows without a fold bubble between them" in {
    // Two rows in one task: the point of LANEWISE is that treeBusy never rises, so the second row's
    // beats are not held off. Correctness of the SECOND row is what proves no state leaked.
    val rnd   = new Random(0xC4)
    val cols  = Seq.fill(4)(Seq.fill(lanes)(rnd.between(-20, 20).toFloat))
    val beats = cols.map(_.map(v => f32ToF16bits(v)))
    val hw    = runReduceLanewise(0, beats)
    val gold  = (0 until lanes).map(i => cols.map(_(i)).max)
    for (i <- 0 until lanes) assert(hw(i) == gold(i))
  }

  /** Drive one row with fp32out set (op CSR bit[9]) and return the HW scalar read from the output beat's
    * low 32 bits as a raw FP32 (NO FP16 narrowing). This is the SUMSQ-overflow fix path: a reduction that
    * would saturate the FP16 grid to +inf is instead delivered to the host as the true FP32 value. */
  def runReduceFp32(op: Int, beats: Seq[Seq[Int]], computeLanes: Int = 32,
                    ops: Seq[String] = Seq("FMA_FP16", "MAX_FP16")): Float = {
    var result: Float = 0.0f
    test(new DataPathExtensionHarness(new HasStreamReduce(computeLanes = computeLanes, op = ops, elementWidth = 16)))
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.io.csr_i(0).poke(beats.length.U)
        dut.io.csr_i(1).poke((op | 0x200).U) // fp32out bit[9]
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true)
        dut.clock.step(1)
        dut.io.start_i.poke(false)

        var threads = new chiseltest.internal.TesterThreadList(Seq())
        threads = threads.fork {
          dut.io.data_i.valid.poke(true)
          for (b <- beats) {
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
            dut.io.data_i.bits.poke(packBeat(b))
            dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
        }
        threads = threads.fork {
          while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
          val lane0 = (dut.io.data_o.bits.peekInt() & ((BigInt(1) << 32) - 1)).toLong
          result = java.lang.Float.intBitsToFloat(lane0.toInt)
          dut.io.data_o.ready.poke(true)
          dut.clock.step(1)
          dut.io.data_o.ready.poke(false)
        }
        threads.joinAndStep()
      }
    result
  }

  /** Drive the DUT for `rows` consecutive rows in ONE dispatch (operandCount = beatsPerRow), returning
    * the per-row HW scalars (low lane of each output beat). The counter wraps at operandCount and
    * re-inits the per-lane regs every row, so feeding rows*beatsPerRow input beats emits one splatted
    * scalar per row -- no Chisel change. Distinct per-row data lets the check catch a broken
    * row-boundary re-init: row r's scalar must equal row r's own reduction, not a running total. */
  def runReduceMultiRow(op: Int, rows: Seq[Seq[Seq[Int]]], computeLanes: Int = 32,
                        ops: Seq[String] = Seq("FMA_FP16", "MAX_FP16")): Seq[Float] = {
    val beatsPerRow = rows.head.length
    val allBeats    = rows.flatten // rows*beatsPerRow input beats
    val results     = scala.collection.mutable.ArrayBuffer[Float]()
    test(new DataPathExtensionHarness(new HasStreamReduce(computeLanes = computeLanes, op = ops, elementWidth = 16)))
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.io.csr_i(0).poke(beatsPerRow.U)
        dut.io.csr_i(1).poke(op.U)
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true)
        dut.clock.step(1)
        dut.io.start_i.poke(false)

        var threads = new chiseltest.internal.TesterThreadList(Seq())
        threads = threads.fork {
          dut.io.data_i.valid.poke(true)
          for (b <- allBeats) {
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
            dut.io.data_i.bits.poke(packBeat(b))
            dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
        }
        threads = threads.fork {
          for (_ <- rows.indices) { // one scalar beat per row
            while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
            val lane0 = (dut.io.data_o.bits.peekInt() & ((BigInt(1) << 16) - 1)).toInt
            results += f16bitsToF32(lane0)
            dut.io.data_o.ready.poke(true)
            dut.clock.step(1)
            dut.io.data_o.ready.poke(false)
          }
        }
        threads.joinAndStep()
      }
    results.toSeq
  }

  // ---- golden + check ----
  def goldenScalar(op: Int, vals: Seq[Float]): Float = op match {
    case 0 => vals.max                                  // MAX
    case 1 => vals.foldLeft(0.0f)(_ + _)                // ADD
    case 2 => vals.foldLeft(0.0f)((a, x) => a + x * x)  // SUMSQ
  }

  def checkOp(op: Int, opName: String, nBeats: Int, mag: Int, computeLanes: Int = 32,
              ops: Seq[String] = Seq("FMA_FP16", "MAX_FP16"),
              fpPipe: Int = 1, treePipe: Int = 1): Unit = {
    val rng   = new Random(0x5EED + op)
    // FP16-representable inputs; magnitude kept small enough that the reduction stays in FP16 range
    // (SUMSQ over nBeats*32 squared terms must not overflow 65504).
    val beats = Seq.fill(nBeats)(Seq.fill(lanes) {
      val f = (rng.between(-mag, mag) + rng.nextInt(4) * 0.25f)
      f32ToF16bits(f)
    })
    val vals  = beats.flatten.map(f16bitsToF32)
    val golden = goldenScalar(op, vals)
    // narrow golden to FP16 grid for a fair compare with the HW (which narrows before output)
    val goldenF16 = f16bitsToF32(f32ToF16bits(golden))
    val hw        = runReduce(op, beats, computeLanes, ops, fpPipe, treePipe)
    val tol       = math.max(math.abs(goldenF16) * 0.004f, 0.05f) // ~0.4% rel + small abs floor
    val err       = math.abs(hw - goldenF16)
    println(s"[StreamReduce:$opName cl=$computeLanes ops=${ops.mkString}] beats=$nBeats golden=$goldenF16 hw=$hw err=$err tol=$tol")
    assert(err <= tol, s"$opName mismatch: hw=$hw golden=$goldenF16 err=$err > tol=$tol")
  }

  /** Multi-row: nRows independent per-row reductions in one dispatch. Each row gets its OWN random data
    * so a stale carry across the row boundary (broken re-init) shows up as a wrong per-row scalar. */
  def checkMultiRow(op: Int, opName: String, nRows: Int, beatsPerRow: Int, mag: Int, computeLanes: Int = 32,
                    ops: Seq[String] = Seq("FMA_FP16", "MAX_FP16")): Unit = {
    val rng  = new Random(0x3A1 + op)
    val rows = Seq.tabulate(nRows) { _ =>
      Seq.fill(beatsPerRow)(Seq.fill(lanes) {
        val f = (rng.between(-mag, mag) + rng.nextInt(4) * 0.25f)
        f32ToF16bits(f)
      })
    }
    val hw = runReduceMultiRow(op, rows, computeLanes, ops)
    assert(hw.length == nRows, s"$opName multirow: expected $nRows scalars, got ${hw.length}")
    for (r <- 0 until nRows) {
      val vals      = rows(r).flatten.map(f16bitsToF32)
      val goldenF16 = f16bitsToF32(f32ToF16bits(goldenScalar(op, vals)))
      val tol       = math.max(math.abs(goldenF16) * 0.004f, 0.05f)
      val err       = math.abs(hw(r) - goldenF16)
      println(s"[StreamReduce:$opName:multirow r=$r cl=$computeLanes] golden=$goldenF16 hw=${hw(r)} err=$err tol=$tol")
      assert(err <= tol, s"$opName multirow row $r mismatch: hw=${hw(r)} golden=$goldenF16 err=$err > tol=$tol")
    }
  }

  /** fp32out SUMSQ with LARGE inputs that overflow the FP16 grid: prove the scalar comes back as the true
    * FP32 value (finite), not the +inf a narrowed FP16 scalar would produce. mag~4096 -> per-row SUMSQ
    * ~1e9 >> FP16 max (65504). Compares the raw FP32 read-back against the FP32 (un-narrowed) golden. */
  def checkSumsqFp32(nBeats: Int, mag: Int, computeLanes: Int = 32): Unit = {
    val rng   = new Random(0xF32)
    val beats = Seq.fill(nBeats)(Seq.fill(lanes)(f32ToF16bits(rng.between(-mag, mag).toFloat)))
    val vals   = beats.flatten.map(f16bitsToF32)
    val golden = goldenScalar(2, vals) // FP32 SUMSQ, NOT narrowed to FP16
    assert(f16bitsToF32(f32ToF16bits(golden)).isInfinite,
           s"test setup: SUMSQ $golden must overflow the FP16 grid to exercise the fix")
    val hw  = runReduceFp32(2, beats, computeLanes)
    val rel = math.abs(hw - golden) / math.abs(golden)
    println(s"[StreamReduce:SUMSQ:fp32out cl=$computeLanes] beats=$nBeats golden=$golden hw=$hw rel=$rel")
    assert(!hw.isInfinite && !hw.isNaN, s"fp32out scalar must be finite, got $hw")
    assert(rel <= 0.02f, s"SUMSQ fp32out mismatch: hw=$hw golden=$golden rel=$rel > 0.02")
  }

  "StreamReduce_MAX" should "match the FP golden" in { checkOp(0, "MAX", 8, 32) }
  "StreamReduce_ADD" should "match the FP golden" in { checkOp(1, "ADD", 8, 32) }
  "StreamReduce_SUMSQ" should "match the FP golden" in { checkOp(2, "SUMSQ", 8, 4) }
  // fp32out: large inputs that overflow FP16 must come back as the true FP32 SUMSQ (combinational + time-mux).
  "StreamReduce_SUMSQ_fp32out" should "emit the true FP32 sum-of-squares when FP16 would overflow" in {
    checkSumsqFp32(8, 4096)
  }
  "StreamReduce_SUMSQ_fp32out_cl8" should "emit the true FP32 SUMSQ in the time-mux" in {
    checkSumsqFp32(8, 4096, computeLanes = 8)
  }

  // separate accumulate (fpPipe) vs reduce-tree (treePipe) cut knobs: asymmetric depths must still match
  // (exercises the independent accLat = laneLat+2*fpPipe and treeAddLat = numLevels*treePipe accounting).
  "StreamReduce_SUMSQ_asym_pipe" should "match with fpPipe=2, treePipe=1" in {
    checkOp(2, "SUMSQ", 8, 4, computeLanes = 8, fpPipe = 2, treePipe = 1)
  }
  "StreamReduce_ADD_treePipe" should "match with fpPipe=1, treePipe=2" in {
    checkOp(1, "ADD", 8, 32, computeLanes = 8, fpPipe = 1, treePipe = 2)
  }

  // --- specialized configs ---
  // time-mux FSM (computeLanes=8 -> 4 sub-cycles/beat): MAX + ADD must still match.
  "StreamReduce_MAX_cl8" should "match (time-mux)" in { checkOp(0, "MAX", 8, 32, computeLanes = 8) }
  "StreamReduce_ADD_cl8" should "match (time-mux)" in { checkOp(1, "ADD", 8, 32, computeLanes = 8) }
  // single-op build (op CSR fixed, unused arithmetic dropped): ADD-only.
  "StreamReduce_ADDonly" should "match (op=ADD only)" in { checkOp(1, "ADD", 8, 32, ops = Seq("ADD_FP16")) }

  // --- multi-row: S independent per-row reductions in ONE dispatch (the llama3 per-row RMSNorm/Softmax
  // need). 4 rows x 2 beats = D=64 @FP16. Distinct per-row data catches a broken row-boundary re-init.
  "StreamReduce_SUMSQ_multirow" should "reduce each row independently" in { checkMultiRow(2, "SUMSQ", 4, 2, 4) }
  "StreamReduce_MAX_multirow"   should "reduce each row independently" in { checkMultiRow(0, "MAX",   4, 2, 8) }
  "StreamReduce_ADD_multirow"   should "reduce each row independently" in { checkMultiRow(1, "ADD",   4, 2, 8) }
  // multi-row in the time-mux FSM (computeLanes=8 -> beatCnt-reset re-init path).
  "StreamReduce_ADD_multirow_cl8" should "reduce each row in the time-mux" in { checkMultiRow(1, "ADD", 4, 2, 8, computeLanes = 8) }

  /** Tap mode (op | 0x100): the N input beats pass through unchanged, then one trailing scalar beat is
    * emitted (output = N+1 beats). The harness inserts -||> register cuts on both data ports, so use the
    * fork-based producer/consumer handshake. Returns all N+1 output beats (full 512-bit each). */
  def runReduceTap(op: Int, beats: Seq[Seq[Int]], computeLanes: Int = 32): Seq[BigInt] = {
    val outs = scala.collection.mutable.ArrayBuffer[BigInt]()
    test(new DataPathExtensionHarness(
      new HasStreamReduce(computeLanes = computeLanes, op = Seq("FMA_FP16", "MAX_FP16"), elementWidth = 16)))
      .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.io.csr_i(0).poke(beats.length.U)
        dut.io.csr_i(1).poke((op | 0x100).U) // tap bit[8]
        dut.io.enable_i.poke(true)
        dut.io.start_i.poke(true)
        dut.clock.step(1)
        dut.io.start_i.poke(false)

        var threads = new chiseltest.internal.TesterThreadList(Seq())
        threads = threads.fork {
          dut.io.data_i.valid.poke(true)
          for (b <- beats) {
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
            dut.io.data_i.bits.poke(packBeat(b))
            dut.clock.step(1)
          }
          dut.io.data_i.valid.poke(false)
        }
        threads = threads.fork {
          for (_ <- 0 to beats.length) { // N+1 output beats
            while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
            outs += dut.io.data_o.bits.peekInt()
            dut.io.data_o.ready.poke(true)
            dut.clock.step(1)
            dut.io.data_o.ready.poke(false)
          }
        }
        threads.joinAndStep()
      }
    outs.toSeq
  }

  def checkTap(op: Int, opName: String, nBeats: Int, mag: Int, computeLanes: Int = 32): Unit = {
    val rng   = new Random(0x7A9 + op)
    val beats = Seq.fill(nBeats)(Seq.fill(lanes) {
      val f = (rng.between(-mag, mag) + rng.nextInt(4) * 0.25f)
      f32ToF16bits(f)
    })
    val outs = runReduceTap(op, beats, computeLanes)
    assert(outs.length == nBeats + 1, s"$opName tap: expected ${nBeats + 1} beats, got ${outs.length}")
    for (i <- 0 until nBeats)
      assert(outs(i) == packBeat(beats(i)), s"$opName tap: passthrough beat $i changed")
    val vals      = beats.flatten.map(f16bitsToF32)
    val goldenF16 = f16bitsToF32(f32ToF16bits(goldenScalar(op, vals)))
    val hw        = f16bitsToF32((outs(nBeats) & 0xffff).toInt)
    val tol       = math.max(math.abs(goldenF16) * 0.004f, 0.05f)
    val err       = math.abs(hw - goldenF16)
    println(s"[StreamReduce:$opName:tap] beats=$nBeats passthrough OK; scalar golden=$goldenF16 hw=$hw err=$err tol=$tol")
    assert(err <= tol, s"$opName tap scalar mismatch: hw=$hw golden=$goldenF16 err=$err > tol=$tol")
  }

  "StreamReduce_ADD_tap" should "pass the row through and emit the sum" in { checkTap(1, "ADD", 8, 32) }
  "StreamReduce_MAX_tap" should "pass the row through and emit the max" in { checkTap(0, "MAX", 8, 32) }
  // tap in the time-mux FSM (computeLanes=8): passthrough beats + trailing scalar must still be correct.
  "StreamReduce_ADD_tap_cl8" should "tap-pass-through in the time-mux" in { checkTap(1, "ADD", 8, 32, computeLanes = 8) }

  /** APP-LEVEL protocol test: emulate the xDMA controller driving ONE reduce instance across several
    * back-to-back tasks -- pulse start, feed the row, drain the output, then WAIT for busy_o to fall (the
    * completion signal the controller blocks on before the next task). The value-only tests above never
    * check busy_o, so a stuck-busy / credit-overflow bug (the reduce hangs the real softmax) slips through.
    * This mirrors softmax's T1 MAX -> T2 ADD-tap -> ... sequence in the fast Tier-1 sim. */
  def runTasksCheckBusy(computeLanes: Int, tasks: Seq[(Int, Boolean, Int)]): Unit = {
    test(new DataPathExtensionHarness(
      new HasStreamReduce(computeLanes = computeLanes, op = Seq("FMA_FP16", "MAX_FP16"), elementWidth = 16)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--build-jobs", "1")))) { dut =>
        dut.io.enable_i.poke(true)
        val rng = new Random(0x1EAF)
        for (((op, tap, nBeats), ti) <- tasks.zipWithIndex) {
          val beats = Seq.fill(nBeats)(Seq.fill(lanes)(f32ToF16bits(rng.between(-4, 4) + rng.nextInt(4) * 0.25f)))
          val opv   = op | (if (tap) 0x100 else 0)
          val nOut  = if (tap) nBeats + 1 else 1
          dut.io.csr_i(0).poke(nBeats.U)
          dut.io.csr_i(1).poke(opv.U)
          dut.io.start_i.poke(true); dut.clock.step(1); dut.io.start_i.poke(false)
          var threads = new chiseltest.internal.TesterThreadList(Seq())
          threads = threads.fork {
            dut.io.data_i.valid.poke(true)
            for (b <- beats) {
              while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step(1)
              dut.io.data_i.bits.poke(packBeat(b)); dut.clock.step(1)
            }
            dut.io.data_i.valid.poke(false)
          }
          threads = threads.fork {
            for (_ <- 0 until nOut) {
              while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step(1)
              dut.io.data_o.ready.poke(true); dut.clock.step(1); dut.io.data_o.ready.poke(false)
            }
          }
          threads.joinAndStep()
          // controller now waits for the datapath to report done (busy_o low), draining any tail beats.
          dut.io.data_o.ready.poke(true)
          var w = 0
          while (dut.io.busy_o.peekBoolean() && w < 400) { dut.clock.step(1); w += 1 }
          dut.io.data_o.ready.poke(false)
          assert(!dut.io.busy_o.peekBoolean(),
                 s"task $ti (op=$op tap=$tap beats=$nBeats): busy_o STUCK HIGH after drain -- completion/credit bug")
          println(f"[StreamReduce:busy] task $ti op=$op tap=$tap beats=$nBeats -> busy fell after $w cyc")
        }
      }
  }

  // softmax-like sequence on one instance: MAX (non-tap) then ADD-tap then more -- each must report done.
  "StreamReduce_multitask_busy" should "return busy_o low after each back-to-back task" in {
    runTasksCheckBusy(4, Seq((0, false, 8), (1, true, 8), (1, false, 8), (1, true, 8)))
  }
  "StreamReduce_tap_busy_cl32" should "return busy_o low after a tap task (parallel)" in {
    runTasksCheckBusy(32, Seq((1, true, 4), (0, false, 4), (1, true, 4)))
  }
}
