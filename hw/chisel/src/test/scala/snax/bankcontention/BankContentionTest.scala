package snax.bankcontention

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** Measure what bank contention costs, on the real streamer RTL.
  *
  * Every case here is a controlled pair: the same streams, the same byte volume, the same
  * number of steps, with ONE thing changed -- where a buffer starts, or how deep its FIFO is.
  * Anything that prices memory by counting bytes predicts no difference in any of them.
  *
  * The measurements are PRINTED, not written to a file. They are a handful of integers whose
  * only job is to reach the cost model's own test, and a generated file sitting in the RTL tree
  * is a thing to accidentally commit rather than a result. Read them off the run.
  */
class BankContentionTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "bank contention"

  val NUM_BANKS   = 32
  val NUM_CHANNEL = 8
  val WORD        = 8
  val SPAN        = NUM_BANKS * WORD          // 256 B -- addresses this far apart share a bank
  val STEPS       = 512     // long enough that STEADY STATE dominates the start-up fill.
                            // At 64 the fill was a third of the run and the depth effect
                            // it was meant to expose was buried in it. Verilator made
                            // this affordable; under the interpreter it was not.
  val BEAT        = NUM_CHANNEL * WORD        // 64 B moved per step per stream

  /** One contiguous stream of `STEPS` beats from `ptr`. */
  def seq(ptr: Int): StreamSpec = StreamSpec(ptr, Seq(STEPS, 1, 1), Seq(BEAT, 0, 0))

  case class Result(cycles: Int, steps: Int, blocked: Int, stall: Seq[Int], retries: Seq[Int],
                    urgent: Seq[Int])

  val results = scala.collection.mutable.LinkedHashMap[String, Result]()

  def run(name: String, specs: Seq[StreamSpec], depth: Int,
          banks: Int = NUM_BANKS): Result = {
    var out: Result = null
    // Verilator, because the interpreter is ~100x slower per node and the five-stream cases
    // need it. The PCH race that forced the default backend earlier is a PARALLEL MAKE problem
    // -- chiseltest drives verilator's makefile itself, where snax_cluster lets
    // `verilator --build` do it -- so `MAKEFLAGS=-j1` in experiments/run_chisel_tests.sh is
    // what makes this work.
    test(new BankContentionHarness(specs, numBanks = banks, numChannel = NUM_CHANNEL,
                                   bufferDepth = depth))
      .withAnnotations(Seq(VerilatorBackendAnnotation)) { dut =>
        dut.io.start.poke(false.B)
        dut.clock.setTimeout(0)   // the guard loop below bounds the run, not chiseltest
        dut.clock.step(4)
        dut.io.start.poke(true.B)
        dut.clock.step(1)
        dut.io.start.poke(false.B)
        val limit = 20 * STEPS + 500     // 20x the worst sane answer
        var guard = 0
        while (!dut.io.done.peek().litToBoolean && guard < limit) {
          dut.clock.step(1); guard += 1
        }
        assert(dut.io.done.peek().litToBoolean,
               s"$name made no progress: ${dut.io.steps.peek().litValue} of $STEPS steps in " +
               s"${dut.io.cycles.peek().litValue} cycles (limit $limit)")
        out = Result(
          dut.io.cycles.peek().litValue.toInt,
          dut.io.steps.peek().litValue.toInt,
          dut.io.blocked.peek().litValue.toInt,
          specs.indices.map(i => dut.io.stall(i).peek().litValue.toInt),
          specs.indices.map(i => dut.io.retries(i).peek().litValue.toInt),
          specs.indices.map(i => dut.io.urgent(i).peek().litValue.toInt)
        )
      }
    results(name) = out
    println(f"[bank-contention] $name%-34s cycles ${out.cycles}%6d  blocked ${out.blocked}%6d  " +
            f"stall ${out.stall.mkString(",")}  retries ${out.retries.mkString(",")}  " +
            f"urgent ${out.urgent.mkString(",")}")
    out
  }

  it should "measure what bank contention costs, across controlled pairs" in {
    // ONE test for all of it: each `it should` gets its own suite instance, so results
    // accumulated across tests are lost by the time the last one writes them.
    val single = run("single", Seq(seq(0)), depth = 8)
    assert(single.cycles <= STEPS + 8, s"a lone conflict-free stream took ${single.cycles}")

    // bases one whole bank array apart: identical banks, in the same order, every cycle
    val d1 = run("congruent, depth 1", Seq(seq(0), seq(SPAN)), depth = 1)
    val depths = Seq(2, 4, 8).map(d => d -> run(s"congruent, depth $d", Seq(seq(0), seq(SPAN)), d))
    val deep = depths.toMap.apply(4)
    assert(d1.cycles > 2 * deep.cycles,
           s"depth 1 (${d1.cycles}) should cost far more than depth 4 (${deep.cycles})")

    // half a beat past the array: the two streams no longer want the same banks
    run("offset, depth 8", Seq(seq(0), seq(SPAN + BEAT / 2)), depth = 8)

    // 4 streams x 8 words = 32 words a cycle against 16 banks: a factor of two, undodgeable
    val four = Seq(0, SPAN, 2 * SPAN, 3 * SPAN).map(seq)
    for (d <- Seq(1, 8)) {
      val r = run(s"oversubscribed x4 on 16 banks, depth $d", four, d, banks = 16)
      assert(r.cycles >= 2 * STEPS,
             s"16 banks served 32 words/cycle in ${r.cycles} cycles -- impossible")
    }

    // ---- THE WRITER CASES -------------------------------------------------------------
    // A depth-1 writer asserts the TCDM priority bit permanently (`count >= depth-1` with
    // depth 1 is `count >= 0`). What that costs a reader sharing its banks is the number every
    // FIFO-budget decision turns on, and no reader-only case can measure it.
    //
    // The reader sits at bank 0 and the writer one whole bank array away, so they want the same
    // banks in the same order -- the worst case, and the one a model is most likely to
    // over-charge. The writer moves a QUARTER of the reader's beats, as a drain port does.
    val slowW = StreamSpec(SPAN, Seq(STEPS, 1, 1), Seq(BEAT * 4, 0, 0), isWriter = true)
    for (d <- Seq(1, 2, 4, 8)) {
      run(s"reader(8) + writer(depth $d)", Seq(seq(0), slowW.copy(fifoDepth = Some(d))), 8)
    }
    // ...and the reader's own depth, with the writer pinned at the shipped 1.
    for (d <- Seq(2, 4, 8)) {
      run(s"reader($d) + writer(depth 1)",
          Seq(seq(0).copy(fifoDepth = Some(d)), slowW.copy(fifoDepth = Some(1))), d)
    }
    // the reader alone at each depth, so the writer's cost is a DIFFERENCE, not a level
    for (d <- Seq(2, 4, 8)) {
      run(s"reader($d) alone", Seq(seq(0).copy(fifoDepth = Some(d))), d)
    }

    // A summary block in the shape the cost model's test keeps its copy in, so transcribing a
    // re-measurement is a paste rather than a transcription.
    println("[bank-contention] ===== MEASURED (paste into test_arbiter_vs_rtl.py) =====")
    results.foreach { case (k, v) =>
      println(s"""[bank-contention]     "$k": ${v.cycles},""")
    }
    println("[bank-contention] ===== end =====")
  }

  // The five-stream regime, one test per depth. Each `it should` gets its own test_run_dir, and
  // with the Verilator backend that matters: chiseltest deletes and rebuilds `verilated/` for
  // every `test(...)` in a block, and throws DirectoryNotEmptyException when two cases in one
  // block race over it.
  //
  // TWO streams agree with the model to a few cycles at every depth, and BOTH say a reader gains
  // nothing above depth 4. At cluster scale the machine disagrees sharply -- dropping the operand
  // readers 8 -> 4 costs FlashAttention 3.74% -- so the sensitivity appears somewhere between two
  // streams and five. This is five, with FA's shape: two fast congruent operand readers, a slow
  // reader and a slow writer sharing the drain's banks, and a neighbour.
  //
  // The SLOW streams must be slow in RATE, not merely in stride. Giving them the same step
  // count with a wider stride leaves them at one step per cycle, which oversubscribes the banks:
  // the engine then idles 70% of the time and no stream's FIFO depth matters, which is the
  // opposite of FlashAttention, whose array is busy 91% of the time. `rateShift = 3` makes a
  // stream move once every eight engine steps, which is the C/D pair's real duty.
  //
  // THE BANKS MUST BE LAID OUT LIKE THE KERNEL'S. FlashAttention's L1 map puts its A operand on
  // bank 0, its B operand on bank 16, the C/D pair on bank 8 and the iDMA's target back on
  // bank 16 -- so A and B do NOT fight each other, and the pressure on each comes from the slow
  // streams. Making both fast readers congruent instead (the obvious synthetic choice) has them
  // collide head-on, which is a different machine and gives the depth sweep the opposite sign.
  // `+ k * SPAN` keeps a stream's bank while moving it out of the others' address range.
  def fiveStreams(d: Int): Seq[StreamSpec] = {
    val slow = (p: Int) => StreamSpec(p, Seq(STEPS / 8, 1, 1), Seq(BEAT, 0, 0), rateShift = 3)
    Seq(seq(0).copy(fifoDepth = Some(d)),                        // A: bank 0
        seq(4 * SPAN + 2 * BEAT).copy(fifoDepth = Some(d)),      // B: bank 16
        slow(8 * SPAN + BEAT).copy(fifoDepth = Some(d)),         // C: bank 8, 1/8 rate
        slow(8 * SPAN + BEAT).copy(isWriter = true, fifoDepth = Some(1)),  // D: bank 8, depth 1
        slow(12 * SPAN + 2 * BEAT).copy(isWriter = true, fifoDepth = Some(8))) // neighbour: bank 16
  }

  for (d <- Seq(2, 4, 8)) {
    it should s"measure five streams with the fast readers at depth $d" in {
      run(s"five streams, fast readers at depth $d", fiveStreams(d), depth = d)
    }
  }

  // ---------------------------------------------------------------------------------------
  // FlashAttention's QK dispatch, with the streamer CSRs the kernel actually writes.
  //
  // Four synthetic five-stream configurations failed to reproduce the reader-depth sensitivity
  // the cluster measures (A/B 8 -> 4 costs FA 3.74%): congruent readers went flat above depth 4,
  // long runs went non-monotonic, and FA's bank layout with linear streams went flat entirely.
  // What none of them had was FA's ACCESS PATTERN. Its A operand walks bounds (32,2,32) with
  // strides (64,0,2048) -- the ZERO on the middle digit means it re-reads the same 32 addresses
  // before moving on, so its working set is revisited rather than streamed. B has the zero on
  // the outer digit instead. That is the shape a real matmul's operand reuse gives the banks,
  // and it is not a property any linear stream has.
  //
  // Bases are the kernel's, reduced to their banks: A on 0, B on 16, the C/D pair on 8.
  def faQK(d: Int): Seq[StreamSpec] = Seq(
    StreamSpec(0,                   Seq(32, 2, 32), Seq(64, 0, 2048), fifoDepth = Some(d)),
    StreamSpec(64 * SPAN + 2 * BEAT, Seq(32, 2, 32), Seq(64, 2048, 0), fifoDepth = Some(d)),
    StreamSpec(96 * SPAN + BEAT,    Seq(4, 2, 32), Seq(128, 512, 1024), fifoDepth = Some(d),
               rateShift = 3),
    StreamSpec(96 * SPAN + BEAT,    Seq(4, 2, 32), Seq(128, 512, 1024), isWriter = true,
               fifoDepth = Some(1), rateShift = 3),
    StreamSpec(128 * SPAN + 2 * BEAT, Seq(256, 1, 1), Seq(64, 0, 0), isWriter = true,
               fifoDepth = Some(8), rateShift = 3))

  for (d <- Seq(2, 4, 8)) {
    it should s"measure FA's QK dispatch with the operand readers at depth $d" in {
      run(s"FA QK dispatch, operand readers at depth $d", faQK(d), depth = d)
    }
  }

  // FlashAttention's PV dispatch: O^T = V^T.P^T, m=8 n=2 k=128, the same 2048 passes as QK but
  // a far deeper accumulation -- and the half that stalls twice as hard on the real cluster
  // (259 cycles against QK's 138). Its three streams are ALL on bank 16 in the kernel's map:
  // A is v_buf, B is p8_0, and C/D is oacc32. That is the congruent case, in the machine.
  def faPV(d: Int): Seq[StreamSpec] = Seq(
    StreamSpec(2 * BEAT,                Seq(128, 2, 8), Seq(64, 0, 8192), fifoDepth = Some(d)),
    StreamSpec(64 * SPAN + 2 * BEAT,    Seq(128, 2, 8), Seq(64, 8192, 0), fifoDepth = Some(d)),
    StreamSpec(96 * SPAN + 2 * BEAT,    Seq(8, 2, 8), Seq(128, 1024, 2048), fifoDepth = Some(d),
               rateShift = 4),
    StreamSpec(96 * SPAN + 2 * BEAT,    Seq(8, 2, 8), Seq(128, 1024, 2048), isWriter = true,
               fifoDepth = Some(1), rateShift = 4),
    StreamSpec(128 * SPAN + 2 * BEAT,   Seq(128, 1, 1), Seq(64, 0, 0), isWriter = true,
               fifoDepth = Some(8), rateShift = 4))

  for (d <- Seq(2, 4, 8)) {
    it should s"measure FA's PV dispatch with the operand readers at depth $d" in {
      run(s"FA PV dispatch, operand readers at depth $d", faPV(d), depth = d)
    }
  }
}
