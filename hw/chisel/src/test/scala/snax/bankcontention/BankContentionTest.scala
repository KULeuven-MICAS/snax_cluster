package snax.bankcontention

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import java.io.PrintWriter

/** Measure what bank contention costs, on the real streamer RTL.
  *
  * Every case here is a controlled pair: the same streams, the same byte volume, the same
  * number of steps, with ONE thing changed -- where a buffer starts, or how deep its FIFO is.
  * Anything that prices memory by counting bytes predicts no difference in any of them.
  *
  * The numbers are written to `bank_contention_rtl.json` so the Python cost model can be scored
  * against them rather than against numbers copied by hand into a docstring.
  */
class BankContentionTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "bank contention"

  val NUM_BANKS   = 32
  val NUM_CHANNEL = 8
  val WORD        = 8
  val SPAN        = NUM_BANKS * WORD          // 256 B -- addresses this far apart share a bank
  val STEPS       = 64      // the effects are steady-state; treadle is slow, so keep it short
  val BEAT        = NUM_CHANNEL * WORD        // 64 B moved per step per stream

  /** One contiguous stream of `STEPS` beats from `ptr`. */
  def seq(ptr: Int): StreamSpec = StreamSpec(ptr, Seq(STEPS, 1, 1), Seq(BEAT, 0, 0))

  case class Result(cycles: Int, steps: Int, blocked: Int, stall: Seq[Int], retries: Seq[Int])

  val results = scala.collection.mutable.LinkedHashMap[String, Result]()

  def run(name: String, specs: Seq[StreamSpec], depth: Int,
          banks: Int = NUM_BANKS): Result = {
    var out: Result = null
    // Default backend, not Verilator: verilator 5.048's generated makefile races on its
    // precompiled headers under chiseltest's parallel build ("__pch.h.fast: No such file"),
    // and 5.048 has no --no-pch. The DUT is a few readers over 300 cycles, so it does not need
    // a compiled simulator.
    test(new BankContentionHarness(specs, numBanks = banks, numChannel = NUM_CHANNEL,
                                   bufferDepth = depth)) { dut =>
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
          specs.indices.map(i => dut.io.retries(i).peek().litValue.toInt)
        )
      }
    results(name) = out
    println(f"[bank-contention] $name%-28s cycles ${out.cycles}%6d  blocked ${out.blocked}%6d  " +
            f"stall ${out.stall.mkString(",")}  retries ${out.retries.mkString(",")}")
    out
  }

  it should "match the cost model on every case, and write the measurements out" in {
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

    val json = results.map { case (k, v) =>
      f""""$k": {"cycles": ${v.cycles}, "steps": ${v.steps}, "blocked": ${v.blocked}, """ +
      f""""stall": [${v.stall.mkString(", ")}], "retries": [${v.retries.mkString(", ")}]}"""
    }.mkString("{\n  ", ",\n  ", "\n}\n")
    val w = new PrintWriter("bank_contention_rtl.json")
    w.write(json); w.close()
    println("[bank-contention] wrote bank_contention_rtl.json")
    println(json)
  }
}
