package snax.utils

import chisel3.Module
import chiseltest.ChiselScalatestTester
import chiseltest.simulator.VerilatorBackendAnnotation
import org.scalatest.TestSuite

/** Mix this in instead of `ChiselScalatestTester` to run a suite on VERILATOR rather than Treadle.
  *
  * chiseltest's default backend is Treadle, a FIRRTL interpreter: every `poke`/`peek` re-evaluates the
  * design's combinational cone in interpreted Scala. That is fine for a handful of vectors and terrible
  * for the thing these FP suites actually do -- sweep thousands of inputs through a datapath built from
  * native-Chisel FP units, where one "cycle" is several hundred gates of shift/normalise/round. Verilator
  * compiles the design to C++ once and then each vector costs essentially nothing, so the cost stops
  * scaling with the sweep and starts being a fixed compile.
  *
  * It is a mixin rather than a `.withAnnotations(...)` at every call site because the choice is a property
  * of the SUITE, not of one test, and because a suite with several `test(...)` blocks would otherwise have
  * to repeat it and could silently disagree with itself.
  *
  *     class FooTester extends AnyFlatSpec with VerilatorTester { ... }   // one word, whole suite
  *
  * Requires `verilator` on PATH -- run sbt through pixi (`pixi run sbt ...`), which is where this repo's
  * verilator lives. A suite that does NOT mix this in keeps the Treadle default, so adopting it elsewhere
  * is opt-in and one suite at a time.
  */
trait VerilatorTester extends ChiselScalatestTester { this: TestSuite =>
  // `TestBuilder` is an INNER class of the trait (it carries an $outer), so the return type is the
  // path-dependent `this.TestBuilder[T]` -- not a companion-object type.
  override def test[T <: Module](dutGen: => T): this.TestBuilder[T] =
    super.test(dutGen).withAnnotations(Seq(VerilatorBackendAnnotation))
}
