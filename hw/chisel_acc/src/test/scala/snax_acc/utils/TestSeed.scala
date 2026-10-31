package snax_acc.utils

import scala.util.Random

/** Seed for the randomized tests. It is fixed by default so that every run is the same; set SNAX_TEST_SEED to try
  * another seed or to re-run a failure with the seed it printed.
  */
object TestSeed {
  val default: Long = 42L
  val seed:    Long = sys.env.get("SNAX_TEST_SEED").map(_.toLong).getOrElse(default)

  /** A new Random seeded with `seed`. Prints the seed under the test's name. */
  def rng(testName: String): Random = {
    println(s"[$testName] SNAX_TEST_SEED=$seed")
    new Random(seed)
  }
}
