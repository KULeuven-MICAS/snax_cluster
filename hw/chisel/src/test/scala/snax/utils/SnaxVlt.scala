package snax.utils

import chiseltest.simulator.VerilatorFlags

/** Verilator's --build-jobs for the tests: SNAX_VLT_BUILD_JOBS, 1 by default. */
object SnaxVlt {
  val buildJobs: Int = sys.env.get("SNAX_VLT_BUILD_JOBS").filter(_.nonEmpty).map(_.toInt).getOrElse(1)

  def buildJobsFlags: VerilatorFlags = VerilatorFlags(Seq("--build-jobs", buildJobs.toString))
}
