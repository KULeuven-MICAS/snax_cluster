package snax.utils

import chiseltest._
import firrtl2.annotations.Annotation

/** VCD dumping for chiseltest, off by default. Run with SNAX_VCD=1 to write a VCD per test into test_run_dir. */
object SnaxVcd {
  def annotations: Seq[Annotation] = if (sys.env.get("SNAX_VCD").contains("1")) Seq(WriteVcdAnnotation) else Seq()
}
