package snax.DataPathExtension

/** What a generator needs to know about an extension without knowing which extension it is.
  *
  * `capabilities` is the set of OPTIONAL things this instance elaborated -- the op or func names it was
  * configured with, plus any build-time feature that changes what software may ask of it. It exists because
  * the generated headers name each EXTENSION and, until this field, said nothing about what that extension
  * can actually do.
  *
  * THAT GAP IS NOT COSMETIC. `SIMD_EXT_STREAMMAP` answers "is there a StreamMap?" and cannot answer "does it
  * have RSQRT?" -- so a kernel writing `func = 3` into the CSR on a cluster whose cfg lists only
  * LINEAR/EXP/SILU is accepted, selects no activation, and returns the LINEAR result. A well-formed tensor of
  * wrong numbers, with nothing reported anywhere. The same hole applies to StreamReduce's MAX/ADD/SUMSQ/FMA
  * and to StreamElementwise's MUL/ADD/FMA, and it reopens for every op added later.
  *
  * An extension populates this with the BARE op names -- no precision suffix, because the transport precision
  * is one per extension and `OpSpec.checkWidth` already ties it to the cfg's `elementWidth`. The generators
  * turn each into a `..._HAS_<CAP>` define beside the extension's own macro, so software gates on what the
  * hardware has rather than on what its cfg was assumed to say.
  */
class DataPathExtensionParam(
  val moduleName:   String,
  val userCsrNum:   Int,
  val dataWidth:    Int         = 512,
  val capabilities: Seq[String] = Nil
) {
  require(dataWidth > 0)
  require(userCsrNum >= 0)
  require(
    capabilities.forall(c => c.nonEmpty && c.forall(ch => ch.isLetterOrDigit || ch == '_')),
    s"$moduleName: a capability becomes part of a C macro name, so it must be non-empty and " +
      s"[A-Za-z0-9_] only; got ${capabilities.mkString(", ")}"
  )
}
