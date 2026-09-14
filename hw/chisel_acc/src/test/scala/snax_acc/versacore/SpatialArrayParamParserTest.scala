package snax_acc.versacore

import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec

class SpatialArrayParamParserTest extends AnyFlatSpec {
  behavior of "SpatialArrayParamParser"

  private val groupSizeKey = "snax_versacore_p2s_chunks_per_group"

  // Eight integer MACs with a 2x2x2 shape produce four 32-bit output chunks.
  // A nondefault group size of two therefore exercises multiple shift groups.
  private def config(): ujson.Obj =
    ujson.Obj(
      "snax_num_rw_csr"                       -> 7,
      "snax_versacore_mac_num"                -> ujson.Arr(8),
      "snax_versacore_input_a_element_width"  -> ujson.Arr(8),
      "snax_versacore_input_b_element_width"  -> ujson.Arr(8),
      "snax_versacore_input_c_element_width"  -> ujson.Arr(32),
      "snax_versacore_output_d_element_width" -> ujson.Arr(32),
      "snax_versacore_input_a_data_type"      -> ujson.Arr("SInt"),
      "snax_versacore_input_b_data_type"      -> ujson.Arr("SInt"),
      "snax_versacore_input_c_data_type"      -> ujson.Arr("SInt"),
      "snax_versacore_output_d_data_type"     -> ujson.Arr("SInt"),
      "snax_versacore_array_input_a_width"    -> 32,
      "snax_versacore_array_input_b_width"    -> 32,
      "snax_versacore_array_input_c_width"    -> 128,
      "snax_versacore_array_output_d_width"   -> 128,
      "snax_versacore_spatial_unrolling"      -> ujson.Arr(ujson.Arr(ujson.Arr(2, 2, 2))),
      "snax_versacore_serial_a_width"         -> 32,
      "snax_versacore_serial_b_width"         -> 32,
      "snax_versacore_serial_c_d_width"       -> 32,
      "snax_versacore_adder_tree_delay"       -> 0,
      "snax_versacore_temporal_unrolling"     -> ujson.Arr("output_stationary")
    )

  private def parse(cfg: ujson.Obj): SpatialArrayParam = SpatialArrayParamParser.parseFromHjsonString(ujson.write(cfg))

  it should "default to four chunks per group when the optional setting is absent" in {
    val params = parse(config())
    assert(params.p2sChunksPerGroup == 4)
  }

  for (groupSize <- Seq(2, 8)) {
    it should s"retain an explicit group size of $groupSize" in {
      val cfg = config()
      cfg(groupSizeKey) = ujson.Num(groupSize)
      assert(parse(cfg).p2sChunksPerGroup == groupSize)
    }
  }

  for (groupSize <- Seq(0.0, 1.0, 3.0, 4.5)) {
    it should s"reject an invalid group size of $groupSize" in {
      val cfg = config()
      cfg(groupSizeKey) = ujson.Num(groupSize)
      intercept[IllegalArgumentException] {
        parse(cfg)
      }
    }
  }

  it should "forward the configured group size into the VersaCore output serializer" in {
    val cfg = config()
    cfg(groupSizeKey) = ujson.Num(2)
    val params = parse(cfg)

    var instantiatedGroupSize: Option[Int] = None
    ChiselStage.emitCHIRRTL {
      val dut = new VersaCore(params)
      instantiatedGroupSize = Some(dut.D_p2s.p.p2sChunksPerGroup)
      dut
    }

    assert(instantiatedGroupSize.contains(2))
  }
}
