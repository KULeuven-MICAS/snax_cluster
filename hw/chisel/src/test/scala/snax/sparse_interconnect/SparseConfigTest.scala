package snax.sparse_interconnect

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SparseConfigTest extends AnyFlatSpec with Matchers {

  "SparsePortDefinition" should "calculate inputsPerBank correctly" in {
    val port = new SparsePortDefinition(width = 8, access_granularity = 2)
    port.inputsPerBank shouldEqual 4
  }

  "SparseConfig" should "calculate inputsPerBank correctly for multiple ports" in {
    val ports  = Seq(
      new SparsePortDefinition(width = 8, access_granularity = 2),
      new SparsePortDefinition(width = 16, access_granularity = 4)
    )
    val config = new SparseConfig(ports)
    config.inputsPerBank shouldEqual 4 + 4 // 8/2 + 16/4
  }

  it should "return the correct index list for a given bank" in {
    val ports   = Seq(
      new SparsePortDefinition(width = 8, access_granularity = 2),
      new SparsePortDefinition(width = 16, access_granularity = 4)
    )
    val config  = new SparseConfig(ports)
    val bank    = 1
    val idxList = config.get_global_idx_list(bank)
    idxList shouldEqual List(1, 3, 5, 7, 9, 13, 17, 21) // Calculated manually
  }

  it should "return the correct global index for a given sparse index and bank" in {
    val ports     = Seq(
      new SparsePortDefinition(width = 8, access_granularity = 2),
      new SparsePortDefinition(width = 16, access_granularity = 4)
    )
    val config    = new SparseConfig(ports)
    val sparseIdx = 2
    val bank      = 1
    val globalIdx = config.get_global_idx(sparseIdx, bank)
    globalIdx shouldEqual 5 // Calculated manually
  }

  it should "create a default SparseConfig with the correct number of inputs" in {
    val defaultConfig = new SparseConfig(Seq(new SparsePortDefinition(4, 1)))
    defaultConfig.inputsPerBank shouldEqual 4
  }

  "SparseConfig" should "correctly determine the port and index for a given input index" in {
    val ports  = Seq(
      new SparsePortDefinition(width = 8, access_granularity = 2),
      new SparsePortDefinition(width = 16, access_granularity = 4),
      new SparsePortDefinition(width = 4, access_granularity = 1)
    )
    val config = new SparseConfig(ports)

    // Test cases
    config.getPortAndIndex(0) shouldEqual (0, 0)  // First port, first index
    config.getPortAndIndex(7) shouldEqual (0, 7)  // First port, last index
    config.getPortAndIndex(8) shouldEqual (1, 0)  // Second port, first index
    config.getPortAndIndex(15) shouldEqual (1, 7) // Second port, last index
    config.getPortAndIndex(24) shouldEqual (2, 0) // Third port, first index
    config.getPortAndIndex(27) shouldEqual (2, 3) // Third port, last index
  }

  it should "throw an exception for an out-of-range input index" in {
    val ports  = Seq(
      new SparsePortDefinition(width = 8, access_granularity = 2),
      new SparsePortDefinition(width = 16, access_granularity = 4)
    )
    val config = new SparseConfig(ports)

    // Out-of-range index
    an[IllegalArgumentException] should be thrownBy config.getPortAndIndex(25)
  }
}
