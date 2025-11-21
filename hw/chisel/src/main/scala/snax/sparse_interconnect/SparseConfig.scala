package snax.sparse_interconnect

class SparsePortDefinition(val width: Int, val access_granularity: Int) {
  def inputsPerBank: Int = width / access_granularity
  def outputsPerPort(NumOut: Int): Int = NumOut / access_granularity
};
class SparseConfig(val ports: Seq[SparsePortDefinition])                {
  def default(numInp: Int): SparseConfig = new SparseConfig(Seq(new SparsePortDefinition(numInp, 1)))
  def inputsPerBank: Int = {
    ports.map(p => p.inputsPerBank).sum;
  };

  def getPortAndIndex(inputIndex: Int):           (Int, Int) = {
    // mapping of index -> port and index
    var cumulativeWidth = 0

    for ((port, portIndex) <- ports.zipWithIndex) {
      val nextCumulativeWidth = cumulativeWidth + port.width

      if (inputIndex < nextCumulativeWidth) {
        // The input index belongs to this port
        val indexWithinPort = inputIndex - cumulativeWidth
        return (portIndex, indexWithinPort)
      }

      cumulativeWidth = nextCumulativeWidth
    }

    throw new IllegalArgumentException(s"Input index $inputIndex is out of range")
  }
  def outputsPerPort(NumOut: Int):                Int        = {
    // find the correct port
    ports.map(p => p.outputsPerPort(NumOut)).sum;
  };
  // Maps the n-th input of a bank (sparse_idx) to the global input index (all inputs across ports)
  def get_global_idx_list(bank: Int):             List[Int]  = {
    var idx_list = List[Int]()
    var acc      = 0;
    for (p <- ports) {
      for (i <- 0 until p.inputsPerBank) {
        idx_list = idx_list :+ acc + i * p.access_granularity + bank % p.access_granularity;
      }
      acc += p.width;
    }
    return idx_list;
  };
  def get_global_idx(sparse_idx: Int, bank: Int): Int        = {
    return get_global_idx_list(bank)(sparse_idx);
  };

};

object SparseConfig {

  // Function to parse the sparse configuration string into SparseConfig
  def parseSparseConfig(configString: String): SparseConfig = {
    val configSeq = configString
      .stripPrefix("[[")
      .stripSuffix("]]")
      .split("\\], \\[")
      .map { pair =>
        val Array(a, b) = pair.split(", ").map(_.toInt)
        (a, b)
      }
      .toSeq

    new SparseConfig(configSeq.map { case (inputsPerBank, accessGranularity) =>
      new SparsePortDefinition(inputsPerBank, accessGranularity)
    })
  }
}
