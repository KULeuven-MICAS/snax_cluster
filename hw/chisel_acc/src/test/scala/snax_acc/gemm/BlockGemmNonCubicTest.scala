package snax_acc.gemm

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** Non-cubic mesh shapes.
  *
  * Every shipped BlockGemm test uses meshRow == tileSize == meshCol (5x5x5, and the
  * default config). A four-engine cluster wants 16x16x4 -- 1024 MAC/cycle at the SAME
  * operand bandwidth as 8x8x8's 512, because operand bytes/MAC is 1/meshRow + 1/meshCol
  * and does not depend on tileSize. That configuration elaborates, builds and links, but
  * on the cluster hart 0 spins forever in the gemmX completion poll.
  *
  * These cases pin down whether the fault is in BlockGemm itself or in the integration
  * around it.
  */
class BlockGemmNonCubicTest extends AnyFlatSpec with ChiselScalatestTester with AbstractBlockGemmTest {

  private def cfg(meshRow: Int, tileSize: Int, meshCol: Int) = new GemmParams(
    dataWidthA          = GemmConstant.dataWidthA,
    dataWidthB          = GemmConstant.dataWidthB,
    dataWidthMul        = GemmConstant.dataWidthMul,
    dataWidthC          = GemmConstant.dataWidthC,
    dataWidthAccum      = GemmConstant.dataWidthAccum,
    subtractionCfgWidth = GemmConstant.subtractionCfgWidth,
    meshRow             = meshRow,
    tileSize            = tileSize,
    meshCol             = meshCol,
    addrWidth           = GemmConstant.addrWidth,
    sizeConfigWidth     = GemmConstant.sizeConfigWidth
  )

  "BlockGemm" should "work with tileSize smaller than the mesh (16x4x16)" in {
    test(new BlockGemmDelayedWrapper(cfg(16, 4, 16))) { dut =>
      dut.clock.setTimeout(20000)
      dut.clock.step(5)
      BlockGemmRandomTest(dut, 1, 1, 1)
      BlockGemmRandomTest(dut, 2, 2, 2)
      BlockGemmRandomTest(dut, 3, 1, 2)
    }
  }

  "BlockGemm" should "work with a rectangular mesh (8x8x16)" in {
    test(new BlockGemmDelayedWrapper(cfg(8, 8, 16))) { dut =>
      dut.clock.setTimeout(20000)
      dut.clock.step(5)
      BlockGemmRandomTest(dut, 1, 1, 1)
      BlockGemmRandomTest(dut, 2, 2, 2)
    }
  }

  "BlockGemm" should "work with tileSize larger than the mesh (8x16x8)" in {
    test(new BlockGemmDelayedWrapper(cfg(8, 16, 8))) { dut =>
      dut.clock.setTimeout(20000)
      dut.clock.step(5)
      BlockGemmRandomTest(dut, 1, 1, 1)
      BlockGemmRandomTest(dut, 2, 2, 2)
    }
  }
}
