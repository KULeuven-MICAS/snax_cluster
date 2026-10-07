package snax.DataPathExtension

import java.io.PrintWriter

import chisel3._

/** Emits the SystemVerilog of the CREST decompressor at the split cluster's parameters (groupBeats 64, escWords 12,
  * k1Lanes 32, k2Lanes 16, planeDepth 8) inside the extension test harness, to the file named by the first argument;
  * with a second argument "zp", the zero-point-aware variant.
  *
  * sbt "Test/runMain snax.DataPathExtension.CrestEmitVerilog out.sv [zp]"
  */
object CrestEmitVerilog {
  def main(args: Array[String]): Unit = {
    val zp  = args.length > 1 && args(1) == "zp"
    val sv  = getVerilogString(
      new DataPathExtensionHarness(new HasCrestDecompressor(groupBeats = 64, escWords = 12, zeroPoint = zp))
    )
    val out = new PrintWriter(args(0))
    out.write(sv)
    out.close()
  }
}
