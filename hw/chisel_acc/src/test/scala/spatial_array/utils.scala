package snax_acc.spatial_array

import chisel3._

import chiseltest._

trait GeMMTestUtils {

  // Function to convert UInt to SInt manually
  def toSInt(value: Int, bitWidth: Int, ifTrans: Boolean): Int = {
    if (!ifTrans) return value
    val mask     = (1L << bitWidth) - 1
    val unsigned = value.toLong & mask
    val signBit  = 1L << (bitWidth - 1)
    if ((unsigned & signBit) != 0) (unsigned - (1L << bitWidth)).toInt
    else unsigned.toInt
  }

  def WaitOrTimeout(signalToAssert: Bool, clock: Clock, timeout: Int = 100) = {
    var waitCnt = 0
    while (!signalToAssert.peekBoolean()) {
      if (waitCnt > timeout) {
        println(s"Timeout at cycle ${clock.getStepCount} for signal ${signalToAssert}")
        assert(false)
      }
      clock.step(1)
      waitCnt += 1
    }
  }

  /** Translate the temporal index (i.e. the cycle) to spatial index (i.e. the matrix index).
    *   - The temporal index ranges from 0 to (M*K*N - 1).
    *   - Matrix A (size MxK) needs to be read in row major order, and is stored in row major order. (The testbench only
    *     sees a 1D array of size M*K.)
    *   - Matrix B (size KxN) needs to be read in column major order, and is stored in column major order.
    *   - The temporal index follows the following loop ordering:
    *     - for (m = 0 to M - 1);
    *       - for (n = 0 to N - 1);
    *         - for (k = 0 to K - 1);
    *           - (do operation)
    */
  def temporalToSpatialIndicesAB(temporalIndex: Int, K: Int, N: Int): (Int, Int) = {
    val m = temporalIndex / (K * N) // Row index of D
    val n = (temporalIndex / K) % N // Column index of D
    val k = temporalIndex       % K // Shared dimension index

    val indexA = m * K + k
    val indexB = n * K + k

    (indexA, indexB)
  }

}

trait fpUtils {

  object fp16 {
    val expWidth = 5
    val sigWidth = 10
  }
  object fp32 {
    val expWidth = 8
    val sigWidth = 23
  }

  // Generalized helper function to encode Float as UInt
  def floatToUInt(expWidth: Int, sigWidth: Int, value: scala.Float): BigInt = {
    val totalWidth = expWidth + sigWidth + 1 // Sign bit + exponent + significand

    // Convert to IEEE 754 32-bit float representation
    val ieee754Bits = java.lang.Float.floatToIntBits(value)

    // Extract sign, exponent, and significand
    val sign        = (ieee754Bits >>> 31) & 0x1
    val exponent    = (ieee754Bits >>> 23) & 0xff
    val significand = ieee754Bits & 0x7fffff

    // Re-normalize the exponent to fit expWidth
    val bias32      = 127 // IEEE 754 bias for 32-bit float
    val biasTarget  = (1 << (expWidth - 1)) - 1
    val newExponent = (exponent - bias32 + biasTarget).max(0).min((1 << expWidth) - 1)

    // Truncate significand to fit sigWidth
    val newSignificand = significand >>> (23 - sigWidth)

    // Assemble the custom float representation
    val customBits = (sign << (expWidth + sigWidth)) | (newExponent << sigWidth) | newSignificand
    BigInt(customBits & ((1L << totalWidth) - 1)) // Mask to ensure valid bit-width
  }

  // Generalized helper function to decode UInt to Float
  def uintToFloat(expWidth: Int, sigWidth: Int, bits: BigInt): scala.Float = {
    expWidth + sigWidth + 1 // Sign bit + exponent + significand

    // Extract sign, exponent, and significand
    val sign        = (bits >> (expWidth + sigWidth)) & 0x1
    val exponent    = (bits >> sigWidth) & ((1 << expWidth) - 1)
    val significand = bits & ((1 << sigWidth) - 1)

    // Re-normalize the exponent back to IEEE 754
    val biasTarget   = (1 << (expWidth - 1)) - 1
    val bias32       = 127 // IEEE 754 bias for 32-bit float
    val ieeeExponent = (exponent.toInt - biasTarget + bias32).max(0).min(255)

    // Re-expand the significand to fit IEEE 754
    val ieeeSignificand = significand.toInt << (23 - sigWidth)

    // Assemble the IEEE 754 representation
    val ieee754Bits = (sign.toInt << 31) | (ieeeExponent << 23) | ieeeSignificand
    java.lang.Float.intBitsToFloat(ieee754Bits)
  }

}
