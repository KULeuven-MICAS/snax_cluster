package snax_acc.utils

import scala.Float

import chisel3._

import snax_acc.versacore.FpType

trait fpUtils {

  /** Generalized helper function to encode Float as UInt */
  def floatToUInt(expWidth: Int, sigWidth: Int, value: Float): BigInt = {
    // Sign bit + exponent + significand
    val totalWidth = expWidth + sigWidth + 1

    // Convert to IEEE 754 32-bit float representation
    val ieee754Bits = java.lang.Float.floatToIntBits(value)

    // Extract sign, exponent, and significand
    val sign        = (ieee754Bits >>> 31) & 0x1
    val exponent    = (ieee754Bits >>> 23) & 0xff
    val significand = ieee754Bits & 0x7fffff

    // Re-normalize the exponent to fit expWidth
    val bias32      = 127 // IEEE 754 bias for 32-bit float
    val biasTarget  = (1 << (expWidth - 1)) - 1
    val newExponent =
      (exponent - bias32 + biasTarget).max(0).min((1 << expWidth) - 1)

    // Truncate significand to fit sigWidth
    val newSignificand = significand >>> (23 - sigWidth)

    // Assemble the custom float representation
    val customBits =
      (sign << (expWidth + sigWidth)) | (newExponent << sigWidth) | newSignificand
    BigInt(
      customBits & ((1L << totalWidth) - 1)
    ) // Mask to ensure valid bit-width
  }

  def floatToUInt(fpType: FpType, value: Float): BigInt = floatToUInt(fpType.expWidth, fpType.sigWidth, value)

  /** Generalized helper function to decode UInt to Float */
  def uintToFloat(expWidth: Int, sigWidth: Int, bits: BigInt): Float = {
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
    val ieee754Bits =
      (sign.toInt << 31) | (ieeeExponent << 23) | ieeeSignificand
    java.lang.Float.intBitsToFloat(ieee754Bits)
  }

  def uintToFloat(fpType: FpType, value: BigInt): Float = uintToFloat(fpType.expWidth, fpType.sigWidth, value)
  def uintToFloat(fpType: FpType, value: UInt):   Float = uintToFloat(fpType.expWidth, fpType.sigWidth, value.litValue)
  def quantize(fpType:    FpType, value: Float):  Float = uintToFloat(fpType, floatToUInt(fpType, value))

  /** Generate a true random value in the given FpType, where exponent and mantissa are sampled independently
    */
  def getTrueRandomValue(fpType: FpType): Float = {
    val r          = new scala.util.Random()
    val randomBits = BigInt(fpType.width, r)
    uintToFloat(fpType, randomBits)
  }

  /** Generated a bounded random float in the given FpType. Maximum value should be calculated such that a large number
    * of operations on randomly sampled numbers will not overflow with high probability
    */
  def genRandomValue(fpType: FpType): Float = {

    val margin      = 16
    val maxExponent = ((1 << (fpType.expWidth - 1)) - 1) / margin
    val maxVal      = (1 << maxExponent).toFloat / margin
    val r           = new scala.util.Random()
    (2 * r.nextFloat() - 1f) * maxVal
  }

  /** Process two floating point numbers in a given format by introducing the hardware limitations of this format
    */
  def fpOperationHardware(
    a:     Float,
    b:     Float,
    typeA: FpType,
    typeB: FpType,
    op:    (Float, Float) => Float
  ) = {
    op(quantize(typeA, a), quantize(typeB, b))
  }

  /** Multiplies two floating point numbers in a given format by introducing the hardware limitations of this format
    */
  def fpOperationHardware(a: Float, typeA: FpType, op: Float => Float) =
    op(
      quantize(typeA, a)
    )

  /** Returns true iff the hardware result a (as UInt) correctly represents the float While hardware modules use RNE
    * (Round to Nearest, ties to Even), the fp arithmetic in here does not model this. Hence, the value from hardware
    * can be 0 or 1 higher than the expected value, but not smaller.
    *
    * -0 and +0 are also accepted as equal.
    */
  def fpEqualsHardware(expected: Float, from_hw: UInt, typeB: FpType) = {
    val expected_bigint     = floatToUInt(typeB, expected)
    val from_hw_bigint      = from_hw.litValue
    val plusEqualsMinusZero = (expected_bigint == 0 && from_hw_bigint == (BigInt(1) << typeB.width - 1))
    from_hw_bigint - expected_bigint <= 1 || plusEqualsMinusZero
  }

  /** Returns true iff the hardware result a (as UInt) correctly represents the float. The result is allowed to differ
    * in `lsbTolerance` LSB bits, as a result from rounding errors propagated through operations.
    */
  def fpAlmostEqualsHardware(expected: Float, from_hw: UInt, typeB: FpType) = {
    val lsbTolerance        = 4
    val expected_bigint     = floatToUInt(typeB, expected)
    val from_hw_bigint      = from_hw.litValue
    val plusEqualsMinusZero = (expected_bigint == 0 && from_hw_bigint == (BigInt(1) << typeB.width - 1))
    (from_hw_bigint - expected_bigint).abs <= (1 << lsbTolerance) - 1 || plusEqualsMinusZero
  }

  /** Define operator symbol for mulFpHardware. Signature: ((Float, FpType), (Float, FpType)) => Float
    */
  implicit class FpHardwareOps(a: (Float, FpType)) {
    def *(b: (Float, FpType)): Float = fpOperationHardware(a._1, b._1, a._2, b._2, _ * _)
    def +(b: (Float, FpType)): Float = fpOperationHardware(a._1, b._1, a._2, b._2, _ + _)

    /** FP results are exactly the same, save for a +1 bit rounding error (RNE instead of floor). Not to be confused
      * with the Chisel3 === operator
      */
    def ===(b: UInt): Boolean = fpEqualsHardware(a._1, b, a._2)

    /** FP results are similar, tolerating an accumulated rounding error */
    def =~=(b: UInt): Boolean = fpAlmostEqualsHardware(a._1, b, a._2)
  }

}
