package snax.utils

/** FP16 <-> Double for testbenches.
  *
  * These exist because `java.lang.Float.floatToFloat16` / `float16ToFloat` are Java 20 APIs and this
  * tree builds on an older JDK -- every tester that called them failed to COMPILE, which takes the whole
  * test tree down with it, not just those files. Doing the conversion here keeps the testers portable and
  * puts one implementation where there were five copies of a one-liner.
  *
  * Round-to-nearest-even, with subnormals and overflow-to-infinity, so a tester can score a device result
  * against the same grid the hardware narrows onto.
  */
object TestFp16 {

  /** Double -> FP16 bit pattern, round-to-nearest-even. */
  def enc(d: Double): Int = {
    val f    = d.toFloat
    val bits = java.lang.Float.floatToIntBits(f)
    val sign = (bits >>> 16) & 0x8000
    val exp  = (bits >>> 23) & 0xff
    val mant = bits & 0x7fffff
    if (exp == 0xff) {                                   // inf / NaN
      if (mant != 0) sign | 0x7e00 else sign | 0x7c00
    } else {
      val e = exp - 127 + 15                             // rebias
      if (e >= 0x1f) sign | 0x7c00                       // overflow -> inf
      else if (e > 0) {                                  // normal
        val m     = mant >>> 13
        val rest  = mant & 0x1fff
        val round = if (rest > 0x1000 || (rest == 0x1000 && (m & 1) == 1)) 1 else 0
        sign | (((e << 10) | m) + round)                 // a mantissa carry walks into the exponent
      } else if (e > -10) {                              // subnormal
        val full  = mant | 0x800000
        val shift = 14 - e
        val m     = full >>> shift
        val rest  = full & ((1 << shift) - 1)
        val half  = 1 << (shift - 1)
        val round = if (rest > half || (rest == half && (m & 1) == 1)) 1 else 0
        sign | (m + round)
      } else sign                                        // underflow -> signed zero
    }
  }

  /** FP16 bit pattern -> Double. */
  def dec(b: Int): Double = {
    val h    = b & 0xffff
    val sign = if ((h & 0x8000) != 0) -1.0 else 1.0
    val exp  = (h >>> 10) & 0x1f
    val mant = h & 0x3ff
    if (exp == 0) sign * mant * math.pow(2.0, -24)
    else if (exp == 0x1f) if (mant == 0) sign * Double.PositiveInfinity else Double.NaN
    else sign * (1.0 + mant / 1024.0) * math.pow(2.0, exp - 15)
  }

  /** Monotonic key: adjacent FP16 values map to adjacent keys, so |key(a)-key(b)| IS the ULP distance. */
  def mono(h: Int): Int = { val m = h & 0x7fff; if ((h & 0x8000) != 0) 0x8000 - m else 0x8000 + m }

  /** FP16-ULP distance between two reals, each first rounded onto the FP16 grid. */
  def ulp(a: Double, b: Double): Int = math.abs(mono(enc(a)) - mono(enc(b)))
}
