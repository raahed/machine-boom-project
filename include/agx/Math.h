/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#include <agx/config.h>

#include <agx/macros.h>
#include <agx/agxCore_export.h>
#include <agx/debug.h>
#include <agx/Real.h>
#include <agx/Integer.h>
#include <agx/Random.h>

#include <cmath>

#ifdef _MSC_VER
  #define _USE_MATH_DEFINES
  #include <math.h>

  #ifndef NOMINMAX
    #define NOMINMAX
  #endif

#else
  #include <math.h>
#endif

#include <stdlib.h>
#include <limits>
#include <algorithm>


#include <float.h>

namespace agx
{
  // physics related constants
  const Real GRAVITY_ACCELERATION = Real(9.80665); // in m/s2, SI standard acceleration due to gravity


  #define NUMERIC_MAX(type) std::numeric_limits<type>::max()
  #define NUMERIC_MIN(type) std::numeric_limits<type>::min()
  const Real AGX_EQUIVALENT_EPSILON = (double)1E-9;
  #define AGX_BIT_MASK(bit) (1 << bit)
  #define AGX_TEST_BIT(val, bit) (val & AGX_BIT_MASK(bit))
  #define AGX_SET_BIT(val, bit) (val |= AGX_BIT_MASK(bit))
  #define AGX_UNSET_BIT(val, bit) (val &= ~AGX_BIT_MASK(bit))

  const Real PI = agx::Real(M_PI);
  const Real PI_2 = agx::Real(M_PI_2);
  const Real PI_4 = agx::Real(M_PI_4);

  // define the standard trig values

  const Real DEG_TO_RAD = agx::Real(0.017453292519943295769236907684886);
  const Real RAD_TO_DEG = agx::Real(57.295779513082320876798154814105);

  AGXCORE_EXPORT extern const Real Infinity;
  AGXCORE_EXPORT extern const Real REAL_SQRT_EPSILON;



  template <typename T>
  AGX_FORCE_INLINE T inverse(const T& value)
  {
    return T(1) / value;
  }

  template <typename T>
  AGX_FORCE_INLINE bool isEven(T val)
  {
    return !(bool)(val & 0x1);
  }


  /**
  Sinc function.
  */
  inline Real sinc( Real x, Real nearZero = Real(1E-4) )
  {
    if ( std::abs( x ) < nearZero )
    {
      return Real(1) - (x * x) / Real(6) + (x * x * x * x) / Real(120);
    }
    else
    {
      return std::sin( x ) / x;
    }
  }

  /**
  Compare two values for equality.
  Meaning that the difference between them is less than an epsilon value
  \return true if lhs and rhs are equivalent
  */
  template<typename T>
  AGX_FORCE_INLINE bool _equivalent( T lhs, T rhs, T epsilon = T(AGX_EQUIVALENT_EPSILON) )
  {
    return (lhs + epsilon >= rhs) && (lhs - epsilon <= rhs);
  }

  /**
  Compare two values for equality.
  Meaning that the difference between them is less than an epsilon value
  \return true if lhs and rhs are equivalent
  */
  AGX_FORCE_INLINE bool equivalent(float lhs, float rhs, float epsilon = (float)AGX_EQUIVALENT_EPSILON)
  {
    return _equivalent< float >(lhs, rhs, epsilon);
  }

  /**
  Compare two values for equality.
  Meaning that the difference between them is less than an epsilon value
  \return true if lhs and rhs are equivalent
  */
  AGX_FORCE_INLINE bool equivalent(double lhs, double rhs, double epsilon = (double)AGX_EQUIVALENT_EPSILON)
  {
    return _equivalent< double >(lhs, rhs, epsilon);
  }

  /**
  Compare two values for equality.
  Meaning that the difference between them is less than an epsilon value
  \return true if lhs and rhs are equivalent
  */
  AGX_FORCE_INLINE bool equivalent(float lhs, float rhs, double epsilon)
  {
    return _equivalent(lhs, rhs, (float)epsilon);
  }


  /**
  Compare two values for relative equality.
  Meaning that the difference between them is less than an epsilon value
  The epsilon value is scaled with the values compared, to a minimum of relativeEpsilon.
  \return true if lhs and rhs are equivalent
  */
  template<typename T>
  AGX_FORCE_INLINE bool _relativelyEquivalent( T lhs, T rhs, T relativeEpsilon = T(AGX_EQUIVALENT_EPSILON) )
  {
    return _equivalent(lhs, rhs, (std::abs(lhs) + std::abs(rhs) + T(1)) * relativeEpsilon);
  }

  /**
  Compare two values for relative equality.
  Meaning that the difference between them is less than an epsilon value
  The epsilon value is scaled with the values compared, to a minimum of relativeEpsilon^2.
  \return true if lhs and rhs are equivalent
  */
  AGX_FORCE_INLINE bool relativelyEquivalent(float lhs, float rhs, float epsilon = (float)AGX_EQUIVALENT_EPSILON)
  {
    return _relativelyEquivalent< float >(lhs, rhs, epsilon);
  }

  /**
  Compare two values for relative equality.
  Meaning that the difference between them is less than an epsilon value
  The epsilon value is scaled with the values compared, to a minimum of relativeEpsilon^2.
  \return true if lhs and rhs are equivalent
  */
  AGX_FORCE_INLINE bool relativelyEquivalent(double lhs, double rhs, double epsilon = (double)AGX_EQUIVALENT_EPSILON)
  {
    return _relativelyEquivalent< double >(lhs, rhs, epsilon);
  }

  /**
  Compare two values for relative equality.
  Meaning that the difference between them is less than an epsilon value
  The epsilon value is scaled with the values compared, to a minimum of relativeEpsilon^2.
  \return true if lhs and rhs are equivalent
  */
  AGX_FORCE_INLINE bool relativelyEquivalent(float lhs, float rhs, double epsilon)
  {
    return _relativelyEquivalent(lhs, rhs, (float)epsilon);
  }


  /// \return true if \p f equals zero
#define INT_EQUALS_ZERO( T ) \
  AGX_FORCE_INLINE bool equalsZero( T f ) \
  { \
    return ( f == 0 ); \
  }

#define INT_IS_NAN( T ) \
  AGX_FORCE_INLINE bool isNaN( T /*v*/ ) \
  { \
    return false; \
  }

#define INT_IS_INF( T ) \
  AGX_FORCE_INLINE bool isInf( T /*v*/ ) \
  { \
  return false; \
  }

#define INT_IS_FINITE( T ) \
  AGX_FORCE_INLINE bool isFinite( T /*v*/ ) \
  { \
  return true; \
  }

#define UINT_ABSOLUTE( T ) \
  AGX_FORCE_INLINE T absolute( T v ) \
  { \
  return v; \
  }

  INT_EQUALS_ZERO( Int8 )
  INT_EQUALS_ZERO( Int16 )
  INT_EQUALS_ZERO( Int32 )
  INT_EQUALS_ZERO( Int64 )

  INT_EQUALS_ZERO( UInt8 )
  INT_EQUALS_ZERO( UInt16 )
  INT_EQUALS_ZERO( UInt32 )
  INT_EQUALS_ZERO( UInt64 )

  INT_IS_NAN( Int8 )
  INT_IS_NAN( Int16 )
  INT_IS_NAN( Int32 )
  INT_IS_NAN( Int64 )

  INT_IS_NAN( UInt8 )
  INT_IS_NAN( UInt16 )
  INT_IS_NAN( UInt32 )
  INT_IS_NAN( UInt64 )

  INT_IS_INF( Int8 )
  INT_IS_INF( Int16 )
  INT_IS_INF( Int32 )
  INT_IS_INF( Int64 )

  INT_IS_INF( UInt8 )
  INT_IS_INF( UInt16 )
  INT_IS_INF( UInt32 )
  INT_IS_INF( UInt64 )

  INT_IS_FINITE( Int8 )
  INT_IS_FINITE( Int16 )
  INT_IS_FINITE( Int32 )
  INT_IS_FINITE( Int64 )

  INT_IS_FINITE( UInt8 )
  INT_IS_FINITE( UInt16 )
  INT_IS_FINITE( UInt32 )
  INT_IS_FINITE( UInt64 )

  UINT_ABSOLUTE( UInt8 )
  UINT_ABSOLUTE( UInt16 )
  UINT_ABSOLUTE( UInt32 )
  UINT_ABSOLUTE( UInt64 )

#undef INT_EQUALS_ZERO
#undef INT_IS_NAN
#undef INT_IS_INF
#undef INT_IS_FINITE
#undef UINT_ABSOLUTE

  /// \return true if \p f is within the range of (-FLT_EPSILON, FLT_EPSLION).
  AGX_FORCE_INLINE bool equalsZero( float f, float eps=FLT_EPSILON )
  {
    return ( f < eps && f > -eps );
  }

  /// \return true if \p d is within the range of (-DBL_EPSILON, DBL_EPSILON).
  AGX_FORCE_INLINE bool equalsZero( double d, double eps=DBL_EPSILON )
  {
    return ( d < eps && d > -eps );
  }


  /** return the absolute value.
  */
  template<typename T>
  AGX_FORCE_INLINE T absolute( T v )
  {
    return std::abs(v);
  }

  template<typename T>
  AGX_FORCE_INLINE bool isInf( T v )
  {
    return std::isinf( v ) != 0;
  }

  template<typename T>
  AGX_FORCE_INLINE bool isNaN( T v )
  {
    return std::isnan( v );
  }

  template<typename T>
  AGX_FORCE_INLINE bool isFinite( T v )
  {
    return std::isfinite(v);
  }

  /**
  \return \param v clamped between \param minimum and \param maximum
  */
  template<typename T1, typename T2, typename T3>
  AGX_FORCE_INLINE T1 clamp( T1 v, T2 minimum, T3 maximum )
  {
    return v < minimum ? minimum : v > maximum ? maximum : v;
  }


  /**
  \return -1 if v < 0 1 if >= 0.
  */
  template<typename T>
  AGX_FORCE_INLINE T sign( T v )
  {
    return v < T(0) ? T(-1) : T(1);
  }

  /// \return v*v
  template<typename T>
  AGX_FORCE_INLINE T square( T v )
  {
    return v*v;
  }

  /// \return v*v*sign(v)
  template<typename T>
  AGX_FORCE_INLINE T signedSquare( T v )
  {
    return v < ( T )0 ? -v*v : v*v;
  }

  /// \return \param angle converted from angle to radians
  AGX_FORCE_INLINE Real degreesToRadians( Real angle )
  {
    return angle * Real(DEG_TO_RAD);
  }

  /// \return \param angle converted from radians to degrees
  AGX_FORCE_INLINE Real radiansToDegrees(Real angle)
  {
    return angle*Real(RAD_TO_DEG);
  }

  /**
  Normalize an angle.

  \param angle - The angle that will be transformed into the range
  \param positiveRange true generates angle in interval [0, 2*Pi], false use interval [-Pi, Pi]
  \return a specified angle in a range either:
  */
  AGXCORE_EXPORT Real normalizedAngle( Real angle, bool positiveRange = false );

  /// \return true if \param a <= \param  b within \param eps
  AGX_FORCE_INLINE bool leq( double a, double b, double eps = (double)AGX_EQUIVALENT_EPSILON )
  {
    return a < b + eps;
  }

  /// \return true if \param a <= \param  b within \param eps
  AGX_FORCE_INLINE bool leq(float a, float b, float eps = (float)AGX_EQUIVALENT_EPSILON)
  {
    return a < b + eps;
  }

  /// \return true if \param a >= \param  b within \param eps
  AGX_FORCE_INLINE bool geq(double a, double b, double eps = (double)AGX_EQUIVALENT_EPSILON)
  {
    return a > b - eps;
  }

  /// \return true if \param a >= \param  b within \param eps
  AGX_FORCE_INLINE bool geq(float a, float b, float eps = (float)AGX_EQUIVALENT_EPSILON)
  {
    return a > b - eps;
  }


  AGX_FORCE_INLINE Int32 log2(size_t val)
  {
    Int32 ret = -1;
    while (val != 0) {
      val >>= 1;
      ret++;
    }
    return ret;
  }


  /**
  \return true if \param value is a power of two
  */
  template <typename T>
  AGX_FORCE_INLINE bool isPowerOfTwo(T value)
  {
    return (value & (value-1)) == 0;
  }

  AGX_FORCE_INLINE UInt32 alignPowerOfTwo(UInt32 value)
  {
    value--;
    value |= (value >> 1);
    value |= (value >> 2);
    value |= (value >> 4);
    value |= (value >> 8);
    value |= (value >> 16);
    value++;
    return value;
  }

  AGX_FORCE_INLINE Int32 alignPowerOfTwo(Int32 value)
  {
    value--;
    value |= (value >> 1);
    value |= (value >> 2);
    value |= (value >> 4);
    value |= (value >> 8);
    value |= (value >> 16);
    value++;
    return value;
  }


  AGX_FORCE_INLINE UInt64 alignPowerOfTwo(UInt64 value)
  {
    value--;
    value |= (value >> 1);
    value |= (value >> 2);
    value |= (value >> 4);
    value |= (value >> 8);
    value |= (value >> 16);
    value |= (value >> 32);
    value++;
    return value;
  }

  AGX_FORCE_INLINE Int64 alignPowerOfTwo(Int64 value)
  {
    value--;
    value |= (value >> 1);
    value |= (value >> 2);
    value |= (value >> 4);
    value |= (value >> 8);
    value |= (value >> 16);
    value |= (value >> 32);
    value++;
    return value;
  }


  template <typename T>
  AGX_FORCE_INLINE T align_ceil(T value, T alignment)
  {
    agxAssert(alignment > 0);
    return value > 0 ? (value + (alignment-1) - ((value-1) % alignment)) : 0;
  }

  template <typename T1, typename T2>
  AGX_FORCE_INLINE T1 *align_ceil(const T1 *ptr, T2 alignment)
  {
    return (T1 *)align_ceil<agx::UInt64>((agx::UInt64)ptr, (agx::UInt64)alignment);
  }


  template <typename T>
  AGX_FORCE_INLINE T align_floor(T value, T alignment)
  {
    agxAssert(alignment > 0);
    return value - (value % alignment);
  }

  template <typename T1, typename T2>
  AGX_FORCE_INLINE T1 *align_floor(const T1 *ptr, T2 alignment)
  {
    return (T1 *)align_floor<agx::UInt64>((agx::UInt64)ptr, (agx::UInt64)alignment);
  }

  template <typename T>
  AGX_FORCE_INLINE bool isAligned(T value, T alignment)
  {
    return value % alignment == 0;
  }

  template <typename T1, typename T2>
  AGX_FORCE_INLINE bool isAligned(const T1 *ptr, T2 alignment)
  {
    return isAligned((uintptr_t)ptr, (uintptr_t)alignment);
  }

  template <typename T>
  AGX_FORCE_INLINE T align_ceil2(T value, T alignment)
  {
    agxAssert(isPowerOfTwo(alignment));
    T tmp = alignment-1;
    return ((value + tmp) & ~tmp);
  }

  template <typename T1, typename T2>
  AGX_FORCE_INLINE T1 *align_ceil2(const T1 *ptr, T2 alignment)
  {
    return (T1 *)align_ceil2<agx::UInt64>((agx::UInt64)ptr, (agx::UInt64)alignment);
  }

  template <typename T>
  AGX_FORCE_INLINE T align_floor2(T value, T alignment)
  {
    agxAssert(isPowerOfTwo(alignment));
    T tmp = alignment-1;
    return (value & ~tmp);
  }

  template <typename T1, typename T2>
  AGX_FORCE_INLINE T1 *align_floor2(const T1 *ptr, T2 alignment)
  {
    return (T1 *)align_floor2<agx::UInt64>((agx::UInt64)ptr, (agx::UInt64)alignment);
  }

  // basic method for generating primes
  AGXCORE_EXPORT int nextPrime(int n);

  /// strtod ignoring global locale
  AGXCORE_EXPORT double strtod(const char *nptr, char **endptr);

  /// strtod ignoring global locale
  AGXCORE_EXPORT float strtof(const char *nptr, char **endptr);



  AGXCORE_EXPORT agx::UniformInt32Generator& getRandGenerator();

  /**
  \return a random value in the range [\p min, \p max)
  */
  AGX_FORCE_INLINE int irandom(int min, int max)
  {
    return getRandGenerator().rand() % (max - min) + min;
  }

  /// \return a random value in the range [\p min, \p max]
  template <typename T>
  AGX_FORCE_INLINE T random(T min, T max)
  {
    Real sample = static_cast<Real>(getRandGenerator().rand());
    Real randEnd = static_cast<Real>(RAND_MAX);
    Real range_fraction = sample / randEnd;
    Real range = static_cast<Real>(max - min);
    T offset = static_cast<T>(range_fraction * range);
    T value = static_cast<T>(min + offset);
    return value;
  }


  /**
  \p a, \p b, \p c, \p d sides of the tetrahedron
  \return the volume of a tetrahedron. */
  template<typename T>
  AGX_FORCE_INLINE Real computeVolume( const T& a, const T& b, const T& c, const T& d )
  {
    return fabsf( ( ( b -c ) ^ ( a - b ) )*( d - b ) );
  }

  /** compute the volume of a prism. */
  template<typename T>
  AGX_FORCE_INLINE Real computeVolume( const T& f1, const T& f2, const T& f3,
                              const T& b1, const T& b2, const T& b3 )
  {
    return computeVolume( f1, f2, f3, b1 ) +
           computeVolume( b1, b2, b3, f2 ) +
           computeVolume( b1, b3, f2, f3 );
  }

  /// Linearly interpolate from \p a to \p b using \p s = {0,..1}
  template<typename T1, typename T2>
  AGX_FORCE_INLINE T1 const lerp(T1 const& a, T1 const& b, T2 s)
  {
    if (s >= 1)
      return b;
    if (s <= 0)
      return a;

    return (T1)(a*(1-s)  + b*s);
  }

  /// logarithmic interpolation from \p a to \p b using \p s = {0,..1}
  template<typename T>
  AGX_FORCE_INLINE T const logInterpolate(T const& a, T const& b, float s)
  {
    return std::pow(T(10), agx::lerp(std::log10(a), std::log10(b), s));
  }

  /**
  Finds and returns n given \p value = 2^n.
  \param value - integer value such that value = 2^n
  \return n in value = 2^n
  */
  template<typename T>
  AGX_FORCE_INLINE agx::UInt32 pow2Exponent( T value )
  {
    static const agx::UInt32 multiplyDeBruijnBitPosition2[ 32 ] =
    {
      0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
      31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
    };

    return multiplyDeBruijnBitPosition2[ ( (unsigned int)value * 0x077CB531U ) >> 27 ];
  }

    /**
    \param n - The value that will be tested for the highest set bit.

    highestBitToIndex(0)    - 0
    highestBitToIndex(0x2)  - 1
    highestBitToIndex(0x3)  - 1
    highestBitToIndex(0x4)  - 2
    highestBitToIndex(0x8)  - 3
    ...

    \return the index (0..) of the highest set bit in the value \p n
    */
    template<typename T>
    inline agx::UInt8 highestBitToIndex(T n)
    {
      if (n == 0)
        return 0;

      agx::UInt8 msb = 0;
      n = T(n / 2);
      while (n != 0) {
        n = T(n / 2);
        msb++;
      }

      return msb;
    }


} // namespace agx
