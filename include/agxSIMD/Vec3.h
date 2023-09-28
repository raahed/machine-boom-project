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

#ifndef AGX_SIMD_VEC3_H
#define AGX_SIMD_VEC3_H


#include <agx/config/AGX_USE_SSE4.h>
#include <agx/config/AGX_USE_SSE.h>
#include <agx/config/AGX_USE_AVX.h>
#include <agx/Integer.h>
#include <agx/Real.h>
#include <agx/Vec3.h>
#include <iosfwd>

#if AGX_USE_SSE()
#include <pmmintrin.h>
#endif

#if AGX_USE_SSE4()
#include <smmintrin.h>
#endif

#if AGX_USE_AVX()
#include <immintrin.h>
#endif

DOXYGEN_START_INTERNAL_BLOCK()
namespace agxSIMD
{
  // Vec3 base class, explicit specializations for Real32 and Real64
  template <typename T>
  class Vec3T
  {
  };

  // 32bit real specialization
  template <>
  class Vec3T<agx::Real32>
  {
  public:
    typedef agx::Real32 RealT;
    typedef agx::Real32 Type;
    static const agx::UInt ALIGNMENT = 16;

  public:

    Vec3T();
    Vec3T(agx::Real32 val);
    Vec3T(agx::Real32 x, agx::Real32 y, agx::Real32 z);

#if AGX_USE_SSE()
    Vec3T(const __m128 _xyzw);
#endif

    explicit Vec3T(const agx::Vec3f& vec);
    explicit Vec3T(const agx::Vec3d&);

    agx::Real32 x() const;
    agx::Real32 y() const;
    agx::Real32 z() const;

    agx::Real32 length() const;
    agx::Real32 length2() const;

    void store(agx::Vec3f& target) const;
    Vec3T absolute() const;
    Vec3T negate() const;
    Vec3T scale(const Vec3T& val) const;

#if AGX_USE_SSE()
    Vec3T scale(const __m128 val) const;
#endif

    Vec3T scale(agx::Real32 val) const;

    static Vec3T madd(const Vec3T& a, const Vec3T& b, const Vec3T& c);
    static Vec3T cross(const Vec3T& lhs, const Vec3T& rhs);

    static agx::Real32 dot(const Vec3T& lhs, const Vec3T& rhs);

    static agx::Real32 innerProduct(const Vec3T& v01, const Vec3T& v02, const Vec3T& v11, const Vec3T& v12);

  public:
#if AGX_USE_SSE()
    __m128 xyzw;
#else
    agx::Vec3f xyzw;
#endif
  };

  // 64bit real specialization
  template <>
  class Vec3T<agx::Real64>
  {
  public:
    typedef agx::Real64 RealT;
    typedef agx::Real64 Type;

#if AGX_USE_AVX()
    static const agx::UInt ALIGNMENT = 32;
#else
    static const agx::UInt ALIGNMENT = 16;
#endif

  public:
    Vec3T();
    Vec3T(agx::Real64 val);
    Vec3T(agx::Real64 x, agx::Real64 y, agx::Real64 z);

#if AGX_USE_AVX()
    Vec3T(const __m256d _xyzw);
#elif AGX_USE_SSE()
    Vec3T(const __m128d _xy, const __m128d _zw);
#endif

    explicit Vec3T(const agx::Vec3d& vec);
    explicit Vec3T(const agx::Vec3f& vec);

    agx::Real64 x() const;
    agx::Real64 y() const;
    agx::Real64 z() const;

    agx::Real64 length() const;
    agx::Real64 length2() const;

    void store(agx::Vec3d& target) const;
    void store(agx::Vec3f& target) const;

    Vec3T absolute() const;
    Vec3T negate() const;
    Vec3T scale(const Vec3T& val) const;

#if AGX_USE_SSE() && !AGX_USE_AVX()
    Vec3T scale(const __m128d val) const;
#endif

    Vec3T scale(agx::Real64 val) const;

    static Vec3T madd(const Vec3T& a, const Vec3T& b, const Vec3T& c);
    static Vec3T cross(const Vec3T& lhs, const Vec3T& rhs);
    static agx::Real64 dot(const Vec3T& lhs, const Vec3T& rhs);


    static agx::Real64 innerProduct(const Vec3T& v01, const Vec3T& v02, const Vec3T& v11, const Vec3T& v12);

  public:
#if AGX_USE_AVX()
    __m256d xyzw;
#elif AGX_USE_SSE()
    __m128d xy;
    __m128d zw;
#else
    agx::Vec3d xyzw;
#endif
  };


  typedef Vec3T<agx::Real32> Vec3f;
  typedef Vec3T<agx::Real64> Vec3d;
  typedef Vec3T<agx::Real> Vec3;

  std::ostream& operator << ( std::ostream& output, const Vec3d& v );
  std::ostream& operator << ( std::ostream& output, const Vec3f& v );



  /* Implementation */

  AGX_FORCE_INLINE Vec3T<agx::Real32>::Vec3T() : Vec3T(0.0f)
  {}

#if AGX_USE_SSE()
  AGX_FORCE_INLINE Vec3T<agx::Real32>::Vec3T(agx::Real32 val) : xyzw(_mm_setr_ps(val, val, val, 0))
  {
  }

  AGX_FORCE_INLINE Vec3T<agx::Real32>::Vec3T(agx::Real32 x, agx::Real32 y, agx::Real32 z) : xyzw(_mm_setr_ps(x, y, z, 0))
  {
  }

  AGX_FORCE_INLINE Vec3T<agx::Real32>::Vec3T(const __m128 _xyzw) : xyzw(_xyzw)
  {}

  AGX_FORCE_INLINE Vec3T<agx::Real32>::Vec3T(const agx::Vec3f& vec) : xyzw(_mm_load_ps(vec.ptr()))
  {
  }

  AGX_FORCE_INLINE Vec3T<agx::Real32>::Vec3T(const agx::Vec3d& vec)
  {
  #if AGX_USE_AVX()
    xyzw = _mm256_cvtpd_ps(_mm256_load_pd(vec.ptr()));
  #else
    __m128 lower = _mm_cvtpd_ps(_mm_load_pd(vec.ptr()));
    __m128 upper = _mm_cvtpd_ps(_mm_load_pd(vec.ptr()+2));
    xyzw = _mm_or_ps(lower, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(upper), 8)));
  #endif
  }

#else

  AGX_FORCE_INLINE Vec3T<agx::Real32>::Vec3T(agx::Real32 val) : xyzw(val)
  {
  }

  AGX_FORCE_INLINE Vec3T<agx::Real32>::Vec3T(agx::Real32 x, agx::Real32 y, agx::Real32 z) : xyzw(x, y, z)
  {
  }

  AGX_FORCE_INLINE Vec3T<agx::Real32>::Vec3T(const agx::Vec3f& vec) : xyzw(vec)
  {
  }

  AGX_FORCE_INLINE Vec3T<agx::Real32>::Vec3T(const agx::Vec3d& vec) : xyzw(vec)
  {
  }
#endif


  AGX_FORCE_INLINE agx::Real32 Vec3T<agx::Real32>::x() const
  {
#if AGX_USE_SSE()
    AGX_ALIGNED( agx::Real32, 16 ) tmp[4];
    _mm_store_ps( tmp, xyzw );
    return tmp[0];
#else
    return xyzw.x();
#endif
  }

  AGX_FORCE_INLINE agx::Real32 Vec3T<agx::Real32>::y() const
  {
#if AGX_USE_SSE()
    AGX_ALIGNED( agx::Real32, 16 ) tmp[4];
    _mm_store_ps( tmp, xyzw );
    return tmp[1];
#else
    return xyzw.y();
#endif
  }


  AGX_FORCE_INLINE agx::Real32 Vec3T<agx::Real32>::z() const
  {
#if AGX_USE_SSE()
    AGX_ALIGNED( agx::Real32, 16 ) tmp[4];
    _mm_store_ps( tmp, xyzw );
    return tmp[2];
#else
    return xyzw.z();
#endif
  }

  AGX_FORCE_INLINE void Vec3T<agx::Real32>::store(agx::Vec3f& target) const
  {
#if AGX_USE_SSE()
    _mm_store_ps(target.ptr(), xyzw);
#else
    target = xyzw;
#endif
  }

  AGX_FORCE_INLINE Vec3f Vec3T<agx::Real32>::absolute() const
  {
#if AGX_USE_SSE()
    const __m128 SIGNMASK = _mm_set1_ps(-0.0f); // -0.0f = 1 << 31 // _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return Vec3T(_mm_andnot_ps(xyzw, SIGNMASK));
#else
    return Vec3T(std::abs(xyzw[0]),
                 std::abs(xyzw[1]),
                 std::abs(xyzw[2]));
#endif
  }

  AGX_FORCE_INLINE Vec3f Vec3T<agx::Real32>::negate() const
  {
#if AGX_USE_SSE()
    const __m128 SIGNMASK = _mm_set1_ps(-0.0f); // -0.0f = 1 << 31 // _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return Vec3T(_mm_xor_ps(xyzw, SIGNMASK));
#else
    return Vec3T(-xyzw);
#endif
  }

  AGX_FORCE_INLINE Vec3f Vec3T<agx::Real32>::scale(const Vec3T& val) const
  {
#if AGX_USE_SSE()
    return Vec3T(_mm_mul_ps(xyzw, val.xyzw));
#else
    return Vec3T(agx::Vec3f::mul(xyzw, val.xyzw));
#endif
  }


#if AGX_USE_SSE()
  AGX_FORCE_INLINE Vec3f Vec3T<agx::Real32>::scale(const __m128 val) const
  {
    return Vec3T(_mm_mul_ps(xyzw, val));
  }
#endif

  AGX_FORCE_INLINE Vec3f Vec3T<agx::Real32>::scale(agx::Real32 val) const
  {
#if AGX_USE_SSE()
    return scale(_mm_set1_ps(val));
#else
    return Vec3T(xyzw * val);
#endif
  }

  AGX_FORCE_INLINE Vec3f Vec3T<agx::Real32>::madd(const Vec3T& a, const Vec3T& b, const Vec3T& c)
  {
#if AGX_USE_SSE()
    return Vec3T( _mm_add_ps( a.xyzw, _mm_mul_ps( b.xyzw, c.xyzw ) ) );
#else
    return Vec3T(a.xyzw + agx::Vec3f::mul(b.xyzw, c.xyzw));
#endif
  }

  AGX_FORCE_INLINE Vec3f Vec3T<agx::Real32>::cross(const Vec3T& lhs, const Vec3T& rhs)
  {
#if AGX_USE_SSE()
    const agx::UInt32 shuffle_3021 = _MM_SHUFFLE(3, 0, 2, 1);
    const agx::UInt32 shuffle_3102 = _MM_SHUFFLE(3, 1, 0, 2);

    const __m128 a = _mm_mul_ps(_mm_shuffle_ps(lhs.xyzw, lhs.xyzw, shuffle_3021), _mm_shuffle_ps(rhs.xyzw, rhs.xyzw, shuffle_3102));
    const __m128 b = _mm_mul_ps(_mm_shuffle_ps(lhs.xyzw, lhs.xyzw, shuffle_3102), _mm_shuffle_ps(rhs.xyzw, rhs.xyzw, shuffle_3021));

    #ifdef AGX_DEBUG
    AGX_ALIGNED( agx::Real32, 16 ) tmp[4];
    _mm_store_ps( tmp, _mm_sub_ps(a,b) );
    agxAssert(tmp[3] == 0);
    #endif

    return Vec3T(_mm_sub_ps(a,b));
#else
    return Vec3T(lhs.xyzw ^ rhs.xyzw);
#endif
  }

  AGX_FORCE_INLINE agx::Real32 Vec3T<agx::Real32>::dot(const Vec3T& lhs, const Vec3T& rhs)
  {
#if AGX_USE_SSE4()
    const int mask = 0x71;
    AGX_ALIGNED( agx::Real32, 16 ) output[4];
    _mm_store_ps( output, _mm_dp_ps(lhs.xyzw, rhs.xyzw, mask) );
    return output[0];
#elif AGX_USE_SSE()
    AGX_ALIGNED( agx::Real32, 16 ) scalarLhs[4];
    AGX_ALIGNED( agx::Real32, 16 ) scalarRhs[4];

    _mm_store_ps( scalarLhs, lhs.xyzw );
    _mm_store_ps( scalarRhs, rhs.xyzw );

    return scalarLhs[0] * scalarRhs[0] + scalarLhs[1] * scalarRhs[1] + scalarLhs[2] * scalarRhs[2];
#else
    return lhs.xyzw * rhs.xyzw;
#endif
  }

  AGX_FORCE_INLINE agx::Real32 Vec3T<agx::Real32>::length2() const
  {
#if AGX_USE_SSE()
    return dot(*this, *this);
#else
    return (agx::Real32)xyzw.length2();
#endif
  }

  AGX_FORCE_INLINE agx::Real32 Vec3T<agx::Real32>::length() const
  {
#if AGX_USE_SSE()

  #if AGX_USE_SSE4()
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(xyzw, xyzw, 0x71)));
  #else
    return std::sqrt(length2());
  #endif

#else
    return (agx::Real32)xyzw.length();
#endif
  }


  AGX_FORCE_INLINE agx::Real32 Vec3T<agx::Real32>::innerProduct(const Vec3T& v01, const Vec3T& v02, const Vec3T& v11, const Vec3T& v12)
  {
#if AGX_USE_SSE()
    AGX_ALIGNED( agx::Real32, 16 ) output[4];

    __m128 tmp1 = _mm_mul_ps( v01.xyzw, v11.xyzw );
    __m128 tmp2 = _mm_mul_ps( v02.xyzw, v12.xyzw );

    tmp1 = _mm_add_ps( tmp1, tmp2 );

    _mm_store_ps( output, tmp1 );

    return output[0]+output[1]+output[2];
#else
    return v01.xyzw * v11.xyzw + v02.xyzw * v12.xyzw;
#endif
  }



  /////////////////////////////////////////////////////////////////////////////////////////////////////

#if AGX_USE_AVX()

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T() : Vec3T(0.0)
  {}

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(agx::Real64 val) : xyzw(_mm256_setr_pd(val, val, val, 0))
  {}

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(agx::Real64 x, agx::Real64 y, agx::Real64 z) : xyzw(_mm256_setr_pd(x, y, z, 0))
  {
  }

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(const __m256d _xyzw) : xyzw(_xyzw)
  {}

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(const agx::Vec3d& vec) : xyzw(_mm256_load_pd(vec.ptr()))
  {}

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(const agx::Vec3f& vec) : xyzw(_mm256_cvtps_pd(_mm_load_ps(vec.ptr())))
  {}

#elif AGX_USE_SSE()

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T()
  {}

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(agx::Real64 val) : xy(_mm_set1_pd(val)), zw(_mm_setr_pd(val, 0))
  {}

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(agx::Real64 x, agx::Real64 y, agx::Real64 z) : xy(_mm_setr_pd(x, y)), zw(_mm_setr_pd(z, 0))
  {
  }


  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(const __m128d _xy, const __m128d _zw) : xy(_xy), zw(_zw)
  {}

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(const agx::Vec3d& vec) : xy(_mm_load_pd(vec.ptr())), zw(_mm_load_pd(vec.ptr()+2))
  {}

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(const agx::Vec3f& vec)
  {
    const __m128 lower = _mm_load_ps(vec.ptr());
    xy = _mm_cvtps_pd(lower);

    const __m128 upper = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(lower), 8));
    zw = _mm_cvtps_pd(upper);
  }

#else

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T()
  {}

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(agx::Real64 val) : xyzw(val)
  {}

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(agx::Real64 x, agx::Real64 y, agx::Real64 z) : xyzw(x, y, z)
  {
  }

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(const agx::Vec3d& vec) : xyzw(vec)
  {}

  AGX_FORCE_INLINE Vec3T<agx::Real64>::Vec3T(const agx::Vec3f& vec) : xyzw(vec)
  {}

#endif

  AGX_FORCE_INLINE agx::Real64 Vec3T<agx::Real64>::x() const
  {
#if AGX_USE_AVX()
    AGX_ALIGNED( agx::Real64, 32 ) tmp[4];
    _mm256_store_pd( tmp, xyzw );
    return tmp[0];
#elif AGX_USE_SSE()
    AGX_ALIGNED( agx::Real64, 16 ) tmp[2];
    _mm_store_pd( tmp, xy );
    return tmp[0];
#else
    return xyzw.x();
#endif
  }

  AGX_FORCE_INLINE agx::Real64 Vec3T<agx::Real64>::y() const
  {
#if AGX_USE_AVX()
    AGX_ALIGNED( agx::Real64, 32 ) tmp[4];
    _mm256_store_pd( tmp, xyzw );
    return tmp[1];
#elif AGX_USE_SSE()
    AGX_ALIGNED( agx::Real64, 16 ) tmp[2];
    _mm_store_pd( tmp, xy );
    return tmp[1];
#else
    return xyzw.y();
#endif
  }

  AGX_FORCE_INLINE agx::Real64 Vec3T<agx::Real64>::z() const
  {
#if AGX_USE_AVX()
    AGX_ALIGNED( agx::Real64, 32 ) tmp[4];
    _mm256_store_pd( tmp, xyzw );
    return tmp[2];
#elif AGX_USE_SSE()
    AGX_ALIGNED( agx::Real64, 16 ) tmp[2];
    _mm_store_pd( tmp, zw );
    return tmp[0];
#else
    return xyzw.z();
#endif
  }

  AGX_FORCE_INLINE void Vec3T<agx::Real64>::store(agx::Vec3d& target) const
  {
#if AGX_USE_AVX()
    _mm256_store_pd(target.ptr(), xyzw);
#elif AGX_USE_SSE()
    _mm_store_pd(target.ptr()  , xy);
    _mm_store_pd(target.ptr()+2, zw);
#else
    target = xyzw;
#endif
  }

  AGX_FORCE_INLINE void Vec3T<agx::Real64>::store(agx::Vec3f& target) const
  {
#if AGX_USE_AVX()
    _mm_store_ps(target.ptr(), _mm256_cvtpd_ps(xyzw));
#elif AGX_USE_SSE()
    __m128 lower = _mm_cvtpd_ps(xy);
    __m128 upper = _mm_cvtpd_ps(zw);
    __m128 result = _mm_or_ps(lower, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(upper), 8)));
    _mm_store_ps(target.ptr(), result);
#else
    target = agx::Vec3f((float)xyzw[0], (float)xyzw[1], (float)xyzw[2]);
#endif
  }

  AGX_FORCE_INLINE Vec3d Vec3T<agx::Real64>::absolute() const
  {
#if AGX_USE_AVX()
    const __m256d SIGNMASK = _mm256_set1_pd(-0.0); // -0.0 = 1 << 63
    return Vec3T(_mm256_andnot_pd(xyzw, SIGNMASK));
#elif AGX_USE_SSE()
    const __m128d SIGNMASK = _mm_set1_pd(-0.0); // -0.0 = 1 << 63
    return Vec3T(_mm_andnot_pd(xy, SIGNMASK), _mm_andnot_pd(zw, SIGNMASK));
#else
    return Vec3T(std::abs(xyzw[0]),
                 std::abs(xyzw[1]),
                 std::abs(xyzw[2]));
#endif
  }

  AGX_FORCE_INLINE Vec3d Vec3T<agx::Real64>::negate() const
  {
#if AGX_USE_AVX()
    const __m256d SIGNMASK = _mm256_set1_pd(-0.0); // -0.0 = 1 << 63
    return Vec3T(_mm256_xor_pd(xyzw, SIGNMASK));
#elif AGX_USE_SSE()
    const __m128d SIGNMASK = _mm_set1_pd(-0.0); // -0.0 = 1 << 63
    return Vec3T(_mm_xor_pd(xy, SIGNMASK), _mm_xor_pd(zw, SIGNMASK));
#else
    return Vec3T(-xyzw);
#endif
  }

  AGX_FORCE_INLINE Vec3d Vec3T<agx::Real64>::scale(const Vec3T& val) const
  {
#if AGX_USE_AVX()
    return Vec3T(_mm256_mul_pd(xyzw, val.xyzw));
#elif AGX_USE_SSE()
    return Vec3T(_mm_mul_pd(xy, val.xy), _mm_mul_pd(zw, val.zw));
#else
    return Vec3T(agx::Vec3d::mul(xyzw, val.xyzw));
#endif
  }

#if AGX_USE_SSE() && !AGX_USE_AVX()
  AGX_FORCE_INLINE Vec3d Vec3T<agx::Real64>::scale(const __m128d val) const
  {
    return Vec3T(_mm_mul_pd(xy, val), _mm_mul_pd(zw, val));
  }
#endif

  AGX_FORCE_INLINE Vec3d Vec3T<agx::Real64>::scale(agx::Real64 val) const
  {
#if AGX_USE_AVX()
    return scale(Vec3T(val));
#elif AGX_USE_SSE()
    return scale(_mm_set1_pd(val));
#else
    return Vec3T(xyzw * val);
#endif
  }

  AGX_FORCE_INLINE Vec3d Vec3T<agx::Real64>::madd(const Vec3T& a, const Vec3T& b, const Vec3T& c)
  {
#if AGX_USE_AVX()
    return Vec3T( _mm256_add_pd( a.xyzw, _mm256_mul_pd( b.xyzw, c.xyzw ) ) );
#elif AGX_USE_SSE()
    const __m128d xy = _mm_add_pd( a.xy, _mm_mul_pd( b.xy, c.xy ) );
    const __m128d zw = _mm_add_pd( a.zw, _mm_mul_pd( b.zw, c.zw ) );

    return Vec3T(xy, zw);
#else
    return Vec3T(a.xyzw + agx::Vec3d::mul(b.xyzw, c.xyzw));
#endif
  }


  AGX_FORCE_INLINE Vec3d Vec3T<agx::Real64>::cross(const Vec3T& lhs, const Vec3T& rhs)
  {
#if AGX_USE_AVX()
#error "This needs to be fixed!"
    // NOTE: Disabled, permute256 semantics are weird :/
    const agx::UInt32 shuffle_3021 = _MM_SHUFFLE(3, 0, 2, 1);
    const agx::UInt32 shuffle_3102 = _MM_SHUFFLE(3, 1, 0, 2);

    const __m256d a = _mm256_mul_pd(_mm256_shuffle_pd(lhs.xyzw, lhs.xyzw, shuffle_3021), _mm256_shuffle_pd(rhs.xyzw, rhs.xyzw, shuffle_3102));
    const __m256d b = _mm256_mul_pd(_mm256_shuffle_pd(lhs.xyzw, lhs.xyzw, shuffle_3102), _mm256_shuffle_pd(rhs.xyzw, rhs.xyzw, shuffle_3021));

    #ifdef AGX_DEBUG
    AGX_ALIGNED( agx::Real64, 32 ) tmp[4];
    _mm256_store_pd( tmp, _mm256_sub_pd(a,b) );
    agxAssert(tmp[3] == 0);
    #endif

    return Vec3T(_mm256_sub_pd(a,b));
#elif AGX_USE_SSE()
    const __m128d SIGN_NP = _mm_set_pd ( 0.0 , -0.0 );


    // lhs.z * rhs.x, lhs.z * rhs.y
    __m128d l1 = _mm_mul_pd ( _mm_unpacklo_pd ( lhs.zw , lhs.zw ) , rhs.xy );

    // rhs.z * lhs.x, rhs.z * lhs.y
    __m128d l2 = _mm_mul_pd ( _mm_unpacklo_pd ( rhs.zw , rhs.zw ) , lhs.xy );
    __m128d m1 = _mm_sub_pd ( l1 , l2 ); // l1 - l2
    m1 = _mm_shuffle_pd ( m1 , m1 , 1 ); // switch the elements
    m1 = _mm_xor_pd ( m1 , SIGN_NP ); // change the sign of the first element

    // lhs.x * rhs.y, lhs.y * rhs.x
    l1 = _mm_mul_pd ( lhs.xy , _mm_shuffle_pd ( rhs.xy , rhs.xy , 1 ) );
    // lhs.x * rhs.y - lhs.y * rhs.x
    __m128d m2 = _mm_sub_sd ( l1 , _mm_unpackhi_pd ( l1 , l1 ) );

    // Clear w-component
    m2 = _mm_move_sd(_mm_setzero_pd(), m2);

    return Vec3T(m1, m2);
#else
    return Vec3T(lhs.xyzw ^ rhs.xyzw);
#endif
  }

  AGX_FORCE_INLINE agx::Real64 Vec3T<agx::Real64>::dot(const Vec3T& lhs, const Vec3T& rhs)
  {
#if AGX_USE_AVX()
    // NOTE: No _mm256_dp_pd instruction available!
    // const int mask = 0x71;
    // AGX_ALIGNED( agx::Real64, 32 ) output[4];
    // _mm256_store_pd( output, _mm256_dp_pd(lhs.xyzw, rhs.xyzw, mask) );
    // return output[0];

    #if 1
    __m128d lhsXY = _mm256_extractf128_pd( lhs.xyzw, 0 );
    __m128d lhsZW = _mm256_extractf128_pd( lhs.xyzw, 1 );
    __m128d rhsXY = _mm256_extractf128_pd( rhs.xyzw, 0 );
    __m128d rhsZW = _mm256_extractf128_pd( rhs.xyzw, 1 );

    const int mask = 0x31;
    AGX_ALIGNED( agx::Real64, 16 ) output[2];

    const __m128d xyDot = _mm_dp_pd(lhsXY, rhsXY, mask);
    const __m128d zDot = _mm_mul_pd(lhsZW, rhsZW);

    _mm_store_pd( output, _mm_add_pd(xyDot, zDot));
    return output[0];

    #else
    __m256d a = _mm256_mul_pd( lhs.xyzw, rhs.xyzw );
    __m256d b = _mm256_hadd_pd( a, a );
    __m128d hi128 = _mm256_extractf128_pd( b, 1 );
    __m128d dotproduct = _mm_add_pd( (__m128d)b, hi128 );

    AGX_ALIGNED( agx::Real64, 16 ) output[2];
    _mm_store_pd( output, dotproduct );
    return output[0];
    #endif

#elif AGX_USE_SSE4()
    const int mask = 0x31;
    AGX_ALIGNED( agx::Real64, 16 ) output[2];

    const __m128d xyDot = _mm_dp_pd(lhs.xy, rhs.xy, mask);
    const __m128d zDot = _mm_mul_pd(lhs.zw, rhs.zw);

    _mm_store_pd( output, _mm_add_pd(xyDot, zDot));
    return output[0];
#elif AGX_USE_SSE()
    AGX_ALIGNED( agx::Real64, 16 ) scalarLhs[4];
    AGX_ALIGNED( agx::Real64, 16 ) scalarRhs[4];

    _mm_store_pd( scalarLhs, lhs.xy );
    _mm_store_pd( scalarLhs + 2, lhs.zw );
    _mm_store_pd( scalarRhs, rhs.xy );
    _mm_store_pd( scalarRhs + 2, rhs.zw );

    return scalarLhs[0] * scalarRhs[0] + scalarLhs[1] * scalarRhs[1] + scalarLhs[2] * scalarRhs[2];
#else
    return lhs.xyzw * rhs.xyzw;
#endif
  }

  AGX_FORCE_INLINE agx::Real64 Vec3T<agx::Real64>::length2() const
  {
    return dot(*this, *this);
  }

  AGX_FORCE_INLINE agx::Real64 Vec3T<agx::Real64>::length() const
  {
#if AGX_USE_AVX() && 0
    // NOTE: No _mm256_dp_pd instruction available!
    AGX_ALIGNED( agx::Real64, 32 ) output[4];
    _mm256_store_pd(output, _mm256_sqrt_pd(_mm256_dp_pd(xyzw, xyzw, 0x71)));
    return output[0];
#elif AGX_USE_SSE()
    return std::sqrt(this->length2());
#else
    return (agx::Real64)xyzw.length();
#endif
  }


  AGX_FORCE_INLINE agx::Real64 Vec3T<agx::Real64>::innerProduct(const Vec3T& v01, const Vec3T& v02, const Vec3T& v11, const Vec3T& v12)
  {
#if AGX_USE_AVX()

    AGX_ALIGNED( agx::Real64, 32 ) output[4];

    __m256d tmp1 = _mm256_mul_pd( v01.xyzw, v11.xyzw );
    __m256d tmp2 = _mm256_mul_pd( v02.xyzw, v12.xyzw );

    tmp1 = _mm256_add_pd( tmp1, tmp2 );

    _mm256_store_pd( output, tmp1 );

    return output[0]+output[1]+output[2];

#elif AGX_USE_SSE()

    // A0     B0         A0 * B0
    // A1     B1         A1 * B1        A0 * B0 + A2 * B2
    // ---------  mul => ------- add => ----------------- hadd => A0 * B0 + A2 * B2 + A1 * B1 + A3 * B3
    // A2     B2         A2 * B2        A1 * B1 + A3 * B3
    // A3     B3         A3 * B3

    __m128d tmp1 = _mm_mul_pd( v01.xy, v11.xy );
    __m128d tmp2 = _mm_mul_pd( v01.zw, v11.zw );

    __m128d tmp3 = _mm_mul_pd( v02.xy, v12.xy );
    __m128d tmp4 = _mm_mul_pd( v02.zw, v12.zw );


    tmp1 = _mm_add_pd( tmp1, tmp2 );
    tmp3 = _mm_add_pd( tmp3, tmp4 );

    AGX_ALIGNED( agx::Real64, 16 ) output[2];

    tmp1 = _mm_hadd_pd( tmp1, tmp3 );
    _mm_store_pd( output, tmp1 );

    return output[0] + output[1];
#else
    return v01.xyzw * v11.xyzw + v02.xyzw * v12.xyzw;
#endif
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////

  AGX_FORCE_INLINE std::ostream& operator << ( std::ostream& output, const Vec3d& v )
  {
    AGX_ALIGNED(agx::Vec3d, 32) tmp;
    v.store(tmp);
    return output << tmp;
  }

  AGX_FORCE_INLINE std::ostream& operator << ( std::ostream& output, const Vec3f& v )
  {
    AGX_ALIGNED(agx::Vec3f, 16) tmp;
    v.store(tmp);
    return output << tmp;
  }

}
DOXYGEN_END_INTERNAL_BLOCK()

#endif /* AGX_SIMD_VEC3_H */
