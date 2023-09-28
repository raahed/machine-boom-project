/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_JACOBIAN_H
#define AGX_JACOBIAN_H

#include <agx/config/AGX_USE_SSE.h>
#include <agx/config/AGX_USE_AVX.h>
#include <agx/config.h>
#include <agxData/Type.h>
#include <agx/Vec3.h>
#include <agx/Matrix3x3.h>
#include <agx/Range.h>

#if AGX_USE_SSE()
#  include <pmmintrin.h>
#endif

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agx
{

  AGX_FORCE_INLINE double innerProd_6( const Vec3d& v_01, const Vec3d& v_02, const Vec3d& v_11, const Vec3d& v_12 )
  {
#if AGX_USE_SSE()
    const double* v_01Ptr = v_01.ptr();
    const double* v_11Ptr = v_11.ptr();

    const double* v_02Ptr = v_02.ptr();
    const double* v_12Ptr = v_12.ptr();

    // A0     B0         A0 * B0
    // A1     B1         A1 * B1        A0 * B0 + A2 * B2
    // ---------  mul => ------- add => ----------------- hadd => A0 * B0 + A2 * B2 + A1 * B1 + A3 * B3
    // A2     B2         A2 * B2        A1 * B1 + A3 * B3
    // A3     B3         A3 * B3

    __m128d tmp1 = _mm_mul_pd( _mm_load_pd( v_01Ptr + 0 ), _mm_load_pd( v_11Ptr + 0 ) );
    __m128d tmp2 = _mm_mul_pd( _mm_load_pd( v_01Ptr + 2 ), _mm_load_pd( v_11Ptr + 2 ) );

    __m128d tmp3 = _mm_mul_pd( _mm_load_pd( v_02Ptr + 0 ), _mm_load_pd( v_12Ptr + 0 ) );
    __m128d tmp4 = _mm_mul_pd( _mm_load_pd( v_02Ptr + 2 ), _mm_load_pd( v_12Ptr + 2 ) );

    tmp1 = _mm_add_pd( tmp1, tmp2 );
    tmp3 = _mm_add_pd( tmp3, tmp4 );

    AGX_ALIGNED( double, 16 ) output[2];

    tmp1 = _mm_hadd_pd( tmp1, tmp3 );
    _mm_store_pd( output, tmp1 );

    return output[0] + output[1];
#else
    return v_01 * v_11 + v_02 * v_12;
#endif
  }

  AGX_FORCE_INLINE float innerProd_6( const Vec3f& v_01, const Vec3f& v_02, const Vec3f& v_11, const Vec3f& v_12 )
  {
#if AGX_USE_SSE()
    const float* v_01Ptr = v_01.ptr();
    const float* v_11Ptr = v_11.ptr();

    const float* v_02Ptr = v_02.ptr();
    const float* v_12Ptr = v_12.ptr();

    AGX_ALIGNED( float, 16 ) output[4];

    __m128 tmp1 = _mm_mul_ps( _mm_load_ps( v_01Ptr ), _mm_load_ps( v_11Ptr ) );
    __m128 tmp2 = _mm_mul_ps( _mm_load_ps( v_02Ptr ), _mm_load_ps( v_12Ptr ) );

    tmp1 = _mm_add_ps( tmp1, tmp2 );

    _mm_store_ps( output, tmp1 );

    return output[0]+output[1]+output[2];

    //return tmp1.m128_f32[ 0 ] + tmp1.m128_f32[ 1 ] + tmp1.m128_f32[ 2 ];
#else
    return v_01 * v_11 + v_02 * v_12;
#endif
  }


  template <typename T>
  class Jacobian6DOFElementT
  {
    public:
      typedef T Vec3T;

    public:
      /**
      Default constructor. All values are zeroed.
      */
      AGX_FORCE_INLINE Jacobian6DOFElementT() {}

      /**
      Copy constructor.
      */
      AGX_FORCE_INLINE Jacobian6DOFElementT( const Jacobian6DOFElementT& other )
      {
        *this = other;
      }

      /**
      Construct given spatial and rotational value.
      \param spat - vector containing spatial values
      \param rot - vector containing rotational values
      */
      explicit AGX_FORCE_INLINE Jacobian6DOFElementT( const T& spat, const T& rot )
      {
        m_data[ 0 ].set( spat );
        m_data[ 1 ].set( rot );
      }

      /**
      Assignment operator, lhs.spatial = rhs.spatial and lhs.rotational = rhs.rotational.
      */
      AGX_FORCE_INLINE Jacobian6DOFElementT& operator = ( const Jacobian6DOFElementT& other )
      {
        m_data[ 0 ] = other.m_data[ 0 ];
        m_data[ 1 ] = other.m_data[ 1 ];
        return *this;
      }

      /**
      \return spatial vector if index == 0, rotational vector if index == 1
      */
      AGX_FORCE_INLINE T& operator [] ( UInt i )
      {
        agxAssert( i < 2 );
        return m_data[ i ];
      }

      /**
      \return spatial vector if index == 0, rotational vector if index == 1
      */
      AGX_FORCE_INLINE const T& operator [] ( UInt i ) const
      {
        agxAssert( i < 2 );
        return m_data[ i ];
      }

      /**
      \return the spatial vector
      */
      AGX_FORCE_INLINE T& spatial()
      {
        return m_data[ 0 ];
      }

      /**
      \return the spatial vector
      */
      AGX_FORCE_INLINE const T& spatial() const
      {
        return m_data[ 0 ];
      }

      /**
      \return the rotational vector
      */
      AGX_FORCE_INLINE T& rotational()
      {
        return m_data[ 1 ];
      }

      /**
      \return the rotational vector
      */
      AGX_FORCE_INLINE const T& rotational() const
      {
        return m_data[ 1 ];
      }

      /**
      Multiplication operator, multiply this (lhs) with another element (rhs).
      */
      AGX_FORCE_INLINE Real operator * ( const Jacobian6DOFElementT& rhs ) const
      {
        return m_data[ 0 ] * rhs.m_data[ 0 ] + m_data[ 1 ] * rhs.m_data[ 1 ];
      }

      /**
      Multiplication operator, scale this (lhs) with scalar (rhs).
      */
      AGX_FORCE_INLINE Jacobian6DOFElementT operator * ( Real rhs ) const
      {
        return Jacobian6DOFElementT( m_data[ 0 ] * rhs, m_data[ 1 ] * rhs );
      }

      AGX_FORCE_INLINE Jacobian6DOFElementT& operator *= ( Real rhs )
      {
        m_data[ 0 ] *= rhs;
        m_data[ 1 ] *= rhs;
        return *this;
      }

      AGX_FORCE_INLINE Jacobian6DOFElementT operator / ( Real rhs ) const
      {
        return Jacobian6DOFElementT( m_data[ 0 ] / rhs, m_data[ 1 ] / rhs );
      }

      AGX_FORCE_INLINE Jacobian6DOFElementT& operator /= ( Real rhs )
      {
        m_data[ 0 ] /= rhs;
        m_data[ 1 ] /= rhs;
        return *this;
      }

      /**
      Scalar product between this and two other vectors. I.e., this * [v1 v2]^T.
      \param v1 - vector to be multiplied with spatial part
      \param v2 - vector to be multiplied with rotational part
      \return the dot product between spatial * v1 + rotational + v2
      */
      AGX_FORCE_INLINE Real mult( const T& v1, const T& v2 ) const
      {
        return innerProd_6( m_data[ 0 ], m_data[ 1 ], v1, v2 );
      }

      /**
      Assignment.
      \param v1 - new spatial part
      \param v2 - new rotational part
      */
      AGX_FORCE_INLINE void set( const T& v1, const T& v2 )
      {
        m_data[ 0 ] = v1;
        m_data[ 1 ] = v2;
      }

      /**
      Add \p v1 to spatial and \p v2 to rotational.
      \param v1 - vector to add to spatial part
      \param v2 - vector to add to rotational part
      */
      AGX_FORCE_INLINE void add( const T& v1, const T& v2 )
      {
        m_data[ 0 ] += v1;
        m_data[ 1 ] += v2;
      }

      AGX_FORCE_INLINE bool operator== (const Jacobian6DOFElementT& other)
      {
        return spatial() == other.spatial() && rotational() == other.rotational();
      }

    protected:
      T m_data[2];
  };

  typedef Jacobian6DOFElementT<Vec3d> Jacobian6DOFElement64;
  typedef Jacobian6DOFElementT<Vec3f> Jacobian6DOFElement32;
  typedef Jacobian6DOFElementT<Vec3> Jacobian6DOFElement;

  // Body-body
  template <typename T>
  class BodyBodyContactJacobianT
  {
  public:
    Jacobian6DOFElementT<T>& normal1();
    Jacobian6DOFElementT<T>& uTangent1();
    Jacobian6DOFElementT<T>& vTangent1();

    Jacobian6DOFElementT<T>& normal2();
    Jacobian6DOFElementT<T>& uTangent2();
    Jacobian6DOFElementT<T>& vTangent2();

    const Jacobian6DOFElementT<T>& normal1() const;
    const Jacobian6DOFElementT<T>& uTangent1() const;
    const Jacobian6DOFElementT<T>& vTangent1() const;

    const Jacobian6DOFElementT<T>& normal2() const;
    const Jacobian6DOFElementT<T>& uTangent2() const;
    const Jacobian6DOFElementT<T>& vTangent2() const;

  private:
    Jacobian6DOFElementT<T> m_body1[3];
    Jacobian6DOFElementT<T> m_body2[3];
  };

  // Particle-particle
  template <typename T>
  class ParticleParticleContactJacobianT
  {
  public:
    T& normal1();
    T& uTangent1();
    T& vTangent1();

    T& normal2();
    T& uTangent2();
    T& vTangent2();

    const T& normal1() const;
    const T& uTangent1() const;
    const T& vTangent1() const;

    const T& normal2() const;
    const T& uTangent2() const;
    const T& vTangent2() const;

  private:
    T m_particle1[3];
    T m_particle2[3];
  };

  // Particle-body
  template <typename T>
  class ParticleBodyContactJacobianT
  {
  public:
    T& normal1();
    T& uTangent1();
    T& vTangent1();

    Jacobian6DOFElementT<T>& normal2();
    Jacobian6DOFElementT<T>& uTangent2();
    Jacobian6DOFElementT<T>& vTangent2();

    const T& normal1() const;
    const T& uTangent1() const;
    const T& vTangent1() const;

    const Jacobian6DOFElementT<T>& normal2() const;
    const Jacobian6DOFElementT<T>& uTangent2() const;
    const Jacobian6DOFElementT<T>& vTangent2() const;

  private:
    T m_particle[3];
    Jacobian6DOFElementT<T> m_body[3];
  };

  typedef BodyBodyContactJacobianT<Vec3d> BodyBodyContactJacobian64;
  typedef BodyBodyContactJacobianT<Vec3f> BodyBodyContactJacobian32;
  typedef BodyBodyContactJacobianT<Vec3> BodyBodyContactJacobian;

  typedef ParticleParticleContactJacobianT<Vec3d> ParticleParticleContactJacobian64;
  typedef ParticleParticleContactJacobianT<Vec3f> ParticleParticleContactJacobian32;
  typedef ParticleParticleContactJacobianT<Vec3> ParticleParticleContactJacobian;

  typedef ParticleBodyContactJacobianT<Vec3d> ParticleBodyContactJacobian64;
  typedef ParticleBodyContactJacobianT<Vec3f> ParticleBodyContactJacobian32;
  typedef ParticleBodyContactJacobianT<Vec3> ParticleBodyContactJacobian;


  template <typename T>
  struct JacobianMetaT
  {
    T localPoint1;
    T localPoint2;

    Vec3f normal;
    Vec3f uTangent;
    Vec3f vTangent;

#if AGX_USE_AVX()
    Vec3f pad;
#endif
  };

  typedef JacobianMetaT<Vec3d> JacobianMeta64;
  typedef JacobianMetaT<Vec3f> JacobianMeta32;
  typedef JacobianMetaT<Vec3> JacobianMeta;

  template <typename T>
  AGX_FORCE_INLINE Real mult( const Jacobian6DOFElementT<T>& G1, const Jacobian6DOFElementT<T>& G2 )
  {
    return innerProd_6( G1.spatial(), G1.rotational(), G2.spatial(), G2.rotational() );
  }

  AGX_FORCE_INLINE Vec3f mult( const Matrix3x3f& m, const Vec3f& v )
  {
#if AGX_USE_SSE()
    const float* mPtr = m.ptr();
    const float* vPtr = v.ptr();
    __m128 r1 = _mm_load_ps( mPtr + 0 );
    __m128 r2 = _mm_load_ps( mPtr + 4 );
    __m128 r3 = _mm_load_ps( mPtr + 8 );

    __m128 v1 = _mm_load_ps( vPtr );

    r1 = _mm_mul_ps( r1, v1 );
    r2 = _mm_mul_ps( r2, v1 );
    r3 = _mm_mul_ps( r3, v1 );

    r1 = _mm_hadd_ps( r1, r2 );
    r3 = _mm_hadd_ps( r3, v1 );

    r1 = _mm_hadd_ps( r1, r3 );

    AGX_ALIGNED( Vec4f, 16 ) output;
    _mm_store_ps( output.ptr(), r1 );
    return Vec3f( output.ptr() );
#else
    return m * v;
#endif
  }


  // BodyBodyContactJacobianT
  template <typename T>
  AGX_FORCE_INLINE Jacobian6DOFElementT<T>& BodyBodyContactJacobianT<T>::normal1() { return m_body1[0]; }

  template <typename T>
  AGX_FORCE_INLINE Jacobian6DOFElementT<T>& BodyBodyContactJacobianT<T>::uTangent1() { return m_body1[1]; }

  template <typename T>
  AGX_FORCE_INLINE Jacobian6DOFElementT<T>& BodyBodyContactJacobianT<T>::vTangent1() { return m_body1[2]; }

  template <typename T>
  AGX_FORCE_INLINE Jacobian6DOFElementT<T>& BodyBodyContactJacobianT<T>::normal2() { return m_body2[0]; }

  template <typename T>
  AGX_FORCE_INLINE Jacobian6DOFElementT<T>& BodyBodyContactJacobianT<T>::uTangent2() { return m_body2[1]; }

  template <typename T>
  AGX_FORCE_INLINE Jacobian6DOFElementT<T>& BodyBodyContactJacobianT<T>::vTangent2() { return m_body2[2]; }


  template <typename T>
  AGX_FORCE_INLINE const Jacobian6DOFElementT<T>& BodyBodyContactJacobianT<T>::normal1() const { return m_body1[0]; }

  template <typename T>
  AGX_FORCE_INLINE const Jacobian6DOFElementT<T>& BodyBodyContactJacobianT<T>::uTangent1() const { return m_body1[1]; }

  template <typename T>
  AGX_FORCE_INLINE const Jacobian6DOFElementT<T>& BodyBodyContactJacobianT<T>::vTangent1() const { return m_body1[2]; }

  template <typename T>
  AGX_FORCE_INLINE const Jacobian6DOFElementT<T>& BodyBodyContactJacobianT<T>::normal2() const { return m_body2[0]; }

  template <typename T>
  AGX_FORCE_INLINE const Jacobian6DOFElementT<T>& BodyBodyContactJacobianT<T>::uTangent2() const { return m_body2[1]; }

  template <typename T>
  AGX_FORCE_INLINE const Jacobian6DOFElementT<T>& BodyBodyContactJacobianT<T>::vTangent2() const { return m_body2[2]; }


  // ParticleParticleContactJacobianT
  template <typename T>
  AGX_FORCE_INLINE T& ParticleParticleContactJacobianT<T>::normal1() { return m_particle1[0]; }

  template <typename T>
  AGX_FORCE_INLINE T& ParticleParticleContactJacobianT<T>::uTangent1() { return m_particle1[1]; }

  template <typename T>
  AGX_FORCE_INLINE T& ParticleParticleContactJacobianT<T>::vTangent1() { return m_particle1[2]; }

  template <typename T>
  AGX_FORCE_INLINE T& ParticleParticleContactJacobianT<T>::normal2() { return m_particle2[0]; }

  template <typename T>
  AGX_FORCE_INLINE T& ParticleParticleContactJacobianT<T>::uTangent2() { return m_particle2[1]; }

  template <typename T>
  AGX_FORCE_INLINE T& ParticleParticleContactJacobianT<T>::vTangent2() { return m_particle2[2]; }


  template <typename T>
  AGX_FORCE_INLINE const T& ParticleParticleContactJacobianT<T>::normal1() const { return m_particle1[0]; }

  template <typename T>
  AGX_FORCE_INLINE const T& ParticleParticleContactJacobianT<T>::uTangent1() const { return m_particle1[1]; }

  template <typename T>
  AGX_FORCE_INLINE const T& ParticleParticleContactJacobianT<T>::vTangent1() const { return m_particle1[2]; }

  template <typename T>
  AGX_FORCE_INLINE const T& ParticleParticleContactJacobianT<T>::normal2() const { return m_particle2[0]; }

  template <typename T>
  AGX_FORCE_INLINE const T& ParticleParticleContactJacobianT<T>::uTangent2() const { return m_particle2[1]; }

  template <typename T>
  AGX_FORCE_INLINE const T& ParticleParticleContactJacobianT<T>::vTangent2() const { return m_particle2[2]; }


  // ParticleBodyContactJacobianT
  template <typename T>
  AGX_FORCE_INLINE T& ParticleBodyContactJacobianT<T>::normal1() { return m_particle[0]; }

  template <typename T>
  AGX_FORCE_INLINE T& ParticleBodyContactJacobianT<T>::uTangent1() { return m_particle[1]; }

  template <typename T>
  AGX_FORCE_INLINE T& ParticleBodyContactJacobianT<T>::vTangent1() { return m_particle[2]; }

  template <typename T>
  AGX_FORCE_INLINE Jacobian6DOFElementT<T>& ParticleBodyContactJacobianT<T>::normal2() { return m_body[0]; }

  template <typename T>
  AGX_FORCE_INLINE Jacobian6DOFElementT<T>& ParticleBodyContactJacobianT<T>::uTangent2() { return m_body[1]; }

  template <typename T>
  AGX_FORCE_INLINE Jacobian6DOFElementT<T>& ParticleBodyContactJacobianT<T>::vTangent2() { return m_body[2]; }


  template <typename T>
  AGX_FORCE_INLINE const T& ParticleBodyContactJacobianT<T>::normal1() const { return m_particle[0]; }

  template <typename T>
  AGX_FORCE_INLINE const T& ParticleBodyContactJacobianT<T>::uTangent1() const { return m_particle[1]; }

  template <typename T>
  AGX_FORCE_INLINE const T& ParticleBodyContactJacobianT<T>::vTangent1() const { return m_particle[2]; }

  template <typename T>
  AGX_FORCE_INLINE const Jacobian6DOFElementT<T>& ParticleBodyContactJacobianT<T>::normal2() const { return m_body[0]; }

  template <typename T>
  AGX_FORCE_INLINE const Jacobian6DOFElementT<T>& ParticleBodyContactJacobianT<T>::uTangent2() const { return m_body[1]; }

  template <typename T>
  AGX_FORCE_INLINE const Jacobian6DOFElementT<T>& ParticleBodyContactJacobianT<T>::vTangent2() const { return m_body[2]; }

}

AGX_TYPE_BINDING(agx::Jacobian6DOFElement32, "Jacobian6DOFElement")
AGX_TYPE_BINDING(agx::Jacobian6DOFElement64, "Jacobian6DOFElement")
AGX_TYPE_BINDING(agx::RangeReal, "RangeReal")
AGX_TYPE_BINDING(agx::BodyBodyContactJacobian32, "BodyBodyContactJacobian")
AGX_TYPE_BINDING(agx::BodyBodyContactJacobian64, "BodyBodyContactJacobian")
AGX_TYPE_BINDING(agx::ParticleParticleContactJacobian32, "ParticleParticleContactJacobian")
AGX_TYPE_BINDING(agx::ParticleParticleContactJacobian64, "ParticleParticleContactJacobian")
AGX_TYPE_BINDING(agx::ParticleBodyContactJacobian32, "ParticleBodyContactJacobian")
AGX_TYPE_BINDING(agx::ParticleBodyContactJacobian64, "ParticleBodyContactJacobian")

AGX_TYPE_BINDING(agx::JacobianMeta32, "JacobianMeta")
AGX_TYPE_BINDING(agx::JacobianMeta64, "JacobianMeta")

#ifdef _MSC_VER
# pragma warning(pop)
#endif


#endif /* AGX_JACOBIAN_H */
