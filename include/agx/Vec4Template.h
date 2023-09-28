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

#ifndef AGX_VEC4_TEMPLATE_H
#define AGX_VEC4_TEMPLATE_H

#include <agx/agxPhysics_export.h>

#include <iosfwd>

#include <agx/agx.h>
#include <agx/Vec3.h>
#include <agx/Math.h>

namespace agx
{
  /**
  A class holding 4 dimensional vectors and providing basic arithmetic.
  */
  template <typename T>
  class Vec4T
  {
    public:
      typedef T Type;

    public:

      /// Copy constructor
      Vec4T(const Vec4T& copy ) = default;

      /// Copy constructor for other types.
      template<typename T2>
      explicit Vec4T(const Vec4T<T2>& copy );

      /**
      Default constructor
      */
      Vec4T();

      /// Constructor, fill all elements with scalar \p r
      explicit Vec4T( T r );

      /// Constructor, initialize elements with the specified scalars
      Vec4T( T x, T y, T z, T w );

      /// Constructor, initialize elements with the specified scalars
      explicit Vec4T(const T v[4] );

      /// Constructor, initialize the first three elements with \p v3 and the last with \p w
      Vec4T( const Vec3T<T>& v3, T w );

      /// \return a random Vec4 in the range of [\p min..\p max]
      static Vec4T random(T min, T max);

      /// \return a random Vec4 in the range of [\p min..\p max]
      static Vec4T random(const Vec4T& min, const Vec4T& max);

      /**
      Equality test
      */
      bool operator == ( const Vec4T& v ) const;

      /**
      In-equality test
      */
      bool operator != ( const Vec4T& v ) const;

      /**
      \return a new vector where each component is the minimum of this and the other vector.
      */
      static Vec4T componentMin(const Vec4T& v1, const Vec4T& v2);

      /**
      \return a new vector where each component is the maximum of this and the other vector.
      */
      static Vec4T componentMax(const Vec4T& v1, const Vec4T& v2);

      /**
      \return the smallest component (in absolute value).
      */
      T minComponent() const;

      /**
      \return the largest component (in absolute value).
      */
      T maxComponent() const;

      /**
      \return the index of the smallest element (in absolute value)
      */
      size_t minElement() const;

      /**
      \return the index of the largest element (in absolute value)
      */
      size_t maxElement() const;

      /**
      Clamp a vector between a lower and upper bound (per component).
      */
      void clamp(const Vec4T& min, const Vec4T& max);

      /**
      \return true if all elements are zero
      */
      bool equalsZero() const;

      /**
      \return a pointer to the data vector
      */
      T* ptr();

      /**
      \return a const pointer to the data vector
      */
      const T* ptr() const;

      /// Set the elements of the vector
      void set( T x, T y, T z, T w );

      /// Set all elements of the vector with scalar value \p value
      void set( T value );

      /// Set the value of the vector with \p rhs
      void set( const Vec4T& rhs );

      /// \return a reference to the i:th element
      T& operator [] ( size_t i );

      /// \return a reference to the i:th element
      const T& operator [] ( size_t i ) const;

      /// \return a reference to the 1:st element
      T& x();

      /// \return a reference to the 2:nd element
      T& y();

      /// \return a reference to the 3:rd element
      T& z();

      /// \return a reference to the 4:th element
      T& w();

      /// \return the 1:st element
      T x() const;

      /// \return the 2:nd element
      T y() const;

      /// \return the 3:rd element
      T z() const;

      /// \return the 4:th element
      T w() const;

      /// \return true if all elements is valid
      bool isValid() const;

      /// \return true if any of the elements is NAN
      bool isNaN() const;

      /// \return true if all of the elements is are non-NaN and non-inf
      bool isFinite() const;

      /**
      \return the Dot product.
      */
      T operator * ( const Vec4T& rhs ) const;

      /**
      \return the result of an Element-wise-multiplication
      */
      const Vec4T operator | ( const Vec4T& rhs ) const;

      /**
      \return a new Vec4 multiplied by scalar \p rhs.
      */
      const Vec4T operator * ( T rhs ) const;

      /**
      \return a reference to the Vec4 Unary multiplied by the scalar \p rhs
      */
      Vec4T& operator *= ( T rhs );

      /**
      Divide by scalar.
      */
      const Vec4T operator / ( T rhs ) const;

      /**
      \return reference to the Vec4 Unary divided by scalar \p rhs.
      */
      Vec4T& operator /= ( T rhs );

      /**
      Binary vector add.
      \return a new Vec4 which is the sum of this Vec4 and \p rhs
      */
      const Vec4T operator + ( const Vec4T& rhs ) const;

      /**
      Unary vector add. Slightly more efficient because no temporary
      intermediate object.
      */
      Vec4T& operator += ( const Vec4T& rhs );

      /**
      Binary vector subtract.
      */
      const Vec4T operator - ( const Vec4T& rhs ) const;

      /**
      Unary vector subtract.
      */
      Vec4T& operator -= ( const Vec4T& rhs );

      /**
      Binary vector add.
      */
      const Vec4T operator + ( const T& rhs ) const;

      /**
      Unary vector add. Slightly more efficient because no temporary
      intermediate object.
      */
      Vec4T& operator += ( const T& rhs );

      /**
      Binary vector subtract.
      */
      const Vec4T operator - ( const T& rhs ) const;

      /**
      Unary vector subtract.
      */
      Vec4T& operator -= ( const T& rhs );

      /**
      Negation operator. Returns the negative of the Vec4T.
      */
      const Vec4T operator - () const;

      /**
      \return length of the vector = sqrt( vec . vec )
      */
      Real length() const;

      /**
      \return spared length of the vector = vec . vec
      */
      Real length2() const;

      /**
      \return squared distance to another vector
      */
      Real distance2(const Vec4T& v2) const;

      /**
      \return Distance to another vector
      */
      Real distance(const Vec4T& v2) const;

      /**
      Normalize the vector so that it has length unity.
      \return the previous length of the vector.
      */
      Real normalize();

      /**
      \return the x, y and z components as a Vec3
      */
      Vec3T< T > asVec3() const;

      /// \return the element at i. Mainly for accessing vec4 in Lua.
      T at( size_t i );

      /// Set the value of element i. Mainly for accessing vec4 in Lua.
      void set( T val, size_t i );

    private:
      T m_data[4];
  };    // end of class Vec4T_template


  /* Implementation */


  template <typename T>
  template<typename T2>
  AGX_FORCE_INLINE Vec4T<T>::Vec4T(const Vec4T<T2>& copy )
  {
    //memcpy(&m_data, copy.m_data, sizeof(agx::Real)*3);
    m_data[ 0 ] = (T)copy[ 0 ];
    m_data[ 1 ] = (T)copy[ 1 ];
    m_data[ 2 ] = (T)copy[ 2 ];
    m_data[ 3 ] = (T)copy[ 3 ];
  }


  template <typename T>
  AGX_FORCE_INLINE Vec4T<T>::Vec4T()
  {
    m_data[0] = T();
    m_data[1] = T();
    m_data[2] = T();
    m_data[3] = T();
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T>::Vec4T( T r )
  {
    m_data[0] = m_data[1] = m_data[2] = m_data[3] = r;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T>::Vec4T( T x, T y, T z, T w )
  {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
    m_data[3] = w;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T>::Vec4T(const T v[4] )
  {
    m_data[0] = v[0];
    m_data[1] = v[1];
    m_data[2] = v[2];
    m_data[3] = v[3];
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T>::Vec4T( const Vec3T<T>& v3, T w )
  {
    m_data[0] = T(v3[0]);
    m_data[1] = T(v3[1]);
    m_data[2] = T(v3[2]);
    m_data[3] = w;
  }


  template <typename T>
  AGX_FORCE_INLINE bool Vec4T<T>::operator == ( const Vec4T<T>& v ) const
  {
    return m_data[0] == v.m_data[0] && m_data[1] == v.m_data[1] && m_data[2] == v.m_data[2] && m_data[3] == v.m_data[3];
  }

  template <typename T>
  AGX_FORCE_INLINE bool Vec4T<T>::operator != ( const Vec4T<T>& v ) const
  {
    return m_data[0] != v.m_data[0] || m_data[1] != v.m_data[1] || m_data[2] != v.m_data[2] || m_data[3] != v.m_data[3];
  }


  template <typename T>
  AGX_FORCE_INLINE Vec4T<T> Vec4T<T>::componentMin(const Vec4T<T>& v1, const Vec4T<T>& v2)
  {
    return Vec4T(std::min(v1[0], v2[0]), std::min(v1[1], v2[1]), std::min(v1[2], v2[2]), std::min(v1[3], v2[3]));
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T> Vec4T<T>::componentMax(const Vec4T<T>& v1, const Vec4T<T>& v2)
  {
    return Vec4T(std::max(v1[0], v2[0]), std::max(v1[1], v2[1]), std::max(v1[2], v2[2]), std::max(v1[3], v2[3]));
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec4T<T>::minComponent() const
  {
    return std::min(std::min(std::min(m_data[0], m_data[1]), m_data[2]), m_data[3]);
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec4T<T>::maxComponent() const
  {
    return std::max(std::max(std::max(m_data[0], m_data[1]), m_data[2]), m_data[3]);
  }

  template <typename T>
  AGX_FORCE_INLINE size_t Vec4T<T>::minElement() const
  {
    T m = std::numeric_limits<T>::infinity();
    size_t idx = 0;
    for(size_t i = 0; i < 4; i++) {
      T a = agx::absolute(m_data[i]);
      if ( a < m) {
        idx = i;
        m = a;
      }
    }
    return idx;
  }

  template <typename T>
  AGX_FORCE_INLINE size_t Vec4T<T>::maxElement() const
  {
    T m = 0;
    size_t idx = 0;
    for(size_t i = 0; i < 4; i++) {
      T a = agx::absolute(m_data[i]);
      if ( a > m) {
        idx = i;
        m = a;
      }
    }
    return idx;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vec4T<T>::clamp(const Vec4T<T>& min, const Vec4T<T>& max)
  {
    m_data[0] = agx::clamp( m_data[0], min.m_data[0], max.m_data[0] );
    m_data[1] = agx::clamp( m_data[1], min.m_data[1], max.m_data[1] );
    m_data[2] = agx::clamp( m_data[2], min.m_data[2], max.m_data[2] );
    m_data[3] = agx::clamp( m_data[3], min.m_data[3], max.m_data[3] );
  }


  template <typename T>
  AGX_FORCE_INLINE bool Vec4T<T>::equalsZero() const
  {
    return (agx::equalsZero(m_data[0]) && agx::equalsZero(m_data[1]) && agx::equalsZero(m_data[2]) && agx::equalsZero(m_data[3]));
  }


  template <typename T>
  AGX_FORCE_INLINE T* Vec4T<T>::ptr()
  {
    return m_data;
  }

  template <typename T>
  AGX_FORCE_INLINE const T* Vec4T<T>::ptr() const
  {
    return m_data;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vec4T<T>::set( T x, T y, T z, T w )
  {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
    m_data[3] = w;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vec4T<T>::set( T value )
  {
    m_data[0] = m_data[1] = m_data[2] = m_data[3] = value;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vec4T<T>::set( const Vec4T& rhs )
  {
    m_data[0] = rhs.m_data[0];
    m_data[1] = rhs.m_data[1];
    m_data[2] = rhs.m_data[2];
    m_data[3] = rhs.m_data[3];
  }

  template <typename T>
  AGX_FORCE_INLINE T& Vec4T<T>::operator [] ( size_t i )
  {
    return m_data[i];
  }

  template <typename T>
  AGX_FORCE_INLINE const T& Vec4T<T>::operator [] ( size_t i ) const
  {
    return m_data[i];
  }

  template <typename T>
  AGX_FORCE_INLINE T& Vec4T<T>::x()
  {
    return m_data[0];
  }

  template <typename T>
  AGX_FORCE_INLINE T& Vec4T<T>::y()
  {
    return m_data[1];
  }

  template <typename T>
  AGX_FORCE_INLINE T& Vec4T<T>::z()
  {
    return m_data[2];
  }

  template <typename T>
  AGX_FORCE_INLINE T& Vec4T<T>::w()
  {
    return m_data[3];
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec4T<T>::x() const
  {
    return m_data[0];
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec4T<T>::y() const
  {
    return m_data[1];
  }
  template <typename T>
  AGX_FORCE_INLINE T Vec4T<T>::z() const
  {
    return m_data[2];
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec4T<T>::w() const
  {
    return m_data[3];
  }

  template <typename T>
  AGX_FORCE_INLINE bool Vec4T<T>::isValid() const
  {
    return !isNaN();
  }

  template <typename T>
  AGX_FORCE_INLINE bool Vec4T<T>::isNaN() const
  {
    return agx::isNaN( m_data[0] ) || agx::isNaN( m_data[1] ) || agx::isNaN( m_data[2] ) || agx::isNaN( m_data[3] );
  }

  template <typename T>
  AGX_FORCE_INLINE bool Vec4T<T>::isFinite() const
  {
    return agx::isFinite( m_data[0] ) && agx::isFinite( m_data[1] ) && agx::isFinite( m_data[2] ) && agx::isFinite( m_data[3] );
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec4T<T>::operator * ( const Vec4T<T>& rhs ) const
  {
    return m_data[0] * rhs.m_data[0] + m_data[1] * rhs.m_data[1] + m_data[2] * rhs.m_data[2] + m_data[3] * rhs.m_data[3];
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec4T<T> Vec4T<T>::operator | ( const Vec4T<T>& rhs ) const
  {
    return Vec4T( m_data[0] * rhs.m_data[0],
                  m_data[1] * rhs.m_data[1],
                  m_data[2] * rhs.m_data[2],
                  m_data[3] * rhs.m_data[3] );
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec4T<T> Vec4T<T>::operator * ( T rhs ) const
  {
    return Vec4T( m_data[0] * rhs, m_data[1] * rhs, m_data[2] * rhs, m_data[3] * rhs );
  }


  template <typename T>
  AGX_FORCE_INLINE Vec4T<T>& Vec4T<T>::operator *= ( T rhs )
  {
    m_data[0] = m_data[0] * rhs;
    m_data[1] = m_data[1] * rhs;
    m_data[2] = m_data[2] * rhs;
    m_data[3] = m_data[3] * rhs;
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec4T<T> Vec4T<T>::operator / ( T rhs ) const
  {
    return Vec4T( m_data[0] / rhs, m_data[1] / rhs, m_data[2] / rhs, m_data[3] / rhs );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T>& Vec4T<T>::operator /= ( T rhs )
  {
    m_data[0] = m_data[0] / rhs;
    m_data[1] = m_data[1] / rhs;
    m_data[2] = m_data[2] / rhs;
    m_data[3] = m_data[3] / rhs;
    return *this;
  }


  template <typename T>
  AGX_FORCE_INLINE const Vec4T<T> Vec4T<T>::operator + ( const Vec4T<T>& rhs ) const
  {
    return Vec4T( m_data[0] + rhs.m_data[0], m_data[1] + rhs.m_data[1], m_data[2] + rhs.m_data[2], m_data[3] + rhs.m_data[3] );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T>& Vec4T<T>::operator += ( const Vec4T<T>& rhs )
  {
    m_data[0] += rhs.m_data[0];
    m_data[1] += rhs.m_data[1];
    m_data[2] += rhs.m_data[2];
    m_data[3] += rhs.m_data[3];
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec4T<T> Vec4T<T>::operator - ( const Vec4T<T>& rhs ) const
  {
    return Vec4T( m_data[0] - rhs.m_data[0], m_data[1] - rhs.m_data[1], m_data[2] - rhs.m_data[2], m_data[3] - rhs.m_data[3] );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T>& Vec4T<T>::operator -= ( const Vec4T<T>& rhs )
  {
    m_data[0] -= rhs.m_data[0];
    m_data[1] -= rhs.m_data[1];
    m_data[2] -= rhs.m_data[2];
    m_data[3] -= rhs.m_data[3];
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec4T<T> Vec4T<T>::operator + ( const T& rhs ) const
  {
    return Vec4T( m_data[0] + rhs, m_data[1] + rhs, m_data[2] + rhs, m_data[3] + rhs );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T>& Vec4T<T>::operator += ( const T& rhs )
  {
    m_data[0] += rhs;
    m_data[1] += rhs;
    m_data[2] += rhs;
    m_data[3] += rhs;
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec4T<T> Vec4T<T>::operator - ( const T& rhs ) const
  {
    return Vec4T( m_data[0] - rhs, m_data[1] - rhs, m_data[2] - rhs, m_data[3] - rhs );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T>& Vec4T<T>::operator -= ( const T& rhs )
  {
    m_data[0] -= rhs;
    m_data[1] -= rhs;
    m_data[2] -= rhs;
    m_data[3] -= rhs;
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec4T<T> Vec4T<T>::operator - () const
  {
    return Vec4T ( -m_data[0], -m_data[1], -m_data[2], -m_data[3] );
  }

  template <typename T>
  AGX_FORCE_INLINE Real Vec4T<T>::length() const
  {
    return std::sqrt( length2() );
  }

  template <typename T>
  AGX_FORCE_INLINE Real Vec4T<T>::length2() const
  {
    return Real(m_data[0] * m_data[0] + m_data[1] * m_data[1] + m_data[2] * m_data[2] + m_data[3] * m_data[3]);
  }


  template <typename T>
  AGX_FORCE_INLINE Real Vec4T<T>::distance(const Vec4T<T>& v2) const
  {
    return std::sqrt(distance2(v2));
  }


  template <typename T>
  AGX_FORCE_INLINE Real Vec4T<T>::normalize()
  {
    Real norm = Vec4T::length();
    if ( norm > 0.0 ) {
      Real inv = Real(1.0) / norm;
      m_data[0] = T(m_data[0] * inv);
      m_data[1] = T(m_data[1] * inv);
      m_data[2] = T(m_data[2] * inv);
      m_data[3] = T(m_data[3] * inv);
    }
    return( norm );
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec4T<T>::at( size_t i )
  {
    return (*this)[i];
  }

  template <typename T>
  AGX_FORCE_INLINE void Vec4T<T>::set( T val, size_t i )
  {
    (*this)[i] = val;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Vec4T<T>::asVec3() const
  {
    return Vec3T<T>( m_data[0], m_data[1], m_data[2] );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T> Vec4T<T>::random(T min, T max)
  {
    return Vec4T(agx::random(min, max), agx::random(min, max), agx::random(min, max), agx::random(min, max));
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T> Vec4T<T>::random(const Vec4T& min, const Vec4T& max)
  {
    return Vec4T(agx::random(min[0], max[0]), agx::random(min[1], max[1]), agx::random(min[2], max[2]), agx::random(min[3], max[3]));
  }


  template<typename T>
  AGX_FORCE_INLINE bool equivalent( const agx::Vec4T<T>& a, const agx::Vec4T<T>& b, T epsilon = T(AGX_EQUIVALENT_EPSILON) )
  {
    return agx::equivalent(a[0], b[0], epsilon) &&
           agx::equivalent(a[1], b[1], epsilon) &&
           agx::equivalent(a[2], b[2], epsilon) &&
           agx::equivalent(a[3], b[3], epsilon);
  }

  template<typename T>
  AGX_FORCE_INLINE Vec4T<T> absolute( const agx::Vec4T<T>& a)
  {
    return Vec4T<T>(std::abs(a[0]),
                    std::abs(a[1]),
                    std::abs(a[2]),
                    std::abs(a[3]));
  }

  template<typename T>
  AGX_FORCE_INLINE std::ostream& operator << ( std::ostream& output, const Vec4T<T>& v )
  {
    output << v[0] << " " << v[1] << " " << v[2] << " " << v[3];
    return output;
  }


  /** Compute the dot product of a (Vec3,1.0) and a Vec4. */
  template <typename T>
  AGX_FORCE_INLINE Real operator * ( const Vec3T<T>& lhs, const Vec4T<T>& rhs )
  {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2] + rhs[3];
  }

  /** Compute the dot product of a Vec4 and a (Vec3,1.0). */
  template <typename T>
  AGX_FORCE_INLINE Real operator * ( const Vec4T<T>& lhs, const Vec3T<T>& rhs )
  {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2] + lhs[3];
  }


  template <typename T>
  AGX_FORCE_INLINE Real Vec4T<T>::distance2(const Vec4T& v2) const
  {
    Vec4T diff(m_data[0] - v2.m_data[0], m_data[1] - v2.m_data[1], m_data[2] - v2.m_data[2], m_data[3] - v2.m_data[3]);
    return diff.length2();
  }


  template<>
  AGX_FORCE_INLINE Real Vec4T<UInt8>::distance2(const Vec4T& v2) const
  {
    // Template specialization to avoid underflow in unsigned types.
    Vec4T diff( (UInt8) (std::max( m_data[0], v2.m_data[0] ) - std::min( m_data[0], v2.m_data[0] )),
                (UInt8) (std::max( m_data[1], v2.m_data[1] ) - std::min( m_data[1], v2.m_data[1] )),
                (UInt8) (std::max( m_data[2], v2.m_data[2] ) - std::min( m_data[2], v2.m_data[2] )),
                (UInt8) (std::max( m_data[3], v2.m_data[3] ) - std::min( m_data[3], v2.m_data[3] )) );
    return diff.length2();
  }


  template<>
  AGX_FORCE_INLINE Real Vec4T<UInt16>::distance2(const Vec4T& v2) const
  {
    // Template specialization to avoid underflow in unsigned types.
    Vec4T diff( (UInt16) (std::max( m_data[0], v2.m_data[0] ) - std::min( m_data[0], v2.m_data[0] )),
                (UInt16) (std::max( m_data[1], v2.m_data[1] ) - std::min( m_data[1], v2.m_data[1] )),
                (UInt16) (std::max( m_data[2], v2.m_data[2] ) - std::min( m_data[2], v2.m_data[2] )),
                (UInt16) (std::max( m_data[3], v2.m_data[3] ) - std::min( m_data[3], v2.m_data[3] )) );
    return diff.length2();
  }


  template<>
  AGX_FORCE_INLINE Real Vec4T<UInt32>::distance2(const Vec4T& v2) const
  {
    // Template specialization to avoid underflow in unsigned types.
    Vec4T diff( std::max( m_data[0], v2.m_data[0] ) - std::min( m_data[0], v2.m_data[0] ),
                std::max( m_data[1], v2.m_data[1] ) - std::min( m_data[1], v2.m_data[1] ),
                std::max( m_data[2], v2.m_data[2] ) - std::min( m_data[2], v2.m_data[2] ),
                std::max( m_data[3], v2.m_data[3] ) - std::min( m_data[3], v2.m_data[3] ) );
    return diff.length2();
  }


  template<>
  AGX_FORCE_INLINE Real Vec4T<UInt64>::distance2(const Vec4T& v2) const
  {
    // Template specialization to avoid underflow in unsigned types.
    Vec4T diff( std::max( m_data[0], v2.m_data[0] ) - std::min( m_data[0], v2.m_data[0] ),
                std::max( m_data[1], v2.m_data[1] ) - std::min( m_data[1], v2.m_data[1] ),
                std::max( m_data[2], v2.m_data[2] ) - std::min( m_data[2], v2.m_data[2] ),
                std::max( m_data[3], v2.m_data[3] ) - std::min( m_data[3], v2.m_data[3] ) );
    return diff.length2();
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T> clamp( const Vec4T<T>& vec, const Vec4T<T>& minimum, const Vec4T<T>& maximum )
  {
    Vec4T<T> result = vec;
    result.clamp(minimum, maximum);
    return result;
  }

}    // end of namespace agx

#endif
