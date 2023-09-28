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



#ifndef AGX_VEC2_TEMPLATE_H
#define AGX_VEC2_TEMPLATE_H

#include <iosfwd>

#include <agx/agx.h>
#include <agx/Math.h>

namespace agx
{
  template <typename T>
  class Vec2T
  {
  public:
    typedef T Type;

  public:

    /// Copy constructor
    Vec2T(const Vec2T& copy ) = default;

    /// Copy constructor for other types.
    template<typename T2>
    AGX_FORCE_INLINE explicit Vec2T(const Vec2T<T2>& copy )
    {
      //memcpy(&m_data, copy.m_data, sizeof(agx::Real)*3);
      m_data[ 0 ] = (T)copy[ 0 ];
      m_data[ 1 ] = (T)copy[ 1 ];
    }

    /**
    Default constructor
    */
    AGX_FORCE_INLINE Vec2T() {
      m_data[0] = T();
      m_data[1] = T();
    }

    explicit AGX_FORCE_INLINE Vec2T( T r ) {
      m_data[0] = m_data[1] = r;
    }

    AGX_FORCE_INLINE Vec2T( T x, T y ) {
      m_data[0] = x;
      m_data[1] = y;
    }

    explicit AGX_FORCE_INLINE Vec2T(const T v[2] ) {
      m_data[0] = v[0];
      m_data[1] = v[1];
    }

    static AGX_FORCE_INLINE Vec2T random(T min, T max);
    static AGX_FORCE_INLINE Vec2T random(const Vec2T& min, const Vec2T& max);

    /**
    Equality test
    */
    AGX_FORCE_INLINE bool operator == ( const Vec2T& v ) const {
      return m_data[0] == v.m_data[0] && m_data[1] == v.m_data[1];
    }

    /**
    In-equality test
    */
    AGX_FORCE_INLINE bool operator != ( const Vec2T& v ) const {
      return m_data[0] != v.m_data[0] || m_data[1] != v.m_data[1];
    }

    /**
    Creates a new vector where each component is the minimum of this and the other vector.
    */
    AGX_FORCE_INLINE static Vec2T componentMin(const Vec2T& v1, const Vec2T& v2) {
      return Vec2T(std::min(v1[0], v2[0]), std::min(v1[1], v2[1]));
    }

    /**
    Creates a new vector where each component is the maximum of this and the other vector.
    */
    AGX_FORCE_INLINE static Vec2T componentMax(const Vec2T& v1, const Vec2T& v2) {
      return Vec2T(std::max(v1[0], v2[0]), std::max(v1[1], v2[1]));
    }

    /**
    \return the smallest component (value).
    */
    AGX_FORCE_INLINE T minComponent() const
    {
      return std::min(m_data[0], m_data[1]);
    }

    /**
    \return the largest component (value).
    */
    AGX_FORCE_INLINE T maxComponent() const
    {
      return std::max(m_data[0], m_data[1]);
    }

    /**
    \return the index of the smallest element (in absolute value)
    */
    AGX_FORCE_INLINE size_t minElement() const
    {
      T m = std::numeric_limits<T>::infinity();
      size_t idx = 0;
      for(size_t i=0;i<2;i++)
      {
        T a = agx::absolute(m_data[i]);
        if ( a < m) {
          idx=i;
          m = a;
        }
      }
      return idx;
    }
    /**
    \return the index of the largest element (in absolute value)
    */
    AGX_FORCE_INLINE size_t maxElement() const
    {
      T m = 0;
      size_t idx = 0;
      for(size_t i=0;i<2;i++)
      {
        T a = agx::absolute(m_data[i]);
        if ( a > m) {
          idx=i;
          m = a;
        }
      }
      return idx;
    }

    /**
    Clamp a vector between a lower and upper bound (per component).
    */
    AGX_FORCE_INLINE void clamp(const Vec2T& min, const Vec2T& max)
    {
      m_data[0] = agx::clamp( m_data[0], min.m_data[0], max.m_data[0] );
      m_data[1] = agx::clamp( m_data[1], min.m_data[1], max.m_data[1] );
    }


    /**
    \return true if all elements are zero
    */
    AGX_FORCE_INLINE bool equalsZero() const
    {
      return (agx::equalsZero(m_data[0]) && agx::equalsZero(m_data[1]));
    }


    /**
    Return a pointer to the data vector
    */
    AGX_FORCE_INLINE T* ptr() {
      return m_data;
    }

    /**
    Return a const pointer to the data vector
    */
    AGX_FORCE_INLINE const T* ptr() const {
      return m_data;
    }

    AGX_FORCE_INLINE void set( T x, T y ) {
      m_data[0] = x;
      m_data[1] = y;
    }

    AGX_FORCE_INLINE void set( T value) {
      m_data[0] = m_data[1] = value;
    }

    AGX_FORCE_INLINE void set( const Vec2T& rhs ) {
      m_data[0] = rhs.m_data[0];
      m_data[1] = rhs.m_data[1];
    }


    AGX_FORCE_INLINE T& operator [] ( size_t i ) {
      return m_data[i];
    }
    AGX_FORCE_INLINE const T& operator [] ( size_t i ) const {
      return m_data[i];
    }

    AGX_FORCE_INLINE T& x() {
      return m_data[0];
    }
    AGX_FORCE_INLINE T& y() {
      return m_data[1];
    }

    AGX_FORCE_INLINE T x() const {
      return m_data[0];
    }
    AGX_FORCE_INLINE T y() const {
      return m_data[1];
    }

    AGX_FORCE_INLINE bool isValid() const {
      return !isNaN();
    }

    AGX_FORCE_INLINE bool isNaN() const {
      return agx::isNaN( m_data[0] ) || agx::isNaN( m_data[1] );
    }

    AGX_FORCE_INLINE bool isFinite() const {
      return agx::isFinite( m_data[0] ) && agx::isFinite( m_data[1] );
    }


    /** Dot product. */
    AGX_FORCE_INLINE T operator * ( const Vec2T& rhs ) const {
      return m_data[0]*rhs.m_data[0] + m_data[1]*rhs.m_data[1];
    }

    /** Cross product. */
    #if 0
    AGX_FORCE_INLINE const Vec2T operator ^ ( const Vec2T& rhs ) const {
      return Vec2T( m_data[1]*rhs.m_data[2] - m_data[2]*rhs.m_data[1],
                  m_data[2]*rhs.m_data[0] - m_data[0]*rhs.m_data[2] );
                   m_data[0]*rhs.m_data[1] - m_data[1]*rhs.m_data[0] );
    }
    #endif

    /** Element-wise-multiplication */
    AGX_FORCE_INLINE const Vec2T operator | ( const Vec2T& rhs ) const {
      return Vec2T( m_data[0] * rhs.m_data[0],
                    m_data[1] * rhs.m_data[1] );
    }

    /** Multiply by scalar. */
    AGX_FORCE_INLINE const Vec2T operator * ( T rhs ) const {
      return Vec2T( m_data[0]*rhs, m_data[1]*rhs );
    }


    /** Unary multiply by scalar. */
    AGX_FORCE_INLINE Vec2T& operator *= ( T rhs ) {
      m_data[0] = m_data[0]*rhs;
      m_data[1] = m_data[1] *rhs;
      return *this;
    }

    /** Divide by scalar. */
    AGX_FORCE_INLINE const Vec2T operator / ( T rhs ) const {
      return Vec2T( m_data[0] / rhs, m_data[1] / rhs );
    }

    /** Unary divide by scalar. */
    AGX_FORCE_INLINE Vec2T& operator /= ( T rhs ) {
      m_data[0] = m_data[0] / rhs;
      m_data[1] = m_data[1] / rhs;
      return *this;
    }


    /** Element-wise-multiplication */
    AGX_FORCE_INLINE static Vec2T mul( const Vec2T& lhs, const Vec2T& rhs )
    {
      return Vec2T( lhs[0] * rhs[0],
                    lhs[1] * rhs[1]);
    }

    AGX_FORCE_INLINE static Vec2T div( const Vec2T& lhs, const Vec2T& rhs )
    {
      return Vec2T( lhs[0] / rhs[0],
                    lhs[1] / rhs[1]);
    }

    /** Binary vector add. */
    AGX_FORCE_INLINE const Vec2T operator + ( const Vec2T& rhs ) const {
      return Vec2T( m_data[0] + rhs.m_data[0], m_data[1] + rhs.m_data[1] );
    }

    /** Unary vector add. Slightly more efficient because no temporary
    * intermediate object.
    */
    AGX_FORCE_INLINE Vec2T& operator += ( const Vec2T& rhs ) {
      m_data[0] += rhs.m_data[0];
      m_data[1] += rhs.m_data[1];
      return *this;
    }

    /** Binary vector subtract. */
    AGX_FORCE_INLINE const Vec2T operator - ( const Vec2T& rhs ) const {
      return Vec2T( m_data[0] - rhs.m_data[0], m_data[1] - rhs.m_data[1] );
    }

    /** Unary vector subtract. */
    AGX_FORCE_INLINE Vec2T& operator -= ( const Vec2T& rhs ) {
      m_data[0] -= rhs.m_data[0];
      m_data[1] -= rhs.m_data[1];
      return *this;
    }


    /** Binary vector add. */
    AGX_FORCE_INLINE const Vec2T operator + ( const T& rhs ) const {
      return Vec2T( m_data[0] + rhs, m_data[1] + rhs );
    }

    /** Unary vector add. Slightly more efficient because no temporary
    * intermediate object.
    */
    AGX_FORCE_INLINE Vec2T& operator += ( const T& rhs ) {
      m_data[0] += rhs;
      m_data[1] += rhs;
      return *this;
    }

    /** Binary vector subtract. */
    AGX_FORCE_INLINE const Vec2T operator - ( const T& rhs ) const {
      return Vec2T( m_data[0] - rhs, m_data[1] - rhs );
    }

    /** Unary vector subtract. */
    AGX_FORCE_INLINE Vec2T& operator -= ( const T& rhs ) {
      m_data[0] -= rhs;
      m_data[1] -= rhs;
      return *this;
    }


    /** Negation operator. Returns the negative of the Vec2T. */
    AGX_FORCE_INLINE const Vec2T operator - () const {
      return Vec2T ( -m_data[0], -m_data[1] );
    }

    /** Length of the vector = sqrt( vec . vec ) */
    AGX_FORCE_INLINE Real length() const {
      return std::sqrt( m_data[0]*m_data[0] + m_data[1]*m_data[1] );
    }

    /** Length squared of the vector = vec . vec */
    AGX_FORCE_INLINE Real length2() const {
      return Real(m_data[0]*m_data[0] + m_data[1]*m_data[1]);
    }

    /** Squared distance to another vector */
    AGX_FORCE_INLINE Real distance2(const Vec2T& v2) const;

    /** Distance to another vector */
    AGX_FORCE_INLINE Real distance(const Vec2T& v2) const {
      return std::sqrt(distance2(v2));
    }



    /** Normalize the vector so that it has length unity.
    * Returns the previous length of the vector.
    */
    AGX_FORCE_INLINE Real normalize() {
      Real norm = length();
      if ( norm > 0.0 ) {
        Real inv = Real(1.0) / norm;
        m_data[0] = T(m_data[0]*inv);
        m_data[1] = T(m_data[1]*inv);
      }
      return( norm );
    }

  private:
    T m_data[2];
  };    // end of class Vec2T_template

  template <typename T>
  AGX_FORCE_INLINE Vec2T<T> Vec2T<T>::random(T min, T max)
  {
    return Vec2T(agx::random(min, max), agx::random(min, max));
  }

  template <typename T>
  AGX_FORCE_INLINE Vec2T<T> Vec2T<T>::random(const Vec2T& min, const Vec2T& max)
  {
    return Vec2T(agx::random(min[0], max[0]), agx::random(min[1], max[1]));
  }


  template<typename T>
  AGX_FORCE_INLINE bool equivalent( const agx::Vec2T<T>& a, const agx::Vec2T<T>& b, T epsilon = T(AGX_EQUIVALENT_EPSILON) )
  {
    return agx::equivalent(a[0], b[0], epsilon) &&
      agx::equivalent(a[1], b[1], epsilon);
  }

  template<typename T>
  AGX_FORCE_INLINE Vec2T<T> absolute( const agx::Vec2T<T>& a)
  {
    return Vec2T<T>(
      std::abs(a[0]),
      std::abs(a[1]));
  }

  template<typename T>
  AGX_FORCE_INLINE std::ostream& operator << ( std::ostream& output, const Vec2T<T>& v )
  {
    output << v[0] << " " << v[1];
    return output;
  }


  template<typename T>
  AGX_FORCE_INLINE Real Vec2T<T>::distance2(const Vec2T& v2) const
  {
    Vec2T diff(m_data[0]-v2.m_data[0], m_data[1]-v2.m_data[1]);
    return diff.length2();
  }


  template<>
  AGX_FORCE_INLINE Real Vec2T<UInt8>::distance2(const Vec2T& v2) const
  {
    // Template specialization to avoid underflow in unsigned types.
    Vec2T diff( (UInt8) (std::max( m_data[0], v2.m_data[0] ) - std::min( m_data[0], v2.m_data[0] )),
                (UInt8) (std::max( m_data[1], v2.m_data[1] ) - std::min( m_data[1], v2.m_data[1] )) );
    return diff.length2();
  }


  template<>
  AGX_FORCE_INLINE Real Vec2T<UInt16>::distance2(const Vec2T& v2) const
  {
    // Template specialization to avoid underflow in unsigned types.
    Vec2T diff( (UInt16) (std::max( m_data[0], v2.m_data[0] ) - std::min( m_data[0], v2.m_data[0] )),
                (UInt16) (std::max( m_data[1], v2.m_data[1] ) - std::min( m_data[1], v2.m_data[1] )) );
    return diff.length2();
  }

  template<>
  AGX_FORCE_INLINE Real Vec2T<UInt32>::distance2(const Vec2T& v2) const
  {
    // Template specialization to avoid underflow in unsigned types.
    Vec2T diff( std::max( m_data[0], v2.m_data[0] ) - std::min( m_data[0], v2.m_data[0] ),
      std::max( m_data[1], v2.m_data[1] ) - std::min( m_data[1], v2.m_data[1] ) );
    return diff.length2();
  }


  template<>
  AGX_FORCE_INLINE Real Vec2T<UInt64>::distance2(const Vec2T& v2) const
  {
    // Template specialization to avoid underflow in unsigned types.
    Vec2T diff( std::max( m_data[0], v2.m_data[0] ) - std::min( m_data[0], v2.m_data[0] ),
      std::max( m_data[1], v2.m_data[1] ) - std::min( m_data[1], v2.m_data[1] ) );
    return diff.length2();
  }


  template <typename T>
  AGX_FORCE_INLINE Vec2T<T> clamp( const Vec2T<T>& vec, const Vec2T<T>& minimum, const Vec2T<T>& maximum )
  {
    Vec2T<T> result = vec;
    result.clamp(minimum, maximum);
    return result;
  }

}    // end of namespace agx


#endif
