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

#ifndef AGX_VEC3_TEMPLATE_H
#define AGX_VEC3_TEMPLATE_H

#include <agx/agxPhysics_export.h>
#include <agx/macros.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <iosfwd>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/agx.h>
#include <agx/Vec2.h>
#include <agx/Math.h>


namespace agx
{

  /**
  A class holding 3 dimensional vectors and providing basic arithmetic.

  The 3D vector class holds an internal representations of vectors which
  can include padding and so on.  At this level of the API, the object is
  provided for user convenience.  In the internal representation used by
  the solvers, 3D vectors may be concatenated into larger vector blocks
  as seems fit.
  */
  template <typename T >
  class Vec3T
  {
    public:
      typedef T Type;

      /// Copy constructor
      Vec3T(const Vec3T& copy ) = default;

      // Copy constructor with a new length
      Vec3T(const Vec3T& copy, Real length);

      // Copy constructor with one modified element, for x, y and z setters.
      Vec3T(const Vec3T& copy, T value, size_t i);

      /// Copy constructor for other types.
      template <typename T2>
      explicit Vec3T(const Vec3T<T2>& copy );

      static Vec3T random(T min = T(0), T max = T(1));

      static Vec3T random(const Vec3T<T>& min, const Vec3T<T>& max);

      /**
      Default constructor
      */
      Vec3T();

      explicit Vec3T( T r );

      Vec3T( T x, T y, T z );

      explicit Vec3T(const T v[3] );

      Vec3T(const Vec2& v2, T zz);

      /**
        Equality test
        */
      bool operator == ( const Vec3T& v ) const;

      /**
      In-equality test
      */
      bool operator != ( const Vec3T& v ) const;

      /**
      Creates a new vector where each component is the minimum of this and the other vector.
      */
      static Vec3T componentMin(const Vec3T& v1, const Vec3T& v2);

      /**
      Creates a new vector where each component is the maximum of this and the other vector.
      */
      static Vec3T componentMax(const Vec3T& v1, const Vec3T& v2);

      /**
      \return the smallest component (value).
      */
      T minComponent() const;

      /**
      \return the largest component (value).
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
      void clamp(const Vec3T& min, const Vec3T& max);

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

      void set( T x, T y, T z );

      void set( T value );

      void set( const Vec3T& rhs );

      T& operator [] ( size_t i );

      const T& operator [] ( size_t i ) const;

      T& x();

      T& y();

      T& z();

      T x() const;

      T y() const;

      T z() const;

      bool isValid() const;

      bool isNaN() const;

      bool isFinite() const; // Non NAN, not infinite.

      /** Dot product. */
      T operator * ( const Vec3T& rhs ) const;

      /** Cross product operator */
      const Vec3T operator ^ ( const Vec3T& rhs ) const;

      /** Cross product method */
      const Vec3T cross(const Vec3T& rhs) const;


      /** Element-wise-multiplication */
      static Vec3T mul( const Vec3T& lhs, const Vec3T& rhs );

      /** Element-wise-division */
      static Vec3T div( const Vec3T& lhs, const Vec3T& rhs );

      /** Multiply by scalar. */
      const Vec3T operator * ( T rhs ) const;

      /** Unary multiply by scalar. */
      Vec3T& operator *= ( T rhs );

      /** Divide by scalar. */
      const Vec3T operator / ( T rhs ) const;

      /** Unary divide by scalar. */
      Vec3T& operator /= ( T rhs );

      /** Binary vector add. */
      const Vec3T operator + ( const Vec3T& rhs ) const;

      /** Unary vector add. Slightly more efficient because no temporary
      * intermediate object.
      */
      Vec3T& operator += ( const Vec3T& rhs );

      /** Binary vector subtract. */
      const Vec3T operator - ( const Vec3T& rhs ) const;

      /** Unary vector subtract. */
      Vec3T& operator -= ( const Vec3T& rhs );

      /** Binary vector add. */
      const Vec3T operator + ( const T& rhs ) const;

      /** Unary vector add. Slightly more efficient because no temporary
      * intermediate object.
      */
      Vec3T& operator += ( const T& rhs );

      /** Binary vector subtract. */
      const Vec3T operator - ( const T& rhs ) const;

      /** Unary vector subtract. */
      Vec3T& operator -= ( const T& rhs );

      /** Negation operator. Returns the negative of the Vec3T. */
      const Vec3T operator - () const;

      /** Length of the vector = sqrt( vec . vec ) */
      Real length() const;

      /** Length squared of the vector = vec . vec */
      Real length2() const;

      /** Squared distance to another vector */
      Real distance2(const Vec3T& v2) const;

      /** Distance to another vector */
      Real distance(const Vec3T& v2) const;

      /**
      Normalize the vector so that it has length unity.
      \return the previous length of the vector.
      */
      Real normalize();

      /**
      Scale the vector so that is has the specified length
      */
      Real setLength(Real newLength);

      /// \return a vec3 containing the normalized version of this vector.
      Vec3T normal() const;

      /**
      \return an arbitrary unit vector perpendicular to this vector
      */
      Vec3T getPerpendicularUnitVector() const;

      /**
      Return a unit vector perpendicular to this and another vector.
      If the two vectors are parallel, an arbitrary perpendicular vector will be chosen.
      \return a unit vector perpendicular to this and another vector.
      */
      Vec3T getPerpendicularUnitVector(const Vec3T& v2) const;

      /// \return a vec3 defining a vector along the x-axis
      static Vec3T X_AXIS();

      /// \return a vec3 defining a vector along the y-axis
      static Vec3T Y_AXIS();

      /// \return a vec3 defining a vector along the z-axis
      static Vec3T Z_AXIS();

      /**
      \param i The axis to choose. 0, 1 or 2.
      \return a vec3 defining an unit vector along the i-th axis.
      */
      static Vec3T AXIS(size_t i);

      /// \return the element at i. Mainly for accessing vec3 in Lua.
      T at( size_t i );

      /// Set the value of element i. Mainly for accessing vec3 in Lua.
      void set( T val, size_t i );

      /// Store the tree components to the given memory area.
      void store3(T* storage) const;

    protected:
      T m_data[4];
  };    // end of class Vec3T_template


  // Implementation
  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>::Vec3T(const Vec3T<T>& copy, Real length)
  {
    m_data[0] = copy.m_data[0];
    m_data[1] = copy.m_data[1];
    m_data[2] = copy.m_data[2];
    m_data[3] = T(0);

    setLength(length);
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>::Vec3T(const Vec3T<T>& copy, T value, size_t i)
  {
    m_data[0] = copy.m_data[0];
    m_data[1] = copy.m_data[1];
    m_data[2] = copy.m_data[2];
    m_data[3] = T(0);

    m_data[i] = value;
  }

  template <typename T>
  template <typename T2>
  AGX_FORCE_INLINE Vec3T<T>::Vec3T(const Vec3T<T2>& copy )
  {
    m_data[ 0 ] = (T)copy[ 0 ];
    m_data[ 1 ] = (T)copy[ 1 ];
    m_data[ 2 ] = (T)copy[ 2 ];
    m_data[ 3 ] = T(0);
  }


  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>::Vec3T()
  {
    m_data[0] = T();
    m_data[1] = T();
    m_data[2] = T();
    m_data[3] = T(0);
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>::Vec3T( T r )
  {
    m_data[0] = m_data[1] = m_data[2] = r;
    m_data[3] = T(0);
  }

template <typename T>
  AGX_FORCE_INLINE Vec3T<T>::Vec3T( T x, T y, T z )
  {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
    m_data[3] = T(0);
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>::Vec3T(const T v[3] )
  {
    m_data[0] = v[0];
    m_data[1] = v[1];
    m_data[2] = v[2];
    m_data[3] = T(0);
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>::Vec3T( const Vec2& v2, T zz )
  {
    m_data[0] = T(v2[0]);
    m_data[1] = T(v2[1]);
    m_data[2] = zz;
    m_data[3] = T(0);
  }

  template <typename T>
  AGX_FORCE_INLINE bool Vec3T<T>::operator == ( const Vec3T<T>& v ) const
  {
    return m_data[0] == v.m_data[0] && m_data[1] == v.m_data[1] && m_data[2] == v.m_data[2];
  }

  template <typename T>
  AGX_FORCE_INLINE bool Vec3T<T>::operator != ( const Vec3T<T>& v ) const
  {
    return m_data[0] != v.m_data[0] || m_data[1] != v.m_data[1] || m_data[2] != v.m_data[2];
  }


  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Vec3T<T>::componentMin(const Vec3T& v1, const Vec3T& v2)
  {
    return Vec3T<T>( std::min( v1[0], v2[0] ), std::min( v1[1], v2[1] ), std::min( v1[2], v2[2] ) );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Vec3T<T>::componentMax(const Vec3T& v1, const Vec3T& v2)
  {
    return Vec3T<T>( std::max( v1[0], v2[0] ), std::max( v1[1], v2[1] ), std::max( v1[2], v2[2] ) );
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec3T<T>::minComponent() const
  {
    return std::min(std::min(m_data[0], m_data[1]), m_data[2]);
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec3T<T>::maxComponent() const
  {
    return std::max(std::max(m_data[0], m_data[1]), m_data[2]);
  }

  template <typename T>
  AGX_FORCE_INLINE size_t Vec3T<T>::minElement() const
  {
    T m = std::numeric_limits<T>::infinity();
    size_t idx = 0;
    for(size_t i = 0; i < 3; i++) {
      T a = agx::absolute(m_data[i]);
      if ( a < m) {
        idx = i;
        m = a;
      }
    }
    return idx;
  }

  template <typename T>
  AGX_FORCE_INLINE size_t Vec3T<T>::maxElement() const
  {
    T m = 0;
    size_t idx = 0;
    for(size_t i = 0; i < 3; i++) {
      T a = agx::absolute(m_data[i]);
      if ( a > m) {
        idx = i;
        m = a;
      }
    }
    return idx;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vec3T<T>::clamp(const Vec3T& min, const Vec3T& max)
  {
    m_data[0] = agx::clamp( m_data[0], min.m_data[0], max.m_data[0] );
    m_data[1] = agx::clamp( m_data[1], min.m_data[1], max.m_data[1] );
    m_data[2] = agx::clamp( m_data[2], min.m_data[2], max.m_data[2] );
  }

  template <typename T>
  AGX_FORCE_INLINE bool Vec3T<T>::equalsZero() const
  {
    return (agx::equalsZero(m_data[0]) && agx::equalsZero(m_data[1]) && agx::equalsZero(m_data[2]));
  }

  template <typename T>
  AGX_FORCE_INLINE T* Vec3T<T>::ptr()
  {
    return m_data;
  }

  template <typename T>
  AGX_FORCE_INLINE const T* Vec3T<T>::ptr() const
  {
    return m_data;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vec3T<T>::set( T x, T y, T z )
  {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vec3T<T>::set( T value )
  {
    m_data[0] = m_data[1] = m_data[2] = value;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vec3T<T>::set( const Vec3T<T>& rhs )
  {
    m_data[0] = rhs.m_data[0];
    m_data[1] = rhs.m_data[1];
    m_data[2] = rhs.m_data[2];
  }

  template <typename T>
  AGX_FORCE_INLINE T& Vec3T<T>::operator [] ( size_t i )
  {
    return m_data[i];
  }

  template <typename T>
  AGX_FORCE_INLINE const T&  Vec3T<T>::operator [] ( size_t i ) const
  {
    return m_data[i];
  }

  template <typename T>
  AGX_FORCE_INLINE T& Vec3T<T>::x()
  {
    return m_data[0];
  }

  template <typename T>
  AGX_FORCE_INLINE T& Vec3T<T>::y()
  {
    return m_data[1];
  }

  template <typename T>
  AGX_FORCE_INLINE T& Vec3T<T>::z()
  {
    return m_data[2];
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec3T<T>::x() const
  {
    return m_data[0];
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec3T<T>::y() const
  {
    return m_data[1];
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec3T<T>::z() const
  {
    return m_data[2];
  }

  template <typename T>
  AGX_FORCE_INLINE bool Vec3T<T>::isValid() const
  {
    return !isNaN();
  }

  template <typename T>
  AGX_FORCE_INLINE bool Vec3T<T>::isNaN() const
  {
    return agx::isNaN( m_data[0] ) || agx::isNaN( m_data[1] ) || agx::isNaN( m_data[2] );
  }

  template <typename T>
  AGX_FORCE_INLINE bool Vec3T<T>::isFinite() const
  {
    return agx::isFinite( m_data[0] ) && agx::isFinite( m_data[1] ) && agx::isFinite( m_data[2] );
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec3T<T>::operator * ( const Vec3T<T>& rhs ) const
  {
    return T(m_data[0] * rhs.m_data[0] + m_data[1] * rhs.m_data[1] + m_data[2] * rhs.m_data[2]);
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec3T<T> Vec3T<T>::operator ^ ( const Vec3T<T>& rhs ) const
  {
    return Vec3T<T>( m_data[1] * rhs.m_data[2] - m_data[2] * rhs.m_data[1],
                  m_data[2] * rhs.m_data[0] - m_data[0] * rhs.m_data[2] ,
                  m_data[0] * rhs.m_data[1] - m_data[1] * rhs.m_data[0] );
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec3T<T> Vec3T<T>::cross(const Vec3T<T>& rhs) const
  {
    return Vec3T<T>(m_data[1] * rhs.m_data[2] - m_data[2] * rhs.m_data[1],
      m_data[2] * rhs.m_data[0] - m_data[0] * rhs.m_data[2],
      m_data[0] * rhs.m_data[1] - m_data[1] * rhs.m_data[0]);
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Vec3T<T>::mul( const Vec3T<T>& lhs, const Vec3T<T>& rhs )
  {    return Vec3T<T>( lhs[0] * rhs[0],
                  lhs[1] * rhs[1],
                  lhs[2] * rhs[2] );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Vec3T<T>::div( const Vec3T<T>& lhs, const Vec3T<T>& rhs )
  {
    return Vec3T<T>( lhs[0] / rhs[0],
                  lhs[1] / rhs[1],
                  lhs[2] / rhs[2] );
  }


  template <typename T>
  AGX_FORCE_INLINE const Vec3T<T> Vec3T<T>::operator * ( T rhs ) const
  {
    return Vec3T<T>( T(m_data[0] * rhs), T(m_data[1] * rhs), T(m_data[2] * rhs) );
  }


  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>& Vec3T<T>::operator *= ( T rhs )
  {
    m_data[0] = T(m_data[0] * rhs);
    m_data[1] = T(m_data[1] * rhs);
    m_data[2] = T(m_data[2] * rhs);
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec3T<T> Vec3T<T>::operator / ( T rhs ) const
  {
    return Vec3T( T(m_data[0] / rhs), T(m_data[1] / rhs), T(m_data[2] / rhs) );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>& Vec3T<T>::operator /= ( T rhs )
  {
    m_data[0] = T(m_data[0] / rhs);
    m_data[1] = T(m_data[1] / rhs);
    m_data[2] = T(m_data[2] / rhs);
    return *this;
  }


  template <typename T>
  AGX_FORCE_INLINE const Vec3T<T> Vec3T<T>::operator + ( const Vec3T<T>& rhs ) const
  {
    return Vec3T( m_data[0] + rhs[0], m_data[1] + rhs[1], m_data[2] + rhs[2] );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>& Vec3T<T>::operator += ( const Vec3T<T>& rhs )
  {
    m_data[0] += rhs.m_data[0];
    m_data[1] += rhs.m_data[1];
    m_data[2] += rhs.m_data[2];
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec3T<T> Vec3T<T>::operator - ( const Vec3T<T>& rhs ) const
  {
    return Vec3T( m_data[0] - rhs[0], m_data[1] - rhs[1], m_data[2] - rhs[2] );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>& Vec3T<T>::operator -= ( const Vec3T<T>& rhs )
  {
    m_data[0] -= rhs.m_data[0];
    m_data[1] -= rhs.m_data[1];
    m_data[2] -= rhs.m_data[2];
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec3T<T> Vec3T<T>::operator + ( const T& rhs ) const
  {
    return Vec3T( m_data[0] + rhs, m_data[1] + rhs, m_data[2] + rhs );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>& Vec3T<T>::operator += ( const T& rhs )
  {
    m_data[0] += rhs;
    m_data[1] += rhs;
    m_data[2] += rhs;
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec3T<T>  Vec3T<T>::operator - ( const T& rhs ) const
  {
    return Vec3T( m_data[0] - rhs, m_data[1] - rhs, m_data[2] - rhs );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>& Vec3T<T>::operator -= ( const T& rhs )
  {
    m_data[0] -= rhs;
    m_data[1] -= rhs;
    m_data[2] -= rhs;
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE const Vec3T<T> Vec3T<T>::operator - () const
  {
    return Vec3T ( -m_data[0], -m_data[1], -m_data[2] );
  }

  template <typename T>
  AGX_FORCE_INLINE Real Vec3T<T>::length() const
  {
    return std::sqrt( agx::Real(m_data[0] * m_data[0] + m_data[1] * m_data[1] + m_data[2] * m_data[2]) );
  }

  template <typename T>
  AGX_FORCE_INLINE Real Vec3T<T>::length2() const
  {
    return Real(m_data[0] * m_data[0] + m_data[1] * m_data[1] + m_data[2] * m_data[2]);
  }

  template <typename T>
  AGX_FORCE_INLINE Real  Vec3T<T>::distance(const Vec3T& v2) const
  {
    return std::sqrt(distance2(v2));
  }

  template <typename T>
  AGX_FORCE_INLINE Real  Vec3T<T>::normalize()
  {
    return setLength((Real) 1.0);
  }

  template<typename T>
  AGX_FORCE_INLINE Real Vec3T<T>::setLength(Real newLength)
  {
    Real norm = Vec3T<T>::length();

    if (norm > 0.0) {
      Real inv = Real(newLength) / norm;
      m_data[0] = T(m_data[0] * inv);
      m_data[1] = T(m_data[1] * inv);
      m_data[2] = T(m_data[2] * inv);

    }
    return(norm);
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>  Vec3T<T>::normal() const
  {
    Vec3T normal = *this;
    normal.normalize();
    return normal;
  }



  template<typename T>
  AGX_FORCE_INLINE Vec3T<T> Vec3T<T>::getPerpendicularUnitVector() const
  {
    // Create a temporary vector that's not parallel to this vector
    Vec3T<T> tmp;
    if (agx::absolute(m_data[0]) < agx::absolute(m_data[1])) {
      if (agx::absolute(m_data[0]) < agx::absolute(m_data[2]))
        tmp.set(T(1), T(0), T(0)); // use x axis.
      else
        tmp.set(T(0), T(0), T(1));
    }
    else if (agx::absolute(m_data[1]) < agx::absolute(m_data[2])) {
      tmp.set(T(0), T(1), T(0));
    }
    else {
      tmp.set(T(0), T(0), T(1));
    }

    // Create and return a vector perpendicular to this and the temporary one
    Vec3T<T> vPerp = (*this) ^ tmp;
    vPerp.normalize();

    return vPerp;
  }



  template<typename T>
  Vec3T<T> Vec3T<T>::getPerpendicularUnitVector(const Vec3T& v2) const
  {
    Vec3T<T> vPerp = (*this) ^ v2;
    if (vPerp.equalsZero())
      return getPerpendicularUnitVector();
    vPerp.normalize();
    return vPerp;
  }



  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Vec3T<T>::X_AXIS()
  {
    return Vec3T<T>(1, 0, 0);
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Vec3T<T>::Y_AXIS()
  {
    return Vec3T<T>(0, 1, 0);
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Vec3T<T>::Z_AXIS()
  {
    return Vec3T<T>(0, 0, 1);
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Vec3T<T>::AXIS(size_t i)
  {
    Vec3T<T> tmp;
    tmp[i] = T(1);
    return tmp;
  }

  template <typename T>
  AGX_FORCE_INLINE T Vec3T<T>::at( size_t i )
  {
    return (*this)[i];
  }

  template <typename T>
  AGX_FORCE_INLINE void Vec3T<T>::set( T val, size_t i )
  {
    (*this)[i] = val;
  }


  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Vec3T<T>::random(T min, T max)
  {
    return Vec3T(agx::random(min, max), agx::random(min, max), agx::random(min, max));
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Vec3T<T>::random(const Vec3T<T>& min, const Vec3T<T>& max)
  {
    return Vec3T(agx::random(min[0], max[0]), agx::random(min[1], max[1]), agx::random(min[2], max[2]));
  }


  template<typename T>
  void Vec3T<T>::store3(T* storage) const
  {
    ::memcpy(storage, this->ptr(), size_t(3)*sizeof(T));
  }


  template<typename T>
  AGX_FORCE_INLINE bool equivalent( const agx::Vec3T<T>& a, const agx::Vec3T<T>& b, T epsilon = T(AGX_EQUIVALENT_EPSILON) )
  {
    return agx::equivalent(a[0], b[0], epsilon) &&
           agx::equivalent(a[1], b[1], epsilon) &&
           agx::equivalent(a[2], b[2], epsilon);
  }

  template<typename T>
  AGX_FORCE_INLINE Vec3T<T> absolute( const agx::Vec3T<T>& a)
  {
    return Vec3T<T>(std::abs(a[0]),
                    std::abs(a[1]),
                    std::abs(a[2]));
  }

  template<typename T>
  AGX_FORCE_INLINE Vec3T<T> asin( const agx::Vec3T<T>& a)
  {
    return Vec3T<T>(std::asin(a[0]),
                    std::asin(a[1]),
                    std::asin(a[2]));
  }

  template<typename T>
  AGX_FORCE_INLINE Vec3T<T> sinc( const agx::Vec3T<T>& a)
  {
    return Vec3T<T>(agx::sinc(a[0]),
                    agx::sinc(a[1]),
                    agx::sinc(a[2]));
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> operator* ( T val, const Vec3T<T>& vec )
  {
    return vec * val;
  }

  template<typename T>
  AGX_FORCE_INLINE Vec3T<T> max( const Vec3T<T>& lhs, const Vec3T<T>& rhs )
  {
    return Vec3T<T>::componentMax(lhs, rhs);
  }

  template<typename T>
  AGX_FORCE_INLINE Vec3T<T> min( const Vec3T<T>& lhs, const Vec3T<T>& rhs )
  {
    return Vec3T<T>::componentMin(lhs, rhs);
  }


  template<typename T>
  AGX_FORCE_INLINE std::ostream& operator << ( std::ostream& output, const Vec3T<T>& v )
  {
    output << v[0] << " " << v[1] << " " << v[2];
    return output;
  }


  template<typename T>
  AGX_FORCE_INLINE Real Vec3T<T>::distance2(const Vec3T& v2) const
  {
    Vec3T diff(m_data[0] - v2.m_data[0], m_data[1] - v2.m_data[1], m_data[2] - v2.m_data[2]);
    return diff.length2();
  }

  template<>
  AGX_FORCE_INLINE Real Vec3T<UInt8>::distance2(const Vec3T& v2) const
  {
    // Template specialization to avoid underflow in unsigned types.
    Vec3T diff( (UInt8) (std::max( m_data[0], v2.m_data[0] ) - std::min( m_data[0], v2.m_data[0] )),
                (UInt8) (std::max( m_data[1], v2.m_data[1] ) - std::min( m_data[1], v2.m_data[1] )),
                (UInt8) (std::max( m_data[2], v2.m_data[2] ) - std::min( m_data[2], v2.m_data[2] )) );
    return diff.length2();
  }

  template<>
  AGX_FORCE_INLINE Real Vec3T<UInt16>::distance2(const Vec3T& v2) const
  {
    // Template specialization to avoid underflow in unsigned types.
    Vec3T diff( (UInt16) (std::max( m_data[0], v2.m_data[0] ) - std::min( m_data[0], v2.m_data[0] )),
                (UInt16) (std::max( m_data[1], v2.m_data[1] ) - std::min( m_data[1], v2.m_data[1] )),
                (UInt16) (std::max( m_data[2], v2.m_data[2] ) - std::min( m_data[2], v2.m_data[2] )) );
    return diff.length2();
  }

  template<>
  AGX_FORCE_INLINE Real Vec3T<UInt32>::distance2(const Vec3T& v2) const
  {
    // Template specialization to avoid underflow in unsigned types.
    Vec3T diff( std::max( m_data[0], v2.m_data[0] ) - std::min( m_data[0], v2.m_data[0] ),
                std::max( m_data[1], v2.m_data[1] ) - std::min( m_data[1], v2.m_data[1] ),
                std::max( m_data[2], v2.m_data[2] ) - std::min( m_data[2], v2.m_data[2] ) );
    return diff.length2();
  }

  template<>
  AGX_FORCE_INLINE Real Vec3T<UInt64>::distance2(const Vec3T& v2) const
  {
    // Template specialization to avoid underflow in unsigned types.
    Vec3T diff( std::max( m_data[0], v2.m_data[0] ) - std::min( m_data[0], v2.m_data[0] ),
                std::max( m_data[1], v2.m_data[1] ) - std::min( m_data[1], v2.m_data[1] ),
                std::max( m_data[2], v2.m_data[2] ) - std::min( m_data[2], v2.m_data[2] ) );
    return diff.length2();
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> clamp( const Vec3T<T>& vec, const Vec3T<T>& minimum, const Vec3T<T>& maximum )
  {
    Vec3T<T> result = vec;
    result.clamp(minimum, maximum);
    return result;
  }


}    // end of namespace agx


#endif
