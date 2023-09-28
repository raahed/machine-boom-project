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



#ifndef AGX_QUATTEMPLATE_H
#define AGX_QUATTEMPLATE_H

#include <agx/agxCore_export.h>

#include <agx/agx.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>
#include <agx/EulerConvention.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 6385) // Disable warning C6385: Reading invalid data
#endif


namespace agx
{
  class EulerAngles;
  class OrthoMatrix3x3;

  template <typename T>
  class AffineMatrix4x4T;

  /**
  The object holding quaternions and providing operations on these.

  The quaternion class holds a representation of quaternion objects and
  provides simple arithmetic as well as conversion utilities.  In
  addition, the class can compute the 3x4 matrices for converting angular
  velocities to quaternion velocities and vice versa.  The class can also
  convert from and to a 3x3 orthonormal matrix.

  The convention adopted for quaternions is that they map a moving frame
  to the fixed global reference.
  */
  template <typename T>
  class QuatT
  {
    public:

      typedef T Type;

      /// Default constructor. Create a Quaternion with an identity rotation
      QuatT();

      /// Create a Quaternion with the specified scalar values
      QuatT( T x, T y, T z, T w );

      /// Create a Quaternion based on the 4D vector (x,y,z,w)
      explicit QuatT( const Vec4T<T>& v );

      /// Create a Quaternion calculated from a rotation of \p angle radians around the rotation axis \p axis
      QuatT( T angle, const Vec3T<T>& axis );

      /**
      Create a Quaternion based on the concatenation between the three specified angle and axis.
      */
      QuatT( T angle1, const Vec3T<T>& axis1,
            T angle2, const Vec3T<T>& axis2,
            T angle3, const Vec3T<T>& axis3 );

      /**
      Create a Quaternion which maps the rotation from vector \p from to the vector \p to
      */
      QuatT( const Vec3T<T>& from, const Vec3T<T>& to );

      /**
      Create a Quaternion based on the rotation from \p euler specified as a EulerAngle
      */
      explicit QuatT( const EulerAngles& euler);


      /// Copy constructor
      QuatT(const QuatT<T>& copy ) = default;

      /// Copy constructor for other types.
      template <typename T2>
      explicit QuatT(const QuatT<T2>& copy );

      /**
      Create a Quaternion based on the rotation part in the specified Affine 4x4 transformation matrix.
      */
      explicit QuatT( const AffineMatrix4x4T<T>& matrix );

      /// Assignment operator
      QuatT<T>& operator = ( const QuatT<T>& v );

      /**
      Test for equality
      \return true if this Quaternion is identical (using == operator) to \p v
      */
      bool operator == ( const QuatT<T>& v ) const;

      /**
      \return true if \p v is not identical to this quaternion
      */
      bool operator != ( const QuatT<T>& v ) const;

      /* ----------------------------------
      Methods to access data members
      ---------------------------------- */

      /// \return the elements in the Quaternion as a Vec4
      Vec4T<T> asVec4() const;

      /// \return the vector part (x,y,z) as a Vec3T<T>
      Vec3T<T> asVec3() const;

      /// Set the 4 elements of the quaternion as 4 scalars.
      void set( T x, T y, T z, T w );

      /// Set the four elements of the quaternion as an Vec4.
      void set( const Vec4T<T>& v );

      /**
      Convert this quaternion to an EulerAngles representation.
      param euler - reference to where the result will be written.
      */
      void get( EulerAngles& euler ) const;

      /**
      Convert to Euler anglers. and retrieve as a Vec3 with the angles stored.
      \param convention - The specified convention for EulerAngles
      */
      Vec3T<T> getAsEulerAngles(agx::EulerConvention::Convention convention = agx::EulerConvention::Convention::DEFAULT_CONVENTION) const;

      /**
      Set the Rotational part of \p matrix from this quaternion
      \param matrix - reference to the matrix where the result will be written.
      */
      void get( AffineMatrix4x4T<T>& matrix ) const;

      /**
      Set an OrthoMatrix3x3 to represent the same rotation as the quaternion.
      \param matrix - reference to the matrix where the result will be written.
      */
      void get(OrthoMatrix3x3& matrix) const;

      /**
      Convert a Quaternion to an axis and angle representation
      \param axis - reference to which the axis will be written
      \param angle - reference to the real value where the angle will be written
      */
      void get( T& angle, agx::Vec3T<T>& axis ) const;


      /**
      Set this quaternion with the rotation described by the EulerAngles representation
      \return reference to this updated quaternion.
      */
      QuatT<T>& set( const EulerAngles& euler );

      /**
      Set this quaternion with the rotational part of \p matrix
      \return a reference to this updated quaternion.
      */
      QuatT<T>& set( const AffineMatrix4x4T<T>& matrix );

      /**
      Set this quaternion with the rotation described by a OrthoMatrix3x3
      \return a reference to this updated quaternion.
      */
      QuatT<T>& set( const OrthoMatrix3x3& matrix );


      /// \return a reference to the i:th element of the quaternion in the order x,y,z,w
      T& operator [] ( size_t i );

      /// \return the value of the i:th element of the quaternion in the order x,y,z,w
      T  operator [] ( size_t i ) const;

      /// \return a reference to the first element of the vector part of the quaternion.
      T& x();

      /// \return a reference to the second element of the vector part of the quaternion.
      T& y();

      /// \return a reference to the third element of the vector part of the quaternion.
      T& z();

      /// \return a reference to the scalar (last) element of the quaternion.
      T& w();


      /// \return the value of the first element of the vector part of the quaternion.
      T x() const;

      /// \return the value of the second element of the vector part of the quaternion.
      T y() const;

      /// \return the value of the third element of the vector part of the quaternion.
      T z() const;

      /// \return the value of the scalar (last) element of the quaternion.
      T w() const;

      /**
      \return true if the QuatT<T> represents a zero rotation, and therefore can be ignored in computations.
      */
      bool zeroRotation() const;


      /* -------------------------------------------------------------
      BASIC ARITHMETIC METHODS
      Implemented in terms of Vec4s.  Some Vec4 operators, e.g.
      operator* are not appropriate for quaternions (as
      mathematical objects) so they are implemented differently.
      Also define methods for conjugate and the multiplicative inverse.
      ------------------------------------------------------------- */
      /// Multiply by scalar \p rhs
      const QuatT<T> operator * ( T rhs ) const;

      /// Unary multiply by scalar \p rhs
      QuatT<T>& operator *= ( T rhs );

      /**
      Binary LEFT  multiply: result is  q = rhs quatmultiply   this,
      even though the notation  is  q = this * rhs ;
      */
      const QuatT<T> operator*( const QuatT<T>& rhs ) const;

      /**
      Unary LEFT multiply: result is   this = rhs quatmultiply this,
      which makes sense since the notation suggests this = rhs * this ;
      */
      QuatT<T>& operator*=( const QuatT<T>& rhs );

      /**
      Binary left multiplication:  q = p * rhs; results in the OTHER WAY
      AROUND:  q = rhs quatmultiply  p;
      */
      const QuatT<T> leftMult( const QuatT<T>& rhs ) const;

      /**
      Binary RIGHT multiplication:  q = p * rhs ; results in
      q = p quatmultiply  rhs ;
      */
      const QuatT<T> rightMult( const QuatT<T>& rhs ) const;


      /// Divide by scalar \p rhs
      QuatT<T> operator / ( T rhs ) const;

      /// Unary divide by scalar \p rhs
      QuatT<T>& operator /= ( T rhs );

      /// Binary divide
      const QuatT<T> operator/( const QuatT<T>& denom ) const;

      /// Unary divide
      QuatT<T>& operator/=( const QuatT<T>& denom );

      /// Binary addition
      const QuatT<T> operator + ( const QuatT<T>& rhs ) const;

      /// Unary addition
      QuatT<T>& operator += ( const QuatT<T>& rhs );

      /// Binary subtraction
      const QuatT<T> operator - ( const QuatT<T>& rhs ) const;

      /// Unary subtraction
      QuatT<T>& operator -= ( const QuatT<T>& rhs );

      /**
      Negation operator - returns the negative of the quaternion.
      Basically just calls operator - () on the Vec4
      */
      const QuatT<T> operator - () const;

      /// \return the length of the quaternion = sqrt( vec . vec )
      T length() const;

      /// \return the squared length of the quaternion = vec . vec
      T length2() const;

      /// \return the conjugate
      QuatT<T> conj () const;

      /// Multiplicative inverse method: q^(-1) = q^*/(q.q^*)
      QuatT<T> inverse () const;

      /**
      Set the rotation of the Quaternion as a rotation \p angle radians around the vector (\p x, \p y, \p z)
      */
      void setRotate( T angle, T x, T y, T z );

      /**
      Set the rotation of this Quaternion as a rotation \p angle radians around the vector \p vec
      */
      void setRotate( T angle, const Vec3T<T>& vec );

      /**
      Set the rotation of this Quaternion as a concatenation of the three angle/axis rotations:

      *this = q1*q2*q3
      */
      void setRotate ( T angle1, const Vec3T<T>& axis1,
                       T angle2, const Vec3T<T>& axis2,
                       T angle3, const Vec3T<T>& axis3 );

      /**
      Make a rotation QuatT<T> which will rotate \p from to \p to
      Generally take a dot product to get the angle between these
      and then use a cross product to get the rotation axis
      Watch out for the two special cases when the vectors
      are co-incident or opposite in direction.
      */
      void setRotate( const Vec3T<T>& from, const Vec3T<T>& to );

      /**
      Static method which constructs and returns a Quaternion from a rotation given as \p angle radians around the vector (\p x, \p y, \p z)
      */
      static QuatT<T> rotate( T angle, T x, T y, T z );

      /**
      Static method which constructs and returns a Quaternion from a rotation given as \p angle radians around the vector \p vec
      */
      static QuatT<T> rotate( T angle, const Vec3T<T>& vec );

      /**
      Static method which constructs and returns a Quaternion from a rotation given as a concatenation of the three angle/axis rotations:

      *this = q1*q2*q3
      */
      static QuatT<T> rotate( T angle1, const Vec3T<T>& axis1,
                          T angle2, const Vec3T<T>& axis2,
                          T angle3, const Vec3T<T>& axis3 );

      /**
      Static method which constructs and returns a Quaternion from rotating vector \p from to vector \p to.
      */
      static QuatT<T> rotate( const Vec3T<T>& from, const Vec3T<T>& to );

      /**
      Get the angle and vector components represented by the quaternion.
      */
      void getRotate( T& angle, T& x, T& y, T& z ) const;

      /**
      Get the angle and vector represented by the quaternion.
      */
      void getRotate( T& angle, Vec3T<T>& vec ) const;

      /**
      Get the rotation angle represented by the quaternion.
      */
      agx::Real getAngle() const;

      /**
      Get the rotation unit vector represented by the quaternion.
      */
      agx::Vec3T<T> getUnitVector() const;

      /**
      Spherical Linear Interpolation.
      As \p t goes from 0 to 1, the QuatT<T> object goes from \p from to \p to.
      */
      void slerp( T t, const QuatT<T>& from, const QuatT<T>& to );

      /**
      Rotate a vector by this quaternion.
      */
      Vec3T<T> operator* ( const Vec3T<T>& v ) const;

      /**
      Normalize the quat so that it has unit length.
      * Returns the previous length of the quat.
      */
      T normalize();


      /// \return the element at i. Mainly for accessing QuatT<T> in Lua.
      T at( size_t i );

      /// Set the value of element i. Mainly for accessing QuatT<T> in Lua.
      void set( T val, size_t i );

      /**
      \return a pointer to the data
      */
      T* ptr();

      /**
      \return a const pointer to the data
      */
      const T* ptr() const;

    protected:
      T m_data[4];
  };


  /* Implementation */
  template <typename T>
  AGX_FORCE_INLINE QuatT<T>::QuatT()
  {
    m_data[0] = 0.0;
    m_data[1] = 0.0;
    m_data[2] = 0.0;
    m_data[3] = 1.0;
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T>::QuatT( T x, T y, T z, T w )
  {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
    m_data[3] = w;
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T>::QuatT( const Vec4T<T>& v )
  {
    m_data[0] = v.x();
    m_data[1] = v.y();
    m_data[2] = v.z();
    m_data[3] = v.w();
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T>::QuatT( T angle, const Vec3T<T>& axis )
  {
    setRotate( angle, axis );
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T>::QuatT( T angle1, const Vec3T<T>& axis1,
                               T angle2, const Vec3T<T>& axis2,
                               T angle3, const Vec3T<T>& axis3 )
  {
    setRotate( angle1, axis1, angle2, axis2, angle3, axis3 );
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T>::QuatT( const Vec3T<T>& from, const Vec3T<T>& to )
  {
    setRotate( from, to );
  }


  template <typename T>
  template <typename T2>
  AGX_FORCE_INLINE QuatT<T>::QuatT(const QuatT<T2>& copy )
  {
    m_data[ 0 ] = (T)copy[ 0 ];
    m_data[ 1 ] = (T)copy[ 1 ];
    m_data[ 2 ] = (T)copy[ 2 ];
    m_data[ 3 ] = (T)copy[ 3 ];
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T>::QuatT( const AffineMatrix4x4T<T>& matrix )
  {
    set( matrix );
  }


  template <typename T>
  AGX_FORCE_INLINE QuatT<T>& QuatT<T>::operator = ( const QuatT<T>& v )
  {
    m_data[0] = v.m_data[0];
    m_data[1] = v.m_data[1];
    m_data[2] = v.m_data[2];
    m_data[3] = v.m_data[3];
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE bool QuatT<T>::operator == ( const QuatT<T>& v ) const
  {
    return m_data[0] == v.m_data[0] && m_data[1] == v.m_data[1] && m_data[2] == v.m_data[2] && m_data[3] == v.m_data[3];
  }

  template <typename T>
  AGX_FORCE_INLINE bool QuatT<T>::operator != ( const QuatT<T>& v ) const
  {
    return m_data[0] != v.m_data[0] || m_data[1] != v.m_data[1] || m_data[2] != v.m_data[2] || m_data[3] != v.m_data[3];
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T> QuatT<T>::asVec4() const
  {
    return Vec4T<T>( m_data[0], m_data[1], m_data[2], m_data[3] );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> QuatT<T>::asVec3() const
  {
    return Vec3T<T>( m_data[0], m_data[1], m_data[2] );
  }

  template <typename T>
  AGX_FORCE_INLINE void QuatT<T>::set( T x, T y, T z, T w )
  {
    m_data[0] = x;
    m_data[1] = y;
    m_data[2] = z;
    m_data[3] = w;
  }

  template <typename T>
  AGX_FORCE_INLINE void QuatT<T>::set( const Vec4T<T>& v )
  {
    m_data[0] = v.x();
    m_data[1] = v.y();
    m_data[2] = v.z();
    m_data[3] = v.w();
  }

  template <typename T>
  AGX_FORCE_INLINE T& QuatT<T>::operator [] ( size_t i )
  {
    agxAssert( i < 4 );
    return m_data[i];
  }

  template <typename T>
  AGX_FORCE_INLINE T   QuatT<T>::operator [] ( size_t i ) const
  {
    agxAssert( i < 4 );
    return m_data[i];
  }

  template <typename T>
  AGX_FORCE_INLINE T& QuatT<T>::x()
  {
    return m_data[0];
  }

  template <typename T>
  AGX_FORCE_INLINE T& QuatT<T>::y()
  {
    return m_data[1];
  }

  template <typename T>
  AGX_FORCE_INLINE T& QuatT<T>::z()
  {
    return m_data[2];
  }

  template <typename T>
  AGX_FORCE_INLINE T& QuatT<T>::w()
  {
    return m_data[3];
  }

  template <typename T>
  AGX_FORCE_INLINE T QuatT<T>::x() const
  {
    return m_data[0];
  }

  template <typename T>
  AGX_FORCE_INLINE T QuatT<T>::y() const
  {
    return m_data[1];
  }

  template <typename T>
  AGX_FORCE_INLINE T QuatT<T>::z() const
  {
    return m_data[2];
  }

  template <typename T>
  AGX_FORCE_INLINE T QuatT<T>::w() const
  {
    return m_data[3];
  }

  template <typename T>
  inline bool QuatT<T>::zeroRotation() const
  {
    return m_data[0] == 0.0 && m_data[1] == 0.0 && m_data[2] == 0.0 && m_data[3] == 1.0;
  }


  template <typename T>
  AGX_FORCE_INLINE const QuatT<T> QuatT<T>::operator * ( T rhs ) const
  {
    return QuatT( m_data[0] * rhs, m_data[1] * rhs, m_data[2] * rhs, m_data[3] * rhs );
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T>& QuatT<T>::operator *= ( T rhs )
  {
    m_data[0] *= rhs;
    m_data[1] *= rhs;
    m_data[2] *= rhs;
    m_data[3] *= rhs;
    return *this;        // enable nesting
  }

  template <typename T>
  AGX_FORCE_INLINE const QuatT<T> QuatT<T>::operator*( const QuatT<T>& rhs ) const
  {
    return QuatT( rhs.m_data[3] * m_data[0] + rhs.m_data[0] * m_data[3] + rhs.m_data[1] * m_data[2] - rhs.m_data[2] * m_data[1],
                 rhs.m_data[3] * m_data[1] - rhs.m_data[0] * m_data[2] + rhs.m_data[1] * m_data[3] + rhs.m_data[2] * m_data[0],
                 rhs.m_data[3] * m_data[2] + rhs.m_data[0] * m_data[1] - rhs.m_data[1] * m_data[0] + rhs.m_data[2] * m_data[3],
                 rhs.m_data[3] * m_data[3] - rhs.m_data[0] * m_data[0] - rhs.m_data[1] * m_data[1] - rhs.m_data[2] * m_data[2] );
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T>& QuatT<T>::operator*=( const QuatT<T>& rhs )
  {
    T x = rhs.m_data[3] * m_data[0] + rhs.m_data[0] * m_data[3] + rhs.m_data[1] * m_data[2] - rhs.m_data[2] * m_data[1];
    T y = rhs.m_data[3] * m_data[1] - rhs.m_data[0] * m_data[2] + rhs.m_data[1] * m_data[3] + rhs.m_data[2] * m_data[0];
    T z = rhs.m_data[3] * m_data[2] + rhs.m_data[0] * m_data[1] - rhs.m_data[1] * m_data[0] + rhs.m_data[2] * m_data[3];
    m_data[3]   = rhs.m_data[3] * m_data[3] - rhs.m_data[0] * m_data[0] - rhs.m_data[1] * m_data[1] - rhs.m_data[2] * m_data[2];

    m_data[2] = z;
    m_data[1] = y;
    m_data[0] = x;

    return ( *this );          // enable nesting
  }

  template <typename T>
  AGX_FORCE_INLINE const QuatT<T> QuatT<T>::leftMult( const QuatT<T>& rhs ) const
  {
    return (*this) * rhs ;
  }

  template <typename T>
  AGX_FORCE_INLINE const QuatT<T> QuatT<T>::rightMult( const QuatT<T>& rhs ) const
  {
    return rhs * (*this) ;
  }


  template <typename T>
  AGX_FORCE_INLINE QuatT<T> QuatT<T>::operator / ( T rhs ) const
  {
    T div = T(1) / rhs;
    return QuatT( m_data[0] * div, m_data[1] * div, m_data[2] * div, m_data[3] * div );
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T>& QuatT<T>::operator /= ( T rhs )
  {
    T div = T(1) / rhs;
    m_data[0] *= div;
    m_data[1] *= div;
    m_data[2] *= div;
    m_data[3] *= div;
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE const QuatT<T> QuatT<T>::operator/( const QuatT<T>& denom ) const
  {
    return ( ( *this ) * denom.inverse() );
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T>& QuatT<T>::operator/=( const QuatT<T>& denom )
  {
    ( *this ) = ( *this ) * denom.inverse();
    return ( *this );          // enable nesting
  }

  template <typename T>
  AGX_FORCE_INLINE const QuatT<T> QuatT<T>::operator + ( const QuatT<T>& rhs ) const
  {
    return QuatT( m_data[0] + rhs.m_data[0], m_data[1] + rhs.m_data[1],
                 m_data[2] + rhs.m_data[2], m_data[3] + rhs.m_data[3] );
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T>& QuatT<T>::operator += ( const QuatT<T>& rhs )
  {
    m_data[0] += rhs.m_data[0];
    m_data[1] += rhs.m_data[1];
    m_data[2] += rhs.m_data[2];
    m_data[3] += rhs.m_data[3];
    return *this;            // enable nesting
  }

  template <typename T>
  AGX_FORCE_INLINE const QuatT<T> QuatT<T>::operator - ( const QuatT<T>& rhs ) const
  {
    return QuatT( m_data[0] - rhs.m_data[0], m_data[1] - rhs.m_data[1],
                 m_data[2] - rhs.m_data[2], m_data[3] - rhs.m_data[3] );
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T>& QuatT<T>::operator -= ( const QuatT<T>& rhs )
  {
    m_data[0] -= rhs.m_data[0];
    m_data[1] -= rhs.m_data[1];
    m_data[2] -= rhs.m_data[2];
    m_data[3] -= rhs.m_data[3];
    return *this;            // enable nesting
  }

  template <typename T>
  AGX_FORCE_INLINE const QuatT<T> QuatT<T>::operator - () const
  {
    return QuatT<T> ( -m_data[0], -m_data[1], -m_data[2], -m_data[3] );
  }

  template <typename T>
  AGX_FORCE_INLINE T QuatT<T>::length() const
  {
    return std::sqrt( m_data[0] * m_data[0] + m_data[1] * m_data[1] + m_data[2] * m_data[2] + m_data[3] * m_data[3] );
  }

  template <typename T>
  AGX_FORCE_INLINE T QuatT<T>::length2() const
  {
    return m_data[0] * m_data[0] + m_data[1] * m_data[1] + m_data[2] * m_data[2] + m_data[3] * m_data[3];
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T> QuatT<T>::conj () const
  {
    return QuatT( -m_data[0], -m_data[1], -m_data[2], m_data[3] );
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T> QuatT<T>::inverse () const
  {
    return conj() / length2();
  }


  template <typename T>
  AGX_FORCE_INLINE QuatT<T> QuatT<T>::rotate( T angle, T x, T y, T z )
  {
    return rotate( angle, Vec3T<T>( x, y, z ) );
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T> QuatT<T>::rotate( T angle, const Vec3T<T>& vec )
  {
    QuatT<T> q;
    q.setRotate( angle, vec );
    return q;
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T> QuatT<T>::rotate( T angle1, const Vec3T<T>& axis1,
                                      T angle2, const Vec3T<T>& axis2,
                                      T angle3, const Vec3T<T>& axis3 )
  {
    QuatT<T> q;
    q.setRotate( angle1, axis1, angle2, axis2, angle3, axis3 );
    return q;
  }

  template <typename T>
  AGX_FORCE_INLINE QuatT<T> QuatT<T>::rotate( const Vec3T<T>& from, const Vec3T<T>& to )
  {
    QuatT<T> q;
    q.setRotate( from, to );
    return q;
  }


  template <typename T>
  inline Vec3T<T> QuatT<T>::operator* ( const Vec3T<T>& v ) const
  {
    // nVidia SDK implementation
    Vec3T<T> uv, uuv;
    Vec3T<T> qvec( m_data[0], m_data[1], m_data[2] );
    uv = qvec ^ v;
    uuv = qvec ^ uv;
    uv *= ( 2.0f * m_data[3] );
    uuv *= 2.0f;
    return v + uv + uuv;
  }

  template <typename T>
  AGX_FORCE_INLINE T QuatT<T>::normalize()
  {
    T norm = this->length();
    if ( norm > 0.0 ) {
      T inv = 1.0f / norm;
      m_data[0] *= inv;
      m_data[1] *= inv;
      m_data[2] *= inv;
      m_data[3] *= inv;
    }
    return( norm );
  }


  template <typename T>
  AGX_FORCE_INLINE T QuatT<T>::at( size_t i )
  {
    return (*this)[i];
  }

  template <typename T>
  AGX_FORCE_INLINE void QuatT<T>::set( T val, size_t i )
  {
    (*this)[i] = val;
  }

  template <typename T>
  AGX_FORCE_INLINE T* QuatT<T>::ptr()
  {
    return m_data;
  }

  template <typename T>
  AGX_FORCE_INLINE const T* QuatT<T>::ptr() const
  {
    return m_data;
  }


  template <typename T>
  AGX_FORCE_INLINE bool equivalent( const agx::QuatT<T>& a, const agx::QuatT<T>& b, T epsilon = T(AGX_EQUIVALENT_EPSILON))
  {
    return
      agx::equivalent(a[0], b[0], epsilon) &&
      agx::equivalent(a[1], b[1], epsilon) &&
      agx::equivalent(a[2], b[2], epsilon) &&
      agx::equivalent(a[3], b[3], epsilon);
  }


  template <typename T>
  inline std::ostream& operator << ( std::ostream& output, const QuatT<T>& quat )
  {
    output << quat[0] << " "
           << quat[1] << " "
           << quat[2] << " "
           << quat[3];
    return output;     // to enable cascading
  }

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif /* AGX_QUATTEMPLATE_H */
