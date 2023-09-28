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

#include <agx/agxCore_export.h>

#include <agx/Vec3.h>
#include <agx/Quat.h>

#include <cstring>

namespace agx
{

  class EulerAngles;

  /**
  Specialized types of matrices for holding orthogonal transformation matrices.
  */
  class AGXCORE_EXPORT OrthoMatrix3x3
  {
  public:
    /**
    Default constructor
    */
    inline OrthoMatrix3x3() {
      setIdentity();
    }

    /**
    Copy constructor
    */
    inline OrthoMatrix3x3( const OrthoMatrix3x3& mat ) {
      set( mat.ptr() );
    }
    inline explicit OrthoMatrix3x3( Real const * const ptr ) {
      set( ptr );
    }
    inline explicit OrthoMatrix3x3( const Quat& quat ) {
      set( quat );
    }

    inline explicit OrthoMatrix3x3( EulerAngles& euler ) {
      set( euler );
    }


    /**
    Set the row of the matrix,
    \param row must be between [0..2]
    \param vec - value of the row
    */
    inline void setRow(size_t row, const agx::Vec3& vec ) {
      m_data[row][0] = vec[0];
      m_data[row][1] = vec[1];
      m_data[row][2] = vec[2];
    }

    /**
    Return the row of the matrix,
    \param row must be between [0..2]
    */
    inline Vec3 getRow(size_t row ) const {
      return Vec3(m_data[row][0], m_data[row][1], m_data[row][2]);
    }


    /**
    Set the column of the matrix,
    \param col must be between [0..2]
    \param vec - value of the column
    */
    inline void setColumn(size_t col, const agx::Vec3& vec ) {
      m_data[0][col] = vec[0];
      m_data[1][col] = vec[1];
      m_data[2][col] = vec[2];
    }


    OrthoMatrix3x3( Real a00, Real a01, Real a02,
                    Real a10, Real a11, Real a12,
                    Real a20, Real a21, Real a22);

    ~OrthoMatrix3x3() {}


    bool operator== ( const OrthoMatrix3x3& m ) const {
      const int bitCompare = memcmp(m_data, m.m_data, sizeof(m_data));
      return (bitCompare == 0);
    }

    bool operator!= ( const OrthoMatrix3x3& m ) const {
      const int bitCompare = memcmp(m_data, m.m_data, sizeof(m_data));
      return (bitCompare != 0);
    }

    inline Real& operator()( size_t row, size_t col ) {
      return m_data[row][col];
    }
    inline Real operator()( size_t row, size_t col ) const {
      return m_data[row][col];
    }

    inline bool isValid() const {
      return !isNaN();
    }
    inline bool isNaN() const {
      return std::isnan( m_data[0][0] ) || std::isnan( m_data[0][1] ) || std::isnan( m_data[0][2] )  ||
             std::isnan( m_data[1][0] ) || std::isnan( m_data[1][1] ) || std::isnan( m_data[1][2] ) ||
             std::isnan( m_data[2][0] ) || std::isnan( m_data[2][1] ) || std::isnan( m_data[2][2] ) ;
    }

    bool isIdentity() const;

    inline OrthoMatrix3x3 inverse() const;
    inline OrthoMatrix3x3 transpose() const;


    inline OrthoMatrix3x3& operator= ( const OrthoMatrix3x3& rhs ) {
      if ( &rhs == this ) return *this;
      set( rhs.ptr() );
      return *this;
    }

    inline void set( const OrthoMatrix3x3& rhs ) {
      set( rhs.ptr() );
    }

    inline void set( Real const * const ptr ) {
      Real* local_ptr = ( Real* )m_data;
      for ( size_t i = 0;i < 9;++i ) local_ptr[i] = ( Real )ptr[i];
    }


    void set( Real a00, Real a01, Real a02,
              Real a10, Real a11, Real a12,
              Real a20, Real a21, Real a22);

    /**
    Set the value of this matrix from the specified quaternion
    \return reference to this matrix
    */
    OrthoMatrix3x3& set( const Quat& q_ );

    /**
    Set the quaternion with the rotation from this matrix
    */
   void get( Quat& q ) const;
   inline Quat get( ) const;

    /**
    Set the rotation of matrix to be the rotation specified in the given EulerAngles object.
    \return a reference to the updated matrix.
    */
    OrthoMatrix3x3& set( EulerAngles& e );

    /**
    Convert the rotation of this matrix into a specified EulerAngles representation.
    */
    void get( EulerAngles& e ) const;

    Real * ptr() {
      return ( Real* )m_data;
    }
    const Real * ptr() const {
      return ( const Real * )m_data;
    }

    void setIdentity();

    void setRotate( const Vec3& from, const Vec3& to );
    void setRotate( Real angle, const Vec3& axis );
    void setRotate( Real angle, Real x, Real y, Real z );
    void setRotate( const Quat& );
    void setRotate( Real angle1, const Vec3& axis1,
                     Real angle2, const Vec3& axis2,
                     Real angle3, const Vec3& axis3 );


    //basic utility functions to create new matrices
    inline static OrthoMatrix3x3 rotate( const Vec3& from, const Vec3& to );
    inline static OrthoMatrix3x3 rotate( Real angle, Real x, Real y, Real z );
    inline static OrthoMatrix3x3 rotate( Real angle, const Vec3& axis );
    inline static OrthoMatrix3x3 rotate( Real angle1, const Vec3& axis1,
                                         Real angle2, const Vec3& axis2,
                                         Real angle3, const Vec3& axis3 );


    inline Vec3 preMult( const Vec3& v ) const;
    inline Vec3 postMult( const Vec3& v ) const;
    inline Vec3 operator* ( const Vec3& v ) const;

    inline Vec3 getScale() const {
      return Vec3( m_data[0][0], m_data[1][1], m_data[2][2] );
    }


    // basic OrthoMatrix3x3 multiplication, our workhorse methods.
    void mult( const OrthoMatrix3x3&, const OrthoMatrix3x3& );
    void preMult( const OrthoMatrix3x3& );
    void postMult( const OrthoMatrix3x3& );

    inline void operator*= ( const OrthoMatrix3x3& other ) {
      if ( this == &other ) {
        OrthoMatrix3x3 temp( other );
        postMult( temp );
      }
      else postMult( other );
    }

    inline OrthoMatrix3x3 operator* ( const OrthoMatrix3x3 &m ) const {
      OrthoMatrix3x3 r;
      r.mult( *this, m );
      return  r;
    }

  protected:

    bool invert( const OrthoMatrix3x3& rhs );
    bool invert_3x3_new(const agx::OrthoMatrix3x3&);

    Real m_data[3][3];
  };


  inline Quat OrthoMatrix3x3::get( ) const
  {
    Quat q;
    get( q );
    return q;
  }

  inline OrthoMatrix3x3 OrthoMatrix3x3::rotate( Real angle, Real x, Real y, Real z )
  {
    OrthoMatrix3x3 m;
    m.setRotate( angle, x, y, z );
    return m;
  }
  inline OrthoMatrix3x3 OrthoMatrix3x3::rotate( Real angle, const Vec3& axis )
  {
    OrthoMatrix3x3 m;
    m.setRotate( angle, axis );
    return m;
  }
  inline OrthoMatrix3x3 OrthoMatrix3x3::rotate( Real angle1, const Vec3& axis1,
      Real angle2, const Vec3& axis2,
      Real angle3, const Vec3& axis3 )
  {
    OrthoMatrix3x3 m;
    m.setRotate( angle1, axis1, angle2, axis2, angle3, axis3 );
    return m;
  }
  inline OrthoMatrix3x3 OrthoMatrix3x3::rotate( const Vec3& from, const Vec3& to )
  {
    OrthoMatrix3x3 m;
    m.setRotate( from, to );
    return m;
  }

  inline OrthoMatrix3x3 OrthoMatrix3x3::inverse() const
  {
    OrthoMatrix3x3 m;
    m.invert( *this );
    return m;
  }

  inline OrthoMatrix3x3 OrthoMatrix3x3::transpose() const
  {
    return OrthoMatrix3x3( m_data[0][0], m_data[1][0], m_data[2][0],
                           m_data[0][1], m_data[1][1], m_data[2][1],
                           m_data[0][2], m_data[1][2], m_data[2][2]);
  }

  inline Vec3 OrthoMatrix3x3::postMult( const Vec3& v ) const
  {
    return Vec3( ( m_data[0][0]*v.x() + m_data[0][1]*v.y() + m_data[0][2]*v.z() ),
                 ( m_data[1][0]*v.x() + m_data[1][1]*v.y() + m_data[1][2]*v.z() ),
                 ( m_data[2][0]*v.x() + m_data[2][1]*v.y() + m_data[2][2]*v.z() ) );
  }
  inline Vec3 OrthoMatrix3x3::preMult( const Vec3& v ) const
  {
    return Vec3( ( m_data[0][0]*v.x() + m_data[1][0]*v.y() + m_data[2][0]*v.z() ),
                 ( m_data[0][1]*v.x() + m_data[1][1]*v.y() + m_data[2][1]*v.z() ),
                 ( m_data[0][2]*v.x() + m_data[1][2]*v.y() + m_data[2][2]*v.z() ) );
  }

  inline Vec3 operator* ( const Vec3& v, const OrthoMatrix3x3& m )
  {
    return m.preMult( v );
  }
  inline Vec3 OrthoMatrix3x3::operator* ( const Vec3& v ) const
  {
    return postMult( v );
  }

  inline std::ostream& operator<< ( std::ostream& os, const OrthoMatrix3x3& m )
  {
    os << "{" << std::endl;
    for ( size_t row = 0; row < 3; ++row ) {
      os << "\t";
      for ( size_t col = 0; col < 3; ++col )
        os << m( row, col ) << " ";
      os << std::endl;
    }
    os << "}" << std::endl;
    return os;
  }


} // namespace agx


