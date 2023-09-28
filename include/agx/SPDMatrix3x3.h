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
#include <agx/AffineMatrix4x4.h>
#include <agx/Matrix3x3.h>
#include <agxData/Type.h>

namespace agx
{
  template<typename T>
  class AffineMatrix4x4T;

  /**
  Specialized type of matrices for holding symmetric positive definite matrices

  \todo Add optimized method for Matrix3x3 * SPD * Matrix3x3^T ?

  Striped down Matrix3x3 with some added stuff to handle:
    matrix + - matrix
    matrix * / *= /= scalar
    scalar * matrix

    Stored in the following order:

    0 1 2
    - 5 3
    - - 4

  */
  class AGXCORE_EXPORT SPDMatrix3x3
  {
  public:
    typedef Real Type;
    SPDMatrix3x3();
    SPDMatrix3x3( const SPDMatrix3x3& mat );

    SPDMatrix3x3( Real a00, Real a01, Real a02,
                  Real a10, Real a11, Real a12,
                  Real a20, Real a21, Real a22);

    explicit SPDMatrix3x3( const Vec3& v );
    explicit SPDMatrix3x3( const Matrix3x3& m );
    explicit SPDMatrix3x3( const AffineMatrix4x4T<Real>& m );

    bool operator == ( const SPDMatrix3x3& m ) const;
    bool operator != ( const SPDMatrix3x3& m ) const;

    Real& operator()( size_t row, size_t col );
    Real operator()( size_t row, size_t col ) const;

#ifndef SWIG
    Real& at( size_t row, size_t col );
#endif
    Real at( size_t row, size_t col ) const;

    bool isValid() const;
    bool isNaN() const;

    bool isIdentity() const;
    void makeIdentity();

    SPDMatrix3x3& operator = ( const SPDMatrix3x3& rhs );
    void set( Real a00, Real a01, Real a02,
              Real a10, Real a11, Real a12,
              Real a20, Real a21, Real a22);
    void set(const Matrix3x3& m);
    void set(const AffineMatrix4x4T<Real>& m);
    void set(const Vec3& v);

    /**
    Set a single element of the matrix
    \param v Value of the element
    \param row Matrix row
    \param col Matrix column
    */
    void set(Real v, size_t row, size_t col);

    Vec3 getDiagonal() const;
    Real determinant() const;
    SPDMatrix3x3 inverse() const;

    void mult( const SPDMatrix3x3& m, const Real& r);
    void add( const SPDMatrix3x3& a, const SPDMatrix3x3& b);
    void sub( const SPDMatrix3x3& a, const SPDMatrix3x3& b);

    void operator *= ( const Real& r );
    void operator /= ( const Real& d);
    void operator += ( const SPDMatrix3x3& m );
    void operator -= ( const SPDMatrix3x3& m );
    SPDMatrix3x3 operator * ( const Real& r ) const;
    SPDMatrix3x3 operator / ( const Real& r ) const;
    SPDMatrix3x3 operator + ( const SPDMatrix3x3& m ) const;
    SPDMatrix3x3 operator - ( const SPDMatrix3x3& m ) const;

    Vec3 preMult( const Vec3& v ) const;
    Vec3 postMult( const Vec3& v ) const;
    Vec3 operator * ( const Vec3& v ) const;


    Real* ptr();
    const Real* ptr() const;

  protected:
    template <typename T>
    friend class Matrix3x3T;


  protected:
    Type m_data[6];
  };

  /* Implementation */

  AGX_FORCE_INLINE SPDMatrix3x3::SPDMatrix3x3()
  {
    this->makeIdentity();
  }

  AGX_FORCE_INLINE SPDMatrix3x3::SPDMatrix3x3( const SPDMatrix3x3& mat )
  {
    *this = mat;
  }

  AGX_FORCE_INLINE SPDMatrix3x3::SPDMatrix3x3(
    Real a00, Real a01, Real a02,
    Real a10, Real a11, Real a12,
    Real a20, Real a21, Real a22)
  {
    this->set(
      a00, a01, a02,
      a10, a11, a12,
      a20, a21, a22);
  }


  AGX_FORCE_INLINE SPDMatrix3x3::SPDMatrix3x3( const Vec3& v )
  {
    this->set(v);
  }

  AGX_FORCE_INLINE bool SPDMatrix3x3::operator == ( const SPDMatrix3x3& m ) const
  {
    return m_data[0] == m.m_data[0] &&
           m_data[1] == m.m_data[1] &&
           m_data[2] == m.m_data[2] &&
           m_data[3] == m.m_data[3] &&
           m_data[4] == m.m_data[4] &&
           m_data[5] == m.m_data[5];
  }

  AGX_FORCE_INLINE bool SPDMatrix3x3::operator != ( const SPDMatrix3x3& m ) const
  {
    return !(*this == m);
  }


  AGX_FORCE_INLINE Real& SPDMatrix3x3::at( size_t row, size_t col )
  {
    agxAssert(row < 3);
    agxAssert(col < 3);
    return this->operator()(row, col);
  }


  AGX_FORCE_INLINE Real SPDMatrix3x3::at( size_t row, size_t col ) const
  {
    agxAssert(row < 3);
    agxAssert(col < 3);
    return this->operator()(row, col);
  }


  AGX_FORCE_INLINE Real& SPDMatrix3x3::operator()( size_t row, size_t col )
  {
    size_t index = row + col;
    if (index == 2 && row == 1)
      index = 5;
    return m_data[index];
  }

  AGX_FORCE_INLINE Real SPDMatrix3x3::operator()( size_t row, size_t col ) const
  {
    size_t index = row + col;
    if (index == 2 && row == 1)
      index = 5;
    return m_data[index];
  }


  AGX_FORCE_INLINE bool SPDMatrix3x3::isValid() const
  {
    // Matrix can not be NaN
    if (this->isNaN())
      return false;
    // Check that it is positive definite
    // by checking that all 3 principal minors are positive;
    // i.e. top left matrix 1x1, 2x2 and 3x3 determinants must be positive
    // 1x1
    if (m_data[0] <= 0)
      return false;
    // 2x2
    if ((m_data[0] * m_data[5]) - (m_data[1] * m_data[1]) <= 0)
      return false;
    // 3x3
    if (determinant() <= 0)
      return false;
    return true;
  }


  AGX_FORCE_INLINE bool SPDMatrix3x3::isNaN() const
  {
    return std::isnan( m_data[0] ) || std::isnan( m_data[1] ) || std::isnan( m_data[2] )  ||
           std::isnan( m_data[3] ) || std::isnan( m_data[4] ) || std::isnan( m_data[5] );
  }

  AGX_FORCE_INLINE bool SPDMatrix3x3::isIdentity() const
  {
    return *this == SPDMatrix3x3();
  }

  AGX_FORCE_INLINE void SPDMatrix3x3::makeIdentity()
  {
    m_data[0] = 1.0;
    m_data[1] = 0;
    m_data[2] = 0;
    m_data[3] = 0;
    m_data[4] = 1.0;
    m_data[5] = 1.0;
  }


  AGX_FORCE_INLINE SPDMatrix3x3& SPDMatrix3x3::operator = ( const SPDMatrix3x3& rhs )
  {
    m_data[0] = rhs.m_data[0];
    m_data[1] = rhs.m_data[1];
    m_data[2] = rhs.m_data[2];
    m_data[3] = rhs.m_data[3];
    m_data[4] = rhs.m_data[4];
    m_data[5] = rhs.m_data[5];

    return *this;
  }

// #define AGX_SPDMATRIX3X3_VERIFY_VALIDITY in order to get validity asserts.
#if (defined AGX_DEBUG && defined AGX_SPDMATRIX3X3_VERIFY_VALIDITY)
  AGX_FORCE_INLINE void SPDMatrix3x3::set(
    Real a00, Real a01, Real a02,
    Real a10, Real a11, Real a12,
    Real a20, Real a21, Real a22)
  {
    agxAssert(
      relativelyEquivalent(a01, a10) &&
      relativelyEquivalent(a02, a20) &&
      relativelyEquivalent(a12, a21));

    m_data[0] = a00;
    m_data[1] = a01;
    m_data[2] = a02;
    m_data[3] = a12;
    m_data[4] = a22;
    m_data[5] = a11;

    agxAssert(this->isValid());
  }
#else
  AGX_FORCE_INLINE void SPDMatrix3x3::set(
    Real a00, Real a01, Real a02,
    Real /*a10*/, Real a11, Real a12,
    Real /*a20*/, Real /*a21*/, Real a22)
  {
    m_data[0] = a00;
    m_data[1] = a01;
    m_data[2] = a02;
    m_data[3] = a12;
    m_data[4] = a22;
    m_data[5] = a11;
  }
#endif

  AGX_FORCE_INLINE void SPDMatrix3x3::set( const Vec3& v )
  {
    m_data[0] = v[0];
    m_data[1] = 0;
    m_data[2] = 0;
    m_data[3] = 0;
    m_data[4] = v[2];
    m_data[5] = v[1];
  }



  AGX_FORCE_INLINE void SPDMatrix3x3::set(Real v, size_t row, size_t col)
  {
    size_t index = row + col;

    // Per definition, (row, col) = (1, 1) corresponds to index 5 (see top of file)
    if (index == 2 && row == 1)
      index = 5;

    m_data[index] = v;
  }



  AGX_FORCE_INLINE Vec3 SPDMatrix3x3::getDiagonal() const
  {
    return Vec3( m_data[0], m_data[5], m_data[4] );
  }

  AGX_FORCE_INLINE Real SPDMatrix3x3::determinant() const
  {
    return m_data[0]*m_data[5]*m_data[4] - m_data[0]*m_data[3]*m_data[3] -
           m_data[1]*m_data[1]*m_data[4] + m_data[1]*m_data[3]*m_data[2] +
           m_data[2]*m_data[1]*m_data[3] - m_data[2]*m_data[5]*m_data[2];
  }

  AGX_FORCE_INLINE Vec3 SPDMatrix3x3::postMult( const Vec3& v ) const
  {
    return Vec3( ( m_data[0]*v.x() + m_data[1]*v.y() + m_data[2]*v.z() ),
                 ( m_data[1]*v.x() + m_data[5]*v.y() + m_data[3]*v.z() ),
                 ( m_data[2]*v.x() + m_data[3]*v.y() + m_data[4]*v.z() ) ) ;
  }

  AGX_FORCE_INLINE Vec3 SPDMatrix3x3::preMult( const Vec3& v ) const
  {
    return Vec3( ( m_data[0]*v.x() + m_data[1]*v.y() + m_data[2]*v.z() ),
                 ( m_data[1]*v.x() + m_data[5]*v.y() + m_data[3]*v.z() ),
                 ( m_data[2]*v.x() + m_data[3]*v.y() + m_data[4]*v.z() ) ) ;
  }

  AGX_FORCE_INLINE void SPDMatrix3x3::mult( const SPDMatrix3x3& m, const Real& r)
  {
    m_data[0] = m.m_data[0] * r;
    m_data[1] = m.m_data[1] * r;
    m_data[2] = m.m_data[2] * r;
    m_data[3] = m.m_data[3] * r;
    m_data[4] = m.m_data[4] * r;
    m_data[5] = m.m_data[5] * r;
  }

  AGX_FORCE_INLINE void SPDMatrix3x3::add( const SPDMatrix3x3& a, const SPDMatrix3x3& b)
  {
    m_data[0] = a.m_data[0] + b.m_data[0];
    m_data[1] = a.m_data[1] + b.m_data[1];
    m_data[2] = a.m_data[2] + b.m_data[2];
    m_data[3] = a.m_data[3] + b.m_data[3];
    m_data[4] = a.m_data[4] + b.m_data[4];
    m_data[5] = a.m_data[5] + b.m_data[5];
  }

  AGX_FORCE_INLINE void SPDMatrix3x3::sub( const SPDMatrix3x3& a, const SPDMatrix3x3& b)
  {
    m_data[0] = a.m_data[0] - b.m_data[0];
    m_data[1] = a.m_data[1] - b.m_data[1];
    m_data[2] = a.m_data[2] - b.m_data[2];
    m_data[3] = a.m_data[3] - b.m_data[3];
    m_data[4] = a.m_data[4] - b.m_data[4];
    m_data[5] = a.m_data[5] - b.m_data[5];
  }

  AGX_FORCE_INLINE void SPDMatrix3x3::operator *= ( const Real& r )
  {
    this->mult(*this, r);
  }

  AGX_FORCE_INLINE void SPDMatrix3x3::operator /= ( const Real& d)
  {
    this->mult(*this, agx::Real(1.0)/d);
  }

  AGX_FORCE_INLINE void SPDMatrix3x3::operator += ( const SPDMatrix3x3& m )
  {
    this->add(*this, m);
  }

  AGX_FORCE_INLINE void SPDMatrix3x3::operator -= ( const SPDMatrix3x3& m )
  {
    this->sub(*this, m);
  }

  AGX_FORCE_INLINE SPDMatrix3x3 SPDMatrix3x3::operator * ( const Real& r ) const
  {
    SPDMatrix3x3 t;
    t.mult( *this, r );
    return  t;
  }

  AGX_FORCE_INLINE SPDMatrix3x3 SPDMatrix3x3::operator / ( const Real& r ) const
  {
    SPDMatrix3x3 t;
    t.mult( *this, agx::Real(1.0)/r );
    return t;
  }

  AGX_FORCE_INLINE SPDMatrix3x3 SPDMatrix3x3::operator + ( const SPDMatrix3x3& m ) const
  {
    SPDMatrix3x3 t;
    t.add( *this, m );
    return t;
  }

  AGX_FORCE_INLINE SPDMatrix3x3 SPDMatrix3x3::operator - ( const SPDMatrix3x3& m ) const
  {
    SPDMatrix3x3 t;
    t.sub( *this, m );
    return t;
  }

  AGX_FORCE_INLINE Vec3 operator* ( const Vec3& v, const SPDMatrix3x3& m )
  {
    return m.preMult( v );
  }

  AGX_FORCE_INLINE Vec3 SPDMatrix3x3::operator* ( const Vec3& v ) const
  {
    return postMult( v );
  }


  AGX_FORCE_INLINE SPDMatrix3x3 operator* ( const Real& r, const SPDMatrix3x3& m )
  {
    SPDMatrix3x3 t;
    t.mult(m, r);
    return t;
  }


  AGX_FORCE_INLINE bool equivalent( const SPDMatrix3x3& m1, const SPDMatrix3x3& m2, Real epsilon = Real(AGX_EQUIVALENT_EPSILON) )
  {
    return (equivalent(m1(0, 0), m2(0, 0), epsilon) &&
            equivalent(m1(0, 1), m2(0, 1), epsilon) &&
            equivalent(m1(0, 2), m2(0, 2), epsilon) &&
            equivalent(m1(1, 1), m2(1, 1), epsilon) &&
            equivalent(m1(1, 2), m2(1, 2), epsilon) &&
            equivalent(m1(2, 2), m2(2, 2), epsilon));
  }


  AGX_FORCE_INLINE Real* SPDMatrix3x3::ptr()
  {
    return m_data;
  }

  AGX_FORCE_INLINE const Real* SPDMatrix3x3::ptr() const
  {
    return m_data;
  }

  inline std::ostream& operator<< ( std::ostream& os, const SPDMatrix3x3& m )
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

AGX_TYPE_BINDING(agx::SPDMatrix3x3, "SPDMatrix3x3")

