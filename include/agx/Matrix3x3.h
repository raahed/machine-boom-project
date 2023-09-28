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
#include <agx/AffineMatrix4x4.h>


namespace agx
{
  class SPDMatrix3x3;
  class EulerAngles;

  template <typename T>
  class Matrix3x3T
  {
  public:
    typedef T Type;

  public:

    /**
    Default constructor, provides an identity matrix
    */
    Matrix3x3T();

    /**
    Copy constructor
    */
    Matrix3x3T( const Matrix3x3T<T>& mat );

    /**
    Init from 12 values in row major order.
    Each row includes a forth element as padding.
    */
    explicit Matrix3x3T( T const * const ptr );

    /**
    Init from the top left 3x3 part of an 4x4 matrix.
    */
    template <typename T2>
    explicit Matrix3x3T( const AffineMatrix4x4T<T2>& m );

    /**
    Init from a quaternion.
    */
    explicit Matrix3x3T( const Quat& q );

    /**
    Init from EulerAngles.
    */
    explicit Matrix3x3T(const EulerAngles& e);

    /**
    Init from a SPDMatrix3x3.
    */
    explicit Matrix3x3T( const SPDMatrix3x3& m );

    /**
    Explicitly set all values.
    */
    Matrix3x3T( T a00, T a01, T a02,
               T a10, T a11, T a12,
               T a20, T a21, T a22);

    /**
    Construct a Matrix34Init with a specified diagonal.
    Effectively specifying a scaling matrix.
    */
    explicit Matrix3x3T( const Vec3T<T>& v );

    /**
    Cast operator.
    */
    operator AffineMatrix4x4T<T>() const;

    /**
    Equality test (no threshold, must be exact same bits).
    */
    bool operator == ( const Matrix3x3T<T>& m ) const;
    bool operator != ( const Matrix3x3T<T>& m ) const;

    /**
    Element accessors.
    */
    T& operator()( size_t row, size_t col );
    T operator()( size_t row, size_t col ) const;


    /**
    State queries.
    */
    bool isValid() const;
    bool isNaN() const;
    bool isIdentity() const;

    /**
    Set all values.
    */
    void makeIdentity();
    Matrix3x3T<T>& operator = ( const Matrix3x3T<T>& rhs );
    void set( T const * const ptr );

    /**
    Set the diagonal values of the Matrix and the rest of the elements to zero
    \param diagonal - The diagonal value of the matrix
    */
    void set( const Vec3T<T>& diagonal );

    /**
    Set all elements of the matrix
    */
    void set( T a00, T a01, T a02,
              T a10, T a11, T a12,
              T a20, T a21, T a22);

    /**
    Set the rotation from a specified quaternion
    \param q - Set the matrix from the quaternion definition
    */
    void set( const Quat& q );

    /**
    Set the rotation from a specified EulerAngles
    \param e - Set the matrix from the EulerAngles definition
    */
    void set(const EulerAngles& e);

    /**
    Access internal buffer.
    */
    T * ptr();
    const T * ptr() const;

    /**
    Get the diagonal elements.
    */
    Vec3T<T> getDiagonal() const;

    /**
    Create a cross product matrix.
    */
    static Matrix3x3T<T> crossMatrix(const Vec3T<T>& v);

    /**
    Calculate the inverse.
    */
    Matrix3x3T<T> inverse() const;

    /**
    Calculate the transpose.
    */
    Matrix3x3T<T> transpose() const;

    /**
    Calculate the determinant.
    */
    T determinant() const;

    /**
    Test if matrix is diagonal.
    */
    bool isDiagonal() const;

    /**
    Set the value of this matrix to the result of the product between the matrix \p m and the scalar \p r
    */
    void mult( const Matrix3x3T<T>& m, const T& r);

    /**
    Set the value of this matrix to the result of the sum between the matrix \a m and the matrix \p b
    */
    void add( const Matrix3x3T<T>& a, const Matrix3x3T<T>& b);

    /**
    Set the value of this matrix to the result of the subtraction between the matrix \a m and the matrix \p b
    */
    void sub( const Matrix3x3T<T>& a, const Matrix3x3T<T>& b);

    /**
    Set the value of this matrix to the result of the multiplication between the matrix \p lhs m and the matrix \p rhs
    */
    void mult(const Matrix3x3T<T>& lhs, const Matrix3x3T<T>& rhs);

    /**
    Set the value of the matrix to the result of the product between this matrix and the scalar \p r
    */
    void operator *= ( const T& r );

    /**
    Set the value of the matrix to the quotient of the division between this matrix and the divisor \p r
    */
    void operator /= ( const T& d);

    /**
    Set the value of the matrix to the result of the sum between this matrix and the matrix \p m
    */
    void operator += ( const Matrix3x3T<T>& m );

    /**
    Set the value of the matrix to the result of the subtraction between this matrix and the matrix \p m
    */
    void operator -= ( const Matrix3x3T<T>& m );

    /**
    \return the product result from the multiplication between this matrix and the scalar \p r
    */
    Matrix3x3T<T> operator * ( const T& r ) const;

    /**
    \return the quotient of the division between this matrix and the divisor \p r
    */
    Matrix3x3T<T> operator / ( const T& r ) const;

    /**
    \return the result of the addition between this matrix and the matrix \p m
    */
    Matrix3x3T<T> operator + ( const Matrix3x3T<T>& m ) const;

    /**
    \return the result of the subtraction between this matrix and the matrix \p m
    */
    Matrix3x3T<T> operator - ( const Matrix3x3T<T>& m ) const;

    /**
    \return the result of the product of the vector \p v and this matrix as v * M
    */
    Vec3T<T> preMult( const Vec3T<T>& v ) const;

    /**
    \return the result of the product of the vector \p v and this matrix as M * v
    */
    Vec3T<T> postMult( const Vec3T<T>& v ) const;

    /**
    Set the value of this matrix to the result of the product \p m * this
    */
    void preMult( const Matrix3x3T<T>& m );

    /**
    Set the value of this matrix to the result of the product this * \p m
    */
    void postMult( const Matrix3x3T<T>& m );

    /**
    \return a scalar which is the result from the dot product between \p vec and the \p col :th column of this matrix
    */
    T dotColumn(const Vec3T<T>& vec, size_t col) const;

    /**
    \returns the post multiplication product between this matrix and the vector \p v
    */
    Vec3T<T> operator* ( const Vec3T<T>& v ) const;


    /**
    \returns the result from the multiplication this * \param m
    */
    Matrix3x3T<T> operator * ( const Matrix3x3T<T> &m ) const;


    /// \return the element at i,j. Mainly for accessing matrix in Lua.
    AGX_FORCE_INLINE Real at(size_t i, size_t j) { return Real((*this)(i,j)); }

    /// Set the value of element i,j. Mainly for accessing matrix in Lua.
    AGX_FORCE_INLINE void set(Real val, size_t i, size_t j) { (*this)(i,j) = T(val); }


  private:
    Matrix3x3T<T> fastInverse(T determinant) const;

  protected:
    T m_data[3][4];
  };


  typedef Matrix3x3T<Real> Matrix3x3;

  typedef Matrix3x3T<Real32> Matrix3x3f;
  typedef Matrix3x3T<Real64> Matrix3x3d;

 }

#include <agx/SPDMatrix3x3.h>

namespace agx {
  /* Implementation */

#define DET2(a, b, c, d) ((a)*(d) - (b)*(c))

#define SET_ROW(row, v1, v2, v3 )    \
  {                                    \
    m_data[(row)][0] = (v1);             \
    m_data[(row)][1] = (v2);             \
    m_data[(row)][2] = (v3);             \
    m_data[(row)][3] = ( 0);             \
  }

  #define INNER_PRODUCT(a,b,r,c)         \
       ((a).m_data[r][0] * (b).m_data[0][c]) \
      +((a).m_data[r][1] * (b).m_data[1][c]) \
      +((a).m_data[r][2] * (b).m_data[2][c])

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T>::Matrix3x3T()
  {
    this->makeIdentity();
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T>::Matrix3x3T( const Matrix3x3T<T>& mat )
  {
    this->set(mat.ptr());
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T>::Matrix3x3T( T const * const ptr )
  {
    this->set(ptr);
  }

  template <typename T> template <typename T2>
  AGX_FORCE_INLINE Matrix3x3T<T>::Matrix3x3T( const AffineMatrix4x4T<T2>& m )
  {
    m_data[0][0] = (T)m(0,0);
    m_data[0][1] = (T)m(0,1);
    m_data[0][2] = (T)m(0,2);
    m_data[0][3] = 0;
    m_data[1][0] = (T)m(1,0);
    m_data[1][1] = (T)m(1,1);
    m_data[1][2] = (T)m(1,2);
    m_data[1][3] = 0;
    m_data[2][0] = (T)m(2,0);
    m_data[2][1] = (T)m(2,1);
    m_data[2][2] = (T)m(2,2);
    m_data[2][3] = 0;
  }



  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T>::Matrix3x3T( const Quat& q )
  {
    this->set(q);
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T>::Matrix3x3T(const EulerAngles& e)
  {
    this->set(e);
  }


  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T>::Matrix3x3T( T a00, T a01, T a02,
                  T a10, T a11, T a12,
                  T a20, T a21, T a22)
  {
    m_data[0][0] = a00;
    m_data[0][1] = a01;
    m_data[0][2] = a02;
    m_data[0][3] = 0;
    m_data[1][0] = a10;
    m_data[1][1] = a11;
    m_data[1][2] = a12;
    m_data[1][3] = 0;
    m_data[2][0] = a20;
    m_data[2][1] = a21;
    m_data[2][2] = a22;
    m_data[2][3] = 0;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T>::Matrix3x3T( const Vec3T<T>& v )
  {
    m_data[0][0] = v[0];
    m_data[0][1] = 0;
    m_data[0][2] = 0;
    m_data[0][3] = 0;
    m_data[1][0] = 0;
    m_data[1][1] = v[1];
    m_data[1][2] = 0;
    m_data[1][3] = 0;
    m_data[2][0] = 0;
    m_data[2][1] = 0;
    m_data[2][2] = v[2];
    m_data[2][3] = 0;
  }

  template <typename T>
  inline Matrix3x3T<T>::operator AffineMatrix4x4T<T>() const
  {
    return AffineMatrix4x4T<T>(
        m_data[0][0], m_data[0][1], m_data[0][2], 0,
        m_data[1][0], m_data[1][1], m_data[1][2], 0,
        m_data[2][0], m_data[2][1], m_data[2][2], 0,
       0, 0, 0, 1);
  }



  template <typename T>
  AGX_FORCE_INLINE bool Matrix3x3T<T>::operator == ( const Matrix3x3T<T>& m ) const
  {
    return m_data[0][0] == m.m_data[0][0] &&
           m_data[0][1] == m.m_data[0][1] &&
           m_data[0][2] == m.m_data[0][2] &&
           m_data[1][0] == m.m_data[1][0] &&
           m_data[1][1] == m.m_data[1][1] &&
           m_data[1][2] == m.m_data[1][2] &&
           m_data[2][0] == m.m_data[2][0] &&
           m_data[2][1] == m.m_data[2][1] &&
           m_data[2][2] == m.m_data[2][2];
  }

  template <typename T>
  AGX_FORCE_INLINE bool Matrix3x3T<T>::operator != ( const Matrix3x3T<T>& m ) const
  {
    return !(*this == m);
  }

  template <typename T>
  AGX_FORCE_INLINE T& Matrix3x3T<T>::operator()( size_t row, size_t col )
  {
    return m_data[row][col];
  }

  template <typename T>
  AGX_FORCE_INLINE T Matrix3x3T<T>::operator()( size_t row, size_t col ) const
  {
    return m_data[row][col];
  }


  template <typename T>
  AGX_FORCE_INLINE bool Matrix3x3T<T>::isValid() const
  {
    return !this->isNaN();
  }

  template <typename T>
  AGX_FORCE_INLINE bool Matrix3x3T<T>::isNaN() const
  {
    return std::isnan( m_data[0][0] ) || std::isnan( m_data[0][1] ) || std::isnan( m_data[0][2] )  ||
           std::isnan( m_data[1][0] ) || std::isnan( m_data[1][1] ) || std::isnan( m_data[1][2] ) ||
           std::isnan( m_data[2][0] ) || std::isnan( m_data[2][1] ) || std::isnan( m_data[2][2] ) ;
  }

  template <typename T>
  AGX_FORCE_INLINE bool Matrix3x3T<T>::isIdentity() const
  {
    return *this == Matrix3x3T();
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::makeIdentity()
  {
    SET_ROW(0, 1, 0, 0);
    SET_ROW(1, 0, 1, 0);
    SET_ROW(2, 0, 0, 1);
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T>& Matrix3x3T<T>::operator = ( const Matrix3x3T<T>& rhs )
  {
    this->set(rhs.ptr());
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::set( T const * const ptr )
  {
    #if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstringop-overflow"
    #endif

    memcpy(m_data, ptr, sizeof(T) * 12);

    #if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic pop
    #endif
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::set( T a00, T a01, T a02,
            T a10, T a11, T a12,
            T a20, T a21, T a22)
  {
    m_data[0][0] = a00;
    m_data[0][1] = a01;
    m_data[0][2] = a02;
    m_data[1][0] = a10;
    m_data[1][1] = a11;
    m_data[1][2] = a12;
    m_data[2][0] = a20;
    m_data[2][1] = a21;
    m_data[2][2] = a22;
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::set( const Vec3T<T>& diagonal )
  {
    m_data[0][0] = diagonal[0];
    m_data[0][1] = T(0);
    m_data[0][2] = T(0);
    m_data[1][0] = T(0);
    m_data[1][1] = diagonal[1];
    m_data[1][2] = T(0);
    m_data[2][0] = T(0);
    m_data[2][1] = T(0);
    m_data[2][2] = diagonal[2];
  }

  template <typename T>
  AGX_FORCE_INLINE T *Matrix3x3T<T>::ptr()
  {
    return (T *)m_data;
  }

  template <typename T>
  AGX_FORCE_INLINE const T *Matrix3x3T<T>::ptr() const
  {
    return (const T *)m_data;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Matrix3x3T<T>::getDiagonal() const
  {
    return Vec3T<T>( m_data[0][0], m_data[1][1], m_data[2][2] );
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T> Matrix3x3T<T>::crossMatrix(const Vec3T<T>& v)
  {
    return Matrix3x3T( 0, -v.z(), v.y(),
                         v.z(), 0, -v.x(),
                         -v.y(), v.x(), 0);
  }

  template <typename T>
  AGX_FORCE_INLINE T Matrix3x3T<T>::determinant() const
  {
    return m_data[0][0]*m_data[1][1]*m_data[2][2] - m_data[0][0]*m_data[1][2]*m_data[2][1] -
           m_data[0][1]*m_data[1][0]*m_data[2][2] + m_data[0][1]*m_data[1][2]*m_data[2][0] +
           m_data[0][2]*m_data[1][0]*m_data[2][1] - m_data[0][2]*m_data[1][1]*m_data[2][0];
  }

  template <typename T>
  AGX_FORCE_INLINE bool Matrix3x3T<T>::isDiagonal() const
  {
    return m_data[0][1] == T(0) && m_data[0][2] == T(0) &&
           m_data[1][0] == T(0) && m_data[1][2] == T(0) &&
           m_data[2][0] == T(0) && m_data[2][1] == T(0);
  }


  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T> Matrix3x3T<T>::transpose() const
  {
    return Matrix3x3T(m_data[0][0], m_data[1][0], m_data[2][0],
                     m_data[0][1], m_data[1][1], m_data[2][1],
                     m_data[0][2], m_data[1][2], m_data[2][2]);
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::mult( const Matrix3x3T<T>& m, const T& r)
  {
    m_data[0][0] = m(0,0) * r;
    m_data[0][1] = m(0,1) * r;
    m_data[0][2] = m(0,2) * r;
    m_data[1][0] = m(1,0) * r;
    m_data[1][1] = m(1,1) * r;
    m_data[1][2] = m(1,2) * r;
    m_data[2][0] = m(2,0) * r;
    m_data[2][1] = m(2,1) * r;
    m_data[2][2] = m(2,2) * r;
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::add( const Matrix3x3T<T>& a, const Matrix3x3T<T>& b)
  {
    m_data[0][0] = a(0,0) + b(0,0);
    m_data[0][1] = a(0,1) + b(0,1);
    m_data[0][2] = a(0,2) + b(0,2);
    m_data[1][0] = a(1,0) + b(1,0);
    m_data[1][1] = a(1,1) + b(1,1);
    m_data[1][2] = a(1,2) + b(1,2);
    m_data[2][0] = a(2,0) + b(2,0);
    m_data[2][1] = a(2,1) + b(2,1);
    m_data[2][2] = a(2,2) + b(2,2);
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::sub( const Matrix3x3T<T>& a, const Matrix3x3T<T>& b)
  {
    m_data[0][0] = a(0,0) - b(0,0);
    m_data[0][1] = a(0,1) - b(0,1);
    m_data[0][2] = a(0,2) - b(0,2);
    m_data[1][0] = a(1,0) - b(1,0);
    m_data[1][1] = a(1,1) - b(1,1);
    m_data[1][2] = a(1,2) - b(1,2);
    m_data[2][0] = a(2,0) - b(2,0);
    m_data[2][1] = a(2,1) - b(2,1);
    m_data[2][2] = a(2,2) - b(2,2);
  }


  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::operator *= ( const T& r )
  {
    this->mult(*this, r);
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::operator /= ( const T& d)
  {
    this->mult(*this, T(1.0)/d);
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::operator += ( const Matrix3x3T<T>& m )
  {
    this->add(*this, m);
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::operator -= ( const Matrix3x3T<T>& m )
  {
    this->sub(*this, m);
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T> Matrix3x3T<T>::operator * ( const T& r ) const
  {
    Matrix3x3T<T> t;
    t.mult( *this, r );
    return  t;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T> Matrix3x3T<T>::operator / ( const T& r ) const
  {
    Matrix3x3T<T> t;
    t.mult( *this, T(1.0)/r );
    return t;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T> Matrix3x3T<T>::operator + ( const Matrix3x3T<T>& m ) const
  {
    Matrix3x3T<T> t;
    t.add( *this, m );
    return t;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T> Matrix3x3T<T>::operator - ( const Matrix3x3T<T>& m ) const
  {
    Matrix3x3T<T> t;
    t.sub( *this, m );
    return t;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Matrix3x3T<T>::postMult( const Vec3T<T>& v ) const
  {
    return Vec3T<T>( ( m_data[0][0]*v.x() + m_data[0][1]*v.y() + m_data[0][2]*v.z() ),
                 ( m_data[1][0]*v.x() + m_data[1][1]*v.y() + m_data[1][2]*v.z() ),
                 ( m_data[2][0]*v.x() + m_data[2][1]*v.y() + m_data[2][2]*v.z() ) ) ;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Matrix3x3T<T>::preMult( const Vec3T<T>& v ) const
  {
    return Vec3T<T>( ( m_data[0][0]*v.x() + m_data[1][0]*v.y() + m_data[2][0]*v.z() ),
                 ( m_data[0][1]*v.x() + m_data[1][1]*v.y() + m_data[2][1]*v.z() ),
                 ( m_data[0][2]*v.x() + m_data[1][2]*v.y() + m_data[2][2]*v.z() ) );
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::preMult( const Matrix3x3T<T>& m )
  {
    T tmp[3];

    // Column 0
    tmp[0] = INNER_PRODUCT(m, *this, 0, 0);
    tmp[1] = INNER_PRODUCT(m, *this, 1, 0);
    tmp[2] = INNER_PRODUCT(m, *this, 2, 0);

    m_data[0][0] = tmp[0];
    m_data[1][0] = tmp[1];
    m_data[2][0] = tmp[2];

    // Column 1
    tmp[0] = INNER_PRODUCT(m, *this, 0, 1);
    tmp[1] = INNER_PRODUCT(m, *this, 1, 1);
    tmp[2] = INNER_PRODUCT(m, *this, 2, 1);

    m_data[0][1] = tmp[0];
    m_data[1][1] = tmp[1];
    m_data[2][1] = tmp[2];

    // Column 2
    tmp[0] = INNER_PRODUCT(m, *this, 0, 2);
    tmp[1] = INNER_PRODUCT(m, *this, 1, 2);
    tmp[2] = INNER_PRODUCT(m, *this, 2, 2);

    m_data[0][2] = tmp[0];
    m_data[1][2] = tmp[1];
    m_data[2][2] = tmp[2];
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::postMult( const Matrix3x3T<T>& m )
  {
    T tmp[3];

    // Row 0
    tmp[0] = INNER_PRODUCT(*this, m, 0, 0);
    tmp[1] = INNER_PRODUCT(*this, m, 0, 1);
    tmp[2] = INNER_PRODUCT(*this, m, 0, 2);

    m_data[0][0] = tmp[0];
    m_data[0][1] = tmp[1];
    m_data[0][2] = tmp[2];

    // Row 1
    tmp[0] = INNER_PRODUCT(*this, m, 1, 0);
    tmp[1] = INNER_PRODUCT(*this, m, 1, 1);
    tmp[2] = INNER_PRODUCT(*this, m, 1, 2);

    m_data[1][0] = tmp[0];
    m_data[1][1] = tmp[1];
    m_data[1][2] = tmp[2];

    // Row 0
    tmp[0] = INNER_PRODUCT(*this, m, 2, 0);
    tmp[1] = INNER_PRODUCT(*this, m, 2, 1);
    tmp[2] = INNER_PRODUCT(*this, m, 2, 2);

    m_data[2][0] = tmp[0];
    m_data[2][1] = tmp[1];
    m_data[2][2] = tmp[2];

  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix3x3T<T>::mult( const Matrix3x3T<T>& lhs, const Matrix3x3T<T>& rhs )
  {
    if (&lhs==this)
    {
        this->postMult(rhs);
        return;
    }
    if (&rhs==this)
    {
        this->preMult(lhs);
        return;
    }

    m_data[0][0] = INNER_PRODUCT(lhs, rhs, 0, 0);
    m_data[0][1] = INNER_PRODUCT(lhs, rhs, 0, 1);
    m_data[0][2] = INNER_PRODUCT(lhs, rhs, 0, 2);
    m_data[1][0] = INNER_PRODUCT(lhs, rhs, 1, 0);
    m_data[1][1] = INNER_PRODUCT(lhs, rhs, 1, 1);
    m_data[1][2] = INNER_PRODUCT(lhs, rhs, 1, 2);
    m_data[2][0] = INNER_PRODUCT(lhs, rhs, 2, 0);
    m_data[2][1] = INNER_PRODUCT(lhs, rhs, 2, 1);
    m_data[2][2] = INNER_PRODUCT(lhs, rhs, 2, 2);
  }

  template <typename T>
  inline T Matrix3x3T<T>::dotColumn(const Vec3T<T>& vec, size_t col) const
  {
    agxAssert(col < 3);
    return vec[0] * m_data[0][col] + vec[1] * m_data[1][col] + vec[2] * m_data[2][col];
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T> Matrix3x3T<T>::operator * ( const Matrix3x3T<T> &m ) const
  {
    Matrix3x3T<T> r;
    r.mult( *this, m );
    return  r;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> operator* ( const Vec3T<T>& v, const Matrix3x3T<T>& m )
  {
    return m.preMult( v );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Matrix3x3T<T>::operator* ( const Vec3T<T>& v ) const
  {
    return postMult( v );
  }


  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T> operator* ( const T& r, const Matrix3x3T<T>& m )
  {
    Matrix3x3T<T> t;
    t.mult(m, r);
    return t;
  }

  namespace {
    template <typename T>
    AGX_FORCE_INLINE Vec2i findMaxElement(const Matrix3x3T<T>& mat, bool skipFlags[3])
    {
      T max = 0;
      // Will return {the first non-skipped row},0 in case of an all-NaN or all-inf matrix.
      Vec2i coord(skipFlags[0] ? (skipFlags[1] ? 2 : 1) : 0, 0);

      for (int i = 0; i < 3; i++)
      {
        if (skipFlags[i])
          continue;

        for (int j = 0; j < 3; j++)
        {
          if (std::abs(mat(i, j)) > max)
          {
            coord.set(i, j);
            max = std::abs(mat(i, j));
          }
        }
      }

      agxAssert(coord[0] >= 0 && coord[1] >= 0);
      return coord;
    }
  }

  template <typename T>
  Matrix3x3T<T> Matrix3x3T<T>::inverse() const
  {
    /* Check if diagonal matrix */
    if (this->isDiagonal())
      return Matrix3x3T<T>(T(1)/m_data[0][0], 0, 0, 0, T(1)/m_data[1][1], 0, 0, 0, T(1)/m_data[2][2]);

    /* Check if determinant is large enough to do fast inverse */
    T determinant = this->determinant();

    if (determinant > agx::RealEpsilon)
      return this->fastInverse(determinant);

    /* Else do robust, but slow inversion */
    const int next[3] = {1, 2, 0};
    Matrix3x3T<T> result;
    Matrix3x3T<T> tmp(*this);

    bool done[3] = {false, false, false};
    int pivotColumns[3] = {-1, -1, -1};

    for (int i = 0; i < 3; i++)
    {
      // Select pivot row
      Vec2i coord = findMaxElement(tmp, done);
      pivotColumns[(int)coord[0]] = (int)coord[1];

      // Gauss-Jordan pivoting
      T *row = &tmp(coord[0], 0);
      T *resultRow = &result(coord[0], 0);

      // Scale target row
      T scale = T(1)/row[coord[1]];
      for (int j = 0; j < 3; j++)
      {
        row[j] *= scale;
        resultRow[j] *= scale;
      }


      // Column operations on other rows
      for (int r = next[coord[0]]; r != coord[0]; r = next[r])
      {
        T *ra = &tmp(r, 0);
        T *rb = &result(r, 0);
        T f = ra[coord[1]];

        for (int j = 0; j < 3; j++)
        {
          ra[j] -= f*row[j];
          rb[j] -= f*resultRow[j];
        }
      }

      done[coord[0]] = true;
    }

    // Do final row permutations
    for (int i = 0; i < 3; i++)
    {
      T *ra = &tmp(pivotColumns[i], 0);
      T *rb = &result(i, 0);
      memcpy(ra, rb, sizeof(T) * 3);
    }

    return tmp;
  }



  template <typename T>
  inline std::ostream& operator<< ( std::ostream& os, const Matrix3x3T<T>& m )
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

  template <typename T>
  AGX_FORCE_INLINE Matrix3x3T<T> Matrix3x3T<T>::fastInverse(T determinant) const
  {
    Matrix3x3T<T> dest;

    T k = T(1)/determinant;

    dest(0, 0) = k * DET2(m_data[1][1], m_data[1][2], m_data[2][1], m_data[2][2]);
    dest(0, 1) = k * DET2(m_data[0][2], m_data[0][1], m_data[2][2], m_data[2][1]);
    dest(0, 2) = k * DET2(m_data[0][1], m_data[0][2], m_data[1][1], m_data[1][2]);

    dest(1, 0) = k * DET2(m_data[1][2], m_data[1][0], m_data[2][2], m_data[2][0]);
    dest(1, 1) = k * DET2(m_data[0][0], m_data[0][2], m_data[2][0], m_data[2][2]);
    dest(1, 2) = k * DET2(m_data[0][2], m_data[0][0], m_data[1][2], m_data[1][0]);

    dest(2, 0) = k * DET2(m_data[1][0], m_data[1][1], m_data[2][0], m_data[2][1]);
    dest(2, 1) = k * DET2(m_data[0][1], m_data[0][0], m_data[2][1], m_data[2][0]);
    dest(2, 2) = k * DET2(m_data[0][0], m_data[0][1], m_data[1][0], m_data[1][1]);

    return dest;
  }




  #undef SET_ROW
  #undef INNER_PRODUCT
  #undef DET2
} // namespace agx

AGX_TYPE_BINDING(agx::Matrix3x3f, "Matrix3x3")
AGX_TYPE_BINDING(agx::Matrix3x3d, "Matrix3x3")

