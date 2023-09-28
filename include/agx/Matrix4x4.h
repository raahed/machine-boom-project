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
#ifndef AGX_MATRIX4X4_H
#define AGX_MATRIX4X4_H

#include <agx/config/AGX_USE_SSE.h>
#include <agx/Math.h>
#include <agx/Quat.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>

#include <agx/Prefetch.h>

#if AGX_USE_SSE()
#include <pmmintrin.h>
#endif

namespace agx
{
  class EulerAngles;
  class OrthoMatrix3x3;

#define AGX_MATRIX4X4_SET_ROW(row, v1, v2, v3, v4 )    \
  m_data[(row)][0] = (v1); \
  m_data[(row)][1] = (v2); \
  m_data[(row)][2] = (v3); \
  m_data[(row)][3] = (v4);

#define AGX_MATRIX4X4_INNER_PRODUCT(a,b,r,c) \
   ((a).m_data[r][0] * (b).m_data[0][c]) \
  +((a).m_data[r][1] * (b).m_data[1][c]) \
  +((a).m_data[r][2] * (b).m_data[2][c]) \
  +((a).m_data[r][3] * (b).m_data[3][c])

  /**
    Matrix class for affine transformations.
    Translations are stored in the last row.
    The last column will be assumed to be [0 0 0 1]' in all operations,
    even if it is set to something else.
  */
  template <typename T>
  class Matrix4x4T
  {
    public:
      typedef T Type;

    /*=====================================================
                        Constructors
    =======================================================*/

    /**
    Creates a new matrix, initialized to be an identity matrix.
    */
    Matrix4x4T();

    /** Copy constructor */
    template <typename T2>
    explicit Matrix4x4T( const Matrix4x4T<T2>& mat );

    /** Create a matrix from a vector of 16 reals */
    explicit Matrix4x4T( T const * const ptr );


    /** Create a matrix from a quaternion */
    Matrix4x4T(const QuatT<T>& rotation, const Vec3T<T>& translation = Vec3T<T>());

    /** Create a matrix from Euler angles */
    Matrix4x4T( const EulerAngles& rotation, const Vec3T<T>& translation = Vec3T<T>());

    /** Create a matrix from a 3x3 rotation matrix */
    Matrix4x4T( const OrthoMatrix3x3& rotation, const Vec3T<T>& translation = Vec3T<T>());

    /** Create a matrix from 16 T scalars */
    Matrix4x4T( T a00, T a01, T a02, T a03,
                     T a10, T a11, T a12, T a13,
                     T a20, T a21, T a22, T a23,
                     T a30, T a31, T a32, T a33 );


    /** Destructor */
    ~Matrix4x4T() = default;

    /*=====================================================
                          Accessors
    =======================================================*/

    /**
    \return true if the matrix is a valid one.
    */
    bool valid() const;

    /**
    \return true if the matrix is a valid one.
    */
    bool isNaN() const;

    /**
    \return true if all the elements are non-NaN and non-inf
    */
    bool isFinite() const;

    /**
    \return true if the matrix is identity.
    */
    bool isIdentity() const;

    /**
    \return true if the matrix is rigid transformation.
    */
    bool isRigidTransformation() const;

    /**
    Matrix inverse.
    \return the inverse
    */
    Matrix4x4T<T> inverse() const;

    /**
    \return the transpose
    */
    Matrix4x4T<T> transpose() const;

    /*=====================================================
                          Mutators
    =======================================================*/

    /**
     - set(arg) sets the entire matrix, overwriting all previous values.
     - setRotate(arg) sets the rotational part of the matrix, leaving the translation intact.
     - setTranslate(arg) sets the translational part, without affecting the rotation.
     - setIdentity() creates the identity matrix

     They all return a reference to the updated matrix, enabling chained commands. E.g:
     matrix.setRotate(q).setTranslate(pos)

     */

     Matrix4x4T<T>& set( T const * const ptr );
     Matrix4x4T<T>& set( T a00, T a01, T a02, T a03,
                      T a10, T a11, T a12, T a13,
                      T a20, T a21, T a22, T a23,
                      T a30, T a31, T a32, T a33 );

    Matrix4x4T<T>& set( const QuatT<T>& q );
    Matrix4x4T<T>& set( const OrthoMatrix3x3& m3 );
    Matrix4x4T<T>& set( const EulerAngles& e );


    /**
    Set the row of the matrix,
    \param row must be between [0..3]
    \param vec - value of the row
    */
    void setRow(size_t row, const agx::Vec4T<T>& vec ) {
      m_data[row][0] = vec[0];
      m_data[row][1] = vec[1];
      m_data[row][2] = vec[2];
      m_data[row][3] = vec[3];
    }

    void setCol(size_t col, const agx::Vec4T<T>& vec ) {
      m_data[0][col] = vec[0];
      m_data[1][col] = vec[1];
      m_data[2][col] = vec[2];
      m_data[3][col] = vec[3];
    }

    agx::Vec4T<T> getRow(size_t row) const
    {
      return agx::Vec4T<T>(
        m_data[row][0],
        m_data[row][1],
        m_data[row][2],
        m_data[row][3]
      );
    }

    agx::Vec4T<T> getCol(size_t col) const
    {
      return agx::Vec4T<T>(
        m_data[0][col],
        m_data[1][col],
        m_data[2][col],
        m_data[3][col]
      );
    }


    Matrix4x4T<T>& setRotate( const QuatT<T>& q );
    Matrix4x4T<T>& setRotate( const OrthoMatrix3x3& m3 );
    Matrix4x4T<T>& setRotate( const EulerAngles& euler );
    Matrix4x4T<T>& setRotate( const Vec3T<T>& from, const Vec3T<T>& to );
    Matrix4x4T<T>& setRotate( T angle, const Vec3T<T>& axis );
    Matrix4x4T<T>& setRotate( T angle, T x, T y, T z );
    Matrix4x4T<T>& setRotate( T angle1, const Vec3T<T>& axis1,
                                T angle2, const Vec3T<T>& axis2,
                                T angle3, const Vec3T<T>& axis3 );

    Matrix4x4T<T>& setTranslate( const Vec3T<T>& t );
    Matrix4x4T<T>& setTranslate(T tx, T ty, T tz);

    Matrix4x4T<T>& addTranslate( const Vec3T<T>& t );
    Matrix4x4T<T>& addTranslate(T tx, T ty, T tz);

    Matrix4x4T<T>& setIdentity();

    /*=====================================================
                          Accessors
    =======================================================*/
    Vec3T<T> getTranslate() const;

    QuatT<T> getRotate() const;

    /**
    \return a quaternion of the rotation part of this matrix (real part > 0 by convention)
    */
    void get( QuatT<T>& q ) const;
    void get( OrthoMatrix3x3& m3 ) const;
    void get( EulerAngles& e ) const;


    /* Get the internal row-major buffer */
    T *ptr();
    const T *ptr() const;

    /*=====================================================
                        Static methods
    =======================================================*/

    /**
    Generates a new matrix of a specific type.
    */
    static Matrix4x4T<T> crossMatrix(const Vec3T<T>& vec);
    static Matrix4x4T<T> translate( const Vec3T<T>& dv );
    static Matrix4x4T<T> translate( T x, T y, T z );
    static Matrix4x4T<T> rotate( const Vec3T<T>& from, const Vec3T<T>& to );
    static Matrix4x4T<T> rotate( T angle, T x, T y, T z );
    static Matrix4x4T<T> rotate( T angle, const Vec3T<T>& axis );
    static Matrix4x4T<T> rotate( T angle1, const Vec3T<T>& axis1,
                                          T angle2, const Vec3T<T>& axis2,
                                          T angle3, const Vec3T<T>& axis3 );

    /*=====================================================
                        Operators
    =======================================================*/
    Vec3T<T> operator* ( const Vec3T<T>& v ) const;
    Vec4T<T> operator* ( const Vec4T<T>& v ) const;

    void operator*= ( const Matrix4x4T<T>& other );
    Matrix4x4T<T> operator* ( const Matrix4x4T<T> &m ) const;
    bool operator== ( const Matrix4x4T<T>& m ) const;
    bool operator!= ( const Matrix4x4T<T>& m ) const;
    T& operator()( size_t row, size_t col );
    T operator()( size_t row, size_t col ) const;
    Matrix4x4T<T>& operator= ( const Matrix4x4T<T>& rhs );

    Vec3T<T>& transform3x3( const Vec3T<T>& vIn, Vec3T<T>& vOut ) const;

    Vec3T<T>& transform3x3Inv( const Vec3T<T>& vIn, Vec3T<T>& vOut ) const;

    Vec3T<T> transform3x3(const Vec3T<T>& vIn) const;

    Vec3T<T> transform3x3Inv(const Vec3T<T>& vIn) const;

    Vec3T<T> preMult(const Vec3T<T>& v) const;

    Vec3T<T> postMult(const Vec3T<T>& v) const;

    Vec4T<T> preMult(const Vec4T<T>& v) const;

    Vec4T<T> postMult(const Vec4T<T>& v) const;

    void mult( const Matrix4x4T<T>&, const Matrix4x4T<T>& );
    void multSSE(const Matrix4x4T<T>& lhs, const Matrix4x4T<T>& rhs);
    void preMult( const Matrix4x4T<T>& );
    void postMult( const Matrix4x4T<T>& );
    Matrix4x4T<T>& preMultTranslate( const Vec3T<T>& v );
    Matrix4x4T<T>& postMultTranslate( const Vec3T<T>& v );


  protected:

    bool invert(const Matrix4x4T<T>& mat);
    static void multSSE32Implementation(Real32 *out, const Real32 *lhs, const Real32 *rhs);
    static void multSSE64Implementation(Real64 *out, const Real64 *lhs, const Real64 *rhs);


    T m_data[4][4];
  };

  typedef Matrix4x4T<Real> Matrix4x4;

  typedef Matrix4x4T<Real32> Matrix4x4f;
  typedef Matrix4x4T<Real64> Matrix4x4d;
}



/* ****************************************** */
/* *           IMPLEMENTATION              ** */
/* ****************************************** */
#include <agx/OrthoMatrix3x3.h>
#include <agx/EulerAngles.h>

namespace agx
{

  /*=====================================================
                      Constructors
  =======================================================*/

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>::Matrix4x4T()
  {
    this->setIdentity();
  }

  template <typename T> template <typename T2>
  AGX_FORCE_INLINE Matrix4x4T<T>::Matrix4x4T( const Matrix4x4T<T2>& mat )
  {
    m_data[0][0] = (T)mat(0, 0);
    m_data[0][1] = (T)mat(0, 1);
    m_data[0][2] = (T)mat(0, 2);
    m_data[0][3] = (T)mat(0, 3);

    m_data[1][0] = (T)mat(1, 0);
    m_data[1][1] = (T)mat(1, 1);
    m_data[1][2] = (T)mat(1, 2);
    m_data[1][3] = (T)mat(1, 3);

    m_data[2][0] = (T)mat(2, 0);
    m_data[2][1] = (T)mat(2, 1);
    m_data[2][2] = (T)mat(2, 2);
    m_data[2][3] = (T)mat(2, 3);

    m_data[3][0] = (T)mat(3, 0);
    m_data[3][1] = (T)mat(3, 1);
    m_data[3][2] = (T)mat(3, 2);
    m_data[3][3] = (T)mat(3, 3);
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>::Matrix4x4T( T a00, T a01, T a02, T a03,
                                           T a10, T a11, T a12, T a13,
                                           T a20, T a21, T a22, T a23,
                                           T a30, T a31, T a32, T a33 )
  {
    m_data[0][0] = a00;
    m_data[0][1] = a01;
    m_data[0][2] = a02;
    m_data[0][3] = a03;

    m_data[1][0] = a10;
    m_data[1][1] = a11;
    m_data[1][2] = a12;
    m_data[1][3] = a13;

    m_data[2][0] = a20;
    m_data[2][1] = a21;
    m_data[2][2] = a22;
    m_data[2][3] = a23;

    m_data[3][0] = a30;
    m_data[3][1] = a31;
    m_data[3][2] = a32;
    m_data[3][3] = a33;

  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>::Matrix4x4T( T const * const ptr )
  {
    set( ptr );
  }


  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>::Matrix4x4T(const QuatT<T>& rotation, const Vec3T<T>& translation)
  {
    this->setIdentity();
    this->set(rotation);
    this->setTranslate(translation);
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>::Matrix4x4T( const OrthoMatrix3x3& rotation, const Vec3T<T>& translation )
  {
    this->setIdentity();
    this->set(rotation);
    this->setTranslate(translation);
  }



  /*=====================================================
                        Queries
  =======================================================*/

  template <typename T>
  AGX_FORCE_INLINE bool Matrix4x4T<T>::valid() const
  {
    return !isNaN();
  }


  template <typename T>
  AGX_FORCE_INLINE bool Matrix4x4T<T>::isNaN() const
  {
    return std::isnan( m_data[0][0] ) || std::isnan( m_data[0][1] )
      || std::isnan( m_data[0][2] ) || std::isnan( m_data[0][3] )
      || std::isnan( m_data[1][0] ) || std::isnan( m_data[1][1] )
      || std::isnan( m_data[1][2] ) || std::isnan( m_data[1][3] )
      || std::isnan( m_data[2][0] ) || std::isnan( m_data[2][1] )
      || std::isnan( m_data[2][2] ) || std::isnan( m_data[2][3] )
      || std::isnan( m_data[3][0] ) || std::isnan( m_data[3][1] )
      || std::isnan( m_data[3][2] ) || std::isnan( m_data[3][3] );
  }


  template <typename T>
  AGX_FORCE_INLINE bool Matrix4x4T<T>::isFinite() const
  {
    return std::isfinite( m_data[0][0] ) && std::isfinite( m_data[0][1] )
      && std::isfinite( m_data[0][2] ) && std::isfinite( m_data[0][3] )
      && std::isfinite( m_data[1][0] ) && std::isfinite( m_data[1][1] )
      && std::isfinite( m_data[1][2] ) && std::isfinite( m_data[1][3] )
      && std::isfinite( m_data[2][0] ) && std::isfinite( m_data[2][1] )
      && std::isfinite( m_data[2][2] ) && std::isfinite( m_data[2][3] )
      && std::isfinite( m_data[3][0] ) && std::isfinite( m_data[3][1] )
      && std::isfinite( m_data[3][2] ) && std::isfinite( m_data[3][3] );
  }

  template <typename T>
  AGX_FORCE_INLINE bool Matrix4x4T<T>::isIdentity() const
  {
    return *this == Matrix4x4T<T>();
  }

  template <typename T>
  AGX_FORCE_INLINE bool Matrix4x4T<T>::isRigidTransformation() const
  {
    bool isRigid = true;
    // rows have norm 1
    isRigid &= equivalent(m_data[0][0]*m_data[0][0] + m_data[0][1]*m_data[0][1] + m_data[0][2]*m_data[0][2], T(1), T(1.0e-3));
    isRigid &= equivalent(m_data[1][0]*m_data[1][0] + m_data[1][1]*m_data[1][1] + m_data[1][2]*m_data[1][2], T(1), T(1.0e-3));
    isRigid &= equivalent(m_data[2][0]*m_data[2][0] + m_data[2][1]*m_data[2][1] + m_data[2][2]*m_data[2][2], T(1), T(1.0e-3));
    // columns have norm 1
    isRigid &= equivalent(m_data[0][0]*m_data[0][0] + m_data[1][0]*m_data[1][0] + m_data[2][0]*m_data[2][0], T(1), T(1.0e-3));
    isRigid &= equivalent(m_data[0][1]*m_data[0][1] + m_data[1][1]*m_data[1][1] + m_data[2][1]*m_data[2][1], T(1), T(1.0e-3));
    isRigid &= equivalent(m_data[0][2]*m_data[0][2] + m_data[1][2]*m_data[1][2] + m_data[2][2]*m_data[2][2], T(1), T(1.0e-3));
    // determinant 1
    isRigid &= equivalent(m_data[0][0]*m_data[1][1]*m_data[2][2] + m_data[0][1]*m_data[1][2]*m_data[2][0] + m_data[0][2]*m_data[1][0]*m_data[2][1]
    -  m_data[2][0]*m_data[1][1]*m_data[0][2] - m_data[1][0]*m_data[0][1]*m_data[2][2] - m_data[0][0]*m_data[2][1]*m_data[1][2],
      T(1), T(1.0e-1));
    // no row scaling
    isRigid &= (m_data[0][3] == T(0));
    isRigid &= (m_data[1][3] == T(0));
    isRigid &= (m_data[2][3] == T(0));
    // no transform scaling
    isRigid &= (m_data[3][3] == T(1));
    return isRigid;
  }



  /*=====================================================
                        Setters
  =======================================================*/
  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::set( T const * const ptr )
  {
    memcpy(m_data, ptr, sizeof(T)*16);
    return *this;
  }


  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::set( T a00, T a01, T a02, T a03,
                                          T a10, T a11, T a12, T a13,
                                          T a20, T a21, T a22, T a23,
                                          T a30, T a31, T a32, T a33 )
  {
   m_data[0][0] = a00;
   m_data[0][1] = a01;
   m_data[0][2] = a02;
   m_data[0][3] = a03;

   m_data[1][0] = a10;
   m_data[1][1] = a11;
   m_data[1][2] = a12;
   m_data[1][3] = a13;

   m_data[2][0] = a20;
   m_data[2][1] = a21;
   m_data[2][2] = a22;
   m_data[2][3] = a23;

   m_data[3][0] = a30;
   m_data[3][1] = a31;
   m_data[3][2] = a32;
   m_data[3][3] = a33;

   return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::set(const QuatT<T>& q)
  {
    this->setRotate(q);
    this->setTranslate(0.0, 0.0, 0.0);

    return *this;
  }


  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::set( const OrthoMatrix3x3& m3 )
  {
    this->setRotate(m3);
    this->setTranslate(0.0, 0.0, 0.0);

    return *this;
  }



  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::setRotate( const OrthoMatrix3x3& m3 )
  {
    (*this)(0,0) = (T)m3(0,0);
    (*this)(0,1) = (T)m3(0,1);
    (*this)(0,2) = (T)m3(0,2);

    (*this)(1,0) = (T)m3(1,0);
    (*this)(1,1) = (T)m3(1,1);
    (*this)(1,2) = (T)m3(1,2);

    (*this)(2,0) = (T)m3(2,0);
    (*this)(2,1) = (T)m3(2,1);
    (*this)(2,2) = (T)m3(2,2);

    return *this;
  }


  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::setRotate( const Vec3T<T>& from, const Vec3T<T>& to )
  {
    QuatT<T> quat;
    quat.setRotate(from, to);
    this->setRotate(quat);
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::setRotate( T angle, const Vec3T<T>& axis )
  {
    QuatT<T> quat;
    quat.setRotate( angle, axis);
    this->setRotate(quat);
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::setRotate( T angle, T x, T y, T z )
  {
    QuatT<T> quat;
    quat.setRotate( angle, x, y, z);
    this->setRotate(quat);
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::setRotate( T angle1, const Vec3T<T>& axis1,
                              T angle2, const Vec3T<T>& axis2,
                              T angle3, const Vec3T<T>& axis3 )
  {
    QuatT<T> quat;
    quat.setRotate(angle1, axis1,
                   angle2, axis2,
                   angle3, axis3);
    this->setRotate(quat);
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::setTranslate( const Vec3T<T>& t )
  {
    m_data[3][0] = t[0];
    m_data[3][1] = t[1];
    m_data[3][2] = t[2];

    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::setTranslate(T tx, T ty, T tz)
  {
    m_data[3][0] = tx;
    m_data[3][1] = ty;
    m_data[3][2] = tz;

    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::addTranslate( const Vec3T<T>& t )
  {
    m_data[3][0] += t[0];
    m_data[3][1] += t[1];
    m_data[3][2] += t[2];

    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::addTranslate(T tx, T ty, T tz)
  {
    m_data[3][0] += tx;
    m_data[3][1] += ty;
    m_data[3][2] += tz;

    return *this;
  }


  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::setIdentity()
  {
    this->set(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
    return *this;
  }

  /*=====================================================
                        Getters
  =======================================================*/

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Matrix4x4T<T>::getTranslate() const
  {
    return Vec3T<T>( m_data[3][0], m_data[3][1], m_data[3][2] );
  }


  template <typename T>
  AGX_FORCE_INLINE QuatT<T> Matrix4x4T<T>::getRotate() const
  {
    QuatT<T> q;
    get(q);
    return q;
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix4x4T<T>::get( OrthoMatrix3x3& m3 ) const
  {
    m3(0,0) = Real((*this)(0,0));
    m3(0,1) = Real((*this)(0,1));
    m3(0,2) = Real((*this)(0,2));

    m3(1,0) = Real((*this)(1,0));
    m3(1,1) = Real((*this)(1,1));
    m3(1,2) = Real((*this)(1,2));

    m3(2,0) = Real((*this)(2,0));
    m3(2,1) = Real((*this)(2,1));
    m3(2,2) = Real((*this)(2,2));
  }


  template <typename T>
  AGX_FORCE_INLINE T *Matrix4x4T<T>::ptr()
  {
    return ( T* )m_data;
  }

  template <typename T>
  AGX_FORCE_INLINE const T *Matrix4x4T<T>::ptr() const
  {
    return ( const T * )m_data;
  }

  /*=====================================================
                      Serialization methods
  =======================================================*/

  /*=====================================================
                      Static methods
  =======================================================*/

  template <typename T>
  inline Matrix4x4T<T> Matrix4x4T<T>::crossMatrix(const Vec3T<T>& vec)
  {
    return Matrix4x4T<T>(
       0,        -vec.z(),  vec.y(), 0,
       vec.z(),   0,       -vec.x(), 0,
      -vec.y(),   vec.x(),  0,       0,
       0,         0,        0,       1);
  }


  template <typename T>
  inline Matrix4x4T<T> Matrix4x4T<T>::translate( const Vec3T<T>& dv )
  {
    Matrix4x4T<T> m;
    m.setTranslate(dv);
    return m;
  }

  template <typename T>
  inline Matrix4x4T<T> Matrix4x4T<T>::translate( T x, T y, T z )
  {
    Matrix4x4T<T> m;
    m.setTranslate(x, y, z);
    return m;
  }

  template <typename T>
  inline Matrix4x4T<T> Matrix4x4T<T>::rotate( const Vec3T<T>& from, const Vec3T<T>& to )
  {
    Matrix4x4T<T> m;
    m.setRotate(from, to);
    return m;
  }

  template <typename T>
  inline Matrix4x4T<T> Matrix4x4T<T>::rotate( T angle, T x, T y, T z )
  {
    Matrix4x4T<T> m;
    m.setRotate(angle, x, y, z);
    return m;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T> Matrix4x4T<T>::rotate( T angle, const Vec3T<T>& axis )
  {
    Matrix4x4T<T> m;
    m.setRotate(angle, axis);
    return m;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T> Matrix4x4T<T>::rotate( T angle1, const Vec3T<T>& axis1,
                                        T angle2, const Vec3T<T>& axis2,
                                        T angle3, const Vec3T<T>& axis3 )
  {
    Matrix4x4T<T> m;
    m.setRotate(angle1, axis1, angle2, axis2, angle3, axis3);
    return m;
  }


  /*=====================================================
                      Operators
  =======================================================*/

  template <typename T>
  AGX_FORCE_INLINE void Matrix4x4T<T>::mult( const Matrix4x4T<T>& lhs, const Matrix4x4T<T>& rhs )
  {
    agx::prefetch<agx::L1>( lhs.m_data );
    agx::prefetch<agx::L1>( rhs.m_data );

    if (&lhs==this)
    {
      postMult(rhs);
      return;
    }
    if (&rhs==this)
    {
      preMult(lhs);
      return;
    }

    m_data[0][0] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 0, 0);
    m_data[0][1] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 0, 1);
    m_data[0][2] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 0, 2);
    m_data[0][3] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 0, 3);

    m_data[1][0] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 1, 0);
    m_data[1][1] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 1, 1);
    m_data[1][2] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 1, 2);
    m_data[1][3] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 1, 3);

    m_data[2][0] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 2, 0);
    m_data[2][1] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 2, 1);
    m_data[2][2] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 2, 2);
    m_data[2][3] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 2, 3);

    m_data[3][0] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 3, 0);
    m_data[3][1] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 3, 1);
    m_data[3][2] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 3, 2);
    m_data[3][3] = AGX_MATRIX4X4_INNER_PRODUCT(lhs, rhs, 3, 3);

  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix4x4T<T>::preMult( const Matrix4x4T<T>& other )
  {

    // more efficient method just use a T[4] for temporary storage.
    T t[4];
    for(size_t col=0; col<4; ++col) {
      t[0] = AGX_MATRIX4X4_INNER_PRODUCT( other, *this, 0, col );
      t[1] = AGX_MATRIX4X4_INNER_PRODUCT( other, *this, 1, col );
      t[2] = AGX_MATRIX4X4_INNER_PRODUCT( other, *this, 2, col );
      t[3] = AGX_MATRIX4X4_INNER_PRODUCT( other, *this, 3, col );
      m_data[0][col] = t[0];
      m_data[1][col] = t[1];
      m_data[2][col] = t[2];
      m_data[3][col] = t[3];
    }
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix4x4T<T>::postMult( const Matrix4x4T<T>& other )
  {

    // more efficient method just use a T[3] for temporary storage.
    T t[4];
    for(size_t row=0; row<4; ++row)
    {
      t[0] = AGX_MATRIX4X4_INNER_PRODUCT( *this, other, row, 0 );
      t[1] = AGX_MATRIX4X4_INNER_PRODUCT( *this, other, row, 1 );
      t[2] = AGX_MATRIX4X4_INNER_PRODUCT( *this, other, row, 2 );
      t[3] = AGX_MATRIX4X4_INNER_PRODUCT( *this, other, row, 3 );

      AGX_MATRIX4X4_SET_ROW(row, t[0], t[1], t[2], t[3] )
    }
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::preMultTranslate( const Vec3T<T>& v )
  {
    for (unsigned i = 0; i < 3; ++i)
    {
      m_data[3][0] += v[i]*m_data[i][0];
      m_data[3][1] += v[i]*m_data[i][1];
      m_data[3][2] += v[i]*m_data[i][2];
      m_data[3][3] += v[i]*m_data[i][3];
    }
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::postMultTranslate( const Vec3T<T>& v )
  {
    for (unsigned i = 0; i < 3; ++i)
    {
      m_data[0][i] += v[i]*m_data[0][3];
      m_data[1][i] += v[i]*m_data[1][3];
      m_data[2][i] += v[i]*m_data[2][3];
      m_data[3][i] += v[i]*m_data[3][3];
    }
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE bool Matrix4x4T<T>::operator== ( const Matrix4x4T<T>& m ) const
  {
    const int bitCompare = memcmp(m_data, m.m_data, sizeof(m_data));
    return (bitCompare == 0);
  }

  template <typename T>
  AGX_FORCE_INLINE bool Matrix4x4T<T>::operator!= ( const Matrix4x4T<T>& m ) const
  {
    const int bitCompare = memcmp(m_data, m.m_data, sizeof(m_data));
    return (bitCompare != 0);
  }

  template <typename T>
  AGX_FORCE_INLINE T& Matrix4x4T<T>::operator()( size_t row, size_t col )
  {
    return m_data[row][col];
  }

  template <typename T>
  AGX_FORCE_INLINE T Matrix4x4T<T>::operator()( size_t row, size_t col ) const
  {
    return m_data[row][col];
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T>& Matrix4x4T<T>::operator= ( const Matrix4x4T<T>& rhs )
  {
    if ( this != &rhs )
      this->set( rhs.ptr() );

    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE void Matrix4x4T<T>::operator*= ( const Matrix4x4T<T>& other )
  {
    if ( this == &other )
    {
      Matrix4x4T<T> temp( other );
      postMult( temp );
    }
    else postMult( other );
  }

  template <typename T>
  AGX_FORCE_INLINE Matrix4x4T<T> Matrix4x4T<T>::operator* ( const Matrix4x4T<T> &m ) const
  {
    Matrix4x4T<T> r;
    r.mult( *this, m );
    return  r;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Matrix4x4T<T>::operator* ( const Vec3T<T>& v ) const
  {
    return postMult( v );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T> Matrix4x4T<T>::operator* ( const Vec4T<T>& v ) const
  {
    return postMult( v );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> operator* ( const Vec3T<T>& v, const Matrix4x4T<T>& m )
  {
    return m.preMult( v );
  }
  template <typename T>
  AGX_FORCE_INLINE Vec4T<T> operator* ( const Vec4T<T>& v, const Matrix4x4T<T>& m )
  {
    return m.preMult( v );
  }



  template <typename T>
  std::ostream& operator<< ( std::ostream& os, const Matrix4x4T<T>& m )
  {
    os << "{" << std::endl;
    for ( size_t row = 0; row < 4; ++row ) {
      os << "\t";
      for ( size_t col = 0; col < 4; ++col )
        os << m( row, col ) << " ";
      os << std::endl;
    }
    os << "}" << std::endl;
    return os;
  }

  /*=====================================================
                      Internal
  =======================================================*/

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>& Matrix4x4T<T>::transform3x3( const Vec3T<T>& vIn, Vec3T<T>& vOut ) const
  {
    vOut.set(
      (m_data[0][0] * vIn.x() + m_data[1][0] * vIn.y() + m_data[2][0] * vIn.z()),
      (m_data[0][1] * vIn.x() + m_data[1][1] * vIn.y() + m_data[2][1] * vIn.z()),
      (m_data[0][2] * vIn.x() + m_data[1][2] * vIn.y() + m_data[2][2] * vIn.z()));
    return vOut;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T>& Matrix4x4T<T>::transform3x3Inv(const Vec3T<T>& vIn, Vec3T<T>& vOut) const
  {
    vOut.set(
      (m_data[0][0] * vIn.x() + m_data[0][1] * vIn.y() + m_data[0][2] * vIn.z()),
      (m_data[1][0] * vIn.x() + m_data[1][1] * vIn.y() + m_data[1][2] * vIn.z()),
      (m_data[2][0] * vIn.x() + m_data[2][1] * vIn.y() + m_data[2][2] * vIn.z()));
    return vOut;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Matrix4x4T<T>::transform3x3( const Vec3T<T>& vIn ) const
  {
    Vec3T<T> vOut;
    transform3x3(vIn, vOut);
    return vOut;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Matrix4x4T<T>::transform3x3Inv( const Vec3T<T>& vIn ) const
  {
    Vec3T<T> vOut;
    transform3x3Inv(vIn, vOut);
    return vOut;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Matrix4x4T<T>::postMult( const Vec3T<T>& v ) const
  {
    T d = T(1.0) / ( m_data[3][0] * v.x() + m_data[3][1] * v.y() + m_data[3][2] * v.z() + T(1.0) ) ;
    return Vec3T<T>( ( m_data[0][0]*v.x() + m_data[0][1]*v.y() + m_data[0][2]*v.z() ) *d,
                 ( m_data[1][0]*v.x() + m_data[1][1]*v.y() + m_data[1][2]*v.z() ) *d,
                 ( m_data[2][0]*v.x() + m_data[2][1]*v.y() + m_data[2][2]*v.z() ) *d ) ;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> Matrix4x4T<T>::preMult( const Vec3T<T>& v ) const
  {
    return Vec3T<T>(
             ( m_data[0][0]*v.x() + m_data[1][0]*v.y() + m_data[2][0]*v.z() + m_data[3][0] ),
             ( m_data[0][1]*v.x() + m_data[1][1]*v.y() + m_data[2][1]*v.z() + m_data[3][1] ),
             ( m_data[0][2]*v.x() + m_data[1][2]*v.y() + m_data[2][2]*v.z() + m_data[3][2] ) );
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T> Matrix4x4T<T>::postMult( const Vec4T<T>& v ) const
  {
    return Vec4T<T>(
             ( m_data[0][0]*v.x() + m_data[0][1]*v.y() + m_data[0][2]*v.z() + m_data[0][3]*v.w() ),
             ( m_data[1][0]*v.x() + m_data[1][1]*v.y() + m_data[1][2]*v.z() + m_data[1][3]*v.w() ),
             ( m_data[2][0]*v.x() + m_data[2][1]*v.y() + m_data[2][2]*v.z() + m_data[2][3]*v.w() ),
             ( m_data[3][0]*v.x() + m_data[3][1]*v.y() + m_data[3][2]*v.z() + m_data[3][3]*v.w() ) ) ;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T> Matrix4x4T<T>::preMult( const Vec4T<T>& v ) const
  {
    return Vec4T<T>(
             ( m_data[0][0]*v.x() + m_data[1][0]*v.y() + m_data[2][0]*v.z() + m_data[3][0]*v.w() ),
             ( m_data[0][1]*v.x() + m_data[1][1]*v.y() + m_data[2][1]*v.z() + m_data[3][1]*v.w() ),
             ( m_data[0][2]*v.x() + m_data[1][2]*v.y() + m_data[2][2]*v.z() + m_data[3][2]*v.w() ),
             ( m_data[0][3]*v.x() + m_data[1][3]*v.y() + m_data[2][3]*v.z() + m_data[3][3]*v.w() ) );
  }


  template <typename T>
  inline Matrix4x4T<T> Matrix4x4T<T>::transpose() const
  {
    return Matrix4x4T<T>( m_data[0][0], m_data[1][0], m_data[2][0], m_data[3][0],
                            m_data[0][1], m_data[1][1], m_data[2][1], m_data[3][1],
                            m_data[0][2], m_data[1][2], m_data[2][2], m_data[3][2],
                            m_data[0][3], m_data[1][3], m_data[2][3], m_data[3][3]);
  }


  template <typename T>
  inline Matrix4x4T<T> Matrix4x4T<T>::inverse() const
  {
    Matrix4x4T<T> m;
    m.invert( *this );
    return m;
  }

  template <typename T>
  AGX_FORCE_INLINE bool equivalent( const agx::Matrix4x4T<T>& a, const agx::Matrix4x4T<T>& b, T epsilon = 1e-6 )
  {
    return
      agx::equivalent(a(0, 0), b(0, 0), epsilon) &&
      agx::equivalent(a(0, 1), b(0, 1), epsilon) &&
      agx::equivalent(a(0, 2), b(0, 2), epsilon) &&
      agx::equivalent(a(0, 3), b(0, 3), epsilon) &&
      agx::equivalent(a(1, 0), b(1, 0), epsilon) &&
      agx::equivalent(a(1, 1), b(1, 1), epsilon) &&
      agx::equivalent(a(1, 2), b(1, 2), epsilon) &&
      agx::equivalent(a(1, 3), b(1, 3), epsilon) &&
      agx::equivalent(a(2, 0), b(2, 0), epsilon) &&
      agx::equivalent(a(2, 1), b(2, 1), epsilon) &&
      agx::equivalent(a(2, 2), b(2, 2), epsilon) &&
      agx::equivalent(a(2, 3), b(2, 3), epsilon) &&
      agx::equivalent(a(3, 0), b(3, 0), epsilon) &&
      agx::equivalent(a(3, 1), b(3, 1), epsilon) &&
      agx::equivalent(a(3, 2), b(3, 2), epsilon) &&
      agx::equivalent(a(3, 3), b(3, 3), epsilon);
  }

#if AGX_USE_SSE()
  template <typename T>
  AGX_FORCE_INLINE void Matrix4x4T<T>::multSSE32Implementation(Real32 *out, const Real32 *lhs, const Real32 *rhs)
  {
    agxAssert(out && lhs && rhs);
    agxAssert(agx::isAligned((agx::UInt)out, (agx::UInt)16));
    agxAssert(agx::isAligned((agx::UInt)lhs, (agx::UInt)16));
    agxAssert(agx::isAligned((agx::UInt)rhs, (agx::UInt)16));

    __m128 lhs1 = _mm_load_ps( lhs );
    __m128 lhs2 = _mm_load_ps( lhs + 4 );
    __m128 lhs3 = _mm_load_ps( lhs + 8 );
    __m128 lhs4 = _mm_load_ps( lhs + 12 );

    __m128 rhs1 = _mm_load_ps( rhs );
    __m128 rhs2 = _mm_load_ps( rhs + 4 );
    __m128 rhs3 = _mm_load_ps( rhs + 8 );
    __m128 rhs4 = _mm_load_ps( rhs + 12 );

    _MM_TRANSPOSE4_PS( rhs1, rhs2, rhs3, rhs4 );

    _mm_store_ps( out    ,  _mm_hadd_ps( _mm_hadd_ps( _mm_mul_ps(lhs1, rhs1), _mm_mul_ps(lhs1, rhs2) ),
                                         _mm_hadd_ps( _mm_mul_ps(lhs1, rhs3), _mm_mul_ps(lhs1, rhs4) ) ));
    _mm_store_ps( out + 4,  _mm_hadd_ps( _mm_hadd_ps( _mm_mul_ps(lhs2, rhs1), _mm_mul_ps(lhs2, rhs2) ),
                                         _mm_hadd_ps( _mm_mul_ps(lhs2, rhs3), _mm_mul_ps(lhs2, rhs4) ) ));
    _mm_store_ps( out + 8,  _mm_hadd_ps( _mm_hadd_ps( _mm_mul_ps(lhs3, rhs1), _mm_mul_ps(lhs3, rhs2) ),
                                         _mm_hadd_ps( _mm_mul_ps(lhs3, rhs3), _mm_mul_ps(lhs3, rhs4) ) ));
    _mm_store_ps( out + 12, _mm_hadd_ps( _mm_hadd_ps( _mm_mul_ps(lhs4, rhs1), _mm_mul_ps(lhs4, rhs2) ),
                                         _mm_hadd_ps( _mm_mul_ps(lhs4, rhs3), _mm_mul_ps(lhs4, rhs4) ) ));

  }

#else

  template <typename T>
  AGX_FORCE_INLINE void Matrix4x4T<T>::multSSE32Implementation(Real32*, const Real32*, const Real32*)
  {
    agxAbort1("AGX_USE_SSE is disabled");
  }
#endif



#if AGX_USE_SSE()
  template <typename T>
  AGX_FORCE_INLINE void Matrix4x4T<T>::multSSE64Implementation(Real64 *out, const Real64 *lhs, const Real64 *rhs)
  {
    agxAssert(out && lhs && rhs);
    agxAssert(out != lhs);
    agxAssert(out != rhs);
    agxAssert(agx::isAligned((agx::UInt)out, (agx::UInt)16));
    agxAssert(agx::isAligned((agx::UInt)lhs, (agx::UInt)16));
    agxAssert(agx::isAligned((agx::UInt)rhs, (agx::UInt)16));

    __m128d tmp1, tmp2, tmp3, tmp4;

    __m128d rhs1 = _mm_load_pd( rhs      );
    __m128d rhs2 = _mm_load_pd( rhs +  4 );
    __m128d rhs3 = _mm_load_pd( rhs +  8 );
    __m128d rhs4 = _mm_load_pd( rhs + 12 );

    tmp1 = _mm_shuffle_pd( rhs1, rhs2, _MM_SHUFFLE2(0, 0) );
    tmp2 = _mm_shuffle_pd( rhs1, rhs2, _MM_SHUFFLE2(1, 1) );
    tmp3 = _mm_shuffle_pd( rhs3, rhs4, _MM_SHUFFLE2(0, 0) );
    tmp4 = _mm_shuffle_pd( rhs3, rhs4, _MM_SHUFFLE2(1, 1) );

    __m128d lhs11 = _mm_load_pd( lhs     );
    __m128d lhs12 = _mm_load_pd( lhs +  2 );
    __m128d lhs21 = _mm_load_pd( lhs +  4 );
    __m128d lhs22 = _mm_load_pd( lhs +  6 );
    __m128d lhs31 = _mm_load_pd( lhs +  8 );
    __m128d lhs32 = _mm_load_pd( lhs + 10 );
    __m128d lhs41 = _mm_load_pd( lhs + 12 );
    __m128d lhs42 = _mm_load_pd( lhs + 14 );

    _mm_store_pd( out,      _mm_hadd_pd( _mm_add_pd( _mm_mul_pd( lhs11, tmp1 ), _mm_mul_pd( lhs12, tmp3 ) ),
                                         _mm_add_pd( _mm_mul_pd( lhs11, tmp2 ), _mm_mul_pd( lhs12, tmp4 ) ) ) );
    _mm_store_pd( out +  4, _mm_hadd_pd( _mm_add_pd( _mm_mul_pd( lhs21, tmp1 ), _mm_mul_pd( lhs22, tmp3 ) ),
                                         _mm_add_pd( _mm_mul_pd( lhs21, tmp2 ), _mm_mul_pd( lhs22, tmp4 ) ) ) );
    _mm_store_pd( out +  8, _mm_hadd_pd( _mm_add_pd( _mm_mul_pd( lhs31, tmp1 ), _mm_mul_pd( lhs32, tmp3 ) ),
                                         _mm_add_pd( _mm_mul_pd( lhs31, tmp2 ), _mm_mul_pd( lhs32, tmp4 ) ) ) );
    _mm_store_pd( out + 12, _mm_hadd_pd( _mm_add_pd( _mm_mul_pd( lhs41, tmp1 ), _mm_mul_pd( lhs42, tmp3 ) ),
                                         _mm_add_pd( _mm_mul_pd( lhs41, tmp2 ), _mm_mul_pd( lhs42, tmp4 ) ) ) );

    rhs1 = _mm_load_pd( rhs +  2 );
    rhs2 = _mm_load_pd( rhs +  6 );
    rhs3 = _mm_load_pd( rhs + 10 );
    rhs4 = _mm_load_pd( rhs + 14 );

    tmp1 = _mm_shuffle_pd( rhs1, rhs2, _MM_SHUFFLE2(0, 0) );
    tmp2 = _mm_shuffle_pd( rhs1, rhs2, _MM_SHUFFLE2(1, 1) );
    tmp3 = _mm_shuffle_pd( rhs3, rhs4, _MM_SHUFFLE2(0, 0) );
    tmp4 = _mm_shuffle_pd( rhs3, rhs4, _MM_SHUFFLE2(1, 1) );

    _mm_store_pd( out +  2, _mm_hadd_pd( _mm_add_pd( _mm_mul_pd( lhs11, tmp1 ), _mm_mul_pd( lhs12, tmp3 ) ),
                                         _mm_add_pd( _mm_mul_pd( lhs11, tmp2 ), _mm_mul_pd( lhs12, tmp4 ) ) ) );
    _mm_store_pd( out +  6, _mm_hadd_pd( _mm_add_pd( _mm_mul_pd( lhs21, tmp1 ), _mm_mul_pd( lhs22, tmp3 ) ),
                                         _mm_add_pd( _mm_mul_pd( lhs21, tmp2 ), _mm_mul_pd( lhs22, tmp4 ) ) ) );
    _mm_store_pd( out + 10, _mm_hadd_pd( _mm_add_pd( _mm_mul_pd( lhs31, tmp1 ), _mm_mul_pd( lhs32, tmp3 ) ),
                                         _mm_add_pd( _mm_mul_pd( lhs31, tmp2 ), _mm_mul_pd( lhs32, tmp4 ) ) ) );
    _mm_store_pd( out + 14, _mm_hadd_pd( _mm_add_pd( _mm_mul_pd( lhs41, tmp1 ), _mm_mul_pd( lhs42, tmp3 ) ),
                                         _mm_add_pd( _mm_mul_pd( lhs41, tmp2 ), _mm_mul_pd( lhs42, tmp4 ) ) ) );
  }

#else

  template <typename T>
  AGX_FORCE_INLINE void Matrix4x4T<T>::multSSE64Implementation(Real64*, const Real64*, const Real64*)
  {
    agxAbort1("AGX_USE_SSE is disabled");
  }
#endif


  template <>
  AGX_FORCE_INLINE void Matrix4x4T<Real32>::multSSE(const Matrix4x4T<Real32>& lhs, const Matrix4x4T<Real32>& rhs)
  {
    #if AGX_USE_SSE()
    multSSE32Implementation(this->ptr(), lhs.ptr(), rhs.ptr());
    #else
    mult(lhs, rhs);
    #endif
  }

  template <>
  AGX_FORCE_INLINE void Matrix4x4T<Real64>::multSSE(const Matrix4x4T<Real64>& lhs, const Matrix4x4T<Real64>& rhs)
  {
    #if AGX_USE_SSE()
    multSSE64Implementation(this->ptr(), lhs.ptr(), rhs.ptr());
    #else
    mult(lhs, rhs);
    #endif
  }


  template <>
  inline Matrix4x4f inverse<Matrix4x4f>(const Matrix4x4f& value)
  {
    return value.inverse();
  }


  template <>
  inline Matrix4x4d inverse<Matrix4x4d>(const Matrix4x4d& value)
  {
    return value.inverse();
  }

}

#include <agxData/Type.h>

AGX_TYPE_BINDING(agx::Matrix4x4f, "Matrix4x4")
AGX_TYPE_BINDING(agx::Matrix4x4d, "Matrix4x4")

#endif /* AGX_MATRIX4X4_H */
