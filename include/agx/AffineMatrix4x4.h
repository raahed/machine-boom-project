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
#ifndef AGX_AFFINEMATRIX4X4_H
#define AGX_AFFINEMATRIX4X4_H

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4714) // Disable warnings about not able to force inline
#endif

#include <agx/config/AGX_USE_SSE.h>
#include <agx/Matrix4x4.h>


namespace agx
{
  class EulerAngles;
  class OrthoMatrix3x3;

#define AGX_AFFINEMATRIX4X4_SET_ROW(row, v1, v2, v3 )    \
  this->m_data[(row)][0] = (v1); \
  this->m_data[(row)][1] = (v2); \
  this->m_data[(row)][2] = (v3);

#define AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(a,b,r,c) \
   ((a).m_data[r][0] * (b).m_data[0][c]) \
  +((a).m_data[r][1] * (b).m_data[1][c]) \
  +((a).m_data[r][2] * (b).m_data[2][c])

  /**
    Matrix class for rigid transformations (translation, rotation).
    Translations are stored in the last row.
    The last column will be assumed to be [0 0 0 1]' in all operations,
    even if it is set to something else.
    The submatrix containing the first three rows and columns contains
    the rotational part and is assumed to be orthonormal.
  */
  template <typename T>
  class AffineMatrix4x4T : public Matrix4x4T<T>
  {
    public:
    /*=====================================================
                        Constructors
    =======================================================*/

    /**
    Creates a new matrix, initialized to be an identity matrix.
    */
    AffineMatrix4x4T();

    /** Copy constructor */
    template <typename T2>
    explicit AffineMatrix4x4T( const AffineMatrix4x4T<T2>& mat );

    /** Create a matrix from a vector of 16 reals */
    explicit AffineMatrix4x4T( T const * const ptr );


    /** Create a matrix from a quaternion */
    explicit AffineMatrix4x4T(const QuatT<T>& rotation, const Vec3T<T>& translation = Vec3T<T>());

    /** Create a matrix from Euler angles */
    explicit AffineMatrix4x4T( const EulerAngles& rotation, const Vec3T<T>& translation = Vec3T<T>());

    /** Create a matrix from a 3x3 rotation matrix */
    explicit AffineMatrix4x4T( const OrthoMatrix3x3& rotation, const Vec3T<T>& translation = Vec3T<T>());

    /** Create a matrix from 16 T scalars */
    AffineMatrix4x4T( T a00, T a01, T a02, T a03,
                     T a10, T a11, T a12, T a13,
                     T a20, T a21, T a22, T a23,
                     T a30, T a31, T a32, T a33 );


    /** Destructor */
    ~AffineMatrix4x4T() = default;

    /*=====================================================
                          Accessors
    =======================================================*/

    /**
    Quick inverse, transpose rotation part, and change sign of translation part.
    \return the inverse
    */
    AffineMatrix4x4T<T> inverse() const;

    /**
    \return The inverse translation
    */
    Vec3T<T> getInvTranslate() const;


    /*=====================================================
                          Mutators
    =======================================================*/

    /**
    Set the entire matrix using values from the array with 16 elements using the following order:

    m_data[0][0] = ptr[0];
    m_data[0][1] = ptr[1];
    m_data[0][2] = ptr[2];
    m_data[0][3] = ptr[3];

    m_data[1][0] = ptr[4];

    etc.

    \param ptr - pointer to a vector with 16 elements
    \return reference of the modified matrix with the new values
    */
    AffineMatrix4x4T<T>& set(T const * const ptr);


    /**
    Set the entire matrix using values from the 16 elements using the following order:

    m_data[0][0] = a00;
    m_data[0][1] = a01;
    m_data[0][2] = a02;
    m_data[0][3] = a03;

    m_data[1][0] = a04;

    etc.

    \param a00,a01, a02,a03,a10,a11,a12, a13,a20,a21,a22,a23,a30,a31,a32,a33 - Matrix elements
    \return a reference to the modified matrix with the new values
    */
    AffineMatrix4x4T<T>& set(T a00, T a01, T a02, T a03,
      T a10, T a11, T a12, T a13,
      T a20, T a21, T a22, T a23,
      T a30, T a31, T a32, T a33);

    /**
    Set the rotational part of the matrix using the specified quaternion
    and the translational part to 0,0,0

    \param q - Specified quaternion
    \returns a reference to the modified matrix
    */
    AffineMatrix4x4T<T>& set(const QuatT<T>& q);

    /**
    Set the rotational part of the matrix using the specified rotation matrix
    and the translational part to 0,0,0

    \param m3 - Specified rotation matrix
    \returns a reference to the modified matrix
    */
    AffineMatrix4x4T<T>& set(const OrthoMatrix3x3& m3);


    /**
    Set the rotational part of the matrix using the specified euler angles
    and the translational part to 0,0,0

    \param euler - Specified euler angles rotation
    \returns a reference to the modified matrix
    */
    AffineMatrix4x4T<T>& set(const EulerAngles& euler);


    /**
    Set the rotational part of the matrix using the specified quaternion
    leaving the translational part untouched

    \param q - Specified quaternion
    \returns a reference to the modified matrix
    */
    AffineMatrix4x4T<T>& setRotate(const QuatT<T>& q);

    /**
    Set the rotational part of the matrix using the specified rotation matrix
    leaving the translational part untouched

    \param m3 - Specified rotation matrix
    \returns a reference to the modified matrix
    */
    AffineMatrix4x4T<T>& setRotate(const OrthoMatrix3x3& m3);


    /**
    Set the rotational part of the matrix using the specified euler angles
    leaving the translational part untouched.

    \param euler - Specified euler angles rotation
    \returns a reference to the modified matrix
    */
    AffineMatrix4x4T<T>& setRotate(const EulerAngles& euler);

    /**
    Set the rotational part of the matrix using the specified two vectors. The
    resulting rotation will be a rotation matrix which rotates a vector from \p from to \p to
    leaving the translational part untouched.

    \param from - start of rotation
    \param to - end of rotation
    \returns a reference to the modified matrix
    */
    AffineMatrix4x4T<T>& setRotate(const Vec3T<T>& from, const Vec3T<T>& to);

    /**
    Set the rotational part of a matrix which rotate \p angle radians around vector \p axis
    \param angle - angle in radians
    \param axis - vector
    \returns A reference to the modified matrix
    */
    AffineMatrix4x4T<T>& setRotate(T angle, const Vec3T<T>& axis);

    /**
    Set the rotational part of a matrix which rotate \p angle radians around vector \p x \p y \p z
    \param angle - angle in radians
    \param x,y,z - vector
    \returns A reference to the modified matrix
    */
    AffineMatrix4x4T<T>& setRotate(T angle, T x, T y, T z);

    /**
    Set the rotational part of a matrix which rotate \p angle1 radians around vector \p axis1 and
    \p angle2 radians around vector \p axis2
    \p angle3 radians around vector \p axis3
    \param angle1 - angle in radians
    \param axis1 - vector
    \param angle2 - angle in radians
    \param axis2 - vector
    \param angle3 - angle in radians
    \param axis3 - vector
    \returns A reference to the modified matrix
    */
    AffineMatrix4x4T<T>& setRotate(T angle1, const Vec3T<T>& axis1,
      T angle2, const Vec3T<T>& axis2,
      T angle3, const Vec3T<T>& axis3);

    /**
    Set the translational part of the matrix using the vector \p t
    \param t - translation vector
    \returns A reference to the modified matrix
    */
    AffineMatrix4x4T<T>& setTranslate(const Vec3T<T>& t);

    /**
    Set the translational part of the matrix using the vector \p x, \p y, \p z
    \param tx,ty,tz - translation vector
    \returns A reference to the modified matrix
    */
    AffineMatrix4x4T<T>& setTranslate(T tx, T ty, T tz);

    /**
    Set the value of the matrix to the identity matrix
    \returns a reference to the modified matrix
    */
    AffineMatrix4x4T<T>& setIdentity();



    /*=====================================================
                        Static methods
    =======================================================*/

    /**
    Generates a new matrix of a specific type.
    */
    static AffineMatrix4x4T<T> crossMatrix(const Vec3T<T>& vec);

    /**
    Return a matrix which translates according to \p dv
    \param dv - translation vector
    \returns A new matrix with specified translation
    */
    static AffineMatrix4x4T<T> translate( const Vec3T<T>& dv );

    /**
    Return a matrix which translates according to vector [x,y,z]
    \param x - x element of vector
    \param y - y element of vector
    \param z - z element of vector
    \returns A new matrix with specified translation
    */
    static AffineMatrix4x4T<T> translate( T x, T y, T z );

    /**
    Return a matrix which rotate a vector from \p from to \p to
    \param from - start of rotation
    \param to - end of rotation
    \returns A new matrix with specified rotation
    */
    static AffineMatrix4x4T<T> rotate( const Vec3T<T>& from, const Vec3T<T>& to );

    /**
    Return a matrix which rotate \p angle radians around vector [x,y,z]
    \param angle - angle in radians
    \param x - x element of vector
    \param y - y element of vector
    \param z - z element of vector
    \returns A new matrix with specified rotation
    */
    static AffineMatrix4x4T<T> rotate( T angle, T x, T y, T z );

    /**
    Return a matrix which rotate \p angle radians around vector \p axis
    \param angle - angle in radians
    \param axis - vector
    \returns A new matrix with specified rotation
    */
    static AffineMatrix4x4T<T> rotate( T angle, const Vec3T<T>& axis );

    /**
    Return a matrix which rotate \p angle1 radians around vector \p axis1 and
    \p angle2 radians around vector \p axis2
    \p angle3 radians around vector \p axis3
    \param angle1 - angle in radians
    \param axis1 - vector
    \param angle2 - angle in radians
    \param axis2 - vector
    \param angle3 - angle in radians
    \param axis3 - vector
    \returns A new matrix with specified rotation
    */
    static AffineMatrix4x4T<T> rotate( T angle1, const Vec3T<T>& axis1,
                                          T angle2, const Vec3T<T>& axis2,
                                          T angle3, const Vec3T<T>& axis3 );
    static AffineMatrix4x4T<T> rotate(const EulerAngles& euler);


    /*=====================================================
                        Operators
    =======================================================*/

    AffineMatrix4x4T<T> operator* ( const AffineMatrix4x4T<T> &m ) const;
    Matrix4x4T<T> operator* ( const Matrix4x4T<T> &m ) const;
    AffineMatrix4x4T<T>& operator= ( const AffineMatrix4x4T<T>& rhs );
    Vec3T<T> operator* ( const Vec3T<T>& v ) const;
    Vec4T<T> operator* ( const Vec4T<T>& v ) const;

    void operator*= ( const AffineMatrix4x4T<T>& other );
    bool operator== ( const AffineMatrix4x4T<T>& m ) const;
    bool operator!= ( const AffineMatrix4x4T<T>& m ) const;

    /**
    Transforms a (mathematical) point, using both rotation and translation.
    */
    agx::Vec3T<T> transformPoint(const agx::Vec3T<T>& point) const;

    /**
    Transforms a (mathematical) vector, using only rotation.
    */
    agx::Vec3T<T> transformVector(const agx::Vec3T<T>& vector) const;

    void mult( const AffineMatrix4x4T<T>&, const AffineMatrix4x4T<T>& );
    void multSSE(const AffineMatrix4x4T<T>& lhs, const AffineMatrix4x4T<T>& rhs);
    using agx::Matrix4x4T<T>::preMult;
    using agx::Matrix4x4T<T>::postMult;
    void preMult( const AffineMatrix4x4T<T>& );
    void postMult( const AffineMatrix4x4T<T>& );
    AffineMatrix4x4T<T>& preMultTranslate( const Vec3T<T>& v );
    AffineMatrix4x4T<T>& postMultTranslate( const Vec3T<T>& v );

    /// \return the element at i,j.
    Real at(size_t i, size_t j) const;

    /// Set the value of element i,j.
    void set(Real val, size_t i, size_t j);

  protected:
    bool invert( const AffineMatrix4x4T<T>& rhs );
  };


  typedef AffineMatrix4x4T<Real> AffineMatrix4x4;

  typedef AffineMatrix4x4T<Real32> AffineMatrix4x4f;
  typedef AffineMatrix4x4T<Real64> AffineMatrix4x4d;
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
  AGX_FORCE_INLINE AffineMatrix4x4T<T>::AffineMatrix4x4T()
  {
    this->setIdentity();
  }

  template <typename T> template <typename T2>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>::AffineMatrix4x4T( const AffineMatrix4x4T<T2>& mat ) : Matrix4x4T<T>( mat )
  {
  }

  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>::AffineMatrix4x4T( T a00, T a01, T a02, T a03,
                                           T a10, T a11, T a12, T a13,
                                           T a20, T a21, T a22, T a23,
                                           T a30, T a31, T a32, T a33 ) : Matrix4x4T<T>(a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33)
  {
    agxAssert(this->isRigidTransformation());
  }

  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>::AffineMatrix4x4T( T const * const ptr )
  {
    set( ptr );
  }


  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>::AffineMatrix4x4T(const QuatT<T>& rotation, const Vec3T<T>& translation) : Matrix4x4T<T>(rotation, translation)
  {
  }

  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>::AffineMatrix4x4T( const OrthoMatrix3x3& rotation, const Vec3T<T>& translation ) : Matrix4x4T<T>(rotation, translation)
  {
  }

  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>::AffineMatrix4x4T(agx::EulerAngles const& rotation, const Vec3T<T>& translation ) : Matrix4x4T<T>(rotation, translation)
  {
  }



  /*=====================================================
                        Queries
  =======================================================*/


  template <typename T>
  inline AffineMatrix4x4T<T> AffineMatrix4x4T<T>::inverse() const
  {
    AffineMatrix4x4T<T> m;
    m.invert( *this );
    return m;
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> AffineMatrix4x4T<T>::getInvTranslate() const
  {
    T x = this->m_data[3][0];
    T y = this->m_data[3][1];
    T z = this->m_data[3][2];

    Vec3T<T> invTranslate;

    invTranslate[0] = -( x * this->m_data[0][0] + y* this->m_data[0][1] + z * this->m_data[0][2]);
    invTranslate[1] = -( x * this->m_data[1][0] + y* this->m_data[1][1] + z * this->m_data[1][2]);
    invTranslate[2] = -( x * this->m_data[2][0] + y* this->m_data[2][1] + z * this->m_data[2][2]);

    return invTranslate;
  }

  template <typename T>
  AGX_FORCE_INLINE bool AffineMatrix4x4T<T>::operator== (const AffineMatrix4x4T<T>& m) const
  {
    const int bitCompare = memcmp(this->m_data, m.m_data, sizeof(this->m_data));
    return (bitCompare == 0);
  }

  template <typename T>
  AGX_FORCE_INLINE bool AffineMatrix4x4T<T>::operator!= (const AffineMatrix4x4T<T>& m) const
  {
    const int bitCompare = memcmp(this->m_data, m.m_data, sizeof(this->m_data));
    return (bitCompare != 0);
  }


  /*=====================================================
                        Setters
  =======================================================*/
  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::set( T const * const ptr )
  {

    #if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstringop-overflow"
    #endif

    memcpy(this->m_data, ptr, sizeof(T)*16);

    #if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic pop
    #endif

    agxAssert(this->isRigidTransformation());
    return *this;
  }


  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::set( T a00, T a01, T a02, T a03,
                                          T a10, T a11, T a12, T a13,
                                          T a20, T a21, T a22, T a23,
                                          T a30, T a31, T a32, T a33 )
  {

    Matrix4x4T<T>::set(a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33);
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::set(const QuatT<T>& q)
  {
    this->setRotate(q);
    this->setTranslate(0.0, 0.0, 0.0);

    return *this;
  }


  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::set( const OrthoMatrix3x3& m3 )
  {
    this->setRotate(m3);
    this->setTranslate(0.0, 0.0, 0.0);

    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::setRotate(const QuatT<T>& q)
  {
    Matrix4x4T<T>::setRotate(q);
    return *this;
  }


  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::setRotate( const OrthoMatrix3x3& m3 )
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
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::setRotate( const Vec3T<T>& from, const Vec3T<T>& to )
  {
    QuatT<T> quat;
    quat.setRotate(from,to);
    this->setRotate(quat);
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::setRotate( T angle, const Vec3T<T>& axis )
  {
    QuatT<T> quat;
    quat.setRotate( angle, axis);
    this->setRotate(quat);
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::setRotate( T angle, T x, T y, T z )
  {
    QuatT<T> quat;
    quat.setRotate( angle, x, y, z);
    this->setRotate(quat);
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::setRotate( T angle1, const Vec3T<T>& axis1,
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
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::setTranslate( const Vec3T<T>& t )
  {
    this->m_data[3][0] = t[0];
    this->m_data[3][1] = t[1];
    this->m_data[3][2] = t[2];

    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::setTranslate(T tx, T ty, T tz)
  {
    this->m_data[3][0] = tx;
    this->m_data[3][1] = ty;
    this->m_data[3][2] = tz;

    return *this;
  }


  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::setIdentity()
  {
    this->set(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
    return *this;
  }


  /*=====================================================
                      Static methods
  =======================================================*/

  template <typename T>
  inline AffineMatrix4x4T<T> AffineMatrix4x4T<T>::crossMatrix(const Vec3T<T>& vec)
  {
    return AffineMatrix4x4T<T>(
       0,        -vec.z(),    vec.y(), 0,
       vec.z(),   0,         -vec.x(), 0,
      -vec.y(),   vec.x(),    0,       0,
       0,         0,          0,       1);
  }


  template <typename T>
  inline AffineMatrix4x4T<T> AffineMatrix4x4T<T>::translate( const Vec3T<T>& dv )
  {
    AffineMatrix4x4T<T> m;
    m.setTranslate(dv);
    return m;
  }

  template <typename T>
  inline AffineMatrix4x4T<T> AffineMatrix4x4T<T>::translate( T x, T y, T z )
  {
    AffineMatrix4x4T<T> m;
    m.setTranslate(x, y, z);
    return m;
  }

  template <typename T>
  inline AffineMatrix4x4T<T> AffineMatrix4x4T<T>::rotate( const Vec3T<T>& from, const Vec3T<T>& to )
  {
    AffineMatrix4x4T<T> m;
    m.setRotate(from, to);
    return m;
  }

  template <typename T>
  inline AffineMatrix4x4T<T> AffineMatrix4x4T<T>::rotate( T angle, T x, T y, T z )
  {
    AffineMatrix4x4T<T> m;
    m.setRotate(angle, x, y, z);
    return m;
  }

  template <typename T>
  inline AffineMatrix4x4T<T> AffineMatrix4x4T<T>::rotate( T angle, const Vec3T<T>& axis )
  {
    AffineMatrix4x4T<T> m;
    m.setRotate(angle, axis);
    return m;
  }

  template <typename T>
  inline AffineMatrix4x4T<T> AffineMatrix4x4T<T>::rotate( T angle1, const Vec3T<T>& axis1,
                                        T angle2, const Vec3T<T>& axis2,
                                        T angle3, const Vec3T<T>& axis3 )
  {
    AffineMatrix4x4T<T> m;
    m.setRotate(angle1, axis1, angle2, axis2, angle3, axis3);
    return m;
  }


  /*=====================================================
                      Operators
  =======================================================*/
  template <typename T>
  AGX_FORCE_INLINE void AffineMatrix4x4T<T>::mult( const AffineMatrix4x4T<T>& lhs, const AffineMatrix4x4T<T>& rhs )
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

    this->m_data[0][0] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(lhs, rhs, 0, 0);
    this->m_data[0][1] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(lhs, rhs, 0, 1);
    this->m_data[0][2] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(lhs, rhs, 0, 2);

    this->m_data[1][0] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(lhs, rhs, 1, 0);
    this->m_data[1][1] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(lhs, rhs, 1, 1);
    this->m_data[1][2] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(lhs, rhs, 1, 2);

    this->m_data[2][0] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(lhs, rhs, 2, 0);
    this->m_data[2][1] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(lhs, rhs, 2, 1);
    this->m_data[2][2] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(lhs, rhs, 2, 2);

    this->m_data[3][0] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(lhs, rhs, 3, 0) + rhs.m_data[3][0];
    this->m_data[3][1] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(lhs, rhs, 3, 1) + rhs.m_data[3][1];
    this->m_data[3][2] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3(lhs, rhs, 3, 2) + rhs.m_data[3][2];

  }

  template <typename T>
  AGX_FORCE_INLINE void AffineMatrix4x4T<T>::preMult( const AffineMatrix4x4T<T>& other )
  {

    // more efficient method just use a T[4] for temporary storage.
    T t[4];
    for(size_t col=0; col<3; ++col) {
      t[0] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3( other, *this, 0, col );
      t[1] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3( other, *this, 1, col );
      t[2] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3( other, *this, 2, col );
      t[3] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3( other, *this, 3, col );
      this->m_data[0][col] = t[0];
      this->m_data[1][col] = t[1];
      this->m_data[2][col] = t[2];
      this->m_data[3][col] = t[3] + this->m_data[3][col];
    }
  }

  template <typename T>
  AGX_FORCE_INLINE void AffineMatrix4x4T<T>::postMult( const AffineMatrix4x4T<T>& other )
  {

    // more efficient method just use a T[3] for temporary storage.
    T t[3];
    for(size_t row=0; row<3; ++row)
    {
      t[0] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3( *this, other, row, 0 );
      t[1] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3( *this, other, row, 1 );
      t[2] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3( *this, other, row, 2 );

      AGX_AFFINEMATRIX4X4_SET_ROW(row, t[0], t[1], t[2] )
    }
    // treat last row differently
    t[0] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3( *this, other, 3, 0 ) + other.m_data[3][0];
    t[1] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3( *this, other, 3, 1 ) + other.m_data[3][1];
    t[2] = AGX_AFFINEMATRIX4X4_INNER_PRODUCT_3( *this, other, 3, 2 ) + other.m_data[3][2];
    AGX_AFFINEMATRIX4X4_SET_ROW(3, t[0], t[1], t[2] )
  }

  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::preMultTranslate( const Vec3T<T>& v )
  {
    for (unsigned i = 0; i < 3; ++i)
    {
      this->m_data[3][0] += v[i]*this->m_data[i][0];
      this->m_data[3][1] += v[i]*this->m_data[i][1];
      this->m_data[3][2] += v[i]*this->m_data[i][2];
      this->m_data[3][3] += v[i]*this->m_data[i][3];
    }
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::postMultTranslate( const Vec3T<T>& v )
  {
    for (unsigned i = 0; i < 3; ++i)
    {
      this->m_data[0][i] += v[i]*this->m_data[0][3];
      this->m_data[1][i] += v[i]*this->m_data[1][3];
      this->m_data[2][i] += v[i]*this->m_data[2][3];
      this->m_data[3][i] += v[i]*this->m_data[3][3];
    }
    return *this;
  }



  template <typename T>
  AGX_FORCE_INLINE AffineMatrix4x4T<T>& AffineMatrix4x4T<T>::operator= ( const AffineMatrix4x4T<T>& rhs )
  {
    if ( this != &rhs )
      this->set( rhs.ptr() );

    return *this;
  }


  template <typename T>
  inline AffineMatrix4x4T<T> AffineMatrix4x4T<T>::operator* ( const AffineMatrix4x4T<T> &m ) const
  {
    AffineMatrix4x4T<T> r;
    r.mult( *this, m );
    return  r;
  }

  template <typename T>
  inline Matrix4x4T<T> AffineMatrix4x4T<T>::operator* ( const Matrix4x4T<T> &m ) const
  {
    Matrix4x4T<T> r;
    r.mult( *this, m );
    return  r;
  }


  template <typename T>
  AGX_FORCE_INLINE void AffineMatrix4x4T<T>::operator*= ( const AffineMatrix4x4T<T>& other )
  {
    if ( this == &other )
    {
      AffineMatrix4x4T<T> temp( other );
      postMult( temp );
    }
    else postMult( other );
  }


  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> AffineMatrix4x4T<T>::operator* (const Vec3T<T>& v) const
  {
    return postMult(v);
  }

  template <typename T>
  AGX_FORCE_INLINE Vec4T<T> AffineMatrix4x4T<T>::operator* (const Vec4T<T>& v) const
  {
    return postMult(v);
  }

  template <typename T>
  AGX_FORCE_INLINE Vec3T<T> operator* (const Vec3T<T>& v, const AffineMatrix4x4T<T>& m)
  {
    return m.preMult(v);
  }
  template <typename T>


  AGX_FORCE_INLINE Vec4T<T> operator* (const Vec4T<T>& v, const AffineMatrix4x4T<T>& m)
  {
    return m.preMult(v);
  }


  template <typename T>
  AGX_FORCE_INLINE bool AffineMatrix4x4T<T>::invert( const AffineMatrix4x4T<T>& rhs)
  {
    T x,y,z;

    // diagonal
    this->m_data[0][0] = rhs.m_data[0][0];
    this->m_data[1][1] = rhs.m_data[1][1];
    this->m_data[2][2] = rhs.m_data[2][2];

    // rotation
    this->m_data[0][1] = rhs.m_data[1][0];
    this->m_data[1][0] = rhs.m_data[0][1];
    this->m_data[0][2] = rhs.m_data[2][0];
    this->m_data[2][0] = rhs.m_data[0][2];
    this->m_data[1][2] = rhs.m_data[2][1];
    this->m_data[2][1] = rhs.m_data[1][2];

    x = rhs.m_data[3][0];
    y = rhs.m_data[3][1];
    z = rhs.m_data[3][2];

    // translation
    this->m_data[3][0] = -( x * this->m_data[0][0] + y* this->m_data[1][0] + z * this->m_data[2][0]);
    this->m_data[3][1] = -( x * this->m_data[0][1] + y* this->m_data[1][1] + z * this->m_data[2][1]);
    this->m_data[3][2] = -( x * this->m_data[0][2] + y* this->m_data[1][2] + z * this->m_data[2][2]);

    return true;
  }

  template <typename T>
  AGX_FORCE_INLINE bool equivalent( const agx::AffineMatrix4x4T<T>& a, const agx::AffineMatrix4x4T<T>& b, T epsilon = T(AGX_EQUIVALENT_EPSILON) )
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


  template <>
  AGX_FORCE_INLINE void AffineMatrix4x4T<Real32>::multSSE(const AffineMatrix4x4T<Real32>& lhs, const AffineMatrix4x4T<Real32>& rhs)
  {
    #if AGX_USE_SSE()
    multSSE32Implementation(this->ptr(), lhs.ptr(), rhs.ptr());
    #else
    mult(lhs, rhs);
    #endif
  }

  template <>
  AGX_FORCE_INLINE void AffineMatrix4x4T<Real64>::multSSE(const AffineMatrix4x4T<Real64>& lhs, const AffineMatrix4x4T<Real64>& rhs)
  {
    #if AGX_USE_SSE()
    multSSE64Implementation(this->ptr(), lhs.ptr(), rhs.ptr());
    #else
    mult(lhs, rhs);
    #endif
  }


  template <>
  inline AffineMatrix4x4f inverse<AffineMatrix4x4f>(const AffineMatrix4x4f& value)
  {
    return value.inverse();
  }


  template <>
  inline AffineMatrix4x4d inverse<AffineMatrix4x4d>(const AffineMatrix4x4d& value)
  {
    return value.inverse();
  }


  template <typename T>
  AGX_FORCE_INLINE Real AffineMatrix4x4T<T>::at(size_t i, size_t j) const
  {
    return Real((*this)(i,j));
  }

  template <typename T>
  AGX_FORCE_INLINE void AffineMatrix4x4T<T>::set(Real val, size_t i, size_t j)
  {
    (*this)(i,j) = T(val);
  }


  template <typename T>
  AGX_FORCE_INLINE agx::Vec3T<T> AffineMatrix4x4T<T>::transformPoint(const agx::Vec3T<T>& point) const
  {
    return point * (*this);
  }

  template <typename T>
  AGX_FORCE_INLINE agx::Vec3T<T> AffineMatrix4x4T<T>::transformVector(const agx::Vec3T<T>& vector) const
  {
    return this->transform3x3(vector);
  }



}

#include <agxData/Type.h>

AGX_TYPE_BINDING(agx::AffineMatrix4x4f, "AffineMatrix4x4")
AGX_TYPE_BINDING(agx::AffineMatrix4x4d, "AffineMatrix4x4")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
