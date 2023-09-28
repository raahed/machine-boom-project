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

#ifndef AGX_SIMD_JACOBIAN_H
#define AGX_SIMD_JACOBIAN_H

#include <agxSIMD/Vec3.h>
#include <agx/Jacobian.h>
// #include <agx/BodyContactJacobian.h>
#include <agxData/StackArray.h>

DOXYGEN_START_INTERNAL_BLOCK()
namespace agxSIMD
{

  template <typename T>
  class Jacobian6DOFElementT
  {
  public:
    typedef T Vec3T;

  public:
    Jacobian6DOFElementT() : m_spatial(0, 0, 0), m_rotational(0, 0, 0){}

    Vec3T& spatial();
    Vec3T& rotational();

    const Vec3T& spatial() const;
    const Vec3T& rotational() const;


    typedef agx::Vec3T<typename T::RealT> NonSSEVec3T;

    void store(agx::Jacobian6DOFElementT<NonSSEVec3T>& target) const;

    void set(const T& spatial, const T& rotational);

    typename T::RealT mult(const T& v1, const T& v2) const;

    static typename T::RealT mult(const Jacobian6DOFElementT<T>& lhs, const Jacobian6DOFElementT<T>& rhs);

    void scale(const Vec3T& multiplier);

    void update(const T& dl, T& linVel, T& angVel) const;

  private:
    Vec3T m_spatial;
    Vec3T m_rotational;
  };

  typedef Jacobian6DOFElementT<Vec3d> Jacobian6DOFElement64;
  typedef Jacobian6DOFElementT<Vec3f> Jacobian6DOFElement32;
  typedef Jacobian6DOFElementT<Vec3> Jacobian6DOFElement;


  typedef agxData::Array<Jacobian6DOFElement32> Jacobian6DOFElement32Array;
  typedef agxData::Array<Jacobian6DOFElement64> Jacobian6DOFElement64Array;
  typedef agxData::Array<Jacobian6DOFElement> Jacobian6DOFElementArray;


#if 0
  template <typename T>
  struct JacobianMetaT
  {
    typedef T Vec3T;

    Vec3T localPoint1;
    Vec3T localPoint2;

    Vec3T normal;
    Vec3T uTangent;
    Vec3T vTangent;

    Vec3T normalNeg;
    Vec3T uTangentNeg;
    Vec3T vTangentNeg;

    JacobianMetaT();
    JacobianMetaT(const agx::JacobianMeta& G);
  };
#endif

  /* Implementation */


  //-----------------------------------------------------------------------------------------------------

  template <typename T>
  AGX_FORCE_INLINE typename Jacobian6DOFElementT<T>::Vec3T& Jacobian6DOFElementT<T>::spatial()
  {
    return m_spatial;
  }

  template <typename T>
  AGX_FORCE_INLINE typename Jacobian6DOFElementT<T>::Vec3T& Jacobian6DOFElementT<T>::rotational()
  {
    return m_rotational;
  }

  template <typename T>
  AGX_FORCE_INLINE const typename Jacobian6DOFElementT<T>::Vec3T& Jacobian6DOFElementT<T>::spatial() const
  {
    return m_spatial;
  }

  template <typename T>
  AGX_FORCE_INLINE const typename Jacobian6DOFElementT<T>::Vec3T& Jacobian6DOFElementT<T>::rotational() const
  {
    return m_rotational;
  }


  template <typename T>
  AGX_FORCE_INLINE void Jacobian6DOFElementT<T>::set(const T& spatial, const T& rotational)
  {
    m_spatial = spatial;
    m_rotational = rotational;
  }

  template <typename T>
  AGX_FORCE_INLINE void Jacobian6DOFElementT<T>::store(agx::Jacobian6DOFElementT<NonSSEVec3T>& target) const
  {
    m_spatial.store(target.spatial());
    m_rotational.store(target.rotational());
  }

  template <typename T>
  AGX_FORCE_INLINE typename T::RealT Jacobian6DOFElementT<T>::mult(const T& v1, const T& v2) const
  {
    return T::innerProduct(m_spatial, m_rotational, v1, v2);
  }

  template <typename T>
  AGX_FORCE_INLINE typename T::RealT Jacobian6DOFElementT<T>::mult(const Jacobian6DOFElementT<T>& lhs, const Jacobian6DOFElementT<T>& rhs)
  {
    return T::innerProduct(lhs.spatial(), lhs.rotational(), rhs.spatial(), rhs.rotational());
  }

  template <typename T>
  AGX_FORCE_INLINE void Jacobian6DOFElementT<T>::scale(const Vec3T& multiplier)
  {
    m_spatial = m_spatial.scale(multiplier);
    m_rotational = m_rotational.scale(multiplier);
  }

  template <typename T>
  AGX_FORCE_INLINE void Jacobian6DOFElementT<T>::update(const T& dl, T& linVel, T& angVel) const
  {
    linVel = T::madd(linVel, m_spatial, dl);
    angVel = T::madd(angVel, m_rotational, dl);
  }

  //-----------------------------------------------------------------------------------------------------


#if 0
  template <typename T>
  AGX_FORCE_INLINE JacobianMetaT<T>::JacobianMetaT()
  {
  }

  template <typename T>
  AGX_FORCE_INLINE JacobianMetaT<T>::JacobianMetaT(const agx::JacobianMetaT<T>& gmeta) :
   localPoint1(gmeta.localPoint1),
   localPoint2(gmeta.localPoint2),
   normal(gmeta.normal),
   uTangent(gmeta.uTangent),
   vTangent(gmeta.vTangent)
  {
    normalNeg = normal.negate();
    uTangentNeg = uTangent.negate();
    vTangentNeg = vTangent.negate();
  }
#endif

}
DOXYGEN_END_INTERNAL_BLOCK()


#endif /* AGX_SIMD_JACOBIAN_H */
