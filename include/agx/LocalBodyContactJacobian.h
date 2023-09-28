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

#ifndef AGX_LOCALBODYCONTACTJACOBIAN_H
#define AGX_LOCALBODYCONTACTJACOBIAN_H

#include <agx/Jacobian.h>
#include <agxSIMD/Jacobian.h>

namespace agx
{
  template <typename JacobianElementT> class LocalBodyContactJacobianT;
  template <typename JacobianElementT> class LocalRollingResistanceJacobianT;
  template <typename JacobianElementT> class LocalTangentJacobianT;

  typedef LocalBodyContactJacobianT<Jacobian6DOFElement32> LocalBodyContactJacobian32;
  typedef LocalBodyContactJacobianT<Jacobian6DOFElement64> LocalBodyContactJacobian64;
  typedef LocalBodyContactJacobianT<Jacobian6DOFElement> LocalBodyContactJacobian;

  typedef LocalRollingResistanceJacobianT<Jacobian6DOFElement32> LocalRollingResistanceJacobian32;
  typedef LocalRollingResistanceJacobianT<Jacobian6DOFElement64> LocalRollingResistanceJacobian64;
  typedef LocalRollingResistanceJacobianT<Jacobian6DOFElement> LocalRollingResistanceJacobian;

  typedef LocalTangentJacobianT<Jacobian6DOFElement32> LocalTangentJacobian32;
  typedef LocalTangentJacobianT<Jacobian6DOFElement64> LocalTangentJacobian64;
  typedef LocalTangentJacobianT<Jacobian6DOFElement> LocalTangentJacobian;

}

DOXYGEN_START_INTERNAL_BLOCK()
namespace agxSIMD
{
  typedef agx::LocalBodyContactJacobianT<agxSIMD::Jacobian6DOFElement32> LocalBodyContactJacobian32;
  typedef agx::LocalBodyContactJacobianT<agxSIMD::Jacobian6DOFElement64> LocalBodyContactJacobian64;
  typedef agx::LocalBodyContactJacobianT<agxSIMD::Jacobian6DOFElement> LocalBodyContactJacobian;

  typedef agx::LocalRollingResistanceJacobianT<agxSIMD::Jacobian6DOFElement32> LocalRollingResistanceJacobian32;
  typedef agx::LocalRollingResistanceJacobianT<agxSIMD::Jacobian6DOFElement64> LocalRollingResistanceJacobian64;
  typedef agx::LocalRollingResistanceJacobianT<agxSIMD::Jacobian6DOFElement> LocalRollingResistanceJacobian;

  typedef agx::LocalTangentJacobianT<agxSIMD::Jacobian6DOFElement32> LocalTangentJacobian32;
  typedef agx::LocalTangentJacobianT<agxSIMD::Jacobian6DOFElement64> LocalTangentJacobian64;
  typedef agx::LocalTangentJacobianT<agxSIMD::Jacobian6DOFElement> LocalTangentJacobian;
}
DOXYGEN_END_INTERNAL_BLOCK()


namespace agx
{
  ///////////////////////////////////////////////////////////////

  template <typename T>
  class LocalTangentJacobianT
  {
  public:
    typedef T JacobianElementT;

  public:
    JacobianElementT& u();
    JacobianElementT& v();

    const JacobianElementT& u() const;
    const JacobianElementT& v() const;

  private:
    JacobianElementT m_u;
    JacobianElementT m_v;
  };

  ///////////////////////////////////////////////////////////////

  template <typename T>
  class LocalRollingResistanceJacobianT
  {
  public:
    typedef T JacobianElementT;

  public:
    JacobianElementT& normal();
    const JacobianElementT& normal() const;

    LocalTangentJacobianT<JacobianElementT>& tangent();
    const LocalTangentJacobianT<JacobianElementT>& tangent() const;

  private:
    JacobianElementT m_normal;
    LocalTangentJacobianT<JacobianElementT> m_tangent;
  };

  ///////////////////////////////////////////////////////////////

  template <typename T>
  class LocalBodyContactJacobianT
  {
  public:
    typedef T JacobianElementT;
    static const bool IsLocal = true;

  public:
    JacobianElementT& normal();
    const JacobianElementT& normal() const;

    LocalTangentJacobianT<JacobianElementT>& tangent();
    const LocalTangentJacobianT<JacobianElementT>& tangent() const;

    LocalRollingResistanceJacobianT<JacobianElementT>& rollingResistance();
    const LocalRollingResistanceJacobianT<JacobianElementT>& rollingResistance() const;

    JacobianElementT& get(size_t rowIndex);
    const JacobianElementT& get(size_t rowIndex) const;

  public:
    // Not used, only for template compatibility
    template <typename T2>
    void setup(const T2&, size_t) const {}

  private:
    JacobianElementT m_normal;
    LocalTangentJacobianT<JacobianElementT> m_tangent;
    LocalRollingResistanceJacobianT<JacobianElementT> m_rollingResistance;
  };



  /* Implementation */

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianElementT& LocalTangentJacobianT<JacobianElementT>::u()
  {
    return m_u;
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const JacobianElementT& LocalTangentJacobianT<JacobianElementT>::u() const
  {
    return m_u;
  }


  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianElementT& LocalTangentJacobianT<JacobianElementT>::v()
  {
    return m_v;
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const JacobianElementT& LocalTangentJacobianT<JacobianElementT>::v() const
  {
    return m_v;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianElementT& LocalRollingResistanceJacobianT<JacobianElementT>::normal()
  {
    return m_normal;
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const JacobianElementT& LocalRollingResistanceJacobianT<JacobianElementT>::normal() const
  {
    return m_normal;
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE LocalTangentJacobianT<JacobianElementT>& LocalRollingResistanceJacobianT<JacobianElementT>::tangent()
  {
    return m_tangent;
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const LocalTangentJacobianT<JacobianElementT>& LocalRollingResistanceJacobianT<JacobianElementT>::tangent() const
  {
    return m_tangent;
  }


  /////////////////////////////////////////////////////////////////////////////////////////////////

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianElementT& LocalBodyContactJacobianT<JacobianElementT>::normal()
  {
    return m_normal;
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const JacobianElementT& LocalBodyContactJacobianT<JacobianElementT>::normal() const
  {
    return m_normal;
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE LocalTangentJacobianT<JacobianElementT>& LocalBodyContactJacobianT<JacobianElementT>::tangent()
  {
    return m_tangent;
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const LocalTangentJacobianT<JacobianElementT>& LocalBodyContactJacobianT<JacobianElementT>::tangent() const
  {
    return m_tangent;
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE LocalRollingResistanceJacobianT<JacobianElementT>& LocalBodyContactJacobianT<JacobianElementT>::rollingResistance()
  {
    return m_rollingResistance;
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const LocalRollingResistanceJacobianT<JacobianElementT>& LocalBodyContactJacobianT<JacobianElementT>::rollingResistance() const
  {
    return m_rollingResistance;
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianElementT& LocalBodyContactJacobianT<JacobianElementT>::get(size_t rowIndex)
  {
    return ((JacobianElementT *)this)[rowIndex];
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const JacobianElementT& LocalBodyContactJacobianT<JacobianElementT>::get(size_t rowIndex) const
  {
    return ((const JacobianElementT *)this)[rowIndex];
  }


}


#endif /* AGX_LOCALBODYCONTACTJACOBIAN_H */
