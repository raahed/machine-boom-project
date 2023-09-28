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

#ifndef AGX_BODYCONTACTJACOBIAN_H
#define AGX_BODYCONTACTJACOBIAN_H

#include <agx/Jacobian.h>
#include <agxSIMD/Jacobian.h>


namespace agx
{
  template <typename JacobianElementT> class BodyContactJacobianT;
  template <typename JacobianElementT> class RollingResistanceJacobianT;
  template <typename JacobianElementT> class TangentJacobianT;

  typedef BodyContactJacobianT<Jacobian6DOFElement32> BodyContactJacobian32;
  typedef BodyContactJacobianT<Jacobian6DOFElement64> BodyContactJacobian64;
  typedef BodyContactJacobianT<Jacobian6DOFElement> BodyContactJacobian;

  typedef RollingResistanceJacobianT<Jacobian6DOFElement32> RollingResistanceJacobian32;
  typedef RollingResistanceJacobianT<Jacobian6DOFElement64> RollingResistanceJacobian64;
  typedef RollingResistanceJacobianT<Jacobian6DOFElement> RollingResistanceJacobian;

  typedef TangentJacobianT<Jacobian6DOFElement32> TangentJacobian32;
  typedef TangentJacobianT<Jacobian6DOFElement64> TangentJacobian64;
  typedef TangentJacobianT<Jacobian6DOFElement> TangentJacobian;

}

DOXYGEN_START_INTERNAL_BLOCK()
namespace agxSIMD
{
  typedef agx::BodyContactJacobianT<agxSIMD::Jacobian6DOFElement32> BodyContactJacobian32;
  typedef agx::BodyContactJacobianT<agxSIMD::Jacobian6DOFElement64> BodyContactJacobian64;
  typedef agx::BodyContactJacobianT<agxSIMD::Jacobian6DOFElement> BodyContactJacobian;

  typedef agx::RollingResistanceJacobianT<agxSIMD::Jacobian6DOFElement32> RollingResistanceJacobian32;
  typedef agx::RollingResistanceJacobianT<agxSIMD::Jacobian6DOFElement64> RollingResistanceJacobian64;
  typedef agx::RollingResistanceJacobianT<agxSIMD::Jacobian6DOFElement> RollingResistanceJacobian;

  typedef agx::TangentJacobianT<agxSIMD::Jacobian6DOFElement32> TangentJacobian32;
  typedef agx::TangentJacobianT<agxSIMD::Jacobian6DOFElement64> TangentJacobian64;
  typedef agx::TangentJacobianT<agxSIMD::Jacobian6DOFElement> TangentJacobian;
  #if 0
  typedef agx::BodyContactJacobian32 BodyContactJacobian32;
  typedef agx::BodyContactJacobian64 BodyContactJacobian64;
  typedef agx::BodyContactJacobian BodyContactJacobian;

  typedef agx::RollingResistanceJacobian32 RollingResistanceJacobian32;
  typedef agx::RollingResistanceJacobian64 RollingResistanceJacobian64;
  typedef agx::RollingResistanceJacobian RollingResistanceJacobian;

  typedef agx::TangentJacobian32 TangentJacobian32;
  typedef agx::TangentJacobian64 TangentJacobian64;
  typedef agx::TangentJacobian TangentJacobian;

  // typedef agx::Jacobian6DOFElement64 Jacobian6DOFElement64;
  // typedef agx::Jacobian6DOFElement32 Jacobian6DOFElement32;
  // typedef agx::Jacobian6DOFElement Jacobian6DOFElement;
  #endif
}
DOXYGEN_END_INTERNAL_BLOCK()

namespace agx
{

  // Templates for binding non-SIMD arrays as SIMD data
  template <typename JacobianElementT> struct ComplementaryElementT {};

  template <> struct ComplementaryElementT<Jacobian6DOFElement64> { typedef agxSIMD::Jacobian6DOFElement64 Type; };
  template <> struct ComplementaryElementT<Jacobian6DOFElement32> { typedef agxSIMD::Jacobian6DOFElement32 Type; };
  template <> struct ComplementaryElementT<agxSIMD::Jacobian6DOFElement64> { typedef agx::Jacobian6DOFElement64 Type; };
  template <> struct ComplementaryElementT<agxSIMD::Jacobian6DOFElement32> { typedef agx::Jacobian6DOFElement32 Type; };

  ////////////////////////////////////////////////////////////////////////////////////////////

  template <typename T>
  class JacobianAccessorT
  {
  public:
    typedef T JacobianElementT;
    typedef agxData::Array<JacobianElementT> JacobianElementArrayT;
    typedef agxData::Array<typename ComplementaryElementT<JacobianElementT>::Type> ComplementaryElementArrayT;
    typedef TangentJacobianT<JacobianElementT> TangentJacobian;
    typedef RollingResistanceJacobianT<JacobianElementT> RollingResistanceJacobian;

  public:

    JacobianAccessorT();
    JacobianAccessorT(const JacobianAccessorT& other);

    JacobianAccessorT(const JacobianElementArrayT& jacobianElements, size_t offset);
    JacobianAccessorT(const ComplementaryElementArrayT& jacobianElements, size_t offset);

#ifndef AGX_DEBUG
    JacobianAccessorT(const JacobianElementT *jacobianElements, size_t offset);
#endif

    void setup(const JacobianElementArrayT& jacobianElements, size_t offset) const;
    void setup(const ComplementaryElementArrayT& jacobianElements, size_t offset) const;

    void setup(JacobianElementArrayT& jacobianElements, size_t offset);
    void setup(ComplementaryElementArrayT& jacobianElements, size_t offset);


    JacobianElementT& get(size_t index);
    const JacobianElementT& get(size_t index) const;

    template <typename AccessorT>
    AccessorT get(size_t index);

    template <typename AccessorT>
    const AccessorT get(size_t index) const;

  private:
    JacobianAccessorT& operator= (const JacobianAccessorT& other);

  private:
#ifdef AGX_DEBUG
    JacobianElementArrayT m_jacobianElements;
#else
    JacobianElementT *m_jacobianElements;
#endif
  };


  ////////////////////////////////////////////////////////////////////////////////////////////

  template <typename JacobianElementT>
  class TangentJacobianT : public JacobianAccessorT<JacobianElementT>
  {
  public:
    typedef JacobianAccessorT<JacobianElementT> BaseT;
    typedef typename BaseT::JacobianElementArrayT JacobianElementArrayT;
    typedef typename BaseT::ComplementaryElementArrayT ComplementaryElementArrayT;

  public:
    TangentJacobianT();
    TangentJacobianT(const JacobianElementArrayT& jacobianElements, size_t offset);
    TangentJacobianT(const ComplementaryElementArrayT& jacobianElements, size_t offset);

    JacobianElementT& u();
    const JacobianElementT& u() const;

    JacobianElementT& v();
    const JacobianElementT& v() const;

#ifndef AGX_DEBUG
    TangentJacobianT(const JacobianElementT *jacobianElements, size_t offset);
#endif
  };

  ////////////////////////////////////////////////////////////////////////////////////////////

  template <typename JacobianElementT>
  class RollingResistanceJacobianT : public JacobianAccessorT<JacobianElementT>
  {
  public:
    typedef JacobianAccessorT<JacobianElementT> BaseT;
    typedef typename BaseT::JacobianElementArrayT JacobianElementArrayT;
    typedef typename BaseT::ComplementaryElementArrayT ComplementaryElementArrayT;
    typedef typename BaseT::TangentJacobian TangentJacobian;

  public:
    RollingResistanceJacobianT();
    RollingResistanceJacobianT(const JacobianElementArrayT& jacobianElements, size_t offset);
    RollingResistanceJacobianT(const ComplementaryElementArrayT& jacobianElements, size_t offset);

    TangentJacobian tangent();
    const TangentJacobian tangent() const;

    JacobianElementT& normal();
    const JacobianElementT& normal() const;

#ifndef AGX_DEBUG
    RollingResistanceJacobianT(const JacobianElementT *jacobianElements, size_t offset);
#endif
  };

  ////////////////////////////////////////////////////////////////////////////////////////////

  template <typename JacobianElementT>
  class BodyContactJacobianT : public JacobianAccessorT<JacobianElementT>
  {
  public:
    typedef JacobianAccessorT<JacobianElementT> BaseT;
    typedef typename BaseT::JacobianElementArrayT JacobianElementArrayT;
    typedef typename BaseT::ComplementaryElementArrayT ComplementaryElementArrayT;
    typedef typename BaseT::TangentJacobian TangentJacobian;
    typedef typename BaseT::RollingResistanceJacobian RollingResistanceJacobian;

    static const bool IsLocal = false;

  public:
    BodyContactJacobianT();
    BodyContactJacobianT(const JacobianElementArrayT& jacobianElements, size_t offset);
    BodyContactJacobianT(const ComplementaryElementArrayT& jacobianElements, size_t offset);

    JacobianElementT& normal();
    const JacobianElementT& normal() const;

    TangentJacobian tangent();
    const TangentJacobian tangent() const;

    RollingResistanceJacobian rollingResistance();
    const RollingResistanceJacobian rollingResistance() const;

#ifndef AGX_DEBUG
    BodyContactJacobianT(const JacobianElementT *jacobianElements, size_t offset);
#endif
  };





  /* Implementation */

  ////////////////////////////////////////////////////////////////////////////////////////////

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianAccessorT<JacobianElementT>::JacobianAccessorT()
    : m_jacobianElements(nullptr)
  {
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianAccessorT<JacobianElementT>::JacobianAccessorT(const ComplementaryElementArrayT& jacobianElements, size_t offset) :
#ifdef AGX_DEBUG
    m_jacobianElements(reinterpret_cast<const JacobianElementArrayT&>(jacobianElements), IndexRange(offset, jacobianElements.size()))
#else
    m_jacobianElements((JacobianElementT *)jacobianElements.ptr() + offset)
#endif
  {
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianAccessorT<JacobianElementT>::JacobianAccessorT(const JacobianElementArrayT& jacobianElements, size_t offset) :
#ifdef AGX_DEBUG
  m_jacobianElements(jacobianElements, IndexRange(offset, jacobianElements.size()))
#else
    m_jacobianElements((JacobianElementT *)jacobianElements.ptr() + offset)
#endif
  {
  }


  template <typename JacobianElementT>
  AGX_FORCE_INLINE void JacobianAccessorT<JacobianElementT>::setup(ComplementaryElementArrayT& jacobianElements, size_t offset)
  {
#ifdef AGX_DEBUG
    m_jacobianElements = JacobianElementArrayT(reinterpret_cast<JacobianElementArrayT&>(jacobianElements), IndexRange(offset, jacobianElements.size()));
#else
    m_jacobianElements = (JacobianElementT *)jacobianElements.ptr() + offset;
#endif
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE void JacobianAccessorT<JacobianElementT>::setup(JacobianElementArrayT& jacobianElements, size_t offset)
  {
#ifdef AGX_DEBUG
    m_jacobianElements = JacobianElementArrayT(jacobianElements, IndexRange(offset, jacobianElements.size()));
#else
    m_jacobianElements = (JacobianElementT *)jacobianElements.ptr() + offset;
#endif
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE void JacobianAccessorT<JacobianElementT>::setup(const ComplementaryElementArrayT& jacobianElements, size_t offset) const
  {
    const_cast<JacobianAccessorT *>(this)->setup(const_cast<ComplementaryElementArrayT&>(jacobianElements), offset);
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE void JacobianAccessorT<JacobianElementT>::setup(const JacobianElementArrayT& jacobianElements, size_t offset) const
  {
    const_cast<JacobianAccessorT *>(this)->setup(const_cast<JacobianElementArrayT&>(jacobianElements), offset);
  }

#ifndef AGX_DEBUG
  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianAccessorT<JacobianElementT>::JacobianAccessorT(const JacobianElementT *jacobianElements, size_t offset) : m_jacobianElements((JacobianElementT *)jacobianElements + offset)
  {
  }
#endif

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianAccessorT<JacobianElementT>::JacobianAccessorT(const JacobianAccessorT& other) : m_jacobianElements(other.m_jacobianElements)
  {
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianAccessorT<JacobianElementT>& JacobianAccessorT<JacobianElementT>::operator= (const JacobianAccessorT& other)
  {
    m_jacobianElements = other.m_jacobianElements;
    return *this;
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianElementT& JacobianAccessorT<JacobianElementT>::get(size_t index)
  {
    return m_jacobianElements[index];
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const JacobianElementT& JacobianAccessorT<JacobianElementT>::get(size_t index) const
  {
    return m_jacobianElements[index];
  }


  template <typename JacobianElementT> template <typename AccessorT>
  AGX_FORCE_INLINE AccessorT JacobianAccessorT<JacobianElementT>::get(size_t index)
  {
    return AccessorT(m_jacobianElements, index);
  }

  template <typename JacobianElementT> template <typename AccessorT>
  AGX_FORCE_INLINE const AccessorT JacobianAccessorT<JacobianElementT>::get(size_t index) const
  {
    return AccessorT(m_jacobianElements, index);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////

  template <typename JacobianElementT>
  AGX_FORCE_INLINE TangentJacobianT<JacobianElementT>::TangentJacobianT() : BaseT()
  {
  }


  template <typename JacobianElementT>
  AGX_FORCE_INLINE TangentJacobianT<JacobianElementT>::TangentJacobianT(const ComplementaryElementArrayT& jacobianElements, size_t offset) : BaseT(jacobianElements, offset)
  {
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE TangentJacobianT<JacobianElementT>::TangentJacobianT(const JacobianElementArrayT& jacobianElements, size_t offset) : BaseT(jacobianElements, offset)
  {
  }

#ifndef AGX_DEBUG
  template <typename JacobianElementT>
  AGX_FORCE_INLINE TangentJacobianT<JacobianElementT>::TangentJacobianT(const JacobianElementT *jacobianElements, size_t offset) : BaseT(jacobianElements, offset)
  {
  }
#endif

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianElementT& TangentJacobianT<JacobianElementT>::u()
  {
    return this->get(0);
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const JacobianElementT& TangentJacobianT<JacobianElementT>::u() const
  {
    return this->get(0);
  }


  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianElementT& TangentJacobianT<JacobianElementT>::v()
  {
    return this->get(1);
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const JacobianElementT& TangentJacobianT<JacobianElementT>::v() const
  {
    return this->get(1);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////

  template <typename JacobianElementT>
  AGX_FORCE_INLINE RollingResistanceJacobianT<JacobianElementT>::RollingResistanceJacobianT() : BaseT()
  {
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE RollingResistanceJacobianT<JacobianElementT>::RollingResistanceJacobianT(const ComplementaryElementArrayT& jacobianElements, size_t offset) : BaseT(jacobianElements, offset)
  {
  }


  template <typename JacobianElementT>
  AGX_FORCE_INLINE RollingResistanceJacobianT<JacobianElementT>::RollingResistanceJacobianT(const JacobianElementArrayT& jacobianElements, size_t offset) : BaseT(jacobianElements, offset)
  {
  }

#ifndef AGX_DEBUG
  template <typename JacobianElementT>
  AGX_FORCE_INLINE RollingResistanceJacobianT<JacobianElementT>::RollingResistanceJacobianT(const JacobianElementT *jacobianElements, size_t offset) : BaseT(jacobianElements, offset)
  {
  }
#endif

  template <typename JacobianElementT>
  AGX_FORCE_INLINE typename RollingResistanceJacobianT<JacobianElementT>::TangentJacobian RollingResistanceJacobianT<JacobianElementT>::tangent()
  {
    return this->template get<TangentJacobian>(0);
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const typename RollingResistanceJacobianT<JacobianElementT>::TangentJacobian RollingResistanceJacobianT<JacobianElementT>::tangent() const
  {
    return this->template get<TangentJacobian>(0);
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianElementT& RollingResistanceJacobianT<JacobianElementT>::normal()
  {
    return this->get(2);
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const JacobianElementT& RollingResistanceJacobianT<JacobianElementT>::normal() const
  {
    return this->get(2);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////

  template <typename JacobianElementT>
  AGX_FORCE_INLINE BodyContactJacobianT<JacobianElementT>::BodyContactJacobianT() : BaseT()
  {
  }


  template <typename JacobianElementT>
  AGX_FORCE_INLINE BodyContactJacobianT<JacobianElementT>::BodyContactJacobianT(
    const ComplementaryElementArrayT& jacobianElements, size_t offset) : BaseT(jacobianElements, offset)
  {
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE BodyContactJacobianT<JacobianElementT>::BodyContactJacobianT(
    const JacobianElementArrayT& jacobianElements, size_t offset) : BaseT(jacobianElements, offset)
  {
  }

#ifndef AGX_DEBUG
  template <typename JacobianElementT>
  AGX_FORCE_INLINE BodyContactJacobianT<JacobianElementT>::BodyContactJacobianT(
    const JacobianElementT *jacobianElements, size_t offset) : BaseT(jacobianElements, offset)
  {
  }
#endif

  template <typename JacobianElementT>
  AGX_FORCE_INLINE JacobianElementT& BodyContactJacobianT<JacobianElementT>::normal()
  {
    return this->get(0);
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const JacobianElementT& BodyContactJacobianT<JacobianElementT>::normal() const
  {
    return this->get(0);
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE typename BodyContactJacobianT<JacobianElementT>::TangentJacobian BodyContactJacobianT<JacobianElementT>::tangent()
  {
    return this->template get<TangentJacobian>(1);
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const typename BodyContactJacobianT<JacobianElementT>::TangentJacobian BodyContactJacobianT<JacobianElementT>::tangent() const
  {
    return this->template get<TangentJacobian>(1);
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE typename BodyContactJacobianT<JacobianElementT>::RollingResistanceJacobian BodyContactJacobianT<JacobianElementT>::rollingResistance()
  {
    return this->template get<RollingResistanceJacobian>(3);
  }

  template <typename JacobianElementT>
  AGX_FORCE_INLINE const typename BodyContactJacobianT<JacobianElementT>::RollingResistanceJacobian BodyContactJacobianT<JacobianElementT>::rollingResistance() const
  {
    return this->template get<RollingResistanceJacobian>(3);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////

}


#endif /* AGX_BODYCONTACTJACOBIAN_H */
