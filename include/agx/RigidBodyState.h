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

#ifndef AGX_RIGIDBODYSTATE_H
#define AGX_RIGIDBODYSTATE_H

#include <agx/Math.h>
#include <agx/BitSet_small.h>
#include <agxData/Type.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agx
{
  class RigidBodyState : public BitSet_small<agx::UInt32>
  {
    public:
      RigidBodyState();
      agx::UInt32 motionControl() const;
      bool handleAsParticle() const;
      bool hasEffectiveMass() const;
      bool hasEffectiveInertia() const;
      bool enabled() const;
      bool removed() const;
      bool sleeping() const;
      bool mergedRootBody() const;
      bool powerline() const;
      bool isKinematicForParticles() const;

      void setKinematicForParticles(bool flag);

    private:
      friend class RigidBody;
      friend class DynamicsSystem;
      friend class MergedBody;
      friend class MassPropertiesUpdate;

      void setMotionControl(agx::UInt32 motionControl);
      void setHandleAsParticle(bool flag);
      void setHasEffectiveMass(bool flag);
      void setHasEffectiveInertia(bool flag);
      void setEnabled(bool flag);
      void setRemoved(bool flag);
      void setSleeping(bool flag);
      void setMergedRootBody(bool flag);
      void setPowerline(bool flag);

      enum Bit : agx::UInt32
      {
        PARTICLE_KINEMATIC_BIT = 29,
        HYDRAULICS_BIT = 28,
        MERGED_ROOT_BODY_BIT = 25,
        SLEEPING_BIT = 24,
        REMOVED_BIT = 20,
        HAS_EFFECTIVE_INERTIA_BIT = 18,
        HAS_EFFECTIVE_MASS_BIT = 17,
        HANDLE_AS_PARTICLE_BIT = 16,
        ENABLE_BIT = 12
      };
  };

  /* Implementation */
  AGX_FORCE_INLINE RigidBodyState::RigidBodyState() : BitSet_small<UInt32>()
  {
    this->set(ENABLE_BIT, true);
  }

  AGX_FORCE_INLINE agx::UInt32 RigidBodyState::motionControl() const
  {
    return (agx::UInt32)(m_state & 0x3);
  }

  AGX_FORCE_INLINE bool RigidBodyState::handleAsParticle() const
  {
    return this->test(HANDLE_AS_PARTICLE_BIT);
  }

  AGX_FORCE_INLINE bool RigidBodyState::hasEffectiveMass() const
  {
    return this->test(HAS_EFFECTIVE_MASS_BIT);
  }

  AGX_FORCE_INLINE bool RigidBodyState::hasEffectiveInertia() const
  {
    return this->test(HAS_EFFECTIVE_INERTIA_BIT);
  }

  AGX_FORCE_INLINE bool RigidBodyState::enabled() const
  {
    return this->test(ENABLE_BIT);
  }

  AGX_FORCE_INLINE bool RigidBodyState::removed() const
  {
    return this->test(REMOVED_BIT);
  }

  AGX_FORCE_INLINE bool RigidBodyState::sleeping() const
  {
    return this->test(SLEEPING_BIT);
  }

  AGX_FORCE_INLINE bool RigidBodyState::mergedRootBody() const
  {
    return this->test(MERGED_ROOT_BODY_BIT);
  }

  AGX_FORCE_INLINE bool RigidBodyState::powerline() const
  {
    return this->test(HYDRAULICS_BIT);
  }

  AGX_FORCE_INLINE bool RigidBodyState::isKinematicForParticles() const
  {
    return this->test(PARTICLE_KINEMATIC_BIT);
  }

  AGX_FORCE_INLINE void RigidBodyState::setMotionControl(agx::UInt32 motionControl)
  {
    m_state = ( m_state & ~0x3 ) | ((agx::UInt32)motionControl);
  }

  AGX_FORCE_INLINE void RigidBodyState::setEnabled(bool flag)
  {
    this->set(ENABLE_BIT, flag);
  }

  AGX_FORCE_INLINE void RigidBodyState::setHandleAsParticle(bool flag)
  {
    this->set(HANDLE_AS_PARTICLE_BIT, flag);
  }

  AGX_FORCE_INLINE void RigidBodyState::setHasEffectiveMass(bool flag)
  {
    this->set(HAS_EFFECTIVE_MASS_BIT, flag);
  }

  AGX_FORCE_INLINE void RigidBodyState::setHasEffectiveInertia(bool flag)
  {
    this->set(HAS_EFFECTIVE_INERTIA_BIT, flag);
  }

  AGX_FORCE_INLINE void RigidBodyState::setRemoved(bool flag)
  {
    this->set(REMOVED_BIT, flag);
  }

  AGX_FORCE_INLINE void RigidBodyState::setSleeping(bool flag)
  {
    this->set(SLEEPING_BIT, flag);
  }

  AGX_FORCE_INLINE void RigidBodyState::setMergedRootBody(bool flag)
  {
    this->set(MERGED_ROOT_BODY_BIT, flag);
  }

  AGX_FORCE_INLINE void RigidBodyState::setPowerline(bool flag)
  {
    this->set(HYDRAULICS_BIT, flag);
  }

  inline void RigidBodyState::setKinematicForParticles(bool flag)
  {
    this->set(PARTICLE_KINEMATIC_BIT, flag);
  }

  std::ostream& operator << ( std::ostream& output, const RigidBodyState& state );
}

AGX_TYPE_BINDING(agx::RigidBodyState, "RigidBodyState")

DOXYGEN_END_INTERNAL_BLOCK()

#endif /* AGX_RIGIDBODYSTATE_H */
