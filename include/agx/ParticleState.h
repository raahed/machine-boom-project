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

#ifndef AGX_PARTICLESTATE_H
#define AGX_PARTICLESTATE_H

#include <agx/Math.h>
#include <agx/BitSet_small.h>
#include <agxData/Type.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agx
{
  typedef agx::UInt32 ParticleStateDataType;

  class ParticleState : public BitSet_small<ParticleStateDataType>
  {
  public:
    ParticleState();
    agx::UInt32 motionControl() const;
    bool enabled() const;
    bool removed() const;
    bool enableMerge() const;
    bool enableRotation() const;
    bool particleClipped() const;
    bool particleFiltered() const;
    bool isTerrainSoilParticle() const;
    bool enableRendering() const;
    bool shouldRenderParticle() const;

    void setMotionControl(agx::UInt32 motionControl);
    void setEnabled(bool flag);
    void setRemoved(bool flag);
    void setEnableMerge(bool flag);
    void setEnableRotation(bool flag);
    void setParticleClipped(bool flag);
    void setParticleFiltered(bool flag);
    void setIsTerrainSoilParticle(bool flag);
    void setEnableRendering(bool flag);

  private:
    friend class RigidBody;
    friend class DynamicsSystem;
    friend class MergedBody;
    friend class MassPropertiesUpdate;

    static const UInt16 ENABLE_RENDERING = 12;
    static const UInt16 TERRAIN_SOIL_PARTICLE = 11;
    static const UInt16 RENDERING_PARTICLE_FILTERED = 10;
    static const UInt16 RENDERING_PARTICLE_CLIPPED = 9;
    static const UInt16 ENABLE_NO_ROTATIONS = 8;
    static const UInt16 ENABLE_MERGE_BIT    = 7;
    static const UInt16 REMOVED_BIT         = 6;
    static const UInt16 ENABLE_BIT          = 5;
  };

  /* Implementation */
  AGX_FORCE_INLINE ParticleState::ParticleState() : BitSet_small<ParticleStateDataType>()
  {
    this->set(ENABLE_BIT,       true);
    this->set(REMOVED_BIT,      false);
    this->set(ENABLE_MERGE_BIT, true);
    this->set(RENDERING_PARTICLE_CLIPPED, false);
    this->set(RENDERING_PARTICLE_FILTERED, false );
    this->set(TERRAIN_SOIL_PARTICLE, false);
    this->set(ENABLE_RENDERING, true);
    this->setMotionControl(3); // This should be the DYNAMIC enum number in GranularBodySystem
  }

  AGX_FORCE_INLINE agx::UInt32 ParticleState::motionControl() const
  {
    return (agx::UInt32)(m_state & 0x3);
  }

  AGX_FORCE_INLINE bool ParticleState::enabled() const
  {
    return this->test(ENABLE_BIT);
  }

  AGX_FORCE_INLINE bool ParticleState::particleClipped() const
  {
    return this->test(RENDERING_PARTICLE_CLIPPED);
  }

  AGX_FORCE_INLINE bool ParticleState::removed() const
  {
    return this->test(REMOVED_BIT);
  }

  AGX_FORCE_INLINE bool ParticleState::enableMerge() const
  {
    return this->test(ENABLE_MERGE_BIT);
  }

  AGX_FORCE_INLINE bool ParticleState::enableRotation() const
  {
    return !this->test(ENABLE_NO_ROTATIONS);
  }

  AGX_FORCE_INLINE bool agx::ParticleState::particleFiltered() const
  {
    return this->test(RENDERING_PARTICLE_FILTERED);
  }

  AGX_FORCE_INLINE bool agx::ParticleState::isTerrainSoilParticle() const
  {
    return this->test( TERRAIN_SOIL_PARTICLE );
  }

  AGX_FORCE_INLINE bool ParticleState::enableRendering() const
  {
    return this->test( ENABLE_RENDERING );
  }

  AGX_FORCE_INLINE void ParticleState::setMotionControl(agx::UInt32 motionControl)
  {
    const ParticleStateDataType mask = (ParticleStateDataType) 0x3;
    m_state = (m_state & ~mask) | ((ParticleStateDataType)motionControl);
  }

  AGX_FORCE_INLINE void ParticleState::setEnabled(bool flag)
  {
    this->set(ENABLE_BIT, flag);
  }

  AGX_FORCE_INLINE void ParticleState::setParticleClipped(bool flag)
  {
    this->set(RENDERING_PARTICLE_CLIPPED, flag);
  }

  AGX_FORCE_INLINE void agx::ParticleState::setParticleFiltered(bool flag)
  {
    this->set(RENDERING_PARTICLE_FILTERED, flag);
  }

  AGX_FORCE_INLINE void ParticleState::setEnableRotation(bool flag)
  {
    this->set(ENABLE_NO_ROTATIONS, !flag);
  }

  AGX_FORCE_INLINE void ParticleState::setRemoved(bool flag)
  {
    this->set(REMOVED_BIT, flag);
  }

  AGX_FORCE_INLINE void ParticleState::setEnableMerge(bool flag)
  {
    this->set(ENABLE_MERGE_BIT, flag);
  }

  AGX_FORCE_INLINE void agx::ParticleState::setIsTerrainSoilParticle( bool flag )
  {
    this->set( TERRAIN_SOIL_PARTICLE, flag );
  }

  AGX_FORCE_INLINE void ParticleState::setEnableRendering(bool flag)
  {
    this->set(ENABLE_RENDERING, flag);
  }

  AGX_FORCE_INLINE bool ParticleState::shouldRenderParticle() const
  {
    return enableRendering() && !( particleClipped() || particleFiltered() );
  }

#ifndef SWIG
  std::ostream& operator << (std::ostream& output, const ParticleState& state);
#endif // !SWIG
}

AGX_TYPE_BINDING(agx::ParticleState, "ParticleState")

DOXYGEN_END_INTERNAL_BLOCK()

#endif /* AGX_RIGIDBODYSTATE_H */
