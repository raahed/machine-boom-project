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

#include <agxTerrain/export.h>

#include <agx/Physics/GranularBodySystem.h>

namespace agxTerrain
{
  class Terrain;

  typedef agx::Physics::GranularBodyPtrVector SoilParticleVector;
  typedef agx::Physics::GranularBodyPtrArray SoilParticleArray;

  AGX_DECLARE_POINTER_TYPES(SoilSimulationInterface);

  /**
  Interface class for accessing data and functions for the internal soil particle simulation used in agxTerrain.
  \note Observe that editing the internal granular data structure might break the terrain algorithm.
  */
  class AGXTERRAIN_EXPORT SoilSimulationInterface : public agx::Referenced
  {
    public:
      /**
      Construct given terrain instance.
      \param terrain - terrain instance
      */
      SoilSimulationInterface(Terrain* terrain);

      /**
      \internal

      Default constructor used in serialization.
      */
      SoilSimulationInterface();

      /**
      Create a soil particle with nominal radius and material in the internal soil simulation.
      \param position - initial position of the created soil particle in world space.
      \param velocity - initial velocity of the created soil particle in world space.
      \return the created soil particle.
      */
      agx::Physics::GranularBodyPtr createSoilParticle(const agx::Vec3& position = agx::Vec3(), const agx::Vec3& velocity = agx::Vec3());

      /**
      Create a soil particle with specified radius and material in the internal soil simulation.
      \param radius - radius of the created soil particle. If negative, no particle will be created.
      \param position - initial position of the created soil particle in world space.
      \param velocity - initial velocity of the created soil particle in world space.
      \return the created soil particle.
      */
      agx::Physics::GranularBodyPtr createSoilParticle(agx::Real radius, const agx::Vec3& position = agx::Vec3(), const agx::Vec3& velocity = agx::Vec3());

      /**
      Tag a soil particle for removal at the last stages of the current time step
      */
      void removeSoilParticle( agx::Physics::GranularBodyPtr ptr );

      /**
      \return true if particle is valid, i.e., not removed
      */
      bool isValid( agx::Physics::GranularBodyPtr ptr ) const;

#if (!defined SWIG ) || ( defined SWIGCSHARP || defined SWIGJAVA)
      /**
      Get the soil particles active in the simulation.
      \note Observe that editing the internal granular data structure might break the terrain algorithm.
      */
      SoilParticleArray getSoilParticles() const;
#endif

      /**
      \return soil particle specific volume (m3) active in the simulation.
      */
      agx::Real calculateSoilParticleSpecificVolume() const;

      /**
      \return soil particle bulk volume (m3) active in the simulation.
      \note - the bulk volume is the specific volume of all particles divided with the
              estimated particle packing fraction (0.67)
      */
      agx::Real calculateSoilParticleBulkVolume() const;

      /**
      Add the specified \p id to a vector of group id, this will make the created soil particles part
      of the group \p id. By default, the created soil particles are not part of any group.

      This can be used to partition the simulation where some groups cannot collide with other groups.
      Which groups that can collide or not is determined by Space and can be set using the method
      agxCollide::Space::setEnablePair(id1,id2);
      */
      void addCollisionGroup(agx::UInt32 id);
      void addCollisionGroup(const agx::Name& name);

      /**
      Remove a group id from the vector of group ids.
      \param id - The group id to be removed
      */
      void removeCollisionGroup(agx::UInt32 id);
      void removeCollisionGroup(const agx::Name& name);

      /**
      Get the number of soil particles active in the soil simulation.
      */
      size_t getNumSoilParticles() const;

      /**
      Get the complete bound of the soil particle system active in the simulation
      */
      agx::Bound3 getSoilParticleBound() const;

      /**
      Returns the particle-geometry contacts in the internal soil simulation.
      */
      const agxCollide::ParticleGeometryContactVector& getParticleGeometryContacts() const;

      /**
      Returns the particle-particle contacts in the internal soil simulation.
      */
      const agxCollide::ParticlePairContactVector& getParticleParticleContacts() const;

      /**
      Returns the internal GranularBodySystem used to simulate the soil particles.
      \note Observe that editing or otherwise change the internal particle data might break the terrain algorithm.
      */
      agx::Physics::GranularBodySystem* getGranularBodySystem() const;

      /**
      \return true if particle geometry contact is valid
      */
      bool isParticleGeometryContactValid(agx::Physics::ParticleGeometryContactPtr contact);

      /**
      \return true if particle-particle contact is valid
      */
      bool isParticlePairContactValid(agx::Physics::ParticlePairContactPtr contact);

      /**
      \return true if the Geometry ptr is valid
      */
      bool isGeometryPtrValid( agx::Physics::GeometryPtr geometry ) const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      void onAddNotification();
      void onRemoveNotification();

      AGXTERRAIN_STORE_RESTORE_INTERFACE;
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      ~SoilSimulationInterface();

      agx::Physics::GranularBodyPtr _createSoilParticle( agx::Real radius, const agx::Vec3& position, const agx::Vec3& velocity, agx::Material* material );
      void initGranularBodySystem();
      bool isLastTerrain();
      bool isTerrainPagerActive();

    private:
      Terrain* m_terrain;
      agx::Physics::GranularBodySystemRef m_soilParticleSystem;
  };

  AGX_FORCE_INLINE bool SoilSimulationInterface::isValid(agx::Physics::GranularBodyPtr ptr) const
  {
    return m_soilParticleSystem != nullptr ? m_soilParticleSystem->isValid(ptr) : false;
  }
}
