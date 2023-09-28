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

#ifndef AGXMODEL_SOIL_PARTICLE_MERGER_H
#define AGXMODEL_SOIL_PARTICLE_MERGER_H

#define USE_SOIL_PARTICLE_SIMULATION 0

#if defined(USE_SOIL_PARTICLE_SIMULATION) && USE_SOIL_PARTICLE_SIMULATION == 1

#include <agxModel/export.h>
#include <agxModel/TerrainParticles.h>

namespace agxModel
{

  typedef agx::HashTable <SoilParticle*,SoilParticlePtrVector> SoilParticlePtrSoilParticlePtrVectorTable;

  typedef agx::HashTable< agxCollide::Geometry*, agxCollide::GeometryRef > GeometryPtrGeometryRefTable;

  class AGXMODEL_EXPORT SoilParticleSimulation : public agxSDK::Simulation
  {
    public:

    SoilParticleSimulation();

    void addSeed(agx::RigidBody *seed);

    void addSeed(agxCollide::Geometry *seed);

    void notifyContact( agxCollide::GeometryContact* geometryContact );

    void prePrepare();

    void prepareForStepForward( agxSDK::Simulation* originalSimulation );

    void updatePositionsAfterOrdinarySolve();

    void postCleanUp( agxSDK::Simulation* originalSimulation );

    bool isActive( ) { return !m_geometriesToStaticGeometriesTable.empty(); }

    void updateTransforms( )
    {
      GeometryPtrGeometryRefTable::iterator it = m_geometriesToStaticGeometriesTable.begin();

      while ( it != m_geometriesToStaticGeometriesTable.end() )
      {
        updateTransform((*it).first);
        ++it;
      }
    }

    protected:
      friend class SoilParticle;

    private:

    void addChild( SoilParticle* parent, SoilParticle* child );

    void mergeParentToParent( SoilParticle* parent1, SoilParticle* parent2, SoilParticlePtrVector& parentVector );

    void mergeToParent( SoilParticle* parent, SoilParticle* child );

    void updateTransform( agxCollide::Geometry* geometry );

    void restoreParticles();

    agx::RigidBody* replaceParticle( SoilParticle* particle );

    AGX_FORCE_INLINE void updateTransform( GeometryPtrGeometryRefTable::iterator it )
    {
      if ( it == m_geometriesToStaticGeometriesTable.end() )
        return;

      (*it).second->setTransform( (*it).first->getTransform() );
    }

    AGX_FORCE_INLINE void insertInTable( SoilParticlePtrVector& children, SoilParticle* child )
    {
#if TERRAIN_USE_PARTICLE_SYSTEM == 0
      agxAssert(!children.contains(child));
      children.push_back(child);
#endif
    }

    agx::Vector< agx::MaterialRef > m_materials;
    GeometryPtrGeometryRefTable m_geometriesToStaticGeometriesTable;
    agxCollide::GeometryRefVector m_geometryReferences;
    SoilParticlePtrSoilParticlePtrVectorTable m_parents;
    SoilParticlePtrSoilParticlePtrVectorTable m_seedParents;
    agx::RigidBodyRefVector m_createdBodies;
    agx::RigidBodyRefVector m_particlePool;
    agx::RigidBodyRefVector m_usedParticles;
    agx::RigidBodyRefVector m_disabledParticles;
    agxCollide::GeometryPtrVector m_particleGeometries;
  };

  typedef agx::ref_ptr<SoilParticleSimulation> SoilParticleSimulationRef;

}

#endif //USE_SOIL_PARTICLE_SIMULATION

#endif
