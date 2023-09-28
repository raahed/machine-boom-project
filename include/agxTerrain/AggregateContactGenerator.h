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

#include <agxTerrain/PrimaryActiveZone.h>
#include <agx/Plane.h>
#include <agxTerrain/SoilParticleAggregate.h>
#include <agxTerrain/AggregateContactUtils.h>

/// \cond INTERNAL_DOCUMENTATION
namespace detail
{
  struct ContactPatchData;
  struct ContactSpanData;
}
/// \endcond

namespace agxTerrain
{
  class TerrainToolCollection;

  typedef agx::HashVector<agxCollide::Geometry*, agxCollide::LocalGeometryContact> GeometryLocalContactTable;
  typedef agx::HashSet<agxCollide::Geometry*> GeometryHashSet;

  AGX_DECLARE_POINTER_TYPES(AggregateContactGenerator);

  class AGXTERRAIN_EXPORT AggregateContactGenerator : public agx::Referenced
  {
  public:
    AggregateContactGenerator();

    void onPre(TerrainToolCollection* collection);

    /**
    \return the geometry contacts created between the shovel geometries and this aggregate
    */
    agxCollide::GeometryContactPtrVector getGeometryContacts(agxSDK::Simulation *simulation) const;

    /**
    Initialize contact materials used in the aggregate contacts
    */
    void initializeContactMaterials(TerrainToolCollection* collection);

    /**
    Resets contact model history in the contact generator
    */
    void resetAggregateContactHistory();

    /**
    Get the aggregate <-> terrain contact depth model
    */
    const AggregateContactDepthModel& getAggregateDepthModel() const;

    /**
    Get the internal shovel <-> aggregate contact material
    */
    agx::ContactMaterial* getAggregateShovelContactMaterial() const;

    /**
    Get the internal aggregate <-> terrain contact material
    */
    agx::ContactMaterial* getAggregateTerrainContactMaterial() const;

  protected:
    virtual ~AggregateContactGenerator();

  private:

    void createGeometryContacts(TerrainToolCollection* collection);

    void createTerrainContacts(TerrainToolCollection* collection,
                               agxCollide::LocalGeometryContactVector& localContacts);

    void createShovelContacts(TerrainToolCollection* collection,
                              agxCollide::LocalGeometryContactVector& localContacts);

    agx::Real getContactPointAngle(const agx::Vec3& midPoint,
                                   const agx::Vec3& zeroPoint,
                                   const agx::Vec3& contactPoint,
                                   const agx::Plane& polarPlane,
                                   const agx::Plane& cutTopPlane) const;

    GeometryHashSet getShovelGeometries( TerrainToolCollection* collection) const;

    void getShovelContactsAndContactRange( TerrainToolCollection* collection,
                                           GeometryHashSet& geometrySet,
                                           agxCollide::ParticleGeometryContactVector& contacts,
                                           agx::RangeReal& range) const;

    agxCollide::LocalParticleGeometryContactVector calculateShovelContactPoints( TerrainToolCollection* collection,
                                                                                 const agxCollide::ParticleGeometryContactVector& shovelContacts,
                                                                                 const agx::RangeReal& range ) const;

    void updateMeanContactNormals( detail::ContactPatchData& contactData,
                                   const agx::Physics::ParticleGeometryContactPtr& contact,
                                   const agx::Plane& cuttingTopPlane,
                                   agx::Real contactAngle ) const;

    void updateContactSpan( detail::ContactPatchData& contactPatchData,
                            detail::ContactSpanData& contactSpanData,
                            const agx::Physics::ParticleGeometryContactPtr& contact,
                            const agx::Plane& middlePlane,
                            agx::Real contactAngle ) const;

    agxCollide::LocalParticleGeometryContactVector finalizeShovelContacts( TerrainToolCollection* collection,
                                                                           detail::ContactPatchData& contactPatchData,
                                                                           const agx::Vec3f& separationNormal,
                                                                           const agx::Vec3& cuttingVector,
                                                                           const detail::ContactSpanData& contactSpanData ) const;

    agx::Real calculateContactArea( const TerrainToolCollection* collection,
                                    const agxCollide::LocalParticleGeometryContactVector& contacts ) const;

  private:
    using ParticleIdSet = agx::HashSet<agx::UInt32>;

    ParticleIdSet                          m_particleIds;
    agxCollide::LocalGeometryContactVector m_localGeometryContacts;
    agx::ContactMaterialRef                m_aggregateTerrainContactMaterial;
    agx::ContactMaterialRef                m_aggregateShovelContactMaterial;
    AggregateContactDepthModel             m_depthModel;
  };
}
