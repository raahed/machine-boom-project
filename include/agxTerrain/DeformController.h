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

#include <agxTerrain/DeformerCollection.h>
#include <agx/Plane.h>
#include <agxTerrain/AggregateContactUtils.h>
#include <array>

namespace agxTerrain
{
  AGX_DECLARE_POINTER_TYPES(DeformController);

  class AGXTERRAIN_EXPORT DeformController : public agx::Referenced
  {
  public:
    DeformController(agx::Frame* parentFrame,
                     const agx::Line& cuttingEdge,
                     const agx::Line& topEdge,
                     const agx::Vec3& forwardVector,
                     const agxCollide::GeometryRefVector& voxelCollisionGeometries);

    void addNotification(agxSDK::Simulation* simulation, Terrain* terrain);

    void removeNotification(agxSDK::Simulation* simulation, Terrain* terrain);

    void onPreCollide(TerrainToolCollection* collection);

    void onPre(TerrainToolCollection* collection);

    void onPost(TerrainToolCollection* collection);

    DeformerCollection* getDeformerCollection(size_t i);

    ActiveZonePtrVector getActiveZones();

    agxCollide::GeometryPtrVector getActiveZoneGeometries();

    SoilParticleAggregatePtrVector getSoilParticleAggregates();

    agxCollide::GeometryContactPtrVector getGeometryContacts(agxSDK::Simulation *simulation) const;

    void onEnableChanged(bool enable);

    void initializeContactMaterials(TerrainToolCollection* collection);

    void resetAggregateContactHistories();

    void resetAggregateContactHistory(agx::UInt i);

    const agxTerrain::AggregateContactDepthModel& getAggregateDepthModel(agx::UInt i) const;

    /**
    Get the internal shovel <-> aggregate contact material of the specified deformer
    denoted by an index.
    \param i - index of the deformer which contact material should be accessed. [0-2]
    \return the contact material of the specified deformer.
    */
    agx::ContactMaterial* getAggregateShovelContactMaterial( agx::UInt i ) const;

    /**
    Get the internal aggregate <-> terrain contact material of the specified deformer
    denoted by an index.
    \param i - index of the deformer which contact material should be accessed. [0-2]
    \return the contact material of the specified deformer.
    */
    agx::ContactMaterial* getAggregateTerrainContactMaterial( agx::UInt i ) const;

  protected:
    virtual ~DeformController();

  private:
    enum DeformSide
    {
      BACK = 0,
      LEFT = 1,
      RIGHT = 2
    };

  private:

    void createEdges(const agx::Line& cuttingEdge,
                     const agx::Line& topEdge,
                     const agx::Vec3& forwardVector,
                     const agxCollide::GeometryRefVector& voxelCollisionGeometries);

    agx::Vec3Vector extractVertices(const agxCollide::GeometryRefVector& voxelCollisionGeometries);

    agx::Vec3 findEdgePoint( agx::Vec3Vector& vertices,
                             agx::Vec3 point,
                             agx::Plane& frontPlane,
                             agx::Plane& bottomPlane,
                             agx::Plane& sidePlane,
                             agx::Real tol );

    void deformSolidMass(TerrainToolCollection* collection);

    bool shouldDeformMass(TerrainToolCollection* collection);

    void createAggregateContactPoints(TerrainToolCollection* collection);

    void createShovelContacts(TerrainToolCollection* collection);

    void createTerrainContacts(TerrainToolCollection* collection);

    agx::Vec3Vector findShovelTerrainIntersectionPoints( TerrainToolCollection* collection,
                                                         DeformerCollection* deformer );

    agx::Vec3Vector findAggregateTerrainIntersectionPoints( const agx::Line& wedgeCuttingEdge,
                                                            const agx::Line& wedgeFailureEdge );

    agx::Vec3 calculateAggregateNormal( const agx::Line& wedgeCuttingEdge,
                                        const agx::Line& wedgeFailureEdge ) const;

    void pushSoilToSides(TerrainToolCollection* collection,
      const agx::Vec3& fromVoxelWorldPos,
      const agx::Real& elementSize,
      const agx::Vec3& topVectorNormalized,
      const agx::Vec3& deformerNormal,
      const agx::Real& timeStep,
      const agx::Real& voxelSpeed,
      const agx::Vec2i& fromTerrainIndex,
      const float& occupancyInVoxel,
      const float& dynamicMassCompaction,
      const agx::Vec3i& voxel,
      agx::Vec3* voxelSideVelocity);

    void DEBUG_DRAW();

  private:

    agx::Frame*                               m_parentFrame;
    std::array<DeformerCollectionRef, 3>      m_deformers;
    std::array<AggregateContactDepthModel, 3> m_depthModels;
    agxCollide::LocalGeometryContactVector    m_contacts;
    std::array<agx::ContactMaterialRef, 3>    m_aggregateTerrainContactMaterials;
    std::array<agx::ContactMaterialRef, 3>    m_aggregateShovelContactMaterials;
  };
}