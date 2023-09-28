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

#include <agx/config/AGX_USE_AGXTERRAIN.h>

#include <agx/agx.h>
#include <agx/Line.h>
#include <agx/RigidBody.h>
#include <agx/Physics/GranularBodySystem.h>

#include <agxCollide/Geometry.h>
#include <agxCollide/ShapePrimitives.h>
#include <agxCollide/Convex.h>

#include <agxRender/RenderSingleton.h>

#include <agxSDK/Simulation.h>
#include <agxSDK/StepEventListener.h>

#include <agxTerrain/export.h>
#include <agxTerrain/Grid.h>
#include <agxTerrain/SoilWedge.h>

namespace agxTerrain
{
  class Terrain;
  class TerrainToolCollection;

  typedef std::pair<agxCollide::MeshRef, agx::AffineMatrix4x4> MeshTransformPair;
  typedef agx::Vector<MeshTransformPair> MeshTransformPairVector;

  using WedgePlaneVector = agx::Vector<std::array<agx::Plane, 3>>;

  AGX_DECLARE_POINTER_TYPES(ActiveZone);
  AGX_DECLARE_VECTOR_TYPES(ActiveZone);

  class AGXTERRAIN_EXPORT ActiveZone : public agx::Referenced
  {
  public:

    enum class Type
    {
      PRIMARY_ACTIVE_ZONE,
      DEFORMER_ACTIVE_ZONE
    };

    ActiveZone(agx::Frame* parentFrame);

    virtual void onPreCollide(Terrain* terrain, const agx::Line& cuttingEdge, const agx::Line& topEdge, const agx::Vec3& forwardVector) = 0;

    virtual void onPre(Terrain* terrain, const agx::Line& cuttingEdge, const agx::Line& topEdge, const agx::Vec3& forwardVector) = 0;

    virtual void onPost(Terrain* terrain, const agx::Line& cuttingEdge, const agx::Line& topEdge, const agx::Vec3& forwardVector) = 0;

    virtual ActiveZone::Type getType() const = 0;

    agx::Line getWedgeFailureEdge() const;

    agx::Line getWedgeCuttingEdge() const;

    agx::Line getWedgeTopEdge() const;

    agx::Bool overlapsVoxel(const agx::Vec3i& index) const;

    agx::Bound3i getVoxelOverlapBound() const;

    agxCollide::Geometry* getGeometry();

    const agxCollide::Geometry* getGeometry() const;

    const agx::Vector<agxCollide::ConvexRef>&  getWedgeShapes() const;

    WedgePlaneVector getSoilWedgePlanes() const;

    bool isSoilWedgeEnabled(agx::UInt index) const;

    bool isAnySoilWedgeEnabled() const;

    /**
    \return true if any soil wedge convex creation failed for this Active Zone
    */
    bool soilWedgeCreationFailed() const;

    agx::Plane getSplittingPlane() const;

    agx::Real getTotalSoilWedgeVolume(const agx::Vec3& upVector) const;

    agx::Real getFailureAngle() const;

    agx::Real getTerrainInclination() const;

    agx::Real getToolInclination() const;

    agx::Real getFailureAngleToTerrainPlane(const agx::Vec3& upVector) const;

    /**
    \internal

    Called when Shovel is disabled. Performs temporary cleanup.
    */
    void onEnableChange( bool enable );

    void removeWedgeShapes();

    agx::String getActiveZoneTypeName() const;

  protected:

    virtual ~ActiveZone();

    void createSoilWedges(const Terrain* terrain,
                          const agx::Line& cuttingEdge,
                          const agx::Line& topEdge,
                          const agx::Vec3& forwardVector,
                          const agx::Vec3& flatForwardVector);

    agx::Vec3 projectVectorToTerrainPlane(const agx::Vec3& vec, const agx::Vec3& upVector) const;

  protected:
    // Parent frame following shovel body model frame.
    agx::Frame* m_parentFrame;
    agx::Bound3i m_voxelOverlapBound;

    // Soil wedge meshes
    agx::UInt m_soilWedgesSize;

    agx::Vector<SoilWedge> m_soilWedges;

    agx::Vector<agxCollide::ConvexRef> m_soilWedgeMeshses;

    agx::Plane m_splittingPlane;

  private:
    void initializeGeometry();

    SoilWedge createSoilWedge(const Terrain* terrain,
                              const agx::UInt wedgeCount,
                              const agx::Line& cuttingEdge,
                              const agx::Line& topEdge,
                              const agx::Vec3& forwardVector,
                              const agx::Vec3& flatForwardVector);

    bool calculateFailurePlaneIntersection(const Terrain* terrain,
                                           SoilWedge& soilWedge,
                                           const agx::Line& cuttingEdge,
                                           const agx::Line& topEdge,
                                           const agx::Vec3& forwardVector,
                                           const agx::Vec3& flatForwardVector);

    void createSoilWedgeConvex(const SoilWedge& soilWedge);

    void createSoilWedgePlanes(SoilWedge& soilWedge);

    bool isSoilWedgeSubmerged( const Terrain* terrain, const SoilWedge& soilWedge );

    bool isPointSubmerged( const Terrain* terrain, const agx::Vec3& point );

    bool doTerrainToolIntersectionTest(const Terrain* terrain,
                                       const agx::AffineMatrix4x4& worldToHeightField,
                                       const SoilWedge& soilWedge,
                                       const agx::Vec3& upVector,
                                       const agx::Vec3& flatForwardVector,
                                       const agx::Vec3& forwardVector,
                                       agx::Real forwardAngle,
                                       agx::Real maxWedgeLength,
                                       agx::Vec3& intersectionPoint) const;

    bool handleMissedIntersection( const Terrain* terrain,
                                   const agx::Vec3& upVector,
                                   const agx::Vec3& flatForwardVector,
                                   const agx::Line& flatForwardLine,
                                   agx::Vec3iVector::iterator& voxelIterator,
                                   const agx::Vec3iVector& voxels,
                                   agx::Vec3& intersectionPoint ) const;

    bool isIntersectionAboveTerrainMesh( const Terrain* terrain,
                                         const SoilWedge& soilWedge,
                                         const agx::AffineMatrix4x4& worldToHeightField,
                                         const agx::Vec3& upVector ) const;

    void updateSoilWedgeInclinations(const Terrain* terrain,
                                     SoilWedge& soilWedge,
                                     const agx::Vec2i terrainIndex,
                                     const agx::Vec3i voxelIndex,
                                     const agx::Real cuttingEdgeHeight,
                                     const agx::Vec3& upVector,
                                     const agx::Vec3& forwardVector,
                                     const agx::Vec3& flatForwardVector) const;

  private:
    // Active zone geometry and inner-shape
    agxCollide::GeometryRef m_geometry;

    bool m_convexCreationFailed;
  };
}
