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

#include <agxTerrain/ActiveZone.h>

namespace agxTerrain
{
  class Shovel;
  class Terrain;

  AGX_DECLARE_POINTER_TYPES( PrimaryActiveZone );

  class AGXTERRAIN_EXPORT PrimaryActiveZone : public ActiveZone
  {
  public:
    PrimaryActiveZone( agx::Frame* parentFrame, Shovel* shovel, const agxCollide::GeometryRefVector& voxelCollisionGeometries );

    ActiveZone::Type getType() const;

    const agxCollide::Convex* getInnerShape() const;

    const agx::Vector<agx::Plane>& getInnerPlanes() const;

    void onPreCollide( Terrain* terrain, const agx::Line& cuttingEdge, const agx::Line& topEdge, const agx::Vec3& forwardVector );

    void onPre( Terrain* terrain, const agx::Line& cuttingEdge, const agx::Line& topEdge, const agx::Vec3& forwardVector );

    void onPost( Terrain* terrain, const agx::Line& cuttingEdge, const agx::Line& topEdge, const agx::Vec3& forwardVector );

    agx::Real getSecondarySeparationAngle() const;

    agx::Vec3 getSecondaryCuttingDirection() const;

    void updateInnerShape( const agxCollide::GeometryRefVector& voxelCollisionGeometries );

  protected:
    virtual ~PrimaryActiveZone();

  private:

    void createInnerShape( const Shovel* shovel, const agxCollide::GeometryRefVector& voxelCollisionGeometries );

    bool finalizeInnerShapeFromVertices( const agx::Vec3Vector& vertices );

    void calculateSecondarySeparationProperties( Shovel* shovel );

    void calculateProjectedVectors( Terrain* terrain,
      const agx::Line& cuttingEdge,
      const agx::Line& topEdge,
      const agx::Vec3& forwardVector,
      agx::Line& cuttingEdgeWorld,
      agx::Line& topEdgeWorld );

    bool findPlaneMeshCenterPoint(
      agx::Vec3 planePoint,
      agx::Vec3 planeNormal,
      agx::Real rayLength,
      const MeshTransformPairVector& meshes,
      agx::Vec3& result );

    bool isTriangleFirstInRay(
      const agx::Vec3& bottom,
      const agx::Vec3& top,
      const agxCollide::Mesh::Triangle& triangle,
      const MeshTransformPairVector& meshes );

    agx::Vec3Vector performSecondIntersectionTest(
      agx::Vec3Vector& vertices,
      const MeshTransformPairVector& meshes,
      const agx::Vec3& center);

  private:
    // Active zone geometry and inner-shape
    agxCollide::ConvexRef m_innerShape;

    // Planes used to sort particles
    agx::Vector<agx::Plane> m_innerPlanes;

    // Pointer to shovel
    Shovel* m_shovel;

    // Secondary separation
    agx::Real m_secondarySeparationAngle;
    agx::Vec3 m_secondaryForwardVector;

    // Projected vectors
    agx::Vec3 m_projectedForwardVector;
    agx::Vec3 m_projectedSecondaryForwardVector;
    agx::Vec3 m_flatForwardVector;
  };
}
