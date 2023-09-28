/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#include <agxTerrain/Grid.h>

namespace agxTerrain
{

  /**
  Class implementing the BasicGrid interface.
  */
  class AGXTERRAIN_EXPORT ColumnHeightGrid : public agxTerrain::Grid
  {

  public:

    using GridData = agx::Vector<agx::Vector<std::pair<int, float>>>;

    ColumnHeightGrid(const float defaultValue, const size_t resolutionX, const size_t resolutionY, const agx::Int depth, agx::Real voxelSize);
    ~ColumnHeightGrid();

    virtual agx::Vec3 getElementSize() const override;

    virtual size_t getMemoryUsage() const override;

    virtual bool empty() override;

    virtual void clearGrid() override;

    virtual void setFloatValue(const GridCoord& ijk, float value) override;

    virtual float getFloatValue(const GridCoord& ijk) const override;

    virtual bool isActiveVoxel(const GridCoord& /*ijk*/) const override;

    virtual size_t getNumActiveVoxels() const override;

    virtual agx::Real getTotal() const override;

    virtual std::tuple<float, float> getTotalAndLargest() const override;

    virtual GridCoordVector getActiveVoxelIndices() const override;

    virtual void sizeHint(size_t /*hint*/) override;

    virtual bool hasIterateSupport() const override;

    virtual void iterateVoxels(GridFunction func) const override;

    virtual agx::HashSet<agx::Vec3i> findGeometryVoxelIntersections(const agx::AffineMatrix4x4& gridTransform,
      const agxCollide::Geometry* geometry,
      agx::Real voxelSize,
      bool /*onlyOccupiedVoxels*/ = true) const override;

    agxCollide::BoundingAABB getBoundingBox() const override;

    virtual void setValuesBelowHeightField(
      const agxCollide::HeightField* heightField,
      agx::Real voxelSize,
      int lowestAllowedVoxelIndex,
      BasicGrid* compactionGrid) override;


    virtual agx::HashSet<agx::Vec3i> findConvexVoxelIntersections(const agx::AffineMatrix4x4& gridTransform,
      const agxCollide::Convex* convex,
      agx::Real voxelSize) override;


    virtual bool findMeshVoxelIntersections(const agx::AffineMatrix4x4& gridTransform,
      const agxCollide::Mesh* mesh,
      agx::Real voxelSize,
      agx::Vec3iVector& result) override;


    virtual agx::HashSet<agx::Vec3i> findVoxelsAboveTriangles(const agx::AffineMatrix4x4& gridTransform,
      const TriangleVector& triangles,
      agx::Real voxelSize,
      agx::Real margin) override;


    virtual ActiveZoneIntersectionData removeVolumeInActiveZone(const agx::AffineMatrix4x4& gridTransformation,
      const ActiveZoneShapes& activeZoneShapes,
      const MeshIntersectionTestData& innerShape,
      const agx::Plane& innerShapeBoundary,
      const agx::Bound3& activeZoneGeometryBound,
      const agx::Vec3iVector voxelsInPrimaryActiveZoneLastTimeStep,
      std::function<float(const agx::Vec3i&)> getCompression,
      std::function<bool(const agx::Vec2i&)> canRemoveIndex,
      agx::Real voxelSize,
      agx::Real minOccupancyRemoval,
      agx::Real minAllowedHeight,
      bool useParticleFreeDeformers) override;

    virtual ActiveZoneIntersectionData removeVolumeInShape( const agx::AffineMatrix4x4& gridTransformation,
                                                            const ShapeIntersectionTestData& shape,
                                                            std::function<float( const agx::Vec3i& )> getCompression,
                                                            std::function<bool( const agx::Vec2i& )> canRemoveIndex,
                                                            std::function<bool( const agx::Vec3&, const agx::Vec3i& )> isWithinBoundry,
                                                            agx::Real voxelSize,
                                                            agx::Real minOccupancyRemoval,
                                                            agx::Real minAllowedHeight ) override;

    virtual agx::Vec3iVector findVoxelsInLineIntersection(const agx::Line& line) const override;

    virtual void findVoxelSurfaceFromIndex(const agx::Vec3i& index, agx::Vec3i& result) override;

    agx::Vec2i getResolution();

    int getSurfaceHeightIndex(const agx::Vec2i terrainIndex);

    std::pair<int, float> getVoxelZAndOccupancyFromLocalHeight(agx::Real height, agx::Real voxelSize);

    void setGridColumnOccupancy(const agx::Vec2i terrainIndex, int voxelZ, float occupancy);

    void setOccupancyInSurfaceVoxel(const agx::Vec2i terrainIndex, float occupancy);

    /**
    \return the maximum allowable voxel index depth that the ColumnHeightGrid has.
    */
    int getDepth() const;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxTerrain::ColumnHeightGrid);

  protected:

    agx::Vec3 getLocalPositionFromVoxelIndex(const agx::Vec3i& voxelIndex, agx::Real voxelSize) const;

    agx::Vec3i getVoxelIndexFromLocalPosition(agx::Vec3 position, agx::Real voxelSize) const;


  protected:
    ColumnHeightGrid();

    void allocateMassArray();

  private:

    GridData    m_solidMassArray;
    float       m_default;
    size_t      m_resolutionX;
    size_t      m_resolutionY;
    agx::Int    m_depth;
    agx::Real   m_voxelSize;
    };
  }

