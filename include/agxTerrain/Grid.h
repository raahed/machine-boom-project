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

#include <tuple>

#include <agxTerrain/export.h>
#include <agxCollide/Mesh.h>
#include <agx/IndexLambdaKernel.h>

#include <mutex>

#include <agxTerrain/export.h>

#include <agx/AffineMatrix4x4.h>
#include <agx/Plane.h>
#include <agx/Line.h>

#include <agxCollide/HeightField.h>
#include <agxCollide/BoundingAABB.h>
#include <agxCollide/Convex.h>

#include <agxStream/InputArchive.h>
#include <agxStream/OutputArchive.h>

namespace agxTerrain
{
  // Container struct for containing data regarding removed volume
  struct ActiveZoneIntersectionData
  {
    agx::Bound3i indicesInActiveZone;
    agx::Vec3iVector removedIndices;
    agx::Real32Vector removedOccupancy;
    agx::Real removedVolume;
  };

  typedef std::tuple<const agxCollide::Mesh*, agx::AffineMatrix4x4, agxCollide::BoundingAABB> MeshIntersectionTestData;
  typedef std::tuple<const agxCollide::Shape*, agx::AffineMatrix4x4, agxCollide::BoundingAABB> ShapeIntersectionTestData;
  typedef agx::Vector<MeshIntersectionTestData> ActiveZoneShapes;
  typedef agx::Vector<std::array<agx::Vec3, 3>> TriangleVector;


  AGX_DECLARE_POINTER_TYPES(BasicGrid);
  AGX_DECLARE_POINTER_TYPES(Grid);

  /**
  Pure virtual interface class for the data storage grid classes we have.

  Makes it possible to have multiple implementations for the grid data storage.

  Provides a number of batch operations that are expected to be faster to
  implement with knowledge about the underlying data structure instead of through
  per-cell element access.
  */


  class AGXTERRAIN_EXPORT BasicGrid : public agx::Referenced, public agxStream::Serializable
  {
    public:

      using GridCoord = agx::Vec3i;

      using GridCoordVector = agx::VectorPOD<GridCoord>;

      using GridFunction = std::function<void(GridCoord, float)>;


    public:
      /**
      Get the element size used for the voxel grid
      */
      virtual agx::Vec3 getElementSize() const = 0;

      /**
      Get the memory usage
      */
      virtual size_t getMemoryUsage() const = 0;

      /**
      Test if the trid is empty
      */
      virtual bool empty() = 0;

      /**
      Clear the grid
      */
      virtual void clearGrid() = 0;

      /**
      Set value for voxel ijk.
      */
      virtual void setFloatValue(const GridCoord& ijk, float value) = 0;

      /**
      Return value for voxel ijk.
      */
      virtual float getFloatValue(const GridCoord& ijk) const = 0;

      /**
      Test if voxel ijk is active.
      */
      virtual bool isActiveVoxel(const GridCoord& /*ijk*/) const = 0;

      /**
      Return number of active voxels in the data structure.
      */
      virtual size_t getNumActiveVoxels() const = 0;

      /**
      Returns the sum of all voxel values
      */
      virtual agx::Real getTotal() const = 0;

      virtual std::tuple<float, float> getTotalAndLargest() const = 0;

      virtual GridCoordVector getActiveVoxelIndices() const = 0;

      /**
      Provide the implementation with a hint about the estimated number of voxels that are needed
      */
      virtual void sizeHint( size_t /*hint*/ ) {}

      /*
      \return true if the implementation supports iterateVoxels
      */
      virtual bool hasIterateSupport() const { return false; }

      /*
      Implementation dependent method that iterates over all voxels in some order
      and calls func( voxel_ijk, voxel_value) for each voxel.

      It is not valid to have the function change the grid while iterating.
      */
      virtual void iterateVoxels( GridFunction /*func*/ ) const {}


      /**
      Find which voxels that overlap with a geometry
      */
      virtual agx::HashSet<agx::Vec3i> findGeometryVoxelIntersections(const agx::AffineMatrix4x4& gridTransform,
                                                                      const agxCollide::Geometry* geometry,
                                                                      agx::Real voxelSize,
                                                                      bool onlyOccupiedVoxels = true ) const = 0;



      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxTerrain::BasicGrid );
      void store( agxStream::OutputArchive& out ) const override = 0;
      void restore( agxStream::InputArchive& in ) override = 0;

  };


  /**
  Extended pure virtual interface for Grid storage classes. Implements
  operations needed for the solid mass grid.
  */
  class AGXTERRAIN_EXPORT Grid : public BasicGrid
  {

  public:



    virtual agxCollide::BoundingAABB getBoundingBox() const = 0;


    virtual void setValuesBelowHeightField(
      const agxCollide::HeightField* heightField,
      agx::Real voxelSize,
      int lowestAllowedVoxelIndex,
      BasicGrid* compactionGrid) = 0;


    virtual agx::HashSet<agx::Vec3i> findConvexVoxelIntersections(const agx::AffineMatrix4x4& gridTransform,
      const agxCollide::Convex* convex,
      agx::Real voxelSize) = 0;


    virtual bool findMeshVoxelIntersections(const agx::AffineMatrix4x4& gridTransform,
      const agxCollide::Mesh* mesh,
      agx::Real voxelSize,
      agx::Vec3iVector& result) = 0;



    virtual agx::HashSet<agx::Vec3i> findVoxelsAboveTriangles(const agx::AffineMatrix4x4& gridTransform,
      const TriangleVector& triangles,
      agx::Real voxelSize,
      agx::Real margin) = 0;



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
      bool useParticleFreeDeformers) = 0;
    
    

    virtual ActiveZoneIntersectionData removeVolumeInShape( const agx::AffineMatrix4x4& gridTransformation,
                                                            const ShapeIntersectionTestData& shape,
                                                            std::function<float( const agx::Vec3i& )> getCompression,
                                                            std::function<bool( const agx::Vec2i& )> canRemoveIndex,
                                                            std::function<bool( const agx::Vec3&, const agx::Vec3i& )> isWithinBoundry,
                                                            agx::Real voxelSize,
                                                            agx::Real minOccupancyRemoval,
                                                            agx::Real minAllowedHeight) = 0;

    virtual agx::Vec3iVector findVoxelsInLineIntersection(const agx::Line& line) const = 0;

    virtual void findVoxelSurfaceFromIndex(const agx::Vec3i& index, agx::Vec3i& result) = 0;

    AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxTerrain::Grid );
  };



}
