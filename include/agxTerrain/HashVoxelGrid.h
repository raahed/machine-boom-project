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
  class AGXTERRAIN_EXPORT HashVoxelGrid : public agxTerrain::BasicGrid
  {
    public:
      using VoxelHashType = agx::HashTable< agx::Vec3i, float >;

      HashVoxelGrid( float defaultValue, agx::Real voxelSize );

      ~HashVoxelGrid();

      virtual agx::Vec3 getElementSize() const override;

      virtual size_t getMemoryUsage() const override;

      virtual bool empty() override;

      virtual void clearGrid() override;

      virtual GridCoordVector getActiveVoxelIndices() const override;

      virtual void setFloatValue(const GridCoord& ijk, float value) override;

      virtual float getFloatValue(const GridCoord& ijk) const override;

      virtual bool isActiveVoxel(const GridCoord& ijk) const override;

      virtual size_t getNumActiveVoxels() const override;

      virtual agx::Real getTotal() const override;

      virtual std::tuple<float, float> getTotalAndLargest() const override;

      virtual void sizeHint( size_t hint ) override;

      virtual bool hasIterateSupport() const override;

      virtual void iterateVoxels( GridFunction /*func*/ ) const override;

      virtual agx::HashSet<agx::Vec3i> findGeometryVoxelIntersections(const agx::AffineMatrix4x4& gridTransform,
                                                                      const agxCollide::Geometry* geometry,
                                                                      agx::Real voxelSize,
                                                                      bool onlyOccupiedVoxels = true ) const override;

      // Serialization
      AGXSTREAM_DECLARE_SERIALIZABLE( agxTerrain::HashVoxelGrid );

    protected:
      HashVoxelGrid();

    private:
        VoxelHashType m_table;
        agx::Vec3 m_voxelSize;

        float m_default;
  };
}

