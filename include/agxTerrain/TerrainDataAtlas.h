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

#include <agxTerrain/export.h>
#include <agxTerrain/Grid.h>
#include <agx/StackArray.h>

namespace agxTerrain
{
  typedef agx::HashTable<agx::Vec3i, agx::Vec3f> VoxelToVelocityTable;

  AGX_DECLARE_POINTER_TYPES(TerrainDataAtlas);

  /*
  Atlas for holding information about terrain mass in the grid system.
  Is active in the same index space as the regular grid.
  */
  class AGXTERRAIN_EXPORT TerrainDataAtlas : public agx::Referenced
  {
    static constexpr float MAX_MASS = 1.0;

  public:
    enum MassType
    {
      SOLID,
      PARTICLE,
      FLUID
    };
    struct MassTypeHash
    {
      AGX_FORCE_INLINE agx::UInt32 operator()(TerrainDataAtlas::MassType key) const
      {
        return agx::hash((agx::UInt32)key);
      }
    };

  public:
    /**
    Basic Constructor.
    \param voxelSize - The uniform voxel dimension used in the internal transformation of the terrain data grids.
    \param resolutionX - The resolution of the voxel grid in X.
    \param resolutionY - The resolution of the voxel grid in Y.
    */
    TerrainDataAtlas(agx::Real voxelSize, size_t resolutionX, size_t resolutionY, agx::Int depth);

    /**
    Get the occupancy mass of specific mass type in a voxel with specified 3D-coordinate in voxel index space.
    \param coord - The 3D index coordinate in voxel space.
    \param massType - The type of mass data that should be extracted from the voxel.
    \returns The occupancy mass of the specified type at the specified coordinate. Will return zero if the specified voxel at the coordinate is inactive.
    */
    float getMass(const Grid::GridCoord& coord, MassType massType) const;

    /**
    Sets the occupancy mass of specific mass type in a voxel with specified 3D-coordinate in voxel index space.
    Setting a mass to the voxel with activate it in the specified mass grid.
    \param coord - The 3D coordinate of the voxel in the voxel index space.
    \param massType - The type of mass data that should be inserted into the voxel.
    \param value - the occupancy value, which can range from [0.0 - 1.0] of the specified mass type that should inserted.
    */
    void setMass(const Grid::GridCoord& coord, MassType massType, float value);

    /**
    Adds occupancy mass of specific mass type in a voxel with specified 3D-coordinate in voxel index space.
    Setting a mass to the voxel with activate it in the specified mass grid.
    \param coord - The 3D coordiante of the voxel in the voxel index space.
    \param massType - The type of mass data that should be inserted into the voxel.
    \param value - The occupancy value of the specific type that should be added to the voxel.
    \return the new total occupancy mass in the voxel
    */
    float addMass(const Grid::GridCoord& coord, MassType massType, float value);

    /**
    Sets compression data in a voxel in the grid. The compression determines how much solid occupancy can exist in a voxel.
    by the following relation: MAX_SOILD_VOLUME = compression * NOMINAL_MAX_VOLUME
    \param coord - The 3D voxel coordinate of the voxel in voxel index space.
    \param value - The compression value to set to the voxel.
    */
    void setCompaction(const Grid::GridCoord& coord, float value);

    /**
    Gets the compression data from a voxel in the grid. The compression determines how much solid occupancy can exist in a voxel.
    by the following relation: MAX_SOILD_VOLUME = compression * NOMINAL_MAX_VOLUME
    \param coord - The 3D voxel coordinate of the voxel in the voxel index space.
    */
    float getCompaction(const Grid::GridCoord& coord) const;

    /**
    Get the occupancy mass in a voxel modified by the compaction grid with specified 3D-coordinate in voxel index space.
    The returned mass is modified by the following formula: return = occupancyMass / compaction. This is mainly used when
    trying to sum compacted soild mass with particle and fluid mass in the same voxel. Particle and fluid occupancy is always defined in nominal compression.
    \param coord - The 3D index coordinate in voxel space.
    \returns The occupancy mass at the specified coordinate, modified my the compaction grid. Will return zero if the specified voxel at the coordinate is inactive.
    */
    float getCompactedSolidMass(const Grid::GridCoord& coord) const;

    /**
    Returns the maximum allowed solid mass in the voxel. This is a function of the current compression in the voxel.
    \param coord - The 3D voxel coordinate of the voxel in the voxel index space.
    \return the maximum allowed solid mass in the terrain.
    */
    float getMaximumAllowedSolidMassInVoxel(const Grid::GridCoord& coord) const;

    /**
    Checks if a voxel is active in data grid of a specific mass type.
    \param coord - The 3D coordiante of the voxel in the voxel index space.
    \param massType - The specified mass type.
    */
    bool isActiveVoxel(const Grid::GridCoord& coord, MassType massType) const;

    /**
    Returns the active indices in voxel index space which contains occupancy mass of a specific mass type.
    \param massType - the specific mass type
    \returns all the active voxel indices which contains occupancy of the specified mass type.
    */
    agx::Vec3iVector getIndicesWithMass(MassType massType) const;

    float getTotalMassInVoxel(const Grid::GridCoord& coord) const;

    void setVelocity(const Grid::GridCoord& coord, const agx::Vec3f& value);
    agx::Vec3f getVelocity(const Grid::GridCoord& coord) const;
    void clearVelocityGrid();

    Grid* getSolidGridMassData() const;

    BasicGrid* getGridMassData(MassType massType) const;
    BasicGrid* getCompactionGridData() const;

    size_t getTotalGridMemoryUsage() const;

    size_t getTotalActiveVoxels() const;

    void clearMassGrid(MassType massType);

    agx::Real getMassInTerrain(MassType massType) const;

    VoxelToVelocityTable& getVelocityTable();

  public:
    DOXYGEN_START_INTERNAL_BLOCK()
    AGXTERRAIN_STORE_RESTORE_INTERFACE;
    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    /**
    Reference counted object - protected destructor.
    */
    virtual ~TerrainDataAtlas();

    void fillGridVector();

  protected:
    BasicGridRef m_solidMassGrid;
    BasicGridRef m_fluidMassGrid;
    BasicGridRef m_particleMassGrid;
    BasicGridRef m_compactionGrid;

    VoxelToVelocityTable m_voxelToVelocityTable;

    agx::StackArray<BasicGrid*, 3> m_gridVector;

    Grid*     m_rawSolidMassGrid; // alias for m_solidMassGrid
  };
}
