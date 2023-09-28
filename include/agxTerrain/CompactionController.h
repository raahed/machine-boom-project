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
#include <agxTerrain/export.h>
#include <agx/Referenced.h>
#include <agxTerrain/TerrainContactGenerator.h>
#include <agx/Vec3.h>

namespace agxCollide
{
  class Geometry;
}

namespace agxTerrain
{
  class Terrain;
  class TerrainDataAtlas;
  class TerrainMassController;

  typedef agx::HashTable<agxCollide::Geometry*, agx::Real> GeometryContactTimeTable;
  typedef agx::HashTable<agx::Vec3i, agx::Real> VoxelStressTable;

  AGX_DECLARE_POINTER_TYPES(CompactionController);
  AGX_DECLARE_VECTOR_TYPES(CompactionController);

  /**
  Controller that handles compaction in the terrain from generated terrain contacts in TerrainContactGenerator
  The controller features two methods that calculate stress depth propagation in the terrain soil:
  BOUSSINESQ - Two part method that calculates the stress field by combining the calculated surface stress and subsoil stress based on the surface contact point loads.
  UNIFORM_DISK - Simple method that approximates the stress propagation by assuming that the surface stress is applied by a circular uniform disk of voxel area in each voxel active in the terrain contact.
  */
  class AGXTERRAIN_EXPORT CompactionController : public agx::Referenced
  {
  public:
    enum StressCalculationMethod
    {
      BOUSSINESQ,
      UNIFORM_DISK
    };

    enum CompactionMethod
    {
      DENSITY_LOG10,
      VOID_RATIO_LN
    };

  public:
    /**
    Construct given terrain instance.
    */
    CompactionController(Terrain* terrain);

    /**
    \internal

    Default constructor used in serialization.
    */
    CompactionController();

    /**
    The onPost function updates compaction from terrain contacts
    */
    void onPost();

    /**
    Set the stress method used when calculating compaction
    \param stressMethod - The stress method to used when calculating compaction
    */
    void setStressMethod(StressCalculationMethod stressMethod);

    /**
    Get the stress method used when calculating compaction
    */
    StressCalculationMethod getStressMethod() const;

    /**
    Set the compaction method used when calculating compaction
    */
    void setCompactionMethod(CompactionMethod compactionMethod);

    /**
    Get the compaction method used when calculating compaction
    */
    CompactionMethod getCompactionMethod() const;

    /**
    Create compaction from an artificially applied surface force.
    \param voxelXY - The 2D terrain coordinate in voxel space
    \param surfaceForce - The surface force to apply on the surface
    \param contactTime - The contact time that should be used when calculating the relaxation time factor
    \return True if the terrain was compacted from the applied force
    */
    bool applyCompactionFromSurfaceForce(const agx::Vec2i& voxelXY, agx::Real surfaceForce, agx::Real contactTime);

    /**
    Get the current active stress in the specified voxelIndex.
    \param voxelIndex - The specified voxel index.
    \return The current stress active in the voxel given by the specified voxel index.
    */
    agx::Real getStressInVoxel(const agx::Vec3i& voxelIndex) const;

    /**
    Get the indices that has been compressed this time step after the post step of the compactionController
    */
    const agx::HashSet<agx::Vec2i>& getCompressedTerrainIndices() const;

  public:
    DOXYGEN_START_INTERNAL_BLOCK()
    AGXTERRAIN_STORE_RESTORE_INTERFACE;
    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    virtual ~CompactionController();

    /**
    Update compaction from terrain contacts using the chosen stress method. Triggered in onPost.
    */
    void updateCompactionFromTerrainContacts(const TerrainContactRefVector& contacts);

    // Compact the terrain using the uniform disk method
    void handleCompactionFromTerrainContactsUniformDisk(const TerrainContactRefVector& contacts);

    // Compact the terrain using the uniform disk method
    agx::Bool compactColumnWithPointForcesUniformDiskLoad(
      const agx::Vec2i& voxelXY, agx::Real surfaceStress, agx::Real loadRadius, agx::Real currentContactTime);

    // Compact the terrain using the BOUSSINESQ method
    void handleCompactionFromTerrainContactsBoussinesq(const TerrainContactRefVector& contacts);

    // Compact the terrain using the BOUSSINESQ method
    agx::Bool compactColumnWithPointForces(const agx::Vec2i& voxelXY,
                                           const TerrainContact::SurfaceForceVector& pointForces,
                                           agx::Real surfaceStress,
                                           bool voxelSurfacecIsInContactArea,
                                           agx::Real minDepth,
                                           agx::Real maxDepth,
                                           agx::Real currentContactTime);

    // Calculate compaction in voxel given a compaction stress and time factor
    bool doCompactionInVoxel(const agx::Vec3i& voxelIndex, agx::Real compactionStress, agx::Real timeFactor);

    // Calculate subsoil stress in the BOUSSINESQ method
    agx::Real calculateTotalSubSoilStress(const agx::Vec3& voxelPoint,
                                          bool voxelSurfacecIsInContactArea,
                                          const TerrainContact::SurfaceForceVector& pointForces) const;

    void debugRenderStressInNode(const agx::Vec3& position, agx::Real stress) const;

    TerrainMassController* getMassController() const;

    friend class Terrain;

  private:
    Terrain* m_terrain;
    VoxelStressTable m_voxelIndexToStress;
    agx::HashSet<agx::Vec2i> m_compressedTerrainIndices;
    StressCalculationMethod m_stressMethod;
    CompactionMethod m_compactionMethod;
  };
}
