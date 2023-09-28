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

#include <agx/HashSet.h>
#include <agx/HashTable.h>

#include <agx/config/AGX_USE_AGXTERRAIN.h>
#include <agxTerrain/export.h>

#include <agxSDK/StepEventListener.h>

#include <agxTerrain/TileSpecification.h>
#include <agxTerrain/TerrainCache.h>

#include <agxStream/StorageStream.h>

namespace agxTerrain
{

  class TerrainCache;
  class TerrainDataSource;

  AGX_DECLARE_POINTER_TYPES(TerrainPager);
  AGX_DECLARE_VECTOR_TYPES(TerrainPager);

  /**
  A pager that will dynamically handle terrain tiles in a agxSDK::Simulation.

  The pager uses a 2D-grid and populates the grid tiles with terrain instances as needed.
  There is a reference point which specifies the (0,0) position for the 2D-grid. There is
  also a reference rotation which should be used to map the grids local Z-axis to the up
  direction at the reference point.

  The reference point should be specified to be near the location in the world
  where the simulation is being performed.

  In a default created Simulation with uniform gravity, the reference rotation can
  be set to the identity matrix since AGX uses Z as default up axis. If the gravity direction
  is changed and e.g. the x-z plane is used for ground and y for up, then the rotation
  must be set accordingly.

  On the other hand, if the Simulation is being performed on a sphere (e.g Earth) with a
  point gravity source, then it is important that the working area is relatively close
  to the reference point. With Earths radius and a distance 10km away from the reference point,
  a height difference of a few meters can be observed due to Earths curvature. The approximation
  within the pager to work with a 2D-grid can handle these heigh differences within the different
  tiles without problem. But if the reference point is set at the north pole and the working area
  is near the equator problems will occur.

  To track which grid tiles that needs to have active terrain instances, one or more bodies
  are added to the TerrainPager. The world position for those bodies will be used
  to determine which tiles should be available.
  Along with the body position, two radiuses are used:
  - One radius for required tiles. These must be available for the simulation to step.
  This will add a small delay in the initial setup.
  - A second radius for preloading. The preloading radius should be larger than the
  required radius and is used so that tiles are made available before they are reached
  by the required radius. Preloaded tiles are inserted into the simulation when ready and
  the simulation can be stepped while the preloading occurs.


  Excavation in the Terrain works best when the excavation is being performed within a tile.
  Therefor tiles can have a margin so that they have a overlap to make the transition from one
  tile to the next work so the excavation is always kept within a single tile and never
  reaches a tile edge. This margin is specified via how many height values that should
  overlap.

  If margins larger than 0 are used, then this affects which data points from each tiles
  heightfield that should be used for rendering to avoid drawing the same part twice.

  NOTE:
  - The agxSDK::Simulation does not support that the update task is rebuilt during stepForward.
  - The TerrainPager must be able to add terrains to the simulation during stepForward.
  - Adding a terrain to the simulation will set some solver parameters and possibly cause
    the update task to be rebuilt.
  - Hence, simulation->add( terrainpager ) prepares the Simulation so that the update task
    will not be rebuilt when adding terrains.
  - It is NOT supported to change the following Solver parameters after a TerrainPager has been
    added to a Simulation: setUseParallelPgs, setUse32bitGranularBodySolver, setUseGranularWarmStarting
  - The Broadphase algorithm will be set to Hierarchical_Grid and it is NOT supported to change
    it back to Sweep-And-Prune.
  */
  class AGXTERRAIN_EXPORT TerrainPager : public agxSDK::StepEventListener
  {
  public:

    /// Callback type
    typedef agx::Callback2< agxTerrain::TileId, agxTerrain::Terrain* > TileCallback;

    /// Event types
    typedef agx::Event2< agxTerrain::TileId, agxTerrain::Terrain* > TileLoadEvent;
    typedef agx::Event2< agxTerrain::TileId, agxTerrain::Terrain* > TileUnloadEvent;

    struct TileAttachments {
      agx::Real               m_lowestPoint;
      agx::Real               m_highestPoint;
      agx::Real               m_zOffset;
      agxTerrain::TerrainRef  m_terrainTile;

      agx::UInt32             m_unusedCount;
    };

    typedef agx::Vector<TerrainPager::TileAttachments*> TileAttachmentPtrVector;

    /**
    Constructor. Specifies tile information and the 2D-plane used for dynamic terrain tiles.

    The TerrainPager do not mirror all the different parameters a terrain exposes,
    instead a template terrain is used for settings.

    To support excavation, the terrain tiles have an optional margin. This margin can
    be set so that the excavation is performed within one tile and correct soil wedges
    are formed.

    Example: tileResolution = 301, tileOverlap 10, tileElementSize = 0.25, ....

    Each tile then becomes (301-1)*0.25 = 75m x 75m
    The overlap on each side will be 10*0.25 = 2.5m

    \param tileResolution    Specifies the number of height values for each dimension for (tile+margin).
    \param tileOverlap       Specifies the number of height values that should overlap between two tiles.
    \param tileElementSize   Size in meters for one element. Same as distance between two height values.
    \param maximumDepth      How deep in meters the terrain can be excavated
    \param refPoint          A world reference point for where the terrain is located
    \param refRotation       A rotation that will transform the Z-axis to the up direction at the referencePoint
    \param templateTerrain   A template terrain whose settings will be used for the terrain tiles.
    */

    TerrainPager(size_t tileResolution, size_t tileOverlap, agx::Real tileElementSize, agx::Real maximumDepth,
                 agx::Vec3 refPoint, agx::Quat refRotation,
                 agxTerrain::Terrain* templateTerrain );


    /**
    Specifies a new reference point for the terrain grid tiles.
    This is an expensive operation and should be avoided if possible.
    - All old data and the cache will be cleared since the transform from 2D-grid to world will be affected
    - All required tiles will have to be reloaded. Can cause some delay on the next simulation stepForward.
    */
    //void setTerrainPlaneLocation( agx::Vec3 referencePoint, agx::Quat referenceRotation );

    /**
    The tile specification is set via the constructor and optionally modified via setTerrainPlanLocation.
    This method gives read access to terrain tile paging setup.
    */
    const TileSpecification& getTileSpecification() const;


    /**
    Specify the location for there terrain tiles should be stored when cached to disk.
    */
    bool setFileCacheDirectory( agx::String dir );

    /**
    Return the location for where files should be temporary cached when not needed in memory.
    */
    agx::String getFileCacheDirectory() const;

    /**
    Sets if the terrain pager should store compaction data for tiles that are paged out.
    \param shouldStoreCompaction Bool flag to set if compction should be stored.
    \param maximumCompactionDepth Sets how deep compaction data should be stored. Specifically how many
    voxels below the surface.
    */
    void setShouldStoreCompaction(bool shouldStoreCompaction, size_t maximumCompactionDepth);

    /**
    Returns whether the terrain pager should store compaction data for tiles that are paged out.
    */
    bool getShouldStoreCompaction() const;

    /**
    Sets if deformation should be enabled for terrains managed by this pager. If this is set to false no 
    dynamic mass particles or fluid mass is created. Additionally, no avalanching or compaction occurs on the terrain.
    This changes and applies the undelying template terrain.
    \param deformable - Whether or not to deformations should occur on this terrain
    */
    void setDeformable( bool deformable );

    /**
    Gets whether or not deformations are enabled for the tiles managed by this pager. If false, then no 
    dynamic mass particles or fluid mass is created. Additionally, no avalanching or compaction occurs on the terrain.
    \return true if deformations are enabled for this pager, false otherwise
    */
    inline bool getDeformable() const;

    /**
    Sets if cached files written out to disk should be embeded in AGX Archives
    when store is called. Default false.
    */
    void setEmbedCacheFilesInArchive( bool embed );

    /**
    Returns if cached files should be embedded in AGX Archives used
    for serialization.
    */
    bool getEmbedCacheFilesInArhive() const;

    /**
    Specify whether the terrain pager should store changes to the terrain every time step.
    Should only be true if restore function is planned to be used.
    */
    void setShouldStoreDeltas(bool shouldStoreDeltas);

    /**
    Return whether the terrain pager should store changes to the terrain every time step.
    */
    bool getShouldStoreDeltas();

    /**
    Returns a vector of the terrains currently tracked by this pager.
    */
    TileAttachmentPtrVector getActiveTileAttachments();

    /**
    Returns the soil simulation interface used by the terrains tracked by this pager or nullptr if there are no active terrains
    */
    SoilSimulationInterface* getSoilSimulationInterface();

    /**
    Adds a RigidBody to the terrain pager. The pager will make sure that terrain tiles
    within the required radius are present in the Simulation before stepping and tiles
    within the preload radius will be loaded and inserted when available.
    */
    bool add( agx::RigidBody* body, agx::Real requiredTileRadius, agx::Real preloadTileRadius );

    /**
    Adds a Shovel and its RigidBody to the terrain pager. The pager will make sure that
    terrain tiles within the required radius are present in the Simulation before stepping
    and tiles within the preload radius will be loaded and inserted when available.
    */
    bool add( agxTerrain::Shovel* shovel, agx::Real requiredTileRadius, agx::Real preloadTileRadius );


    /**
    Remove body so that the terrain pager no longer uses the body when determining
    which terrain tiles that are needed.

    */
    bool remove( agx::RigidBody* body );

    /**
    Removes a shovel object from the terrain pager, if it exists.

    \return true if successful, false otherwise
    */
    bool remove( agxTerrain::Shovel* shovel );

    /**
    Fetches the template terrain.
    The settings for the template terrain are used on new terrain tiles
    when they are inserted into the simulation.

    \see applyChangesToTemplateTerrain
    */
    agxTerrain::Terrain* getTemplateTerrain();

    /**
    Signals to the TerrainPager that the template terrain has been updated
    and that any changes to the template terrain should be applied to the
    active terrain tiles in the pager.
    */
    void applyChangesToTemplateTerrain();



    /**
    Set tile radiuses for a already added RigidBody.
    If a radius is negative, then it is ignored making it possible to have a body that does not
    require any tiles but can be used to preload tiles.
    */
    bool setTileLoadRadiuses( agx::RigidBody* body, agx::Real requiredTileRadius, agx::Real preloadTileRadius );

    /**
    Get the tile readiuses for a body. If body is not found, (-1,-1) is returned.
    The first number in the pair is the required radius, the second the preload radius
    */
    agx::RealPair getTileLoadRadius( agx::RigidBody* body );

    /**
    Returns the data source the pager uses for fetching height data for tiles.
    */
    TerrainDataSource* getTerrainDataSource();

    /**
    Sets the data source that the pager needs to fetch terrain tile data.
    \see TerrainRasterizer
    */
    void setTerrainDataSource( TerrainDataSource* tds );



    /**
    Notification to the TerrainPager when being added to a Simulation
    */
    virtual void addNotification() override;

    /**
    Performs cleanup when removed from the simulation.
    */
    virtual void removeNotification() override;



    /**
    Callback from Simulation. Ready preloaded terraintiles are inserted at this stage.
    */
    virtual void preCollide( const agx::TimeStamp& time ) override;


    /**
    Callback from Simulation.
    Height synchronization for overlapping parts.
    No longer needed tiles are removed at this stage.
    */
    virtual void last( const agx::TimeStamp& time ) override;


    /**
    Enum with different types of result from the raycast method.
    */
    enum TerrainRaycastResult {

      RAYCAST_NO_RESULT,                // No hit
      RAYCAST_HIT_ACTIVE_TERRAIN_TILE,  // Hit active paged in tile
      RAYCAST_RESULT_FROM_DATASOURCE    // Hit via TerrainDataSource
    };

    /**
    Perform raycasting against the paged terrain.

    Test via active tiles:
    - hit: return RAYCAST_HIT_ACTIVE_TILE
    - miss: Perform test via TerrainDataSource if supported
    - hit: return RAYCAST_RESULT_FROM_DATASOURCE
    - miss/not supported: return RAYCAST_NO_RESULT

    If a terrain tile is modified and then paged out, a result from the datasource
    will not show those modifications. For best results, raycasting should be
    performed against paged in tiles.
    */
    TerrainRaycastResult raycast( agx::Vec3 start, agx::Vec3 end, agx::Vec3& result );


    using StepEventListener::store;

    using StepEventListener::restore;

    /**
    Stores internal data into stream.
    */
    void store(/*agxStream::StorageStream& str*/agxTerrain::TileModificationVector& storeData);


    /**
    Restores internal data from stream.
    */
    void restore(/*agxStream::StorageStream& str*/ const agxTerrain::TileModificationVector& restoreData);


    /**
    An event that the TerrainPager will trigger after a new tile as been loaded
    and added to the simulation.
    */
    TileLoadEvent tileLoadEvent;

    /**
    An event that the TerrainPager will trigger just before it removes a tile
    from the simulation.
    */
    TileUnloadEvent tileUnloadEvent;

    /**
    Calculates the total mass of all currently paged in terrain tiles.
    The mass of the areas where tiles are overlapping is not counted twice.
    NOTE: The mass of the areas where tiles are overlapping might be out of
    sync if this method is called in a time-step where one or more vertices
    within the overlapping area has been modified. Which might make this
    calculation invalid.
     */
    agx::Real getTotalTerrainMass();

    DOXYGEN_START_INTERNAL_BLOCK()
      AGXSTREAM_DECLARE_SERIALIZABLE(agxTerrain::TerrainPager);
    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    /**
    Serializable. Default constructor.
    */
    TerrainPager();

    virtual ~TerrainPager();

    void updateActiveTiles();

    void unloadUnusedTiles( bool partOfRemoveNotification );

    void handleNewTerrainTile( agxTerrain::TerrainTile );

    void unloadAllTiles();

    void prepareSimulation(bool fromRestore);

    agx::Real getOverlappingAreaSolidMass(agxTerrain::TileId, agxTerrain::TileId);

  private:
    mutable TerrainCache m_cache;

    TileSpecification    m_tileSpec;

    bool                 m_shouldStoreDeltas;

    bool                 m_justRestored;

    struct BodyAttachments {
      //agx::RigidBodyRef     m_body;
      agx::Real             m_requiredRadius;
      agx::Real             m_preloadRadius;
      agxTerrain::ShovelRef m_shovel;
    };

    typedef std::pair<TileId, agx::Vec2i> GlobalTerrainIndex;

    // Bodies this TerrainPager keeps track of
    agx::HashTable< agx::RigidBody*, BodyAttachments > m_bodies;

    // Tiles this TerrainPager keeps track of
    agx::HashTable< TileId, TileAttachments > m_terrainTiles;

    // Stores heightfield changes since the last call to the store method
    agx::HashTable< GlobalTerrainIndex, agx::Real > m_heightFieldDelta;
    //agx::HashTable< TileId, TerrainIndexHeight > m_heightFieldDelta;

    agx::Physics::GranularBodySystemRef m_gbs;

    agx::Vector<agxTerrain::TileId>  m_missingRequiredTiles;
    agx::HashSet<agxTerrain::TileId> m_pendingRequests;



    void fillTileAttachment( agxTerrain::TerrainTile tile, agxTerrain::TerrainPager::TileAttachments& ta );

    void getCompactionFromNeighbourTiles(agxTerrain::TerrainTile tile);

    void fillHeightDelta(const GlobalTerrainIndex& gtid, agx::Real height);

  };


  inline void TerrainPager::fillHeightDelta(const GlobalTerrainIndex& gtid, agx::Real height)
  {
    m_heightFieldDelta[gtid] = height;
  }


  inline void TerrainPager::setEmbedCacheFilesInArchive( bool embed )
  {
    m_cache.setEmbedCacheFilesInArchive( embed );
  }


  inline bool TerrainPager::getEmbedCacheFilesInArhive() const
  {
    return m_cache.getEmbedCacheFilesInArhive();
  }

  inline bool TerrainPager::getDeformable() const
  {
    return m_cache.getTemplateTerrain()->getDeformable();
  }
}

