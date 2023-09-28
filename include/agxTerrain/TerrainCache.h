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

#include <agx/config/AGX_USE_AGXTERRAIN.h>

#include <agx/Thread.h>

#include <agxUtil/agxUtil.h>

#include <agxTerrain/export.h>
#include <agxTerrain/Terrain.h>
#include <agxTerrain/TileSpecification.h>

#include <agxTerrain/TerrainDataSource.h>


namespace agxTerrain
{

  /**
  Struct with data to describe a terrain tile
  */
  struct TerrainTile {
    agxTerrain::TileId     id;
    agxTerrain::TerrainRef terrain;
    agx::RealPair          zMinMax;
  };

  typedef agx::VectorPOD<agx::RealVector> modifiedCompactionVector;

  /**
  The TerrainCache is used internally by the TerrainPager to provide requested terrain tiles.

  The first time a tile is requested, the tile is fetched from a TerrainDataSource.
  On later reuse, the tile is handled via the paging system where it might reside
  in memory or on disk.
  */
  class AGXTERRAIN_EXPORT TerrainCache
  {
  public:
    enum TilePriority {
      TP_HIGH,       /// Will be placed first in queue for processing
      TP_LOW         /// Will be places last in queue for processing
    };


    /*
    Possible state transitions:

    MISSING  -> LOADED -> ACTIVE -> RETURNED
    RETURNED -> LOADED
    RETURNED -> ON_DISK -> LOADED
    RETURNED -> IN_RAM  -> LOADED
    */
    enum TileStatus {
      TS_MISSING,    /// Tile not seen before, must be loaded from data source
      TS_LOADED,     /// Tile is loaded and ready to be used by the TerrainPager
      TS_ACTIVE,     /// Tile has been fetched by the TerrainPager
      TS_RETURNED,   /// Tile has been returned to the Cache by the TerrainPager
      TS_IN_RAM,     /// Tile is paged out, heights held in memory
      TS_ON_DISK     /// Tile is paged out, heights stored in a file on disk
    };



    /**
    Constructor
    */
    TerrainCache();

    /**
    Destructor
    */
    virtual ~TerrainCache();

    /**
    Create agxTerrain::Terrain instances so that there are poolSize number
    of Terrains in the pool for future use. Creating the Terrains in advance
    should reduce the risk for interruptions during the simulation.
    */
    void prepareTerrainInstances( const TileSpecification& ts, size_t poolSize );


    /**
    Load a terrain tile, either from cache (memory or disk) or request the tile from a data source.
    The priority prio determines if the tile is loaded immediately or scheduled for being loaded in
    the background.
    */
    void loadTerrainTile( const TileSpecification& ts, TileId id, TilePriority prio);

    /**
    Schedule a tile for unloading.
    Unloads data from a terrain tile and stores it. The agxTerrain::Terrain instance
    will be cleared and returned to the internal pool so it can be reused later when needed.
    */
    void unloadTerrainTile( TerrainTile tile );

    /**
    Returns the template terrain used for properties of new terrain tiles.
    */
    agxTerrain::TerrainRef getTemplateTerrain();


    /**
    Sets the template terrain used for properties of new terrain tiles.
    */
    void setTemplateTerrain(agxTerrain::TerrainRef templateTerrain);

    /**
    Sets if the terrain pager should store compaction data for tiles that are paged out.
    */
    void setShouldStoreCompaction(bool shouldStoreCompaction);

    /**
    Returns whether the terrain pager should store compaction data for tiles that are paged out.
    */
    bool getShouldStoreCompaction() const;

    void setMaximumCompactionDepth(size_t maximumCompactionDepth);

    agx::HashTable<agxTerrain::TileId, agx::Vec2iVector>* getCompressedIndices();


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
    Returns the data source used for height data.
    */
    TerrainDataSource* getTerrainDataSource() const;

    /**
    Sets the data source for height data. This operation
    will block until ongoing load requests have completed.
    */
    void setTerrainDataSource( TerrainDataSource* tds );


    /**
    */
    TileStatus getTileStatus( TileId id ) const;


    /**
    Specifies where paged out height data for terrain should be stored.
    */
    bool setFileCacheDirectory( agx::String dir );

    /**
    \return The directory where paged out data is stored
    */
    agx::String getFileCacheDirectory() const;

    /**
    */
    void applyChangesToTemplateTerrain(agxTerrain::Terrain* t);

    /**
    Waits until one tile is ready.
    */
    void waitForTile();


    /**
    Checks if one or more tiles are ready.
    */
    bool haveReadyTile();


    /**
    Without blocking, this method tries to return a ready processed terrain tile.
    If no tile is available, the terrain member in the tile will be a null pointer.
    */
    agxTerrain::TerrainTile getReadyTile();


    /**
    Specifies the restore data that is used to apply changes the first
    time they are needed after restore.

    At first, the tile heights are requested from the TerrainDataSource,
    then the modifications are applied. Finally the terrain tile is placed
    in the ready queue and can be fetched with getReadyTile.
    */
    void setRestoreData( const TileModificationVector& state );




    /**
    Removes all internal state.
    All height values will have to be queried again from the data source.
    */
    void clearCache();

    /**
    Checks if there is staged terrain which can be moved to the pool for future usage
    */
    void handleStagedTerrain();


  public:

    DOXYGEN_START_INTERNAL_BLOCK()
    void store( agxStream::OutputArchive& out );
    void restore( agxStream::InputArchive& in, const agxTerrain::TileSpecification& tileSpec );
    DOXYGEN_END_INTERNAL_BLOCK()


  protected:
      void pauseBackgroundThread();

      void resumeBackgroundThread();

  private:

    TileSpecification m_tileSpec; // we need our own copy

    TileModificationVector m_restoreData;
    agx::HashTable < TileId, std::pair<size_t, size_t>> m_restoreDataTileIdTable;

    /*

    */
    struct CachedTile {
      agx::RealPair               zMinMax;
      agx::RealVector             heights;
      size_t                      numberOfModifiedVoxels;
      agx::Vector<agx::RealVector> compactionVector;
      //modifiedCompactionVector    compactionVector;
      agx::Vec2iVector  compactedIndices;
    };


    size_t readTerrainHeightsAndCompaction(agxTerrain::Terrain* t, agx::RealVector& heightsOutputVector, agx::Vector<agx::RealVector>& compactionOutputVector, agx::Vec2iVector& compactedIndices, agxTerrain::TileId id);

    void readTerrainHeights( agxTerrain::Terrain* t, agx::RealVector& outputVector );

    //
    void backgroundOperations();


    void performLoadTile( agxTerrain::TileSpecification ts, agxTerrain::TileId id );

    void performUnloadTile( TerrainTile tile );

    agx::String getFilenameForTile( agxTerrain::TileId id );

    bool writeCacheFile( agx::String filename, const CachedTile& tile );

    bool readCacheFile( agx::String filename, CachedTile& output );


    void storeCachedTile( agxStream::OutputArchive& out, const TileId& tileId, const CachedTile& tile );
    void restoreCachedTile( agxStream::InputArchive& in, TileId& tileId, CachedTile& tile );

    /**
    For performance reasons, upon creation of the TerrainCache, a number of
    agxTerrains are created and stored in a pool.

    This method fetches an item from the pool.

    If the pool is empty, a new instance will be created and placed in staging.
    The main thread will then move the staged terrain to the pool.
    This handling is so that the data will reside in the correct storages and
    not require any locks when simulation->add is performed.
    */
    agxTerrain::TerrainRef getTerrainInstance( const TileSpecification& ts );

    agxTerrain::Terrain* createTerrainInstance( const TileSpecification& ts );


    // Terrain instances held in the pool must have their data in storages
    // belonging to the simulation or main thread
    agxTerrain::TerrainRefVector     m_terrainPool;

    // These must be moved to the pool in a thread safe way
    agxTerrain::TerrainRefVector     m_terrainStaging;

    agxTerrain::TerrainDataSourceRef m_terrainDataSource;
    agxTerrain::TerrainRef           m_templateTerrain;

    agx::HashTable< agxTerrain::TileId, agx::Vec2iVector> m_compressedIndices;
    agx::HashTable< agxTerrain::TileId, CachedTile > m_tiles;

    agx::String m_fileCacheDir;


    std::thread             m_backgroundThread;

    std::mutex              m_mutex;
    std::mutex              m_createTerrainMutex;

    std::condition_variable m_producerCondition; // background thread waits on this condition when it's pause or has nothing to do

    std::condition_variable m_consumerCondition; // main thread waits on this condition when it requires a tile to consume

    std::condition_variable m_clearCondition;    // main thread waits on this condition when the background thread should clear stuff

    bool                    m_backgroundThreadPaused; // Flag to indicate that the background thread has reached the waiting/pause block.
                                                      // There is more than one reason why the background thread could wake the main thread
                                                      // and it must be possible to see why.

    bool                    m_shutdownThread;
    bool                    m_pauseThread;
    bool                    m_performClear;

    bool                    m_shouldStoreCompaction;
    bool                    m_shouldEmbedCachedFiles;

    size_t                  m_maximumCompactionDepth;


    // Shared structures:
    agx::HashTable< agxTerrain::TileId, TileStatus  > m_tileStatusTable;


    std::deque< std::pair<agxTerrain::TileSpecification, agxTerrain::TileId> > m_loadRequests;
    //std::deque< std::pair<agxTerrain::TerrainTile, agxTerrain::TileId> > m_unloadRequests;
    std::deque< agxTerrain::TerrainTile > m_unloadRequests;

    std::queue< agxTerrain::TerrainTile > m_readyTiles;
  };



  AGX_FORCE_INLINE void agxTerrain::TerrainCache::setShouldStoreCompaction(bool shouldStoreCompaction)
  {
    m_shouldStoreCompaction = shouldStoreCompaction;
  }

  AGX_FORCE_INLINE bool agxTerrain::TerrainCache::getShouldStoreCompaction() const
  {
    return m_shouldStoreCompaction;
  }

  AGX_FORCE_INLINE void agxTerrain::TerrainCache::setMaximumCompactionDepth(size_t maximumCompactionDepth)
  {
    m_maximumCompactionDepth = maximumCompactionDepth;
  }

  AGX_FORCE_INLINE agx::HashTable<agxTerrain::TileId, agx::Vec2iVector>* agxTerrain::TerrainCache::getCompressedIndices()
  {
    return &m_compressedIndices;
  }

  AGX_FORCE_INLINE TerrainDataSource* TerrainCache::getTerrainDataSource() const
  {
    return m_terrainDataSource;
  }

  AGX_FORCE_INLINE agxTerrain::TerrainRef TerrainCache::getTemplateTerrain()
  {
    return m_templateTerrain;
  }

  AGX_FORCE_INLINE void  TerrainCache::setTemplateTerrain(agxTerrain::TerrainRef templateTerrain)
  {
    m_templateTerrain = templateTerrain;
  }


  AGX_FORCE_INLINE agx::String TerrainCache::getFileCacheDirectory() const
  {
    return m_fileCacheDir;
  }


  AGX_FORCE_INLINE void TerrainCache::setEmbedCacheFilesInArchive( bool embed )
  {
    m_shouldEmbedCachedFiles = embed;
  }

  AGX_FORCE_INLINE bool TerrainCache::getEmbedCacheFilesInArhive() const
  {
    return m_shouldEmbedCachedFiles;
  }



}

