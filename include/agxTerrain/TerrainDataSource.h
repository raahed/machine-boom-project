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
#include <agx/Referenced.h>
#include <agxStream/Serializable.h>
#include <agx/ThreadSynchronization.h>

#include <agxTerrain/export.h>
#include <agxTerrain/TileSpecification.h>

#include <agxCollide/Geometry.h>
#include <agxCollide/Line.h>


namespace agxTerrain
{

  AGX_DECLARE_POINTER_TYPES( TerrainDataSource );

  /**
  Abstract interface for requesting height data for a Terrain tile.
  */
  class AGXTERRAIN_EXPORT TerrainDataSource : public agx::Referenced, public agxStream::Serializable
  {

    public:
      /**
      Datatype used for returning height values
      */
      typedef agx::RealVector TerrainHeightType;


      /**
      Return height data for requested tile according to the TileSpecification.

      The returned vector should have ts.getTileResolution squared number of height values.
      If the TerrainDataSource is not able to provide data, an empty vector can be returned.

      Upon no data returned, the TerrainCache will reply to the TerrainPager with the requested TileId id
      but having no terrain instance signaling that the request was processed but could not be fullfilled.
      In this scenario, there will be an empty tile where it should have been a terrain instance and the
      same tile will be requested again on the next stepForward.

      \param ts - The specification of how new tiles are created by the data source
      \param id - The id or position of the tile relative to other paged terrain tiles

      \return A vector containing the height data for the new tile
      */
      virtual TerrainHeightType fetchTerrainTile( const TileSpecification& ts, agxTerrain::TileId id ) = 0;

      /**
      Perform raycasting against the terrain. This can be called by the TerrainPager
      when TerrainPager::raycast is used and no active tile is hit.

      Default implementation returns false. Classes inheriting from TerrainDataSource
      can overload this method and add support if possible.

      \param start - The start position of the raycast
      \param end - The end position of the raycast
      \param raycastResult - A reference to a Vec3 in which to store the point where the raycast hits the data source,

      \return true if terrain is hit and raycastResult is then updated with position for the hit.
      */
      virtual bool raycast( agx::Vec3 start, agx::Vec3 end, agx::Vec3& raycastResult );

      // Serialization
      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxTerrain::TerrainDataSource );
      virtual void store( agxStream::OutputArchive& out ) const override;
      virtual void restore( agxStream::InputArchive& in ) override;
  };



  /**
  This class performs raycasting against the source geometry to get height
  data for tiles.

  The reference position for the grid tiles defined by the TileSpecification is used
  to determine where to sample the source geometry. This defines the local
  X- and Y-coordinates. The local height values are not known and depends
  on the source geometry.

  Therefor a sampleRayLength should be defined so that a ray from
  0.5 * sampleRayLength to -0.5 * sampleRayLength hits the source
  geometry.

  Changing the ray length will only affect future tiles that will be sampled,
  not any previous requested tiles that are cached.


  The TerrainRasterizer will only return height data for a terrain tile
  when it was able to sample the entire tile. If parts of the tile or
  the entire tile is lacking source geometry, no data will be returned
  and the TerrainPager will not be able to page in a terrain instace for that
  region.
  */
  class AGXTERRAIN_EXPORT TerrainRasterizer : public TerrainDataSource
  {
    public:

      /**
      Constructor
      */
      TerrainRasterizer( agx::Real sampleRayLength = 1000.0 );

      /**
      \return Length of ray used for ray casting to get height data
      */
      agx::Real getRayLength() const;


      /**
      Set the length of the ray
      */
      void setRayLength( agx::Real length );


      /**
      The rasterizer can have one or more source geometries which
      are used when creating the terrain tiles.

      These geometries are not required to be part of a agxSDK::Simulation or
      agxCollide::Space. They will be used directly by the rasterizer.

      It is required that they are positioned correctly in the world and it should
      not be repositioned after it has been given to the rasterizer.
      */
      bool addSourceGeometry( agxCollide::Geometry* geom );

      /**
      Removes a geometry from the TerrainRasterizer.

      Areas that has been sampled which hit the geometry being removed
      will not be resampled and cached heights will still be used.
      */
      bool removeSourceGeometry( agxCollide::Geometry* geom );

      /**
      Return height data for requested tile according to the TileSpecification.
      \param ts - The specification of how new tiles are created by the data source
      \param id - The id or position of the tile relative to other paged terrain tiles
      \return A vector containing the height data for the new tile
      */
      virtual TerrainHeightType fetchTerrainTile( const TileSpecification& ts, agxTerrain::TileId id ) override;

      /**
      Overload of TerrainDataSource::raycast for testing against source geometry
      \param start - The start position of the raycast
      \param end - The end position of the raycast
      \param raycastResult - A reference to a Vec3 in which to store the point where the raycast hits the data source,

      \return true if terrain is hit and raycastResult is then updated with position for the hit.
      */
      virtual bool raycast( agx::Vec3 start, agx::Vec3 end, agx::Vec3& raycastResult ) override;


      // Serialization methods
      AGXSTREAM_DECLARE_SERIALIZABLE( agxTerrain::TerrainRasterizer );


    protected:
      ~TerrainRasterizer();

    private:
      agxCollide::GeometryRefVector m_sourceGeometry;
      agxCollide::LineRef m_line;
      agx::Real m_sampleLength;

      agx::ReaderWriterLock m_rwLock;
  };

  /**
  This class provides a concrete, serializable base class which classes written
  in other languages can inherit from to create data sources. This class NOT meant 
  to be used in C++ code or instantiated, only to provide an interface to child classes.
  
  The need for this class arises from the TerrainDataSource class being an abstract class 
  which in turn is not serializable, meaning that simulations containing external data source classes 
  inheriting from TerrainDataSource fails during serialization.
  
  The solution is to instead inherit from this class which disables serialization of the data source.
  */
  class AGXTERRAIN_EXPORT ExternalTerrainDataSource : public TerrainDataSource
  {
    public:
      /**
      Constructor. Note that class should not be instantiated, only inherited
      */
      ExternalTerrainDataSource( );

      /**
      Overload of TerrainDataSource::fetchTerrainTile for fetching terrain data
      \param ts - The specification of how new tiles are created by the data source
      \param id - The id or position of the tile relative to other paged terrain tiles
      \return A vector containing the height data for the new tile
      */
      virtual TerrainHeightType fetchTerrainTile( const TileSpecification& ts, agxTerrain::TileId id ) override;

      /**
      Overload of TerrainDataSource::raycast for performing raycast against the data source
      \param start - The start position of the raycast
      \param end - The end position of the raycast
      \param raycastResult - A reference to a Vec3 in which to store the point where the raycast hits the data source,

      \return true if terrain is hit and raycastResult is then updated with position for the hit.
      */
      virtual bool raycast( agx::Vec3 start, agx::Vec3 end, agx::Vec3& raycastResult ) override;

      // Serialization methods
      AGXSTREAM_DECLARE_SERIALIZABLE( agxTerrain::ExternalTerrainDataSource );

    protected:
      ~ExternalTerrainDataSource() {};

    private:
      // When ExternalTerrainDataSource is loaded from the archive the base class is used by design. This means that
      // ExternalTerrainDataSource::fetchTerrainTile/raycast will be called. This would normaly result in a warning.
      // This flag keeps track of if this instance was loaded from an archive and, if so, supresses warnings.
      bool m_loadedFromArchive;
  };

  AGX_FORCE_INLINE agx::Real TerrainRasterizer::getRayLength() const
  {
    return m_sampleLength;
  }

  AGX_FORCE_INLINE void TerrainRasterizer::setRayLength( agx::Real length )
  {
    m_sampleLength = length;
  }

}
