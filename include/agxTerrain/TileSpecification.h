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
#include <agxTerrain/export.h>

#include <agx/Integer.h>
#include <agx/Vec3.h>
#include <agx/Quat.h>
#include <agx/AffineMatrix4x4.h>
#include <agx/HashFunction.h>

#include <agxStream/StorageStream.h>

#include <utility>

namespace agx {

  typedef std::pair< Real, Real > RealPair;

}

namespace agxTerrain
{
  class Terrain;


  /**
  A TileId is a 2D-coordinate that specifies
  where a in grid a tile is located. Together with a TileSpecification,
  a location in world coordinates can be calculated.
  */
  typedef agx::Vec2i32 TileId;


  /**
  TileModifications are used to provide a delta for paged terrain changes.
  */
  class AGXTERRAIN_EXPORT TileModification
  {
    public:
      agxTerrain::TileId tileId;
      agx::Vec2i32       vertexIndex;
      float              height;
  };

  typedef agx::VectorPOD<TileModification>   TileModificationVector;


  /**
  A TileSpecification determines where in the world a 2D TileId is located
  and contains utility methods for coordinate transforms.
  */
  class AGXTERRAIN_EXPORT TileSpecification
  {
    public:

      /**
      Constructor
      \param refPoint    Reference point for 2d-grid used to positioning terrain tiles
      \param refRotation Reference rotation for 2d-grid
      \param tileSize    Size in meter for a grid tile
      \param marginSize  Number of height values that should overlap on neighbor tiles
      \param resolution  Resolution for the tiles
      \param maxDepth    Maximum excavation depth for a tile
      */
      TileSpecification( agx::Vec3 refPoint, agx::Quat refRotation,
                         agx::Real tileSize, size_t marginSize,
                         size_t resolution, agx::Real maxDepth );

      TileSpecification();

      /**
      \return The world reference point
      */
      agx::Vec3 getReferencePoint() const;

      /**
      \return Reference rotation for Z-axis
      */
      agx::Quat getReferenceRotation() const;

      /**
      The full size of a tile
      \return The side length for a square tile
      */
      agx::Real getTileSize() const;

      /**
      \return The margin size
      */
      size_t getTileMarginSize() const;


      /**
      \return How many height values that are used per dimension for tile
      */
      size_t getTileResolution() const;

      /**
      \return Maximum excavation depth for a tile
      */
      agx::Real getMaximumDepth() const;



      /**
      Return position in coordinate system defined by the XY-plane with local Z upwards
      */
      agx::Vec3 convertWorldCoordinateToTilePosition( agx::Vec3 worldPos ) const;

      /**
      \return The TileId which contains the position pos
      */
      agxTerrain::TileId convertWorldCoordinateToTileId( agx::Vec3 worldPos ) const;

      /**
      Convert a local position in a tile to world coordinates
      \param id Which tile
      \param tilePos local position in tile
      \return World position for a 2D position in a given tile
      */
      agx::Vec3 convertTilePositionToWorld( agxTerrain::TileId id, agx::Vec2 tilePos ) const;


      /**
      The 2D version of this method is the same as calling this method with agx::Vec3(x,y,0)
      \param id Which tile
      \param tilePos local position in tile
      \return World position for a 3D position in a given tile.
      */
      agx::Vec3 convertTilePositionToWorld( agxTerrain::TileId id, agx::Vec3 tilePos ) const;


      /**
      Return the transform that can be used to place the tile id correctly.
      */
      agx::AffineMatrix4x4 getTileTransform( agxTerrain::TileId id, agx::Real zPos = 0 ) const;

      /**
      Return a transform that can be used to position something at localOffset in the tile with id
      */
      agx::AffineMatrix4x4 getTileTransform( agxTerrain::TileId id, agx::Vec3 localOffset ) const;

      /**
      Return the local center position for tile id in the plane with tiles
      */
      agx::Vec3 getTileCenter( agxTerrain::TileId id ) const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      void store(agxStream::OutputArchive& out) const;
      void restore(agxStream::InputArchive& in);
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      agx::Vec3 m_referencePoint;
      agx::Quat m_referenceRotation;

      agx::Real m_tileSize;
      size_t m_tileMarginSize;
      size_t    m_tileResolution;

      agx::Real m_maximumDepth;

      agx::AffineMatrix4x4 m_gridToWorld;
      agx::AffineMatrix4x4 m_worldToGrid;
  };




  AGX_FORCE_INLINE agx::Vec3 TileSpecification::getReferencePoint() const
  {
    return m_referencePoint;
  }

  AGX_FORCE_INLINE agx::Quat TileSpecification::getReferenceRotation() const
  {
    return m_referenceRotation;
  }

  AGX_FORCE_INLINE agx::Real TileSpecification::getTileSize() const
  {
    return m_tileSize;
  }

  AGX_FORCE_INLINE size_t TileSpecification::getTileMarginSize() const
  {
    return m_tileMarginSize;
  }

  AGX_FORCE_INLINE size_t TileSpecification::getTileResolution() const
  {
    return m_tileResolution;
  }

  AGX_FORCE_INLINE agx::Real TileSpecification::getMaximumDepth() const
  {
    return m_maximumDepth;
  }

  AGX_FORCE_INLINE agx::Vec3 TileSpecification::convertTilePositionToWorld( agxTerrain::TileId id, agx::Vec2 tilePos ) const
  {
    return convertTilePositionToWorld( id, agx::Vec3( tilePos, 0.0 ) );
  }


  AGX_FORCE_INLINE agx::Vec3 TileSpecification::convertWorldCoordinateToTilePosition( agx::Vec3 worldPos ) const
  {
    return worldPos * m_worldToGrid;
  }



  AGX_FORCE_INLINE agx::AffineMatrix4x4 TileSpecification::getTileTransform( agxTerrain::TileId id, agx::Real zPos ) const
  {
    agx::Vec3 center = getTileCenter( id );
    center[2] += zPos;

    return agx::AffineMatrix4x4::translate( center ) * m_gridToWorld;
  }


  AGX_FORCE_INLINE agx::AffineMatrix4x4 TileSpecification::getTileTransform( agxTerrain::TileId id, agx::Vec3 localOffset ) const
  {
    agx::Vec3 center = getTileCenter( id );

    return agx::AffineMatrix4x4::translate( center + localOffset ) * m_gridToWorld;

  }
}

