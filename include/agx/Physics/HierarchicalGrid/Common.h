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

#ifndef AGX_HIERARCHICAL_GRID_COMMON_H
#define AGX_HIERARCHICAL_GRID_COMMON_H

#include <agx/Vec2.h>

#include <agx/HashTable.h>
#include <agx/Physics/HierarchicalGrid/CellEntity.h>
#include <agx/Physics/HierarchicalGrid/Cell2DEntity.h>
#include <agx/Physics/HierarchicalGrid/ContactZoneEntity.h>
#include <agx/Physics/HierarchicalGrid/ContactZoneDependencyEntity.h>

#include <agxData/LocalVector.h>

#include "Util.h"
#include "GridTables.h"

#define POST_FILTER_OBB_TESTS 1

#define GRID_SUBSYSTEM_GEOMETRY 0
#define GRID_SUBSYSTEM_PARTICLE 1
#define GRID_NUM_SUBSYSTEMS 2

namespace agx
{

  enum CellState
  {
    CELL_DEAD,
    CELL_UNINITIALIZED,
    CELL_ACTIVE
  };

  /////////////////////////////////////////////////////////////////

  struct LocalOverlap
  {
    AGX_FORCE_INLINE LocalOverlap() {}
    AGX_FORCE_INLINE LocalOverlap(UInt32 object1In, UInt32 object2In) { this->set(object1In, object2In); }
    AGX_FORCE_INLINE void set(UInt32 object1In, UInt32 object2In) { this->object1 = object1In; this->object2 = object2In; }
    
    UInt32 object1;
    UInt32 object2;
  };

  typedef agxData::LocalVector<LocalOverlap> LocalOverlapVector;

  /////////////////////////////////////////////////////////////////

  struct CellSortEntry
  {
    inline CellSortEntry() {}
    inline CellSortEntry(UInt64 _linearId, UInt8 _tier, UInt32 _cellIndex) : linearId(_linearId), cellIndex(_cellIndex), tier(_tier) {}
    
    UInt64 linearId;
    UInt32 cellIndex;
    UInt8 tier;
  };
  
  AGX_FORCE_INLINE bool compareCellSortEntries(const CellSortEntry& p1, const CellSortEntry& p2)
  {
    // Tier as primary sort key, use linearId if same tier
    return p1.tier == p2.tier ? (p1.linearId < p2.linearId) : (p1.tier < p2.tier);
  }
  

  AGX_FORCE_INLINE Vec3 cornerOffsets2D( Real32 radius, size_t corner )
  {
    static Vec3 cornerOffsets[4] = {
      Vec3(-radius,  radius, 0.0 ), Vec3( radius,  radius, 0),
      Vec3( radius, -radius, 0.0 ), Vec3(-radius, -radius, 0)};
    return cornerOffsets[corner];
  }

  template< typename CellIdT>
  AGX_FORCE_INLINE CellIdT getCornerId( size_t /*corner*/, Real32 /*radius*/, Vec3f /*centerPosition*/, Real32 /*invCellSize*/ )
  {
    agxAbortN( "Need getCornerId template specialization for %s.", typeid(CellIdT).name() );
    return CellIdT();
  }

  template<>
  AGX_FORCE_INLINE Vec2i getCornerId( size_t corner, Real32 radius, Vec3f centerPosition, Real32 invCellSize )
  {
    Vec3f cornerPosition = centerPosition + (Vec3f)cornerOffsets2D( radius, corner );
    return agx::calculateCellId( cornerPosition.x(), cornerPosition.y(), invCellSize );
  }

  /////////////////////////////////////////////////////////////////
  
  // typedef HashTable< Vec3i, agx::UInt32 > GridCellTable;
  // typedef HashTable< Vec2i, agx::UInt32 > GridCell2DTable;

  typedef CellTable< Vec3i > GridCellTable;
  typedef CellTable< Vec2i > GridCell2DTable;

  typedef agx::HashTable< Vec3i, agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZoneTable;
  typedef agx::HashTable< UInt64, agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr > ZoneDependencyTable;

  /////////////////////////////////////////////////////////////////

  // Special wrappers due to 2d implementation using 3d particle positions
  AGX_FORCE_INLINE void calculateCellId(const Vec3& pos, Real invCellSize, Vec3i& id)
  {
    id = agx::calculateCellId(pos, invCellSize);
  }
  
  AGX_FORCE_INLINE void calculateCellId(const Vec3& pos, Real invCellSize, Vec2i& id)
  {
    id = agx::calculateCellId(pos[0], pos[1], invCellSize);
  }
  
  /////////////////////////////////////////////////////////////////
  
  AGX_FORCE_INLINE UInt calculateCellChildIndex(const Vec3i& cellId)
  {
    return (cellId[0] & 0x1) | ((cellId[1] & 0x1) << 1) | ((cellId[2] & 0x1) << 2);
  }

  AGX_FORCE_INLINE UInt calculateCellChildIndex(const Vec2i& cellId)
  {
    return (cellId[0] & 0x1) | ((cellId[1] & 0x1) << 1);
  }
  
  
  /////////////////////////////////////////////////////////////////

  AGX_FORCE_INLINE UInt calculateRootCellIndex(const Vec3i& cellId)
  {
    return (UInt(cellId[0] >= 0) & 0x1) | ((UInt(cellId[1] >= 0) & 0x1) << 1) | ((UInt(cellId[2] >= 0) & 0x1) << 2);
  }

  AGX_FORCE_INLINE UInt calculateRootCellIndex(const Vec2i& cellId)
  {
    return (UInt(cellId[0] >= 0) & 0x1) | ((UInt(cellId[1] >= 0) & 0x1) << 1);
  }

  /////////////////////////////////////////////////////////////////
  
  AGX_FORCE_INLINE Vec3i calculateChildCellId(const Vec3i& cellId, UInt8 childIndex)
  {
    Vec3i result = cellId * 2;
    if (childIndex & 0x1) result[0]++;
    if (childIndex & 0x2) result[1]++;
    if (childIndex & 0x4) result[2]++;
    return result;
  }
  
  AGX_FORCE_INLINE Vec2i calculateChildCellId(const Vec2i& cellId, UInt8 childIndex)
  {
    Vec2i result = cellId * 2;
    if (childIndex & 0x1) result[0]++;
    if (childIndex & 0x2) result[1]++;
    return result;
  }
  
  /////////////////////////////////////////////////////////////////

  AGX_FORCE_INLINE Vec3i calculateNonNegativeId(const Vec3i& cellId, UInt8 /*tier*/)
  {
    // TODO Use offset to make sure id becomes positive, some power of two (bit shift) based on tier id
    return cellId;
  }
  
  AGX_FORCE_INLINE Vec2i calculateNonNegativeId(const Vec2i& cellId, UInt8 /*tier*/)
  {
    // TODO Use offset to make sure id becomes positive, some power of two (bit shift) based on tier id
    return cellId;
  }
  
  /////////////////////////////////////////////////////////////////

  AGX_FORCE_INLINE Vec3i calculateParentCellId(const Vec3i& cellId)
  {
    Vec3i parentId = cellId / 2;
    
    if (cellId[0] < 0 && (cellId[0] & 0x1)) parentId[0]--;
    if (cellId[1] < 0 && (cellId[1] & 0x1)) parentId[1]--;
    if (cellId[2] < 0 && (cellId[2] & 0x1)) parentId[2]--;
    
    return parentId;
  }
  
  AGX_FORCE_INLINE Vec2i calculateParentCellId(const Vec2i& cellId)
  {
    Vec2i parentId = cellId / 2;
    
    if (cellId[0] < 0 && (cellId[0] & 0x1)) parentId[0]--;
    if (cellId[1] < 0 && (cellId[1] & 0x1)) parentId[1]--;
    
    return parentId;
  }

  /////////////////////////////////////////////////////////////////

  /**
   * Calculates the tier where the given object size will fit.
   * \return If a suitable tier is found, then its index is returned.
   *         If no grid is found, then InvalidIndex is returned.
   */
  AGX_FORCE_INLINE UInt8 calculateTier( Real minCellSize, const agxData::Array<Real>& gridTier_size )
  {
    size_t numTiers = gridTier_size.size();
    if ( numTiers == 0 )
      return InvalidIndex;

    UInt8 tier;
  #if 0
    // Experimental log2 way of finding the correct tier. Might be slower than
    // the short loop below and I'm worried about precision problems. Also, if
    // the required minimum cell size is larger than the currently largest
    // tier, then this code will (probably) return the smallest (numTiers-1)
    // tier.
    // Don't think we should use this. Comments?
    tier = (UInt8)( log2(largestCellSize) - log2(minCellSize) );
    if ( tier >= numTiers )
      tier = numTiers-1;
  #else
    // Safe loop for finding the correct tier. Searches down the hierarchy.
    // May produce the largest tier even if even it is too small. Also, 
    // will produce the smallest tier even if the object really should be
    // in a even smaller tier.
    for ( tier = 0; tier < numTiers-1; ++tier )
    {
      // Test if the NEXT tier would be too small.
      if ( gridTier_size[ tier+1 ] < minCellSize  )
        break;
    }
  #endif

    // Make sure we found a tier that isn't too small.
    if ( minCellSize > gridTier_size[ tier ] )
      tier = InvalidIndex;

    // Make sure the object shouldn't fit better in a even smaller tier.
    // NOTE: Important that there is only one valid tier for each requested cell size.
    // We must therefore use '<=' instead of '<' to prevent ambiguous tier assignments.
    else if ( Real(2) * minCellSize <= gridTier_size[ tier ] )
      tier = InvalidIndex;
    
    return tier;
  }
}


#endif /* AGX_HIERARCHICAL_GRID_COMMON_H */
