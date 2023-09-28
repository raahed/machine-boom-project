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

#ifndef AGX_PHYSISCS_HIERARCHICAL_GRID_UTIL_H
#define AGX_PHYSISCS_HIERARCHICAL_GRID_UTIL_H

// #include <agx/Physics/HierarchicalGrid/ContactZoneEntity.h>

#include <agx/Vec3.h>
#include <agx/Vec2.h>
#include <agx/Math.h>
#include <agx/debug.h>
#include <agxData/LocalVector.h>
#include <agx/Physics/HierarchicalGrid/CellEntity.h>


#define NUM_ITERATION_ZONE_JOBS 2

namespace agx { namespace Physics { namespace HierarchicalGrid {

  /**
  The following ordering of neighbor coordinates is used for all cell/zone interactions
    
    == 3D ================================
    
   -1, -1, -1
   -1, -1,  0
   -1, -1,  1
   -1,  0, -1
   -1,  0,  0
   -1,  0,  1
   -1,  1, -1
   -1,  1,  0
   -1,  1,  1
  
    0, -1, -1
    0, -1,  0
    0, -1,  1
    0,  0, -1
  
    0,  0,  0 // Self
  
    0,  0,  1
    0,  1, -1
    0,  1,  0
    0,  1,  1
  
    1, -1, -1
    1, -1,  0
    1, -1,  1
    1,  0, -1
    1,  0,  0
    1,  0,  1
    1,  1, -1
    1,  1,  0
    1,  1,  1
    
    == 2D =================================
    
    -1, -1
    -1,  0
    -1,  1
     0,  1
     0,  0 // Self
     0, -1
     1, -1
     1,  0
     1,  1
    
    =======================================
  */
  
  /**
  \return The neighbor offset vector for a specified neighbor index.
  */
  AGXPHYSICS_EXPORT const Vec3i& getNeighborOffset(Index index);

  /**
  \return The neighbor offset vector for a specified neighbor index, in 2d
  */
  AGXPHYSICS_EXPORT const Vec2i& getNeighborOffset2d(Index index);
  
  
  ///\return direct access to the neighbor table
  AGXPHYSICS_EXPORT const Vec3i *getNeighborTable();
  AGXPHYSICS_EXPORT const Vec2i *getNeighborTable2d();
  
  /**
  \return The neighbor index given a neighbor offset vector.
  */
  AGXPHYSICS_EXPORT Index calculateNeighborIndex(const Vec3i& offset);
  AGXPHYSICS_EXPORT Index calculateNeighborIndex(const Vec2i& offset);

  /**
  \return true if the specified offset is a neighbor offset.
  */
  AGXPHYSICS_EXPORT bool isNeighborOffset(const Vec3i& offset);
  AGXPHYSICS_EXPORT bool isNeighborOffset(const Vec2i& offset);

  /**
  Return the parent cell N tiers above the specified cell.
  */
  AGXPHYSICS_EXPORT Index getParentCell(CellData& cell, Index cellIndex, UInt level);
  
  /**
  Align cell indices to the same tier
  */
  AGXPHYSICS_EXPORT Index alignCellTiers(CellData& cell, Index tier1, Index tier2, Index& cellIndex1, Index& cellIndex2);
  
  /**
  Calculates which one of the 27 internal zones the cell is part of.
  */ 
  AGXPHYSICS_EXPORT UInt8 calculateZoneType(Vec3i id, UInt tierAccumulation);

  /**
  The 27 internal zones are following categories (box features) and counts:
  Internal: 1
  Face: 6
  Edge: 12
  Corner: 8
  */
  enum ZoneCategory
  {
    ZONE_INTERNAL,
    ZONE_FACE,
    ZONE_EDGE,
    ZONE_CORNER
  };
  
  /**
  \return The category for one of the 27 zone types.
  */
  AGXPHYSICS_EXPORT ZoneCategory getZoneCategory(Index zoneType);
  
  /**
  \return The name of a zone category.
  */
  AGXPHYSICS_EXPORT const char *getZoneCategoryName(Index category);

  
  /**
  \return The zone category for a specified zone id.
  */
  AGXPHYSICS_EXPORT ZoneCategory calculateZoneCategory(const Vec3i& zoneId);
  
  
  /**
  Corners and edges are extended to make all edges independent of each other, and the same for all faces.
  */
  AGXPHYSICS_EXPORT Vec3i calculateBoundaryZoneId(CellData& cell, Index cellIndex1, Index cellIndex2, UInt tierAccumulation);
  

  struct AGXPHYSICS_EXPORT NeighborCellPath
  {
    UInt8 neighborParentIndex;
    UInt8 neighborChildIndex;
  };


  AGXPHYSICS_EXPORT const NeighborCellPath *getNeighborPathTable(size_t childIndex);
  AGXPHYSICS_EXPORT const UInt8 *getNeighborPathOrderTable(size_t childIndex);


  template <typename ZoneDataT>
  UInt countNumContacts(ZoneDataT& contactZone, Index zoneIndex)
  {
    return contactZone.particleParticleContacts[zoneIndex].size() + 
           contactZone.particleGeometryContacts[zoneIndex].size() +
           contactZone.geometryGeometryContacts[zoneIndex].size();
  }
  
  template <typename ZonePtrT>
  UInt countNumContacts(ZonePtrT zone)
  {
    agxData::EntityStorage *storage = zone.getStorage();
    return countNumContacts(*storage->getData<typename ZonePtrT::DataType>(), zone.calculateIndex());
  }





  /* Implementation */
  extern const ZoneCategory __private_categoryTable[];
  extern const Vec3i __private_neighborTable[];
  extern const Vec2i __private_neighborTable_2D[];
  
  AGX_FORCE_INLINE const Vec3i *getNeighborTable() { return __private_neighborTable; }
  AGX_FORCE_INLINE const Vec2i *getNeighborTable2d() { return __private_neighborTable_2D; }
  
  AGX_FORCE_INLINE const Vec3i& getNeighborOffset(Index index)
  {
    agxAssert(index < 27);
    return __private_neighborTable[index];
  }

  AGX_FORCE_INLINE const Vec2i& getNeighborOffset2d(Index index)
  {
    agxAssert(index < 9);
    return __private_neighborTable_2D[index];
  }
  
  /////////////////////////////////////////////////////////////////
  
  AGX_FORCE_INLINE Index calculateNeighborIndex(const Vec3i& offset)
  {
    agxAssert(isNeighborOffset(offset));
    Vec3i shiftedOffset = offset + 1;
    Index neighborIndex = (Index)((shiftedOffset[0] * 9) + (shiftedOffset[1] * 3) + shiftedOffset[2]);
    agxAssert(getNeighborOffset(neighborIndex) == offset);
    return neighborIndex;
  }

  AGX_FORCE_INLINE Index calculateNeighborIndex(const Vec2i& offset)
  {
    agxAssert(isNeighborOffset(offset));
    Vec2i shiftedOffset = offset + 1;
    Index neighborIndex = (Index)((shiftedOffset[0] * 2) + shiftedOffset[1]);
    agxAssert(getNeighborOffset2d(neighborIndex) == offset);
    return neighborIndex;
  }

  /////////////////////////////////////////////////////////////////
  
  AGX_FORCE_INLINE bool isNeighborOffset(const Vec3i& offset)
  {
    return (UInt)(offset[0]+1) <= 2 && (UInt)(offset[1]+1) <= 2 && (UInt)(offset[2]+1) <= 2;
  }

  AGX_FORCE_INLINE bool isNeighborOffset(const Vec2i& offset)
  {
    return (UInt)(offset[0]+1) <= 2 && (UInt)(offset[1]+1) <= 2;
  }
  
  /////////////////////////////////////////////////////////////////
  
  AGX_FORCE_INLINE Index getParentCell(CellData& cell, Index cellIndex, UInt level)
  {
    for (UInt i = 0; i < level; ++i)
    {
      if (cell.parent[cellIndex] == InvalidIndex)
        break;

      cellIndex = cell.parent[cellIndex];
    }

    return cellIndex;
  }
  
  /*
  If the particles belong to different tiers, use the lower tier
  */
  AGX_FORCE_INLINE Index alignCellTiers(CellData& cell, Index tier1, Index tier2, Index& cellIndex1, Index& cellIndex2)
  {
    Index minCellIndex = InvalidIndex;

    if (tier1 > tier2)
    {
      minCellIndex = cellIndex1;
      cellIndex1 = getParentCell(cell, cellIndex1, tier1 - tier2);
    }
    else
    {
      minCellIndex = cellIndex2;
      cellIndex2 = getParentCell(cell, cellIndex2, tier2 - tier1);
    }
    
    return minCellIndex;
  }
  
  
  // Generate a bitmask with numBits LSB bits set
  AGX_FORCE_INLINE UInt generateBitMask(UInt numBits)
  {
    return (UInt(1) << numBits) - UInt(1);
  }
  
  
  
  AGX_FORCE_INLINE ZoneCategory getZoneCategory(Index zoneType)
  {
    agxAssert(zoneType < 27);
    return __private_categoryTable[zoneType];
  }
  
  AGX_FORCE_INLINE ZoneCategory calculateZoneCategory(const Vec3i& zoneId)
  {
    Vec3i offset = Vec3i(zoneId[0] & 0x1, zoneId[1] & 0x1, zoneId[2] & 0x1);
    return getZoneCategory(calculateNeighborIndex(offset));
  }
  
  
  AGX_FORCE_INLINE Vec3i calculateBoundaryZoneId(CellData& cell, Index cellIndex1, Index cellIndex2, UInt tierAccumulation)
  {
    ZoneCategory zoneType1 = getZoneCategory(cell.zoneType[cellIndex1]);
    ZoneCategory zoneType2 = getZoneCategory(cell.zoneType[cellIndex2]);

    // Extended zones, using the dominant zone type
    Index dominantCellIndex = zoneType1 > zoneType2 ? cellIndex1 : cellIndex2;

    Vec3i zoneId1 = cell.id[dominantCellIndex];
    Vec3i zoneId2 = cell.id[dominantCellIndex] + getNeighborOffset(cell.zoneType[dominantCellIndex]);

    zoneId1[0] >>= tierAccumulation;
    zoneId1[1] >>= tierAccumulation;
    zoneId1[2] >>= tierAccumulation;

    zoneId2[0] >>= tierAccumulation;
    zoneId2[1] >>= tierAccumulation;
    zoneId2[2] >>= tierAccumulation;

    Vec3i boundaryId = zoneId1 + zoneId2;
    
    // std::cout << "dominant: " << getZoneCategoryName(foo1) << ", extended: " << getZoneCategoryName(foo2) << std::endl;
    // std::cout << "zone: " << getZoneCategoryName(calculateZoneCategory(boundaryId)) << std::endl;
    

    ////////////////////////////
    
    UInt cornerShift = UInt(1) << (tierAccumulation-1);
    Vec3i cornerId1 = cell.id[cellIndex1] + cornerShift;
    Vec3i cornerId2 = cell.id[cellIndex2] + cornerShift;
    
    cornerId1[0] >>= tierAccumulation;
    cornerId1[1] >>= tierAccumulation;
    cornerId1[2] >>= tierAccumulation;

    cornerId2[0] >>= tierAccumulation;
    cornerId2[1] >>= tierAccumulation;
    cornerId2[2] >>= tierAccumulation;
    
    if (cornerId1 == cornerId2)
    {
      // std::cout << "boundaryId: " << boundaryId << ", extended corner: " << (cornerId1 + cornerId2 - 1) << std::endl;
      return cornerId1 + cornerId2 - 1;
    }
    else
    {
      return boundaryId;
    }
    
#if 0
    ////////////////////////////
    
    Vec3i offset = cell.id[cellIndex1] - cell.id[cellIndex2];
    
    if ((offset[0] & offset[1] & offset[2] & 0x1) && zoneType1 == ZONE_EDGE && zoneType2 == ZONE_EDGE && cell.zoneType[cellIndex1] != cell.zoneType[cellIndex2] * -1) 
    {
      std::cout << "detected special edge/edge diagonal, cell1: " << cell.id[cellIndex1] << ", cell2: " << cell.id[cellIndex2] << std::endl;
      
      const Vec3i& dir1 = getNeighborOffset(cell.zoneType[cellIndex1]);
      const Vec3i& dir2 = getNeighborOffset(cell.zoneType[cellIndex2]);
      
      Vec3i cornerId1 = cell.id[cellIndex1] + dir1 + dir2;
      Vec3i cornerId2 = cell.id[cellIndex2] + dir1 + dir2;
      
      // agxAssert(getZoneCategory(calculateZoneType(cornerId1, tierAccumulation)) == ZONE_CORNER);
      // agxAssert(getZoneCategory(calculateZoneType(cornerId2, tierAccumulation)) == ZONE_CORNER);
      
      cornerId1[0] >>= tierAccumulation;
      cornerId1[1] >>= tierAccumulation;
      cornerId1[2] >>= tierAccumulation;

      cornerId2[0] >>= tierAccumulation;
      cornerId2[1] >>= tierAccumulation;
      cornerId2[2] >>= tierAccumulation;

      return cornerId1 + cornerId2;
    }
    
    // if (zoneType1 <= ZONE_FACE || zoneType2 <= ZONE_FACE)
    if (false)
    {
      Index parentIndex1 = getParentCell(cell, cellIndex1, tierAccumulation);
      Index parentIndex2 = getParentCell(cell, cellIndex2, tierAccumulation);
      return cell.id[parentIndex1] + cell.id[parentIndex2];
    }
    ////////////////////////////

    return boundaryId;
#endif
  }
  
  AGX_FORCE_INLINE UInt8 calculateZoneType(Vec3i id, UInt tierAccumulation)
  {
    Int mask = (Int)generateBitMask(tierAccumulation);

    id[0] &= mask;
    id[1] &= mask;
    id[2] &= mask;

    UInt zoneType = 0;
    zoneType += (id[0] == mask ? 18 : (id[0] == 0 ? 0 : 9));
    zoneType += (id[1] == mask ? 6 : (id[1] == 0 ? 0 : 3));
    zoneType += (id[2] == mask ? 2 : (id[2] == 0 ? 0 : 1));

    agxAssert(zoneType < 27);
    return (UInt8)zoneType;
  }
  
  
  //////////////////////////////////////////////////////////////////////////////////
  
  AGXPHYSICS_EXPORT extern NeighborCellPath __private_NeighborPathTable[8][32];
  AGXPHYSICS_EXPORT extern UInt8 __private_NeighborPathOrderTable[8][32];
  
  AGX_FORCE_INLINE const NeighborCellPath *getNeighborPathTable(size_t childIndex)
  {
    agxAssert(childIndex < 8);
    if (childIndex < 8)
      return __private_NeighborPathTable[childIndex];
    else
      return nullptr;
  }
  
  AGX_FORCE_INLINE const UInt8 *getNeighborPathOrderTable(size_t childIndex)
  {
    agxAssert(childIndex < 8);
    if (childIndex < 8)
      return __private_NeighborPathOrderTable[childIndex];
    else
      return nullptr;
  }
  
  
}}}


#endif /* AGX_PHYSISCS_HIERARCHICAL_GRID_UTIL_H */
