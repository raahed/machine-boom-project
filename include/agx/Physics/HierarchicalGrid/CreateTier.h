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

#ifndef AGX_CREATETIER_H
#define AGX_CREATETIER_H

#include "Common.h"

namespace agx
{
  // Used for temporary storage of new grid tiers.
  template <typename GridCellTableT>
  struct GridTier
  {
    GridTier( Real cellSize, GridCellTableT* cellTable ) : cellSize(cellSize), cellTable(cellTable)
    {
    }

    Real cellSize;
    GridCellTableT* cellTable;
  };

  template <typename GridCellTableT>
  AGX_FORCE_INLINE Real calculateBoundingRadius(Real /*tierSize*/)
  {
    agxAbort();
    return 0;
  }

  template <>
  AGX_FORCE_INLINE Real calculateBoundingRadius<agx::GridCellTable>(Real tierSize)
  {
    Real halfSize = tierSize * Real(0.5);
    return std::sqrt(halfSize * halfSize * Real(3)) + halfSize; // Add halfSize to include objects reaching outside cell
  }

  template <>
  AGX_FORCE_INLINE Real calculateBoundingRadius<agx::GridCell2DTable>(Real tierSize)
  {
    Real halfSize = tierSize * Real(0.5);
    return std::sqrt(halfSize * halfSize * Real(2)) + halfSize; // Add halfSize to include objects reaching outside cell
  }

  template <typename GridTierDataT, typename GridCellTableT>
  AGX_FORCE_INLINE void createTier
  (
    size_t tier,
    Real tierSize,
    agxData::Buffer *cellIdBuffer,
    GridTierDataT& gridTier
  )
  {
    gridTier.getStorage()->createInstance();
    gridTier.size[ tier ] = tierSize;
    gridTier.invSize[ tier ] = Real(1.0) / tierSize;
    gridTier.boundingRadius[ tier ] = (Real32)calculateBoundingRadius<GridCellTableT>(tierSize);
    gridTier.cellTable[ tier ] = new GridCellTableT(cellIdBuffer);
    gridTier.zoneTable[ tier ] = new ContactZoneTable();
    // gridTier.zoneDependencyTable[ tier ] = new ZoneDependencyTable();
  }

  // Align a value to a power of two multiple/fraction of a specified alignment
  static Real align(Real value, Real alignment)
  {
    if (value > alignment)
    {
      UInt multiplier = agx::alignPowerOfTwo(UInt(value / alignment));
      Real alignedSize = alignment * (Real)multiplier;
      if (alignedSize < value)
        alignedSize *= 2;

      return alignedSize;
    }
    else
    {
      return align(value, alignment/2);
    }
  }

  template <typename GridTierDataT, typename CellDataT, typename CellT, typename CellIdT, typename CellPosT, typename GridCellTableT, size_t NumNeighbors>
  size_t createMissingTiers
  (
    Real minCellSize,
    agxData::Array< agx::UInt32 >& deadCells,
    agxData::Array< Physics::HierarchicalGrid::ContactZonePtr >& emptyZones,
    GridTierDataT& gridTier,
    CellDataT& cell,
    agxData::Array< CellT >& rootCells,
    const CellIdT *NeighborTable,
    agx::Bool& hasNewTiers,
    Physics::HierarchicalGrid::SolveBodyManager* solveBodyManager,
    const agx::UInt& contactZoneAccumulationLevel
  )
  {
    // std::cout << "Creating new tiers for cell size " << minCellSize << ", currently there are " << gridTier.numElements() << " tiers" << std::endl;
    size_t tier = InvalidIndex;
    size_t numTiers = gridTier.numElements();

    if ( numTiers == 0 )
    {
      if (cell.sizeAlignment > 0)
      {
        Real alignedSize = align(minCellSize, cell.sizeAlignment);
        UInt multiplier = (UInt)round(alignedSize > cell.sizeAlignment ? alignedSize/cell.sizeAlignment : cell.sizeAlignment/alignedSize);
        agxVerify(agx::isPowerOfTwo(multiplier));
        // std::cout << "Aligning cell size of initial tier from " << minCellSize << " to " << alignedSize << " (alignment: " << cell.sizeAlignment << ", multiplier " << multiplier << ")" << std::endl;
        minCellSize = alignedSize;
      }

      // There are not tiers at all, create a fitting root tier.
      tier = 0;
      createTier<GridTierDataT, GridCellTableT>(tier, minCellSize, cell.id.buffer(), gridTier);
    }
    else if ( Real(2.0)*minCellSize <= gridTier.size[ numTiers-1 ] )
    {
      // There is no tier small enough.

      tier = numTiers - 1; // Make 'tier' point to the currently smallest tier.
      Real currentTierSize = gridTier.size[ tier ];

      // Create smaller tier(s).
      while( Real(2.0)*minCellSize <= currentTierSize )
      {
        ++tier;
        currentTierSize = Real(0.5)*gridTier.size[ tier - 1 ];
        createTier<GridTierDataT, GridCellTableT>(tier, currentTierSize, cell.id.buffer(), gridTier);
      }

      numTiers = gridTier.numElements();
    }
    else if ( minCellSize > gridTier.size[0] )
    {
      // There is no tier large enough.
      Real currentTierSize = gridTier.size[ 0 ];

      // Create proxies for the required new grids in temporary storage.
      agx::Vector< GridTier<GridCellTableT> > newTiers;
      while ( minCellSize > currentTierSize )
      {
        currentTierSize *= Real(2.0);
        // std::cout << " ==> new root tier with size " << currentTierSize << std::endl;
        newTiers.push_back( GridTier<GridCellTableT>(currentTierSize, nullptr ) );
      }

      GridCellTableT& rootTierTable = *gridTier.cellTable[0];

      // Allocate slots in the real GridTier storage and make room on the lower indices.
      size_t numNewTiers = newTiers.size();
      agxVerify1( numNewTiers > 0, "Expected at least one tier to be created. Why wasn't it?" );
      gridTier.getStorage()->createInstances( numNewTiers );
      size_t tierOffset = gridTier.numElements();
      // std::cout << "  creating " << numNewTiers << " tiers, num total " << gridTier.numElements() << std::endl;

      if( numTiers > 0 )
      {
        // Move any old tiers down the hierarchy.
        for ( Int n = (Int)numTiers-1 ; n >= 0 ; --n )
        {
          size_t newTierIndex = n + numNewTiers;
          gridTier.size[ newTierIndex ] = gridTier.size[ n ];
          gridTier.invSize[ newTierIndex ] = gridTier.invSize[ n ];
          gridTier.boundingRadius[ newTierIndex ] = gridTier.boundingRadius[ n ];
          gridTier.cellTable[ newTierIndex ] = gridTier.cellTable[ n ];
          gridTier.zoneTable[ newTierIndex ] = gridTier.zoneTable[ n ];
          gridTier.instance[ newTierIndex ].swapId(gridTier.instance[ n ]);
        }
      }
      //numTiers = gridTier.numElements();

      // Copy temporary data into TierGrid storage, write backwards.
      for ( size_t newTierIndex = 0; newTierIndex < newTiers.size(); ++newTierIndex )
      {
        size_t destIndex = numNewTiers - newTierIndex - 1;
        gridTier.size[ destIndex ] = newTiers[ newTierIndex ].cellSize;
        gridTier.invSize[ destIndex ] = Real(1.0) / newTiers[ newTierIndex ].cellSize;
        gridTier.boundingRadius[ destIndex ] = (Real32)calculateBoundingRadius<GridCellTableT>(gridTier.size[ destIndex ]);
        gridTier.cellTable[ destIndex ] = new GridCellTableT(cell.id.buffer());
        gridTier.zoneTable[ destIndex ] = new ContactZoneTable();

      }


      // Update cell-to-tier references.
      // std::cout << "  Update " << cell.numElements() << " cells" << std::endl;
      agx::Vector<Physics::HierarchicalGrid::ContactZonePtr> invalidatedZones;
      for ( size_t n = 0 ; n < cell.numElements() ; ++n )
      {
        // std::cout << n << ": Update tier from " << (UInt)cell.tier[n] << " to " << (UInt)(cell.tier[n]+numNewTiers) << ", size: " << gridTier.size[cell.tier[n] + numNewTiers] << std::endl;

        cell.tier[ n ] = agx::UInt8(cell.tier[n] + numNewTiers);


        /**
        The internal contact zones are bound to the corresponding grid cell, but in a larger sized tier,
        following parents links, with offset == contactZoneAccumulationLevel. However, if the zone tier is close to 0
        the offset may be smaller (parent can not be above the root), which is ok, but when we get a new root tier we
        need to invalidate all such internal zones, so they can find a new ancestor connection.
        */
        auto& internalZone = cell.internalZone[n];
        if (internalZone)
        {
          UInt zoneTierOffset = internalZone.tier().calculateIndex() - cell.tier[n];

          if (zoneTierOffset != contactZoneAccumulationLevel) {
            // std::cout << "Invalidate internal zone for " << cell.id[n] << ", n=" << n << ", internal zone id: " << internalZone.id() << std::endl;
            invalidatedZones.push_back(internalZone);
            internalZone.invalidate();
          }
        }
      }

      if (!invalidatedZones.empty()) {

        // Build hash set to avoid duplicate entries
        agx::HashSet<Physics::HierarchicalGrid::ContactZonePtr> uniqueEmptyZones;
        for(auto&& zone : emptyZones) {
          uniqueEmptyZones.insert(zone);
        }

        auto oldSize = emptyZones.buffer()->size();
        size_t numNewEmpty = 0;
        emptyZones.buffer()->resize(oldSize + invalidatedZones.size());


        for(auto&& zone : invalidatedZones) {
          if (!uniqueEmptyZones.contains(zone)) {
            emptyZones[oldSize + numNewEmpty] = zone;
            numNewEmpty++;
          }
        }

        // Adjusting for duplicates
        emptyZones.buffer()->resize(oldSize + numNewEmpty);
      }


      // Old root cells need to find their parent
      for (typename GridCellTableT::iterator it = rootTierTable.begin(); it != rootTierTable.end(); ++it)
      {
        UInt32 oldRootCell = *it;

        connectToParentCell<GridTierDataT, CellDataT, CellT, CellIdT, CellPosT, GridCellTableT, NumNeighbors>
        (
          deadCells,
          emptyZones,
          oldRootCell,
          cell.tier[ oldRootCell ],
          cell.id[ oldRootCell ],
          gridTier, cell,
          rootCells,
          NeighborTable,
          hasNewTiers,
          solveBodyManager,
          contactZoneAccumulationLevel
        );
      }

      // Store the requested tier, not necessarily tier zero (due to recursive tier construction)
      tier = gridTier.numElements() - tierOffset;
    }
    else
    {
      agxAbort1("The appropriate tier for a particle doesn't exist, but the current tier range is neither too large nor too small. Precision problems perhaps?");
    }

    hasNewTiers = true;
    return tier;
  }
}


#endif /* AGX_CREATETIER_H */
