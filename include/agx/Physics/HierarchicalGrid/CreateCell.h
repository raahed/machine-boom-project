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


#ifndef AGX_CREATE_CELL_H
#define AGX_CREATE_CELL_H

#include <agx/Morton.h>
#include <agx/Prefetch.h>

#include "Common.h"
#include "Util.h"
#include "SolveBodyManager.h"


namespace agx
{

  //-----------------------------------------------------------------------------------------------------

  // Forward declarations
  template <typename GridTierDataT, typename CellDataT, typename CellT, typename CellIdT, typename CellPosT, typename GridCellTableT, size_t NumNeighbors>
  UInt32 connectToParentCell
  (
    agxData::Array< agx::UInt32 >& deadCells,
    agxData::Array< Physics::HierarchicalGrid::ContactZonePtr >& emptyZones,
    Index cellIndex,
    UInt8 tier,
    const CellIdT cellId, // NOTE: Must be copy, not reference, can get invalid buffer read otherwise, see #2182
    GridTierDataT& gridTier,
    CellDataT& cell,
    agxData::Array< CellT >& rootCells,
    const CellIdT *NeighborTable,
    agx::Bool& hasNewTiers,
    Physics::HierarchicalGrid::SolveBodyManager* solveBodyManager,
    const agx::UInt& contactZoneAccumulationLevel

  );

  template <typename CellDataT, typename CellIdT>
  AGX_FORCE_INLINE void connectToParentCell
  (
    Index cellIndex,
    const CellIdT cellId, // NOTE: Must be copy, not reference, can get invalid buffer read otherwise, see #2182
    Index parentCellIndex,
    CellDataT& cell,
    Physics::HierarchicalGrid::SolveBodyManager* solveBodyManager,
    const agx::UInt& contactZoneAccumulationLevel
  );



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
  );


  //-----------------------------------------------------------------------------------------------------

  /**
   * Create and connect a cell. Will create all parent cells as well.
   */
  template <typename GridTierDataT, typename CellDataT, typename CellT, typename CellIdT, typename CellPosT, typename GridCellTableT, size_t NumNeighbors>
  UInt32 createCell
  (
    agxData::Array< agx::UInt32 >& deadCells,
    agxData::Array< Physics::HierarchicalGrid::ContactZonePtr >& emptyZones,
    UInt8 tier,
    const CellIdT cellId,
    GridTierDataT& gridTier,
    CellDataT& cell,
    agxData::Array< CellT >& rootCells,
    const CellIdT *NeighborTable,
    agx::Bool& hasNewTiers,
    Physics::HierarchicalGrid::SolveBodyManager* solveBodyManager,
    const agx::UInt& contactZoneAccumulationLevel
  )
  {
    UInt32 parentCellIndex = InvalidIndex;

    if (tier > 0)
    {
      // Make sure parent cell is available, we want big tier cells first in list of new cells for better performance in InitializedNewCells kernel
      UInt8 parentTier = (UInt8)(tier - 1);
      CellIdT parentId = calculateParentCellId(cellId);

      GridCellTableT& parentTierTable = *gridTier.cellTable[ parentTier ];
      parentCellIndex = parentTierTable.find( parentId );

      // Create parent cell if needed (recursive)
      if ( parentCellIndex == InvalidIndex )
      {
        size_t oldNumTiers = gridTier.cellTable.size();

        parentCellIndex = createCell
        <
          GridTierDataT, CellDataT, CellT, CellIdT, CellPosT, GridCellTableT, NumNeighbors
        >
        (
          deadCells, emptyZones, parentTier, parentId, gridTier, cell,
          rootCells, NeighborTable, hasNewTiers, solveBodyManager,
          contactZoneAccumulationLevel
        );

        // Important: update current tier
        tier = agx::UInt8(tier + (gridTier.numElements() - oldNumTiers));
      }
    }

    UInt32 cellIndex = InvalidIndex;


    // Check if we have a dead cell to use
    if (!deadCells.empty())
    {
      cellIndex = deadCells.back();
      deadCells.range().end()--;
      // std::cout << "Reusing dead cell " << cellIndex << " at index " << deadCells.size() << std::endl;

      // Must explicitly initialize parameters when reusing an old cell
      cell.parent[cellIndex] = InvalidIndex;
      cell.numChildren[cellIndex] = 0;
      cell.invDepth[cellIndex] = 0;
    }
    else
    {
      cellIndex = cell.createInstance();
    }



    cell.tier[ cellIndex ] = tier;
    cell.id[ cellIndex ] = cellId;
    cell.state[ cellIndex ] = CELL_UNINITIALIZED;

    GridCellTableT& cellTable = *gridTier.cellTable[ tier ];
    agxAssert( cellTable.find( cellId ) == InvalidIndex );
    cellTable.insert(cellId, cellIndex);

    // std::cout << cellIndex << ": Creating cell " << cellId <<  " in tier " << (UInt)tier << " with parent " << parentCellIndex << std::endl;

    if ( tier > 0 )
    {
      connectToParentCell<CellDataT, CellIdT>(cellIndex, cellId, parentCellIndex, cell, solveBodyManager, contactZoneAccumulationLevel);
    }
    else // Special handling of root cell
    {
      /**
      Make sure top tier contain a single root cell for this 'quadrant'. Around the origin the
      cells on different sides will never have a common parent, therefore we have a separate
      root cell for each 'quadrant' (4 in 2d, 8 in 3d).
      */

      UInt rootIndex = calculateRootCellIndex(cellId);
      // std::cout << "Creating root " << cellId << ", cellIndex: " << cellIndex << ", cell size: " << gridTier.size[tier] << std::endl;
      solveBodyManager->registerCell(cell.instance[cellIndex]);

      if (!rootCells[rootIndex])
      {
        agx::Physics::HierarchicalGrid::CellPtr root = agxData::EntityPtr::createWithIndex(cell.getStorage(), cellIndex);
        rootCells[rootIndex] = root;

        // Set up neighbor connections to other root cells, including self
        UInt32 *neighborList = &cell.neighbors[CellDataT::neighborsArraySize * cellIndex];

        for (size_t i = 0; i < NumNeighbors; ++i)
          neighborList[i] = InvalidIndex;

        for (size_t i = 0; i < rootCells.size(); ++i)
        {
          if (!rootCells[i])
            continue;

          UInt32 otherIndex = rootCells[i].calculateIndex();

          CellIdT offset = cell.id[otherIndex] - cellId;
          if (agx::Physics::HierarchicalGrid::isNeighborOffset(offset))
          {
            UInt localNeighborIndex = agx::Physics::HierarchicalGrid::calculateNeighborIndex(offset);
            neighborList[localNeighborIndex] = otherIndex;

            UInt32 *reverseList = &cell.neighbors[CellDataT::neighborsArraySize * otherIndex];
            reverseList[ NumNeighbors-1-localNeighborIndex ] = cellIndex;
          }
        }
      }
      else
      {
        // Remove current root cells
        for (size_t i = 0; i < rootCells.size(); ++i)
          rootCells[i] = agxData::EntityPtr();

        // Create new top tier
        createMissingTiers
        <
          GridTierDataT, CellDataT, CellT, CellIdT, CellPosT, GridCellTableT, NumNeighbors
        >
        (
          gridTier.size[0] * Real(2),
          deadCells,
          emptyZones,
          gridTier,
          cell,
          rootCells,
          NeighborTable,
          hasNewTiers,
          solveBodyManager,
          contactZoneAccumulationLevel
       );
      }

    }

    return cellIndex;
  }


  template <typename CellDataT, typename CellIdT>
  AGX_FORCE_INLINE void connectToParentCell
  (
    Index cellIndex,
    const CellIdT cellId,
    Index parentCellIndex,
    CellDataT& cell,
    Physics::HierarchicalGrid::SolveBodyManager* solveBodyManager,
    const agx::UInt& contactZoneAccumulationLevel
  )
  {
    // Connect both ways
    // std::cout << "Connecting " << cellIndex << " with parent " << parentCellIndex << std::endl;
    agxAssert(parentCellIndex != InvalidIndex);
    cell.parent[ cellIndex ] = (UInt32)parentCellIndex;
    UInt32 *childList = &cell.children[CellDataT::childrenArraySize * parentCellIndex];
    UInt childIndex = calculateCellChildIndex(cellId);

    const UInt numChildren = cell.numChildren[ parentCellIndex ];
    UInt compactedIndex = 0;
    for (; compactedIndex < numChildren; ++compactedIndex)
    {
      if (childIndex < childList[ compactedIndex ])
        break;
    }

    for (UInt n = numChildren; n > compactedIndex; --n)
      childList[n] = childList[n-1];

    childList[compactedIndex] = (UInt32)cellIndex;

    cell.numChildren[ parentCellIndex ]++;
    agxAssert(cell.numChildren[parentCellIndex] <= CellDataT::childrenArraySize);

    // Update inv depth of parents if parent was a leaf node
    if (numChildren == 0)
    {
      UInt8 invDepth = cell.invDepth[cellIndex];

      while (parentCellIndex != InvalidIndex && cell.invDepth[parentCellIndex] <= invDepth)
      {
        // std::cout << "cell " << cell.indexToId(parentCellIndex) << ", id: " << cell.id[parentCellIndex] << ", tier: " << (UInt)cell.tier[parentCellIndex]  << " now has inv depth " << (UInt)(invDepth + 1) << std::endl;
        cell.invDepth[parentCellIndex] = ++invDepth;

        if ( (invDepth >= contactZoneAccumulationLevel || cell.tier[parentCellIndex] <= contactZoneAccumulationLevel) && !cell.solveBodyOffsets[parentCellIndex])
          solveBodyManager->registerCell(cell.instance[parentCellIndex]);

        parentCellIndex = cell.parent[parentCellIndex];
      }
    }
  }


  template <typename GridTierDataT, typename CellDataT, typename CellT, typename CellIdT, typename CellPosT, typename GridCellTableT, size_t NumNeighbors>
  AGX_FORCE_INLINE UInt32 connectToParentCell
  (
    agxData::Array< agx::UInt32 >& deadCells,
    agxData::Array< Physics::HierarchicalGrid::ContactZonePtr >& emptyZones,
    Index cellIndex,
    UInt8 tier,
    const CellIdT cellId,
    GridTierDataT& gridTier,
    CellDataT& cell,
    agxData::Array< CellT >& rootCells,
    const CellIdT *NeighborTable,
    agx::Bool& hasNewTiers,
    Physics::HierarchicalGrid::SolveBodyManager* solveBodyManager,
    const agx::UInt& contactZoneAccumulationLevel
  )
  {
    UInt8 parentTier = UInt8(tier - 1);
    CellIdT parentId = calculateParentCellId(cellId);

    GridCellTableT& parentTierTable = *gridTier.cellTable[ parentTier ];
    UInt32 parentCellIndex = parentTierTable.find( parentId );

    // Create parent cell if needed (recursive)
    if ( parentCellIndex == InvalidIndex )
    {
      parentCellIndex = createCell
      <
        GridTierDataT, CellDataT, CellT, CellIdT, CellPosT, GridCellTableT, NumNeighbors
      >
      (
        deadCells, emptyZones, parentTier, parentId, gridTier, cell,
        rootCells, NeighborTable, hasNewTiers, solveBodyManager,
        contactZoneAccumulationLevel
      );

    }

    connectToParentCell<CellDataT, CellIdT>(cellIndex, cellId, parentCellIndex, cell, solveBodyManager, contactZoneAccumulationLevel);

    return parentCellIndex;
  }

}

#endif
