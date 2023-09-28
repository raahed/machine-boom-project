/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code, 
tutorials, scene files and technical white papers, is copyrighted, proprietary 
and confidential material of Algoryx Simulation AB. You may not download, read, 
store, distribute, publish, copy or otherwise disseminate, use or expose this 
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB. 
*/

#ifndef AGX_HIERARCHICAL_GRID_ZONETABLE_H
#define AGX_HIERARCHICAL_GRID_ZONETABLE_H

#include "CellTable.h"
// #include <agx/Physics/HierarchicalGrid/ContactZoneEntity.h>
// #include <agx/Physics/HierarchicalGrid/ContactZoneDependencyEntity.h>

namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {
      class ContactZonePtr;
      class ContactZoneDependencyPtr;
    }
  }
  
  
  typedef agx::HashTable< Vec3i, agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZoneTable;
  typedef agx::HashTable< UInt64, agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr > ZoneDependencyTable;

  /////////////////////////////////////////////////////////////////
  
  // typedef HashTable< Vec3i, agx::UInt32 > GridCellTable;
  // typedef HashTable< Vec2i, agx::UInt32 > GridCell2DTable;

  typedef CellTable< Vec3i > GridCellTable;
  typedef CellTable< Vec2i > GridCell2DTable;
  
}


#endif /* AGX_HIERARCHICAL_GRID_ZONETABLE_H */
