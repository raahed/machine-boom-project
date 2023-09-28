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

#ifndef AGX_HIERARCHICALGRID_CELL_REORDERING_H
#define AGX_HIERARCHICALGRID_CELL_REORDERING_H

#include <agx/Component.h>
#include <agx/Vec3.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(CellReordering);
  class CellReordering : public Component
  {
  public:
    CellReordering() : Component("CellReordering")
    {
    }

  protected:
    virtual ~CellReordering() {}


  public:
    agxData::EntityPtr *instance;
    UInt8 *tier;
    UInt8 *invDepth;
    Vec3i *id;
    Vec3 *position;
    Real32 *emptyTime;
    UInt8 *state;
    UInt32 *parent;
    UInt32 *children;
    UInt8 *numChildren;
    UInt32 *neighbors;
    IndexRange32 *collisionObjects;
    agxData::EntityPtr *internalZone;
    UInt8 *zoneType;
    AtomicValue **solveBodyOffsets;
    // UInt32 *sortedIndex;
  };

}

#endif

