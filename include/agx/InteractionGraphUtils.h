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

#ifndef AGX_INTERACTION_GRAPH_UTILS_H
#define AGX_INTERACTION_GRAPH_UTILS_H

#include <agx/agx.h>
#include <agx/Physics/GraphNodeEntity.h>

namespace agx
{
  class AGXPHYSICS_EXPORT IslandEntry
  {
    public:
    UInt offset;
    UInt size;
    int weight;
    AGX_FORCE_INLINE IslandEntry() : offset(0), size(0), weight(0) {}
    AGX_FORCE_INLINE IslandEntry(UInt _offset, UInt _size, int _weight) : offset(_offset), size(_size), weight(_weight) {}
    static size_t weightOfInteractionNode( const agx::Physics::GraphNodeData& graphNode, size_t bodyNodeIndex, size_t& loops );
  };



} //namespace agx


#endif
