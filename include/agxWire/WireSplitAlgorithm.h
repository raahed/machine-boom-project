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

#pragma once

#include <agx/SolveIslandSplitAlgorithm.h>
#include <agxWire/Node.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agx
{
  namespace Physics
  {
    class GraphNodeData;
  }
}

namespace agxWire
{
  class Wire;

  /// Implements a Wire model with adaptive resolution.
  class AGXPHYSICS_EXPORT WireSplitNode : public agx::SplitNode
  {
    public:
      WireSplitNode( BodyFixedNode* node, size_t weight, agx::Index index );

      virtual void setChoosenToSplit(bool splittable) override;
      /**
      Weight for estimate of solver cost for one body fixed node
      */
      static size_t singleNodeSolveCost() { return size_t(25); }
    protected:
      virtual ~WireSplitNode();

    private:
      BodyFixedNode* m_bfn;
  };
  typedef agx::ref_ptr<WireSplitNode> WireSplitNodeRef;

  /**
  Interface and placeholder of controllers/helpers for wires.
  */
  class AGXPHYSICS_EXPORT WireSplitAlgorithm : public agx::SolveIslandSplitAlgorithm
  {
    public:
     WireSplitAlgorithm( agxWire::Wire* wire, size_t splitInfectedDistance = 1 );

     virtual void collectSplitInformation( agx::Physics::GraphNodeData* graphNode ) override;

     virtual bool getSplitEnabled( NodeConstIterator bfnIt ) const;

     virtual void clearSplitState() override;

     virtual void postSolveCleanup() override;

    protected:
      virtual ~WireSplitAlgorithm() {}

      bool createSplitNode(NodeConstIterator it, size_t weight);
      bool createSplitSeparatorNode( agx::Physics::GraphNodeData* graphNode, NodeConstIterator it);
      bool updateSplitInfo( agx::Physics::GraphNodeData* graphNode, NodeConstIterator firstSplitNode, const size_t numAllowedConnections = 5 );

      NodeConstIterator findNextSplitNodeOrSeparator( agx::Physics::GraphNodeData* graphNode,
        const NodeContainer& nodes,
        const NodeConstIterator previousSplitnode,
        const NodeConstIterator itEnd,
        const size_t numAllowedConnections );

      NodeConstIterator findNextPossibleSplitNodeIterator(  const agx::Physics::GraphNodeData* graphNode,
                                                                                const NodeContainer& nodes,
                                                                                NodeConstIterator it,
                                                                                const NodeConstIterator itEnd,
                                                                                const size_t numAllowedConnections = 5);

      agxWire::Wire* m_wire;
  };

}

DOXYGEN_END_INTERNAL_BLOCK()
