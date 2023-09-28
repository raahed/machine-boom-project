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

#ifndef AGX_SOLVEISLANDSPLITALGORITHM_H
#define AGX_SOLVEISLANDSPLITALGORITHM_H

#include <agx/agx.h>
#include <agx/Referenced.h>
#include <agxCollide/Geometry.h>
#include <agx/RigidBody.h>

namespace agx
{
  namespace Physics
  {
    class GraphNodeData;
  }
}

namespace agx
{

  class AGXPHYSICS_EXPORT SplitNode : public agx::Referenced
  {
  public:

    virtual void setChoosenToSplit(bool splittable) = 0;

    agx::Index getIslandIndex() const;
    agx::Index getBodyIndex() const;
    size_t getWeight() const;
    void setWeight( size_t weight );
    void setIslandIndex( agx::Index );
    bool isSplittable() const;
    agx::Vec3 getOldAcceleration() const;
    void setOldAcceleration( agx::Vec3 oldAcc );
    agx::RigidBody* getRigidBody(const agx::Physics::GraphNodeData* graphNode) const;
    agxCollide::Geometry* getFirstGeometry(const agx::Physics::GraphNodeData* graphNode) const;
  protected:
    virtual ~SplitNode() {}

  protected:
    size_t         m_weight;
    agx::Index     m_bodyId;
    bool           m_splittable;
    agx::Index     m_islandIndex;
    agx::Vec3      m_oldAcc;
  };


  typedef agx::ref_ptr<SplitNode> SplitNodeRef;
  typedef agx::Vector<SplitNodeRef> SplitNodeRefVector;
  typedef SplitNodeRefVector::iterator SplitNodeIterator;

  class AGXPHYSICS_EXPORT SplitNodeSeparator : public SplitNode
  {
    public:
      SplitNodeSeparator( size_t weight, agx::Index index );
      virtual void setChoosenToSplit(bool splittable);
    protected:
      virtual ~SplitNodeSeparator();
  };
  typedef agx::ref_ptr<SplitNodeSeparator> SplitNodeSeparatorRef;


  /**
  The rigid body class, combining a geometric model and a frame of reference.
  */
  class AGXPHYSICS_EXPORT SolveIslandSplitAlgorithm : public agx::Referenced
  {
  public:

     virtual void collectSplitInformation(agx::Physics::GraphNodeData* graphNode) = 0;

     SplitNodeRefVector& getSplitInfo();

     virtual bool getUseHighFidelitySplitting() const;

     virtual void setUseHighFidelitySplitting( bool highFidelity );

     virtual void clearSplitState() = 0;

     virtual void postSolveCleanup() = 0;

     void clear();

    protected:
      virtual ~SolveIslandSplitAlgorithm() {}

    protected:
      /**
      First and last node must be a no-split-node, and every other node must be a splittable node.
      */
      void addSplitNode(SplitNode* splitNode);
      void addSplitSeparatorNode( agx::Physics::GraphNodeData* graphNode, SplitNode* splitNode );

      SplitNodeRefVector m_splitInfo;
      /**
      How many split nodes surrounding the split node in both directions can NOT be split next time step
      */
      size_t             m_splitInfectedDistance;
      bool               m_useHighFidelitySplitting;
  };

  typedef agx::ref_ptr<SolveIslandSplitAlgorithm> SolveIslandSplitAlgorithmRef;

} // namespace agx


#endif
