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


#include <agx/agxPhysics_export.h>
#include <agxWire/WireDistanceCompositeConstraint.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxWire
{

  //Forward declaration
  class WireFrictionConstraintImplementation;

  class AGXPHYSICS_EXPORT WireFrictionController : public agx::Referenced
  {
    public:
      WireFrictionController();

      /**
      Initializes all nodes with active materials, i.e., a call to this method defines a reference state.
      */
      void initialize();

      void updateDirection();

      void toggleDirection( NodeIterator node );
      void reset( NodeConstIterator node, bool useCurrentLength = false ) const;

      /**
      Updates group of nodes on same geometry that could share one friction constraint.
      \param begin - first node to analyze (the frictionConstraint will start or end from this node)
      \param end - end of nodes in container
      \return true if this node should have a friction constraint
      */
      bool updateRowOfSlidingNodes( NodeIterator begin, NodeIterator end );

      /**
      Updates material for nodes that has or should have friction constraints.
      \param node - node to analyze
      \param normalForceMagnitude - Use as initial normal force magnutude
      \return true if this node should have a friction constraint
      */
      bool update( NodeIterator node, agx::Real normalForceMagnitude = 0 );

      /**
      Creates friction constraint given one sliding node. Will find if there are many nodes of the
      same sliding type on the same geometry.
      \param firstSlidingNode - contact or eye that is first sliding node
      \param nodeAfterLastSlidingNode - reference to node after first sliding node. Will change if there are many nodes on a row
      \param wfci - friction constraint
      \param compliancePerUnitLength - Compliance of friction constraint
      \param damping - damping of friction constraint
      \returns true - if constraint was needed
      */
      bool createFrictionConstraint( const NodeIterator firstSlidingNode, NodeIterator& nodeAfterLastSlidingNode, WireFrictionConstraintImplementation* wfci, agx::Real compliancePerUnitLength, agx::Real damping  );

      /**
      Enable/disable this friction controller.
      \param enable - true to enable
      */
      void setEnable( bool enable );

      /**
      \return true if this friction controller is enabled
      */
      inline bool getEnable() const
      {
        return m_enable;
      }

      /**
      calculates rest length for stick nodes when lump is removed or added
      */
      void updateFrictionRestlength( NodeConstIterator A, NodeConstIterator B, NodeConstIterator middleLump, bool remove ) const;

      /**
      Associate a constraint to this friction controller.
      */
      void setWire( WireDistanceCompositeConstraint* wire );

      /**
      Get associated constraint
      */
      WireDistanceCompositeConstraint* getWire() const { return m_wire; }

    private:

      void updateToggledRestLength( agxWire::BodyFixedNode* nextBfn, agxWire::NodeMaterial* nMaterial );
      void initialize( NodeIterator node ) const;
      agx::Vec3 calculateNormalForce( NodeIterator node ) const;
      agx::Bool isSlidingNode(Node* node) const;

    private:
      agx::observer_ptr< WireDistanceCompositeConstraint > m_wire;
      bool m_enable;
  };

  typedef agx::ref_ptr< WireFrictionController > WireFrictionControllerRef;
}


DOXYGEN_END_INTERNAL_BLOCK()
