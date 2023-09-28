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
#include <agxWire/WireMaterialController.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxWire {

  class WireParameterController;

  class AGXPHYSICS_EXPORT WireStabilityController : public agx::Referenced
  {
    public:
      /**
      Utility method to calculate the stability factor given a lumped node,
      or for a hypothetic lumped node at the node position. If this
      stability factor is larger than 1 the node is considered unstable.
      \param node - node to check stability factor for
      \param wire - wire node is part of (failing if node is not part of wire)
      \param scaleConstant - scales the factor such that larger scale constant gives smaller stability factor
      \param smoothed - Should the smoothed or the raw tension be used
      \return stability factor for a lumped node - 0 for unknown nodes
      */
      static agx::Real  getStabilityFactor( const agxWire::Node* node, const agxWire::WireDistanceCompositeConstraint* wire, agx::Real scaleConstant = agx::Real( 1 ), bool smoothed = false );

      /**
      Utility method to calculate the stability factor given a lumped node,
      or for a hypothetic lumped node at the node position. If this
      stability factor is larger than 1 the node is considered unstable.
      \param nodeIt - lumped node iterator to check stability factor for
      \param wire - wire node is part of (failing if node is not part of wire)
      \param scaleConstant - scales the factor such that larger scale constant gives smaller stability factor
      \param smoothed - Should the smoothed or the raw tension be used
      \return stability factor for a lumped node - 0 for unknown nodes
      */
      static agx::Real  getStabilityFactor( agxWire::NodeConstIterator nodeIt, const agxWire::WireDistanceCompositeConstraint* wire, agx::Real scaleConstant = agx::Real( 1 ), bool smoothed = false );

      /**
      Analyze tension for one lumped node that feel tension from two sides
      \param tensionLeft  - tension in distance constraint from before in node list
      \param tensionRight - tension in distance constraint from after in node list
      \param lengthLeft   - distance to closest node before the lumped node in node list
      \param lengthRight  - distance to closest node before the lumped node in node list
      \param lumpMass     - weight of lumped node
      \param constant     - ScaleConstant/(timeStep*timeStep)
      \return true if node is stable
      */
      static bool isStable( agx::Real tensionLeft, agx::Real tensionRight, agx::Real lengthLeft, agx::Real lengthRight, agx::Real lumpMass, agx::Real constant );

    public:
      WireStabilityController( WireMaterialController* lmc );

      virtual void update( agxWire::WireParameterController* parameterController, agx::Real timeStep );

      void setWire( WireDistanceCompositeConstraint* lDCC )
      {
        m_wire = lDCC;
      }

      bool hasWire( )
      {
        return m_wire != nullptr;
      }

      /**
      Changes mass for two lumped nodes (the two first or the two last). Given a density the mass changes due to the change in segment length, dl.
      \param dl   - Change in constraint length
      \param bfnA - body fixed node to change mass for
      \param bfnB - body fixed node to change mass for
      */
      void changeMass( agx::Real dl, BodyFixedNode* bfnA, BodyFixedNode* bfnB ) const;

      /**
      \return the material controller
      */
      WireMaterialController* getMaterialController() { return m_materialController; }
      const WireMaterialController* getMaterialController() const { return m_materialController; }

      /**
      Utility method to test if this is, by definition, is a stable mass node on the wire.
      */
      bool  isStable( const agxWire::BodyFixedNode* node ) const;

      /**
      Check if two distances will "fit" between two nodes
      */
      bool isGeometricallyStable( NodeConstIterator prevIt, NodeConstIterator nextIt, agx::Real d1, agx::Real d2 ) const;

      /**
      Returns the factor used for the stability verification. (scaleFactor / (timestep * timestep))
      */
      agx::Real getTimeStepDependentStabilityConstant() const;

      /**
      Shape contact nodes have mass in a co-simulation.
      The mass distribution has taken these into account.
      The previous stability update must be supplemented with
      this sweep that removes nodes that became unstable.
      Some mass will be distributed to the remaining lumped nodes.
      */
      void removeUnstableNodesNearShapeContacts();

      /**
      Utility method to calculate the stability factor given a lumped node. If this
      stability factor is larger than 1 the node is considered unstable.
      \param node - lumped node to check stability factor for
      \param scaleConstant - scales the factor such that larger scale constant gives smaller stability factor
      \return stability factor for a lumped node - 0 for unknown nodes
      */
      agx::Real  getStabilityFactor( const agxWire::BodyFixedNode* node, agx::Real scaleConstant = agx::Real( 1 ) ) const;

      /**
      Utility method to calculate the stability factor given a lumped node. If this
      stability factor is larger than 1 the node is considered unstable.
      \param nodeIt - lumped node iterator to check stability factor for
      \param scaleConstant - scales the factor such that larger scale constant gives smaller stability factor
      \return stability factor for a lumped node - 0 for unknown nodes
      */
      agx::Real  getStabilityFactor( agxWire::NodeConstIterator nodeIt, agx::Real scaleConstant = agx::Real( 1 ) ) const;

    protected:
      enum MassNodeStatus {

        STABLE = 1,           /**< Node is in a stable state*/
        MOVABLE = (1<<2),     /**< Node could be moved, to get a better wire configuration */
        REMOVABLE = (1<<3),    /**< Node is under to much tension, must be moved*/
        LOW_RESOLUTION = (1<<4) /**< Node removed due to low resolution*/

      };

      struct InsertNodeData
      {
        agx::Real halfRestlengthFromPreviousToNext;
        agx::Real distanceFromABSegmentMiddle;
        agx::Real restlengthFromA;
        agx::Real minLengthFactor;
      };

      typedef agx::List< InsertNodeData > InsertNodeDataList;

      /**
      Check if a lumped node at will collide with an eye or contact next time step
      \param bfn - The node to check
      \param itBefore - iterator to previous node
      \param itAfter - iterator to next node
      \returns true - if collision will happen
      */
      bool isCollidingSlidingNode( BodyFixedNode* bfn,NodeConstIterator itBefore, NodeConstIterator itAfter ) const;

      /**
      find if lumped node position is ok for the local resolution of the line
      */
      bool resolutionHighEnoughForPosition( NodeConstIterator bfnIt, NodeConstIterator prevBfn, NodeConstIterator nextBfn, const agx::Real restlengthToBfn, const agx::Real minDistanceToLump, const agx::Real aroundLumpAngle, const agx::Real eps = agx::Real(1E-6) ) const;

      /**
      Stability analysis of lumped node
      \param bfnIt - Lumped node to analyze
      \param prevBfn - iterator pointing to node before this
      \param nextBfn - iterator pointing to node after this
      \param restlengthToBfn - distance to bfn along line
      \param minDistanceToLump - distance to nearest body fixed node
      \param lastBfnRemovedByResolution - indicates if the node to analyze can't be removed due to low resolution.
      \return Current status of lumped node
      */
      MassNodeStatus analyze( NodeConstIterator bfnIt, NodeConstIterator prevBfn, NodeConstIterator nextBfn, const agx::Real restlengthToBfn ,const agx::Real minDistanceToLump, const bool lastBfnRemovedByResolution ) const;

      /**
      test that a point along the line is within at least one of the ranges in insertRanges
      \param insertRanges - vector with valid ranges
      \param distance - a real value to test for the valid ranges
      \returns true if distance is within any of the "insertRanges"
      */
      bool isWithinRange( const agx::Vector<WireNodeRange>& insertRanges, const agx::Real distance ) const;

      void insertSorted( InsertNodeDataList& insertData, InsertNodeData ind ) const;

      /**
      Find all (closest to middle point between bfnA and bfnB ) valid insert distances for all middle points of the node-node segments between bfnA and bfnB
      \param itA - start lump node for segment
      \param itB - end lump node for segment
      \param lumpMass - potential new mass of the lump
      \param restlengthAB - rest length between bfnA and bfnB
      \param insertData - here we store (sorted with large values at front) the valid node-node middle point distances where we can insert and some additional info (  )
      \param minDistFromStart - don't insert node inside this distance from bfnA
      \param minDistanceFromEnd - don't insert node inside this distance from bfnB
      */
      void findMiddleInsertPositions(
        NodeConstIterator itA, NodeConstIterator itB, const agx::Real lumpMass, const agx::Real restlengthAB,
        InsertNodeDataList& insertData, const agx::Real minDistFromStart, const agx::Real minDistanceFromEnd) const;

      /**
      Find all (closest to middle point between bfnA and bfnB ) valid insert distances for all middle points of the node-node segments between bfnA and bfnB
      \param insertRanges - valid ranges within segment where resolution is high enough to insert a node
      \param insertData - here we store (sorted with large values at front) the valid node-node middle point distances where we can insert and some additional info (  )

      */
      agx::Real findRangeEndInsertPosition(const agx::Vector<WireNodeRange>& insertRanges,const InsertNodeDataList& insertData) const;

      /**
      Stability analysis of two lumped nodes to see if it is possible to add a new one between
      \param bfnAIt - Begin lumped node to analyze
      \param bfnBIt - End lumped node to analyze
      \param restlengthToA - restLength from startOfWire to A
      \param minDistFromStart - don't insert node inside this distance from bfnA
      \param minDistanceFromEnd - don't insert node inside this distance from bfnB
      \return Distance (in world coordinates) from bfnA to insert the new lump
      */
      agx::Real analyze( NodeConstIterator bfnAIt, NodeConstIterator bfnBIt, const agx::Real restlengthToA, const agx::Real minDistFromStart, const agx::Real minDistanceFromEnd ) const;

      inline agx::Real calculateLengthFactor( agx::Real l1, agx::Real l2 ) const
      {
        if ( agx::equalsZero(l1) || agx::equalsZero(l2) )
          return 0;
        return ( l1*l2/(l1+l2) );
      }

      /**
      get minimum length factor ( tension/Mass < (lengthRight*lengthLeft)/(lengthRight + lengthLeft) ) for mass and tension
      */
      inline agx::Real minimumLengthFactor( agx::Real tension, agx::Real mass, agx::Real constant ) const
      {
        return tension / (mass * constant);
      }

      /**
      Moves a lumped node along the wire to a more stable position, that also brings the node configuration closer to evenly spread masses
      \param bfn - lumped node to move
      \return Success
      */
      bool move( BodyFixedNode* bfn );

    protected:
      agx::observer_ptr< WireDistanceCompositeConstraint> m_wire;
      agx::Real m_scaleConstant;
      agx::Real m_timeStep;
      agx::observer_ptr< WireMaterialController> m_materialController;
      agx::Real m_noLumpNearDistance;
  };

  typedef agx::ref_ptr< WireStabilityController > WireStabilityControllerRef;

}

DOXYGEN_END_INTERNAL_BLOCK()
