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


#include <agxWire/Node.h>
#include <agxWire/WireTensionData.h>

#include <agx/Constraint.h>
#include <agx/BitState.h>

#include <agxUtil/SmoothingFilter.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxSDK
{
  class SimulationProxy;
}

namespace agxWire
{
  // Forward declarations
  class WireAttachmentController;
  class WireMaterialController;
  class WireFrictionController;
  class WireSimplifyController;
  class WireDistanceCompositeConstraintImplementation;
  class WirePropertyController;

  /**
  Class to get debug render calls.
  */
  class AGXPHYSICS_EXPORT ConstraintRenderer : public agx::Referenced
  {
    public:
      ConstraintRenderer() : m_enable( true ) {}
      virtual void render( agxRender::RenderManager *mgr, float scale ) = 0;
      inline bool getEnable() const { return m_enable; }
      inline void setEnable( bool enable ) { m_enable = enable; }

    protected:
      virtual ~ConstraintRenderer() {}
      bool m_enable;
  };

  typedef agx::ref_ptr< ConstraintRenderer > ConstraintRendererRef;

  /**
  Internal class.
  */
  class AGXPHYSICS_EXPORT WireListener : public agx::Referenced
  {
    public:
      WireListener() {}

      virtual void onAdd( agxWire::NodeIterator /*nodeIt*/ ) {}
      virtual void onInsert( agxWire::NodeIterator /*nodeIt*/ ) {}
      virtual void onRemove( agxWire::NodeIterator /*nodeIt*/ ) {}

      virtual void postReverse() {}

      virtual void preSolve() {}
      virtual void postSolve() {}

    protected:
      virtual ~WireListener() {}
  };
  typedef agx::ref_ptr< WireListener > WireListenerRef;

  /**
  Internal class.
  */
  class AGXPHYSICS_EXPORT WireDistanceCompositeConstraint : public agx::Constraint
  {
    public:
      /**
      Arguments to cut method.
      */
      struct AGXPHYSICS_EXPORT CutArgs
      {
        /**
        Construct given distance along wire (NOT including any winch pulled in length), minimum
        resolution of the new wire and a clone of the top level wire object.

        \param distanceAlongWire - distance along this wire where the cut should be made
        \param minResolution - minimum number of lumps in the new wire (if very short for example)
        \param newWire - top level clone of the wire this WireDistanceCompositeConstraint is constraint to
        */
        CutArgs( agx::Real distanceAlongWire, agx::UInt minResolution, class Wire* newWire );

        agx::Real distanceAlongWire;
        agx::UInt minResolution;
        class Wire* newWire;

        private:
          CutArgs() {}
      };

    public:
      WireDistanceCompositeConstraint( agxWire::WirePropertyController* propertyController, agxWire::WireDistanceCompositeConstraintImplementation* implementation = nullptr );

      /**
      \return wire property listener
      */
      agxWire::WirePropertyController* getPropertyController() const;

      /**
      During initialization, add nodes. Constraints will be created on the fly during
      this operation.
      \param node - node to add
      \return true if the node was successfully added
      */
      bool add( agxWire::Node* node );

      /**
      Insert node after current distance \p distanceFromStart.
      \param node - node to insert
      \param distanceFromStart - current distance from the first node to insert the node
      \return true if the insert was successful - otherwise false
      */
      bool insert( agxWire::Node* node, agx::Real distanceFromStart );

      /**
      Insert node \p node after \p afterThisNode in the list.
      \return true if the insert were successful - otherwise false
      */
      bool insertAfter( agxWire::Node* node, const agxWire::Node* afterThisNode );

      /**
      Insert node \p node before \p afterThisNode in the list.
      \return true if the insert were successful - otherwise false
      */
      bool insertBefore( agxWire::Node* node, const agxWire::Node* beforeThisNode );

      /**
      Remove node from this wire.
      \param node - node to remove
      \return true if node is present and was successfully removed - otherwise false
      */
      bool remove( agxWire::Node* node );

      /**
      Reverse order dependent structures.
      \return true if success - otherwise false
      */
      bool reverse();

      /**
      Merge this wire with \p other wire. The end of this wire and the begin of the
      other wire has to be free. The end point of this and the start of the other
      will be merged.

      All properties and parameters from this wire will be used. E.g., if this and \p other
      has different radii, the radius of this wire is the final.
      \param other - other wire to merge this wire with
      \return true if merge was successful (\p other will be empty and removed from simulation) - otherwise false
      */
      bool merge( agxWire::WireDistanceCompositeConstraint* other );

      /**
      Cut this wire given distance from start (including winches). If successful cut, a new wire is
      returned with at least \p minimumResolution number of lumps. The new wire is added to the
      simulation if this wire was in a simulation.
      \param args - arguments to this cut method, see CutArgs for further info
      \return new wire if cut was successful - otherwise 0
      */
      agxWire::WireDistanceCompositeConstraint* cut( CutArgs args );

      /**
      Attach this wire to a new object given new begin or end node and at which end. Valid types
      of \p node is Node::BODY_FIXED and Node::CONNECTING. It is not valid to attach from an end
      where a winch is located.
      \param node - new begin or end node (body fixed or connecting)
      \param begin - if true, \p node will be new begin attachment - otherwise new end attachment
      \return true if attach was successful (correct node and valid wire) - otherwise false
      */
      bool attach( agxWire::Node* node, bool begin );

      /**
      Silent attach, i.e., just change first or last node to free and then execute attach.
      \param node - new begin or end node (body fixed or connecting)
      \param begin - if true, \p node will be new begin attachment - otherwise new end attachment
      \return true if attach was successful (correct node and valid wire) - otherwise false
      */
      bool silentAttach( agxWire::Node* node, bool begin );

      /**
      Detach begin or end of this wire (if attached to something).
      \param begin - if true, begin of this wire will be detached - otherwise end
      \return true if a detach was performed - otherwise false
      */
      bool detach( bool begin );

      /**
      Initialize this distance composite constraint. Ideal is that nothing special
      is done during this call.
      \return true if successful - otherwise false
      */
      bool initialize();

      /**
      Updates linear velocity damping for the nodes given the linear velocity damping scalar
      assign to this wire. I.e., this method makes sure there's no damping along this wire.
      */
      void updateDamping();

      /**
      Add rendering class to get render calls when this wire gets rendered.
      \param renderer - rendering class
      */
      void add( ConstraintRenderer* renderer );

      /**
      Remove rendering class.
      \param renderer - rendering class
      */
      void remove( ConstraintRenderer* renderer );

      /**
      Add wire lump add/remove listener.
      \param lumpListener - wire lump add/remove listener
      */
      void add( WireListener* lumpListener );

      /**
      Remove wire lump add/remove listener.
      \param lumpListener - wire lump add/remove listener
      */
      void remove( WireListener* lumpListener );

      /**
      Will take a lumped node from the pool, and if a simulation is passed to this method the
      rigid body will be added to the simulation.
      \param particle - true if the lumped body should be treated as particle, false as rigid body
      \return new body fixed wire node (for a lump)
      */
      BodyFixedNode* getNewLumpedNode( bool particle = true );

      /**
      Places the lumped node back to the pool and removes the body from the simulation
      \param bfn - the body fixed node
      */

      void removeLumpedNodeBody( BodyFixedNode* bfn );

      /**
      Calculates the rest length (loop over all body fixed nodes and sum their rest length). If this
      constraint hasn't been initialized rest length is zero.
      \return rest length of this constraint
      */
      agx::Real getRestLength() const;

      /**
      Calculates the current length (sum the length between all nodes). If this
      constraint hasn't been initialized current length is zero.
      \return current length of this constraint
      */
      agx::Real getCurrentLength() const;

      /**
      \return the sum of all discrete node masses in this wire (i.e., the current, total simulated wire mass)
      */
      agx::Real getMass() const;

      /**
      This function is used by getNode, but can be used of some nodes in the beginning or end of the wire should be excluded.
      \param restLengthFromStart - current distance from start (restLength)
      \param distanceOnSegment - distance to point from the returned iterator
      \return iterator to node before rest length from start.
      */
      agxWire::NodeIterator getIterator( agx::Real restLengthFromStart, agx::Real& distanceOnSegment  );

      /**
      \return iterator to \p node
      */
      agxWire::NodeIterator getIterator( agxWire::Node* node );
      agxWire::NodeConstIterator getIterator( const agxWire::Node* node ) const;

      /**
      \return iterator to begin of wire
      */
      agxWire::NodeIterator getBeginIterator();
      agxWire::NodeConstIterator getBeginIterator() const;

      /**
      \return reverse_iterator to begin of wire
      */
      NodeReverseIterator getReverseBeginIterator();
      NodeConstReverseIterator getReverseBeginIterator() const;

      /**
      \return iterator to end of wire
      */
      agxWire::NodeIterator getEndIterator();
      agxWire::NodeConstIterator getEndIterator() const;

      /**
      \return reverse_iterator to end of wire
      */
      agxWire::NodeReverseIterator getReverseEndIterator();
      agxWire::NodeConstReverseIterator getReverseEndIterator() const;

      /**
      \return iterator to the first node in the list where e.g., geometries or rendering, should start
      \sa getGeometryEndIterator
      */
      agxWire::NodeIterator getGeometryBeginIterator();
      agxWire::NodeConstIterator getGeometryBeginIterator() const;

      /**
      \return iterator to the last node in the list where e.g, geometries or rendering, should end.
      Note that this iterator points to the last node of the purpose, so if you want to iterate from begin
      to end, do:
      iterator end = c->getGeometryEndIterator();
      ++end;
      for ( iterator i = c->getBeginGeometryIterator(); i != end; ++i ) ...
      or
      while ( i != end ) ...
      in order to get the all the nodes.
      \sa getGeometryBeginIterator
      */
      agxWire::NodeIterator getGeometryEndIterator();
      agxWire::NodeConstIterator getGeometryEndIterator() const;

      /**
      Set the geometry end iterator.
      DANGEROUS, should only be called if you know exactly what you are doing.
      */
      void setGeometryEndIterator( agxWire::NodeIterator endIterator );

      /**
      \return true if the geometry list is empty - otherwise false
      */
      bool getGeometryListEmpty() const;

      /**
      Search vector/list container defining a wire for the 3D point -> 1D wire projection. I.e., given any 3D point,
      this function will find the distance along the wire closest to \p point.
      \param pulledInBegin - amount of wire hauled in, in for example a winch at the beginning of the wire
      \param point - any 3D point
      \param shortestDistanceSquared - shortest distance squared to the point on the wire used in the returned distance
      \return distance along wire closest to the 3D point
      */
      agx::Real findRestLengthFromStartGivenPoint( agx::Real pulledInBegin, const agx::Vec3& point, agx::Real& shortestDistanceSquared );

      /**
      Will loop through node list to find the next body fixed node, among the lumped nodes (not the last node in m_nodes)
      \param node - a node in m_nodes
      \return iterator to lump. on fail nullptr
      */
      agxWire::BodyFixedNode* findNextLumpedNode( const agxWire::Node* node ) const;
      agxWire::NodeIterator findNextLumpedNode( agxWire::NodeIterator node ) const;
      agxWire::NodeConstIterator findNextLumpedNode( agxWire::NodeConstIterator node ) const;

      /**
      Will loop through node list to find the previous body fixed node, among the lumped nodes (not the first node in m_nodes)
      \param node - a node in m_nodes
      \return iterator to lump. on fail nullptr
      */
      agxWire::BodyFixedNode* findPrevLumpedNode( const agxWire::Node* node ) const;
      agxWire::NodeIterator findPrevLumpedNode( agxWire::NodeIterator node ) const;
      agxWire::NodeConstIterator findPrevLumpedNode( agxWire::NodeConstIterator node ) const;

      /**
      \return first lumped node in this wire (0 if not initialized)
      */
      agxWire::BodyFixedNode* getFirstLump();
      const agxWire::BodyFixedNode* getFirstLump() const;

      /**
      \return iterator to first lumped node (end if not initialized)
      */
      agxWire::NodeIterator getFirstLumpIterator();
      agxWire::NodeConstIterator getFirstLumpIterator() const;

      /**
      \return last lump in this wire (0 if not initialized)
      */
      agxWire::BodyFixedNode* getLastLump();
      const agxWire::BodyFixedNode* getLastLump() const;

      /**
      \return iterator to last lumped node (end if not initialized)
      */
      agxWire::NodeIterator getLastLumpIterator();
      agxWire::NodeConstIterator getLastLumpIterator() const;

      /**
      \return the first node in the list if this constraint has a valid configuration - otherwise null
      */
      agxWire::Node* getFirstNode() const;

      /**
      \return the last node in the list if this constraint has a valid configuration - otherwise null
      */
      agxWire::Node* getLastNode() const;

      /**
      \return true if \p node is part of this wire - otherwise false
      */
      bool hasNode( const agxWire::Node* node ) const;

      /**
      Calculate current length between two nodes
      \param A - is wire node before B in node list
      \param B - is wire node after A in node list
      \return Length on success, -1 on fail
      */
      agx::Real getLength( const agxWire::Node* A, const agxWire::Node* B ) const;

      /**
      Calculate rest length between two nodes on
      \param A - is wire node before B in node list
      \param B - is wire node after A in node list
      \return rest length on success, -1 on fail
      */
      agx::Real getRestLength( const agxWire::Node* A, const agxWire::Node* B ) const;

      /**
      Calculate current length between two nodes
      \param itA - is wire node before B in node list
      \param itB - is wire node after A in node list
      \return Length on success, -1 on fail
      */
      agx::Real getLength( agxWire::NodeConstIterator itA, agxWire::NodeConstIterator itB ) const;

      /**
      Calculate rest length between two nodes on
      \param itA - is wire node before B in node list
      \param itB - is wire node after A in node list
      \return rest length on success, -1 on fail
      */
      agx::Real getRestLength( agxWire::NodeConstIterator itA, agxWire::NodeConstIterator itB ) const;

      /**
      Scale current length in world coordinates to rest length
      \param currentLength - length in world coordinates
      \param nodeIt - node iterator where the scale should be calculated
      \return rest length for segment
      */
      agx::Real scaleToRestlength( agx::Real currentLength, agxWire::NodeConstIterator nodeIt ) const;

      /**
      Given node on this wire, calculates the transform from current length to rest length.
      \param nodeIt - node iterator on this wire, close to where the scale is about to be used
      \return the scaler for scaling a current length to a rest length given the violation of the wire near the node
      */
      agx::Real getRestLengthScaler( agxWire::NodeConstIterator nodeIt ) const;

      /**
      Scale rest length to current length
      \param restLength - length in world coordinates
      \param nodeIt - node iterator where the scale should be calculated
      \return current length for segment
      */
      agx::Real scaleToCurrentlength( agx::Real restLength, agxWire::NodeConstIterator nodeIt ) const;

      /**
      Given node on this wire, calculates the transform from rest length to current length.
      \param nodeIt - node iterator on this wire, close to where the scale is about to be used
      \return the scaler for scaling a current length to a rest length given the violation of the wire near the node
      */
      agx::Real getCurrentLengthScaler( agxWire::NodeConstIterator nodeIt ) const;

      /**
      Get mass for wire segment of length dist
      \param dist - length of wire segment
      \return Mass of wire
      */
      agx::Real getMass( agx::Real dist ) const;

      /**
      Set a new begin attachment controller to this constraint.
      \param beginAttachment - new begin attachment controller
      */
      void setBeginAttachment( agxWire::WireAttachmentController* beginAttachment );

      /**
      Set a new end attachment controller to this constraint.
      \param endAttachment - new end attachment controller
      */
      void setEndAttachment( agxWire::WireAttachmentController* endAttachment );

      /**
      Return pointer to the attachment controller used at begin of the wire constraint.
      \returns - the begin attachment.
      */
      const agxWire::WireAttachmentController* getBeginAttachment() const;

      /**
      Return pointer to the attachment controller used at end of the wire constraint.
      \returns - the end attachment.
      */
      const agxWire::WireAttachmentController* getEndAttachment() const;

      /**
      Insert node to list before nodeIter
      \param nodeIter - iterator to insert node before
      \param node - The node to be inserted
      \return success
      */
      bool insertInList( agxWire::NodeIterator nodeIter, agxWire::Node* node );

      /**
      Remove \p node from the list and any mapping associated to it.
      \param node - node to remove from list
      \return true if successful - otherwise false
      */
      bool removeFromList( agxWire::Node* node );

      /**
      Updates the compliance to use when initializing distance constraints
      \param compliancePerUnitLength - compliance per unit length
      */
      void setCompliancePerUnitLength( agx::Real compliancePerUnitLength );

      /**
      Return compliance pre unit length
      */
      agx::Real getCompliancePerUnitLength();

      /**
      Updates the currentLength for a body fixed node, this must be called if a node has been removed from the segment that the bfn is storing rest and current length for
      \param bfn - body fixed node
      \return The current length of the segment
      */
      agx::Real setCurrentLength( agxWire::BodyFixedNode* bfn ) const;

      /**
      Assign smoothing filter for the wire to use to smooth tension values.
      \param smoothingFilter - new smoothing filter
      */
      void setSmoothingFilter( agxUtil::SmoothingFilter* smoothingFilter );

      /**
      Internal method. Don't use this method unless it's for internal and well defined use.
      */
      const agxWire::NodeContainer& getNodes() const;

      /**
      Collect nodes in this wire.
      */
      void getNodes( agxWire::NodePtrVector& nodes ) const;

      /**
      \return radius of this wire (material controller has to be present)
      */
      agx::Real getRadius() const;

      /**
      enable/disable the ability to automatically split the wire between two body fixed nodes.
      That means replacing the distance constraint by a force.

      Internal method chooses the body fixed nodes, and if it is possible to split.
      */
      void setEnableSplitting( bool splitting );

      /**
      \return true - if wire has splitting enabled
      */
      bool getEnableSplitting();

      /**
      Replaces \p node with a lumped node. I.e., \p node will be removed from this wire and
      a new lumped node will be inserted instead.
      \param node - node to change to a lumped node
      \return the new lumped node if successful - otherwise 0
      */
      agxWire::BodyFixedNode* changeNodeToBodyFixed( agxWire::Node* node );

      /**
      Change a body fixed node to contact node and inserts that node into this wire.
      \param bfn - body fixed node to remove and be replaced by a contact node
      \param cn - the new contact node (i.e., has to be created and configured before this call)
      \return true if the change was successful - otherwise false
      */
      bool changeBodyFixedToContactNode( agxWire::BodyFixedNode* bfn, agxWire::ContactNode* cn );

      /**
      Spool in/out wire given rest length. Taking care of masses in wire, scales rest length and current length equally(%).
      Takes care of friction constraints that are connected to the end-lumped-node
      \param deltaRestLength - the changed rest length of the wire
      \param atBegin if the wire length change should occur at begin
      */
      void changeLengthGivenRestLength( const agx::Real deltaRestLength, const bool atBegin );

      /**
      Spool in/out wire given current length. Taking care of masses in wire, scales rest length and current length equally(%).
      Takes care of friction constraints that are connected to the end-lumped-node
      \param deltaCurrentLength - the changed current length of the wire
      \param atBegin if the wire length change should occur at begin
      */
      void changeLengthGivenCurrentLength( const agx::Real deltaCurrentLength, const bool atBegin );

      /**
      Remove the lumped node when it is impossible to maintain stability
      \param bfn lumped node to remove
      \return Iterator to the wire node still on wire that was positioned after (now has same position in list) the removed lumped node
      */
      agxWire::NodeIterator remove( agxWire::BodyFixedNode* bfn );

      /**
      Remove the lumped node when it is impossible to maintain stability
      \param bfnIt lumped node to remove
      \return Iterator to the wire node still on wire that was positioned after (now has same position in list) the removed lumped node
      */
      agxWire::NodeIterator remove(agxWire::NodeIterator bfnIt );

      /**
      Add a lumped node to the wire
      \param bfnA - Begin lumped node to
      \param bfnB - End lumped node to analyze
      \param dist - Distance (in world coordinates) from bfnA to insert the new lump
      \return Iterator to the new lumped node
      */
      agxWire::NodeIterator insert( const agxWire::BodyFixedNode* bfnA, const agxWire::BodyFixedNode* bfnB, agx::Real dist );

      /**
      Add a lumped node to the wire
      \param bfnAIt - Begin lumped node to
      \param bfnBIt - End lumped node to analyze
      \param dist - Distance (in world coordinates) from bfnA to insert the new lump
      \return Iterator to the new lumped node
      */
      agxWire::NodeIterator insert(agxWire::NodeConstIterator bfnAIt, agxWire::NodeConstIterator bfnBIt, agx::Real dist);

      /**
      Increase a segment (the one before \p lumpedNode) by a delta current distance > 0.
      \note Calling this method will NOT change the total length of the wire.
      \param lumpedNode - lumped node defining the segment
      \param dCurrentLength - change in current length
      \return true if \p lumpedNode has been moved - otherwise false
      */
      bool increaseSegmentByCurrentLength( agxWire::BodyFixedNode* lumpedNode, agx::Real dCurrentLength );

      /**
      Will return the number of contacts in a row from wire end (begin or end), if body != nullptr, the contact must be on body
      \param begin - from begin or end(begin == false)
      \param body  - body which contacts should have
      \return number of contacts
      */
      int getNumberOfContactsFromWireEnd( bool begin, agx::RigidBody* body = nullptr );

      /**
      \return the tension for the segment of node \p node
      */
      agxWire::WireNodeTensionData getTension( const agxWire::Node* node ) const;

      /**
      \return the tension for the segment of node \p node
      */
      agxWire::WireNodeTensionData getTension( agxWire::NodeConstIterator node ) const;

      /**
      \return the tension for the segment after \p distanceFromStart
      \note This part of the wire implementation doesn't know anything about winches etc., so the distance can not be given
            including winches pulled in length.
      */
      agxWire::WireSegmentTensionData getTension( agx::Real distanceFromStart ) const;

      /**
      transform all nodes in wire
      */
      void transformNodes();

      /**
      Changes body in CONTACT NODES related to currentBody with a new body, newBody.
      */
      bool replaceBodyInNodes( agx::RigidBody* currentBody, agx::RigidBody* newBody, agxCollide::Geometry* geometry = nullptr );

      /**
      \return Simulation
      */
      agxSDK::SimulationProxy* getSimulationProxy();
      const agxSDK::SimulationProxy* getSimulationProxy() const;

      /**
      Associate the wire material controller to this constraint. Used for example to calculate bend compliance.
      */
      void setMaterialController( agxWire::WireMaterialController* materialController );

      /**
      \return the material controller of this wire (0 if not initialized)
      */
      agxWire::WireMaterialController* getMaterialController() const;

      /**
      Associate a friction controller to this constraint.
      */
      void setFrictionController( agxWire::WireFrictionController* frictionController );

      /**
      \return the friction controller
      */
      agxWire::WireFrictionController* getFrictionController() const;

      /**
      Associate a simplify controller to this constraint.
      */
      void setSimplifyController( agxWire::WireSimplifyController* simplifyController );

      /**
      \return the simplify controller
      */
      agxWire::WireSimplifyController* getSimplifyController() const;

      /**
      \return the number of DOF for this constraint, not including secondary constraints. -1 if num DOF is undefined for the constraint.
      */
      virtual int getNumDOF() const override;

      /**
      Update the geometry iterators, could be very important if addToList or removeFromList has been called
      */
      void updateGeometryIterators( agxWire::Node* node );

      /**
      Search for the geometry end iterator of this constraint.
      \return the geometry end iterator of this constraint
      */
      agxWire::NodeIterator findGeometryEndIterator();

      /**
      Call when a attachment controller is about to delete its attachment constraint.
      */
      void aboutToDeleteAttachmentConstraint( const agx::Constraint* constraint );

      /**
      Shorten this wire to a given node. All nodes from this wire's end to \p toNode
      will be removed.
      \note The begin/end has to be free.
      \param toNode - node to short this wire to
      \param fromBegin - true if shorten this wire from begin, false from end
      \return true if successful - otherwise false
      */
      bool shorten( const agxWire::Node* toNode, bool fromBegin );

      /**
      USED BY WINCHES.
      Explicitly replace attachment controller at begin or end with \p connectingNode. The attachment
      controller will not be uninitialized, i.e., the stop node will still be part of
      this wire.
      \note This wire has to be initialized.
      \param connectingNode - connecting node to replace winch controller
      \param begin - true to replace winch controller at begin - false for at end
      \return true if successful - otherwise false
      */
      bool explicitReplaceAttachmentController( agxWire::ConnectingNode* connectingNode, bool begin );

    protected:
      virtual ~WireDistanceCompositeConstraint();

      virtual void render( agxRender::RenderManager *mgr, float scale  ) const override;

    private:
      class WireDistanceCompositeConstraintImplementation* m_implementation;
  };

  typedef agx::ref_ptr< WireDistanceCompositeConstraint > WireDistanceCompositeConstraintRef;
}


DOXYGEN_END_INTERNAL_BLOCK()
