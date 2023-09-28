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
#include <agxWire/WireWinchController.h>

#include <agx/Referenced.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxWire
{
  // Forward declarations
  namespace Composite
  {
    class Segment;
  }

  /**
  Basic wire route controller. Handles initial route (add, remove, insert)
  and route changes during simulation (cut, merge, remove, insert).
  */
  class AGXPHYSICS_EXPORT WireRouteController : public agx::Referenced
  {
    public:
      WireRouteController();

      /**
      Add node to this route. Note that this call only is valid during initial
      route, i.e., it is not possible to add nodes when a wire is initialized.
      \param node - node to add to this route
      \return true if the node was added to this route - otherwise false (line initialized for example)
      */
      virtual bool add( agxWire::Node* node );

      /**
      Add winch to this route. Maximum number of winches are two and they can only
      be located at the ends.
      \param winchController - winch
      \param pulledInLength - pulled in length
      \return 0 for begin 1 for end and -1 for fail
      */
      virtual int add( agxWire::WireWinchController* winchController, agx::Real pulledInLength );

      /**
      Remove node from this route.
      \param node - node to remove
      \return true if the node was successfully removed from this route - otherwise false (node not in this route for example)
      */
      virtual bool remove( agxWire::Node* node );

      /**
      Remove winch from this route.
      \param winchController - winch controller to remove
      \return true if remove was successful - otherwise false (e.g., winch to part of this route)
      */
      virtual bool remove( agxWire::WireWinchController* winchController );

      /**
      Insert a node in this given distance along route. If distanceFromStart exceeds the current total route length the
      node will be added instead.
      \param node - node to insert
      \param distanceFromStart - distance from start of route where the node should be inserted
      \return true if the node was successfully inserted into this route - otherwise false
      */
      virtual bool insert( agxWire::Node* node, agx::Real distanceFromStart );

      /**
      Insert given \p node and \p beforeThisNode.
      \param node - node to insert before other node
      \param beforeThisNode - iterator to node
      \return true if insert was successful - otherwise false
      */
      virtual bool insert( agxWire::Node* node, agxWire::NodeIterator beforeThisNode );

      /**
      Calls Node::transform() on all added nodes.
      */
      virtual void updateNodeTransforms();

      /**
      Reverse route.
      \param initialized - if true, the routed nodes will not be touched - if false, a call to reverse will be made on each node
      \return true if successful - otherwise false
      */
      virtual bool reverse( bool initialized );

      /**
      Cut this route at a given distance along route, including begin winch controller if present.
      \param distanceAlongRoute - distance along this route where the cut should be made
      \return new route controller if cut was successful - otherwise 0
      */
      virtual WireRouteController* cut( agx::Real distanceAlongRoute );

      /**
      Attach this wire to a new object given new begin or end node and at which end. If this wire is
      attached (i.e., first or last node != free node) a detach will be performed before attach.
      Valid types of \p node is Node::BODY_FIXED and Node::CONNECTING. It is not valid to attach
      from an end where a winch is located.
      \param node - new begin or end node (body fixed or connecting)
      \param begin - if true, \p node will be new begin attachment - otherwise new end attachment
      \return true if attach was successful (correct node and valid wire) - otherwise false
      */
      virtual bool attach( agxWire::Node* node, bool begin );

      /**
      Attach a winch to a free end of this wire. If an object is attached to \p begin, it
      will be detached, and this winch controller will be attached at this position instead.
      \param winchController - winch controller to attach
      \param begin - true if \p winchController should be attached at begin, false at end
      \return true if the winch controller was attached
      */
      virtual bool attach( agxWire::WireWinchController* winchController, bool begin );

      /**
      Detach begin or end of this wire (if attached to something). It is not valid to detach a winch
      controller via this interface.
      \param begin - if true, begin of this wire will be detached - otherwise end
      \return true if a detach was performed - otherwise false
      */
      virtual bool detach( bool begin );

      /**
      Set the rest length and resolution per unit length for this wire route.
      \param resolutionPerUnitLength - number of lumped nodes per unit length (all real number >= 0 is valid)
      */
      virtual void setResolutionPerUnitLength( agx::Real resolutionPerUnitLength );

      /**
      \return the resolution per unit length of the wire
      */
      inline agx::Real getResolutionPerUnitLength() const;

      /**
      Calculates the current length between the nodes in the route. If \p includeConnecting
      is true, the route length of any connecting node will be included, otherwise not.
      \return the current length of this route
      */
      agx::Real getCurrentLength( bool includeConnecting = true ) const;

      /**
      Clear routing data
      */
      virtual void clear();

      /**
      Calculates the current length between node1 and node2 in wire coordinates. Note that node1 has to be before node2
      in this wire, i.e., if you travel along the wire from the beginning, the first node of the two you
      meet is node1.
      \param node1 - first wire node
      \param node2 - second wire node
      \return the current length between the node. If something goes wrong, the return value will be -1.
      */
      agx::Real getCurrentLength( const agxWire::Node* node1, const agxWire::Node* node2 ) const;

      /**
      Returns the wire node given distance from start. The wire node returned is the node before or at the given distance.
      \param distanceFromStart - distance from start
      \return the node before or at distanceFromStart
      */
      agxWire::Node* getNode( agx::Real distanceFromStart ) const;

      /**
      \return the nodes in the route
      */
      const agxWire::NodeContainer& getNodes() const { return m_nodes; }
      agxWire::NodeContainer& getNodes() { return m_nodes; }

      /**
      \return the winch controllers
      */
      const agxWire::WireWinchControllerRefVector& getWinchControllers() const { return m_winchControllers; }
      agxWire::WireWinchControllerRefVector& getWinchControllers() { return m_winchControllers; }

      /**
      \return true if the node is in this route - otherwise false
      */
      inline bool hasNode( const agxWire::Node* node ) const;

      /**
      \return iterator to end of node list
      */
      inline agxWire::NodeIterator getEndIterator();
      inline agxWire::NodeConstIterator getEndIterator() const;

      /**
      \param node - node to find iterator for
      \return iterator to \p node if present - otherwise end iterator
      */
      inline agxWire::NodeIterator getIterator( agxWire::Node* node );
      inline agxWire::NodeConstIterator getIterator( const agxWire::Node* node ) const;

    protected:
      virtual ~WireRouteController();

      friend class Wire;
      friend class Composite::Segment;

      /**
      Copy constructor, could be used by agxWire::Wire::setRouteController.
      */
      explicit WireRouteController( const agxWire::WireRouteController& other )
        : agx::Referenced()
      {
        *this = other;
      }

      agxWire::WireRouteController& operator = ( const agxWire::WireRouteController& other );

      inline void setInitialized( bool initialized );
      inline bool getInitialized() const;

      /**
      Inserts the node after the given node.
      \param node - node to insert
      \param afterThisNode - node in route that the node should be inserted after
      \return true if the insert is successful - otherwise false
      */
      bool insertAfter( agxWire::Node* node, agxWire::Node* afterThisNode );

      /**
      Adds \p node to list and map it to the iterator.
      */
      inline void addToList( agxWire::Node* node );

      /**
      Erases node from list.
      */
      inline bool removeFromList( agxWire::Node* node );

    protected:
      NodeContainer                 m_nodes;
      NodePtrNodeIterHash           m_nodeToIterator;
      bool                          m_initialized;
      agx::Real                     m_resolutionPerUnitLength;
      WireWinchControllerRefVector  m_winchControllers;
  };

  typedef agx::ref_ptr< WireRouteController > WireRouteControllerRef;

  // Inline methods ------------------------------------------------------------
  inline void WireRouteController::setInitialized( bool initialized )
  {
    m_initialized = initialized;
  }

  inline bool WireRouteController::getInitialized() const
  {
    return m_initialized;
  }

  inline bool WireRouteController::hasNode( const Node* node ) const
  {
    if ( node )
      return m_nodeToIterator.contains( const_cast< Node* >( node ) );
    return false;
  }

  inline agx::Real WireRouteController::getResolutionPerUnitLength() const
  {
    return m_resolutionPerUnitLength;
  }

  inline void WireRouteController::addToList( Node* node )
  {
    m_nodes.push_back( node );
    m_nodeToIterator.insert( node, --m_nodes.end() );
  }

  inline bool WireRouteController::removeFromList( Node* node )
  {
    // We need the iterator, so instead of hasNode.
    NodePtrNodeIterHash::iterator nodeToIterIt = m_nodeToIterator.find( node );
    if ( nodeToIterIt == m_nodeToIterator.end() )
      return false;

    NodeIterator nodeIt = nodeToIterIt->second;
    m_nodeToIterator.erase( nodeToIterIt );
    m_nodes.erase( nodeIt );

    return true;
  }

  inline NodeIterator WireRouteController::getEndIterator()
  {
    return m_nodes.end();
  }

  inline NodeConstIterator WireRouteController::getEndIterator() const
  {
    return m_nodes.end();
  }

  inline NodeIterator WireRouteController::getIterator( Node* node )
  {
    if ( node == nullptr )
      return m_nodes.end();
    NodePtrNodeIterHash::iterator i = m_nodeToIterator.find( node );
    return i != m_nodeToIterator.end() ? i->second : m_nodes.end();
  }

  inline NodeConstIterator WireRouteController::getIterator( const Node* node ) const
  {
    if ( node == nullptr )
      return m_nodes.end();
    NodePtrNodeIterHash::const_iterator i = m_nodeToIterator.find( const_cast< Node* >( node ) );
    return i != m_nodeToIterator.end() ? i->second : m_nodes.end();
  }
  // ---------------------------------------------------------------------------
}

DOXYGEN_END_INTERNAL_BLOCK()

