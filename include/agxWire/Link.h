/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#include <agxWire/ILinkNode.h>
#include <agxWire/WireParallelCallbacksHandler.h>

#define LINK_V2_SERIALIZATION_VERSION agx::UInt16( 37 )

namespace agxSDK
{
  class SimulationProxy;
}

namespace agxWire
{
  class Winch;

  AGX_DECLARE_POINTER_TYPES( Link );

  /**
  Object that defines the relation between different types of wires (agxWire::Wire). I.e.,
  this link object enables functionality to (for example) move nodes from one wire to
  another.
  */
  class AGXPHYSICS_EXPORT Link : public agx::Referenced, public agxStream::Serializable
  {
    public:
      AGX_DECLARE_POINTER_TYPES( Algorithm );
      typedef agx::Vector< agxWire::ILinkNodeRef > ConnectionsContainer;
      typedef agx::Vector< agxWire::LinkRef > LinkConnectionsContainer;

    public:
      /**
      Connection type of the wires connected to this link, i.e., if this
      link should be attached to begin or end of the wire.
      */
      enum ConnectionType
      {
        WIRE_BEGIN         = 0,    /**< Link attached at begin of wire. */
        WIRE_END           = 1,    /**< Link attached at end of wire. */
        LINK                  ,    /**< Link attached to another link. */
        INVALID_CONNECTION = 0xFF  /**< Invalid connection, e.g., if this link isn't
                                        connected to a give wire. */
      };

      /**
      State when a link is interacting with a winch.
      */
      enum class WinchState
      {
        PULLED_IN,    /**< The link is completely pulled into a winch - the rigid body
                           is disabled and the transform isn't updated. */
        INSIDE,       /**< The link is inside the winch, not visible if the winch
                           is a hole, the rigid body is enabled and constrained with a
                           cylindrical joint and geometries are disabled. */
        OVERLAPPING,  /**< The link is partly inside the winch, the winch node is removed
                           so that the two connecting nodes may pass. */
        NONE          /**< The link is not interacting with a winch. */
      };

    public:
      /**
      \param connectingNode - wire or link connecting node
      \param findClosestIfLinkConnections - if true and \p connectingNode is connected to a link,
                                            the search will continue over all link connections
                                            until first wire connection
      \return the wire connection given connecting node
      */
      static agxWire::Wire* getWire( const agxWire::ConnectingNode* connectingNode,
                                     agx::Bool findClosestIfLinkConnections = false );

      /**
      \return the link given connecting node
      */
      static agxWire::Link* getLink( const agxWire::ConnectingNode* connectingNode );

      /**
      \return other link connected to the connecting node
      */
      static agxWire::Link* getConnectingLink( const agxWire::ConnectingNode* connectingNode );

      /**
      \return the connecting type given connecting node
      */
      static agxWire::Link::ConnectionType getConnectionType( const agxWire::ConnectingNode* connectingNode );

      /**
      Utility function to connect two wires with a link. \p link will be connected at WIRE_END of \p wire1 and
      WIRE_BEGIN at \p wire2.
      \param wire1 - first wire with \p link connected at end of its route
      \param relativeTranslate1 - position in this links frame where the connection point to wire 1 is
      \param link - The link that should be connected between the two wires
      \param wire2 - second wire with \p link connected at begin of its route
      \param relativeTranslate2 - position in this links frame where the connection point to wire 2 is
      \return true if successful, otherwise false
      */
      static agx::Bool connect( agxWire::Wire* wire1,
                                const agx::Vec3& relativeTranslate1,
                                agxWire::Link* link,
                                agxWire::Wire* wire2,
                                const agx::Vec3& relativeTranslate2 );

    public:
      /**
      Default constructor.
      */
      Link( agx::RigidBody* rb );

      /**
      \return the rigid body of this link
      */
      agx::RigidBody* getRigidBody() const;

      /**
      Maps this link to \p wire at given connection type (WIRE_BEGIN/WIRE_END). This method will only
      map \p wire to this link, it will not actually be attached.
      \sa attach( agxWire::Wire* wire, const agx::Vec3& relativeTranslate )
      \param wire - wire to connect
      \param relativeTranslate - position in this links frame where the connection point is
      \param type - connection type at WIRE_BEGIN or WIRE_END
      \return true if successful, otherwise false
      */
      agx::Bool connect( agxWire::Wire* wire, const agx::Vec3& relativeTranslate, agxWire::Link::ConnectionType type );

      /**
      Connect this link to another link with a constraint.
      \param link - other link to connect to this link
      \param constraint - constraint connecting the links
      \return true when connect is successful, otherwise false
      */
      agx::Bool connect( agxWire::Link* link, agx::Constraint* constraint );

      /**
      Connects/maps and attaches \p wire to this link. The current attachment at \p type will be detached
      and replaced by this link.
      \param wire - wire to attach
      \param relativeTranslate - position in this links frame where the connection point is
      \param type - connection type at WIRE_BEGIN or WIRE_END
      \return true if successful, otherwise false
      */
      agx::Bool attach( agxWire::Wire* wire, const agx::Vec3& relativeTranslate, agxWire::Link::ConnectionType type );

      /**
      Disconnect, or detach, this link from \p wire.
      \param wire - wire to disconnect
      \return true if disconnected, otherwise false
      */
      agx::Bool disconnect( agxWire::Wire* wire );

      /**
      Disconnect, or detach, this link from \p otherLink.
      \param otherLink - link to detach
      \return true if disconnected, otherwise false
      */
      agx::Bool disconnect( agxWire::Link* otherLink );

      /**
      Access default wire connection properties for each new wire that's connected to this link.
      \note If any value is changed the current connections will not be updated with this value.
            Only new connection will receive this new default value.
      \return default connection properties
      */
      agxWire::ILinkNode::ConnectionProperties* getDefaultWireConnectionProperties();

      /**
      \return default wire connection properties
      */
      const agxWire::ILinkNode::ConnectionProperties* getDefaultWireConnectionProperties() const;

      /**
      Add a link algorithm.
      \param algorithm - algorithm to add
      \return true if added, false if already part of this link or not added
      */
      agx::Bool add( agxWire::Link::Algorithm* algorithm );

      /**
      Remove link algorithm.
      \param algorithm - algorithm to remove
      \return true if removed, otherwise false
      */
      agx::Bool remove( agxWire::Link::Algorithm* algorithm );

      /**
      \note If there are several algorithms of same type, the first will be returned.
      \return the first algorithm given type
      */
      template< typename T >
      T* getAlgorithm() const;

      /**
      Transforms all the connecting nodes when for example the rigid body has been moved.
      */
      void transform();

      /**
      \return the number of connections connected to the ends of this link
      */
      agx::UInt getNumConnections() const;

      /**
      \return all connections
      */
      const agxWire::Link::ConnectionsContainer& getConnections() const;

      /**
      \return true if a connection is defined for this wire, otherwise false
      */
      agx::Bool hasConnection( const agxWire::Wire* wire ) const;

      /**
      \return connecting node given wire, null if no connection defined
      */
      agxWire::ILinkNode* getConnectingNode( const agxWire::Wire* wire ) const;

      /**
      \return connecting node given link, null if no connection defined
      */
      agxWire::ILinkNode* getConnectingNode( const agxWire::Link* link ) const;

      /**
      \return the connection if it exists between this link and the other \p link, otherwise null
      */
      agx::Constraint* getConnectingConstraint( const agxWire::Link* link ) const;

      /**
      \return a set of connecting nodes given predicate
      */
      template<typename Pred>
      agxWire::Link::ConnectionsContainer getConnectingNodes( Pred pred ) const;

      /**
      \return connecting node given predicate
      */
      template<typename Pred>
      agxWire::ILinkNode* findConnectingNode( Pred pred ) const;

      /**
      \return true if this link is enabled - otherwise false
      */
      agx::Bool getEnable() const;

      /**
      Shows connection points in debug rendering view.
      */
      void debugRenderConnections() const;

      /**
      Finds a state how this link is interacting with any winch.

      \note If the state is WinchState::NONE and this link is disabled
            it means that this link is part of a disabled wire configuration.
      \note If this link is connected to another link only one of the links
            will have state INSIDE. The other link will have states OVERLAPPING
            then PULLED_IN (in the case going into the winch) or PULLED_IN then
            OVERLAPPING to NONE when going out.
      \return the current winch state
      */
      WinchState findWinchState() const;

      /**
      Finds the winch this link is completely pulled into, meaning
      this link is inactive with WinchState::PULLED_IN. If this
      link is enabled or when no winch is found, nullptr is returned.
      \return winch this link is completely pulled into, otherwise nullptr
      */
      agxWire::Winch* findPulledInWinch() const;

    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      /**
      Initializes this link.
      */
      void initialize( agxSDK::SimulationProxy* simulationProxy );

      /**
      Enable/disable this link.
      \param simulationProxy - simulation proxy this link belongs/should belong to
      \param enable - true to enable, false to disable this link
      */
      void setEnable( agx::Bool enable, agxSDK::SimulationProxy* simulationProxy );

      /**
      Callback when a wire is being removed from the simulation.
      */
      void onRemoveNotification( agxWire::Wire* wire, agxWire::ConnectingNode* connectingNode );

      AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::Link );

    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      State of the link.
      */
      enum State
      {
        NONE           = 0,
        INITIALIZED    = 1 << 0, /**< Initial route point if view. */
        ENABLE_TOGGLED = 1 << 1  /**< Enabled or disabled - to fire onEnable events. */
      };

      typedef agx::Vector< agxWire::Link::AlgorithmRef > AlgorithmContainer;

    protected:
      /**
      Default constructor during restore.
      */
      Link();

      /**
      Reference counted object - protected destructor.
      */
      virtual ~Link();

      /**
      Checks connections and removes them where the wires have been deleted.
      */
      void collectGarbage();

      /**
      Add connection, register to LinkController.
      */
      void addConnection( agxWire::Link::ConnectionsContainer& connections, agxWire::ILinkNode* node );

      /**
      Erase connection, register to LinkController.
      */
      void eraseConnection( agxWire::Link::ConnectionsContainer& connections, agxWire::ILinkNode* node );

      /**
      Add state.
      */
      void addState( agxWire::Link::State state );

      /**
      Remove state.
      */
      void removeState( agxWire::Link::State state );

      /**
      \return true if state is set
      */
      agx::Bool hasState( agxWire::Link::State state ) const;

    protected:
      friend class LinkController;

      /**
      Garbage collects and checks if this link is ready for update calls.
      \return true if this link wants update calls for given simulation proxy
      */
      virtual agx::Bool executeUpdate( const agxSDK::SimulationProxy* simulationProxy );

      /**
      Update call.
      */
      virtual void update( agxWire::WireParallelCallbacksHandler::CallbackTypes type, agxSDK::SimulationProxy* simulationProxy );

    protected:
      agx::RigidBodyRef               m_rb;
      AlgorithmContainer              m_algorithms;
      AlgorithmContainer              m_newlyAddedAlgorithms;
      agx::Int32                      m_state;
      ILinkNode::ConnectionProperties m_defaultWireConnectionProperties;
      ILinkNode::ConnectionProperties m_defaultLinkConnectionProperties;

    private:
      ConnectionsContainer            m_connections;
  };

  /**
  Base class for an algorithm present in the agxWire::Link context. This algorithm gets callbacks on
  preCollide, pre, post and when the enable-state of the link has been toggled.
  */
  class AGXPHYSICS_EXPORT Link::Algorithm : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Called when link enable state is changed. \p enable is the new state of the link.
      \param enable - true if the link is enabled, false if disabled
      \param link - link which state is changed
      \param simulationProxy - simulation proxy
      */
      virtual void onLinkEnable(agx::Bool enable, agxWire::Link* link, agxSDK::SimulationProxy* simulationProxy);

      /**
      Simulation pre-collide callback.
      \param link - link this algorithm is connected to
      \param simulationProxy - simulation proxy
      */
      virtual void preCollide(agxWire::Link* link, agxSDK::SimulationProxy* simulationProxy);

      /**
      Simulation pre callback.
      \param link - link this algorithm is connected to
      \param simulationProxy - simulation proxy
      */
      virtual void pre(agxWire::Link* link, agxSDK::SimulationProxy* simulationProxy);

      /**
      Simulation post callback.
      \param link - link this algorithm is connected to
      \param simulationProxy - simulation proxy
      */
      virtual void post(agxWire::Link* link, agxSDK::SimulationProxy* simulationProxy);

      /**
      Add unique child algorithm.
      \return true if added - false if null, already present or already has a parent
      */
      agx::Bool addChild( agxWire::Link::Algorithm* algorithm );

      /**
      Remove child algorithm.
      \return true if removed - false if null or not present as a child.
      */
      agx::Bool removeChild( agxWire::Link::Algorithm* algorithm );

      /**
      \return the number of children of this algorithm
      */
      agx::UInt getNumChildren() const;

      /**
      \return child at index 'i' - bound check, null if out of bound
      */
      agxWire::Link::Algorithm* getChild( agx::UInt i ) const;

      /**
      \return the parent of this algorithm
      */
      agxWire::Link::Algorithm* getParent() const;

      /**
      Internal method.
      Called before children get update calls.
      */
      void preChildrenUpdate();

      /**
      Internal method.
      Called after children get update calls.
      */
      void postChildrenUpdate();

      /**
      \return algorithm of given type
      */
      template< typename T >
      T* find() const;

      /**
      \return this algorithm as given type - null if type mismatch
      */
      template< typename T >
      const T* as() const;

      /**
      \return this algorithm as given type - null if type mismatch
      */
      template< typename T >
      T* as();

    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      /**
      Store this object to stream.
      */
      virtual void store( agxStream::OutputArchive& out ) const override;

      /**
      Restore this object from stream.
      */
      virtual void restore( agxStream::InputArchive& in ) override;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( Link::Algorithm );

    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Abstract class - protected constructor.
      */
      Algorithm();

    private:
      enum State { IDLE = 0, UPDATING_CHILDREN = (1<<0) };

    private:
      typedef agx::Vector< agxWire::Link::AlgorithmRef > ChildrenContainer;

    private:
      ChildrenContainer m_children;
      ChildrenContainer m_removedChildren;
      Link::Algorithm*  m_parent;
      agx::Int32        m_state;
  };

  DOXYGEN_START_INTERNAL_BLOCK()

  inline void Link::Algorithm::onLinkEnable(agx::Bool /*enable*/, agxWire::Link* /*link*/, agxSDK::SimulationProxy* /*simulationProxy*/) {}

  inline void Link::Algorithm::preCollide(agxWire::Link* /*link*/, agxSDK::SimulationProxy* /*simulationProxy*/) {}

  inline void Link::Algorithm::pre(agxWire::Link* /*link*/, agxSDK::SimulationProxy* /*simulationProxy*/) {}

  inline void Link::Algorithm::post(agxWire::Link* /*link*/, agxSDK::SimulationProxy* /*simulationProxy*/) {}

  template<typename T>
  T* Link::getAlgorithm() const
  {
    T* algorithm = nullptr;
    for ( agx::UInt i = 0, numAlgorithms = m_algorithms.size(); algorithm == nullptr && i < numAlgorithms; ++i )
      algorithm = m_algorithms[ i ]->find<T>();

    return algorithm;
  }

  DOXYGEN_END_INTERNAL_BLOCK()

  template<typename Pred>
  Link::ConnectionsContainer Link::getConnectingNodes( Pred pred ) const
  {
    ConnectionsContainer ret;
    ret.reserve( m_connections.size() );
    for ( agx::UInt i = 0, numConnections = m_connections.size(); i < numConnections; ++i )
      if ( pred( m_connections[ i ] ) )
        ret.push_back( m_connections[ i ] );
    return ret;
  }

  template<typename Pred>
  ILinkNode* Link::findConnectingNode( Pred pred ) const
  {
    for ( agx::UInt i = 0, numConnections = m_connections.size(); i < numConnections; ++i )
      if ( pred( m_connections[ i ] ) )
        return m_connections[ i ];
    return nullptr;
  }

  template<typename T>
  T* Link::Algorithm::find() const
  {
    T* algorithm = const_cast<Link::Algorithm*>( this )->as<T>();
    for ( agx::UInt i = 0; algorithm == nullptr && i < m_children.size(); ++i )
      algorithm = m_children[ i ]->find<T>();

    return algorithm;
  }

  template<typename T>
  const T* Link::Algorithm::as() const
  {
    return dynamic_cast<const T*>( this );
  }

  template<typename T>
  T* Link::Algorithm::as()
  {
    return dynamic_cast<T*>( this );
  }
}


