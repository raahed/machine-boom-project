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

#include <agxStream/Serializable.h>

#include <agx/RigidBody.h>
#include <agxCollide/Geometry.h>

#include <agx/List.h>

#include <agx/LinearProbingHashTable.h>
#include <agx/QuadraticProbingHashTable.h>

namespace agxWire
{
  // Forward declarations
  class Node;
  class FreeNode;
  class BodyFixedNode;
  class EyeNode;
  class ConnectingNode;
  class StopNode;
  class ContactNode;
  class ShapeContactNode;
  class WireFrictionController;
  class WireTensionController;
  class WireFrictionConstraintImplementation;
  class SpatialState;
  class Wire;
  class Link;

  typedef agx::HashVector< agxCollide::GeometryRef, agx::Vec3 > GeometryRefDepthTable;
  // Container typedefs
  typedef agx::ref_ptr< Node > NodeRef;
  typedef agx::List< NodeRef > NodeRefListA;
  typedef NodeRefListA NodeContainer;
  typedef NodeContainer::iterator NodeIterator;
  typedef NodeContainer::reverse_iterator NodeReverseIterator;
  typedef NodeContainer::const_reverse_iterator NodeConstReverseIterator;
  typedef NodeContainer::const_iterator NodeConstIterator;
  typedef agx::Vector<NodeRef> NodeRefVector;

#include <HashImplementationSwitcher.h>
#if HASH_FOR_WIRE == HASH_NEW
  using NodePtrNodeIterHash = agx::LinearProbingHashTable<Node*, NodeIterator>;
#elif HASH_FOR_WIRE == HASH_OLD
  typedef agx::QuadraticProbingHashTable< Node*, NodeIterator > NodePtrNodeIterHash;
#else
  #error
#endif

  typedef agx::List< Node* > NodePtrListA;
  typedef NodePtrListA NodePtrContainer;
  typedef NodePtrContainer::iterator NodePtrIterator;
  typedef NodePtrContainer::const_iterator NodePtrConstIterator;

  typedef agx::Vector< Node* > NodePtrVector;

  enum class WireGeometryType
  {
    SEGMENT,
    LUMPED_SPHERE,
    CONTACT_SENSOR
  };

  /**
  Node coordinate frame in another frame. Holds data for original
  local translate, current world (transformed) translate and
  relative (transformed) translate.
  */
  class AGXPHYSICS_EXPORT NodeFrame
  {
    public:
      enum Elements
      {
        ORG_TRANSLATE,      /**< Original translate. */
        TRANSLATE,          /**< Current transformed translate. */
        REL_BODY_TRANSLATE, /**< Current transformed relative */
        ENUM_SIZE
      };

      /**
      Default constructor.
      */
      NodeFrame();

      /**
      Destructor.
      */
      ~NodeFrame();

      /**
      Data access.
      \param element - index (see Elements)
      \return data for current index
      */
      agx::Vec3& getElement( Elements element );

      /**
      Data access.
      \param element - index (see Elements)
      \return data for current index
      */
      const agx::Vec3& getElement( Elements element ) const;

      /**
      Recalculates TRANSLATE and REL_BODY_TRANSLATE.
      */
      void transform();

      /**
      Set original translate (world frame if this node frame doesn't have a rigid body).
      \param translate - position in parent frame (world or body)
      */
      void setTranslate( const agx::Vec3& translate );

      /**
      Set original translate (world frame if this node frame doesn't have a rigid body).
      \param x - x position in parent frame (world or body)
      \param y - y position in parent frame (world or body)
      \param z - z position in parent frame (world or body)
      */
      void setTranslate( agx::Real x, agx::Real y, agx::Real z );

      /**
      \return original translate in parent frame (world or body)
      */
      const agx::Vec3& getTranslate() const;

      /**
      Manly for writing Jacobian data to the solver.

      Calculates the vector from center of mass of the rigid body (world origin if rigid body is null)
      to this node.
      \note Assumes 'transform' has been called, i.e., all data is up to date.
      \param rb - rigid body from which the vector starts, world if null
      \return the vector from center of mass to the node
      */
      agx::Vec3 calculateCmOffset( const agx::RigidBody* rb ) const;

      /**
      \return the world position
      */
      const agx::Vec3& getWorldPosition() const;

      /**
      Set rigid body.
      */
      void setRigidBody( agx::RigidBody* body );

      /**
      \return rigid body of this frame
      */
      agx::RigidBody* getRigidBody() const;

    protected:
      agx::RigidBodyRef m_rb;
      agx::Vec3         m_data[ ENUM_SIZE ];
  };

  /**
  Material definition for a wire node Specifies the friction of a wire for sliding through an EyeNode
  */
  class AGXPHYSICS_EXPORT NodeMaterial
  {
    public:
      /**
      Friction direction in a wire.
      */
      enum Direction
      {
        NEGATIVE = 0, /**< Direction towards begin of the wire. */
        POSITIVE,     /**< Direction towards end of the wire. */
        BOTH          /**< For mutators only, assigns both values. */
      };

      /**
      Internal. Direction of friction.
      */
      enum FrictionDirection
      {
        GO_FORWARD = 1,                 /**< Friction direction is forward. */
        IS_DIRTY = (1<<1),              /**< Friction direction has been changed. */
        FIX_FRICTION_DIRECTION = (1<<2) /**< Friction direction is by definition on in one direction. */
      };

      /**
      Default constructor.
      */
      NodeMaterial();

      /**
      Internal method.
      \return true if the rest length is valid (i.e., >= 0)
      */
      bool hasValidRestLength() const;

      /**
      Internal method.
      \return true if either negative, positive direction or both are stick
      */
      bool isStick() const;

      /**
      Internal method.
      \return true if the contact state is stick
      */
      bool isStick( Direction dir ) const;

      /**
      Internal method.
      Set if this material should generate constraints or not.
      \param active - true to activate constraints
      */
      void setActive( bool active );

      /**
      Internal method.
      \return true if this material has been assigned to generate constraints
      */
      bool getActive() const;

      /**
      Internal method.
      Set that negative, positive or both direction has stick friction.
      */
      void setStick( bool stick, Direction dir );

      /**
      \return the wire friction coefficient in negative or positive direction (Direction::BOTH is not defined)
      */
      agx::Real getFrictionCoefficient( Direction dir ) const;

      /**
      \return the edge velocity scaler in negative or positive direction (Direction::BOTH is not defined)
      */
      agx::Real getKinematicContactNodeVelocityScale( Direction dir ) const;

      /**
      Set stick friction coefficient in one or both directions.
      \param coeff - stick friction coefficient
      \param dir - direction (Default: BOTH, valid NEGATIVE, POSITIVE or BOTH)
      */
      void setFrictionCoefficient( agx::Real coeff, Direction dir = BOTH );

      /**
      Set slide friction coefficient in one or both directions.
      \param scaler - edge velocity scaler
      \param dir - direction (Default: BOTH, valid NEGATIVE, POSITIVE or BOTH)
      */
      void setKinematicContactNodeVelocityScale( agx::Real scaler, Direction dir = BOTH );

      /**
      \return true if this material is enabled
      */
      bool getEnable() const;

      /**
      \return true if this material is enabled
      */
      bool isEnabled() const;

      /**
      \return the active direction given forward or not for friction constraints (transforms from FrictionDirection to Direction)
      */
      Direction getActiveDirection() const;

      /**
      \return the force in direction \p dir for the friction constraints force range
      */
      agx::Real getMaxForce( Direction dir );

      /**
      \return true if there is infinite friction in \p dir and \p stick (true) coefficient or edge velocity scaler (stick = false)
      */
      bool isInfCoefficient( Direction dir ) const;

      /**
      \return the assigned normal force magnitude
      */
      agx::Real getNormalForceMagnitude() const;

      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      Toggle the direction of the friction constraint to maintain positive tension.
      */
      void toggleGoForward();

      /**
      \return true if the friction constraints direction is forward (positive direction)
      */
      bool getGoForward() const;

      /**
      \return true if the friction direction has been changed
      */
      bool goForwardIsDirty() const;

      /**
      call this function if friction massless should change direction (if tension is negative)
      The direction could stay, if there is a winch in that direction
      */
      void setGoForwardIsDirty();

      /**
      Resets the dirty flag for go forward.
      */
      void resetGoForwardIsDirty();

      /**
      Internal method.
      Update friction stick state.
      */
      void updateStick( agx::Real tension );

      /**
      Invalidate rest length. Triggers a recalculate of a friction constraint length.
      */
      void setInvalidRestLength();

      /**
      Save current state.
      */
      void store( agxStream::OutputArchive& out ) const;

      /**
      Restore from disc.
      */
      void restore( agxStream::InputArchive& in, agxWire::Node* node );

      /**
      Internal method.
      Reverse all direction dependent parameters in this material.
      */
      void reverse();

      /**
      Internal method.
      \return the rest length of the constraint used
      */
      agx::Real getRestLength() const;

      /**
      Internal method.
      \return the current friction violation
      */
      agx::Real getViolation() const;
      DOXYGEN_END_INTERNAL_BLOCK()


    protected:
      friend class WireFrictionController;
      friend class Node;
      friend class EyeNode;
      friend class WireFrictionConstraintImplementation;
      friend class WireOldContactController;
      friend class WireDistanceCompositeConstraintImplementation;

      /**
      Assign the rest length of the friction constraint.
      \param restLength - new rest length
      */
      void setRestLength( agx::Real restLength );

      /**
      Assign current violation in the friction constraint.
      \param violation - current violation
      */
      void setViolation( agx::Real violation );

      /**
      Assign violation per unit length.
      \param violationPerUnitLength - new violation per unit length
      */
      void setViolationPerUnitLength( agx::Real violationPerUnitLength );

      /**
      \return the violation per unit length
      */
      agx::Real getViolationPerUnitLength();

      /**
      Assign the current normal force magnitude.
      \param normalForce - current normal force magnitude
      */
      void setNormalForceMagnitude( agx::Real normalForce );

    protected:
      enum CurrentType { VELOCITY_SCALER = 0, FRICTION_COEFFICIENT };

    protected:
      agx::Real m_frictionCoeffs[ 2 ][ BOTH ];  /**< VELOCITY_SCALER, FRICTION_COEFFICIENT and POSITIVE, NEGATIVE. */
      agx::Real m_restLength;                   /**< Rest length of the friction constraint. */
      agx::Real m_normalForceMagnitude;         /**< Magnitude of the normal force. */
      bool m_stick[ BOTH ];                     /**< NEGATIVE and POSITIVE */
      char m_goForward;                         /**< Bit mask for GO_FORWARD, IS_DIRTY... */
      bool m_activeConstraint;                  /**< Tells you if the friction constraint is active. */
      agx::Real m_violationPerUnitLength;       /**< Violation of the friction constraint */
  };

  /**
  Class to control variable node density along wires through routed nodes.
  All nodes contain a high resolution range.
  */
  class AGXPHYSICS_EXPORT HighResolutionRange
  {
    public:
      /**
      Default constructor. Start distance: 0, end distance: 0, nodes per unit distance: invalid.
      */
      HighResolutionRange();

      /**
      Function to activate the high resolution for a node in a wire.
      The wire will use the highest local resolution when updating its resolution.
      \param startDistance - defines a point at distance along the wire from the node where the local resolution starts. (could be +-)
      \param endDistance - defines a point at distance along the wire from the node where the local resolution ends. (could be +-)
      \param nodesPerUnitDistance - defines a local resolution within startDistance to endDistance
      */
      void set( agx::Real startDistance, agx::Real endDistance, agx::Real nodesPerUnitDistance );

      /**
      \return true if this node will affect the resolution of the wire the node is on
      */
      bool isActive() const;

      /**
      \return the start distance parameter
      */
      agx::Real getStartDistance() const;

      /**
      \return the end distance parameter
      */
      agx::Real getEndDistance() const;

      /**
      \return the number of nodes per unit distance parameter
      */
      agx::Real getNodesPerUnitDistance() const;

      /**
      Assign new start distance. Start distance defines a point at distance
      along the wire from the node where the local resolution starts. (could be +-)
      \param startDistance - new start distance (valid < current end distance)
      \note Assigning start distance higher than the current end distance will make the range invalid.
      */
      void setStartDistance( agx::Real startDistance );

      /**
      Assign new end distance. End distance defines a point at distance
      along the wire from the node where the local resolution ends. (could be +-)
      \param endDistance - new end distance (valid < current end distance)
      \note Assigning end distance lower than the current start distance will make the range invalid.
      */
      void setEndDistance( agx::Real endDistance );

      /**
      Assign new resolution (number of nodes per unit distance). Any real value larger than
      zero is valid. E.g., 0.5 (and unit meters) means one mass node every two meters.
      \note Assigning resolution to less or equal to zero will make the range invalid.
      */
      void setNodesPerUnitDistance( agx::Real nodesPerUnitDistance );

      /**
      Reverse range (distances).
      */
      void reverse();

      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      Save current state.
      */
      void store( agxStream::OutputArchive& out ) const;

      /**
      Restore from disc.
      */
      void restore( agxStream::InputArchive& in );

      DOXYGEN_END_INTERNAL_BLOCK()

     protected:
      agx::Real m_startDistance;
      agx::Real m_endDistance;
      agx::Real m_nodesPerUnitDistance;
  };

  /**
  Class defining a node that is part of a wire.
  */
  class AGXPHYSICS_EXPORT Node : public agx::Referenced, public virtual agxStream::Serializable
  {
    public:
      /**
      The supported node types.
      */
      enum Type
      {
        NOT_DEFINED = 0,      /**< Default type is undefined. */
        EYE         = (1<<0), /**< Eye nodes are nodes that slides along the wire. */
        MISSING     = (1<<1), /**< Node not used. */
        CONNECTING  = (1<<2), /**< Connecting nodes contains two nodes, one at center of mass
                                   and the other at the user defined position. Used to handle
                                   heavy loads and carries special constraints for stability. */
        FREE        = (1<<3), /**< Free nodes are used to route the wire. */
        CONTACT     = (1<<4), /**< Contact nodes may appear during contact with other objects. */
        BODY_FIXED  = (1<<5), /**< Body fixed node has a fixed position relative a parent (body or world). */
        STOP = (1 << 6),  /**< Stop nodes are an internal node used round winches. */
        SHAPE_CONTACT = (1<<7)  /**< Stop nodes are an internal node used round winches. */
      };

      /**
      Internal state a node may have.
      */
      enum StateInfo
      {
        STANDARD                    = 0,       /**< Default state. */
        WAS_CONTACT                 = (1<< 0), /**< If this node was a contact, it may live independent of the resolution. */
        IS_HIGH_RES                 = (1<< 1), /**< States if this node is a high resolution node. */
        IGNORE_BEND                 = (1<< 2), /**< States if the bend constraint should be forced to be disable over this node.*/
        NO_LUMP_NEAR                = (1<< 3), /**< If set, lumped/mass elements may not be closer than a predefined distance. */
        NO_CIRCLE_CONTACT           = (1<< 4), /**< States if this node can live on a circle (e.g., disc area on a cylinder). */
        GYPSY_CONTACT               = (1<< 5), /**< Internal state. */
        NO_MOVEMENT_RANGE           = (1<< 6), /**< States if this node can move along a contact edge or not. */
        NO_FRICTION                 = (1<< 7), /**< No friction constraints may be created from this node. */
        SPLIT_ENABLED               = (1<< 8), /**< Internal state. If set, this node splits the wire. */
        INSIDE_OBJECT               = (1<< 9), /**< Internal state. */
        WAS_CONTACT_LAST_TIMESTEP   = (1<<10), /**< If this node was contact last time step, it may not be removed for stability reasons. */
        PERMANENT_LUMP              = (1<<11), /**< This node has a permanent position in the wire and can not be removed by the wire it self. */
        REPLACE_SUPERBEND_WITH_BEND = (1<<12), /**< Internal state. */
        LUMPED_NODE                 = (1<<13), /**< Set if this node is an internal mass node of a wire. */
        LUMP_CONTACT                = (1<<14), /**< Internal state. */
        WW_STABILIZED               = (1<<15)  /**< Internal state. */
      };

    public:
      /**
      Class used to track nodes when CCD is used.
      */
      DOXYGEN_START_INTERNAL_BLOCK()
#ifndef SWIG
      class SpatialState
      {
        public:
          /**
          Data holder for spatial state used for CCD wire collision detection.
          */
          struct Data
          {
            /**
            Default constructor.
            */
            Data();

            /**
            Operator to sort the spatial state.
            */
            bool operator < ( const Data& other ) const;

            agx::Real t;
            agx::Vec3 pos;
          };

          typedef agx::VectorPOD< Data > DataContainer;

        public:
          /**
          Default constructor.
          */
          SpatialState();

          /**
          Public destructor.
          */
          ~SpatialState();

          /**
          Used to store state for the first node (a node that doesn't have
          a previous node.
          \param t - current tension
          \param curr - current node (curr->getSpatialState() == this)
          */
          void store( const agx::Real t, const agxWire::Node* curr );

          /**
          Store spatial state for this (next) node given current tension, total length of
          the wire and two nodes. (next->getSpatialState() == this)
          \param currT - current tension (0 < t < 1)
          \param totLen - total length of the system
          \param curr - current node
          \param next - next node (next->getSpatialState() == this)
          \return new currT
          */
          agx::Real store( const agx::Real currT, const agx::Real totLen, const agxWire::Node* curr, const agxWire::Node* next );

          /**
          Call when \p curr is being inserted. Intermediate points (before this) will be
          added to this and erased from next.
          */
          void onInsert( agxWire::Node* prev, agxWire::Node* curr, agxWire::Node* next );

          /**
          Call when \p curr is being removed. The current position of this node
          (\p curr) will be stored in \p next which means that this position
          will be an intermediate position for next spatial state.
          \param prev - previous node
          \param curr - current node (i.e., curr->getSpatialState() == this)
          \param next - next node
          */
          void onRemove( agxWire::Node* prev, agxWire::Node* curr, agxWire::Node* next );

          /**
          Sort intermediate points 0 < t < 1.
          */
          void sort();

          /**
          \return local data, i.e., the data for the current node
          */
          Data* local();

          /**
          \return local data, i.e., the data for the current node
          */
          const Data* local() const;

          /**
          \return the removed container
          */
          const DataContainer& removed() const { return m_removed; }

          /**
          Between two points (the actual node is last), get points between. I.e.,
          you are at currNode position and want to get all points between, including
          this node's state - call intermediate( i ) until 0 is returned. If
          intermediate( i ) == 0 you have that intermediate( i - 1 ) == local().
          \param i - index to intermediate point
          \return data to a point between two nodes - 0 if i - 1 == local()
          */
          const Data* intermediate( agx::UInt i ) const;

          /**
          Save current state to output stream.
          */
          void store( agxStream::OutputArchive& out ) const;

          /**
          Restore state from input stream.
          */
          void restore( agxStream::InputArchive& in );

        protected:
          Data m_local;
          DataContainer m_removed;
      };
#endif
      DOXYGEN_END_INTERNAL_BLOCK()

    public:
      /**
      Finds initial node state given geometry.
      \param geometry - the geometry
      \return the node state
      */
      static int findNodeState( const agxCollide::Geometry* geometry );

      static agx::Bool isContactType(const Node* node);
      static agx::Bool isContactType(const Node::Type& type);

      /**
      \return the type of this node
      */
      Node::Type getType() const;

      /**
      Test several types, e.g., node->isType( agxWire::Node::EYE | agxWire::Node::CONNECTING ).
      \return true if the type mask matches this nodes type
      */
      agx::Bool isType( agx::Int typeMask ) const;

      /**
      \deprecated Deprecated interface.

      \sa getType
      \return the type of this node
      */
      Type getNodeType() const;

      /**
      \return this nodes frame
      */
      NodeFrame* getFrame();

      /**
      \return this nodes frame
      */
      const NodeFrame* getFrame() const;

      /**
      \deprecated Deprecated interface.
      \sa getFrame
      \return this nodes frame
      */
      NodeFrame& getNodeFrame();

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      \return the current spatial state
      */
      SpatialState* getSpatialState();

      /**
      \return the current spatial state
      */
      const SpatialState* getSpatialState() const;
      DOXYGEN_END_INTERNAL_BLOCK()

      /**
      \return the world position of this node
      */
      const agx::Vec3& getWorldPosition() const;

      /**
      \return the original relative translate in the parent frame (if no body, world frame otherwise body frame)
      */
      const agx::Vec3& getTranslate() const;

      /**
      \deprecated Deprecated interface.
      */
      const agx::Vec3& getPosition() const; // support old code

      /**
      Set the relative translate of this node in parent frame (body frame if body is set, otherwise world frame)
      */
      virtual void setTranslate( const agx::Vec3& translate );

      /**
      Set the relative translate of this node in parent frame (body frame if body is set, otherwise world frame)
      */
      void setTranslate( agx::Real x, agx::Real y, agx::Real z );

      /**
      \return rigid body (0 if none) of this node
      */
      agx::RigidBody* getRigidBody() const;

      /**
      Only for internal use! Reverses all, internal, direction dependent structures.
      */
      virtual void reverse();

      /**
      Transforms the frame.
      */
      void transform();

      /**
      Set custom data to this node.
      \param customData - custom data
      */
      void setCustomData( agx::Referenced* customData );

      /**
      \return custom data
      */
      agx::Referenced* getCustomData() const;

      /**
      \return the high resolution range to control dynamic, or variable, node density in a wire
      */
      HighResolutionRange* getHighResolutionRange();

      /**
      \return the high resolution range to control dynamic, or variable, node density in a wire
      */
      const HighResolutionRange* getHighResolutionRange() const;

      /**
      Calculates the position of this node at time t from now.
      \param t - delta time from now
      \param onlyTranslational - if true (default) the angular velocity of the body will not be taken into account
      \return the world position of this node at current time + t
      \note The position gets less and less accurate for delta times larger than the time step.
      */
      agx::Vec3 getIntegratedPosition( agx::Real t, bool onlyTranslational = true ) const;

      /**
      \return the velocity of this node
      */
      virtual agx::Vec3 getVelocity() const;

      /**
      Flag if bend constraints should be ignored over this node.
      */
      void setIgnoreBend( bool ignoreBend );

      /**
      \return true if bend should be ignored over this node
      */
      bool getIgnoreBend() const;

      /**
      \return the current state mask
      */
      int getState() const;

      /**
      Add state mask to the current state.
      \param state - state to add
      */
      void addState( int state );

      /**
      Remove state mask from the current state.
      \param state - state to remove
      */
      void removeState( int state );

      /**
      \return the material
      */
      NodeMaterial* getMaterial();

      /**
      \return the material
      */
      const NodeMaterial* getMaterial() const;

      /**
      \return this node as free node (0 if this isn't an free node)
      */
      FreeNode* getAsFree();

      /**
      \return this node as free node (0 if this isn't an free node)
      */
      const FreeNode* getAsFree() const;

      /**
      \return this node as a body fixed node (0 if this isn't a body fixed node)
      */
      BodyFixedNode* getAsBodyFixed();

      /**
      \return this node as a body fixed node (0 if this isn't a body fixed node)
      */
      const BodyFixedNode* getAsBodyFixed() const;

      /**
      \return this node as eye node (0 if this isn't an eye node)
      */
      EyeNode* getAsEye();

      /**
      \return this node as eye node (0 if this isn't an eye node)
      */
      const EyeNode* getAsEye() const;

      /**
      \deprecated Deprecated interface.
      \return node as eye node (0 if this isn't an eye node)
      */
      EyeNode* getAsEyeLineNode();

      /**
      \deprecated Deprecated interface.
      \return node as eye node (0 if this isn't an eye node)
      */
      const EyeNode* getAsEyeLineNode() const;

      /**
      \return this node as connecting node (0 if this isn't a connecting node)
      */
      ConnectingNode* getAsConnecting();

      /**
      \return this node as connecting node (0 if this isn't a connecting node)
      */
      const ConnectingNode* getAsConnecting() const;

      /**
      \return this node as a stop node (0 if this isn't a stop node)
      */
      StopNode* getAsStop();

      /**
      \return this node as a stop node (0 if this isn't a stop node)
      */
      const StopNode* getAsStop() const;

      /**
      \return this node as a contact node (0 if this isn't a contact node)
      */
      ContactNode* getAsContact();

      /**
      \return this node as a contact node (0 if this isn't a contact node)
      */
      const ContactNode* getAsContact() const;

      /**
      \return this node as a shape contact node (0 if this isn't a contact node)
      */
      const ShapeContactNode* getAsShapeContact() const;

      /**
      \return this node as a shape contact node (0 if this isn't a contact node)
      */
      ShapeContactNode* getAsShapeContact();


      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      Internal method.
      Flag this node is inside an object.
      */
      void setInsideObject(bool insideObject);

      /**
      \return true if this node is inside an object
      */
      bool getInsideObject() const;

      /**
      \return true if this node has been a contact
      */
      bool getWasContact() const;

      /**
      \return true if this node was a contact last time step
      */
      bool getWasContactLastTimestep() const;

      /**
      Internal method.
      Flag this node that it has been a contact.
      */
      void setWasContact( bool wasContact );

      /**
      Internal method.
      Flag this node that it was contact last time step.
      */
      void setWasContactLastTimestep( bool wasContact );

      /**
      Flag if lumped nodes may be close to this node or not.
      */
      void setNoLumpNear( bool noLumpNear );

      /**
      \return true if this node has been flagged to not have lumps close
      */
      bool getNoLumpNear() const;

      /**
      Internal method.
      */
      void setNoCircleContact( bool disableCircleContact );

      /**
      Internal method.
      */
      bool getNoCircleContact() const;

      /**
      Internal method.
      A gypsy contact is defined by putting the shape translate at the middle of the edge of a cylinder.
      */
      void setGypsyContact(bool gypsyContact);

      /**
      Internal method.
      \return true if the contact should only be at the middle of the edge of a cylinder
      */
      bool getGypsyContact() const;

      /**
      Set if this contact node should have movement range or not. If no
      movement range is allowed this node will not move relative its
      geometry.
      \param disableMovementRange - true to disable movement range
      */
      void setNoMovementRange(bool disableMovementRange);

      /**
      \return true if movement range is disabled for this node
      */
      bool getNoMovementRange() const;

      /**
      Internal method.
      Flag if friction constraints may be generated due to this node.
      */
      void setNoFriction( bool noFriction );

      /**
      Internal method.
      \return true if no friction constraints may be generated from this node
      */
      bool getNoFriction() const;

      /**
      Internal method for store restoring. Don't use this unless you know exactly what you are doing.
      */
      void setState( int nodeState );

      /**
      Internal class used for communication during serialization procedures.
      I.e., we don't have to explicitly store this object.
      */
      class AGXPHYSICS_EXPORT StoreRestoreData
      {
        public:
          StoreRestoreData();

          bool& ignore();
          int& geometryIndex();
          bool& atBegin();

          bool getIgnored() const;
          int getGeometryIndex() const;
          bool getAtBegin() const;

          void reset();

        private:
          int m_geometryIndex;
          bool m_ignore;
          bool m_atBegin;
      };

      /**
      Internal method.
      */
      StoreRestoreData* getStoreRestoreData();

      /**
      Internal method.
      */
      const StoreRestoreData* getStoreRestoreData() const;


      AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::Node );
      DOXYGEN_END_INTERNAL_BLOCK()

      DOXYGEN_START_INTERNAL_BLOCK()

    public:
      struct AGXPHYSICS_EXPORT Tension
      {
        Tension() : raw( agx::Real( 0 ) ), smoothed( agx::Real( 0 ) ) {}
        agx::Real raw;
        agx::Real smoothed;
      };

      /**
      \return tension data
      */
      Tension* getTension();

      /**
      \return tension data
      */
      const Tension* getTension() const;

      struct AGXPHYSICS_EXPORT ContactForce
      {
        ContactForce() {}

        agx::Vec3 normalForce;
        agx::Vec3 frictionForce;
      };

      /**
      \return contact force
      */
      ContactForce* getContactForce();

      /**
      \return contact data
      */
      const ContactForce* getContactForce() const;

      /**
      \returns the normal force calculated by the wire tension controller.
      */
      agx::Vec3 getNormalForce() const;
DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Constructor given general node type.
      */
      Node( Type type );

      /**
      Default constructor.
      */
      Node();

      /**
      Destructor.
      */
      virtual ~Node();

    protected:
      NodeFrame                       m_nf;
      Type                            m_type;
      HighResolutionRange             m_highResolutionRange;
      NodeMaterial                    m_material;
      int                             m_nodeState;
      Tension                         m_tension;
      ContactForce                    m_contactForce;

      agx::ref_ptr< agx::Referenced > m_customData;

      FreeNode*                       m_free;
      BodyFixedNode*                  m_bodyFixed;
      EyeNode*                        m_eye;
      ConnectingNode*                 m_connecting;
      StopNode*                       m_stop;
      ContactNode*                    m_contact;
      ShapeContactNode*               m_shapeContact;

      StoreRestoreData                m_storeRestoreData;
      SpatialState                    m_spatialState;

      agx::observer_ptr< Node >       m_parent; // Pointer back if this is a child node.

    private:
      DOXYGEN_START_INTERNAL_BLOCK()
      friend class agx::InternalData;
      /**
      Internal method.
      \return merge-split data for this wire
      */
      agx::Referenced* getInternalData() const;

      /**
      Internal method.
      Associates merge-split data with this wire.
      */
      void setInternalData( agx::Referenced* data );
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      agx::ref_ptr<agx::Referenced> m_internalData;
  };

  /**
  Free node class used to route wires. This node is only defined by its
  world position, i.e., no parent transform is used.
  */
  class AGXPHYSICS_EXPORT FreeNode : public Node
  {
    public:
      /**
      Constructs given world position.
      \param position - a position given in world coordinates
      */
      FreeNode( const agx::Vec3& position );

      /**
      Constructs given world position.
      \param x - x position given in world coordinates
      \param y - y position given in world coordinates
      \param z - z position given in world coordinates
      */
      FreeNode( agx::Real x, agx::Real y, agx::Real z );

      AGXSTREAM_DECLARE_SERIALIZABLE(agxWire::FreeNode);

    protected:
      /**
      Not defined to create this node without a position.
      */
      FreeNode();

      /**
      Destructor.
      */
      virtual ~FreeNode();

    protected:
      /**
      Create method for all constructors.
      */
      void create( const agx::Vec3& position );
  };

  typedef agx::ref_ptr< FreeNode > FreeNodeRef;

  /**
  The body fixed node is attached to a body at a given offset. If the body
  is 0, this node type will be attached in world.
  \note Is it not defined to have this node type in the middle of a wire.
        I.e., as a user, it is only valid to have this node at the beginning
        and/or at end of a wire.
  \note This node type is used by the wire as mass/lumped nodes. The lumped nodes
        has a flag isLumpedNode() == true.
  */
  class AGXPHYSICS_EXPORT BodyFixedNode : public Node
  {
    public:
      /**
      Construct a body fixed wire node given a rigid body and relative position (rigid body frame).
      \param rb - rigid body this body fixed node belongs to (if 0, relativeTranslate is relative world frame)
      \param relativeTranslate - position of this node given in local coordinate system (default: At model center)
      */
      BodyFixedNode( agx::RigidBody* rb, const agx::Vec3& relativeTranslate = agx::Vec3() );

      /**
      Construct a body fixed wire node given a rigid body and relative position (rigid body frame).
      \param rb - rigid body this body fixed node belongs to
      \param relTranslate_x - x position of this node given in local coordinate system
      \param relTranslate_y - x position of this node given in local coordinate system
      \param relTranslate_z - x position of this node given in local coordinate system
      */
      BodyFixedNode( agx::RigidBody* rb, agx::Real relTranslate_x, agx::Real relTranslate_y, agx::Real relTranslate_z );

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Internal method.
      \note Manipulating this parameter without careful thoughts WILL cause undefined behaviors in a simulation.
      */
      void setRestLength( agx::Real restLength );

      /**
      Internal method.
      \note Manipulating this parameter without careful thoughts WILL cause undefined behaviors in a simulation.
      */
      void changeRestLength( agx::Real dl );

      /**
      Internal method.
      */
      agx::Real getRestLength() const;

      /**
      Internal method.
      \note Manipulating this parameter without careful thoughts WILL cause undefined behaviors in a simulation.
      */
      void setCurrentLength( agx::Real currentLength );

      /**
      Internal method.
      \note Manipulating this parameter without careful thoughts WILL cause undefined behaviors in a simulation.
      */
      void changeCurrentLength( agx::Real dl );

      /**
      Internal method.
      */
      agx::Real getCurrentLength() const;

      /**
      Internal method.
      */
      agx::Real getViolation() const;

      /**
      Internal method.
      */
      void setIsLumpedNode( bool lumpedNode );

      /**
      Internal method.
      */
      bool isLumpedNode() const;

      /**
      Internal method.
      */
      void setIsLumpContact( bool lumpContact );

      /**
      Internal method.
      */
      bool isLumpContact() const;

      /**
      Internal method.
      Enable split for this node.
      */
      void setSplitEnabled( bool splitEnabled );

      /**
      \return true if split is enabled for this node
      */
      bool getSplitEnabled() const;

      /**
      Internal method.
      Flag this node that it is within a high resolution region.
      */
      void setIsHighRes( bool wasHighRes );

      /**
      \return true if this node is within a high resolution region
      */
      bool getIsHighRes() const;

      /**
      Internal method.
      Change state 'permanent lump'.
      */
      void setPermanentLump( bool permantentLump );

      /**
      Internal method.
      \return true if this is a permanent lump
      */
      bool getPermanentLump() const;

      /**
      Internal method.
      Assign parent node.
      */
      void setParent( ConnectingNode* connectingNode );

      /**
      Internal method.
      \return parent connecting node (given this is the cm-node of the parent connecting node)
      */
      ConnectingNode* getParent() const;

      DOXYGEN_END_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE(agxWire::BodyFixedNode);

    protected:
      /**
      Protected default constructor.
      */
      BodyFixedNode();

      /**
      Destructor.
      */
      virtual ~BodyFixedNode();

      /**
      Create method used by all constructors.
      */
      void create( agx::RigidBody* rb, const agx::Vec3& relativeTranslate );

    protected:
      agx::Real m_restLength;
      agx::Real m_currentLength;
  };

  typedef agx::ref_ptr< BodyFixedNode > BodyFixedNodeRef;

  /**
  Eye node is a type of node that can slide along a wire with or without friction.
  */
  class AGXPHYSICS_EXPORT EyeNode : public Node
  {
    public:
      /**
      Construct a sliding node given a body and a radius. This node will
      be located at model center of the body.
      \param rb - the body this node should be attached to
      \param radius - radius of this eye node (default: 0)
      */
      EyeNode( agx::RigidBody* rb, agx::Real radius = agx::Real( 0 ) );

      /**
      Construct a sliding node given a body, model center offset and a radius.
      \param rb - the body this node should be attached to
      \param relativeTranslate - body model center offset
      \param radius - radius of this node (default: 0)
      */
      EyeNode( agx::RigidBody* rb, const agx::Vec3& relativeTranslate, agx::Real radius = agx::Real( 0 ) );

      /**
      Construct a sliding node given a body, model center offset and a radius.
      An approach to get the correct geometry of something sliding along the wire,
      pass the vector from \p relativeTranslate pointing out where an extra eye
      node will be created.
      \param rb - the body this node should be attached to
      \param relativeTranslate - body model center offset
      \param extraEyeVector - vector pointing out where to put an extra node to achieve plausible geometry of an eye node
      \param radius - radius of this node (default: 0)
      */
      EyeNode( agx::RigidBody* rb, const agx::Vec3& relativeTranslate, const agx::Vec3& extraEyeVector, agx::Real radius = agx::Real( 0 ) );

      /**
      Construct a sliding node given a body, model center offset and a radius.
      Optional; to get the correct geometry of something sliding along the wire,
      pass the vector from \p relativeTranslate pointing out where an extra eye
      node will be created.
      \param rb - the body this node should be attached to
      \param relTranslate_x - x body model center offset
      \param relTranslate_y - y body model center offset
      \param relTranslate_z - z body model center offset
      \param extraEyeVector - vector pointing out where to put an extra node to achieve plausible geometry of an eye node
      \param radius - radius of this node (default: 0)
      */
      EyeNode( agx::RigidBody* rb, agx::Real relTranslate_x, agx::Real relTranslate_y, agx::Real relTranslate_z, const agx::Vec3& extraEyeVector = agx::Vec3(), agx::Real radius = agx::Real( 0 ) );

      /**
      Set the radius of this eye node.
      \param radius - new radius (default 0)
      */
      void setRadius( agx::Real radius );

      /**
      \return the radius of this eye node (default 0)
      */
      agx::Real getRadius() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      \deprecated Deprecated interface.
      \sa setReverseAddd
      */
      void setEnableReverseAddOfNormalNode( bool reverseAdd );
      DOXYGEN_END_INTERNAL_BLOCK()

      /**
      If this node has a child node, this parameter controls in which order
      the nodes (parent and child) gets added/inserted into the wire. Default
      is that the parent node gets added before the child node.
      \param reverseAdd - if true, this child node will be added/inserted before the parent node
      */
      void setReverseAdd( bool reverseAdd );

      /**
      \return true if reverse add is enabled
      */
      bool getReverseAdd() const;

      /**
      \return child node if this node carries one
      */
      EyeNode* getChildNode();

      /**
      \return child node if this node carries one
      */
      const EyeNode* getChildNode() const;

      /**
      \return the parent node if this is a child
      */
      EyeNode* getParentNode() const;

      /**
      \return the distance from this node to its child node (if child present)
      */
      agx::Real getDistanceToChild() const;

      /**
      \return the relative translation from this node to its child node (if child present)
      */
      agx::Vec3 getVectorToChild() const;

      /**
      set the relative translation from this node to its child node (if child present)
      */
      void setVectorToChild( const agx::Vec3& vec );

      /**
      Only for internal use! Reverse all direction dependent internal structures.
      */
      virtual void reverse() override;

      /**
      Assign fixed friction direction for this eye node. This is usually
      used for eye nodes with infinite friction.
      \param fixFrictionDirection - true to enable fixed friction direction
      \param goForward - true if the friction constraints should be attached
                         before this node, false means after
      */
      void setFixFrictionDirection( bool fixFrictionDirection, bool goForward );

      /**
      \return true if this eye has fixed friction direction enabled
      */
      bool getFixFrictionDirection() const;

      /**
      Calculates the speed in which this eye node travels along a wire.
      \param wire - the wire containing this eye node
      \return speed of this node along wire (0 if not on wire or wire not initialized)
      */
      agx::Real findSpeedAlongWire( const agxWire::Wire* wire ) const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxWire::EyeNode);

    protected:
      /**
      Protected default constructor.
      */
      EyeNode();

      /**
      Protected constructor used by other nodes.
      \param type - node type
      \param radius - radius
      */
      EyeNode( Type type, agx::Real radius = agx::Real( 0 ) );

      /**
      Destructor.
      */
      virtual ~EyeNode();

      /**
      Create method used by all public constructors.
      \param rb - the body
      \param relativeTranslate - model frame offset
      \param extraEyeVector - if |v| > 0 an extra eye will be created
      \param radius - radius
      */
      void create( agx::RigidBody* rb, const agx::Vec3& relativeTranslate, const agx::Vec3& extraEyeVector, agx::Real radius );

    protected:
      agx::ref_ptr< EyeNode >       m_childNode;  // Extra eye.
      bool                          m_reverseAdd;
      agx::Real                     m_radius;
      agx::Vec3                     m_normal;
  };

  typedef agx::ref_ptr< EyeNode > EyeNodeRef;

  /**
  Connecting nodes contains two nodes, one at center of mass
  and the other at the user defined position. Used to handle
  heavy loads and this type carries special constraints for
  stability.
  */
  class AGXPHYSICS_EXPORT ConnectingNode : public EyeNode
  {
    public:
      /**
      Stability parameters object defining the behavior of the
      constraints associated to this connecting node.
      */
      struct AGXPHYSICS_EXPORT StabilityParameters
      {
        StabilityParameters( agx::Real youngsModulus, agx::Real damping, agx::Bool active );

        agx::Real youngsModulus;
        agx::Real damping;
        agx::Bool active;
      };

    public:
      /**
      Construct connecting node given body, model frame offset and radius.
      \param rb - body this node should be attached to
      \param relativeTranslate - model frame offset
      \param radius - radius of this node
      */
      ConnectingNode( agx::RigidBody* rb, const agx::Vec3& relativeTranslate, agx::Real radius );

      /**
      \return the extra node located at center of mass of the rigid body
      */
      BodyFixedNode* getCmNode() const;

      /**
      Internal method used by serialization to assign the
      stored center of mass node.
      \param cmNode - new center of mass node
      */
      void setCmNode( agxWire::BodyFixedNode* cmNode );

      /**
      \return the length from model center to this node
      */
      agx::Real getOffsetLength() const;

      /**
      Transform this and (if present) the center of mass node.
      */
      void fullTransform();

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Internal method.
      */
      void setSuperBendReplacedWithBend( bool replaceSuperBend );

      /**
      Internal method.
      */
      bool getSuperBendReplacedWithBend() const;

      /**
      Internal method.
      */
      virtual void updateConnection( agxWire::Wire* /*wire*/ ) {}

      /**
      Internal method.
      */
      virtual agxWire::ConnectingNode::StabilityParameters calculateStabilityParameters( agx::Real radius, agx::Real defaultYoungsModulus, agx::Real defaultDamping, const agxWire::Node* nodes[ 3 ], const agxWire::WireTensionController* tensionController ) const;

      DOXYGEN_END_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE(agxWire::ConnectingNode);

    protected:
      /**
      Protected default constructor.
      */
      ConnectingNode();

      /**
      Destructor.
      */
      virtual ~ConnectingNode();

      /**
      Create method used by public constructors.
      \param rb - the body
      \param relativeTranslate - model frame offset
      */
      void create( agx::RigidBody* rb, const agx::Vec3& relativeTranslate );

    protected:
      BodyFixedNodeRef m_cmNode;
  };

  typedef agx::ref_ptr< ConnectingNode > ConnectingNodeRef;

  /**
  Stop node, wire end node used by winches. So consider
  this node as internal.
  */
  class AGXPHYSICS_EXPORT StopNode : public EyeNode
  {
    public:
      /**
      Construct a stop node (mostly used internally). A stop node is a sliding node but will in some
      cases be handled in a separate way - for example to define begin or end of a wire.
      \param rb - body this node should be attached to
      \param relativeTranslate - model frame offset
      */
      StopNode( agx::RigidBody* rb, const agx::Vec3& relativeTranslate );

      /**
      Assign new value for the normal.
      \note This will not affect the translate of this node.
      */
      void setNormal( agx::Vec3 normal );

      /**
      \return the current normal
      */
      agx::Vec3 getNormal() const;


      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Internal method. Given this is a reference node, returns the distance between this node and the actual node.
      \return defined and constant reference distance (default: 1)
      */
      agx::Real getReferenceDistance() const;

      /**
      Internal method. Set the reference distance for this stop node.
      \param referenceDistance - new referenceDistance (default: 1)
      */
      void setReferenceDistance( agx::Real referenceDistance );

      DOXYGEN_END_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE(agxWire::StopNode);

    protected:
      /**
      Protected default constructor.
      */
      StopNode();

      /**
      Destructor.
      */
      virtual ~StopNode();

      /**
      Create method used by public constructors.
      \param rb - the body
      \param relativeTranslate - model frame offset
      */
      void create( agx::RigidBody* rb, const agx::Vec3& relativeTranslate );

    protected:
      agx::Real m_referenceDistance;
  };

  typedef agx::ref_ptr< StopNode > StopNodeRef;

  /**
  Contact node class is a node that, currently, is defined
  to live on an edge on a geometry (agxCollide::Geometry).
  */
  class AGXPHYSICS_EXPORT ContactNode : public Node
  {
    public:
      /**
      Clamp definition.
      */
      enum MovementRange
      {
        MOVEMENT_RANGE_START = 1, /**< Clamped at edge start. */
        MOVEMENT_RANGE_END   = 2  /**< Clamped at edge end. */
      };

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Internal class.
      */
      class AGXPHYSICS_EXPORT MeshIndex
      {
        public:
          MeshIndex();

          int getMeshCurrentTriangleIndex() const;
          int getMeshNextTriangleIndexStart() const;
          int getMeshNextTriangleIndexEnd() const;
          int getMeshCurrentEdgeIndex() const;
          int getMeshNextEdgeIndexStart() const;
          int getMeshNextEdgeIndexEnd() const;
          void setMeshCurrentIndex( int meshCurrentTriangleIndex, int meshCurrentEdgeIndex );
          void setMeshNextIndexStart( int meshNextTriangleIndexStart, int meshNextEdgeIndexStart );
          void setMeshNextIndexEnd( int meshNextTriangleIndexEnd, int meshNextEdgeIndexEnd );

          void reset();
          void store( agxStream::OutputArchive& out ) const;
          void restore( agxStream::InputArchive& in );

        private:
          int m_meshCurrentTriangleIndex;
          int m_meshCurrentEdgeIndex;
          int m_meshNextTriangleIndexStart;
          int m_meshNextEdgeIndexStart;
          int m_meshNextTriangleIndexEnd;
          int m_meshNextEdgeIndexEnd;
      };

      DOXYGEN_END_INTERNAL_BLOCK()

    public:
      /**
      Construct a contact node given geometry, translate, edge start and end and the
      edge direction (all points/vectors must be given in shape coordinate system).
      \param geometry - geometry with one shape
      \param shapeRelativeTranslate - translate relative the shape
      \param shapeEdgeStart - edge start given in shape coordinates
      \param shapeEdgeEnd - edge end given in shape coordinates
      \param shapeEdge - edge direction given in shape coordinates
      */
      ContactNode( agxCollide::Geometry* geometry, const agx::Vec3& shapeRelativeTranslate, const agx::Vec3& shapeEdgeStart, const agx::Vec3& shapeEdgeEnd, const agx::Vec3& shapeEdge );

      /**
      Like a copy constructor. This node will be a copy of \p cn.
      \param cn - contact node to copy
      */
      ContactNode( ContactNode* cn );

      /**
      Construct a contact node given geometry and relative translate in shape coordinates.
      Note that the translate will be corrected to fit the shape. A movement range and edge
      will be calculated when this node is added to a wire.
      \param geometry - a geometry with one supported shape (supported shapes: box and cylinder)
      \param shapeRelativeTranslate - translate in shape coordinates
      */
      ContactNode( agxCollide::Geometry* geometry, const agx::Vec3& shapeRelativeTranslate = agx::Vec3() );

      /**
      \return this the geometry this code is in contact with
      */
      agxCollide::Geometry* getGeometry() const;

      /**
      Assign new geometry to this node. The node state will be updated but
      not the relative offset.
      \param geometry - new geometry
      */
      void setGeometry( agxCollide::Geometry* geometry );

      /**
      Set translate of this contact wire node in shape coordinates.
      Internal method
      \param shapeTranslate - translate given in shape coordinate system
      */
      virtual void setTranslate( const agx::Vec3& shapeTranslate ) override;

      /**
      Enable this and the normal force vs gravity force test is ignored.
      */
      void setNoStickContact( bool noStickyContact );

      /**
      \return true if no stick contact is enabled
      */
      bool getNoStickContact() const;

      /**
      \return the mesh index class
      */
      MeshIndex* getMeshIndex();

      /**
      \return the mesh index class
      */
      const MeshIndex* getMeshIndex() const;


      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      Internal method
      \return this contact node next geometry
      */
      agxCollide::Geometry* getNextGeometry() const;

      /**
      Set new geometry to this contact node. This call will not invoke any other actions.
      Internal method
      \param nextGeometry - the new geometry
      */
      void setNextGeometry(agxCollide::Geometry* nextGeometry);

      /**
      \return translate in shape coordinates
      */
      const agx::Vec3& getShapeTranslate() const;

      /**
      Set the edge in shape coordinate system.
      */
      void setShapeEdge(const agx::Vec3& shapeEdge);

      /**
      \return the edge in shape coordinate system
      */
      const agx::Vec3& getShapeEdge() const;

      /**
      Resets the movement range given direction.
      */
      void resetMovementRange(int direction);

      /**
      Sets the bounds for how the contact node can be moved along the geometry the wire collided with.
      Any previous movement range including clamped edges will be overwritten.
      I.e., this could be the edge on a box or a wire parallel to a cylinder axis.
      */
      void setMovementRange(const agx::Vec3& start, const agx::Vec3& end);

      /**
      Returns the bounds for how the contact node can be moved on a wire by the wire
      solver and still be in contact with the geometry.
      */
      void getMovementRange(agx::Vec3& start, agx::Vec3& end) const;

      /**
      Returns the Full bounds for how the contact node can be moved on a wire by the
      solver and still be in contact with the geometry.
      */
      void getFullMovementRange(agx::Vec3& start, agx::Vec3& end) const;

      /**
      \return true if this node is stuck at a (by definition) open angle
      */
      bool getIfOpenAngle() const;

      /**
      Assign open angle flag.
      */
      void setOpenAngle(bool openAngle);

      /**
      \return the clamped range information
      */
      size_t getClampedRangeInformation() const;

      /**
      Returns the previous position of the node in shape coordinates.
      */
      const agx::Vec3& getPreviousPosition() const;

      /**
      Sets the previous position of the node in shape coordinates.
      */
      void setPreviousPosition(agx::Real t, agx::Vec3 start, agx::Vec3 edge);

      /**
      Sets the previous position to the current position
      */
      void resetPreviousPosition();

      /**
      Clamp movement range given position and direction.
      */
      bool clampMovementRange(const agx::Vec3& pos, MovementRange direction);

      /**
      Set offset vector from shape surface to shapeEdge, in shapeCoordinates
      */
      void setShapeEdgeOffset(const agx::Vec3& offset);

      /**
      \return the shape edge offset
      */
      agx::Vec3 getShapeEdgeOffset() const;

      /**
      A geometry on another body that the sensor collided with.
      */
      void setCollidedGeometry(agxCollide::Geometry* geometry, const agx::Vec3& depthVector);

      /**
      \return the list of geometries this contact node collided with
      */
      const GeometryRefDepthTable& getCollidedGeometries() const;

      /**
      Reset list of geometries this node collided with.
      */
      void resetCollidedGeometries();

      /**
      \return the shape coordinate position on the edge
      */
      agx::Vec3 getParameterizedPosition(agx::Real t) const;

      /**
      Find the 'time' along the edge given a shape position.
      */
      agx::Real findParameterizedValue(agx::Vec3 localPosition) const;

      /**
      Find the 'time' along the full edge (ignoring clamped information) given a shape position.
      */
      agx::Real findParameterizedValueFullRange(agx::Vec3 localPosition) const;


      /**
      Checks for removed geometries. Resets movement range if removed geometries found.
      */
      void analyzeCollidedGeometries();

      /**
      Copy collided geometries from one contact node to another.
      */
      void copyGeometries(agxWire::ContactNode* cn);

      /**
      \return the velocity calculated for this node last time step
      */
      agx::Real getEdgeVelocity() const;

      /**
      Assign edge velocity of this node.
      \param vel - new edge velocity
      */
      void setEdgeVelocity(agx::Real vel);

      /**
      \returns true if the contact node is on a geometry that allows dynamic wire contacts
      */
      bool getCreatedFromDynamicContact() const;

      /**
      \returns true if this and \p otherNodeIt are contacts and on same geometry
      */
      bool onSameGeometry( const agxWire::NodeConstIterator otherNodeIt ) const;

      /**
      \returns true if this and \p otherNodeIt are contacts and on same geometry
      */
      bool onSameGeometry( const agxWire::Node* otherNode ) const;

      /**
      Corrects the translate, calculates movement range and edge.
      \param wireRadius - radius of the wire
      \param wireRadiusMultiplier - radius multiplier
      \return true if this is a valid contact node
      */
      bool initialize( agx::Real wireRadius, agx::Real wireRadiusMultiplier );

      AGXSTREAM_DECLARE_SERIALIZABLE(agxWire::ContactNode);

      DOXYGEN_END_INTERNAL_BLOCK()


    protected:
      /**
      Protected default constructor.
      */
      ContactNode();

      /**
      Destructor.
      */
      virtual ~ContactNode();

      /**
      Create method all public constructors should use.
      \param geometry - geometry this node is relative to
      \param shapeRelativeTranslate - shape offset (shape coordinate system)
      \param shapeEdgeStart - start of edge in shape coordinates
      \param shapeEdgeEnd - end of edge in shape coordinates
      \param shapeEdge - direction of the edge in shape coordinates
      */
      void create( agxCollide::Geometry* geometry, const agx::Vec3& shapeRelativeTranslate, const agx::Vec3& shapeEdgeStart, const agx::Vec3& shapeEdgeEnd, const agx::Vec3& shapeEdge );

    protected:
      agx::observer_ptr< agxCollide::Geometry > m_geometry;
      agx::observer_ptr< agxCollide::Geometry > m_nextGeometry;
      agx::Vec3                                 m_shapeTranslate;
      agx::Vec3                                 m_shapeEdgeStart;
      agx::Vec3                                 m_shapeEdgeEnd;
      agx::Vec3                                 m_shapeEdge;
      agx::Vec3                                 m_shapeFullRangeStart;
      agx::Vec3                                 m_shapeFullRangeEnd;
      agx::Vec3                                 m_shapePreviousPosition;
      agx::Vec3                                 m_edgeOffsetFromSurface;
      GeometryRefDepthTable                     m_collidedGeometries;
      size_t                                    m_clamped;
      bool                                      m_openAngle;
      agx::Real                                 m_edgeVelocity;
      MeshIndex                                 m_meshIndex;
  };

  AGX_FORCE_INLINE const agx::Vec3& NodeFrame::getWorldPosition() const
  {
    return m_data[ TRANSLATE ];
  }

  AGX_FORCE_INLINE agx::RigidBody* NodeFrame::getRigidBody() const
  {
    return m_rb;
  }

  DOXYGEN_START_INTERNAL_BLOCK()
  AGX_FORCE_INLINE const Node::SpatialState::Data* Node::SpatialState::intermediate( agx::UInt i ) const
  {
    return i < m_removed.size() ? m_removed.ptr() + i : i == m_removed.size() ? &m_local : nullptr;
  }

  AGX_FORCE_INLINE Node::SpatialState::Data* agxWire::Node::SpatialState::local()
  {
    return &m_local;
  }

  AGX_FORCE_INLINE const Node::SpatialState::Data* agxWire::Node::SpatialState::local() const
  {
    return &m_local;
  }

  AGX_FORCE_INLINE agxWire::Node::SpatialState* Node::getSpatialState()
  {
    return &m_spatialState;
  }

  AGX_FORCE_INLINE const agxWire::Node::SpatialState* Node::getSpatialState() const
  {
    return &m_spatialState;
  }

  inline agx::Referenced* Node::getInternalData() const
  {
    return m_internalData;
  }

  DOXYGEN_END_INTERNAL_BLOCK()

  AGX_FORCE_INLINE Node::Type Node::getType() const
  {
    return m_type;
  }

  AGX_FORCE_INLINE agx::Bool Node::isType( agx::Int typeMask ) const
  {
    return (agx::Int( m_type ) & typeMask) != 0;
  }

  AGX_FORCE_INLINE NodeFrame* Node::getFrame()
  {
    return &m_nf;
  }

  AGX_FORCE_INLINE const NodeFrame* Node::getFrame() const
  {
    return &m_nf;
  }

  AGX_FORCE_INLINE const agx::Vec3& Node::getWorldPosition() const
  {
    return m_nf.getWorldPosition();
  }

  AGX_FORCE_INLINE agx::RigidBody* Node::getRigidBody() const
  {
    return m_nf.getRigidBody();
  }

  AGX_FORCE_INLINE BodyFixedNode* Node::getAsBodyFixed()
  {
    return m_bodyFixed;
  }

  AGX_FORCE_INLINE const BodyFixedNode* Node::getAsBodyFixed() const
  {
    return m_bodyFixed;
  }

  AGX_FORCE_INLINE EyeNode* Node::getAsEye()
  {
    return m_eye;
  }

  AGX_FORCE_INLINE const EyeNode* Node::getAsEye() const
  {
    return m_eye;
  }

  AGX_FORCE_INLINE ConnectingNode* Node::getAsConnecting()
  {
    return m_connecting;
  }

  AGX_FORCE_INLINE const ConnectingNode* Node::getAsConnecting() const
  {
    return m_connecting;
  }

  AGX_FORCE_INLINE StopNode* Node::getAsStop()
  {
    return m_stop;
  }

  AGX_FORCE_INLINE const StopNode* Node::getAsStop() const
  {
    return m_stop;
  }

  AGX_FORCE_INLINE ContactNode* Node::getAsContact()
  {
    return m_contact;
  }

  AGX_FORCE_INLINE const ContactNode* Node::getAsContact() const
  {
    return m_contact;
  }

  AGX_FORCE_INLINE const ShapeContactNode* Node::getAsShapeContact() const
  {
    return m_shapeContact;
  }

  AGX_FORCE_INLINE ShapeContactNode* Node::getAsShapeContact()
  {
    return m_shapeContact;
  }

  AGX_FORCE_INLINE ContactNode::MeshIndex* ContactNode::getMeshIndex()
  {
    return &m_meshIndex;
  }

  AGX_FORCE_INLINE const ContactNode::MeshIndex* ContactNode::getMeshIndex() const
  {
    return &m_meshIndex;
  }

  typedef agx::ref_ptr< ContactNode > ContactNodeRef;

}

