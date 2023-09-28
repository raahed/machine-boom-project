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

#include <agxWire/WireParameterController.h>

#include <agxWire/WireRouteController.h>
#include <agxWire/WireInitializationController.h>
#include <agxWire/WireWinchController.h>
#include <agxWire/WireStabilityController.h>
#include <agxWire/WireMaterialController.h>
#include <agxWire/WireGeometryController.h>
#include <agxWire/WireFrictionController.h>
#include <agxWire/WireSimplifyController.h>
#include <agxWire/WireTensionData.h>

#include <agxWire/WireSimulationProxy.h>
#include <agxWire/RenderIterator.h>

#include <agx/ICloneable.h>

namespace agxUtil
{
  class SmoothingFilter;
}

namespace agx
{
  class SolveIslandSplitAlgorithm;

  namespace Physics
  {
    class GraphNodeData;
  }
}

/// Implements a Wire model with adaptive resolution.
namespace agxWire
{
  class Winch;
  AGX_DECLARE_POINTER_TYPES( Wire );
  AGX_DECLARE_VECTOR_TYPES( Wire );

  /**
  Interface and placeholder of controllers/helpers for wires.
  */
  class AGXPHYSICS_EXPORT Wire : public SimulationProxyT< agxSDK::StepEventListener >
  {
    public:
      typedef SimulationProxyT< agxSDK::StepEventListener > base;

    public:
      /**
      Structure containing relevant tension data given one look-up. E.g., tension, smoothed tension...
      */
      struct AGXPHYSICS_EXPORT TensionData
      {
        TensionData();
        TensionData( agx::Real rawTension, agx::Real smoothedTension );
        TensionData( const agxWire::Node::Tension* tension );

        agx::Real raw;
        agx::Real smoothed;
      };

    public:

      agx::Real getContactNodeMomentum() const;

      /**
      \param rb - rigid body to check if a wire lumped node
      \return true if \p rb is a wire lumped node
      */
      static agx::Bool isLumpedNode( const agx::RigidBody* rb );

      /**
      \param rb - any rigid body
      \return the lumped node object if \p rb is a lumped node, otherwise null
      */
      static agxWire::BodyFixedNode* getLumpedNode( agx::RigidBody* rb );

      /**
      \param rb - any rigid body
      \return the lumped node object if \p rb is a lumped node, otherwise null
      */
      static const agxWire::BodyFixedNode* getLumpedNode( const agx::RigidBody* rb );

      /**
      \param rb - any rigid body
      \return wire if \p rb is an active wire lumped node, otherwise null
      */
      static agxWire::Wire* getWire( agx::RigidBody* rb );

      /**
      \param rb - any rigid body
      \return wire if \p rb is an active wire lumped node, otherwise null
      */
      static const agxWire::Wire* getWire( const agx::RigidBody* rb );

      /**
      \param geometry - any geometry
      \return wire if \p geometry is an active wire geometry, otherwise null
      */
      static agxWire::Wire* getWire( agxCollide::Geometry* geometry );

      /**
      \param geometry - any geometry
      \return wire if \p geometry is an active wire geometry, otherwise null
      */
      static const agxWire::Wire* getWire( const agxCollide::Geometry* geometry );

      /**
      Find wire instance given simulation and name.
      \param simulation - simulation
      \param name - name of wire instance to find
      \return wire instance with given name if found - otherwise null
      */
      static agxWire::Wire* find( agxSDK::Simulation* simulation, const agx::Name& name );

      /**
      Find wire instance given simulation and unique id.
      \param simulation - simulation
      \param uuid - UUID of wire instance to find
      \return wire instance with given UUID if found - otherwise null
      */
      static agxWire::Wire* find( agxSDK::Simulation* simulation, const agx::Uuid& uuid );

      /**
      Finds all agxWire::Wire instances in the given simulation.
      \param simulation - simulation
      \return wire instances found in the given simulation
      */
      static agxWire::WirePtrVector findAll( const agxSDK::Simulation* simulation );

      /**
      Finds all wire instances given simulation and name.
      \param simulation - simulation
      \param name - name of wire instances to find
      \return wire instances with given name
      */
      static agxWire::WirePtrVector findAll( const agxSDK::Simulation* simulation, const agx::Name& name );

    public:
      /**
      Constructor given the basic parameters to create a wire.
      \param radius - radius of this wire given in dimension of length
      \param resolutionPerUnitLength - resolution per unit length, i.e., how many discrete nodes that may appear per unit length (length = 10 m, resolutionPerUnitLength = 2 gives at maximum 20 nodes, length 10 m, resolutionPerUnitLength = 0.5 gives at maximum 5 nodes)
      \param enableCollisions - default true. If false, this wire can not collide with anything and will not handle or insert geometries
      */
      Wire( agx::Real radius, agx::Real resolutionPerUnitLength, bool enableCollisions = true );

      /**
      Clone this wire object and return an empty, uninitialized wire. Used during cut.
      \param newWire - if not null this object is interpreted as the clone (i.e., no 'new' will be created in clone)
      \return a new empty wire
      */
      virtual agxWire::Wire* clone( agxWire::Wire* newWire = nullptr );

      /**
      Add node to this wire. Note that this call only is valid during initial
      route, i.e., it is not possible to add nodes when this wire is initialized.
      \param node - node to add to this wire
      \return true if the node was added to this wire - otherwise false (line initialized for example)
      */
      virtual bool add( agxWire::Node* node );

      /**
      Add winch to this wire. Note that this call only is valid during initial route. I.e., it's not possible
      to add winches when this wire is initialized.
      \param winchController - winch to add
      \return true if the winch was successfully added to this wire - otherwise false
      */
      virtual bool add( agxWire::WireWinchController* winchController );

      /**
      Add winch to this wire. Note that this call only is valid during initial route. I.e., it's not possible
      to add winches when this wire is initialized.
      \param winchController - winch
      \param pulledInLength - length rolled up on the winch
      \return true if the winch was successfully added to this wire - otherwise false
      */
      virtual bool add( agxWire::WireWinchController* winchController, agx::Real pulledInLength );

      /**
      Add link to the route. Default behavior is that the route determines the connection type
      of the link connection and overwrites the old connection type. If \p updateConnectionType
      is false this call will fail if the connection types mismatches.
      \param link - link with predefined connection to this wire
      \param updateConnectionType - default true to update connection in the link given the current route of this wire, false to fail if connection type mismatch
      \return true if successful - otherwise false
      */
      virtual bool add( agxWire::Link* link, bool updateConnectionType = true );

      /**
      Add link to the route. This method will connect this wire to the link and add the
      link to the route, i.e., equivalent to:
        link->connect( wire, relativeTranslate, connectionType );
        wire->add( link );
      The connection type will be determined given the current route.
      \param link - link to connect to and add to route
      \param relativeTranslate - position in this links frame where the connection point is
      \return true if successful - otherwise false
      */
      virtual bool add( agxWire::Link* link, const agx::Vec3& relativeTranslate );

      /**
      Remove node from this wire.
      \param node - node to remove
      \return true if the node was successfully removed from this wire - otherwise false (node not in this wire for example)
      */
      virtual bool remove( agxWire::Node* node );

      /**
      Remove winch from this wire.
      \sa detach
      \param winchController - winch controller to remove
      \return if the winch controller was removed
      */
      virtual bool remove( agxWire::WireWinchController* winchController );

      /**
      Insert a node in this given distance along this wire. If distanceFromStart exceeds the current total route length the
      node will be added instead.
      \param node - node to insert
      \param distanceFromStart - distance from start, including pulled in length in begin winch, of this wire where the node should be inserted
      \param includeWinchPulledInLength - default true and should always be true unless a very special winch is used
      \return true if the node was successfully inserted into this wire - otherwise false
      \note If distance from start is less than the pulled in length the insert will fail because it's not possible to insert nodes before the first node
            or add to (insert after) the last node.
      */
      bool insert( agxWire::Node* node, agx::Real distanceFromStart, bool includeWinchPulledInLength = true );

      /**
      Insert a node given a point in world coordinate system. \p worldPoint will be projected onto the wire so the
      insert point will be the closest point on the wire to \p worldPoint.
      \param node - node to insert
      \param worldPoint - point in world
      \return true if the node was successfully inserted into this wire - otherwise false
      */
      bool insert( agxWire::Node* node, const agx::Vec3& worldPoint );

      /**
      Reverse this wire - use to match merge where first wire end is connected to second wire begin.
      \return true if successful reverse - otherwise false
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
      bool merge( agxWire::WireRef other );

      /**
      Cut this wire given distance from start (including winches). If successful cut, a new wire is
      returned with at least \p minimumResolution number of lumps. The new wire is added to the
      simulation if this wire was in a simulation.
      \param distanceAlongWire - distance along this wire where the cut should be made
      \param minimumResolution - minimum number of lumps the wires (this and the new) after cut (if very short for example)
      \param includeWinchPulledInLength - default true, if false any pulled in length in begin winch will be ignored
      \return new wire if cut was successful - otherwise 0
      */
      agxWire::WireRef cut(agx::Real distanceAlongWire, size_t minimumResolution, bool includeWinchPulledInLength = true);

      /**
      Cut this wire given any point in world. The cut position will be the one closest to \p worldPoint
      on this wire. If successful cut, a new wire is returned with at least \p minimumResolution number
      of lumps. The new wire is added to the simulation if this wire was in a simulation.
      \param worldPoint - any point in world
      \param minimumResolution - minimum number of lumps the wires (this and the new) after cut (if very short for example)
      \return new wire if cut was successful - otherwise 0
      */
      agxWire::WireRef cut( const agx::Vec3& worldPoint, size_t minimumResolution );

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
      Detach begin or end of this wire (if attached to something).
      \param begin - if true, begin of this wire will be detached - otherwise end
      \return true if a detach was performed - otherwise false
      */
      virtual bool detach( bool begin );

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
      \return true if this wire has been properly initialized - otherwise false
      */
      bool initialized() const;

      /**
      Assign initialization controller for this wire. If \p initializationController is zero
      the default controller will be assigned.
      \param initializationController - new initialization controller
      */
      void setInitializationController( agxWire::WireInitializationController* initializationController );

      /**
      Associate new, own implemented, route controller to this wire. Implement add, remove and insert in your own
      route controller to define the behaviors during these calls.
      \note This method can only be used before this wire has been initialized.
      \param routeController - new route controller, ignored if null
      \param copyCurrentRouteToNew - if true, and a route is partially done before this call, the routed nodes will
                                     be added to the new route
      \return true if the new route controller is accepted - otherwise false
      */
      bool setRouteController( agxWire::WireRouteController* routeController, bool copyCurrentRouteToNew = false );

      /**
      \return the current route controller
      */
      agxWire::WireRouteController* getRouteController() const;

      /**
      \return the wire parameter controller holding all parameters used to simulate this wire
      */
      agxWire::WireParameterController* getParameterController() const;

      /**
      \return the stability controller
      */
      agxWire::WireStabilityController* getStabilityController() const;

      /**
      \return the material controller
      */
      agxWire::WireMaterialController* getMaterialController() const;

      /**
      \return pointer to geometry controller
      */
      agxWire::WireGeometryController* getGeometryController() const;

      /**
      \return pointer to contact controller
      */
      agxWire::WireContactController* getActiveContactController() const;

      /**
      Calculates the rest length of this wire. If the winches are included their pulled in length will be added
      to the simulated rest length. Note that this wire doesn't have a rest length before it's initialized.
      \param includeWinches - if true, their pulled in length will be added to the simulated rest length
      \return the rest length of this wire, route length if this wire isn't initialized
      */
      virtual agx::Real getRestLength( bool includeWinches = true ) const;

      /**
      Calculates the rest length between two nodes (node1 and node2). Note that node1 has to be before node2
      in this wire, i.e., if you travel along the wire from the beginning, the first node of the two you
      meet is node1.
      \note It's not defined to call this method if this wire is uninitialized - return value will be -1 until
            the wire is initialized.
      \param node1 - first wire node
      \param node2 - second wire node
      \return the rest length between the node. If something goes wrong, the return value will be -1.
      */
      agx::Real getRestLength( const agxWire::Node* node1, const agxWire::Node* node2 ) const;

      /**
      Extended functionality, compared to getRestLength, to find rest length of this
      wire in a agxWire::Winch and agxWire::Link system.

      If this wire is attached to a winch and a link is being pulled in, getRestLength
      will either return a misleading or zero length but this method will find the
      actual rest length.
      \param includeWinches - if true, the winches pulled in length will be added to the simulated rest length.
      \return the rest length of this wire, 0 if this wire isn't initialized.
      */
      agx::Real findRestLength( bool includeWinches = true ) const;

      /**
      Calculates the current length of this wire (stretched/compressed included). If the winches are included their pulled in
      length will be added to the simulated current length.
      \param includeWinches - if true, their pulled in length will be added to the simulated current length
      \return the current length of this wire
      */
      agx::Real getCurrentLength( bool includeWinches = true ) const;

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
      Scale current length in world coordinates to rest length
      \param currentLength - length in world coordinates
      \param node - node where the scale should be calculated
      \return rest length for segment
      */
      agx::Real scaleToRestlength( agx::Real currentLength, const agxWire::Node* node ) const;

      /**
      Given node on this wire, calculates the transform from current length to rest length.
      \param node - node on this wire, close to where the scale is about to be used
      \return the scaler for scaling a current length to a rest length given the violation of the wire near the node
      */
      agx::Real getRestLengthScaler( const agxWire::Node* node ) const;

      /**
      Scale rest length to current length
      \param restLength - length in world coordinates
      \param node - node where the scale should be calculated
      \return current length for segment
      */
      agx::Real scaleToCurrentlength( agx::Real restLength, const agxWire::Node* node ) const;

      /**
      Given node on this wire, calculates the transform from rest length to current length.
      \param node - node on this wire, close to where the scale is about to be used
      \return the scaler for scaling a current length to a rest length given the violation of the wire near the node
      */
      agx::Real getCurrentLengthScaler( const agxWire::Node* node ) const;

      /**
      \return winch controller i, where i is 0 or 1, 0 means begin winch and 1 end. Returns zero if \p i is larger than
              2 or if there's no winch at the desired position.
      */
      agxWire::WireWinchController* getWinchController( size_t i ) const;

      /**
      \return the sum of all discrete node masses in this wire (i.e., the current, total simulated wire mass)
      */
      agx::Real getMass() const;


      /**
      \return the iterator to the first node of the visual part of this wire
      \note Check so that the rendering list is not empty before use of this method.
      \sa getRenderListEmpty, getRenderEndIterator
      */
      agxWire::RenderIterator getRenderBeginIterator() const;

      /**
      \return the iterator the node AFTER the last node that is part of the visual part of this wire
      \note Check so that the rendering list is not empty before use of this method.
      \sa getRenderListEmpty, getRenderBeginIterator
      */
      agxWire::RenderIterator getRenderEndIterator() const;

      /**
      \return true if the render list is empty - otherwise false
      */
      bool getRenderListEmpty() const;

      /**
      Given any point in 3D, this method will find the distance along this wire which is closest to the given point.
      \param point - any point
      \param includeWinchesPulledInLength - if true the algorithm will take the pulled in length, of the begin winch, into account
      \return the distance from start of this wire, closest to the given point
      */
      agx::Real findRestLengthFromStart( const agx::Vec3& point, bool includeWinchesPulledInLength = true ) const;

      /**
      Returns a point on this wire \p distanceFromStart meters from the beginning of this wire. Any winch (etc) that can hold/hide parts
      of this wire is included in the distance. I.e., if a winch, at begin, has 500 meters pulled in and distanceFromStart = 501,
      the point at 1 meter from the winch is returned.
      \param distanceFromStart - distance from start including winch pulled in length
      \return point in this wire \p distanceFromStart away
      */
      agx::Vec3 findPoint( agx::Real distanceFromStart ) const;

      /**
      Returns a point on this wire \p distanceFromStart meters from the beginning of this wire. If
      \p includesWinchesPulledInLength is true, \p distanceFromStart is assumed to include any pulled
      in wire at the beginning of this wire. If \p includesWinchesPulledInLength is false, the distance
      is assumed to be given from the first visible node of the wire.
      \param distanceFromStart - distance from start
      \param includesWinchesPulledInLength - true if \p distanceFromStart include a begin winch pulled in length,
                                             false if \p distanceFromStart does not include a begin winch pulled in length
      \return point in this wire \p distanceFromStart away
      */
      agx::Vec3 findPoint( agx::Real distanceFromStart, bool includesWinchesPulledInLength ) const;

      /**
      Given any point in 3D, this method will find a point on this wire which is closest to the given point.
      \param point - any point
      \return a point on this wire closest to \p point
      */
      agx::Vec3 findPoint( const agx::Vec3& point ) const;

      /**
      Given any point in 3D, this method will find a point on this wire which is closest to the given point and returns the
      current tension in that point.
      \param point - any point
      \return current tension closest to \p point
      */
      agxWire::WireSegmentTensionData getTension( const agx::Vec3& point ) const;

      /**
      Returns the current tension in a point given distance from start of this wire. Any winch (etc) that can hold/hide parts
      of this wire is included in the distance. I.e., if a winch, at begin, has 500 meters pulled in and distanceFromStart = 501,
      the tension at 1 meter from the winch is returned.
      \param distanceFromStart - distance from start including winch pulled in length
      \return current tension
      */
      agxWire::WireSegmentTensionData getTension( agx::Real distanceFromStart ) const;

      /**
      \return the current tension at the current position of this node
      */
      agxWire::WireNodeTensionData getTension( const agxWire::Node* node ) const;

      /**
      \return the material for this wire
      */
      agx::Material* getMaterial() const;

      /**
      Sets the material for this wire.
      \param material - new material
      */
      virtual void setMaterial( agx::Material* material );

      /**
      This determines if the wire should have a bend constraint or not
      \param isBendResistant - true if we want a bend constraint
      */
      void setIsBendResistent( bool isBendResistant );

      /**
      Property container for all rigid bodies and geometries that are owned by this wire.
      \param propertyContainer - new property container
      */
      void setPropertyContainer( agx::PropertyContainer* propertyContainer );

      /// \return a pointer to the PropertyContainer. If no one exists, a new will be created and returned
      agx::PropertyContainer* getPropertyContainer();

      /// \return a pointer to the PropertyContainer
      const agx::PropertyContainer* getPropertyContainer() const;

      /// \return true if it has an initialized PropertyContainer
      bool hasPropertyContainer() const;

      /**
      \return the radius of this wire
      */
      agx::Real getRadius() const;

      /**
      \return the radius of this wire including the offset distance for contact nodes
      */
      agx::Real getRadiusIncludingContactNodeOffsetDistanceMultiplier() const;

      /**
      Set the radius of this wire.
      \param radius - new radius > 0
      */
      virtual void setRadius( agx::Real radius );

      /**
      Assign new resolution per unit length.
      \param resolutionPerUnitLength - resolution per unit length, i.e., how many discrete nodes that may
                                       appear per unit length (length = 10 m, resolutionPerUnitLength = 2
                                       gives at maximum 20 nodes, length 10 m, resolutionPerUnitLength = 0.5
                                       gives at maximum 5 nodes)
      */
      void setResolutionPerUnitLength( agx::Real resolutionPerUnitLength );

      /**
      \return the current resolution per unit length
      */
      agx::Real getResolutionPerUnitLength() const;

      /**
      Enable or disable collisions between a geometry and this wire.
      \param geometry - geometry to enable/disable collisions against
      \param enable - if true, collisions will be enabled, if false, disabled
      */
      void setEnableCollisions( agxCollide::Geometry* geometry, bool enable );

      /**
      Enable or disable collisions between all geometries in a rigid body and this wire.
      \param rb - rigid body
      \param enable - if true, collisions will be enabled, if false, disabled
      */
      void setEnableCollisions( agx::RigidBody* rb, bool enable );

      /**
      Enable or disable this wire collision handling (i.e., if false, this wire will only contain bodies and constraints, no geometries)
      \param enable - if true, collision handling will be enabled, false disabled
      */
      void setEnableCollisions( bool enable );

      /**
      Choose the type of WireContactController to use
      Default is the WireShapeContactController.
      The alternative is WireOldContactController,
      */
      void setActiveContactControllerType(const WireContactControllerType type);
      WireContactControllerType getActiveContactControllerType() const;

      /**
      \return true if collisions is enabled for this wire - otherwise false
      */
      bool getEnableCollisions() const;

      /**
      \return true if collisions are enabled between the geometry and this wire - otherwise false
      */
      bool getEnableCollisions( const agxCollide::Geometry* geometry ) const;

      /**
      \return true if collisions are enabled between ALL the geometries in the rigid body and this wire - otherwise false
      */
      bool getEnableCollisions( const agx::RigidBody* rb ) const;

      /**
      Enable/Disable a group (unique group id) against all wires.
      */
      static void setEnableGroupWireCollision(agxCollide::Space* space,const agx::UInt32 group, bool enable);

      /**
      enable/disable the ability to automatically split the wire at a body fixed node.
      That means temporarily making that body fixed node kinematic, and ignoring the bend constraint with that node in the middle.

      Internal method chooses the body fixed nodes, and if it is possible to split.
      */
      void setEnableSplitting( bool splitting );

      /**
      \return true - if wire has splitting enabled
      */
      bool getEnableSplitting() const;

      /**
      Add group for all geometries handled by this collision controller.
      \param id - id for the new group
      */
      void addGroup( uint32_t id );

      /**
      Add group for all geometries handled by this collision controller.
      \param name - name for the new group
      */
      void addGroup( const agx::Name& name );

      /**
      Find if wire geometries contains group
      */
      bool hasGroup( const uint32_t id ) const;

      /**
      Find if wire geometries contains group
      */
      bool hasGroup( const agx::Name& name ) const;

      /**
      Remove group for all geometries handled by this collision controller.
      \param id - id for the group to remove
      */
      void removeGroup( uint32_t id );

      /**
      Remove group for all geometries handled by this collision controller.
      \param name - name for the group to remove
      */
      void removeGroup( const agx::Name& name );

      /**
      \return true if the (cylinder or capsule) geometry belongs to this wires geometries - false otherwise
      */
      bool hasGeometry( const agxCollide::Geometry* geometry ) const;

      /**
      \return true if this wire has \p node - otherwise false
      */
      bool hasNode( const agxWire::Node* node ) const;

      /**
      \return the first node in this wire - null if no nodes
      */
      agxWire::Node* getFirstNode() const;

      /**
      \return the last node in this wire - null if no nodes
      */
      agxWire::Node* getLastNode() const;

      /**
      \return the constraint used by this line
      */
      agxWire::WireDistanceCompositeConstraint* getConstraint() const;

      /**
      Fills \p geoms with all geometries used by this wire.
      \param geoms - geometry vector to fill
      */
      void getGeometries( agxCollide::GeometryPtrVector& geoms ) const;

      /**
      Removes all geometries
      */
      void removeGeometries();

      /**
      Updates all node transforms.
      */
      void updateNodeTransforms();

      /**
      This wire has a frame which is the parent frame of all internal bodies.
      \return frame of this wire
      */
      agx::Frame* getFrame() const;

      /**
      Assign new parent frame to this wire. All objects in this wire will inherit this parent transform.
      */
      void setFrame( agx::Frame* frame );

      /**
      Assigns linear velocity damping to all lumped nodes (i.e., rigid bodies owned by this wire) in
      this wire. Since all rigid bodies are particles it's defined to have direction dependent damping.
      \param linearVelocityDamping - linear velocity damping > 0
      */
      void setLinearVelocityDamping( agx::Real linearVelocityDamping );

      /**
      \return current linear velocity damping used
      */
      agx::Real getLinearVelocityDamping() const;

      /**
      \return true if wire has contact nodes on \p rb
      */
      bool hasContactNodesOn( const agx::RigidBody* rb ) const;

      /**
      \return true if wire has contact nodes on \p geometry
      */
      bool hasContactNodesOn( const agxCollide::Geometry* geometry ) const;

      /**
      Given \p currentBody, go through all bodies related to this Wire and replace them with \p newBody. If
      \p geom != nullptr then only operate on the  contactnodes which is associated with the specified geometry (geom).
      \param currentBody - The body currently present in the LineComposite
      \param newBody - The body that will replace the currentBody
      \param geometry -
      \return true if currentBody was successfully replaced.
      */
      bool replaceBodyInNodes( agx::RigidBody* currentBody, agx::RigidBody* newBody, agxCollide::Geometry* geometry = nullptr );

      /**
      Utility method to change rigid body in any node, given it has a valid node type. Valid node types are:
      agxWire::Node::BODY_FIXED, agxWire::Node::CONNECTING and agxWire::Node::EYE.

      Notes:
      BODY_FIXED: If this fixed node is part of a connecting node (cm-node) the body will be changed in the connecting node as well.
      CONNECTING: If connecting node its center of mass node (fixed) will also change rigid body.
      EYE: If eye node, and "double eye" was created, both eye nodes will change rigid body.

      The node will have the same position in world coordinates after the call to this method. I.e., a new local
      offset is calculated as well.
      \param node - node to change rigid body on (see "Notes" above)
      \param newRigidBody - the new rigid body (if 0, the node will be connected in world)
      \return true if the change were successful
      */
      bool changeRigidBody( agxWire::Node* node, agx::RigidBody* newRigidBody ) const;

      /**
      Assign smoothed tension (can be useful after certain events, like cut) to this wire.
      \param newSmoothedTension - the new smoothed tension
      */
      void setSmoothedTension( agx::Real newSmoothedTension );

      /**
      Assign smoothed tension (can be useful after certain events, like cut) to part of this wire.
      \param newSmoothedTension - the new smoothed tension
      \param from - from node
      \param to - to node
      */
      void setSmoothedTension( agx::Real newSmoothedTension, agxWire::NodeIterator from, agxWire::NodeIterator to );

      /**
      Assign filter that will be used to filter out the tension values for smoothed tension
      readings. If no filter is set (or 0) in the argument, no filter will be used.
      \param filter - filter which will be used to smooth the smoothed tension values
      */
      void setSmoothingFilter( agxUtil::SmoothingFilter *filter );

      /**
      \return the current smoothing filter used
      */
      const agxUtil::SmoothingFilter* getSmoothingFilter() const;

      /**
      Update geometries.
      */
      virtual void preCollide( const agx::TimeStamp& ) override;

      /**
      Run stability update
      */
      virtual void pre( const agx::TimeStamp& t ) override;

      /**
      Clean up after splitting
      */
      virtual void post( const agx::TimeStamp& t ) override;

      /**
      Simplify this wire.
      \return true if simplified (successful)
      */
      bool simplify( const agx::Frame* parentFrame = nullptr );

      /**
      Unsimplify this wire.
      \return true if unsimplified (successful)
      */
      bool unsimplify();

      /**
      \return true if this wire is simplified
      */
      bool isSimplified() const;

      /**
      \return the simplify controller used by this wire
      */
      agxWire::WireSimplifyController* getSimplifyController() const;

      /**
      Set solve type, either DIRECT, ITERATIVE or DIRECT_AND_ITERATIVE.
      */
      void setSolveType( agx::Constraint::SolveType solveType);

      /**
      Get pointer to solve algorithm
      */
      agx::SolveIslandSplitAlgorithm* getSplitAlgorithm() const;

      /**
      Shorten this wire to a given node. All nodes from this wire's end to \p toNode
      will be removed.
      \note The begin/end has to be free and this wire has to be initialized.
      \param toNode - node to short this wire to
      \param fromBegin - true if shorten this wire from begin, false from end
      \return true if successful - otherwise false
      */
      bool shorten( const agxWire::Node* toNode, bool fromBegin );

      /**
      Replace all contact nodes with shape contact nodes
      */
      void replaceContactNodesWithShapeContacts();

      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      Internal method. Uninitialize and deletes the constraint. I.e., this wire has to
      be routed and initialized again after call to this method.
      */
      void uninitializeConstraint();

      /**
      Internal method.
      A wire is by definition static if first and last lumped node is static. A wire
      that is static will not receive any update callbacks.
      \return true if this wire is, by definition, static
      */
      bool isStatic() const;

      /**
      \return the current number of nodes in this wire (explicit and implementation dependent - for internal or relative use only)
      */
      agx::UInt getNumNodes() const;

      /**
      Internal method.
      Explicitly replace winch controller at begin or end with \p connectingNode. The winch
      controller will not be uninitialized, i.e., the stop node will still be part of
      this wire.
      \note This wire has to be initialized.
      \param connectingNode - connecting node to replace winch controller
      \param begin - true to replace winch controller at begin - false for at end
      \return true if successful - otherwise false
      */
      bool explicitReplaceWinchController( agxWire::ConnectingNode* connectingNode, bool begin );

      /**
      Internal method.
      Callback when merge split properties have been instantiated for this wire.
      */
      void onCreateMergeSplitProperties();

      /**
      Internal method.
      \return our property controller
      */
      agxWire::WirePropertyController* getPropertyController() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::Wire );
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Serializable. Default constructor.
      */
      Wire();

      /**
      Reference counted class, protected destructor.
      */
      virtual ~Wire();

      /**
      Called when this wire is added to a simulation, if this wire hasn't been initialized before this call
      it will initialize it self during this call.
      */
      virtual void addNotification() override;
      using SimulationProxyT< agxSDK::StepEventListener >::addNotification;

      /**
      Called when this wire is removed from the simulation.
      */
      virtual void removeNotification() override;
      using SimulationProxyT< agxSDK::StepEventListener >::removeNotification;

      /**
      Loop through all nodes and update mass (and material NOT IMPLEMENTED)
      */
      void updateMaterial();

      /**
      Hide eventListenerType getType()
      */
#ifndef SWIG
      agxSDK::EventListener::Type getType() const;
#endif

      friend class agxWire::Winch;
      /**
      Assign new winch controller to this wire.
      \param winchController - new winch controller
      \param begin - true if new winch controller is at begin, false at end
      */
      virtual void setWinchController( agxWire::WireWinchController* winchController, bool begin );

      /**
      Return pointer to the attachment controller used at begin of the wire constraint.
      \returns - the begin attachment.
      */
      agxWire::WireAttachmentController* getBeginAttachment();

      /**
      Return pointer to the attachment controller used at end of the wire constraint.
      \returns - the end attachment.
      */
      agxWire::WireAttachmentController* getEndAttachment();

    protected:
      WireDistanceCompositeConstraintRef  m_constraint;

      // Controllers
      WireRouteControllerRef              m_routeController;
      WireInitializationControllerRef     m_initializationController;
      WireParameterController             m_parameterController;
      WireWinchControllerRefVector        m_winchControllers;          /**< Winch controller vector, always of size 2, default both 0 */
      WireMaterialControllerRef           m_materialController;
      WireStabilityControllerRef          m_stabilityController;
      WireGeometryControllerRef           m_geometryController;
      WireFrictionControllerRef           m_frictionController;
      WireSimplifyControllerRef           m_simplifyController;

      agxUtil::SmoothingFilterRef         m_smoothingFilter;
      bool                                m_initialized;
      agx::FrameRef                       m_frame;
      agx::Real                           m_linearVelocityDamping;     /**< Only linear velocity damping as real is defined for lumped nodes. */
      bool                                m_splitEnabled;
      agx::Constraint::SolveType          m_solveType;

      agx::ref_ptr<agx::Referenced>       m_splitAlgorithm;
      agx::ref_ptr<agx::Referenced>       m_internalData;
      agx::ref_ptr<agx::Referenced>       m_propertyController;
      mutable agx::PropertyContainerRef   m_propertyContainer;

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
  };

  // Inline methods *******************************************************************************************

  inline bool Wire::initialized() const
  {
    return m_initialized;
  }

  inline WireParameterController* Wire::getParameterController() const
  {
    return const_cast< WireParameterController* >( &m_parameterController );
  }

  inline WireRouteController* Wire::getRouteController() const
  {
    return m_routeController;
  }

  inline WireStabilityController* Wire::getStabilityController() const
  {
    return m_stabilityController;
  }

  inline WireMaterialController* Wire::getMaterialController() const
  {
    return m_materialController;
  }

  inline WireGeometryController* Wire::getGeometryController() const
  {
     return m_geometryController;
  }

  inline WireDistanceCompositeConstraint* Wire::getConstraint() const
  {
    return m_constraint;
  }

  inline agx::Referenced* Wire::getInternalData() const
  {
    return m_internalData;
  }

  AGX_FORCE_INLINE agx::PropertyContainer* Wire::getPropertyContainer()
  {
    if (!m_propertyContainer)
      m_propertyContainer = new agx::PropertyContainer;
    return m_propertyContainer;
  }

  AGX_FORCE_INLINE bool Wire::hasPropertyContainer() const
  {
    return (m_propertyContainer.isValid());
  }

  AGX_FORCE_INLINE const agx::PropertyContainer* Wire::getPropertyContainer() const
  {
    if (!m_propertyContainer)
      m_propertyContainer = new agx::PropertyContainer;

    return m_propertyContainer;
  }

  // **********************************************************************************************************

  // Static utility functions *********************************************************************************
  /**
  Extracts all the nodes given the render list in the wire. I.e., only "visible" nodes will be added.
  \param nodes - container to fill with nodes
  \param wire - a wire
  */
  AGXPHYSICS_EXPORT void extractNodes( NodePtrVector& nodes, const Wire* wire );
  // **********************************************************************************************************
}

