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

#include <agxWire/WireMaterialController.h>
#include <agxWire/WireParameterController.h>
#include <agxWire/WireImpactController.h>

#include <agxCollide/WireShape.h>

#include <agx/LinearProbingHashTable.h>
#include <agx/LinearProbingHashSet.h>

DOXYGEN_START_INTERNAL_BLOCK()

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 6011) // Disable warningC6011: dereferencing nullptr pointer
#endif


namespace agxWire
{
#define AGX_USE_WIRE_CCD 1

#if AGX_USE_WIRE_CCD
#  define AGX_WIRE_SHAPE agxCollide::WireShape
#  define AGX_WIRE_SHAPE_TYPE agxCollide::Shape::WIRE_SHAPE
#else
#  define AGX_WIRE_SHAPE agxCollide::Capsule
#  define AGX_WIRE_SHAPE_TYPE agxCollide::Shape::CAPSULE
#endif

  // Typedefs
  typedef agx::HashVector< agx::RigidBody*, agxCollide::GeometryRef > RigidBodyPtrGeometryRefTable;
  typedef agx::HashVector< agxCollide::Geometry*, agx::observer_ptr< agxCollide::Geometry > > GeometryPtrGeometryObsTable;
  typedef std::pair< agxWire::Node*, agxWire::Node* > NodePtrPair;

#include <HashImplementationSwitcher.h>
#if HASH_FOR_WIRE == HASH_NEW
  using GeometryPtrNodeObsPtrPairTable = agx::LinearProbingHashTable<const agxCollide::Geometry*, NodePtrPair>;
  using ContactNodePtrGeometryRefTable = agx::LinearProbingHashTable<ContactNode*, agxCollide::GeometryRef>;
  using GeometryPtrContactNodePtrTable = agx::LinearProbingHashTable<agxCollide::Geometry*, ContactNode*>;
  using GeometryPtrHashSet = agx::LinearProbingHashSet<agxCollide::Geometry*>;
  using UnsignedHashSet = agx::LinearProbingHashSet<uint32_t>;
  using NameHashSet = agx::LinearProbingHashSet<agx::Name>;
#elif HASH_FOR_WIRE == HASH_OLD
  typedef agx::QuadraticProbingHashTable< const agxCollide::Geometry*, NodePtrPair > GeometryPtrNodeObsPtrPairTable;
  typedef agx::QuadraticProbingHashTable< ContactNode*, agxCollide::GeometryRef> ContactNodePtrGeometryRefTable;
  typedef agx::QuadraticProbingHashTable< agxCollide::Geometry*, ContactNode*> GeometryPtrContactNodePtrTable;
  typedef agx::HashSet< agxCollide::Geometry* > GeometryPtrHashSet;
  typedef agx::HashSet< uint32_t > UnsignedHashSet;
  using NameHashSet = agx::HashSet<agx::Name>;
#else
  #error
#endif

  // Forward declarations
  class WireParameterController;
  class Wire;
  class WireContactController;
  class WireStabilityController;
  class WireShapeContactController;
  class WireOldContactController;
  class ShapeContactColliderUtils;

  template< typename T >
  class AGXPHYSICS_EXPORT ConfigureGeometryStruct
  {
  public:
    /**
    \returns true if configured properly
    */
    static bool configureWireGeometry(agxCollide::Geometry* wireGeometry, const agx::Vec3& pStart, const agx::Vec3& pEnd, const agx::Vec3& oldPStart, const agx::Vec3& oldPEnd, agx::Real radius, agxCollide::BoundingAABB& ret_localBound);
  };

  enum WireContactControllerType
  {
    OLD_CONTACT_CONTROLLER = 0,
    SHAPE_CONTACT_CONTROLLER
  };

  /**
  Geometry controller that updates the geometries in the wire given current node configuration.
  */
  class AGXPHYSICS_EXPORT WireGeometryController : public WireListener
  {
    public:
      static void onEnableCollisions( agxWire::Wire* wire1, agxWire::Wire* wire2, agx::Bool enable );
      static agx::Real getGlobalWireRadiusScaleIncrement();

    public:
      /**
      Arguments: ( agxCollide::Geometry* geometry, agxWire::WireGeometryType geometryType, agx::Bool enabled )
      */
      typedef std::function<void( agxCollide::Geometry*, agxWire::WireGeometryType, agx::Bool )> GeometryVisitor;

    public:
      // Geometry groups for the spheres and for the wire geometries. These two groups can't collide.
      enum GeometryGroups
      {
        WIRE_SPHERE_GEOMETRY_GROUP = 170000, /**< Wire sphere shapes tag (i.e., a geometry with this group has a sphere shape created by a wire). */
        WIRE_WIRE_GEOMETRY_GROUP,            /**< Wire shape tag (i.e., a geometry with this group has a wire shape created by a wire). */
        WIRE_DYNAMIC_CONTACTS_GROUP,         /**< Geometries with this group has 'dynamics contact model' enabled. */
        WIRE_SENSOR_SPHERE_CONTACTS_GROUP    /**< Geometries with this group has 'dynamics contact model' enabled. */
      };

      /**
      Default, and only, constructor.
      */
      WireGeometryController();

      /**
      Set the type of wire contact controller used for contacts
      created by geometries controlled by this geometry controller.
      */
      void setActiveContactControllerType(Wire* wire, const WireContactControllerType type);
      WireContactControllerType getActiveContactControllerType() const;


      const ShapeContactColliderUtils* getShapeContactControllerUtils() const;

#ifndef SWIG
      /**
      Traverse all geometries created by this geometry controller. If \p onlyEnabled
      is true, only geometries currently used by the wire are traversed, otherwise all (pooled).
      \param visitor - wire geometry visitor
      \param onlyEnabled - true to only traverse geometries in use by the wire, false to traverse all
      */
      void traverse( GeometryVisitor visitor, agx::Bool onlyEnabled ) const;
#endif

      /**
      Call from the wire when it's time to update this controller and all the controllers
      this controller is responsible for. Do stuff that has to be done before collision detection.
      */
      void preCollideUpdate();

      /**
      Call from the wire when it's time to update this controller and all the controllers
      this controller is responsible for. Do stuff where we need info from collision detection.
      Is responsible for the stability update of the lumped nodes
      returns true - if it has done the stability update.
      */
      agx::Bool preUpdate( );

      /**
      Updates internal state of wire configuration (for collision detection).
      */
      void updateState();

      /**
      Set the wire constraint.
      \param wire - the wire
      \param space - space to enable/disable collisions
      \param simulationProxy - simulation proxy to use during this session
      */
      virtual void setWire( agxWire::Wire* wire, agxCollide::Space* space, agxSDK::SimulationProxy* simulationProxy );

      /**
      Remove notification, remove geometries and clear internal structures.
      */
      virtual void removeNotification();

      /**
      Update the geometries given a node list.
      */
      void updateGeometries();

      /**
      Updates the material of all wire geometries
      */
      void updateMaterial( agx::Material* material );

      /**
      Initializes internal structures.
      */
      void initialize();

      /**
      Removes the wire-, node- and sensor geometries from the simulation.
      \param safeRemoveAll - if true, calls to simulation->remove will be done for all geometries in the pools as well
      */
      void removeGeometries( bool safeRemoveAll = false );

      /**
      Force add/remove a sphere on a lumped node.
      */
      agx::Bool forceAddRemoveSphereGeometryOnLumpedNode(agxWire::BodyFixedNode* bfn, agx::Bool add);

      /**
      Enable or disable collisions between the wire and a geometry.
      \param geometry - geometry to enable/disable collisions with
      \param enable - if true the collisions will be enabled, if false - disabled
      */
      bool setEnableCollisions( agxCollide::Geometry* geometry, bool enable );

      /**
      \return true if collisions are enabled against this wire and the geometry
      */
      bool getEnableCollisions( const agxCollide::Geometry* geometry ) const;

      /**
      Copy disabled geometries settings and group id's from other wire geometry controller.
      \param otherLGC - the other wire geometry controller
      */
      void copyCollisionSettings( agxWire::WireGeometryController* otherLGC );

      /**
      Collects all geometries (sphere and wire, not sensors) in this wire.
      */
      void getGeometries( agxCollide::GeometryPtrVector& geoms );

      /**
      Add group for all geometries handled by this collision controller.
      \param id - id for the new group
      */
      void addGroup( agx::UInt32 id );

      /**
      Add group for all geometries handled by this collision controller.
      \param name - name for the group
      */
      void addGroup( const agx::Name& name );

      /**
      Find if wire geometries contains group
      */
      bool hasGroup( const agx::UInt32 id ) const;

      /**
      Find if wire geometries contains group
      */
      bool hasGroup( const agx::Name& name ) const;

      /**
      Remove group for all geometries handled by this collision controller.
      \param id - id for the group to remove
      */
      void removeGroup( agx::UInt32 id );

      /**
      Remove group for all geometries handled by this collision controller.
      \param name - name for the group to remove
      */
      void removeGroup( const agx::Name& name );

      /**
      \return true if rb has contact nodes on some geometry
      */
      bool hasContactNodesOn( const agx::RigidBody* rb ) const;

      /**
      Loop over all contact nodes, transform them and update their related geometries.
      */
      void updateSensorTransforms();

      /**
      Enable or disable the sensor mapped to a contact wire node.
      \param cln - the contact wire node
      \param enable - enable flag
      */
      void enableContactWireNodeSensor( agxWire::ContactNode* cln, bool enable );

      /**
      \return true if the (cylinder or capsule) geometry belongs to this wires geometries - false otherwise
      */
      bool hasGeometry( const agxCollide::Geometry* geometry ) const;

      /**
      \return the unique ID for contact listeners to use to only get collisions from this wire (WireGeometryFilter)
      */
      inline agx::UInt32 getWireUniqueGroupID() const { agxAssert( m_wireUniqueGroupID ); return m_wireUniqueGroupID; }

      /**
      \return the unique ID for contact listeners to use to only get collisions from this wire (WireGeometryFilter)
      */
      inline agx::UInt32 getSensorUniqueGroupID() const { agxAssert( m_sensorUniqueGroupID ); return m_sensorUniqueGroupID; }

      /**
      \return the id this wire uses to disable collisions against other objects
      */
      agx::UInt32 getDisabledCollisionsGroupId() const;

      /**
      \return group ids added to this wire
      */
      agx::UInt32Vector getGroupIds() const;

      /**
      \return group names added to this wire
      */
      agx::NameVector getGroupNames() const;

      /**
      \return the sensor belonging to contact wire node \p cln
      */
      agxCollide::Geometry* getSensor( agxWire::ContactNode* cln );
      const agxCollide::Geometry* getSensor( const agxWire::ContactNode* cln ) const;

      /**
      \return contact wire node given contact wire node sensor \p sensor
      */
      agxWire::ContactNode* getContactWireNode( agxCollide::Geometry* sensor );

      /**
      \param geometry - a wire geometry defined between two nodes
      \return the pair of nodes that defines \p geometry - (0, 0) if the geometry doesn't belong to this wire
      */
      const NodePtrPair& getNodePair( const agxCollide::Geometry* geometry ) const;

      /**
      \return the contact controller
      */
      agxWire::WireContactController* getActiveContactController() const;

      /**
      \return the old contact controller
      */
      agxWire::WireOldContactController* getOldContactController() const;

      /**
      \return the shape contact controller
      */
      agxWire::WireShapeContactController* getShapeContactController() const;

      /**
      \return true if this geometry controller is enabled/valid - otherwise false
      */
      bool getEnable() const;

      /**
      \return true if this geometry controller is enabled/valid - otherwise false
      */
      bool isEnabled() const;


      /**
      \return the m_enable flag, use with consideration
      */
      bool getEnableFlag() const;

      /**
      Set enable flag, will not invoke any other actions.
      \param enable - true to enable
      */
      void setEnable( bool enable );

      /**
      \return true if enabled and initialized
      */
      bool isActive() const;

      /**
      \return the wire radius
      */
      agx::Real getRadius() const;

      /**
      \return the wire radius including the multiplier
      */
      agx::Real getRadiusIncMultiplier() const;

      /**
      \return the maximum distance a contact may move one time step
      */
      agx::Real getMaximumContactMovementOneTimestep() const;

      /**
      \return max of the maximum movement one time step and double wireRadius
      */
      agx::Real getSensorRadius( agx::Real wireRadius ) const;

      /**
      Gives a geometry contact given two geometries
      */
      agxCollide::LocalGeometryContact getGeometryContact( agxCollide::Geometry* geom1, agxCollide::Geometry* geom2 );

      /**
      Will verify if two geometries are colliding
      */
      bool areColliding( agxCollide::Geometry* geom1, agxCollide::Geometry* geom2 );

      /**
      Remove any geometry references kept internally to this lumped node body. This
      action has to be performed if one for example want to use other geometries
      than the default on lumped nodes (not recommended).
      \param lumpedNodeBody - body with geometry belonging to (created by) this geometry controller
      */
      void removeGeometryReference( agx::RigidBody* lumpedNodeBody );

      /**
      \return the impact controller
      */
      agxWire::WireImpactController* getImpactController() const;

      /**
      Assign simulation proxy for this controller to use.
      */
      void setSimulationProxy( agxSDK::SimulationProxy* simulationProxy );

      /**
      \param geometry - a geometry
      \return true if \p geometry is part of the wire this impact controller handles
      */
      bool match( agx::Physics::GeometryPtr geometry ) const;

      /**
      Serialize important objects.
      */
      void store( agxStream::OutputArchive& out ) const;

      /**
      Restore important objects.
      */
      void restore( agxStream::InputArchive& in );

    protected:
      friend class WireContactController;

      /**
      Referenced counted object - protected destructor.
      */
      virtual ~WireGeometryController();


      /**
      Creates a WireContactController of the chosen type of m_contactControllerType;
      */
      void createContactController(Wire* wire);

      /**
      Callback from constraint AFTER a new node has been added (during routing).
      \param nodeIt - iterator to node inserted
      */
      virtual void onAdd( agxWire::NodeIterator nodeIt ) override;

      /**
      Callback from constraint AFTER a new node has been inserted.
      \param nodeIt - iterator to node inserted
      */
      virtual void onInsert( agxWire::NodeIterator nodeIt ) override;

      /**
      Callback from constraint BEFORE a node is removed.
      \param nodeIt - iterator to node about to be removed
      */
      virtual void onRemove( agxWire::NodeIterator nodeIt ) override;

      /**
      Callback from constraint before solve (affects geometry state).
      */
      virtual void preSolve() override;

      /**
      Callback from constraint after solve (affects geometry state).
      */
      virtual void postSolve() override;

      /**
      Add a sphere to a body fixed node
      */
      agx::Bool insertSphere(agxWire::BodyFixedNode* node);

      /**
      Remove a sphere from a body fixed node
      */
      agx::Bool removeSphere(agxWire::BodyFixedNode* node);

      /**
      When a body fixed wire node is enabled/added/inserted.
      \param node - node
      */
      void onAdd( agxWire::BodyFixedNode* node );

      /**
      When a body fixed wire node is disabled/removed.
      \param node - node
      */
      void onRemove( agxWire::BodyFixedNode* node );

      /**
      When a contact node is added/inserted.
      */
      void onAdd( agxWire::ContactNode* cn );

      /**
      When a contact node is removed.
      */
      void onRemove( agxWire::ContactNode* cn );

      /**
      \return the simulation
      */
      agxSDK::SimulationProxy* getSimulationProxy() const;

      /**
      Allocates a pool of the current wire geometry type and geometry filter for that wire geometry.
      */
      void allocateWireGeometries( agxCollide::GeometryRefVector& geometryVector, size_t size ) const;

      /**
      Allocates a pool of sensor geometries.
      */
      void allocateSensorGeometries( agxCollide::GeometryRefVector& geometryVector, size_t size ) const;

      /**
      Takes sphere from pool if the pool isn't empty (if the pool is empty this method will resize it)
      \return a geometry with a sphere shape (default radius)
      */
      agxCollide::GeometryRef createDefaultSphere();

      /**
      Takes sensor from pool if the pool isn't empty (if the pool is empty this method will resize it)
      \return a sensor geometry with a sphere shape (default radius)
      */
      agxCollide::GeometryRef createSensorSphere();

      /**
      Will move one geometry from the disabled geometries vector to the active wire geometries vector.
      */
      void pushToEnabledWireGeometries();

      /**
      Enable/disable first and last lumped node geometries given the current wire configuration.
      \param wire - wire constraint
      */
      void handleFirstAndLastNodeSphereGeometry( agxWire::WireDistanceCompositeConstraint* wire ) const;

      template <class IdT, class ContainerT>
      void addGroupT( IdT id, ContainerT& container );

      template <class IdT, class ContainerT>
      void removeGroupT( IdT id, ContainerT& container );

    protected:
      agx::observer_ptr< WireDistanceCompositeConstraint >  m_wire;
      WireImpactControllerRef                               m_impactController;
      agx::ref_ptr<agx::Referenced>                         m_shapeContactController;
      agx::ref_ptr<agx::Referenced>                         m_oldContactController;
      agxCollide::GeometryRefVector                         m_enabledWireGeometries;
      agxCollide::GeometryRefVector                         m_disabledWireGeometries;
      GeometryPtrHashSet                                    m_enabledGeometries;
      RigidBodyPtrGeometryRefTable                          m_lumpedSphereGeometries;
      ContactNodePtrGeometryRefTable                        m_nodeSensors;
      GeometryPtrContactNodePtrTable                        m_sensorNodes;
      agxCollide::GeometryRefVector                         m_disabledSensorGeometries;
      agx::UInt32                                           m_disabledCollisionsGroupIDOtherGeometries; /**< Unique group id to add to other geometries that are disabled against this line. */
      agx::UInt32                                           m_disabledCollisionsGroupIDWire;            /**< Unique group id that all line components (cylinders/capsules, spheres, sensors etc) has. */
      agx::UInt32                                           m_sensorUniqueGroupID;                      /**< Group ID for sensor geometries. */
      agx::UInt32                                           m_wireUniqueGroupID;                        /**< Unique ID for this wire. */
      GeometryPtrGeometryObsTable                           m_externalDisabledGeometries;
      UnsignedHashSet                                       m_groupIDs;
      NameHashSet                                           m_groupNames;
      GeometryPtrNodeObsPtrPairTable                        m_geometryNodePairTable;
      agx::observer_ptr< WireMaterialController >           m_materialController;
      WireParameterController*                              m_parameters;
      bool                                                  m_enable;
      bool                                                  m_initialized;
      agxCollide::GeometryRefVector                         m_wireGeometryPool;
      agxSDK::SimulationProxyRef                            m_simulationProxy;
      WireContactControllerType                             m_contactControllerType;
      agx::ref_ptr<agx::Referenced>                         m_colliderUtils;

      private:
        static NodePtrPair s_nullObsPtrPair;

  };

  typedef agx::ref_ptr< WireGeometryController > WireGeometryControllerRef;

  // Inline methods *****************************************************************************

  inline bool WireGeometryController::isActive() const
  {
    return m_enable && m_initialized;
  }

  inline bool WireGeometryController::hasGeometry( const agxCollide::Geometry* geometry ) const
  {
    agxAssert( m_wireUniqueGroupID > 0 && (m_wire == nullptr || m_geometryNodePairTable.contains( geometry ) == geometry->hasGroup( m_wireUniqueGroupID )) );
    return m_wire != nullptr && geometry->hasGroup( m_wireUniqueGroupID );
  }

  inline void WireGeometryController::pushToEnabledWireGeometries()
  {
    if ( m_disabledWireGeometries.empty() )
      allocateWireGeometries( m_disabledWireGeometries, 32 );
    m_enabledWireGeometries.push_back( m_disabledWireGeometries.back() );
    m_disabledWireGeometries.pop_back();
  }

  inline const NodePtrPair& WireGeometryController::getNodePair( const agxCollide::Geometry* geometry ) const
  {
    GeometryPtrNodeObsPtrPairTable::const_iterator i = m_geometryNodePairTable.find( geometry );
    return i != m_geometryNodePairTable.end() ? i->second : s_nullObsPtrPair;
  }

  inline agxSDK::SimulationProxy* WireGeometryController::getSimulationProxy() const
  {
    return m_simulationProxy;
  }

  inline agxCollide::Geometry* WireGeometryController::getSensor( ContactNode* cln )
  {
    ContactNodePtrGeometryRefTable::iterator i = m_nodeSensors.find(cln);
    if ( i != m_nodeSensors.end() )
      return i->second;
    return nullptr;
  }

  inline ContactNode* WireGeometryController::getContactWireNode( agxCollide::Geometry* sensor )
  {
    GeometryPtrContactNodePtrTable::iterator i = m_sensorNodes.find( sensor );
    if ( i != m_sensorNodes.end() )
      return i->second;
    return nullptr;
  }

  inline bool WireGeometryController::getEnable() const
  {
    return m_enable && m_wire.isValid() && m_materialController.isValid() && m_parameters;
  }

  inline bool WireGeometryController::isEnabled() const
  {
    return getEnable();
  }

  inline void WireGeometryController::setEnable( bool enable )
  {
    m_enable = enable;
  }

  inline bool WireGeometryController::getEnableFlag() const
  {
    return m_enable;
  }

  inline agx::Real WireGeometryController::getRadius() const
  {
    agxAssert( m_materialController );
    return m_materialController->getRadius();
  }

  inline agx::Real WireGeometryController::getRadiusIncMultiplier() const
  {
    agxAssert( m_materialController && m_parameters );
    return m_materialController->getRadius() * m_parameters->getRadiusMultiplier(m_materialController->getRadius());
  }

  inline agx::Real WireGeometryController::getMaximumContactMovementOneTimestep() const
  {
    agxAssert( m_materialController && m_parameters );
    return m_parameters->getMaximumContactMovementOneTimestep();
  }

  inline agx::Real WireGeometryController::getSensorRadius( agx::Real wireRadius ) const
  {
    agxAssert( m_materialController && m_parameters );
    return m_parameters->getSensorRadius(wireRadius );
  }

  inline WireImpactController* WireGeometryController::getImpactController() const
  {
    return m_impactController;
  }

  AGX_FORCE_INLINE bool WireGeometryController::match( agx::Physics::GeometryPtr geometry ) const
  {
    agx::Physics::Geometry::ShapePtr shape = geometry.shape();
    return shape && shape.type() == agxCollide::Shape::WIRE_SHAPE && ((agx::Physics::Geometry::WireShapePtr)shape).wireId() == m_wireUniqueGroupID;
  }

  // ********************************************************************************************
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

DOXYGEN_END_INTERNAL_BLOCK()
