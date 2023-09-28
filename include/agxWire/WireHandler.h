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
#include <agxWire/WireParallelCallbacksHandler.h>

#include <agxUtil/agxUtil.h>

#include <agx/agxPhysics_export.h>

#include <agx/LinearProbingHashTable.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxWire
{
  class WireDistanceCompositeConstraint;
  class WireGeometryController;
  class WireWireContactConstraint;
  typedef WireWireContactConstraint* WireWireContactConstraintPtr;
  class WireContactConstraint;
  typedef WireContactConstraint* WireContactConstraintPtr;
  class WireSegmentContactConstraint;
  typedef WireSegmentContactConstraint* WireSegmentContactConstraintPtr;
  class WireContactSegment;

  struct ContactMode
  {
    enum Enum {
      UNKNOWN              = 0,      /**< Default contact state is - "it shouldn't matter". */
      FACE_CONTACT         = (1<<0), /**< The contact point is on a face, i.e., not on an edge. */
      EDGE_SAFE            = (1<<1), /**< Convex, well defined object, e.g., a box. */
      SPHERE_CONTACT       = (1<<2), /**< Contact point on capsule sphere. */
      SPHERE_CONTACT_BEGIN = (1<<3), /**< Contact point on capsule sphere (negative side). */
      SPHERE_CONTACT_END   = (1<<4), /**< Contact point on capsule sphere (positive side). */
      OLD_CONTACT_MODEL    = (1<<5)  /**< Old contact model should be used for contact other than SPHERE_CONTACT. */
    };

    static agx::UInt find( const agxCollide::ContactPoint& point, const agx::AffineMatrix4x4& invWireShapeTransform, const WireContactSegment* segment, bool useKinematicContactNodes );
  };

  /**
  Class to handle wire wire contacts. Valid to have ONE per simulation instance.
  */
  class AGXPHYSICS_EXPORT WireHandler : public agxSDK::StepEventListener
  {
    public:
      typedef agx::VectorPOD< WireWireContactConstraintPtr >                              WireWireContactConstraintPtrContainer;
      typedef agx::VectorPOD< WireContactConstraintPtr >                                  WireContactConstraintPtrContainer;
      typedef agx::VectorPOD< WireSegmentContactConstraintPtr >                           WireSegmentContactConstraintPtrContainer;
      typedef agx::List< WireContactConstraintPtr >                                       WireContactConstraintPtrList;
      typedef agx::List< WireSegmentContactConstraintPtr >                                WireSegmentContactConstraintPtrList;

      typedef agx::SymmetricPair< agxWire::Node* >                                        SymmetricNodePtrPair;
      typedef agx::SymmetricPair< SymmetricNodePtrPair >                                  WireWireKey;

#include <HashImplementationSwitcher.h>
#if HASH_FOR_WIRE == HASH_NEW
      using NodePtrUIntTable = agx::LinearProbingHashTable<const agxWire::Node*, agx::UInt, agx::HashFn<const agxWire::Node*>>;
      using WireWireKeyWireWireContactConstraintPtrTable = agx::LinearProbingHashTable<WireWireKey, WireWireContactConstraintPtr, agx::HashFn<WireWireKey>>;
      using NodePtrWireContactConstraintPtrTable = agx::LinearProbingHashTable<Node*, WireContactConstraintPtrList, agx::HashFn<Node*>>;
      using NodePtrPairWireSegmentContactConstraintPtrTable = agx::LinearProbingHashTable<SymmetricNodePtrPair, WireSegmentContactConstraintPtrList, agx::HashFn<SymmetricNodePtrPair>>;
#elif HASH_FOR_WIRE == HASH_OLD
      typedef agx::QuadraticProbingHashTable< const agxWire::Node*, agx::UInt >                           NodePtrUIntTable;
      typedef agx::QuadraticProbingHashTable< WireWireKey, WireWireContactConstraintPtr >                 WireWireKeyWireWireContactConstraintPtrTable;
      typedef agx::QuadraticProbingHashTable< Node*, WireContactConstraintPtrList >                       NodePtrWireContactConstraintPtrTable;
      typedef agx::QuadraticProbingHashTable< SymmetricNodePtrPair, WireSegmentContactConstraintPtrList > NodePtrPairWireSegmentContactConstraintPtrTable;
#endif

    public:
      /**
      Default constructor.
      */
      WireHandler();

      /**
      Creates contact nodes given two wires and one geometry contact.
      \param wire1 - first wire
      \param wire1GeometryController - geometry controller of wire1
      \param wire1Geometry - collision geometry on wire1
      \param wire2 - second wire
      \param wire2GeometryController - geometry controller of wire2
      \param wire2Geometry - collision geometry on wire2
      \param geometryContact - the geometry contact
      \return true if the contact should be removed
      */
      bool handle( agxWire::WireDistanceCompositeConstraint* wire1, agxWire::WireGeometryController* wire1GeometryController, const agxCollide::Geometry* wire1Geometry, agxWire::WireDistanceCompositeConstraint* wire2, agxWire::WireGeometryController* wire2GeometryController, const agxCollide::Geometry* wire2Geometry, agxCollide::GeometryContact* geometryContact );

      /**
      Handles general geometry contact.
      */
      bool handle( agxWire::WireDistanceCompositeConstraint* wire, agxWire::WireGeometryController* geometryController, agxCollide::Geometry* otherGeometry, agxCollide::GeometryContact* geometryContact, bool onlySphereContact );

      /**
      Callback when \p wire is being removed from simulation.
      All constraints associated to \p wire will be removed.
      \param wire - wire about to be removed
      */
      void onRemoveNotification( agxWire::WireDistanceCompositeConstraint* wire );

      /**
      \return the number of constraints \p node belongs to
      */
      agx::UInt getNodeCount( const agxWire::Node* node ) const;

      /**
      Internal method.
      */
      void executeCallback( agxWire::WireParallelCallbacksHandler::CallbackTypes type );

    protected:
      virtual ~WireHandler();

      /**
      Clears all internal structures and removes them from the simulation (if != 0).
      */
      void clear( agxSDK::Simulation* simulation );

      friend class WireContactSegment;
      /**
      Removes node from the wire and returns the rigid body and the node to each pool.
      Note that the rigid body will not be removed if its counter > 0 and the node wont
      be removed from the wire if the wire is zero.
      \sa release
      */
      void remove( agxWire::Node* node, agxWire::WireDistanceCompositeConstraint* wire );

      /**
      Removes contact constraints related to a node
      Only valid for wire-geometry contacts (nodes with wire-wire contacts wont be affected)
      \returns true if the node should be removed
      */
      bool removeContactConstraints( agxWire::Node* node );


      /**
      Release a contact node. Instead of removing it the node is transformed to a
      node that is still on the wire.
      \sa remove
      */
      void release( agxWire::Node* node );

      /**
      Internal method. Updates the counter internal nodes have.
      \param node - node to increase or decrease counter for
      \param increment - number of steps, decrements if negative
      \return the current count
      */
      agx::UInt updateCounter( const agxWire::Node* node, int increment );

      /**
      \return constraint between \p node1 and \p node2 if present - otherwise 0
      */
      agxWire::WireWireContactConstraintPtr getConstraint( agxWire::WireHandler::WireWireKey key ) const;

      /**
      \return an already existing wire contact constraint (i.e., if present) - otherwise 0
      */
      agxWire::WireContactConstraintPtr getConstraint( agxWire::Node* node, const agxCollide::Geometry* geometry ) const;

      /**
      \return an already existing wire segment contact constraint (i.e., if present) - otherwise 0
      */
      agxWire::WireSegmentContactConstraintPtr getConstraint( const agxWire::WireContactSegment& segment, const agxCollide::Geometry* geometry ) const;

      /**
      \return all wire contact constraints for this wire node
      */
      WireContactConstraintPtrList* getConstraints( agxWire::Node* node ) const;

      /**
      Add wire-wire interaction.
      \param constraint - wire-wire constraint to add
      */
      void add( agxWire::WireWireContactConstraintPtr constraint );

      /**
      Add wire interaction.
      \param constraint wire constraint to add
      */
      void add( agxWire::WireContactConstraintPtr constraint );

      /**
      Add wire interaction.
      \param constraint wire constraint to add
      */
      void add( agxWire::WireSegmentContactConstraintPtr constraint );

      /**
      Removes constraint and returns it to pool if present.
      */
      void remove( agxWire::WireWireContactConstraintPtr constraint );

      /**
      Removes constraint and returns it to pool if present.
      */
      void remove( agxWire::WireContactConstraintPtr constraint );

      /**
      Removes constraint and returns it to pool if present.
      */
      void remove( agxWire::WireSegmentContactConstraintPtr constraint );

      /**
      Given two contact segments, constraints are created. Assumes
      both contact segments are valid and the contact nodes as well.
      */
      void collide( agxWire::WireContactSegment& contactSegment1, agxWire::WireContactSegment& contactSegment2, agx::Real32 normalScale );

      /**
      Collide contact segment and another geometry (NOT wire-wire).
      */
      bool collide( agxWire::WireContactSegment& contactSegment, const agxCollide::Geometry* wireGeometry, agxCollide::Geometry* otherGeometry, agx::UInt contactModeMask, agx::Real32 normalScale );

      void createOrUpdateConstraint( agxWire::WireContactSegment& contactSegment1, agxWire::WireContactSegment& contactSegment2, agx::Real32 normalScale );
      void createOrUpdateConstraint( agxWire::Node* node, agxWire::WireContactSegment& contactSegment, agxCollide::Geometry* otherGeometry, agx::Real32 normalScale, bool changeNodeType = true );
      void createOrUpdateConstraint( agxWire::WireContactSegment& contactSegment, agxCollide::Geometry* otherGeometry, agx::Real32 normalScale, bool changeNodeType = true );

      virtual void addNotification() override;
      virtual void removeNotification() override;

      virtual void preCollide( const agx::TimeStamp& ) override;
      virtual void pre( const agx::TimeStamp& ) override;
      virtual void post( const agx::TimeStamp& ) override;

      typedef int (*WireContactUpdateCallback)( agxWire::WireContactConstraintPtr, agxSDK::Simulation*, agxCollide::LocalGeometryContactVector& );
      typedef int (*WireSegmentContactUpdateCallback)( agxWire::WireSegmentContactConstraintPtr, agxSDK::Simulation*, agxCollide::LocalGeometryContactVector& );
      typedef int (*WireWireContactUpdateCallback)( agxWire::WireWireContactConstraintPtr, agxSDK::Simulation*, agxCollide::LocalGeometryContactVector& );
      void handleCallback( WireContactUpdateCallback wireContactUpdater, WireSegmentContactUpdateCallback wireSegmentContactUpdater, WireWireContactUpdateCallback wireWireContactUpdater );

      agxWire::WireWireContactConstraintPtr    getNewWireWireContactConstraint();
      void                                     returnConstraint( agxWire::WireWireContactConstraintPtr constraint );

      agxWire::WireContactConstraintPtr        getNewWireContactConstraint();
      void                                     returnConstraint( agxWire::WireContactConstraintPtr constraint );

      agxWire::WireSegmentContactConstraintPtr getNewWireSegmentContactConstraint();
      void                                     returnConstraint( agxWire::WireSegmentContactConstraintPtr constraint );

      void expand( WireWireContactConstraintPtrContainer& constraints, agx::UInt numElements ) const;
      void expand( WireContactConstraintPtrContainer& constraints, agx::UInt numElements ) const;
      void expand( WireSegmentContactConstraintPtrContainer& constraints, agx::UInt numElements ) const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::WireHandler );

    protected:
      WireWireContactConstraintPtrContainer           m_wireWireContactConstraintPool;
      WireContactConstraintPtrContainer               m_wireContactConstraintPool;
      WireSegmentContactConstraintPtrContainer        m_wireSegmentContactConstraintPool;
      WireWireKeyWireWireContactConstraintPtrTable    m_nodesToWireWireContactConstraint;
      NodePtrWireContactConstraintPtrTable            m_nodeToWireContactConstraint;
      NodePtrPairWireSegmentContactConstraintPtrTable m_nodesToWireSegmentContactConstraint;
      agxUtil::ConstraintHolderRef                    m_wireWireContactConstraints;
      agxUtil::ConstraintHolderRef                    m_wireContactConstraints;
      agxUtil::ConstraintHolderRef                    m_wireSegmentContactConstraints;
      NodePtrUIntTable                                m_nodeCounter;
      agx::PropertyContainerRef                       m_tempPropertyContainer;
      agx::Mutex                                      m_mutex;
  };

  typedef agx::ref_ptr< WireHandler > WireHandlerRef;
}

DOXYGEN_END_INTERNAL_BLOCK()
