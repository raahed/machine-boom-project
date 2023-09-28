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


#include <agx/BitState.h>
#include <agx/AddedMassInteraction.h>
#include <agx/ConstraintImplementation.h>
#include <agx/InternalData.h>

#include <agxCollide/LocalContactPoint.h>

#include <agx/Physics/ParticlePairContactEntity.h>
#include <agx/Physics/ParticleGeometryContactEntity.h>

#define MERGED_BODY_SERIALIZATION_VERSION_1 49
#define MERGED_BODY_SERIALIZATION_VERSION_2 75

namespace agxSDK
{
  class Simulation;
  class MergeSplitHandler;
}

namespace agxWire
{
  class BodyFixedNode;
}

namespace agx
{
  AGX_DECLARE_POINTER_TYPES( MergedBody );
  AGX_DECLARE_VECTOR_TYPES( MergedBody );
  typedef agx::SetVector<agx::MergedBodyRef> MergedBodyRefSetVector;

  /**
  Structure holding several "normal" rigid bodies. The "normal" rigid bodies will
  not be seen by the solver and they will not move relative each other. A rigid body
  added to this structure can still interact with other bodies in a simulation.
  */
  class AGXPHYSICS_EXPORT MergedBody : public agx::Referenced, public agxStream::Serializable
  {
    public:
      // Interactions.
      AGX_DECLARE_POINTER_TYPES( EdgeInteraction );
      AGX_DECLARE_POINTER_TYPES( EmptyEdgeInteraction );
      AGX_DECLARE_POINTER_TYPES( ContactGeneratorEdgeInteraction );
      AGX_DECLARE_POINTER_TYPES( GeometryContactEdgeInteraction );
      AGX_DECLARE_POINTER_TYPES( BinaryConstraintEdgeInteraction );
      AGX_DECLARE_POINTER_TYPES( ParticleParticleInteraction );

      /**
      Internal state of this merged body.
      */
      enum State
      {
        INTERACTION_GRAPH_DIRTY = 1 << 0,
        RESTORED                = 1 << 1,
        READY_FOR_SOLVER        = 1 << 2,
        AUTO_CLEAN_WHEN_EMPTY   = 1 << 3
      };

      /**
      Object with index of merged body and the original index of
      a merged object.
      */
      struct MergedBodyIndex
      {
        /**
        Default constructor initializes both indices to Invalid Index.
        */
        inline MergedBodyIndex() : index( agx::InvalidIndex ), originalIndex( agx::InvalidIndex ) {}

        /**
        Construct given object associated to index, and an original object
        associated to originalIndex.
        */
        inline MergedBodyIndex( agx::Physics::RigidBodyPtr rb, agx::Physics::RigidBodyPtr orgRb )
          : index( rb ? rb.calculateIndex() : agx::InvalidIndex ), originalIndex( orgRb ? orgRb.calculateIndex() : agx::InvalidIndex ) {}

        /**
        \return true if index != originalIndex, i.e., object considered merged
        */
        inline agx::Bool isMerged() const { return originalIndex != agx::InvalidIndex && index != originalIndex; }

        UInt index;
        UInt originalIndex;
      };

    public:
      // Listeners and visitors.
      AGX_DECLARE_POINTER_TYPES( Listener );
      typedef std::function< void( agx::RigidBody* ) > RigidBodyVisitor;
      typedef std::function< void( EdgeInteraction* ) > EdgeInteractionVisitor;
      typedef agx::BitState< State > InternalState;
      typedef agx::Vector< EdgeInteractionRef > EdgeInteractionRefContainer;

    public:
      /**
      This method returns an agx::MergedBody if \p rb has been added to one.
      This method returns an agx::MergedBody regardless of if the agx::MergedBody
      is in a simulation or not.
      \param rb - rb to find the agx::MergedBody for
      \return merged body that rb is part of, null if rb isn't merged
      */
      static const agx::MergedBody* get( const agx::RigidBody* rb );

      /**
      This method returns an agx::MergedBody if \p rb has been added to one.
      This method returns an agx::MergedBody regardless of if the agx::MergedBody
      is in a simulation or not.
      \param rb - rb to find the agx::MergedBody for
      \return merged body that rb is part of, null if rb isn't merged
      */
      static agx::MergedBody* get( agx::RigidBody* rb );

      /**
      This method returns an agx::MergedBody that is active and in a simulation.
      \p rb - rb to find the active agx::MergedBody for
      \return merged body that rb is part of, null if rb isn't merged or the merged body isn't active
      */
      static const agx::MergedBody* getActive( const agx::RigidBody* rb );

      /**
      This method returns an agx::MergedBody that is active and in a simulation.
      \p rb - rb to find the active agx::MergedBody for
      \return merged body that rb is part of, null if rb isn't merged or the merged body isn't active
      */
      static agx::MergedBody* getActive( agx::RigidBody* rb );

      /**
      \param rb - rigid body instance
      \return true if \p rb is root body of a MergedBody instance, otherwise false
      */
      static agx::Bool isRoot( const agx::RigidBody* rb );

      /**
      Similar to filter( const agx::RigidBody* rb ).
      \return index of either \p rb or a merged body that \p rb is part of
      */
      static MergedBodyIndex findMergedBodyIndex( agx::Physics::RigidBodyPtr rb );

      /**
      Filters \p rb and returns the merged rigid body if \p rb is part
      of one. Otherwise \p rb is returned.
      \param rb - rb to filter
      \return a merged rigid body if \p rb is present in one, otherwise \p rb
      */
      static const agx::RigidBody* filter( const agx::RigidBody* rb );
      static agx::RigidBody* filter( agx::RigidBody* rb );

      /**
      Split rigid body if merged.
      \return true if the rigid body was merged and successfully removed from the merged body, otherwise false
      */
      static agx::Bool split( agx::RigidBody* rb );

      /**
      This method returns a merged body if the constraint has been involved in it, e.g.,
      by having an edge created given it.
      \note This method ignores the bodies involved.
      \param constraint - constraint to check if involved in a merged body
      \return the merged body the constraint is part of, otherwise nullptr
      */
      static agx::MergedBody* get( agx::Constraint* constraint );

      /**
      This method returns a merged body if the constraint has been involved in it, e.g.,
      by having an edge created given it.
      \note This method ignores the bodies involved.
      \param constraint - constraint to check if involved in a merged body
      \return the merged body the constraint is part of, otherwise nullptr
      */
      static const agx::MergedBody* get( const agx::Constraint* constraint );

      /**
      \return the edge interaction associated to the constraint, otherwise nullptr
      */
      static BinaryConstraintEdgeInteraction* getEdge( const agx::Constraint* constraint );

      /**
      \return the edge interaction associated to the constraint, otherwise nullptr
      */
      static BinaryConstraintEdgeInteraction* getEdge( const agx::HighLevelConstraintImplementation* constraint );

      /**
      Removes the edge defined by the constraint from the merged body, if any.
      \return true if were merged and the edge was removed, otherwise false
      */
      static agx::Bool split( agx::Constraint* constraint );

      /**
      Removes the edge defined by the constraint from the merged body, if any.
      \return true if were merged and the edge was removed, otherwise false
      */
      static agx::Bool split( agx::HighLevelConstraintImplementation* constraint );

      /**
      \internal

      True if MergedBody::getActive == nullptr or when rb has been merged by
      MergeSplitHandler in post-solve this step.
      */
      static agx::Bool shouldIntegratePosition( const agx::RigidBody* rb );

    public:
      /**
      Construct merged body, default containing a root rigid body.
      */
      MergedBody();

      /**
      Creates an empty clone of this merged body where each object that supports cloning
      also are cloned.
      \note The returned clone will not contain any bodies
      \returns an empty clone of this merged body
      */
      virtual MergedBodyRef clone();

      /**
      \return the name of this MergedBody
      */
      const agx::Name& getName() const;

      /**
      Assign new name to this MergedBOdy
      Default: ""
      */
      void setName(const agx::Name& name);

      /**
      \return the root rigid body of this merged body
      */
      agx::RigidBody* getRigidBody() const;

      /**
      Add an edge interaction defining a base interaction between one
      or two rigid bodies that will be considered merged if successful.
      \param edge - edge interaction
      \return true if merge was successful, otherwise false
      */
      agx::Bool add( EdgeInteraction* edge );

      /**
      Merges this with \p other and clears \p other.
      \param other - other root to merge with this
      \param edge - new edge between this and \p other
      \return true if successful, otherwise false
      */
      agx::Bool merge( agx::MergedBody* other, EdgeInteraction* edge );

      /**
      Remove an edge interaction from this merged body.
      \param edge - edge interaction to remove
      \return true if removed
      */
      agx::Bool remove( EdgeInteraction* edge );

      /**
      Split/Remove a rigid body from this merged body.
      \note If \p rb isn't part of this merged body this action will fail.
      \param rb - rigid body to split
      \return true if successful, otherwise false
      */
      agx::Bool remove( agx::RigidBody* rb );

      /**
      Add listener to receive callbacks during certain events.
      \param listener - listener to add
      \return true if the listener was added (or is already present), otherwise false
      */
      agx::Bool add( Listener* listener );

      /**
      Remove listener from this merged body.
      \param listener - listener to remove
      \return true if removed, otherwise false
      */
      agx::Bool remove( Listener* listener );

      /**
      Splits into islands.
      \return the other islands created and added to the simulation
      \note This merged body has to be in a simulation for this action to be performed.
      */
      agx::MergedBodyRefVector splitIslands();

      /**
      \return true if \p rb1 is merged with \p rb2, i.e., there's an edge interaction present between them
      */
      agx::Bool merged( const agx::RigidBody* rb1, const agx::RigidBody* rb2 ) const;

      /**
      \return the number of pair interactions in this merged body
      */
      agx::UInt getNumPairs() const;

      /**
      \return true if this merged body doesn't contain any bodies
      */
      agx::Bool isEmpty() const;

      /**
      \return true if this merged body is in a simulation and assumed to be active
      */
      agx::Bool isActive() const;

      /**
      \return the edges defined between rb1 and rb2, null if none (e.g., rb1 and rb2 isn't merged in this merged body, or aren't neighbors)
      */
      const EdgeInteractionRefContainer* getEdges( const agx::RigidBody* rb1, const agx::RigidBody* rb2 ) const;

      /**
      \return bodies \p rb is merged with, null if not merged
      */
      const agx::RigidBodyPtrVector* getNeighbors( const agx::RigidBody* rb ) const;

      /**
      If true, this merged body will remove itself from the simulation
      and clear all internal states when it becomes empty.

      This state is disabled by default and can be used when handling
      merged bodies implicitly by doing agx::MergedBody::get/getActive( rb ).
      \note This state is unnecessary as long as some other than agxSDK::Simulation
            is holding a reference to this object.
      \param enable - true to enable, false to disable
      */
      void setEnableAutomaticCleanupWhenEmpty( agx::Bool enable );

      /**
      \return true if automatic cleanup when empty is enabled - otherwise false
      */
      agx::Bool getEnableAutomaticCleanupWhenEmpty() const;

#ifndef SWIG
      /**
      Traverse the internal interaction graph and receive callback for all rigid bodies (unique).
      \param visitor - visitor function
      \param includeDisabled - true to receive callback for all bodies, false to only receive
                               callback for enabled and simulated bodies
      */
      void traverse( RigidBodyVisitor visitor, agx::Bool includeDisabled = false ) const;

      /**
      Traverse the internal interaction graph and receive callback for each edge interaction.
      \param visitor - visitor function
      \param includeDisabled - true to receive callback for all edge interactions, including ones
                               with disabled and/or not simulating bodies. False to only receive
                               callback for edge interactions containing enabled and simulated bodies.
      */
      void traverse( EdgeInteractionVisitor visitor, agx::Bool includeDisabled = false ) const;

      /**
      Traverse the internal interaction graph and receive callback for each edge interaction
      where \p rb is part of.
      \param rb - rigid body to check whether part of an edge interaction
      \param visitor - visitor function
      \param includeDisabled - true to receive callback for all edge interactions, including ones
                               with disabled and/or not simulating bodies. False to only receive
                               callback for edge interactions containing enabled and simulated bodies.
      */
      void traverse( const agx::RigidBody* rb, EdgeInteractionVisitor visitor, agx::Bool includeDisabled = false ) const;

      /**
      Traverse neighboring bodies to \p rb. Neighboring bodies are the bodies with an
      edge interaction to \p rb.
      \param rb - rb with neighbors
      \param visitor - visitor function
      \param includeDisabled - true to receive callback for all bodies, false to only receive
                               callback for enabled and simulated bodies
      */
      void traverse( const agx::RigidBody* rb, RigidBodyVisitor visitor, agx::Bool includeDisabled = false ) const;
#endif

      /**
      \return current internal state
      */
      const InternalState& getState() const;

      /**
      Debug prints (cout) interactions to \p rb.
      */
      void debugPrint( const agx::RigidBody* rb ) const;

      /**
      Color of this merged body used for debug rendering.
      */
      void getDebugRenderColor( agx::Vec3& color ) const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      AGXSTREAM_DECLARE_SERIALIZABLE( agx::MergedBody );

      /**
      Called after solve. Update velocity of the merged bodies.
      */
      virtual void updateVelocities();

      /**
      Called after solve (after updateVelocities) and stores the relative transform
      between the root body and the merged bodies.
      */
      virtual void storeRelativeTransforms();

      /**
      Called after transforms have been integrated.
      */
      void postIntegrate();

      /**
      Clears the list generated interactions from removed edge interactions. Use
      this method when the generated interactions can be considered redundant -
      e.g., the edges were removed before collision detection.
      */
      void clearGeneratedInteractions();
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~MergedBody() throw();

      friend class agxSDK::Simulation; // Remove of bodies and constraints.
      friend class agxCollide::Space;  // Remove of geometries from space.
      friend class agx::RigidBody;     // Remove of geometries from bodies.
      /**
      Called during agxSDK::Simulation::add and agxSDK::Simulation::remove.
      If removed, simulation == null.
      */
      virtual void setSimulation( agxSDK::Simulation* simulation );

      /**
      Called when a rigid body, present in this merged body, is removed from the simulation.
      */
      virtual void onRemove( agx::RigidBody* rb );

      /**
      Called when a constraint, present in this merged body, is removed from the simulation.
      */
      virtual void onRemove( agx::Constraint* constraint );

      /**
      Called when a geometry, probably present in this merged body, is removed from space or body.
      */
      virtual void onRemove( agxCollide::Geometry* geometry );

      /**
      Called before collision detection.
      */
      virtual void preCollide();

      /**
      Called before solver executes.
      */
      virtual void preStep();

      /**
      Called after solver (new velocities and POSSIBLY (or not) new transforms).
      \sa postIntegrate
      */
      virtual void postStep();

      /**
      Called when the simulation is cleaned.
      */
      virtual void onCleanup();

    protected:
      /**
      \return the simulation this merged body is part of
      */
      agxSDK::Simulation* getSimulation();

      /**
      \return the simulation this merged body is part of
      */
      const agxSDK::Simulation* getSimulation() const;

      /**
      Updates mass, position, velocity etc of the root body given the current state.
      \return true if all is valid and the root can be added to the solver
      */
      agx::Bool updateProperties();

    private:
      struct PairData : public agx::Referenced
      {
        EdgeInteractionRefContainer edges;

        protected:
          virtual ~PairData() {}
      };
      typedef agx::ref_ptr<PairData> PairDataRef;

      typedef agx::SymmetricPair<const RigidBody*> Key;
      typedef agx::Vector<EdgeInteractionConstRef> EdgeInteractionConstRefContainer;
      typedef agx::HashVector<Key, PairDataRef> PairInteractionsContainer;
      typedef PairInteractionsContainer::iterator PairInteractionsIterator;
      typedef PairInteractionsContainer::const_iterator PairInteractionsConstIterator;
      typedef agx::Vector<RigidBodyRef> RigidBodyContainer;
      typedef agx::Vector<ListenerRef> ListenerContainer;

    private:
      friend class agxSDK::MergeSplitHandler;

      /**
      Additional data for the MergedBody::m_rb.
      */
      struct RootBodyData : public agx::Referenced
      {
        RootBodyData( agx::MergedBody* mb, agx::StrongInteraction* si ) : rootBody( mb ), strongInteraction( si ) {}

        MergedBody*               rootBody;          /**< Only assigned for the root body going to the solver. */
        agx::StrongInteractionRef strongInteraction; /**< Strong interaction used to handle damping. */
      };
      typedef agx::ref_ptr<RootBodyData> RootBodyDataRef;

      /**
      Data stored in the rigid body.
      */
      struct AGXPHYSICS_EXPORT MergedData : public agx::Referenced
      {
        enum Flag : agx::UInt16
        {
          INTEGRATE_POSITIONS = 1 << 0 /**< If automatically merged from postSolve, this body has to have
                                            IntegratePositions executed once before MergedBody::preStep. This
                                            flag is set by MergeSplitHandler and removed in MergedBody::updateProperties. */
        };
        using Flags = agx::BitState<Flag, agx::UInt16>;

        /**
        Construct given rigid body.
        */
        MergedData()
          : otherBodies(), relativeTransform(), mergedBody( nullptr ), id( agx::InvalidIndex ),
            mergedWithWorldCounter( 0 ), rootBodyData( nullptr ), flags( agx::UInt16( 0 ) )
        {
        }

        agx::RigidBodyPtrVector otherBodies;            /**< Other, neighboring bodies this body is merged with. */
        agx::AffineMatrix4x4    relativeTransform;      /**< Relative transform to our parent body. */
        MergedBody*             mergedBody;             /**< If merged, this is the parent body. */
        agx::UInt               id;                     /**< Id in the merged body. */
        agx::Int32              mergedWithWorldCounter; /**< Ref-counter how many edges the rigid body has to world. */
        RootBodyDataRef         rootBodyData;           /**< MergedBody::m_rb has this data. */
        Flags                   flags;

        protected:
          virtual ~MergedData() {}
      };
      typedef agx::ref_ptr<MergedData> MergedDataRef;

      /**
      Data stored in a constraint.
      */
      struct AGXPHYSICS_EXPORT ConstraintMergedData : public agx::Referenced
      {
        ConstraintMergedData()
          : mergedBody( nullptr ), edge( nullptr ) {}

        MergedBody*      mergedBody; /**< If merged, this is the body the constraint belongs to. */
        EdgeInteraction* edge;       /**< Edge in the merged body. */

        protected:
          virtual ~ConstraintMergedData() {}
      };
      typedef agx::ref_ptr<ConstraintMergedData> ConstraintMergedDataRef;

      /**
      Called by MergeSplitHandler to write the edges of this MergedBody to a
      .dot connectivity graph.
      */
      void writeConnectivityGraph( std::ostream& stream ) const;

    private:
      /**
      \return already created merged data for a given object, nullptr if no data has been created
      */
      template<typename DataT, typename ObjT>
      static DataT* getMergedData( const ObjT& obj );

      /**
      \return already create or creates new merged data for a given object. Invalid if obj == nullptr.
      */
      template<typename DataT, typename ObjT>
      static DataT* getOrCreateMergedData( ObjT obj );

      /**
      \return data for the root solver body
      */
      static RootBodyData* getRootBodyData( const agx::RigidBody* rb );

      /**
      \return true if merged
      */
      static agx::Bool isMerged( const MergedData* data );

    private:
      /**
      \return key given edge interaction
      */
      Key createKey( EdgeInteraction* edge ) const;

      /**
      \return key given two bodies
      */
      Key createKey( const agx::RigidBody* rb1, const agx::RigidBody* rb2 ) const;

      /**
      Internal method that creates merged data and adds a body to the global body array.
      Fires onAdd event if desired.
      \param rb - body to add
      \param fireOnAddEvent - true to fire onAdd event for the body (ONLY FALSE IF YOU EXPLICITLY SEND THE EVENT LATER)
      \return merged data for the body
      */
      MergedData* add( agx::RigidBody* rb, agx::Bool fireOnAddEvent = true );

      /**
      Internal method that erases the body from the global body array and resets the
      data associated to this merged body. Fires onRemove event if desired.
      \param rb - body to erase
      \param fireOnRemoveEvent - true to fire onRemove event for the body (ONLY FALSE IF YOU EXPLICITLY SEDN THE EVENT LATER)
      */
      void erase( agx::RigidBody* rb, agx::Bool fireOnRemoveEvent = true );

      /**
      Removes all connections to the body and erases the body from this merged body.
      \param rb - body to remove
      \param saveRemovedEdges - true to save removed edges (in order to generate interactions later)
      */
      void remove( agx::RigidBody* rb, MergedData* rbData, agx::Bool saveRemovedEdges );

      /**
      Removes edge and optionally stores the edge for later to generate interactions.
      \param edge - edge to remove
      \param saveRemovedEdge - true to save the edge for later to generate interaction data, false to ignore
      \return true if removed - otherwise false
      */
      agx::Bool remove( EdgeInteraction* edge, agx::Bool saveRemovedEdge );

      /**
      Remove connection between two bodies. Assumes no edges between the bodies has been
      removed. If one or both bodies doesn't have any more connections, the body will
      be erased from this merged body as well.
      */
      void remove( PairInteractionsIterator it );

      /**
      Fires onAdd event to listeners.
      */
      void fireOnAdd( agx::RigidBody* rb ) const;

      /**
      Fires onRemove event to listeners.
      */
      void fireOnRemove( agx::RigidBody* rb ) const;

      /**
      Fires onAdd event to listeners.
      */
      void handleOnAdd( EdgeInteraction* edge );

      /**
      Fires onRemove event to listeners.
      */
      void handleOnRemove( const EdgeInteractionRefContainer& edges );

      /**
      Fires onMoveTo event to listeners.
      */
      void handleOnMovedTo( const EdgeInteractionRefContainer& edges, MergedBody* newMergedBody );

      /**
      Updates mass properties given all bodies, center of mass position and the total mass.
      */
      void updateDynamicsProperties( agx::RigidBody::MotionControl motionControl,
                                     const agx::Vec3& centerOfMassPosition,
                                     agx::Real totalMass );

      /**
      Adds the generated interactions to the simulation. Interactions that are
      generated after collision detection and the solver are added automatically.
      */
      void commitGeneratedInteractions();

    private:
      RigidBodyRef                m_rb;
      PairInteractionsContainer   m_pairInteractions;
      EdgeInteractionRefContainer m_removedEdges;
      ListenerContainer           m_listeners;
      RigidBodyContainer          m_bodies;
      agxSDK::Simulation*         m_simulation;
      mutable InternalState       m_state;
      agx::Mutex                  m_mutex;
      agx::Name                   m_name;
  };

  /**
  Base class for merged body listener listener.
  */
  class AGXPHYSICS_EXPORT MergedBody::Listener : public agx::Referenced
  {
    public:
      /**
      This listener will be cloned when e.g., splitIslands is used.
      \note It's valid if the clone return itself.
      \return a clone of this listener
      */
      virtual ListenerRef clone() = 0;

      /**
      Called when a body is added to the merged body.
      */
      virtual void onAdd( agx::RigidBody* /*rb*/, const agx::MergedBody* /*mergedBody*/ ) const {}

      /**
      Called when a body is removed from the merged body.
      */
      virtual void onRemove( agx::RigidBody* /*rb*/, const agx::MergedBody* /*mergedBody*/ ) const {}

      /**
      Called when an edge is added to the merged body.
      */
      virtual void onAdded( EdgeInteraction* /*edge*/, const agx::MergedBody* /*mergedBody*/ ) const {}

      /**
      Called when an edge is removed from the merged body.
      */
      virtual void onRemoved( const MergedBody::EdgeInteractionRefContainer& /*edges*/, const agx::MergedBody* /*mergedBody*/ ) const {}

      /**
      Called when the edges between two bodies are moved from one merged body to another.
      */
      virtual void onMovedFromTo( const MergedBody::EdgeInteractionRefContainer& /*edges*/, const agx::MergedBody* /*fromMergedBody*/, const agx::MergedBody* /*toMergedBody*/ ) const {}

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~Listener();
  };

  /**
  Base type for general interaction with two rigid bodies.
  */
  class AGXPHYSICS_EXPORT MergedBody::EdgeInteraction : public agx::Referenced, public agxStream::Serializable
  {
    public:
#ifndef SWIG
      enum InteractionTag
      {
        NONE = 0,
        CONTACT = 1 << 0,
        CONTACT_GENERATOR = 1 << 1,
        CONSTRAINT = 1 << 2,
        BINARY_CONSTRAINT = CONSTRAINT | (1 << 3),
        PARTICLE_GENERATOR = 1 << 4
      };
#endif

    public:
      /**
      \return first rigid body
      */
      agx::RigidBody* getRigidBody1() const;

      /**
      \return second rigid body
      */
      agx::RigidBody* getRigidBody2() const;

      /**
      \return true if one or both objects has been deleted
      */
      agx::Bool containsDeletedBodies() const;

      /**
      \return true if none of the objects has been deleted
      */
      agx::Bool getValid() const;

#ifndef SWIG
      /**
      \return true if this edge has the given tag
      */
      agx::Bool isTagged( InteractionTag tag ) const;
#endif

    public:
      /**
      Generate interaction or do nothing. This method is called when this
      edge interaction is split, i.e., when removed from a merged body.
      */
      virtual void generateInteraction( agxSDK::Simulation* simulation, agxCollide::LocalGeometryContactVector& newContacts ) = 0;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( MergedBody::EdgeInteraction );

      /**
      Stores the two bodies and the construction state.
      */
      virtual void store( agxStream::OutputArchive& out ) const override;

      /**
      Restores the two bodies and the construction state.
      */
      virtual void restore( agxStream::InputArchive& in ) override;

      /**
      Store non-structural data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore non-structural data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
#ifndef SWIG
      /**
      Construct given two rigid bodies and a valid flag.
      \param rb1 - first rigid body
      \param rb2 - second rigid body
      \param tag - Tag type of the interaction
      \param valid - true if valid, false if invalid
      */
      EdgeInteraction( agx::RigidBody* rb1, agx::RigidBody* rb2, InteractionTag tag, agx::Bool valid = true );
#endif

      /**
      Reference counted object, protected destructor.
      */
      virtual ~EdgeInteraction();

      /**
      Assign bodies during restore.
      */
      void setBodies( agx::RigidBody* rb1, agx::RigidBody* rb2 );

    private:
      agx::RigidBodyObserver m_bodies[ 2 ];
      agx::Bool              m_hadValidRb[ 2 ];
      agx::Bool              m_valid;
      agx::Int32             m_tag;
  };

  /**
  Empty edge interaction class containing only the two bodies.
  This object can be used when e.g., the two bodies have a constraint
  defined between them.
  */
  class AGXPHYSICS_EXPORT MergedBody::EmptyEdgeInteraction : public MergedBody::EdgeInteraction
  {
    public:
      /**
      Construct given two rigid bodies. Both rigid bodies have to be valid.
      */
      EmptyEdgeInteraction( agx::RigidBody* rb1, agx::RigidBody* rb2 );

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( MergedBody::EmptyEdgeInteraction );

      /**
      Store non-structural data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore non-structural data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      /**
      Default constructor used by serialization.
      */
      EmptyEdgeInteraction();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~EmptyEdgeInteraction();

      /**
      Does nothing.
      */
      virtual void generateInteraction( agxSDK::Simulation* /*simulation*/, agxCollide::LocalGeometryContactVector& /*newContacts*/ ) override {}
  };

  /**
  Edge interaction that calls colliders for each pair of geometries in
  the bodies during split.
  */
  class AGXPHYSICS_EXPORT MergedBody::ContactGeneratorEdgeInteraction : public MergedBody::EdgeInteraction
  {
    public:
      /**
      Construct given two rigid bodies. Both rigid bodies have to be valid.
      */
      ContactGeneratorEdgeInteraction( agx::RigidBody* rb1, agx::RigidBody* rb2 );

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( MergedBody::ContactGeneratorEdgeInteraction );

      /**
      Store non-structural data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore non-structural data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      /**
      Default constructor used by serialization.
      */
      ContactGeneratorEdgeInteraction();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~ContactGeneratorEdgeInteraction();

      /**
      Checks for contacts between the bodies and adds them.
      */
      virtual void generateInteraction( agxSDK::Simulation* simulation, agxCollide::LocalGeometryContactVector& newContacts ) override;
  };

  /**
  Interaction data for two (previously contacting) merged objects.
  */
  class AGXPHYSICS_EXPORT MergedBody::GeometryContactEdgeInteraction : public MergedBody::EdgeInteraction
  {
    public:
      /**
      Contact point data store in first rigid body local frame.
      */
      struct AGXPHYSICS_EXPORT Point
      {
        enum Strength
        {
          NORMAL_DIRECTION,
          TANGENT_U_DIRECTION,
          TANGENT_V_DIRECTION
        };

        Point();
        Point( agx::Physics::ContactPointPtr pointPtr, agx::Physics::ContactMaterialPtr contactMaterial, const agx::AffineMatrix4x4& toLocal, const agx::RigidBody* rb1, const agx::RigidBody* rb2 );

        agx::Real depth;
        agx::Vec3 localPoint;
        agx::Vec3 localNormal;
        agx::Vec3 localTangentU;
        agx::Vec3 localTangentV;
        agx::Vec3 strength;
      };
      typedef agx::VectorPOD< Point > PointContainer;

    public:
      /**
      Construct given a geometry contact data.
      \param gcPtr - geometry contact data
      */
      explicit GeometryContactEdgeInteraction( agx::Physics::GeometryContactPtr gcPtr );

      /**
      Construct given two geometries. Use this method when this edge is being
      restored using light data.
      \param geometry1 - first geometry
      \param rb1 - First rigid body
      \param geometry2 - second geometry
      \param rb2 - Second rigid body
      */
      GeometryContactEdgeInteraction( agxCollide::Geometry* geometry1, agx::RigidBody* rb1, agxCollide::Geometry* geometry2, agx::RigidBody* rb2 );

      /**
      \return first geometry
      */
      agxCollide::Geometry* getGeometry1() const;

      /**
      \return second geometry
      */
      agxCollide::Geometry* getGeometry2() const;

      /**
      \return vector with contact point data
      */
      const PointContainer& getPoints() const;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( MergedBody::GeometryContactEdgeInteraction );

      /**
      Store non-structural data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore non-structural data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

      DOXYGEN_START_INTERNAL_BLOCK()
#ifndef SWIG
      /**
      Callback function of the form:
      agxCollide::Geometry* geometryUpdateFunction( agxCollide::Geometry* currentGeometry, agx::RigidBody* currentRigidBody )
      */
      typedef std::function<agxCollide::Geometry*( agxCollide::Geometry*, agx::RigidBody* )> GeometryUpdateFunction;

      /**
      Internal method.

      Solves (light) restored edge where the geometry hasn't been created when
      this edge was created given (nullptr, rb1, geometry2, rb2). When the
      geometry has been created it's possible to update the pointer to the
      geometries, assuming the bodies are the same.
      */
      void updateGeometries( GeometryUpdateFunction geometryUpdateFunction );
#endif
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Default constructor used by serialization.
      */
      GeometryContactEdgeInteraction();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~GeometryContactEdgeInteraction();

      /**
      Transforms and adds the local contact data as a new geometry contact.
      */
      virtual void generateInteraction( agxSDK::Simulation* simulation, agxCollide::LocalGeometryContactVector& newContacts ) override;

    private:
      PointContainer m_points;
      agxCollide::GeometryObserver m_geometries[ 2 ];
  };

  /**
  Interaction data for a (one or) two body constraint.
  */
  class AGXPHYSICS_EXPORT MergedBody::BinaryConstraintEdgeInteraction : public MergedBody::EdgeInteraction
  {
    public:
      /**
      Construct given a one or two body constraint.
      \param constraint - one or two body constraint
      */
      explicit BinaryConstraintEdgeInteraction( agx::Constraint* constraint );

      /**
      Construct given constraint implementation of a binary constraint.
      \param constraint - binary constraint
      */
      explicit BinaryConstraintEdgeInteraction( agx::HighLevelConstraintImplementation* constraint );

      /**
      \return the constraint
      */
      agx::HighLevelConstraintImplementation* getConstraint();

      /**
      \return the constraint
      */
      const agx::HighLevelConstraintImplementation* getConstraint() const;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( MergedBody::BinaryConstraintEdgeInteraction );

      /**
      Store non-structural data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore non-structural data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      /**
      Default constructor used by serialization.
      */
      BinaryConstraintEdgeInteraction();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~BinaryConstraintEdgeInteraction();

      /**
      Activates the constraint.
      */
      virtual void generateInteraction( agxSDK::Simulation* simulation, agxCollide::LocalGeometryContactVector& newContacts ) override;

    private:
      agx::HighLevelConstraintImplementation* m_constraint;
  };

  /**
  Edge interaction that handles connection between rigid bodies that are converted from granular bodies. Functionality is the same as Empty Edge.
  */
  class AGXPHYSICS_EXPORT MergedBody::ParticleParticleInteraction : public MergedBody::EdgeInteraction
  {
  public:
    /**
    Construct given two rigid bodies. Both rigid bodies have to be valid.
    */
    ParticleParticleInteraction(agx::RigidBody* substituteBody1, agx::RigidBody* substituteBody2);

  public:
    AGXSTREAM_DECLARE_SERIALIZABLE(MergedBody::ParticleParticleInteraction);

    /**
    Store non-structural data to stream.
    */
    virtual void storeLightData(agxStream::StorageStream& str) const override;

    /**
    Restore non-structural data from stream.
    */
    virtual void restoreLightData(agxStream::StorageStream& str) override;

  protected:
    /**
    Default constructor used by serialization.
    */
    ParticleParticleInteraction();

    /**
    Reference counted object, protected destructor.
    */
    virtual ~ParticleParticleInteraction();

    /**
    Checks for contacts between the bodies and adds them.
    */
    virtual void generateInteraction(agxSDK::Simulation* /*simulation*/, agxCollide::LocalGeometryContactVector& /*newContacts*/) override;
  };

  template<typename DataT, typename ObjT>
  AGX_FORCE_INLINE DataT* MergedBody::getMergedData( const ObjT& obj )
  {
    return agx::InternalData::get<DataT>( obj, agx::InternalData::MERGED_BODY );
  }

  template<typename DataT, typename ObjT>
  AGX_FORCE_INLINE DataT* MergedBody::getOrCreateMergedData( ObjT obj )
  {
    agxAssert( obj != nullptr );
    return agx::InternalData::getOrCreate<DataT>( obj, agx::InternalData::MERGED_BODY );
  }

  AGX_FORCE_INLINE const MergedBody* MergedBody::get( const RigidBody* rb )
  {
    const MergedData* data = getMergedData<MergedData>( rb );
    return data != nullptr ? data->mergedBody : nullptr;
  }

  AGX_FORCE_INLINE MergedBody* MergedBody::get( RigidBody* rb )
  {
    MergedData* data = getMergedData<MergedData>( rb );
    return data != nullptr ? data->mergedBody : nullptr;
  }

  AGX_FORCE_INLINE const MergedBody* MergedBody::getActive( const RigidBody* rb )
  {
    const MergedBody* mb = MergedBody::get( rb );
    return mb != nullptr && mb->isActive() ? mb : nullptr;
  }

  AGX_FORCE_INLINE MergedBody* MergedBody::getActive( RigidBody* rb )
  {
    MergedBody* mb = MergedBody::get( rb );
    return mb != nullptr && mb->isActive() ? mb : nullptr;
  }

  inline Bool MergedBody::isRoot( const RigidBody* rb )
  {
    return rb != nullptr &&
           rb->getEntity().state().mergedRootBody();
  }

  AGX_FORCE_INLINE MergedBody::MergedBodyIndex MergedBody::findMergedBodyIndex( Physics::RigidBodyPtr rb )
  {
    if ( !rb )
      return MergedBodyIndex();
    const MergedBody* root = getActive( rb.model() );
    return root != nullptr ? MergedBodyIndex( root->getRigidBody()->getEntity(), rb ) : MergedBodyIndex( rb, rb );
  }

  AGX_FORCE_INLINE const RigidBody* MergedBody::filter( const RigidBody* rb )
  {
    const MergedBody* root = getActive( rb );
    return root != nullptr ? root->getRigidBody() : rb;
  }

  AGX_FORCE_INLINE RigidBody* MergedBody::filter( RigidBody* rb )
  {
    MergedBody* root = getActive( rb );
    return root != nullptr ? root->getRigidBody() : rb;
  }

  AGX_FORCE_INLINE Bool MergedBody::isMerged( const MergedData* data )
  {
    return data != nullptr && data->mergedBody != nullptr;
  }

  AGX_FORCE_INLINE Bool MergedBody::isActive() const
  {
    return m_simulation != nullptr && m_state.Is( READY_FOR_SOLVER );
  }

  AGX_FORCE_INLINE MergedBody::RootBodyData* MergedBody::getRootBodyData( const RigidBody* rb )
  {
    MergedData* data = getMergedData<MergedData>( rb );
    return data != nullptr ? data->rootBodyData : nullptr;
  }

  DOXYGEN_START_INTERNAL_BLOCK()
  inline Bool MergedBody::EdgeInteraction::isTagged(InteractionTag tag) const
  {
    return ( m_tag & (Int32)tag ) != 0;
  }
  DOXYGEN_END_INTERNAL_BLOCK()
}

