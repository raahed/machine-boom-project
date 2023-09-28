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

#include <agxSDK/MergeSplitAlgorithm.h>
#include <agxSDK/MergeSplitProperties.h>

#include <agxCollide/Space.h>

namespace agxWire
{
  class Wire;
}

namespace agxSDK
{
  /**
  Base class for a merge split algorithm handling a set of merged bodies.
  */
  class AGXPHYSICS_EXPORT MergeSplitHandler : public agx::Referenced
  {
    public:
      typedef agx::Vector<agxSDK::MergeSplitAlgorithmRef> MergeSplitAlgorithmContainer;
      typedef agx::Vector<agx::MergedBodyRef> MergedBodyRefContainer;

    public:
      /**
      Creates (or returns already present) merge-split parameters given a rigid body.
      This method will return null if the rigid body is null.
      \param rb - rigid body to get merge-split parameters for
      \return merge-split parameters for the rigid body
      */
      static agxSDK::MergeSplitProperties* getOrCreateProperties( agx::RigidBody* rb );

      /**
      \param rb - rigid body with parameters
      \return the merge-split parameters for \p rb if present, otherwise null
      */
      static const agxSDK::MergeSplitProperties* getProperties( const agx::RigidBody* rb );

      /**
      \param rb - rigid body with parameters
      \return the merge-split parameters for \p rb if present, otherwise null
      */
      static agxSDK::MergeSplitProperties* getProperties( agx::RigidBody* rb );

      /**
      Creates (or returns already present) merge-split parameters given a geometry.
      If this geometry currently doesn't have any merge-split properties but has
      a rigid body with merge-split properties as parent - the merge-split properties
      of the rigid body will be cloned.
      This method will return null if the geometry is null.
      \param geometry - geometry to get merge-split parameters for
      \return merge-split parameters for the geometry
      */
      static agxSDK::MergeSplitProperties* getOrCreateProperties( agxCollide::Geometry* geometry );

      /**
      Merge-split properties for the geometry or inherited by the rigid body this geometry belongs to.
      To create merge-split properties for this geometry, use getOrCreateProperties( geometry ).
      \param geometry - geometry with parameters
      \return the merge-split parameters for \p geometry or parent rigid body if present, otherwise null
      */
      static const agxSDK::MergeSplitProperties* getProperties( const agxCollide::Geometry* geometry );

      /**
      Merge-split properties for the geometry or inherited by the rigid body this geometry belongs to.
      To create merge-split properties for this geometry, use getOrCreateProperties( geometry ).
      \param geometry - geometry with parameters
      \return the merge-split parameters for \p geometry or parent rigid body if present, otherwise null
      */
      static agxSDK::MergeSplitProperties* getProperties( agxCollide::Geometry* geometry );

      /**
      Creates (or returns already present) merge-split parameters given a constraint.
      This method will return null if the constraint is null.
      \param constraint - constraint to get merge-split parameters for.
      \return merge-split parameters for the constraint.
      */
      static agxSDK::MergeSplitProperties* getOrCreateProperties( agx::Constraint* constraint );

      /**
      Merge-split properties for the constraint, if created - otherwise null.
      \param constraint - constraint with parameters.
      \return the merge-split parameters for \p constraint if present, otherwise null.
      */
      static agxSDK::MergeSplitProperties* getProperties( agx::Constraint* constraint );

      /**
      Merge-split properties for the constraint, if created - otherwise null.
      \param constraint - ConstraintImplementation with parameters.
      \return the merge-split parameters for \p constraint if present, otherwise null.
      */
      static const agxSDK::MergeSplitProperties* getProperties( const agx::Constraint* constraint );

      /**
      Creates (or returns already present) merge-split parameters given a constraint implementation.
      Currently only the 'merge' property is used.
      This method will return null if the constraint is null.
      \param constraint - constraint implementation to get merge-split parameters for.
      \return merge-split parameters for the constraint implementation.
      */
      static agxSDK::MergeSplitProperties* getOrCreateProperties( agx::ConstraintImplementation* constraint );

      /**
      Merge-split properties for the constraint, if created - otherwise null.
      \param constraint - ConstraintImplementation with parameters.
      \return the merge-split parameters for \p constraint if present, otherwise null.
      */
      static agxSDK::MergeSplitProperties* getProperties( agx::ConstraintImplementation* constraint );

      /**
      Merge-split properties for the constraint, if created - otherwise null.
      \param constraint - ConstraintImplementation with parameters.
      \return the merge-split parameters for \p constraint if present, otherwise null.
      */
      static const agxSDK::MergeSplitProperties* getProperties( const agx::ConstraintImplementation* constraint );

      /**
      Creates (or returns already present) merge-split parameters given a wire.
      This method will return null if the wire is null.
      \param wire - wire to get merge-split parameters for
      \return merge-split parameters for the wire
      */
      static agxSDK::MergeSplitProperties* getOrCreateProperties( agxWire::Wire* wire );

      /**
      \param wire - wire with parameters
      \return the merge-split parameters for \p wire if present, otherwise null
      */
      static agxSDK::MergeSplitProperties* getProperties( agxWire::Wire* wire );

      /**
      \param wire - wire with parameters
      \return the merge-split parameters for \p wire if present, otherwise null
      */
      static const agxSDK::MergeSplitProperties* getProperties( const agxWire::Wire* wire );

      /**
      Transfers merge-split properties from source geometry to another geometry instance.
      Then \p destination instance will share merge-split properties instance with \p source.
      \param source - source geometry instance
      \param destination - destination geometry instance
      */
      static void transferProperties( const agxCollide::Geometry* source, agxCollide::Geometry* destination );

      /**
      Transfers merge-split properties from source rigid body to another rigid body instance.
      Then \p destination instance will share merge-split properties instance with \p source.
      \param source - source rigid body instance
      \param destination - destination rigid body instance
      */
      static void transferProperties( const agx::RigidBody* source, agx::RigidBody* destination );

      /**
      Transfers merge-split properties from the wire to the rigid body.
      \param wire - parent wire object
      \param rb - body, part of \p wire, to get the merge-split properties of the wire
      */
      static void transferProperties( const agxWire::Wire* wire, agx::RigidBody* rb );

      /**
      Transfers merge-split properties from the wire to the constraint.
      \param wire - parent wire object
      \param constraint - constraint in \p wire, to get the merge-split properties of the wire
      */
      static void transferProperties( const agxWire::Wire* wire, agx::Constraint* constraint );

      /**
      Determine if a body is merged explicitly (added to an agx::MergedBody by the user) or merged
      by an agxSDK::MergeSplitHandler.
      \return true if the body is merged and is merged by an agxSDK::MergeSplitHandler - otherwise false
      */
      static agx::Bool isMergedByHandler( const agx::RigidBody* rb );

      /**
      Split a rigid body that has been merged by a merge split handler.
      \param rb - body to split
      \return true if the body were merged by a merge split handler and now is successfully removed, otherwise false
      */
      static agx::Bool split( agx::RigidBody* rb );

    public:
      /**
      Enable/disable merge split.
      \param enable - true to enable merge split
      */
      void setEnable( agx::Bool enable );

      /**
      \return true if merge split is enable
      */
      agx::Bool getEnable() const;

      /**
      Enable/disable serialization of this object. Default: enabled.
      \param enable - true to enable, false to disable serialization
      */
      void setEnableSerialization( agx::Bool enable );

      /**
      \return true if serialization of this object is enabled - otherwise false
      */
      agx::Bool getEnableSerialization() const;

      /**
      Enable/disable global merge and split for all objects not having their MergeSplitProperties set.
      \param enable - true to enable, false to disable
      */
      void setEnableGlobalMergeSplit( agx::Bool enable );

      /**
      \return true if merge-split is enabled globally for objects not having MergeSplitProperties
      */
      agx::Bool getEnableGlobalMergeSplit() const;

      /**
      Enable/disable global merge for all objects not having their MergeSplitProperties set.
      \param enable - true to enable, false to disable
      */
      void setEnableGlobalMerge( agx::Bool enable );

      /**
      \return true if merge is enabled globally for objects not having MergeSplitProperties
      */
      agx::Bool getEnableGlobalMerge() const;

      /**
      Enable/disable global split for all objects not having their MergeSplitProperties set.
      \param enable - true to enable, false to disable
      */
      void setEnableGlobalSplit( agx::Bool enable );

      /**
      \return true if split is enabled globally for objects not having MergeSplitProperties
      */
      agx::Bool getEnableGlobalSplit() const;


      /**
      Enable/disable split on separation events from Space.
      \param enable - true to enable, false to disable.
      */
      void setEnableSplitOnSeparation(agx::Bool enable);

      /**
      \return true if split on separation is enabled.
      */
      agx::Bool getEnableSplitOnSeparation() const;

      /**
      If enable is false, from now on reject merges between bodies in the two
      groups even when the MergeSplitProperties of the two bodies would allow
      the merge to happen.

      Merge groups cannot be used to enable merge for bodies whose
      MergeSplitProperty does not allow the merge to happen. It is a reject
      filter only.

      \param group1 - The first group.
      \param group2 - The second group.
      \param enable - True if merges should be rejected. False to restore default behavior.
      */
      void setEnableMergePair(const agx::Name& group1, const agx::Name& group2, agx::Bool enable);

      /**
      If enable is false, from now on reject merges between bodies in the two
      groups even when the MergeSplitProperties of the two bodies would allow
      the merge to happen.

      Merge groups cannot be used to enable merge for bodies whose
      MergeSplitProperty does not allow the merge to happen. It is a reject
      filter only.

      \param group1 - The first group.
      \param group2 - The second group.
      \param enable - True if merges should be rejected. False to restore default behavior.
      */
      void setEnableMergePair(agx::UInt32 group1, agx::UInt32 group2, agx::Bool enable);

      /**
      If enable is false, from now on reject merges between bodies in the two
      groups even when the MergeSplitProperties of the two bodies would allow
      the merge to happen.

      Merge groups cannot be used to enable merge for bodies whose
      MergeSplitProperty does not allow the merge to happen. It is a reject
      filter only.

      \param group1 - The first group.
      \param group2 - The second group.
      \param enable - True if merges should be rejected. False to restore default behavior.
      */
      void setEnableMergePair(agx::UInt32 group1, const agx::Name& group2, agx::Bool enable);

      /**
      If enable is false, from now on reject merges between bodies in the two
      groups even when the MergeSplitProperties of the two bodies would allow
      the merge to happen.

      Merge groups cannot be used to enable merge for bodies whose
      MergeSplitProperty does not allow the merge to happen. It is a reject
      filter only.

      \param group1 - The first group.
      \param group2 - The second group.
      \param enable - True if merges should be rejected. False to restore default behavior.
      */
      void setEnableMergePair(const agx::Name& group1, agx::UInt32 group2, agx::Bool enable);

      /// \return False if merging between the two groups has been disabled.
      agx::Bool getEnableMergePair(const agx::Name& group1, const agx::Name& group2);

      /// \return False if merging between the two groups has been disabled.
      agx::Bool getEnableMergePair(agx::UInt32 group1, agx::UInt32 group2);

      /// \return False if merging between the two groups has been disabled.
      agx::Bool getEnableMergePair(agx::UInt32 group1, const agx::Name& group2);

      /// \return False if merging between the two groups has been disabled.
      agx::Bool getEnableMergePair(const agx::Name& group1, agx::UInt32 group2);

      /**
      Add new merge-split algorithm.
      \param mergeSplitAlgorithm - new merge-split algorithm
      \return true if added, false if already present or invalid
      */
      agx::Bool add( agxSDK::MergeSplitAlgorithm* mergeSplitAlgorithm );

      /**
      Remove split algorithm.
      \param mergeSplitAlgorithm - merge-split algorithm to remove
      \return true if removed, false if not present or invalid
      */
      agx::Bool remove( agxSDK::MergeSplitAlgorithm* mergeSplitAlgorithm );

      /**
      \return the merge split algorithms
      */
      MergeSplitAlgorithmContainer getAlgorithms() const;

      /**
      \return the merged bodies created and handled by this handler
      */
      const MergedBodyRefContainer& getMergedBodies() const;

      /**
      Merge bodies given an edge interaction.
      \param edgeInteraction - edge interaction between the bodies
      \return true if the result is that the bodies are merged
      */
      agx::Bool merge( agx::MergedBody::EdgeInteraction* edgeInteraction );

      /**
      Creates a new, empty, merged body object associated to this merge split handler.
      I.e., bodies added to this merged body may be split, merged etc by the algorithms
      in this handler.
      */
      agx::MergedBodyRef createMergedBody() const;

      /**
      Register merged body to this merge-split handler. After this call, \p mergedBody may
      be split and merged with others. Even removed from the simulation.
      \param mergedBody - merged body to hand over the control of to this handler
      */
      void registerMergedBody( agx::MergedBody* mergedBody );

      /**
      Global thresholds for contacts. These thresholds are used for all geometries/bodies
      that hasn't got explicitly assigned GeometryContactMergeSplitThresholds instances.
      \return the global contact thresholds
      */
      agxSDK::GeometryContactMergeSplitThresholds* getGlobalContactThresholds() const;

      /**
      Global thresholds for constraints. These thresholds are used for all constraints
      that hasn't got explicitly assigned ConstraintMergeSplitThresholds instances.
      \return the global constraint thresholds
      */
      agxSDK::ConstraintMergeSplitThresholds* getGlobalConstraintThresholds() const;

      /**
      Global thresholds for wires. These thresholds are used for all wires
      that hasn't got explicitly assigned WireMergeSplitThresholds instances.
      \return the global wire thresholds
      */
      agxSDK::WireMergeSplitThresholds* getGlobalWireThresholds() const;

      /**
      \return true if merge is enabled between two objects with merge-split properties
      */
      agx::Bool mayMerge( const agx::RigidBody* rb1, const agx::RigidBody* rb2 ) const;

      /**
      \return true if merge is enabled between the two geometries
      */
      agx::Bool mayMerge( const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2 ) const;

      /**
      \return true if the bodies may merge given associated properties (the properties could belong to a geometry in the body)
      */
      agx::Bool mayMerge( const agx::RigidBody* rb1,
                          const agxSDK::MergeSplitProperties* p1,
                          const agx::RigidBody* rb2,
                          const agxSDK::MergeSplitProperties* p2 ) const;

      /**
      \return true if split is possible for the body
      */
      agx::Bool maySplit( const agx::RigidBody* rb ) const;

      /**
      \return true if split is enabled in the given properties or globally enabled
      */
      agx::Bool maySplit( const agx::RigidBody* rb, const agxSDK::MergeSplitProperties* properties ) const;

      /**
      \return true if merge is enabled in the given properties or globally enabled
      */
      agx::Bool mayMergeProperties( const agxSDK::MergeSplitProperties* properties ) const;

      /**
      \return true if split is enabled in the given properties or globally enabled
      */
      agx::Bool maySplitProperties( const agxSDK::MergeSplitProperties* properties ) const;

      /**
      \return the simulation
      */
      const agxSDK::Simulation* getSimulation() const;

      /**
      Write the MergedBody graph as a .dot file to disk.
      */
      bool writeConnectivityGraph(const char* filename) const;

#ifndef SWIG
      /**
      Traverse all edges currently active in all merged bodies handled by this handler.
      \param visitor - visitor function
      */
      void traverse( agx::MergedBody::EdgeInteractionVisitor visitor ) const;

      /**
      \return the algorithm of given type
      */
      template<typename T>
      T* getAlgorithm() const;
#endif

    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      static void storePersistentData( const agx::RigidBody* rb, agxStream::OutputArchive& out );
      static void storePersistentData( const agxCollide::Geometry* geometry, agxStream::OutputArchive& out );
      static void storePersistentData( const agx::HighLevelConstraintImplementation* constraint, agxStream::OutputArchive& out );
      static void storePersistentData( const agxWire::Wire* wire, agxStream::OutputArchive& out );

      static void restorePersistentData( agx::RigidBody* rb, agxStream::InputArchive& in );
      static void restorePersistentData( agxCollide::Geometry* geometry, agxStream::InputArchive& in );
      static void restorePersistentData( agx::HighLevelConstraintImplementation* constraint, agxStream::InputArchive& in );
      static void restorePersistentData( agxWire::Wire* wire, agxStream::InputArchive& in );

      template< typename T >
      static T* getOrCreateCustomData( agx::HighLevelConstraintImplementation* constraint );

      template< typename T >
      static T* getCustomData( const agx::HighLevelConstraintImplementation* constraint );

      static agx::Bool isMergedGranulate( const agx::RigidBody* rb );
      static void setIsMergedGranulate( agx::RigidBody* rb, agx::Bool flag );
      DOXYGEN_END_INTERNAL_BLOCK()

    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Callback after the IntegratePositions task has been executed.
      */
      virtual void postIntegrate();
      DOXYGEN_END_INTERNAL_BLOCK()

    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Callback from solver with interaction graph edge.
      */
      void postSolve( const agxSDK::MergeSplitPostSolveData& postSolveData );

      /**
      \return locally stored thresholds in obj or global thresholds defined in this merge split handler
      */
      template<typename T, typename ObjT>
      const T* getThresholds( ObjT obj, agxSDK::MergeSplitAlgorithm::Callback callback ) const;

      /**
      \return locally stored thresholds in properties or global thresholds defined in this merge split handler
      */
      template<typename T>
      const T* getThresholdsFromProperties( const agxSDK::MergeSplitProperties* properties, agxSDK::MergeSplitAlgorithm::Callback callback ) const;

      /**
      \return true if the given thresholds is the default
      */
      agx::Bool isDefault( const agxSDK::MergeSplitThresholds* thresholds, agxSDK::MergeSplitAlgorithm::Callback callback ) const;
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~MergeSplitHandler();

      /**
      Pre-collide callback from simulation.
      */
      virtual void preCollide();

      /**
      Separation ballback from simulation.
      \param separations
      */
      virtual void separations(const agxCollide::SeparationPairVector& separations);

      /**
      Pre-solver callback from simulation.
      */
      virtual void preStep();

      /**
      Post-solver callback from simulation.
      */
      virtual void postStep();

      /**
      Reset call during simulation cleanup.
      \param resetState - if true the internal state will also be set to default
      */
      virtual void reset( agx::Bool resetState = true );

      /**
      Call before merge-split callbacks are being processed.
      */
      virtual void mergeSplitBegin();

      /**
      Call after merge-split callbacks have been processed.
      */
      virtual void mergeSplitEnd();

      /**
      Store call from the simulation.
      */
      virtual void store( agxStream::OutputArchive& out ) const;

      /**
      Restore call from the simulation.
      */
      virtual void restore( agxStream::InputArchive& in );

    private:
      /**
      State of this handler, e.g., to enable merge and/or split globally.
      */
      enum EState
      {
        ENABLED               = 1 << 0, /**< Enable flag (setEnable( true/false )). */
        GLOBAL_MERGE_ENABLED  = 1 << 1, /**< Global, forced merge enable flag. */
        GLOBAL_SPLIT_ENABLED  = 1 << 2, /**< Global, forced split enable flag. */
        SERIALIZATION_ENABLED = 1 << 3, /**< Enable flag (setEnableSerialization( true/false )). */
        SPLIT_ON_SEPARATION   = 1 << 4, /**< Split bodies on separation events from Space. */
        IN_POST_SOLVE         = 1 << 5  /**< Set during postSolve. */
      };
      typedef agx::BitState< EState, agx::Int32 > State;

      /**
      Internal data base class for objects supported by this handler.
      */
      struct HandlerData : public agx::ICloneable
      {
        agxSDK::MergeSplitPropertiesRef properties;
      };

      /**
      Internal data for supported objects that can be merged. E.g.,
      RigidBody, Constraint, Geometry. When merged by this handler,
      the object will have a pointer back to this for this to
      receive callbacks when the object is deleted or removed.
      */
      struct MergeableHandlerData : public HandlerData
      {
        enum EState
        {
          MERGED_BY_HANDLER = 1 << 0,
          MERGED_GRANULATE  = 1 << 1
        };
        typedef agx::BitState< EState, agx::Int32 > State;

        State state;
        MergeSplitHandler* handler;

        protected:
          MergeableHandlerData() : state(), handler( nullptr ) {}
          virtual ~MergeableHandlerData() {}

        private:
          virtual agx::ICloneableRef clone( agx::ICloneable* /*child*/ ) override
          {
            // It's not supported to clone a merged object?
            // Implement this if the object is cloneable.
            return agx::ICloneableRef( nullptr );
          }
      };
      typedef agx::ref_ptr<MergeableHandlerData> HandlerDataRef;

      /**
      Handler data for bodies.
      */
      struct BodyHandlerData : public MergeableHandlerData
      {
        protected:
          virtual ~BodyHandlerData() {}
      };

      /**
      Handler data for geometries.
      */
      struct GeometryHandlerData : public MergeableHandlerData
      {
        protected:
          virtual ~GeometryHandlerData() {}
      };

      /**
      Handler data for constraints.
      */
      struct ConstraintHandlerData : public MergeableHandlerData
      {
        agx::ref_ptr<agx::Referenced> customData;

        protected:
          virtual ~ConstraintHandlerData() {}
      };
      typedef agx::ref_ptr< ConstraintHandlerData > ConstraintHandlerDataRef;

      /**
      Handler data for wires.
      */
      struct WireHandlerData : public HandlerData
      {
        virtual agx::ICloneableRef clone( agx::ICloneable* child ) override
        {
          if ( child == nullptr )
            child = new WireHandlerData();

          child->as<WireHandlerData>()->properties = properties->clone()->as<MergeSplitProperties>();

          return agx::ICloneableRef( child );
        }
      };

    private:
      using MergeSplitThresholdsContainer = agx::Vector<MergeSplitThresholdsRef>;

    private:
      /**
      \return already created handler data for a given object, nullptr if the object hasn't got any data
      */
      template<typename DataT, typename ObjT>
      static DataT* getHandlerData( const ObjT& obj );

      /**
      \return already created or creates new handler data for a given object (fails/invalid if obj == nullptr)
      */
      template<typename DataT, typename ObjT>
      static DataT* getOrCreateHandlerData( ObjT obj );

      /**
      Transfers properties from source to destination (sharing properties instance).
      */
      template<typename DataT, typename SourceT, typename DestT>
      static void _transferProperties( const SourceT* source, DestT* destination );

      /**
      Sets default state.
      */
      static void setDefaultState( State& state );

    private:
      friend class Simulation;
      /**
      Construction only valid from friend Simulation.
      \param simulation - simulation for this handler
      */
      MergeSplitHandler( agxSDK::Simulation* simulation );

      /**
      Callback from simulation when the constraint is being removed.
      */
      void onRemove( agx::Constraint* constraint );

      friend class MergeSplitMergedBodyListener;
      /**
      Callback from our MergedBody listener when a body is added to a merged body.
      */
      void onAddToMergedBody( agx::RigidBody* rb, const agx::MergedBody* mergedBody );

      /**
      Callback from our MergedBody listener when a body is removed from a merged body.
      */
      void onRemovedFromMergedBody( agx::RigidBody* rb, const agx::MergedBody* mergedBody );

      /**
      Callback from our MergedBody listener when an edge is added to a merged body.
      */
      void onAddToMergedBody( agx::MergedBody::EdgeInteraction* edge, const agx::MergedBody* mergedBody );

      /**
      Callback from our MergedBody listener when a set of edges are removed from a merged body.
      */
      void onRemovedFromMergedBody( const agx::MergedBody::EdgeInteractionRefContainer& edges, const agx::MergedBody* mergedBody );

      /**
      Callback from our MergedBody listener when a set of edges are moved from one merged body to another.
      */
      void onMovedFromToMergedBody( const agx::MergedBody::EdgeInteractionRefContainer& edges, const agx::MergedBody* fromMergedBody, const agx::MergedBody* toMergedBody );

    private:
      /**
      Removes added algorithms and adds the default ones.
      */
      void clearAndAddDefaultAlgorithms();

      /**
      Handle possible Granulate Bodies
      */
      void handlePossibleGranulateBodies(bool updateSpace);

      /**
      \return true if the body is merged by this handler, otherwise false
      */
      agx::Bool getMergedByHandler( const agx::RigidBody* rb ) const;

      /**
      Checks state of merged bodies and adds/removes the root bodies to/from
      the simulation.
      \param commitGeneratedInteractions - true to add generated interactions to the simulation before
                                           an empty merged body is removed. Normally true when in preStep.
      */
      void updateMergedBodiesSimulationState( agx::Bool commitGeneratedInteractions );

      /**
      Checks merged bodies states and performs island splitting if necessary.
      */
      void performIslandSplitting();

      /**
      Applies action to state.
      */
      agx::Bool apply( const agxSDK::MergeSplitAction& action );

      /**
      Merge given state.
      */
      agx::Bool merge( const agxSDK::MergeSplitAction& action );

      /**
      Split rigid body from a merged body.
      */
      agx::Bool split( const agxSDK::MergeSplitAction& action );

    private:
      agxSDK::Simulation*           m_simulation;
      State                         m_state;
      MergeSplitAlgorithmContainer  m_mergeSplitAlgorithms;
      MergedBodyRefContainer        m_mergedBodies;
      MergeSplitActionContainer     m_currentActions;
      agx::MergedBody::ListenerRef  m_mergedBodyListener;
      MergeSplitThresholdsContainer m_globalThresholds;
      MergeIgnoreFilter             m_mergeIgnoreFilter;

      agx::RigidBodyRefVector       m_possibleGranulateBodies;
  };

  AGX_FORCE_INLINE const MergeSplitProperties* MergeSplitHandler::getProperties( const agx::RigidBody* rb )
  {
    const HandlerData* data = getHandlerData<HandlerData>( rb );
    return data != nullptr ? data->properties : nullptr;
  }

  AGX_FORCE_INLINE MergeSplitProperties* MergeSplitHandler::getProperties( agx::RigidBody* rb )
  {
    HandlerData* data = getHandlerData<HandlerData>(rb);
    return data != nullptr ? data->properties : nullptr;
  }

  AGX_FORCE_INLINE const MergeSplitProperties* MergeSplitHandler::getProperties( const agxCollide::Geometry* geometry )
  {
    const HandlerData* data = getHandlerData<HandlerData>( geometry );
    if ( data != nullptr && data->properties != nullptr )
      return data->properties;
    return geometry != nullptr ? MergeSplitHandler::getProperties( geometry->getRigidBody() ) : nullptr;
  }

  AGX_FORCE_INLINE MergeSplitProperties* MergeSplitHandler::getProperties( agxCollide::Geometry* geometry )
  {
    const HandlerData* data = getHandlerData<HandlerData>( geometry );
    if ( data != nullptr && data->properties != nullptr )
      return data->properties;

    // Special case so that the user doesn't have to call getOrCreateProperties for the geometry
    // when the properties are inherited. That's why we have to const_cast this case.
    return geometry != nullptr ? const_cast<MergeSplitProperties*>( MergeSplitHandler::getProperties( geometry->getRigidBody() ) ) : nullptr;
  }

  inline MergeSplitProperties* MergeSplitHandler::getProperties( agx::Constraint* constraint )
  {
    return constraint != nullptr ? MergeSplitHandler::getProperties( constraint->getRep() ) : nullptr;
  }

  inline const MergeSplitProperties* MergeSplitHandler::getProperties( const agx::Constraint* constraint )
  {
    return constraint != nullptr ? MergeSplitHandler::getProperties( constraint->getRep() ) : nullptr;
  }

  AGX_FORCE_INLINE MergeSplitProperties* MergeSplitHandler::getProperties( agx::ConstraintImplementation* constraint )
  {
    const HandlerData* data = getHandlerData<HandlerData>( constraint );
    return data != nullptr ? data->properties : nullptr;
  }

  AGX_FORCE_INLINE const MergeSplitProperties* MergeSplitHandler::getProperties( const agx::ConstraintImplementation* constraint )
  {
    const HandlerData* data = getHandlerData<HandlerData>( constraint );
    return data != nullptr ? data->properties : nullptr;
  }

  inline agx::Bool MergeSplitHandler::getMergedByHandler( const agx::RigidBody* rb ) const
  {
    const MergeableHandlerData* handlerData = getHandlerData<MergeableHandlerData>( rb );
    return handlerData != nullptr && handlerData->state.Is( MergeableHandlerData::MERGED_BY_HANDLER ) && handlerData->handler == this;
  }

  template<typename DataT, typename ObjT>
  inline DataT* MergeSplitHandler::getHandlerData( const ObjT& obj )
  {
    return agx::InternalData::get<DataT>( obj, agx::InternalData::MERGE_SPLIT );
  }

  template<typename DataT, typename ObjT>
  inline DataT* MergeSplitHandler::getOrCreateHandlerData( ObjT obj )
  {
    agxAssert( obj != nullptr );
    return agx::InternalData::getOrCreate<DataT>( obj, agx::InternalData::MERGE_SPLIT );
  }

  template<typename DataT, typename SourceT, typename DestT>
  void MergeSplitHandler::_transferProperties( const SourceT* source, DestT* destination )
  {
    if ( static_cast<const void*>( source ) == static_cast<void*>( destination ) ||
         source == nullptr ||
         destination == nullptr )
      return;

    const auto sourceData = MergeSplitHandler::getHandlerData<HandlerData>( source );
    if ( sourceData == nullptr )
      return;

    auto destinationData = MergeSplitHandler::getOrCreateHandlerData<DataT>( destination );
    destinationData->properties = sourceData->properties;
  }

  template< typename T >
  T* MergeSplitHandler::getOrCreateCustomData( agx::HighLevelConstraintImplementation* constraint )
  {
    ConstraintHandlerData* data = MergeSplitHandler::getOrCreateHandlerData<ConstraintHandlerData>( constraint );
    if ( data->customData == nullptr )
      data->customData = new T();

    return data->customData->as<T>();
  }

  template< typename T >
  T* MergeSplitHandler::getCustomData( const agx::HighLevelConstraintImplementation* constraint )
  {
    const ConstraintHandlerData* data = MergeSplitHandler::getHandlerData<ConstraintHandlerData>( constraint );
    return data != nullptr && data->customData != nullptr ? data->customData->as<T>() : nullptr;
  }

  inline agx::Bool MergeSplitHandler::isMergedGranulate( const agx::RigidBody* rb )
  {
    const BodyHandlerData* data = getHandlerData<BodyHandlerData>( rb );
    return data != nullptr && data->state.Is( MergeableHandlerData::MERGED_GRANULATE );
  }

  inline void MergeSplitHandler::setIsMergedGranulate( agx::RigidBody* rb, agx::Bool flag )
  {
    BodyHandlerData* data = getOrCreateHandlerData<BodyHandlerData>( rb );
    data->state.update( MergeableHandlerData::MERGED_GRANULATE, flag );
  }

  template<typename T>
  T* MergeSplitHandler::getAlgorithm() const
  {
    for ( const auto& algorithm : m_mergeSplitAlgorithms )
      if ( algorithm->template is<T>() )
        return algorithm->template as<T>();
    return nullptr;
  }

  template<typename T, typename ObjT>
  const T* MergeSplitHandler::getThresholds( const ObjT obj, MergeSplitAlgorithm::Callback callback ) const
  {
    return getThresholdsFromProperties<T>( getProperties( obj ), callback );
  }

  template<typename T>
  const T* MergeSplitHandler::getThresholdsFromProperties( const MergeSplitProperties* properties,
                                                           MergeSplitAlgorithm::Callback callback ) const
  {
    const T* thresholds       = nullptr;
    const auto callbackIndex  = MergeSplitProperties::findCallbackIndex( callback );
    if ( properties == nullptr || (thresholds = properties->getMergeSplitThresholds<T>( callbackIndex )) == nullptr )
      thresholds = (T*)m_globalThresholds[ callbackIndex ].get();

    return thresholds;
  }

  inline agx::Bool MergeSplitHandler::isDefault( const MergeSplitThresholds* thresholds, MergeSplitAlgorithm::Callback callback ) const
  {
    return thresholds != nullptr && thresholds == m_globalThresholds[ MergeSplitProperties::findCallbackIndex( callback ) ];
  }

  inline agx::Bool MergeSplitHandler::mayMergeProperties( const MergeSplitProperties* properties ) const
  {
    return ( properties != nullptr && properties->getEnableMerge() ) ||     // Merge enabled in properties?
           ( properties == nullptr && m_state.Is( GLOBAL_MERGE_ENABLED ) ); // Only check global flag if this object hasn't got properties!

  }

  inline agx::Bool MergeSplitHandler::maySplitProperties( const MergeSplitProperties* properties ) const
  {
    return ( properties != nullptr && properties->getEnableSplit() ) ||     // Enabled in properties?
           ( properties == nullptr && m_state.Is( GLOBAL_SPLIT_ENABLED ) ); // Only check global flag if this object hasn't got properties!

  }
}
