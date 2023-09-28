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

#include <agxVehicle/TrackWheel.h>
#include <agxVehicle/TrackRoute.h>
#include <agxVehicle/TrackNodeRange.h>
#include <agxVehicle/TrackInternalMergeProperties.h>
#include <agxVehicle/TrackNodeOnInitializeCallback.h>
#include <agxVehicle/TrackProperties.h>

#include <agxSDK/LinkedStructure.h>

namespace agxVehicle
{
  AGX_DECLARE_POINTER_TYPES( Track );
  using TrackPtrVector = agx::VectorPOD<Track*>;

  /**
  Assembly object representing a continuous track with a given number of shoes (nodes).
  */
  class AGXVEHICLE_EXPORT Track : public agxSDK::LinkedStructure
  {
    public:
      using TrackWheelContainer      = agx::Vector<TrackWheelRef>;
      using RigidBodyTrackWheelTable = agx::HashTable<const agx::RigidBody*, TrackWheel*>;
      using Iterator                 = TrackNodeIterator;
      using NodeRange                = TrackNodeRange;
      using NodeRangeContainer       = agx::VectorPOD<NodeRange>;
      using OnNodeInitializeCallback = std::function<void( const TrackNode& )>;

    public:
      /**
      \param rb - rigid body
      \return track instance \p rb belongs to, given \p rb is a TrackNode body
      */
      static Track* get( agx::RigidBody* rb );

      /**
      \param rb - rigid body
      \return track instance \p rb belongs to, given \p rb is a TrackNode body
      */
      static const Track* get( const agx::RigidBody* rb );

      /**
      \param wheel - track wheel
      \return track instance \p wheel belongs to, nullptr if \p wheel isn't a TrackWheel or not added to a track
      */
      static Track* get( Wheel* wheel );

      /**
      \param wheel - track wheel
      \return track instance \p wheel belongs to, nullptr if \p wheel isn't a TrackWheel or not added to a track
      */
      static const Track* get( const Wheel* wheel );

      /**
      Find track instance given simulation and name.
      \param simulation - simulation
      \param name - name of track instance to find
      \return track instance with given name if found - otherwise null
      */
      static Track* find( agxSDK::Simulation* simulation, const agx::Name& name );

      /**
      Find track instance given simulation and unique id.
      \param simulation - simulation
      \param uuid - UUID of track instance to find
      \return track instance with given UUID if found - otherwise null
      */
      static Track* find( agxSDK::Simulation* simulation, const agx::Uuid& uuid );

      /**
      Finds all agxVehicle::Track instances in the given simulation.
      \param simulation - simulation
      \return track instances found in the given simulation
      */
      static agxVehicle::TrackPtrVector findAll( agxSDK::Simulation* simulation );

      /**
      Finds all track instances given simulation and name.
      \param simulation - simulation
      \param name - name of track instances to find
      \return track instances with given name
      */
      static agxVehicle::TrackPtrVector findAll( agxSDK::Simulation* simulation, const agx::Name& name );

    public:
      /**
      Construct track given number of nodes, size and initial tension distance.
      \param numberOfNodes - number of nodes in the track
      \param width - width of the track (this is an implicit definition, the geometry may be any size)
      \param thickness - thickness of the track (this is an implicit definition, the geometry may be any size)
      \param initialDistanceTension - value (distance) of how much shorter each node should be which causes tension in the
                                      system of tracks and wheels. Ideal case
                                        track_tension = initialDistanceTension * track_constraint_compliance.
                                      Since contacts and other factors are included it's not possible to know
                                      the exact tension after the system has been created.
      */
      Track( agx::UInt numberOfNodes, agx::Real width, agx::Real thickness, agx::Real initialDistanceTension = agx::Real( 0 ) );

      /**
      Construct given a track route.
      \param route - track route
      */
      Track( agxVehicle::TrackRoute* route );

      /**
      Add track wheel to this track before this track has been initialized.
      \param wheel - track wheel to add
      \return true if added - otherwise false (e.g., already added, track initialized, part of another track, null rigid body)
      */
      agx::Bool add( agxVehicle::TrackWheelRef wheel );

      /**
      Remove track wheel.
      \param wheel - wheel to remove
      \return true if removed - otherwise false
      */
      agx::Bool remove( agxVehicle::TrackWheelRef wheel );

      /**
      \return track wheels added to this track
      */
#ifndef SWIG // for the work-around until SWIG upgrade to work
      const agxVehicle::Track::TrackWheelContainer& getWheels() const;
#endif

      /**
      Reference wheel's rotation axis is defining the plane in which this
      track will be created. The reference wheel is (in priority order)
      either a (first) sprocket, a (first) idler or the first wheel added
      to this track.
      */
      agxVehicle::TrackWheel* findReferenceWheel() const;

      /**
      \return the track route used to initialize this track
      */
      agxVehicle::TrackRoute* getRoute() const;

      /**
      \return node range spanning all nodes in this track
      */
#ifndef SWIG // for the work-around until SWIG upgrade to work
      NodeRange nodes() const;

      /**
      \return node range given index range
      */
      NodeRange nodes( IndexRange range ) const;
#endif

      /**
      \return track wheel instance given rigid body - null when the body isn't a wheel in this track
      */
      agxVehicle::TrackWheel* getWheel( const agx::RigidBody* rb ) const;

      /**
      \return the number of nodes in this track
      */
      agx::UInt getNumNodes() const;

      /**
      \return the node of given index
      */
      agxVehicle::TrackNode* getNode( agx::UInt index ) const;

      /**
      \note This method as template to handle ambiguity with index 0 and null.
      \return track node instance given rigid body - null when the body isn't a node in this track
      */
      template<typename T>
      agxVehicle::TrackNode* getNode( const T* obj ) const;

      /**
      \return track node index given rigid body - agx::InvalidIndex when the body isn't a node in this track
      */
      agx::UInt getNodeIndex( const agx::RigidBody* rb ) const;

      /**
      \return track node iterator given node index
      */
      Iterator getIterator( agx::UInt index ) const;

      /**
      \note This method as template to handle ambiguity with index 0 and null.
      \return track node iterator given rigid body - null when the body isn't a node in this track
      */
      template<typename T>
      Iterator getIterator( const T* obj ) const;

      /**
      \return node to node merge properties and thresholds
      */
      agxVehicle::TrackInternalMergeProperties* getInternalMergeProperties() const;

      /**
      \return the properties of this track
      */
      agxVehicle::TrackProperties* getProperties() const;

      /**
      Assign properties to this track.
      \param properties - new properties of this track, if null, a new default instance is created
      */
      void setProperties( agxVehicle::TrackProperties* properties );

      agx::UInt32Vector onPreGetNodeContacts( const agxVehicle::TrackWheel* wheel ) const;

    public:
#ifndef SWIG
      /**
      Initialize this track given already assigned route, added wheels, number of nodes
      and an initialize node callback. If \p onInitializeCallback isn't given, a default
      function will be used that will add a box shape to the node rigid body.
      \param onInitializeCallback - callback when node has been created and is about to be
                                    added to this track. If desired, this callback should
                                    add shape(s) to the node rigid body.
      \return true if initialized, i.e., more than two nodes were created - false if
              already initialized, no wheels in this track, track route failed to find
              node configuration
      */
      virtual agx::Bool initialize( OnNodeInitializeCallback onInitializeCallback );
#endif

      /**
      Initialize this track given already assigned route, added wheels, number of nodes
      and an initialize node callback. If \p onInitializeCallback isn't given, a default
      function will be used that will add a box shape to the node rigid body.
      \param onInitializeCallback - callback when node has been created and is about to be
                                    added to this track. If desired, this callback should
                                    add shape(s) to the node rigid body.
      \return true if initialized, i.e., more than two nodes were created - false if
              already initialized, no wheels in this track, track route failed to find
              node configuration
      */
      virtual agx::Bool initialize( TrackNodeOnInitializeCallback* onInitializeCallback = nullptr );

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Re-initialize this track given number of nodes, node width and thickness, and
      initial distance tension. The old nodes will be removed but the wheels are kept.
      \param numberOfNodes - number of nodes in the track
      \param width - width of the track (this is an implicit definition, the geometry may be any size)
      \param thickness - thickness of the track (this is an implicit definition, the geometry may be any size)
      \param initialDistanceTension - value (distance) of how much shorter each node should be which causes tension in the
                                      system of tracks and wheels. Ideal case
                                        track_tension = initialDistanceTension * track_constraint_compliance.
                                      Since contacts and other factors are included it's not possible to know
                                      the exact tension after the system has been created.
      \return true if successfully re-initialized, otherwise false
      */
      virtual agx::Bool reinitialize( agx::UInt numberOfNodes,
                                      agx::Real width,
                                      agx::Real thickness,
                                      agx::Real initialDistanceTension = agx::Real( 0 ) );
      DOXYGEN_END_INTERNAL_BLOCK()

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxVehicle::Track );

      virtual void handleGeometryContact( agxCollide::GeometryContact* gc );

    protected:
      enum StateEnum : agx::UInt32
      {
        INITIALIZED = 1 << 0, // After add notification and route successfully initialized.
      };
      using State = agx::BitState<StateEnum, agx::UInt32>;

    protected:
      Track();
      virtual ~Track();

      virtual void onAddNotification( agxSDK::Simulation* simulation ) override;
      virtual void onRemoveNotification( agxSDK::Simulation* simulation ) override;
      virtual void onPreCollideStep() override;
      virtual void onPreStep() override;
      virtual void onPostStep() override;

    private:
      struct ContactData
      {
        using GeometryContactContainer = agx::Vector<agx::Physics::GeometryContactPtr>;
        using NodeContacts             = agx::HashVector<agx::UInt, GeometryContactContainer>;
        using WheelNodeContacts        = agx::HashTable<const TrackWheel*, NodeContacts>;

        WheelNodeContacts wheelNodeContacts;
        NodeContacts nodeContacts;
      };

      struct TrackNodeTriplet
      {
        agx::Bool isValid() const { return prev != nullptr && node != nullptr && next != nullptr; }

        TrackNode* prev;
        TrackNode* node;
        TrackNode* next;

        agx::UInt prevIndex;
        agx::UInt nodeIndex;
        agx::UInt nextIndex;
      };

    private:
      TrackNodeTriplet findTrackNodeTriplet( agx::UInt nodeIndex ) const;
      TrackNodeTriplet findTrackNodeTriplet( const agx::RigidBody* nodeRb ) const;

      void onPreStepMergeNodes( TrackWheel& wheel, const ContactData::NodeContacts* contacts, agx::UIntVector& mergedNodeIndices );
      void onPreStepHandleNonMergedNodes( TrackWheel& wheel, agx::UIntVector& mergedNodeIndices );
      void onPreStepSplitWheelMergedNodes( TrackWheel& wheel, agx::UIntVector& mergedNodeIndices );
      void onPreInternalMergeNodes( const TrackInternalMergeProperties& properties );

      void addAndConstrain( TrackNodeRef node, TrackNodeRef prev );
      void constrain( TrackNode& curr, TrackNode& prev );
      void merge( TrackWheel& wheel, TrackNode& node );
      void merge( TrackNode& node1, TrackNode& node2, agx::MergedBody& mergedBody );
      void split( TrackNode& node );
      void transformToWheel( TrackNode& node, const TrackWheel& wheel ) const;

    private:
      TrackWheelContainer m_wheels;
      RigidBodyTrackWheelTable m_rbWheelTable;
      TrackRouteRef m_route;
      ContactData m_contactData;
      State m_state;
      TrackInternalMergeProperties m_internalMergeProperties;
      TrackPropertiesRef m_properties;
  };

  inline Track* Track::get( agx::RigidBody* rb )
  {
    return LinkedStructure::getT<Track>( rb );
  }

  inline const Track* Track::get( const agx::RigidBody* rb )
  {
    return LinkedStructure::getT<Track>( rb );
  }

  inline Track::TrackNodeTriplet Track::findTrackNodeTriplet( agx::UInt nodeIndex ) const
  {
    if ( nodeIndex == agx::InvalidIndex )
      return { nullptr, nullptr, nullptr, agx::InvalidIndex, agx::InvalidIndex, agx::InvalidIndex };

    Iterator nodeIt = getIterator( nodeIndex );
    agxAssert( nodeIt.index() < getNumNodes() );
    Iterator prevIt = nodeIt - 1ul;
    Iterator nextIt = nodeIt + 1ul;
    return
    {
      *prevIt, *nodeIt, *nextIt,
      prevIt.index(), nodeIt.index(), nextIt.index()
    };
  }

  inline Track::TrackNodeTriplet Track::findTrackNodeTriplet( const agx::RigidBody* nodeRb ) const
  {
    return findTrackNodeTriplet( getNodeIndex( nodeRb ) );
  }

  inline TrackWheel* Track::getWheel( const agx::RigidBody* rb ) const
  {
    return m_rbWheelTable.get_or_return_default( rb );
  }

  inline agx::UInt Track::getNumNodes() const
  {
    return getNumSegments();
  }

  inline TrackNode* Track::getNode( agx::UInt index ) const
  {
    return getSegment( index )->as<TrackNode>();
  }

  template<>
  inline TrackNode* Track::getNode( const agx::RigidBody* obj ) const
  {
    auto node = LinkedStructure::getSegment( obj );
    return node != nullptr ? node->as<TrackNode>() : nullptr;
  }

  inline agx::UInt Track::getNodeIndex( const agx::RigidBody* rb ) const
  {
    return LinkedStructure::getSegmentIndex( rb );
  }

  inline Track::Iterator Track::getIterator( agx::UInt index ) const
  {
    return LinkedStructure::getSegmentIterator<TrackNode>( index, true );
  }

  template<>
  inline Track::Iterator Track::getIterator( const agx::RigidBody* obj ) const
  {
    return LinkedStructure::getSegmentIterator<TrackNode>( obj, true );
  }

  template<>
  inline Track::Iterator Track::getIterator( const TrackNode* obj ) const
  {
    if ( obj == nullptr )
      return Iterator::end( getSegmentsContainer() );
    return getIterator( obj->getRigidBody() );
  }
}
