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

#include <agxSDK/Assembly.h>
#include <agxSDK/LinkedSegmentRange.h>
#include <agxSDK/LinkedStructureComponent.h>
#include <agxSDK/LinkedStructureObjectData.h>

#include <agx/BitState.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES( LinkedStructure );

  /**
  Basic implementation of any structure where bodies are linked/constrained
  together in a well defined consecutive order. E.g., cables, beams,
  tracks etc.
  */
  class AGXPHYSICS_EXPORT LinkedStructure : public agxSDK::Assembly
  {
    public:
      /**
      \return The LinkedStructure the given geometry is part of, or nullptr if there is no such LinkedStructure.
       */
      static LinkedStructure* get(agxCollide::Geometry* geometry);

      /**
      \return The LinkedStructure the given geometry is part of, or nullptr if there is no such LinkedStructure.
       */
      static const LinkedStructure* get(const agxCollide::Geometry* geometry);

      /**
      \return The LinkedStructure the given body is part of, or nullptr if there is no such LinkedStructure.
       */
      static LinkedStructure* get(agx::RigidBody* body);

      /**
      \return The LinkedStructure the given body is part of, or nullptr if there is no such LinkedStructure.
       */
      static const LinkedStructure* get(const agx::RigidBody* body);

    public:
      /**
      Assign material to all segments in this linked structure.
      \param material - material for the segments
      */
      virtual void setMaterial( agx::Material* material );

      /**
      \return the material used for by the segments in this linked structure
      */
      virtual agx::Material* getMaterial() const;

      /**
      \return the unique id of this linked structure
      */
      virtual agx::UInt32 getUniqueId() const;

      /**
      \return the collision id of this linked structure
      */
      virtual agx::UInt32 getDisabledCollisionGroupId() const;

      /**
      Enable/disable collisions between the rigid body and all segments of this linked structure.
      \param rb - rigid body to enable/disable collisions with
      \param enable - true to enable, false to disable
      */
      virtual void setEnableCollisions( agx::RigidBody* rb, agx::Bool enable );

      /**
      \return true if collisions are enabled - otherwise false
      */
      virtual agx::Bool getEnableCollisions( const agx::RigidBody* rb ) const;

      /**
      Enable/disable collisions between the geometry and all segments of this linked structure.
      \param geometry - geometry to enable/disable collisions with
      \param enable - true to enable, false to disable
      */
      virtual void setEnableCollisions( agxCollide::Geometry* geometry, agx::Bool enable );

      /**
      \return true if collisions are enabled - otherwise false
      */
      virtual agx::Bool getEnableCollisions( const agxCollide::Geometry* geometry ) const;

      /**
      Add group id to all segments in this linked structure.
      \param id - id to add
      */
      virtual void addGroup( agx::UInt32 id );

      /**
      Remove group id from all segments in this linked structure.
      \param id - id to remove
      */
      virtual void removeGroup( agx::UInt32 id );

      /**
      \return true if the id has been added to this linked structure - otherwise false
      */
      virtual agx::Bool hasGroup( agx::UInt32 id ) const;

      /**
      Add group name to all segments in this linked structure.
      \param name - name to add
      */
      virtual void addGroup( const agx::Name& name );

      /**
      Remove group name from all segments in this linked structure.
      \param name - name to remove
      */
      virtual void removeGroup( const agx::Name& name );

      /**
      \return true if the name has been added to this linked structure - otherwise false
      */
      virtual agx::Bool hasGroup( const agx::Name& name ) const;

      /**
      \return collection of group names and id's that has been added to this linked structure
      */
      virtual agxCollide::GroupIdCollection findGroupIdCollection() const;

      /**
      Assigns linear velocity damping to all segments in this linked structure.
      \note This value will be assigned to all segments in this linked structure.
            I.e., new segments will be assigned this value but it doesn't prevent
            having different damping values along this linked structure.
      \param linearVelocityDamping - linear velocity damping (default: (0, 0, 0))
      */
      virtual void setLinearVelocityDamping( const agx::Vec3& linearVelocityDamping );

      /**
      \return the given linear velocity damping previously assigned to this linked structure (default: (0, 0, 0))
      */
      virtual agx::Vec3 getLinearVelocityDamping() const;

      /**
      Assigns angular velocity damping to all segments in this linked structure.
      \note This value will be assigned to all segments in this linked structure.
            I.e., new segments will be assigned this value but it doesn't prevent
            having different damping values along this linked structure.
      \param angularVelocityDamping - angular velocity damping (default: (0, 0, 0))
      */
      virtual void setAngularVelocityDamping( const agx::Vec3& angularVelocityDamping );

      /**
      \return the given angular velocity damping previously assigned to this linked structure (default: (0, 0, 0))
      */
      virtual agx::Vec3 getAngularVelocityDamping() const;

      /**
      Calculates and returns the total length of this linked structure, at rest.
      \return rest length of this linked structure
      */
      virtual agx::Real getRestLength() const;

      /**
      Calculates and returns the total length of this linked structure, at current state.
      \return current length of this linked structure
      */
      virtual agx::Real getCurrentLength() const;

      /**
      Calculates and returns the total mass of this linked structure. I.e.,
      the total sum of all segments masses.
      \return total mass of this linked structure
      */
      virtual agx::Real getMass() const;

      /**
      Calculates and returns the total volume of this linked structure, at rest.
      \return total volume of this linked structure
      */
      virtual agx::Real getRestVolume() const;

      /**
      Finds center point on this structure given rest length from the start. If this structure
      isn't initialized, agx::Vec3(0, 0, 0) is returned.
      \param restLengthFromStart - rest length from the start
      \return center point on this structure, \p restLengthFromStart
      */
      virtual agx::Vec3 findPoint( agx::Real restLengthFromStart ) const;

      /**
      Given any world point, find a center point on this structure that is closest
      to the given \p worldPoint.
      \param worldPoint - any point in world
      \return closest center point on this structure to \p worldPoint
      */
      virtual agx::Vec3 findPoint( const agx::Vec3& worldPoint ) const;

      /**
      Calculates the current length of a given rest length. E.g., if this structure is
      stretched, the current length will be larger than the rest length and corresponds
      to the actual distance from the start.
      \p restLengthFromStart - rest length to convert to current length
      \return the corresponding current length
      */
      virtual agx::Real findCurrentLength( agx::Real restLengthFromStart ) const;

      /**
      Calculates the rest length of a given current length from start.
      \param currentLengthFromStart - current length to convert to rest length
      \return the corresponding rest length
      */
      virtual agx::Real findRestLength( agx::Real currentLengthFromStart ) const;

      /**
      Finds rest length to the start of the given segment.
      \param segment - segment in this structure
      \return rest length to the start of the given segment, total rest length if not part of this structure
      */
      virtual agx::Real findRestLengthTo( const agxSDK::LinkedSegment* segment ) const;

      /**
      Make the current relative positions of all nodes the rest state. That is,
      when this call returns no constraint in this linked structure will be violated.
      \return true if successful - false if not initialized
      */
      virtual agx::Bool rebind();

      /**
      Add new component to this linked structure.
      \note This component must be unique to this object, i.e., it cannot be added to
            several instances of LinkedStructure.
      \param component - component to add
      \return true if the component is added, false if null or already added to this or some other linked structure
      */
      virtual agx::Bool addComponent( agxSDK::LinkedStructureComponent* component );

      /**
      Remove an added component from this linked structure.
      \param component - component to remove
      \return true if removed, otherwise false (null or not present in this linked structure)
      */
      virtual agx::Bool removeComponent( agxSDK::LinkedStructureComponent* component );

      /**
      \return first component of given type
      */
      template<typename T>
      T* getComponent() const;

      /**
      \return first component of given name
      */
      agxSDK::LinkedStructureComponent* getComponent( const agx::Name& name ) const;


      /**
       * \return A segment range over all segments in this LinkedStructure.
       */
      agxSDK::LinkedSegmentRange<agxSDK::LinkedSegment> segments();

      /**
      \return segment index if rigid body is a segment in this linked structure - otherwise agx::InvalidIndex
      */
      agx::UInt getSegmentIndex( const agx::RigidBody* rb ) const;

      /**
      \return segments container
      */
      const agxSDK::LinkedSegmentContainer& getSegmentsContainer() const;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::LinkedStructure );

      /**
      Called when this linked structure is added to the simulation. This
      callback is reserved for LinkedStructure but onAddNotification will
      be called in a similar way.
      */
      void addNotification( agxSDK::Simulation* simulation ) final override;

      /**
      Called when this linked structure is added to the simulation. This
      callback is reserved for LinkedStructure but onRemoveNotification will
      be called in a similar way.
      */
      void removeNotification( agxSDK::Simulation* simulation ) final override;

      using Assembly::addNotification;
      using Assembly::removeNotification;

    protected:
      /**
      Group id 'union' of int id and name id. Either id == agx::InvalidIndex or
      name == "" representing group of either name or id. If data == IdData()
      the group is empty and shouldn't be handled.
      */
      struct IdData
      {
        /**
        Default constructor, group id is invalid with id == agx::InvalidIndex and name == "".
        */
        IdData() : id( agx::InvalidIndex ), name( "" ) {}

        /**
        Construct group given int id where name == "".
        */
        IdData( agx::UInt32 id ) : id( id ), name( "" ) {}

        /**
        Construct group given name where id == agx::InvalidIndex.
        */
        IdData( const agx::Name& name ) : id( agx::InvalidIndex ), name( name ) {}

        /**
        \return true if other id and name equals id and name of this
        */
        agx::Bool operator== ( const IdData& other ) const { return id == other.id && name == other.name; }

        /**
        \return true if other id and name are not equal to id and name of this
        */
        agx::Bool operator!= ( const IdData& other ) const { return !(*this == other); }

        agx::UInt32 id;
        agx::Name name;
      };

    protected:
      /**
      Called when addNotification is executed when this linked
      structure is added to the simulation.
      */
      virtual void onAddNotification( agxSDK::Simulation* simulation );

      /**
      Called when removeNotification is executed when this linked
      structure is removed from the simulation.
      */
      virtual void onRemoveNotification( agxSDK::Simulation* simulation );

      /**
      Called when a new material has been assigned.
      */
      virtual void onMaterialNew();

      /**
      Called when a new group has been added.
      */
      virtual void onGroupAdd( const IdData& group );

      /**
      Called when an existing group has been removed.
      */
      virtual void onGroupRemove( const IdData& group );

      /**
      Called when collision state changed for given geometry.
      */
      virtual void onSetEnableCollisions( agxCollide::Geometry* geometry, agx::Bool enable );

      /**
      Called on simulation pre-collide step event - after user preCollide.
      */
      virtual void onPreCollideStep();

      /**
      Called on simulation pre step event - after user preStep.
      */
      virtual void onPreStep();

      /**
      Called on simulation post step event - after user postStep.
      */
      virtual void onPostStep();

      /**
      Called on simulation last step event - after user lastStep.
      */
      virtual void onLastStep();

    protected:
      /**
      \return this as T if rb is a segment in this linked structure - otherwise null
      */
      template<typename T>
      static T* getT( agx::RigidBody* rb );

      /**
      \return this as T if rb is a segment in this linked structure - otherwise null
      */
      template<typename T>
      static const T* getT( const agx::RigidBody* rb );

    protected:
      /**
      Hidden default constructor.
      */
      LinkedStructure();

      /**
      Reference counted object - protected destructor.
      */
      virtual ~LinkedStructure();

      /**
      Add segment to this linked structure. The segment will inherit our material,
      unique id and all externally added groups.
      \note Assumed \p segment to be a valid pointer and not already added.
      \param segment segment to add
      */
      void add( LinkedSegment* segment );

      /**
      \return the number of segments in this linked structure
      */
      agx::UInt getNumSegments() const;

      /**
      Constructs a segment range over all segments.
      \return segment range over all segments
      */
      template<typename T>
      LinkedSegmentRange<T> getSegments() const;

      /**
      Range given begin and end segment index. If end index is before begin,
      begin will become boundaryless. E.g., size = 10, begin = 8, end = 3,
      the resulting range will be 8, 9, 0, 1, 2 (3 is end).
      \return segment range given range
      */
      template<typename T>
      LinkedSegmentRange<T> getSegments( agx::UInt beginIndex, agx::UInt endIndex ) const;

      /**
      \return segment given index
      */
      LinkedSegment* getSegment( agx::UInt index ) const;

      /**
      \return segment given rest length from start, nullptr if no segments has been added
      */
      template<typename T>
      T* getSegment( agx::Real restLengthFromStart ) const;

      /**
      \return segment given rigid body
      */
      template<typename ObjT>
      LinkedSegment* getSegment( const ObjT* rb ) const;

      /**
      \return segment iterator given index and iterator state
      */
      template<typename T>
      LinkedSegmentIterator<T> getSegmentIterator( agx::UInt index, agx::Bool boundaryless = false ) const;

      /**
      \return segment iterator given rigid body and iterator state
      */
      template<typename T, typename ObjT>
      LinkedSegmentIterator<T> getSegmentIterator( const ObjT* rb, agx::Bool boundaryless = false ) const;

      /**
      \return internal data for rigid body if it exists and is the body is mapped to this linked structure
      */
      LinkedStructureObjectData* getData( const agx::RigidBody* rb ) const;

    private:
      // Events from the simulation.
      friend class Simulation;

      enum SimulationEvent
      {
        PRE_COLLIDE_STEP,
        PRE_STEP,
        POST_STEP,
        LAST_STEP
      };

      using IdDataContainer = agx::Vector<IdData>;

    private:
      void trigger( SimulationEvent simulationEvent );
      void addGroup( const IdData& group );
      void removeGroup( const IdData& group );
      agx::Bool hasGroup( const IdData& group ) const;

    private:
      agx::Uuid m_uniqueId;
      agx::Uuid m_disabledCollisionsId;
      LinkedSegmentContainer m_segments;
      agx::MaterialRef m_material;
      IdDataContainer m_externallyAddedGroups;
      LinkedStructureComponentContainer m_components;
      agx::Vec3 m_linearVelocityDamping;
      agx::Vec3 m_angularVelocityDamping;
  };

  template<typename T>
  T* LinkedStructure::getT( agx::RigidBody* rb )
  {
    auto data = LinkedStructureObjectData::get<LinkedStructureObjectData>( rb );
    return data != nullptr && data->linkedStructure != nullptr ? data->linkedStructure->template asSafe<T>() : nullptr;
  }

  template<typename T>
  const T* LinkedStructure::getT( const agx::RigidBody* rb )
  {
    return LinkedStructure::getT<T>( const_cast<agx::RigidBody*>( rb ) );
  }

  template<typename T>
  T* LinkedStructure::getComponent() const
  {
    for ( auto component : m_components ) {
      auto typed = component->template asSafe<T>();
      if ( typed != nullptr )
        return typed;
    }

    return nullptr;
  }

  inline const LinkedSegmentContainer& LinkedStructure::getSegmentsContainer() const
  {
    return m_segments;
  }

  inline agx::UInt LinkedStructure::getNumSegments() const
  {
    return m_segments.size();
  }

  template<typename T>
  LinkedSegmentRange<T> LinkedStructure::getSegments() const
  {
    return LinkedSegmentRange<T>( LinkedSegmentIterator<T>::begin( m_segments ),
                                  LinkedSegmentIterator<T>::end( m_segments ) );
  }

  template<typename T>
  LinkedSegmentRange<T> LinkedStructure::getSegments( agx::UInt beginIndex, agx::UInt endIndex ) const
  {
    return LinkedSegmentRange<T>( getSegmentIterator<T>( beginIndex ),
                                  getSegmentIterator<T>( endIndex ) );
  }

  inline LinkedSegment* LinkedStructure::getSegment( agx::UInt index ) const
  {
    agxAssert( index < m_segments.size() );
    return m_segments[ index ];
  }

  template<typename T>
  T* LinkedStructure::getSegment( agx::Real restLengthFromStart ) const
  {
    if ( m_segments.empty() )
      return nullptr;

    if ( restLengthFromStart <= 0.0 )
      return m_segments.front()->template as<T>();

    agx::Real restLengthAlongStructure = 0.0;
    for ( auto segment : this->template getSegments<T>() ) {
      if ( restLengthAlongStructure + segment->getLength() < restLengthFromStart )
        restLengthAlongStructure += segment->getLength();
      else
        return segment->template as<T>();
    }

    return m_segments.back()->template as<T>();
  }

  template<>
  inline LinkedSegment* LinkedStructure::getSegment( const agx::RigidBody* rb ) const
  {
    auto index = getSegmentIndex( rb );
    return index != agx::InvalidIndex ?
             m_segments[ index ] :
             nullptr;
  }

  inline agx::UInt LinkedStructure::getSegmentIndex( const agx::RigidBody* rb ) const
  {
    auto data = getData( rb );
    return data != nullptr && data->segment != nullptr && data->linkedStructure == this ?
             data->segment->getIndex() :
             agx::InvalidIndex;
  }

  template<typename T>
  LinkedSegmentIterator<T> LinkedStructure::getSegmentIterator( agx::UInt index, agx::Bool boundaryless /* = false */ ) const
  {
    return LinkedSegmentIterator<T>( index, boundaryless, m_segments );
  }

  template<typename T, typename ObjT>
  LinkedSegmentIterator<T> LinkedStructure::getSegmentIterator( const ObjT* rb, agx::Bool boundaryless /* = false */ ) const
  {
    auto segment = this->getSegment( rb );
    if ( segment == nullptr )
      return LinkedSegmentIterator<T>::end( m_segments );
    return getSegmentIterator<T>( segment->getIndex(), boundaryless );
  }

  inline LinkedStructureObjectData* LinkedStructure::getData( const agx::RigidBody* rb ) const
  {
    auto data = LinkedStructureObjectData::get<LinkedStructureObjectData>( rb );
    // Accepting data when linkedStructure == null since it can be considered implicitly
    // related to this linked structure (data->linkedStructure = this to be made).
    return data != nullptr && ( data->linkedStructure == this || data->linkedStructure == nullptr ) ?
             data :
             nullptr;
  }
}
