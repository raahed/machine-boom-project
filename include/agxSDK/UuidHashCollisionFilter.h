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

#include <agxSDK/ExecuteFilter.h>

namespace agxSDK
{
  /**
  Contact event filter matching UUID of instances, such as agxCollide::Geometry,
  agx::RigidBody, agxWire::Wire, agxCable::Cable, agxVehicle::Track and/or
  agxTerrain::Terrain. This filter is especially convenient when you have to
  map AGX Dynamics instances with other representations, e.g.,
      
      auto uuid = agxSDK::UuidHashCollisionFilter::findUuid( myRep->getAgxInstance() );
      assert( uuid > 0u );
      uuidRepTable[ uuid ] = uuid;
      uuidFilter->add( uuid );

      ...

      void handleContact( agxCollide::GeometryContact* gc )
      {
        auto uuid1 = agxSDK::UuidHashCollisionFilter::findCorrespondingUuid( gc->geometry( 0 ) );
        auto uuid2 = agxSDK::UuidHashCollisionFilter::findCorrespondingUuid( gc->geometry( 1 ) );
        // Check if uuid1 and/or uuid2 is in our uuidRepTable.
      }

  A corresponding UUID is the UUID the given geometry is representing. E.g., if
  you have a wire and do filter->add( wire->getUuid().hash() ), then you'll find
  that:
      UuidHashCollisionFilter::findCorrespondingUuid( geometryContact->geometry(1) ) == wire->getUuid().hash()
  with an exception for agx::RigidBody which is tightly coupled to its geometries.
  If an UUID of a rigid body is added to this filter, the filter will match but
  UuidHashCollisionFilter::findCorrespondingUuid( geometryContact->geometry(0) ) will
  return the UUID of the given geometry.

  This filter is matching a pair of corresponding UUIDs given the following modes:
      MATCH_ALL: Any pair of geometries is a match (listen to all contacts).
      MATCH_OR:  At least one of the corresponding UUID must match our UUID set.
      MATCH_AND: Both corresponding UUIDs must match our UUID set.
  */
  class AGXPHYSICS_EXPORT UuidHashCollisionFilter : public ExecuteFilter
  {
    public:
      enum Mode
      {
        MATCH_ALL, /**< Any object results in a match. Default. */
        MATCH_OR,  /**< At least one of the two objects must match our UUIDs. */
        MATCH_AND  /**< Both objects must match our UUID set. */
      };

    public:
      /**
      Finds UUID hash of the given \p instance. This utility method is mainly
      convenient in other languages where, e.g., getUuid of an object isn't
      exported and this method contains some logic matching the filtering that's
      made when this filter is executed. E.g., the UUID of an agxTerrain::Terrain
      instance is of its geometry.

      If it's not possible to determine the UUID, 0 is returned.
      */
      static agx::UInt32 findUuid( const agx::Referenced* instance );

      /**
      Searches for corresponding object's (e.g., Wire, Cable, Track etc.) UUID
      given geometry.
      */
      static agx::UInt32 findCorrespondingUuid( const agxCollide::Geometry* geometry );

    public:
      /**
      Default constructor, default mode is MATCH_ALL.
      */
      UuidHashCollisionFilter();

      /**
      Set mode how the filtering is performed.
      \param mode - new mode
      */
      void setMode( Mode mode );

      /**
      \return the current filtering mode
      */
      Mode getMode() const;

      /**
      Add UUID hash to the matching set.
      \param hash - UUID hash to add to the matching set
      */
      void add( agx::UInt32 hash );

      /**
      Remove \p hash from the matching set.
      \param hash - UUID hash to remove from the matching set
      */
      void remove( agx::UInt32 hash );

      /**
      \param hash - UUID hash to check
      \return true if this filter is matching given UUID \p hash, otherwise false
      */
      agx::Bool contains( agx::UInt32 hash ) const;

    public:
      virtual bool match( const agxCollide::Geometry* geometry1,
                          const agxCollide::Geometry* geometry2 ) const override;
      
      using ExecuteFilter::match;

    protected:
      virtual ~UuidHashCollisionFilter();

    private:
      agx::Bool isMatch( const agxCollide::Geometry* geometry ) const;

    private:
      using UInt32Set = agx::HashSet<agx::UInt32>;

    private:
      Mode m_mode;
      UInt32Set m_uuids;
  };

  inline bool UuidHashCollisionFilter::match( const agxCollide::Geometry* geometry1,
                                              const agxCollide::Geometry* geometry2 ) const
  {
    return m_mode == MATCH_ALL ||
           ( m_mode == MATCH_OR  && ( isMatch( geometry1 ) || isMatch( geometry2 ) ) ) ||
           ( m_mode == MATCH_AND && ( isMatch( geometry1 ) && isMatch( geometry2 ) ) );
  }

  inline agx::Bool UuidHashCollisionFilter::isMatch( const agxCollide::Geometry* geometry ) const
  {
    const auto rbUuid = geometry->getRigidBody() != nullptr ?
                          geometry->getRigidBody()->getUuid().hash() :
                          0u;
    return contains( findCorrespondingUuid( geometry ) ) ||
           contains( rbUuid );
  }
}
