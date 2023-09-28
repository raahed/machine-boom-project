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

#ifndef AGXSDK_COLLISIONGROUPFILTER_H
#define AGXSDK_COLLISIONGROUPFILTER_H

#include <agx/agxPhysics_export.h>
#include <agxSDK/agxSDK.h>
#include <agxCollide/Geometry.h>
#include <agxSDK/ExecuteFilter.h>
#include <agx/TimeStamp.h>
#include <agxCollide/Contacts.h>

#include <agx/Vector.h>


namespace agxSDK
{

  AGX_DECLARE_POINTER_TYPES(CollisionGroupFilter);
  AGX_DECLARE_VECTOR_TYPES(CollisionGroupFilter);

  /**
  An ExecuteFilter that is used to filter out contacts matching a specified criteria based on
  Collision groups.
  Use this class to specify callbacks for certain combinations of geometry collision events.
  */
  class AGXPHYSICS_EXPORT CollisionGroupFilter : public agxSDK::ExecuteFilter
  {
    public:

      /// Default constructor
      CollisionGroupFilter();

      /**
      This filter will select contacts that contain Geometry \p geometry
      0 can be considered a wild card and matches all geometries. It will then
      match the collision groups with the collider. If they belong to the same collision
      group, the filter will select the contact.
      */
      CollisionGroupFilter(agxCollide::Geometry* geometry);

      /**
      Sets the geometry of the filter
      */
      void setGeometry(agxCollide::Geometry* geometry);

      /**
      Gets the geometry of the filter
      */
      agxCollide::Geometry* getGeometry();
      const agxCollide::Geometry* getGeometry() const;

      /**
       Adds a collision group to the filter. This will only register collisions with geometries
       that has the specified groups in it
       */
      void addGroup(const agx::Name& group);
      void addGroup(agx::UInt32 group);

      /**
        Removes the specified group \p form the filter
      */
      void removeGroup(const agx::Name& group);
      void removeGroup(agx::UInt32 group);

      /**
        Removes all the groups the filter listens too
      */
      void removeAllGroups();

      /**
        Returns true if the filter contains and listens to \p group
      */
      bool hasGroup(const agx::Name& group);
      bool hasGroup(agx::UInt32 group);

      /**
        Returns the named groups that the filter listens to
      */
      agx::Vector<agx::Name> getNamedGroups();
      const agx::Vector<agx::Name> getNamedGroups() const;

      /**
        Returns the uint groups that the filter listens to
      */
      agx::Vector<agx::UInt32> getUIntGroups();
      const agx::Vector<agx::UInt32> getUIntGroups() const;

      bool match(const agxCollide::Geometry* geo0, const agxCollide::Geometry* geo1) const override;
      using ExecuteFilter::match;

  protected:

      // Check if a geometry has one of the collision groups of this filter
      bool hasMemberGroup( const agxCollide::Geometry* geom ) const;

      /// Destructor
      virtual ~CollisionGroupFilter();


    protected:
      agx::Vector<agx::Name> m_namedCollisionGroups;
      agx::Vector<agx::UInt32> m_uintCollisiontGroups;
      agxCollide::Geometry* m_geometry;
  };

  // Implementation

  // Had to switch to regular inline since compiler complains about AGX_FORCE_INLINE
  inline bool CollisionGroupFilter::match(const agxCollide::Geometry* geo0, const agxCollide::Geometry* geo1) const
  {

    if (m_geometry == nullptr || (geo0 == m_geometry || geo1 == m_geometry))
    {
      // Match all collisions if no collision groups are set
      if (m_uintCollisiontGroups.size() <= 0 && m_namedCollisionGroups.size() <= 0)
        return true;

      if (m_geometry == nullptr)
      {
        if (hasMemberGroup( geo0 )) return true;
        if (hasMemberGroup( geo1 )) return true;
      }
      else // geo0 == m_geometry || geo1 == m_geometry
      {
        const agxCollide::Geometry* collider = (m_geometry == geo0) ? geo1 : geo0;
        if (hasMemberGroup( collider )) return true;
      }
    }
    return false;
  }

  inline bool CollisionGroupFilter::hasMemberGroup(const agxCollide::Geometry* geom) const
  {
    // Named collision groups
    for (size_t i = 0; i < m_namedCollisionGroups.size(); i++)
    {
      if (geom->hasGroup( m_namedCollisionGroups[i] ))
        return true;
    }

    // UInt collision groups
    for (size_t i = 0; i < m_uintCollisiontGroups.size(); i++)
    {
      if (geom->hasGroup( m_uintCollisiontGroups[i] ))
        return true;
    }
    return false;
  }

  AGX_FORCE_INLINE agxCollide::Geometry *CollisionGroupFilter::getGeometry() { return m_geometry; }
  AGX_FORCE_INLINE const agxCollide::Geometry *CollisionGroupFilter::getGeometry() const { return m_geometry; }

  inline agx::Vector<agx::Name> CollisionGroupFilter::getNamedGroups() {return m_namedCollisionGroups; }
  inline const agx::Vector<agx::Name> CollisionGroupFilter::getNamedGroups() const {return m_namedCollisionGroups;}

  inline agx::Vector<agx::UInt32> CollisionGroupFilter::getUIntGroups() {return m_uintCollisiontGroups; }
  inline const agx::Vector<agx::UInt32> CollisionGroupFilter::getUIntGroups() const {return m_uintCollisiontGroups;}
}

#endif /*AGXMODEL_COLLISIONGROUPFILTER_H*/
