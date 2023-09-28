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

#include <agxSDK/ContactEventListener.h>
#include <agxSDK/StepEventListener.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES( MultiCollisionFilterContactHandler );

  /**
  A step event listener managing N contact event listeners with different
  filters. The filter is used to access the geometry contacts. This handler
  doesn't support separation events.

  NOTE: This step event listener should have priority as low as possible because
        the contact data is cleared in 'pre' and 'post' and is therefore no longer
        accessible. I.e., add this listener with low priority and access the contact
        data in another pre and/or post with a higher priority.
  */
  class AGXPHYSICS_EXPORT MultiCollisionFilterContactHandler : public agxSDK::StepEventListener
  {
    public:
      /**
      Default constructor.
      */
      MultiCollisionFilterContactHandler();

      /**
      Add a contact filter with an activation mask, e.g.,
      agxSDK::ContactEventListener::IMPACT | agxSDK::ContactEventListener::CONTACT.
      \param filter - contact filter to add
      \param activationMask - activation mask of the resulting contact event listener
      */
      void add( ExecuteFilter* filter, int activationMask );

      /**
      Removes the given filter and its corresponding contact event listener.
      \param filter - filter to remove
      */
      virtual void remove( ExecuteFilter* filter );

      /**
      Assign new activation mask to the contact listener representing the given filter.
      \param filter - filter to assign new activation mask to
      \param activationMask - new activation mask
      */
      void setActivationMask( ExecuteFilter* filter, int activationMask );

      /**
      \param filter - filter to get activation mask for
      \return the activation mask of the given filter, 0 if the filter isn't part of this contact handler
      */
      int getActivationMask( const ExecuteFilter* filter ) const;

      /**
      \return the total number of geometry contacts the added filters has matched
      */
      agx::UInt32 getNumGeometryContacts() const;

      /**
      \return the total number of contact points the added filters has matched
      */
      agx::UInt32 getNumContactPoints() const;

      /**
      Collect geometry contact indices for the given filter. The geometry contact
      is accessible by calling getGeometryContact( contactIndex ).
      \param filter - filter to collect geometry contact indices for
      \param[out] contactIndices - resulting geometry contact indices
      */
      void collectContactIndices( const ExecuteFilter* filter, agx::UInt32Vector& contactIndices ) const;

      /**
      \param filter - filter to receive geometry contact indices for
      \return the geometry contact indices for the given filter
      */
      const agx::UInt32Vector& getContactIndices( const ExecuteFilter* filter ) const;

      /**
      \param filter - filter to receive geometry separations for
      \param[out] separations - geometry separations for the given filter
      */
      void collectSeparations( const ExecuteFilter* filter, agxCollide::GeometryPairVector& separations) const;

      /**
      \param index - index of the geometry contact
      \return the geometry contact for the given \p index, nullptr if \p index is out of bounds
      */
      agxCollide::GeometryContact* getGeometryContact( agx::UInt32 index ) const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      void addNotification() override;

      void removeNotification() override;

      void pre( const agx::TimeStamp& time ) override;

      void post( const agx::TimeStamp& time ) override;

      /**
      \return index of an already added geometry contact OR adds it to the contact set
      */
      agx::UInt32 getContactIndex( agxCollide::GeometryContact* geometryContact );
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Reference counted object - protected destructor.
      */
      virtual ~MultiCollisionFilterContactHandler();

    private:
      /**
      Resets all data, e.g., in pre, post and when removed.
      */
      void reset();

    private:
      agxCollide::GeometryContactPtrVector m_geometryContacts;
      agx::HashTable<agxCollide::GeometryContact*, agx::UInt32> m_geometryContactIndexTable;
      agx::HashVector<const ExecuteFilter*, ContactEventListenerRef> m_contactListeners;
      agx::UInt32 m_contactPointCounter = 0;
  };

  inline agxCollide::GeometryContact* MultiCollisionFilterContactHandler::getGeometryContact( agx::UInt32 index ) const
  {
    if ( index >= (agx::UInt32)m_geometryContacts.size() )
      return nullptr;
    return m_geometryContacts[ index ];
  }
}
