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

#include <agxSDK/agxSDK.h>

#include <agxCollide/Geometry.h>
#include <agxSDK/EventListener.h>

#include <agx/TimeStamp.h>
#include <agxCollide/Contacts.h>

#include <agxSDK/ExecuteFilter.h>

namespace agxSDK
{

  /**
  Derive from this class to implement a listener for Collision events.
  The following methods can be implemented to catch events of various types:

  - impact()    : When two geometries actually intersect. Here you can get GeometryContact with points, depths and normals
  - contact()   : When two geometries have already had an Impact and are now in continuous contact.
  - separation(): When two geometries are no more in contact.

  Which methods are executed for a ContactEventListener can be determined using setMask().
  IMPORTANT: By default impact() and separation() are called.
  If you want to deactivate the listener, just call setMask(0) and no events will be activated.
  */
  class AGXPHYSICS_EXPORT ContactEventListener : public EventListener
  {
    public:

      /// Defines the event states for which a listener will be activated, a mask can be bitwise OR:ed from these members.
      enum ActivationMask {
        IMPACT = 1 << 0,               /**< At First contact between two geometries */
        CONTACT = 1 << 1,              /**< After Impact, before separation, continuous contact. */
        SEPARATION = 1 << 2,           /**< At separation between two Geometries */
        POST = 1 << 3,                 /**< Before StepEventListener post are invoked. */
        DEFAULT = IMPACT | SEPARATION, /**< The default activation mask */
        ALL = IMPACT | CONTACT | SEPARATION | POST
      };

      /// Defines whether contact should be kept or removed after contact/impact methods returns.
      enum KeepContactPolicy {
        KEEP_CONTACT,               /**< Keep the contact */
        REMOVE_CONTACT,             /**< Remove the contact AFTER all collision handlers have been executed,
                                       BEFORE the solver get the contacts
                                    */
        REMOVE_CONTACT_IMMEDIATELY  /**< Remove the contact immediately AFTER THIS collision handler returns.
                               No contact handler executed after this will get this contact.*/
      };

      /// Specifies a bit mask which determines which event types that will activate this listener.
      virtual void setMask(int f);

      /**
        Default constructor, sets the activation mask by default  to Impact and Separation
        \param m - Execution mask, by default to IMPACT and SEPARATION
        \param filter - Execution filter, by default nullptr, means matching all contacts
      */
      ContactEventListener(int m = DEFAULT, ExecuteFilter* filter = nullptr );


      //=================== Override these methods:
      /**
      Called upon impact event if getFilter() contain IMPACT.
      Implement this method in the derived class to get callbacks.

      \param time - Last committed time in the simulation
      \param geometryContact - Pointer to the contact data for the two impacting geometries.
      \return Return KEEP_CONTACT if the contact data should be kept.
      */
      virtual KeepContactPolicy impact(const agx::TimeStamp& time, agxCollide::GeometryContact *geometryContact);

      /**
      Called upon contact event if getFilter() contain CONTACT.
      Implement this method in the derived class to get callbacks.

      \param time - Last committed time in the simulation
      \param geometryContact - Pointer to the contact data for the two contacting geometries.
      \return Return KEEP_CONTACT if the contact data should be kept.
      */
      virtual KeepContactPolicy contact(const agx::TimeStamp& time, agxCollide::GeometryContact* geometryContact);

      /**
      Called upon separation event if getFilter() contain SEPARATION.
      Implement this method in the derived class to get callbacks.

      \param time - Last committed time in the simulation
      \param geometryPair - Pointer to the contact data for the two separating Geometries.
      For a separation no contact points, depth nor normals are defined.
      */
      virtual void separation(const agx::TimeStamp& time, agxCollide::GeometryPair&  geometryPair);

      /**
      Called before agxSDK::StepEventListener::post now that the geometry contacts
      has contact forces from the solver. Implement this method in the derived class
      to get callbacks.

      \param time - Last committed time in the simulation
      \param geometryContact - Pointer to the contact data for the two interacting geometries with contact force data.
      */
      virtual void post(const agx::TimeStamp& time, agxCollide::GeometryContact* geometryContact);

      //=================== End Override

      /// Replaces the current filter with a new one
      void setFilter( ExecuteFilter* filter );

      ///\return the current execute filter
      ExecuteFilter* getFilter( );

      ///\return the current execute filter
      const ExecuteFilter* getFilter( ) const;

    protected:

      /// Destructor
      virtual ~ContactEventListener();

      ExecuteFilterRef m_executeFilter;
  };

  typedef agx::ref_ptr<ContactEventListener> ContactEventListenerRef;


  AGX_FORCE_INLINE ContactEventListener::KeepContactPolicy ContactEventListener::impact(const agx::TimeStamp&, agxCollide::GeometryContact*)
  {
    return ContactEventListener::KEEP_CONTACT;
  }

  AGX_FORCE_INLINE ContactEventListener::KeepContactPolicy ContactEventListener::contact(const agx::TimeStamp&, agxCollide::GeometryContact*)
  {
    return ContactEventListener::KEEP_CONTACT;
  }

  AGX_FORCE_INLINE void ContactEventListener::separation(const agx::TimeStamp&, agxCollide::GeometryPair&) {}

  AGX_FORCE_INLINE void ContactEventListener::post(const agx::TimeStamp&, agxCollide::GeometryContact*)
  {
  }

  AGX_FORCE_INLINE ExecuteFilter* ContactEventListener::getFilter( )
  {
    return m_executeFilter;
  }

  AGX_FORCE_INLINE const ExecuteFilter* ContactEventListener::getFilter( ) const
  {
    return m_executeFilter;
  }

}
