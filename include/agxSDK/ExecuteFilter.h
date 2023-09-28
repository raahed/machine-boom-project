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

#include <agx/Referenced.h>
#include <agxCollide/GeometryPair.h>
#include <agxCollide/Contacts.h>


namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES(ExecuteFilter);

  /**
  Abstract base class that implements a filter that selects which events should trigger a Listener.
  */
  class AGXPHYSICS_EXPORT ExecuteFilter : public agx::Referenced
  {
    public:
      /**
      Called when narrow phase tests have determined overlap (when we have detailed intersection between the
      actual geometries).
      \param gc - geometry contact of the overlapping geometries
      \return true when this filter has determined that the GeometryContact should be used for
              triggering either an impact or a contact event
      */
      virtual bool match(const agxCollide::GeometryContact& gc) const;

      /**
      Called once when narrow phase no longer finds overlaps between the two geometries.
      \param geometryPair - the two geometries that no longer overlaps
      \return true when this filter has determined that the geometry pair should be used
              for triggering a separation event
      */
      virtual bool match(const agxCollide::GeometryPair& geometryPair) const;

      /**
      Called by the other two match functions if they are not overridden.
      \return true if either impact, contact or separation events should see these matching geometries
      */
      virtual bool match(const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2) const;

    protected:
      /**
      Default, hidden constructor.
      */
      ExecuteFilter();

      /**
      Destructor
      */
      virtual ~ExecuteFilter();
  };



  /// For templated version of ExecuteFilter. Has to be specialized for each class.
  template <typename T>
  inline bool matchFilter(const agxCollide::Geometry*, const T*);

  /**
  Templated version of ExecuteFilter, for objects which contain geometries and inherit
  from agx::Referenced, such as agxCollide::Geometry, agx::RigidBody, or agxSDK::Assembly.
  */
  template <typename T1, typename T2>
  class ExecuteFilterT : public ExecuteFilter
  {
    public:
      /**
      Create an ExecuteFilter with one or two conditions.
      null-pointers mean "match everything".
      */
      ExecuteFilterT( const T1* t1, const T2* t2 = nullptr );

      /// Inherited from agxSDK::ExecuteFilter.
      virtual bool match( const agxCollide::Geometry*, const agxCollide::Geometry* ) const override;

      using ExecuteFilter::match;

    protected:
      virtual ~ExecuteFilterT();


    protected:
      agx::ref_ptr<const T1> m_t1;
      agx::ref_ptr<const T2> m_t2;
  };


  /// Implementation


  AGX_FORCE_INLINE bool ExecuteFilter::match(const agxCollide::GeometryContact& gc) const
  {
    return match(gc.geometry(0), gc.geometry(1));
  }


  AGX_FORCE_INLINE bool ExecuteFilter::match(const agxCollide::GeometryPair& geometryPair) const
  {
    return match(geometryPair.first, geometryPair.second);
  }

  AGX_FORCE_INLINE bool ExecuteFilter::match(const agxCollide::Geometry*, const agxCollide::Geometry*) const
  {
    return false;
  }


  template <typename T1, typename T2>
  ExecuteFilterT<T1, T2>::ExecuteFilterT( const T1* t1, const T2* t2 )
    : m_t1( t1 ), m_t2( t2 )
  {
  }


  template <typename T1, typename T2>
  bool ExecuteFilterT<T1, T2>::match(const agxCollide::Geometry* geo1, const agxCollide::Geometry* geo2) const
  {
    const bool f =
      (matchFilter(geo1, m_t1.get()) && matchFilter(geo2, m_t2.get()))  ||    // Both match
      (matchFilter(geo2, m_t1.get()) && matchFilter(geo1, m_t2.get()));       // Both match

    return f;
  }


  template<typename T1, typename T2>
  ExecuteFilterT<T1, T2>::~ExecuteFilterT()
  {
  }


} // namespace agxSDK

