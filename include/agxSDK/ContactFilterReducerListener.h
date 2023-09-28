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

#include <agxSDK/MultiCollisionFilterContactHandler.h>

namespace agxSDK
{

  AGX_DECLARE_POINTER_TYPES(ContactFilterReducerListener);

  /**
  Class that execute the agxCollide::ContactReducer::reduce on contact points that are collected via a specific ExecuteFilter.
  Each filter will have its own set of contact points that will be reduced.
  The reduction will be performed in a PRE_STEP.
  */
  class AGXPHYSICS_EXPORT ContactFilterReducerListener : public agxSDK::MultiCollisionFilterContactHandler
  {
  public:
    ContactFilterReducerListener();

    /**
    Add a contact filter with an activation mask, e.g.,
    agxSDK::ContactEventListener::IMPACT | agxSDK::ContactEventListener::CONTACT.
    \param binsPerDimension - The number of bins used for contact reduction.
    \param filter - contact filter to add
    \param activationMask - activation mask of the resulting contact event listener
    */
    void add(size_t binsPerDimension, ExecuteFilter* filter, int activationMask);

    /**
    Removes the given filter and its corresponding contact event listener.
    \param filter - filter to remove
    */
    void remove(ExecuteFilter* filter) override;

    /**
    \param filter - filter for which we want to change the number of bins.
    \param binsPerDimension - New dimension parameter for the contact reduction algorithm
    \return True if the filter existed. Otherwise false.
    */
    bool setNumBinsPerDimension(ExecuteFilter* filter, size_t binsPerDimension);

    /**
    \param filter - The filter for which we want to recieve the num bins per dimension parameter.
    \return The dimension parameter for the contact reduction algorithm for the specific filter. Returns 0 if the filter does not exist.
    */
    size_t getNumBinsPerDimension(ExecuteFilter* filter) const;

    /**
    This method returns the number of reduced contact points last time step.
    \note: It is cleared in the::pre callback.This means that if you haven't activated the pre mask, the number of contact points will just increase.
    \return number of reduced contact points in the previous time step.
    */
    size_t getNumReducedContacts() const;

  protected:

    // These methods should not be public
    void add(ExecuteFilter*, int ) {}

    /**
    Reference counted object - protected destructor.
    */
    virtual ~ContactFilterReducerListener();

    void pre(const agx::TimeStamp& time) override;

    size_t reduceContacts();

  private:
    size_t m_numReducedContacts;

    typedef agx::HashVector<ExecuteFilter*, size_t> FilterBinsHashTable;

    FilterBinsHashTable m_filterBinsTable;

    agx::Vector<agxCollide::ContactPoint> m_contactPoints;
  };
}
