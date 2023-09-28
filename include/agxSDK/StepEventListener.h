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

#include <agxSDK/EventListener.h>
#include <agx/TimeStamp.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES(StepEventListener);
  AGX_DECLARE_VECTOR_TYPES(StepEventListener);
  /**
  Derive from this class to implement a listener for simulation step events. The following methods can be implemented to catch events:
  Method/Mask
  ------------
  - preCollide()/PRE_COLLIDE - Before collision detection
  - pre()/PRE_STEP - after collision detection, but before a dynamics step is taken
  - post()/POST_STEP - after a dynamics step (integration) step is taken.
  - last()/LAST_STEP - Triggered at the end of Simulation::stepForward() with the _updated_ simulation time.
  */
  class AGXPHYSICS_EXPORT StepEventListener : public EventListener
  {
    public:

      /// Defines the event states for which a listener will be activated, a mask can be bitwise OR:ed from these members.
      enum ActivationMask {
        STEP_NONE = 0x0,
        PRE_COLLIDE = 0x1,       /**< Triggered just BEFORE a simulation step is taken */
        PRE_STEP = 0x2,               /**< Triggered just BEFORE a Dynamics stepForward step is taken */
        POST_STEP = 0x4,              /**< Triggered just AFTER a Dynamics stepForward step is taken */
        LAST_STEP = 0x8,            /**< Triggered last in a Simulation::stepForward call */
        DEFAULT = PRE_STEP | POST_STEP, /**< The default activation mask */
        ALL = PRE_COLLIDE | PRE_STEP | POST_STEP | LAST_STEP /**< All events */
      };

      /// Default constructor, sets the default activation mask to all (POST_STEP and PRE_STEP) events.
      StepEventListener( int mask = DEFAULT );

      /**
      Specifies a bitmask which determines which event types will activate this listener.
      \param mask - bitwise or:ed mask for the event types (ActivationMask) for which events this event listener will be activated
      */
      virtual void setMask(int mask) override;


      // ================ Override these three methods to catch events
      /**
      Called before collision detection is performed in the simulation
      Implement this method in the derived class to get callbacks.
      \param time - the current simulation time
      */
      virtual void preCollide(const agx::TimeStamp& time);

      /**
      Called before a step is taken in the simulation
      Implement this method in the derived class to get callbacks.
      \param time - the current simulation time
      */
      virtual void pre(const agx::TimeStamp& time);

      /**
      Called after a step is taken in the simulation
      Implement this method in the derived class to get callbacks.
      \param time - the current simulation time
      */
      virtual void post(const agx::TimeStamp& time);

      /**
      Called after a step is taken in the simulation
      Implement this method in the derived class to get callbacks.
      \note In this call, the simulation time is the new updated time.
      \param time - The new updated simulation time incremented by the time step.
      */
      virtual void last(const agx::TimeStamp& time);

#ifndef SWIG
      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::StepEventListener );
#endif

    protected:


      /// Destructor
      virtual ~StepEventListener();
  };



  inline void StepEventListener::preCollide(const agx::TimeStamp& /*time*/) { }
  inline void StepEventListener::pre(const agx::TimeStamp& /*time*/) { }
  inline void StepEventListener::post(const agx::TimeStamp& /*time*/) { }
  inline void StepEventListener::last(const agx::TimeStamp& /*time*/) { }


}
