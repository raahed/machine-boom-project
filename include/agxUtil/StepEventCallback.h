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

#include <agxSDK/StepEventListener.h>

namespace agxUtil
{
  /**
  Helper to use lambdas as preCollide, pre, post and last step events. The arguments
  after the lambda are bind to the arguments. A pointer to the simulation is mandatory.
  Usage:
    // Pre-collide callback that prints time.
    agxUtil::StepEventCallback::preCollide( []( agxSDK::Simulation* simulation )
    {
      std::cout << "Simulation time: " << simulation->getTimeStamp() << " s." << std::endl;
    }, simulation );

    // Post-step callback with a counter.
    agxUtil::StepEventCallback::post( []( agxSDk::Simulation*, int& counter )
    {
      std::cout << "# time steps: " << ++counter << std::endl;
    }, simulation, int() );

  \note The internally created agxSDK::StepEventListener has to be removed either
        implicitly when calling agxSDK::Simulation::cleanup or explicitly by doing
        agxUtil::StepEventCallback::uninitialize( simulation ).
  */
  class AGXPHYSICS_EXPORT StepEventCallback : public agxSDK::StepEventListener
  {
    public:
      /**
      Removes internal step event listener from the simulation (if created).

      \note It's in general not necessary to call this as long as
            agxSDK::Simulation::cleanup is called with flag ALL.
      */
      static void uninitialize( agxSDK::Simulation* simulation );

      /**
      Register a pre-collide step event callback.
      \param callback - the callback function
      \param simulation - the simulation the callback should be coupled to
      \param args - rest of the arguments to the callback
      */
      template<typename FuncT, typename... Args>
      static void preCollide( FuncT callback, agxSDK::Simulation* simulation, Args&&... args );

      /**
      Register a pre step event callback.
      \param callback - the callback function
      \param simulation - the simulation the callback should be coupled to
      \param args - rest of the arguments to the callback
      */
      template<typename FuncT, typename... Args>
      static void pre( FuncT callback, agxSDK::Simulation* simulation, Args&&... args );

      /**
      Register a post step event callback.
      \param callback - the callback function
      \param simulation - the simulation the callback should be coupled to
      \param args - rest of the arguments to the callback
      */
      template<typename FuncT, typename... Args>
      static void post( FuncT callback, agxSDK::Simulation* simulation, Args&&... args );

      /**
      Register a last step event callback.
      \param callback - the callback function
      \param simulation - the simulation the callback should be coupled to
      \param args - rest of the arguments to the callback
      */
      template<typename FuncT, typename... Args>
      static void last( FuncT callback, agxSDK::Simulation* simulation, Args&&... args );

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~StepEventCallback();

      /**
      Add notification callback.
      */
      virtual void addNotification() override;

      /**
      Remove notification callback - will unregister this from the static instances.
      */
      virtual void removeNotification() override;

      virtual void preCollide( const agx::TimeStamp& t ) override;
      virtual void pre( const agx::TimeStamp& t ) override;
      virtual void post( const agx::TimeStamp& t ) override;
      virtual void last( const agx::TimeStamp& t ) override;

    private:
      enum CallbackType
      {
        PRE_COLLIDE_TYPE,
        PRE_TYPE,
        POST_TYPE,
        LAST_TYPE,
        NUM_CALLBACK_TYPES
      };

      /**
      Callback: []( agxSDK::Simulation* simulation, ... ) {}
      where ... are arbitrary arguments. Note that we're by
      default binding 'simulation' to the first argument.
      */
      using Callback = std::function<void()>;
      using StepEventCallbackRef = agx::ref_ptr<StepEventCallback>;

      using InstanceType = agx::HashTable<const agxSDK::Simulation*, StepEventCallbackRef>;
      using Callbacks = agx::Vector<agx::Vector<Callback>>;

    private:
      /**
      Register new callback of \p callbackType for \p simulation.
      \param callback - new callback
      \param callbackType - callback type
      \param simulation - simulation
      */
      static void registerCallback( Callback callback, CallbackType callbackType, agxSDK::Simulation* simulation );

    private:
      /**
      Hidden constructor - this object is only instantiated when the user
      calls one of the static methods.
      */
      StepEventCallback();

      /**
      Execute all callbacks of given callback type.
      */
      void executeCallbacks( const agx::TimeStamp& t, CallbackType callbackType );

    private:
      static InstanceType s_instances;
      Callbacks m_callbacks;
  };

  template<typename FuncT, typename... Args>
  void StepEventCallback::preCollide( FuncT callback, agxSDK::Simulation* simulation, Args&&... args )
  {
    if ( simulation == nullptr )
      return;

    registerCallback( std::bind( callback, simulation, std::forward<Args>( args )... ), PRE_COLLIDE_TYPE, simulation );
  }

  template<typename FuncT, typename... Args>
  void StepEventCallback::pre( FuncT callback, agxSDK::Simulation* simulation, Args&&... args )
  {
    if ( simulation == nullptr )
      return;

    registerCallback( std::bind( callback, simulation, std::forward<Args>( args )... ), PRE_TYPE, simulation );
  }

  template<typename FuncT, typename... Args>
  void StepEventCallback::post( FuncT callback, agxSDK::Simulation* simulation, Args&&... args )
  {
    if ( simulation == nullptr )
      return;

    registerCallback( std::bind( callback, simulation, std::forward<Args>( args )... ), POST_TYPE, simulation );
  }

  template<typename FuncT, typename... Args>
  void StepEventCallback::last( FuncT callback, agxSDK::Simulation* simulation, Args&&... args )
  {
    if ( simulation == nullptr )
      return;

    registerCallback( std::bind( callback, simulation, std::forward<Args>( args )... ), LAST_TYPE, simulation );
  }
}
