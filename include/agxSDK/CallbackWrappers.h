/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#include <agx/config/AGX_USE_WEBSOCKETS.h>

// The callback wrappers are tightly coupled to the agxSDK::SimulationController,
// and that class doesn't exist when we're building without web sockets.
#if AGX_USE_WEBSOCKETS()

#include <agx/Real.h>
#include <agx/Callback.h>

namespace agxSDK
{
  class SimulationController;

  /**
  A bit of extra layer in order to avoid templated classes in SWIG bindings. When a
  C# application need to get callback from a Application instance, then it should
  create a callback receiver class that inherits from CallbackWrapper and register
  it using one of the Application::register[Event]Callback methods.
  
  When the event is triggered, the 'callback' method will be called.
  
  Additional subclasses of this class will be created as the need for callback with
  specific argument types are required.
  */
  class AGXPHYSICS_EXPORT CallbackWrapperBase : public agx::Referenced
  {
  public:
    CallbackWrapperBase();
  protected:
    virtual ~CallbackWrapperBase() {}
    friend class SimulationController;
    void setSimulationController( SimulationController* controller );
    SimulationController* m_controller;
  };


  class AGXPHYSICS_EXPORT CallbackWrapper : public CallbackWrapperBase
  {
  public:
    CallbackWrapper();
    virtual void callback( SimulationController* /*application*/) {}
    agx::Callback* getCallback();
  protected:
    virtual ~CallbackWrapper() {}

  private:
    // The method called by 'm_callback'. Calls the virtual 'callback' method.
    void bridge();

    /// This is the actual callback object that is passed around behind the scenes.
    agx::Callback m_callback;
  };


  class AGXPHYSICS_EXPORT CallbackWrapper_Real : public CallbackWrapperBase
  {
  public:
    CallbackWrapper_Real();
    virtual void callback( SimulationController* /*application*/, agx::Real /*value*/) {}
    agx::Callback1<agx::Real>* getCallback();

  protected:
      virtual ~CallbackWrapper_Real() {}

  private:
    // The method called by 'm_callback'. Calls the virtual 'callback' method.
    void bridge(agx::Real value);

    /// This is the actual callback object that is passed around behind the scenes.
    agx::Callback1<agx::Real> m_callback;
  };

  class AGXPHYSICS_EXPORT CallbackWrapper_Bool : public CallbackWrapperBase
  {
    public:
      CallbackWrapper_Bool();
      virtual void callback( SimulationController* /*application*/, bool /*value*/) {}
      agx::Callback1<bool>* getCallback();
    protected:
      virtual ~CallbackWrapper_Bool() {}

    private:
      // The method called by 'm_callback'. Calls the virtual 'callback' method.
      void bridge(bool value);

      /// This is the actual callback object that is passed around behind the scenes.
      agx::Callback1<bool> m_callback;
  };

  class AGXPHYSICS_EXPORT CallbackWrapper_String : public CallbackWrapperBase
  {
    public:
      CallbackWrapper_String();
      virtual void callback( SimulationController* /*application*/, agx::String /*value*/) {}
      agx::Callback1<agx::String>* getCallback();
    protected:
      virtual ~CallbackWrapper_String() {}

    private:
      // The method called by 'm_callback'. Calls the virtual 'callback' method.
      void bridge(agx::String value);

      /// This is the actual callback object that is passed around behind the scenes.
      agx::Callback1<agx::String> m_callback;
  };
}

#endif // AGX_USE_WEBSOCKETS()