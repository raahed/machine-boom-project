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

#ifndef AGXFMI2_IMPORT_MASTER_H
#define AGXFMI2_IMPORT_MASTER_H

#include <agx/config/AGX_USE_FMI.h>
#include <agx/config/AGX_USE_WEBPLOT.h>

#if AGX_USE_FMI()

#include <agxFMI2/import/Module.h>

#include <agx/Component.h>
#include <agxSDK/Simulation.h>
#include <agx/RealTimeTrigger.h>

#if AGX_USE_WEBSOCKETS()
#include <agxNet/WebSocket.h>
#endif

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!

extern "C"
{
  #include <JM/jm_callbacks.h>
}

#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxOSG
{
  class ExampleApplication;
}


namespace agxFMI2
{
  namespace Import
  {
    AGX_DECLARE_POINTER_TYPES(Master);

    /**
    Master
    */
    class AGXFMI_EXPORT Master : public agx::Component
    {
    public:
      Master(Module::LogLevel logLevel = Module::LOG_LEVEL_INFO);
      Master(agxSDK::Simulation *simulation, Module::LogLevel logLevel = Module::LOG_LEVEL_INFO);

      void enableDirectApplicationControl(agxOSG::ExampleApplication *app);

      agxSDK::Simulation *getSimulation();
      const agxSDK::Simulation *getSimulation() const;

      void load(const agx::String& xmlPath);
      Module *loadModule(const agx::String& path);

      const agx::String& getDiskRootPath() const;

      Module *getModule(const agx::Name& name);

      void setTimeStep(agx::Real dt);

      void init(agx::Real tStart, agx::Real tEnd, bool stopTimeDefined = true);

      void bind(Variable *input, Variable *output);

      void execute();
      void terminate();


      agx::Real getStartTime() const;
      agx::Real getEndTime() const;
      agx::Real getTimeStep() const;

      void setControlChannelFrequency(agx::Real frequency);

      void setDebug(bool flag);
      bool getDebug() const;


    protected:
      virtual ~Master();

    private:
      AGX_DECLARE_POINTER_TYPES(Binding);
      AGX_DECLARE_VECTOR_TYPES(Binding);

      Variable *getVariable(const agx::String& path);
      void preTickCallback(agx::Clock *clock);
      void stepForward();

#if AGX_USE_WEBSOCKETS()
      void ccGetModules(agxNet::WebSocket::ControlChannel *, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
      void ccGetModuleDescription(agxNet::WebSocket::ControlChannel *, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
      void ccGetInitialState(agxNet::WebSocket::ControlChannel *, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
      void ccSetParameter(agxNet::WebSocket::ControlChannel *, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
      void ccGetParameter(agxNet::WebSocket::ControlChannel *, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
#endif

    private:
      agxSDK::SimulationRef m_simulation;
      agx::Clock::TickEvent::CallbackType m_preTickCallback;

      agx::String m_diskRootPath;
      jm_callbacks m_jmCallbacks;

      ModuleRefVector m_modules;
      BindingRefVector m_bindings;
      agx::Real m_startTime;
      agx::Real m_endTime;
      agx::Real m_timeStep;
      agx::Real m_currentTime;
      agx::RealTimeTrigger m_rtTrigger;
      bool m_debug;
      Module::LogLevel m_logLevel;
#if AGX_USE_WEBSOCKETS()
      agxNet::WebSocket::ControlChannelRef m_controlChannel;
#endif
    };

#ifndef SWIG
/*  To-do: remove the necessity of this conditional using cleaner declaration

    Reason: SWIG's primitive parser gets confused as this decl appears twice
            in the AST SWIG produces, making it impossible ignore _properly_. */

    class Master::Binding : public agx::Referenced
    {
    public:
      Binding(Variable *input, Variable *output);

      Variable *getInput();
      Variable *getOutput();

      void transferData();

    protected:
      virtual ~Binding();

    private:
      VariableRef m_input;
      VariableRef m_output;
    };
#endif
  }
}

#endif /* AGX_USE_FMI */

#endif /* AGX_IMPORT_MASTER_H */
