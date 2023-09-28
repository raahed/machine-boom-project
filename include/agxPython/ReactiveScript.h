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
#ifndef AGXPYTHON_REACTIVECRIPT_H
#define AGXPYTHON_REACTIVECRIPT_H 1

#include <agx/config/AGX_USE_PYTHON.h>

#if AGX_USE_PYTHON()

#include <agxPython/export.h>

#include <agx/Name.h>
#include <agx/Uuid.h>
#include <agxSDK/StepEventListener.h>
#include <agxPython/ScriptConsole.h>
#include <agxPython/ScriptRuntimeListener.h>
#include <vector>

namespace agxSDK
{
  class Simulation;
}

/**
ReactiveScripts are Python scripts for use in strong
coupling with the Simulation instance it is assigned
and the stepping in time it grants.

Derived from the agxSDK::StepEventListener, the idea
is to design the appropriate storage for generic metadata,
Python runtime data and Simulation context data. Only the
metadata is independent of any attachment to a Simulation
and thus permanent relative the other two which only
lasts as long the assignment to the SImulation does.

An EventListener is notified upon being added to a Simulation
and later on its removal from the Simulation. This
pair of callbacks defines the start and finish of the
initialized Script. Only during this time can we
access the agxSDK::Simulation pointer the Reactive
API require to work.

Normally, an EventListener is managed by the Simulation
it is added to. Because we want the ScriptManager to
manage the ReactiveScripts and their execution context
data, we also need a mechanism for each ReactiveScript to
announce when and how it is invoked when triggered as
a StepEventListener of its Simulation and again before
returning. These restores the context state to the Python
environment before Script execution resumes as well stores
the updated context state from the Python runtime.

At the bare minimum, a ReactiveScript consists of initializing
code such as initial values to locals, functions, ReactiveScript
callbacks or even classes. All code found in the global scope
counts as initialization code and is run only once per assignment
to a Simulation. The objects created during initialization is
stored afterwards until the script is called upon again and
restored to Python. This context state is also scanned for
specially named functions to use as callable objects back
into the script from the overridden StepEventListener methods.
If found, the appropriate method is unlocked and will be called
by the Simulation object.

So what does this solve? The scriptable StepEventListener
with potentially to make serializable storage aside, ReactiveScripts
can also manage themselves in response to assignment to Simulation
objects and later removal, they also manage all Python runtime data
required to execute any Python code (assuming the ScriptManager has
initiated the Python C API in embedded mode). Only one ReactiveScript
can execute at a time/be active (for now) because only one context
state can be current.


Algorithm:
   ScriptManager->registerReactiveScript(ReactiveScript* script);
   script->setSource(pythonScriptString);
   Simulation->add(script);
   ScriptManager->
*/

namespace agxPython
{

  //class ScriptConsole;
  //class ScriptRuntimeListener;
  /**
  The ReactiveScript class is derived from
  the agxSDK::StepEventListener class and require an
  agxSDK::Simulation instance to invoke any callbacks. The
  most important callbacks are overridden from
  agxSDK::EventListener in ReactiveScript:

  virtual void addNotification() override;
  virtual void removeNotification() override;

  Designed as objects managed by agxPython::ScriptManager instead
  of agxSDK::Simulation, these callbacks helps us with knowing when
  to initialize the script and when to cleanup. Because they are
  called as part of the operation of adding or removing EventListeners,
  any such operation must complete and return before we act on them.
  */

  class AGXPYTHON_EXPORT ReactiveScript : public agxSDK::StepEventListener
  {

  public:

    friend class ScriptManager;

    ReactiveScript(const agx::Name& name);

    ~ReactiveScript();

    void setSource(const agx::String& source);

    bool setSourceFromFile(const agx::String& path);

    const agx::String& getSource() const;

    void setScriptConsole(ScriptConsole* scriptConsole);

    ScriptConsole* getScriptConsole();

    void setScriptRuntimeListener(ScriptRuntimeListener* scriptRuntimeListener);

    ScriptRuntimeListener* getScriptRuntimeListener();

    void resetError();

    const agx::String& getLastErrorMessage() const;

    long getLastErrorLine() const;

    bool errorOccurred() const;

    const agx::Name& getName() const;

    const agx::Uuid& getUuid() const;

    void setPriority(agx::UInt32 priority);

    bool isReadOnly();

    void setReadOnly(bool readOnly);

    virtual void post(const agx::TimeStamp& timeStamp) override;

    virtual void onStart(const agx::TimeStamp& timeStamp);

    virtual void onStop(const agx::TimeStamp& timeStamp);

    virtual void addNotification() override;

    virtual void removeNotification() override;

    void update();

#ifndef SWIG
    struct Runtime
    {
      Runtime();
      ~Runtime();
      agx::ref_ptr<ScriptConsole> console;
      agx::ref_ptr<ScriptRuntimeListener> listener;
      void* compiled;
      void* cbStep;
      void* cbStart;
      void* cbStop;
    };

    enum Action
    {
      ActionSimulationRemove = 1,
      ActionStateInitialize,
      ActionStateClear,
    };
#endif

  protected:

    void pushAction(Action action, void* param = nullptr);

    void setLastErrorMessage(const agx::String &lastErrorMessage);

    void setLastErrorLine(long lastErrorLine);

  private:

    std::vector<std::pair<Action, void*> > m_pendingActions;

    agx::Name m_name;

    agx::Uuid m_uuid;

    agx::String m_source;

    agx::String m_lastErrorMessage;

    long m_lastErrorLine;

    Runtime m_runtime;

    bool m_readOnly;

  };


}



#endif



#endif
