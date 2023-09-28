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

#ifndef AGXPYTHON_SCRIPTCONTEXT_H
#define AGXPYTHON_SCRIPTCONTEXT_H


#include <agx/config/AGX_USE_PYTHON.h>
#include <agx/config.h>

#if AGX_USE_PYTHON()

#include <agx/config/AGX_USE_FMI.h>
#include <agx/Referenced.h>
#include <agx/ref_ptr.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES(Simulation);
}

namespace agxOSG
{
  class Group;
  class ExampleApplication;
  class SceneDecorator;
}

#include <agxPython/export.h>

#include <agx/Timer.h>
#include <agx/agx_vector_types.h>

#include <map>
#include <functional>

namespace osg
{
  class Group;
}


namespace agxFMI2
{
  namespace Export
  {
    class Module;
  }
}

namespace agxPython
{

  class ScriptContext;
  class ScriptContextInterface;

  AGX_DECLARE_POINTER_TYPES(ScriptSimulationParameter);

  /**
  Class describing the contextual environment in which
  a script is run. Because this class is wrapped by SWIG,
  there is no need to do it ourselves, as long the values
  and objects returned by the getters are as well.
  */
  class AGXPYTHON_EXPORT ScriptContext_environment : public agx::Referenced
  {
  public:

    friend class ScriptContext;
    friend class ScriptContextInterface;

    agxSDK::Simulation*         getSimulation();
    osg::Group*                 getSceneRoot();
    agxOSG::ExampleApplication* getApplication();
    agxOSG::SceneDecorator*     getSceneDecorator();

    agx::String                 getName();

#if AGX_USE_FMI()
    agxFMI2::Export::Module*    getFMI2Module();
    int                         getFMIVersion();
#endif

  protected:

    ScriptContext_environment();
    ~ScriptContext_environment();

  private:
    agxSDK::Simulation         *m_simulation;
    osg::Group                 *m_group;
    agxOSG::ExampleApplication *m_exampleApplication;
    agx::String m_name;
#if AGX_USE_FMI()
    int                         m_fmiVersion;
    void                       *m_fmiModule;
#endif
    agxOSG::SceneDecorator     *m_sceneDecorator;

  };




  /**
  Class which manages a runtime bridge between a script
  and the Python ScriptManager.
  */
  class AGXPYTHON_EXPORT ScriptContext : public agx::Referenced
  {

  public:

    friend class ScriptManager;
    friend class ScriptContextInterface;

    agxPython::ScriptContextInterface *getInterface();

    void setInterface(agxPython::ScriptContextInterface *i);

    ScriptContext_environment* environment; // See m_environment
  protected:

    ScriptContext(int contextType);
    virtual ~ScriptContext();

    void setEnvironment(ScriptContext_environment *e);

  private:
    static bool verify_apply_destroy(ScriptContext *scriptContext, agxPython::ScriptContextInterface *scriptContextInterface);

    int m_contextType;

    agx::ref_ptr<ScriptContextInterface>    m_interface;

    /*
    We have both environment and m_environment holding the same value.
    m_environment is needed to hold the reference and environment likely
    existed first (and clearly was not enough).
    The public variable is used in python boiler plate code in tutorials
    and other snippets:

      agxPython.getContext().environment.getSimulation()

    */
    agx::ref_ptr<ScriptContext_environment> m_environment;
  };

  /**
  Internal class which interfaces to a ScriptContext in
  an abstract and safe manner. It gives AGX some control
  over the environment in which scripts are run.
  */
  class AGXPYTHON_EXPORT ScriptContextInterface : public agx::Referenced
  {

  public:

    friend class ScriptManager;
    friend class ScriptContext;

    ScriptContextInterface() =delete;

    void setSimulation(agxSDK::Simulation* simulation) const;
    void setApplication(agxOSG::ExampleApplication* application) const;
    void setSceneRoot(osg::Group* group) const;
    osg::Group* getSceneRoot() const;
    void setSceneDecorator(agxOSG::SceneDecorator* decorator) const;
    void setName(const agx::String& name) const;

#if AGX_USE_FMI()
    void setFMI2Module(agxFMI2::Export::Module *module) const;
#endif

    int getType() const;

    /// Returns true on success, which also destroys the instance. If the
    /// script context is not set up properly for its type, it returns false
    /// without side-effects. Otherwise on error, an exception is thrown.
    bool verifyAndApply();

    agxPython::ScriptContext* getContext();
  protected:

    ScriptContextInterface(ScriptContext *scriptContext);
    virtual ~ScriptContextInterface();

  private:

    agx::observer_ptr<ScriptContext> m_scriptContext;

  };

}

#endif

#endif
