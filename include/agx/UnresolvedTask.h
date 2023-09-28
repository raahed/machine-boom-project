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


#ifndef AGX_UNRESOLVED_TASK_H
#define AGX_UNRESOLVED_TASK_H


#include <agx/config/AGX_USE_OPENCL.h>
#include <agx/Task.h>
#include <agxData/Value.h>
#include <agx/Pointer.h>
#include <agx/ThreadSynchronization.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4275 ) //  warning C4275: non dll-interface class
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agx
{
  AGX_DECLARE_POINTER_TYPES(UnresolvedTask);
  AGX_DECLARE_VECTOR_TYPES(UnresolvedTask);
  class AGXCORE_EXPORT UnresolvedTask : public SerialTask
  {
  public:
    static agx::Model* ClassModel();

    static Task* load(TiXmlElement* eTask, Device* device);
    virtual void configure(TiXmlElement* eTask) override;
    void configureSource(TiXmlElement* eTask);
    static void deleteAllConfiguredProxies();

    typedef Callback1<UnresolvedTask *> InjectCallback;

    static Task* parse(const String& function, const InjectCallback& injectCallback = InjectCallback(&UnresolvedTask::injectInstance));
  public:
    UnresolvedTask(
      const Name& instanceName, Device* device, const Path& taskPath, const Name& implementationName,
      const Path& basePath = Path(), const InjectCallback& injectCallback = InjectCallback(&UnresolvedTask::injectInstance));

    virtual void run() override;

    Task* getInstance();

    const Name& getInstanceName() const;
    const Name& getImplementationName() const;

    // virtual agx::Job* getPreJob();
    // virtual agx::Job* getPostJob();

  protected:
    virtual ~UnresolvedTask();


  private:
    void retain();
    void valueUpdateCallback(Parameter* value);
    bool instantiateProxy();
    void performLoadAttempts();
    void tryPlugin();
    void tryKernel();
    void tryKernel(const Path& path);
    void tryTask();
    void swapDevice();
    void tryInstanciateTask(TiXmlElement* eTask);
    void addProxyParameter(UnresolvedParameter* parameter);
    void configureInstance();
    static void splitParameters(const String& function, StringVector& result);
    void configure(const String& function);

    using Task::configure;

    static void injectInstance(UnresolvedTask* proxy);
    // static Parameter* loadParameter(TiXmlElement* eValue);

#if AGX_USE_OPENCL()
    void createVerifierTask();
#endif

    static void glInitCallback();
    static agx::HashSet<UnresolvedTask *> s_glTasks;
    static Callback s_glInitCallback;

  private:
    // typedef HashTable<String, ParameterRef> ParameterTable;
    ParameterTable m_proxyParameters;
    Path m_basePath;
    Path m_taskPath;
    Task* m_instance;
    Name m_instanceName;
    Name m_implementation;
    Device* m_device;
    Parameter::Event::CallbackType m_valueUpdateCallback;
    ParameterPtrVector m_nonInstanceParameters;
    InjectCallback m_injectCallback;
    Path m_triggerPath;
    TaskRef m_retain;
    PointerRefVector m_aliases;
    TiXmlElement* m_eTask;
#if AGX_USE_OPENCL()
    Bool m_shouldVerify;
#endif
  };
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif


#endif /* _AGX_UNRESOLVED_TASK_H_ */
