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

#ifndef AGX_TASKMANAGER_H
#define AGX_TASKMANAGER_H

#include <agxIO/ResourceManager.h>
#include <agx/Vector.h>
#include <agx/HashTable.h>
#include <agx/Task.h>
#include <agx/PluginMacros.h>
#include <agx/ThreadSynchronization.h>
#include <exception>
#include <stdexcept>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4275 ) //  warning C4275: non dll-interface class
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif

namespace agx
{
  class TaskPluginImplementation;
  class TiXmlElement;

  class AGXCORE_EXPORT TaskLoadError : public std::runtime_error
  {
  public:
    TaskLoadError(const String& path, const String& implementation, const ParameterTable& parameterList, Device *device, const char* message) : std::runtime_error(message), m_path(path), m_implementation(implementation), m_parameterList(parameterList), m_device(device) {}
    virtual ~TaskLoadError() throw() {}

    const String& getPath() const;
    const String& getImplementation() const;
    const Device *getDevice() const;
    const ParameterTable& getParameterList();

  private:
    TaskLoadError& operator = ( const TaskLoadError& ) { agxAbort(); return *this; }

    String m_path;
    String m_implementation;
    const ParameterTable& m_parameterList;
    Device *m_device;
  };


  class AGXCORE_EXPORT TaskPlugin
  {
  public:
    TaskPlugin(const char* modelPath, Device::Type deviceType);
    ~TaskPlugin();

    const char *getPath() const;
    Device::Type getDeviceType() const;

    Task* create(Device* device, const Name& implementation);
    Task* create(Device* device, const ParameterTable& parameterList, const Name& implementation = Name());

    void registerModels();

    size_t getNumImplementations() const;
  private:
    friend class TaskPluginImplementation;
    friend class TaskManager;
    //PluginHandle m_handle;
    const char *m_path;
    Device::Type m_deviceType;
    size_t m_numImplementations;
    TaskPluginImplementation *m_implementationList;
  };

  typedef HashTable<const char *, TaskPlugin *> TaskPluginTable;


  class AGXCORE_EXPORT TaskPluginImplementation
  {
  public:
    TaskPluginImplementation(TaskPlugin *plugin, const char *name = "");

    TaskPlugin *getPlugin();
    const char *getName() const;

    void registerModel();

    virtual Task *create(Device *device) = 0;
    virtual void destroy(Task *task) = 0;

    virtual ~TaskPluginImplementation() {}

  private:
    friend class TaskPlugin;
    TaskPlugin *m_plugin;
    const char *m_name;
    TaskPluginImplementation *m_next;
  };

  template <typename T>
  class TaskPluginImplementationT : public TaskPluginImplementation
  {
  public:
    TaskPluginImplementationT(TaskPlugin *plugin, const char *name = "") : TaskPluginImplementation(plugin, name) {}
    virtual ~TaskPluginImplementationT() {}

    // TaskPluginImplementation(const String& filePath, const String& name) : TaskPluginImplementation(filePath, name) {}

    virtual Task *create(Device *device) { return new T(device); }
    virtual void destroy(Task *task) { agxAssert(dynamic_cast<T *>(task)); static_cast<T *>(task)->~T(); }
  private:
  };


  class AGXCORE_EXPORT TaskManager : public agx::Singleton
  {
  public:
    static TaskManager *instance();

    Task *create(const Path& path, Device *device, const Name& implementation);
    Task *create(const Path& path, Device *device, const ParameterTable& parameterList, const Name& implementation = Name());
    Task *create(const Path& path, const Name& implementation, agx::ParameterBindingVector& parameterList, bool allowFormatPromotions, Device *device);

    // void scan(const String& path);
    // void rescan();

    SINGLETON_CLASSNAME_METHOD();

  protected:
    virtual ~TaskManager();
    virtual void shutdown() override;

  private:
    friend class TaskPlugin;
    friend class UnresolvedTask;
    TaskManager();

    String getFileSystemPath(const String& resourcePath, const agx::String& extension);
    String getPluginPath(Path resourcePath);

    void loadCallback(TiXmlElement *eKernel);

    #if 0
    typedef Callback1<const String&> ScanCallback;
    void scan(const String& path, ScanCallback callback);
    void load(const String& path);
    void scanComponents(const String& baseDir);
  private:
    typedef HashTable<String, TaskPluginRef> TaskPluginTable;

    TaskPluginTable m_taskPathPluginTable;
    TaskPluginTable m_taskNamePluginTable;
    StringVector m_taskDirectories;
    #endif

  private:
    // Load callback communication
    Device *m_loadDevice;
    Name m_loadImplementation;
    const ParameterTable *m_loadParameters;
    Task *m_loadResult;
    static ReentrantMutex s_mutex;
    agx::Vector< PluginHandle > m_pluginHandles;
  };
  /* Implementation */
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif /* _AGX_TASKMANAGER_H_ */
