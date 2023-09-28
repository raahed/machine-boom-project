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

#ifndef AGX_COMPONENTMANAGER_H
#define AGX_COMPONENTMANAGER_H


#include <agx/Referenced.h>
#include <agx/Vector.h>
#include <agx/HashTable.h>
#include <agx/Callback.h>
#include <agx/Component.h>
#include <agx/Device.h>
#include <agxIO/FileState.h>
#include <agxIO/ResourceManager.h>
#include <agx/Logger.h>
#include <agx/ThreadSynchronization.h>

DOXYGEN_START_INTERNAL_BLOCK()
namespace agx
{
  typedef Component* ( *ComponentConstructor )( Device *device );
  typedef void ( *ComponentDestructor )( Component * );
  typedef const char* ( *ComponentNameFunction )();

  AGX_DECLARE_POINTER_TYPES(AbstractComponentPlugin);
  AGX_DECLARE_VECTOR_TYPES(AbstractComponentPlugin);

  class AGXCORE_EXPORT AbstractComponentPlugin
  {
  public:
    AbstractComponentPlugin(const String& path);
    virtual ~AbstractComponentPlugin();

    const String& getPath() const;

    virtual Component *create(Device *device) = 0;
    virtual void destroy(Component *component) = 0;
    // void reload();


  private:
    #if 0
    template <typename T>
    void loadSymbol(T& function, const char *symbol);

    PluginHandle m_handle;

    ComponentConstructor m_constructor;
    ComponentDestructor m_destructor;
    ComponentNameFunction m_nameFunction;
    #endif

  private:
    String m_path;
    // String m_name;
    // bool m_initialized;
    // ComponentRefVector m_instances;
    // agxIO::FileState m_fileState;
  };


  template <typename T>
  class ComponentPlugin : public AbstractComponentPlugin
  {
  public:
    ComponentPlugin(const String& path);
    virtual Component *create(Device *device);
    virtual void destroy(Component *component);

  private:
  };

  template <typename T>
  ComponentPlugin<T>::ComponentPlugin(const String& path) : AbstractComponentPlugin(path)
  {
  }

  template <typename T>
  Component *ComponentPlugin<T>::create(Device *device)
  {
    return new T(device);
  }

  template <typename T>
  void ComponentPlugin<T>::destroy(Component * /*component*/)
  {
    LOGGER_DEBUG() << "TODO?" << LOGGER_ENDL();
  }

  typedef HashTable<String, AbstractComponentPlugin*> ComponentPluginTable;

  class AGXCORE_EXPORT ComponentManager : public agx::Singleton
  {
  public:
    static ComponentManager *instance();

    Component *create(const Path& path, const Name& implementation = Name(), Device *device = CpuDevice::instance());

    // void scan(const String& path);
    // void rescan();

    SINGLETON_CLASSNAME_METHOD();

  private:
    virtual ~ComponentManager();
    friend class AbstractComponentPlugin;
    void registerPlugin(AbstractComponentPlugin *plugin);
    void unregisterPlugin(AbstractComponentPlugin *plugin);
    virtual void shutdown() override;

    agx::String getFileSystemPath(const agx::String& resourcePath, const agx::String& extension);
    agx::String getPluginFileSystemPath(const agx::String& resourcePath);


  private:
    ComponentManager();
    #if 0
    typedef Callback1<const String&> ScanCallback;
    void scan(const String& path, ScanCallback callback);
    void load(const String& path);
    void scanComponents(const String& baseDir);
    #endif

  private:
    ComponentPluginTable m_pluginTable;
    ReentrantMutex m_mutex;
  };



  /* Implementation */

}
DOXYGEN_END_INTERNAL_BLOCK()


#endif /* _AGX_COMPONENTMANAGER_H_ */
