#pragma once

#include <agx/Singleton.h>
#include <agx/DynamicLibrary.h>
#include <agxDotnet/agxDotnet_export.h>

// Header files copied from https://github.com/dotnet/core-setup
#include <external/dotnet/coreclr_delegates.h>
#include <external/dotnet/hostfxr.h>

namespace agxDotnet {

  /**
   * Ref for error codes: https://github.com/dotnet/runtime/blob/master/docs/design/features/host-error-codes.md
   */


  class AGXDOTNET_EXPORT HostedRuntime : public agx::Singleton {
  public:
    class AGXDOTNET_EXPORT Function {
    public:
      Function(component_entry_point_fn handle);

      int call(void *argData, size_t argNumBytes);

    private:
      component_entry_point_fn m_fn;
    };

    AGX_DECLARE_POINTER_TYPES( Assembly );
    class AGXDOTNET_EXPORT Assembly : public agx::Referenced {
    public:
      agx::String getName() const;
      agx::String getRootPath() const;
      agx::String getConfigPath() const;
      agx::String getLibPath() const;

      Function loadEntrypoint(const agx::String& classPath, const agx::String& methodName, agx::String& error);

    protected:
      ~Assembly() override;

    private:
      friend class HostedRuntime;
      explicit Assembly(const agx::String& rootPath, const agx::String& name);

    private:
      const agx::String m_rootPath;
      const agx::String m_name;
      load_assembly_and_get_function_pointer_fn m_loadFn;
    };

  public:
    static HostedRuntime *instance();
    SINGLETON_CLASSNAME_METHOD();

    static HostedRuntime* init(const agx::String& distRoot, const agx::String& mainAssemblyName);

    Assembly* getMainAssembly();

    Assembly* assemblyFromConfig(const agx::String& distRoot, const agx::String& mainAssemblyName);

  protected:
    void shutdown() override;
    ~HostedRuntime() override;

  private:
    void loadHostFxr();

  private:
    hostfxr_initialize_for_runtime_config_fn m_initFn;
    hostfxr_get_runtime_delegate_fn m_getDelegateFn;
    hostfxr_close_fn m_closeFn;
    agx::DynamicLibraryRef m_hostFxrLibrary;
    AssemblyRef m_mainAssembly;
  };
}
