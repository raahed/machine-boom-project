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

#ifndef AGXLUA_SCRIPTMANAGER_H
#define AGXLUA_SCRIPTMANAGER_H

#ifdef _MSC_VER
# pragma warning( push )
# pragma warning( disable: 4512 ) // Disable warning about assignment operator could not be generated due to reference members.
# endif


#include <agx/config/AGX_USE_LUA.h>
#include <agx/config.h>

#if AGX_USE_LUA()

#include <agx/ref_ptr.h>
#include <agx/DynamicLibrary.h>
#include <agxLua/export.h>


#include <agx/Referenced.h>
#include <agx/Logger.h>
#include <agx/Timer.h>
#include <agxLua/GarbageListener.h>

extern "C" {
#  include "lua.h"
#  include "lualib.h"
#  include "lauxlib.h"
}



namespace agxSDK
{
  class Simulation;
}

namespace agxLua {

#define LUA_SCRIPT_PLUGIN_VERSION "1.0"

  class AGXLUA_EXPORT ScriptManager : public agx::Singleton
  {
  public:


    /**
    Return the singleton object
    If erase is true then this singleton will be de-inititialized
    */
    static ScriptManager* instance( bool erase=false );


    class ScriptPlugin : public agx::Referenced
    {
    public:
      friend class ScriptManager;

      /** Constructor. Stores a reference to the script manager so it won't
      Deleted until all plugins are removed
      This requires that someone calls ScriptManager::shutdown() explicitly
      */
      ScriptPlugin() { m_script_mgr = agxLua::ScriptManager::instance(); }

      /** Should return the name of this plugin
      The library loaded will be "luaplugin_getName()"
      */
      virtual std::string getName()=0;

      /// Return the compiled version of this plugin
      std::string getPluginVersion() { return LUA_SCRIPT_PLUGIN_VERSION; }

      /**
      Add a plugin name to the list of dependent plugins.
      These plugins will be loaded in FIFO order.
      */
      void addDependentPlugin(const std::string pluginName )
      {
        m_dependentPlugins.push_back( pluginName );
      }

      /**
      This method is called before init is executed.
      It should request any plugins that is required for this specific plugin.
      For example: g_luaScriptManager->requestScriptPlugin("osg"); // This plugin depends on osg.
      Will be called before init
      */
      void registerDependentPlugins( void );

      /// Initialize the plugin in the current lua state
      virtual void init(lua_State* lua)=0;

      typedef agx::Vector<std::string> StringVector;
      const StringVector& getDependentPlugins() const { return m_dependentPlugins; }

    protected:

      /// Deinitialize the plugin
      virtual void close()=0;

      // Destructor
      virtual ~ScriptPlugin() {}
      ScriptManager* m_script_mgr;

      StringVector m_dependentPlugins;

    };


    template<class T>
    /// Class for instantiating a plugin that is loaded from a dynamic library
    class RegisterScriptPluginProxy
    {
    public:
      RegisterScriptPluginProxy()
      {
        if (ScriptManager::instance())
        {
          m_plugin = new T;

          // Check so that the plugin version matches the one of the ScriptManager
          if (m_plugin->getPluginVersion() != ScriptManager::instance()->getPluginVersion())
            LOGGER_ERROR() <<
            "Plugin version (" << m_plugin->getPluginVersion() <<
            ") does not match required version (" <<
            ScriptManager::instance()->getPluginVersion() << ")" << LOGGER_END();

          // Make sure that any plugins that this plugin depends on is initialized before.
          ScriptPlugin::StringVector::const_iterator it = m_plugin->getDependentPlugins().begin();
          for(; it != m_plugin->getDependentPlugins().end(); ++it)
          {
            if (!ScriptManager::instance()->requestScriptPlugin( *it ))
              LOGGER_ERROR() << "Error loading plugin: " << *it << LOGGER_END();
          }

          // Initialize the plugin
          m_plugin->init( ScriptManager::instance()->getLua());

          // Add the plugin to the list of plugins at the script manager
          ScriptManager::instance()->addScriptPlugin(m_plugin.get());
        }
      }

      ~RegisterScriptPluginProxy()
      {
        if (ScriptManager::instance())
        {
          // Unlist the plugin
          ScriptManager::instance()->removeScriptPlugin(m_plugin.get());
        }
      }


      agx::ref_ptr<T> getPlugin() { return m_plugin; }

    protected:
      agx::ref_ptr<T> m_plugin;
    };

    /**
    This method is used for registering a reference in a lua table.
    The reason is the following:

    Assume you have a object called Event. You create a Event inside a function as a local variable, local to avoid a
    clash with other global variables with the same name.
    Now this variable perhaps has a function defined: callback. So you write
    function myFunc()
    local e = Event:new()
    function e:callback()
    -- do stuff
    end

    -- register the event
    manager:registerEvent(e)
    end

    Now assume that the manager later on calls the callback method from the lua C API. As the variable e was declared local
    in the function myFunc, the call to e:callback() will fail.

    This is the reason for registering a reference of the Event instance in a global table.
    This table is initially empty and is created by the tolua export file.


    */
    bool registerLuaEntry(const void* ptr, const char* functionName);//, const std::string& class_name);
    bool unregisterLuaEntry(const void* ptr);//, const std::string& class_name);

#define ScriptManager_REGISTER_OBJECT( P, NAME ) agxLua::ScriptManager::instance()->registerLuaEntry(P, NAME) //this, typeid(this).name())
#define ScriptManager_UNREGISTER_OBJECT( P ) agxLua::ScriptManager::instance()->unregisterLuaEntry(P) //this, typeid(this).name())

    /// Add A script-plugin to the list of registered plugins
    void addScriptPlugin(ScriptPlugin *);

    /// Remove a script plugin from the list of registered plugins
    void removeScriptPlugin(ScriptPlugin *);

    /// Try to load and register a script plugin with the given libraryName
    bool requestScriptPlugin(const std::string& libraryName);

    /// \return true if the named plugin is loaded
    bool isLoaded(const std::string& libraryName);

    /// Deallocates any allocated memory
    void shutdown() override;

    /// Print debug information to the log-path (set in setLogPath)
    void setDebugMode(bool flag = true);

    /// Set the filename of the log file containing debug information from lua
    void setLogPath(const std::string& path) { m_log_path=path; }

    /// Initializes the ScriptManager
    void init(lua_State* L=nullptr, unsigned int stacksize=1024);

    /// Return the current lua state (cast to lua_State)
    lua_State* getLua() { return m_lua_state; }

    /// Return the path to the log file for lua debug text
    const char* getLogPath() { return m_log_path.c_str(); }

    /// Execute a lua script file. The file will be searched for in the specified filepath
    bool doFile(const std::string& path);

    /// Execute a string containing lua code. Returns true upon success
    bool doString(const std::string& string);

    /**
    Request garbage collection in lua, if currentTime < 0 then do it immediately.
    Otherwise do garbage collection only if it was more than 1/getGarbageCollectionFrequency() seconds since it was last performed.
    \return true if actual garbage collection was performed.
    */
    bool doGarbageCollection( agx::Real now = agx::Real(-1.0) );

    void setGarbageCollectionEnabled( bool b );
    bool getGarbageCollectionEnabled() const { return m_garbage_collection_enabled; };

    void setGarbageCollectionFrequency( agx::Real f ) { m_garbage_collection_frequency = f; };
    agx::Real getGarbageCollectionFrequency() const { return m_garbage_collection_frequency; };

    std::string getPluginVersion() { return m_plugin_version; }

    bool registerSimulation( agxSDK::Simulation* simulation );

    bool getTreatInvalidAsErrors() const {return m_treatInvalidAsErrors;}
    void setTreatInvalidAsErrors(const bool treatInvalidAsErrors) {m_treatInvalidAsErrors = treatInvalidAsErrors;}

    /// \return num kbytes allocated by LUA
    int getMemoryUsed() const;

    std::string getLastErrorMessage() const { return m_lastErrorMessage; }
    void setLastErrorMessage(const std::string& errorMessage) { m_lastErrorMessage = errorMessage; }
  private:

    SINGLETON_CLASSNAME_METHOD();

    // Register various functions accessible from Lua.
    void registerFunctions();


    agx::ref_ptr<GarbageListener> m_garbageListener;
    bool m_garbage_collection_enabled;
    bool m_garbage_collection_requested;
    agx::Real m_garbage_collection_frequency;
    agx::Real m_last_garbage_collection;

    typedef agx::Vector< agx::ref_ptr<ScriptPlugin> > ScriptPluginVector;
    typedef ScriptPluginVector::iterator ScriptPluginVectorIterator;

    ScriptPluginVector m_plugin_vector;

    typedef agx::Vector< agx::ref_ptr<agx::DynamicLibrary> >  DynamicLibraryList;
    bool closeLibrary(const std::string& fileName);

    /// Return the extension for dynamic library for the current os-platform
    std::string createLibraryNameForExtension(const std::string& ext);

    /// return an iterator to a named dynamic library
    DynamicLibraryList::iterator getLibraryItr(const std::string& fileName);

    /// Return a pointer to a named dynamic library
    agx::DynamicLibrary* getLibrary(const std::string& fileName);


    DynamicLibraryList m_dllist;

    std::string m_log_path;
    lua_State* m_lua_state;

    /// Destructor
    virtual ~ScriptManager();

    /// Constructor
    ScriptManager( void );

    const std::string m_plugin_version;


    typedef agx::HashTable<const void *,int> ObjectReferenceMap;
    ObjectReferenceMap m_object_references;
    bool m_initialized;
    int m_maxMemSize;
    int m_incrementalSize;
    bool m_shutdownInProgress;
    bool m_treatInvalidAsErrors;
    std::string m_lastErrorMessage;

    static ScriptManager* s_instance;
  };

#define g_luaScriptManager agxLua::ScriptManager::instance()


  class AGXLUA_EXPORT LuaCallback
  {
  public:
    LuaCallback(void* self, const char* className) : m_self(self), m_lua(g_luaScriptManager->getLua())
    {
      ScriptManager_REGISTER_OBJECT(self, className);
    }

    virtual ~LuaCallback()
    {
      ScriptManager_UNREGISTER_OBJECT(m_self);
    }
  protected:
    void* m_self;
    lua_State* m_lua;
  };


} // namespace agxLua



#if defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
#  if defined( AGX_LIBRARY_STATIC )
#    define AGXLUAPLUGIN_EXPORT
#  else
#    define AGXLUAPLUGIN_EXPORT   __declspec(dllexport)
#  endif
#elif !defined(_MSC_VER)
  // Non Win32
  #if __GNUC__ >= 4
    #define AGXLUAPLUGIN_EXPORT __attribute__ ((visibility("default")))
  #else
    #define AGXLUAPLUGIN_EXPORT
  #endif
#else
  #define AGXLUAPLUGIN_EXPORT
#endif


#endif

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
