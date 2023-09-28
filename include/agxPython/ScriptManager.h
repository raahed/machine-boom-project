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

#ifndef AGXPYTHON_SCRIPTMANAGER_H
#define AGXPYTHON_SCRIPTMANAGER_H

#include <agx/config/AGX_USE_PYTHON.h>
#include <agx/config/AGX_USE_WEBSOCKETS.h>
#include <agx/config.h>

#if AGX_USE_PYTHON()

#include <agx/ref_ptr.h>
#include <agxPython/export.h>

#include <agx/Referenced.h>
#include <agx/Logger.h>
#include <agx/Timer.h>
#include <agx/agx_vector_types.h>
#include <agx/Uuid.h>

#if AGX_USE_WEBSOCKETS()
#include <agxSDK/CallbackWrappers.h>
#endif

#include <agxPython/ScriptContext.h>
#include <agxPython/ReactiveScript.h>


#ifdef _MSC_VER
# pragma warning( push )
# pragma warning( disable: 4512 ) // Disable warning about assignment operator could not be generated due to reference members.
# pragma warning( disable: 4275 ) // Disable non dll-interface class std::exception used as base for dll-interface class agxPython::ScriptError
# endif

#include <agxPython/ScriptIDE.h>

#include <string>
#include <map>
#include <set>
#include <exception>

#if AGX_USE_WEBSOCKETS()
namespace agxSDK
{
  class SimulationController;
}
#endif

namespace agxPython {

  enum ScriptContextType
  {
    SCRIPTCONTEXT_INVALID = 0,
    SCRIPTCONTEXT_PYTHON,
    SCRIPTCONTEXT_AGXVIEWER,
    SCRIPTCONTEXT_FMI,
    SCRIPTCONTEXT_QTVIEWER
  };

  /**
  Exception base class for all fatal runtime errors caused by
  Python scripts. Thrown only by the ScriptManager, whose source
  file also implements multiple derived ScriptError classes for
  each main error type. Catching exceptions of this base type
  is enough wherever error handling of script execution should
  happen. It derives from STLs std::exception if any handler
  wish to rethrow the exception upwards (e.g for logging or
  other purposes)
  */
  class AGXPYTHON_EXPORT ScriptError : public std::exception
  {

  public:

    ScriptError();
    ScriptError(const ScriptError& other);
    ScriptError(std::string scriptError, std::string scriptFile);

    virtual ~ScriptError();

    virtual const char* what() const AGX_NOEXCEPT override; //=0;

  private:

    std::string m_msg;

  };


#if AGX_USE_WEBSOCKETS()
  AGX_DECLARE_POINTER_TYPES(StartPlaybackSignalCallback);
  class AGXPYTHON_EXPORT StartPlaybackSignalCallback : public agxSDK::CallbackWrapper_Real
  {
  public:
    StartPlaybackSignalCallback() : agxSDK::CallbackWrapper_Real() {}

  protected:

    virtual ~StartPlaybackSignalCallback() {}

  private:
    void callback(agxSDK::SimulationController*, agx::Real value) override;
  };

  AGX_DECLARE_POINTER_TYPES(StartRecordingSignalCallback);
  class AGXPYTHON_EXPORT StartRecordingSignalCallback : public agxSDK::CallbackWrapper_Real
  {
  public:
    StartRecordingSignalCallback() : agxSDK::CallbackWrapper_Real() {}

  protected:

    virtual ~StartRecordingSignalCallback() {}

  private:
    void callback(agxSDK::SimulationController*, agx::Real value) override;
  };

  AGX_DECLARE_POINTER_TYPES(StopPlaybackSignalCallback);
  class AGXPYTHON_EXPORT StopPlaybackSignalCallback : public agxSDK::CallbackWrapper_Real
  {
  public:
    StopPlaybackSignalCallback() : agxSDK::CallbackWrapper_Real() {}

  protected:

    virtual ~StopPlaybackSignalCallback() {}

  private:
    void callback(agxSDK::SimulationController*, agx::Real value) override;
  };

  AGX_DECLARE_POINTER_TYPES(StopRecordingSignalCallback);
  class AGXPYTHON_EXPORT StopRecordingSignalCallback : public agxSDK::CallbackWrapper_Real
  {
  public:
    StopRecordingSignalCallback() : agxSDK::CallbackWrapper_Real() {}

  protected:

    virtual ~StopRecordingSignalCallback() {}

  private:
    void callback(agxSDK::SimulationController*, agx::Real value) override;
  };
#endif

  class ReactiveScript;
  class ScriptContext;

  /**
  Class that manages the embedded Python interpreter and
  a script context.
  */
  class AGXPYTHON_EXPORT ScriptManager : public agx::Singleton
  {

  public:

    friend class ReactiveInitializer;

    /**
    \return the current Python script context
    */
    static ScriptContext* getContext();

    static bool hasInstance();
    static bool isStartedFromExternalPython();

    /**
    Return the singleton object
    If ignoreEnvironment is true then environment variables and site packages will be ignored
    */
    static ScriptManager* instance(const std::string& requestor = "", const std::string& pythonPaths="", const std::string& pythonHome="", bool ignoreEnvironment = false);

    /// Deallocates any allocated memory
    void shutdown() override;

    /**
    Fetches an error from the script context and call setLastErrorMessage for storing it.
    \param prefix - A string that will begin the error message
    */
    void registerError(const std::string& prefix = "");

    /**

    */
    bool breakFlag();

    /**
    Prints an error report, e.g from an exception, to the appropriate error stream
    */
    void reportError(const std::string& errorReport) const;

    /**
    Appends the full path to the directory to the Python sys.path list
    of paths to search for importable modules.
    \param path - A ut8 string containing a path to a directory.
    \return true if path is ut8 and to a directory and sys.path object exists
    */
    bool addSearchPath(const agx::String& path);

    /**
    Register the given ReactiveScript to the ScriptManager
    for Simulation-driven execution using per-Script storage
    of context data about runtime state, unhandled errors and
    the access pointer for the ReactiveScript API to interface
    with the underlying agxSDK::Simulation containing the
    ReactiveScript.

    Registering a ReactiveScript makes it accessible via its
    Uuid and prepares the internal context data with a clean
    runtime state.

    */
    bool registerReactiveScript(ReactiveScript* script);
    bool unregisterReactiveScript(const agx::Uuid& uuid);
    bool isRegisteredReactiveScript(const agx::Uuid& uuid) const;
    void clearReactiveScripts();
    void acquireReactiveScriptContext(const agx::Uuid& scriptId, const char* func);
    void releaseReactiveScriptContext(const agx::Uuid& scriptId, bool unlock = true);
    void addScriptToSimulation(const agx::Uuid& scriptId, agxSDK::Simulation* simulation);

#if AGX_USE_WEBSOCKETS()
    void updateReactiveScripts(agxSDK::SimulationController* simulationController);
#endif
    void updateReactiveScripts();

    void onStartPlaybackReactiveScripts(agx::Real value);
    void onStartRecordingReactiveScripts(agx::Real value);
    void onStopPlaybackReactiveScripts(agx::Real value);
    void onStopRecordingReactiveScripts(agx::Real value);


    /**
    Get the last stored error message
    \return A string containing the last stored error message
    */
    std::string getLastErrorMessage() const;

    /**
    Get the line of the last stored error message
    \return -1 if none, otherwise the line of the last stored error message
    */
    int getLastErrorLine() const;

    /**
    Get if there has been an error since last reset.
    \return true if there has been an error, otherwise false.
    */
    bool receivedErrorMessage() const;

    /**
    After errors been handled, reset the error message status.
    */
    void resetError();

    /**
    Store an error message
    */
    void setLastErrorMessage(const std::string& errorMessage);
    void setLastErrorLine(int line);

    /**
    Return a ScriptIDE ref_ptr to the ScriptIDE with the given name,
    creating it if no ScriptIDE with such a name already exists. ScriptIDE
    instances represent a virtual IDE in order to help with writing code
    editors tailored for writing agxPython scripts, providing facilities such
    as a debugger and support SDK for source code navigation and autocompletion.
    */
    ScriptIDERef requestVirtualIDE(const agx::Name& identifyingName);

   /**
   This is a way to set the simulation for momentum api if not using it
   through reactive scripts.
   */
    void setCurrentScriptSimulationOverride(agxSDK::Simulation* simulation);

#if AGX_USE_WEBSOCKETS()
    void registerSimulationControllerCallbacks(agxSDK::SimulationController* simulationController);
    void unregisterSimulationControllerCallbacks(agxSDK::SimulationController* simulationController);
#endif

    // Destroy the main module for the given script.
    void destroyMainModule(const agx::Uuid& scriptId);

#if !defined(SWIGJAVA)

    void enableEmbeddedMode(const std::string& programName);

    /// Initializes the Python interpreter state
    void init(const std::string& pythonPaths="");

    /// Resets the Python interpreter state and reinitializes it
    void reset();

    /// Initialize the script context in which scripts should execute and return
    /// the interface object to be used for setting up the script environment.
    agxPython::ScriptContextInterface *initScriptContext(agxPython::ScriptContextType contextType);

    /// Get the script context interface object to an already initialized
    /// script context.
    agxPython::ScriptContextInterface* getScriptContext();

    // Create a separate main module for the given script.
    void createMainModule(const agx::Uuid& scriptId);

    // Swap to the main module for the given script.
    void swapMainModule(const agx::Uuid& scriptId);

    // Swap to the global main module.
    void restoreGlobalMainModule(bool unlock = true);

    bool hasMainModule(const agx::Uuid& uuid) const;

    void* getMainModule();

    /**
    Checks if the stored reactive scripts have errors.
    \param printWarnings - true if warnings should be printed to LOGGER_WARNING, false if not.
    \return true if there are errors in the active reactive scripts, false otherwise.
    */
    bool hasReactiveScriptErrors( bool printWarnings = true );

    /**
    Load a Python script file. The file will be searched for using agxIO::Environment settings.
    \param path - full or partial path to a python script file
    \param resolvePath - resolve path using AGX resource lookup
    \return true if file was found and loaded successfully
    */
    bool doFile(const std::string& path, bool resolvePath = true);

    /**
    Run a string containing Python code. Returns true upon success
    \param code - String with python code
    \param origin - An identification string shown if an error occurs
    */
    bool doString(const std::string& code, const std::string& origin);


    /**
    Run an already compiled script as a function global to the __main__
    module using the given locals dictionary. This dictionary of locals
    will contain all declarations and objects the script defines, useful
    for measuring validity of state post-initialization.
    */
    bool doCompiledScript(void* compiled);
    void* doCompiledFunction(void* funcObject, void **args, int numArgs);

    /**
    Invoke a global function
    \param function - name of the function to be called
    \param args - Arguments to the function
    \return true if the call to the function was successful
    */
    bool doFunction(const std::string& function, void* args = nullptr);



    agxSDK::Simulation* getCurrentScriptSimulation();

#if AGX_USE_WEBSOCKETS()
    agxSDK::SimulationController* getCurrentScriptSimulationController();
#endif


    /**
    Check if global function exists
    \param function - name of a function that we will be looking for
    \return true if the function exists
    */
    bool hasFunction(const std::string& function);

    bool getTreatInvalidAsErrors() const;
    void setTreatInvalidAsErrors(const bool treatInvalidAsErrors);


    /**
    Returns the script's "__main__" module __dict__ dictionary, equivalen t
    to the globals() builtin function in Python, as a void-cast PyObject*
    pointer, valid only between calls to ScriptManager::doString or
    ScriptManager::doFile (which calls doString).

    Py_XINCREF is used on the pointer before returning, caller is
    responsible to use Py_XDECREF when done with it. Using Py_XDECREF on NULL
    pointers is completely valid (it does check internally), so no need to check
    for NULL.

    Use "PyDict_"-family of Python C API functions to access it with keys guaranteed
    to be Python string objects.

    Calling code will need to include Python.h for the Py_XDECREF macro and
    "PyDict_"-family of functions.
    */
    void* getGlobals();

    /**
    Get the name of the program that instantiated the ScriptManager
    */
    std::string getProgramName();


  protected:

    typedef std::map<std::string, std::string> PythonDefMap;

    PythonDefMap dumpGlobals(void* globals, std::ostream* logStream = nullptr);

#endif

  public:

    /// Internal method
    static void startProfiling(void* obj, const char* decl);
    /// Internal method
    static void endProfiling(void* obj, const char* decl);
    /// Internal method
    static void abortProfiling(void* obj, const char* decl);


  private:

    SINGLETON_CLASSNAME_METHOD();

    /// Destructor
    virtual ~ScriptManager();

    /// Constructor
    ScriptManager(const std::string& program, const std::string& pythonPaths, const std::string& pythonHome, bool ignoreEnvironment);

    agx::ref_ptr<ScriptContext> m_scriptContext;

    std::map<agx::Uuid, agx::ref_ptr<ReactiveScript> > m_reactiveScripts;
    std::map<agx::Uuid, void*> m_reactiveScriptMainModules;

    void* m_globalMainModule;
    void* m_currentMainModule;
    ReactiveScript* m_currentReactiveScriptContext;

    agxSDK::Simulation* m_currentScriptSimulationOverride;

    bool m_initialized;
    bool m_breakFlag;
    std::string m_programName;
    std::set<std::string> m_pythonPath;
    wchar_t* m_pythonHome;
    bool m_shutdownInProgress;
    bool m_treatInvalidAsErrors;
    std::string m_lastErrorMessage;
    int m_lastErrorLine;
    std::string m_scriptPath;
    bool m_ignoreEnvironment;

    agx::Vector<ScriptIDERef> m_ideInstances;


#if AGX_USE_WEBSOCKETS()
    StartPlaybackSignalCallbackRef m_startPlaybackSignalCallback;
    StartRecordingSignalCallbackRef m_startRecordingSignalCallback;
    StopPlaybackSignalCallbackRef m_stopPlaybackSignalCallback;
    StopRecordingSignalCallbackRef m_stopRecordingSignalCallback;


    agxSDK::SimulationController* m_simulationController;
#endif

    agx::ReentrantMutex m_mutex;

    static ScriptManager* s_instance;

    /**
    Variable to hold the Algoryx Momentum program name
    */
    const char* const MOMENTUM_PROGRAM_NAME = "AlgoryxMomentum";

    friend class ScriptIDEAutocompleteCollection;

  };

  AGXPYTHON_EXPORT ScriptContext* getContext();

  AGXPYTHON_EXPORT bool scriptIsAttached();

  AGXPYTHON_EXPORT bool doFile(const std::string& path);

} // namespace agxPython


#define g_pythonScriptManager agxPython::ScriptManager::instance("")
#define g_pythonScriptManagerForProgram(PROGRAM) agxPython::ScriptManager::instance(PROGRAM)
#define AGXPYTHON_REPORT_ERROR(errorMessage)                                 \
  if (g_pythonScriptManager->getTreatInvalidAsErrors() || agxUnit::isUnittestingEnabled()) \
  {                                                                       \
    LOGGER_ERROR() << (errorMessage) << std::endl << LOGGER_END();        \
  }                                                                       \
  else                                                                    \
  {                                                                       \
    LOGGER_WARNING() << (errorMessage) << std::endl << LOGGER_END();      \
  }


#endif

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
