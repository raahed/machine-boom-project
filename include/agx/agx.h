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

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
#endif

#ifndef AGX_AGX_H
#define AGX_AGX_H

#include <agx/config.h>
#include <agx/macros.h>
#include <agx/build_flags.h>

#include <agx/agxPhysics_export.h>
#include <agx/agxCore_export.h>
#include <string.h>

#include <agx/Math.h>
#include <agx/stdint.h>
#include <agx/debug.h>

// Workaround for a bug in Visual Studio 2008-2013 where a C2039 could be issued when referencing a function foo
// defined in the same unnamed namespace with ::foo.
// See http://stackoverflow.com/questions/5503901/how-to-correctly-reference-a-function-in-an-anonymous-namespace
namespace {}

/**
\namespace agx
\brief The agx namespace contains the dynamics/math part of the AGX Dynamics API.
*/
namespace agx
{
  /**
  Initialize agx. Must be called before the agx API is used.
  */
  void AGXPHYSICS_EXPORT init();

  /**
  Shutdown agx. Must be called before end of main()/unload of dll:s etc.
  */
  void AGXPHYSICS_EXPORT shutdown();

  /**
  \return true if shutdown is called
  */
  bool AGXPHYSICS_EXPORT isShutdown();

  /**
  Convenience class to automatically call agx::init / agx::shutdown. Usage:

  int main (int argc, char const *argv[])
  {
    agx::AutoInit agxInit;
    doStuffWithAgX();
    return 0;
  }
  */
  class AGXPHYSICS_EXPORT AutoInit
  {
  public:
    AutoInit();
    ~AutoInit();
  };



  /// Specifies flags set at build time. Used when querying what is enabled in this current build
  enum BuildConfiguration {
    USE_64BIT_ARCHITECTURE,
    SABRE_USE_METIS,
    USE_OPENGL_INSTANCING,
    USE_PREFETCHING,
    USE_SSE,
    SABRE_USE_SSE3,
    SABRE_USE_PADDING,
    USE_LUA,
    USE_TOLUA,
    USE_TINYXML,
    USE_OPENGL,
    USE_OSG,
    UNITTEST_ENABLED,
    USE_AGXSENSOR,
    USE_AGXCALLABLE,
    USE_PARTICLE_SYSTEM,
    USE_OPENCL,
    USE_DEBUG,
    USE_COMPOSITE,
    WINDOWS_PLATFORM,
    USE_ASSIMP,
    NUM_BUILD_CONFIGURATIONS
  };

  /// \return true if the specified build configuration is enabled at build time
  bool AGXCORE_EXPORT isBuiltWith(BuildConfiguration config);


  /**
  \param config - configuration to be translated to a string
  \return a string with the specified BuildConfiguration
  */
  std::string AGXCORE_EXPORT getBuildConfigurationString(BuildConfiguration config);

  /**
  Set the number of threads to use (including the main thread).
  \param numThreads - The total available number of threads. If 0, then all CPU:s/Cores will be used.
  */
  void AGXCORE_EXPORT setNumThreads(size_t numThreads);

  /**
  \return The number of threads including the main thread but not including user
          threads registered with agx::Thread::registerAsAgxThread.
  */
  agx::Index AGXCORE_EXPORT getNumThreads();

  /**
  \return The number of threads including the main thread and user threads
          registered with agx::Thread::registerAsAgxThread.
  */
  agx::Index AGXCORE_EXPORT getNumThreadsIncludingRegistered();

  class Thread;
  /**
  \return The current thread.
  */
  AGXCORE_EXPORT Thread*  getCurrentThread();


  /**
  Iff true you can move the entity to another thread (but you may only use it from one thread _at the time_).
  Iff false you may only use and destroy an Entity in the same thread it was created in (!!!) UNLESS you transfer ownership to a simulation.
  Default: false
  */
  AGXCORE_EXPORT void setEntityCreationThreadSafe( bool );

  /**
  \return true iff it is safe to transfer ownership of an object to another thread. Default: false.
  */
  AGXCORE_EXPORT bool getEntityCreationThreadSafe();

  /**
  \return The CPU tick value when agx was initialized.
  */
  UInt64 AGXCORE_EXPORT getStartTick();

  /**
  Reset the startTick value to the current tick.
  */
  void AGXCORE_EXPORT resetStartTick();

  /**
  Tag for invalid UInt values.
  E.g. agx::UInt index = agx::InvalidIndex;
  */
  struct AGXCORE_EXPORT InvalidIndexStruct {
    template <typename T>
    AGX_FORCE_INLINE operator T() const {
      return (T) - 1;
    }

    // Comparison operators
    template <typename T>
    AGX_FORCE_INLINE bool operator == (T val) const {
      return val == (T) - 1;
    }

    template <typename T>
    AGX_FORCE_INLINE bool operator != (T val) const {
      return val != (T) - 1;
    }
  };


  // Comparison operators
  template <typename T>
  AGX_FORCE_INLINE bool operator == (T val, InvalidIndexStruct)
  {
    return val == (T) - 1;
  }

  template <typename T>
  AGX_FORCE_INLINE bool operator != (T val, InvalidIndexStruct)
  {
    return val != (T) - 1;
  }

  extern AGXCORE_EXPORT const InvalidIndexStruct InvalidIndex;
}

/**
\namespace agxCore
\brief Contains function for initialization and shutdown of agxCore library when agxPhysics is not used.

NOTE: Only use these when using agxCore without agxPhysics
*/
namespace agxCore
{
  /**
  Deprecated, no need for explicit initialization.
  */
  void AGXCORE_EXPORT init();

  /**
  Shutdown the AGX API, after this call no AGX API calls should be done.
  */
  void AGXCORE_EXPORT shutdown();

  /**
  \return true if shutdown() is called.
  */
  bool AGXCORE_EXPORT isShutdown();
}

#ifdef _MSC_VER
#  pragma warning(pop)
#endif

#endif
