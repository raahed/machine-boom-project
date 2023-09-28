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

#ifndef AGX_SINGLETONMANAGER_H
#define AGX_SINGLETONMANAGER_H

#include <agx/agxCore_export.h>
#include <agx/agx.h>
#include <agx/Referenced.h>
#include <agx/ThreadSynchronization.h>
#include <agx/HashSet.h>
#include <agx/DynamicLibrary.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agx
{

  class Singleton;


  /// Singleton Class for handling shutdown of registered singletons
  class AGXCORE_EXPORT SingletonManager : public agx::Referenced
  {
    public:

      /// Default constructor
      SingletonManager();

      /// \return instance to the static SingletonManager
      static SingletonManager* instance();

      /// Register a singleton to this SingletonManager
      void registerSingleton( Singleton* );

      bool unregisterSingleton( Singleton* singleton );

      /// Call shutdown for all registered singletons
      void shutdown();

      /// Print the registered Singleton to std::cerr in priority order
      void print( ) const;

      void registerDynamicLibrary(agx::DynamicLibrary*);

      void unregisterDynamicLibrary(agx::DynamicLibrary*);

    protected:
      virtual ~SingletonManager();

    private:
      typedef agx::Vector<Singleton*> SingletonQueue;
      SingletonQueue m_singletons;
      SingletonQueue m_addedSingletons;
      bool m_dirty;
      bool m_shutdownCalled;

      agx::HashSet< agx::ref_ptr<agx::DynamicLibrary> > m_libraries;

      Mutex m_mutex;
  };

  AGX_DECLARE_POINTER_TYPES(SingletonManager);
}
DOXYGEN_END_INTERNAL_BLOCK()


#endif
