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

#ifndef AGXCOLLIDE_PLUGINMANAGER_H
#define AGXCOLLIDE_PLUGINMANAGER_H

#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()

#include <agx/agxPhysics_export.h>

#include <agx/Singleton.h>


#include <agx/Vector.h>
#include <agx/HashTable.h>
#include <agx/HashSet.h>

#include <string>

#include <agxCollide/agxCollide.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y

#else //assume POSIX...
#include <dlfcn.h>
#include <sys/types.h>
#include <dirent.h>
#endif


namespace agxCollide
{
  class ShapeCollider;
  class ColliderTable;
  class Shape;
  class ShapeColliderPlugin;


  typedef ShapeCollider* ( *ColliderConstructor )();
  typedef void ( *ColliderDestructor )( ShapeCollider * );

  //typedef agx::HashTable<int, ShapeCollider *> ColliderMap;

  /**
  Singleton class used by the collision engine to manage collision plugins.

  The plugin manager loads plugins (called colliders) and provides functionality
  to find the correct plugin for testing if a pair of shapes are colliding.
  */
  class AGXPHYSICS_EXPORT PluginManager : public agx::Singleton
  {
    public:
      static PluginManager *instance();

      void registerPlugin(ShapeColliderPlugin *plugin);

      SINGLETON_CLASSNAME_METHOD();

    protected:
      /**
      Destructor.
      */
      virtual ~PluginManager();

      /// Deallocates all memory used for this class (not the class itself).
      virtual void shutdown() override;

    private:

      /**
      Default constructor, can only be called from instance
      */
      PluginManager();


  protected:
    static PluginManager *s_instance;
    uint32_t m_pluginCount;
  };

  typedef agx::ref_ptr<PluginManager> PluginManagerRef;
}
DOXYGEN_END_INTERNAL_BLOCK()

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
