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

#ifndef AGX_REFERENCEDMANAGER_H
#define AGX_REFERENCEDMANAGER_H

#include <agx/config/AGX_DEBUG_TRACK_REFERENCED_OBJECTS.h>

#if AGX_DEBUG_TRACK_REFERENCED_OBJECTS()
#include <agx/agxPhysics_export.h>



// hashtable would be better but causes problems with includes
#include <map>
#include <iostream>

namespace agx {


  class AGXPHYSICS_EXPORT ReferencedManager {

    public:
      static ReferencedManager* instance();

      inline void ref() const
      {
        _refCount++;
      }

      inline void unref() const
      {
        _refCount--;
        if ( _refCount == 0 )
        {
          delete this;
          _refCount = 0;
        }
      }

      inline void add(const void *p)
      {
        ++m_added;
        if ( m_referencedObjects.find(p) != m_referencedObjects.end() )
        {
          std::cout << "Duplicate: " << p << std::endl;
        }
        m_referencedObjects.insert( std::make_pair(p, 1) );
      }

      inline void remove(const void *p)
      {
        ++m_removed;
        m_referencedObjects.erase(p);
      }

      void report();

    protected:
      ReferencedManager() : m_added(0), m_removed(0) {}
    private:
      ~ReferencedManager();
      int m_added;
      int m_removed;
      mutable int                     _refCount;
      // void*                          _observers;

      std::map<const void*,int> m_referencedObjects;
  };

}


#endif
#endif

