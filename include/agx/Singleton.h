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

#ifndef AGX_SINGLETON_H
#define AGX_SINGLETON_H

#include <agx/agxCore_export.h>
#include <agx/agx.h>
#include <agx/Referenced.h>
#include <agxStream/ClassInformation.h>


namespace agx
{

  /// Base class for Singletons that should have its shutdown called explicitly before exit of the application
  class AGXCORE_EXPORT Singleton
  {
    public:

      /// Specifies the ranges of the priority which can be assigned to a singleton at the time of creation.
      enum Priority {
        HIGHEST_PRIORITY = 2147483647, /**< The highest priority that can be given to a singleton */
        LOWEST_PRIORITY = 10000,       /**< The lowest priority that can be given to a singleton */
        DEFAULT_PRIORITY = 1073741824  /**< The default priority given to a singleton */
      };

      /**
      Default constructor.
      \param priority - The higher priority, the earlier it will get its shutdown executed during application shutdown.
      So if there is someone which should be shutdown early, use a high priority. Usually default should work.
      */
      Singleton( int priority = DEFAULT_PRIORITY );

      /// Implement this method to cleanup your Singleton class.
      virtual void shutdown() {}

      /// Return the shutdown priority of the singleton instance
      int getPriority() const;

      friend class SingletonManager;

      virtual std::string getClassName() const = 0;

#ifdef SWIG
# define SINGLETON_CLASSNAME_METHOD()
#else
# define SINGLETON_CLASSNAME_METHOD() \
  std::string getClassName() const override { agxStream::ClassInformation ci(AGX_FUNCTION); return ci.getClassType(); }
#endif

    protected:

      /// Destructor
      virtual ~Singleton();

      void _shutdown();

    private:
      int m_priority;
      bool m_shutdownCalled;
  };

  typedef ref_ptr<Singleton> SingletonRef;

}

#endif
