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

#ifndef AGXIO_ENVIRONMENT_H
#define AGXIO_ENVIRONMENT_H

#include <agx/agxCore_export.h>
#include <agxIO/FilePathContainer.h>
#include <agx/Singleton.h>

namespace agxIO
{
  /**
  Class for setting search path for locating dynamic libraries, scripts etc.
  Environment has three paths, of which user should use: RESOURCE_PATH and RUNTIME_PATH

  RESOURCE_PATH - lists directories where AGX can find license files, kernel files, configuration files and scripts.
  BINARY_PATH - lists directories where AGX should search for collider plugins, kernel binaries etc.

  \code
  // Tell AGX where to find some additional scripts.
  agxIO::Environment::instance()->getFilePath(agxIO::Environment::RESOURCE_PATH).addFilePath("c:\\myScriptDir");

  // Point out where to find plugins
  agxIO::Environment::instance()->getFilePath(agxIO::Environment::RUNTIME_PATH).addFilePath("c:\\AgX\bin\x64\plugins");

  \endcode
  */
  class AGXCORE_EXPORT Environment : public agx::Singleton
  {
    public:

      /// Specifies which path to access
      enum Type {
        RESOURCE_PATH,      /**< Specifies paths for where to search for resource files such as scripts, images, models etc. */
        RUNTIME_PATH,       /**< Specifies paths for where to search for plugin's/runtime libraries */
        COMPONENT_DEV_PATH, /**< Specifies paths for where to search for component definitions and source code (only for developers) */

        NUM_TYPES
      };

      /**
      Constructor that will initialize the FilePathContainer with data:

      RESOURCE_PATH: Paths from the environment variable AGX_FILE_PATH
      RUNTIME_PATH: Paths from the environment variable PATH/LD_LIBRARY_PATH
      */
      Environment( );

      /// \return a pointer to the Environment singleton.
      static Environment* instance( void );

      /**
      \return reference to the FilePathContainer for finding files given the current setup filepath:s.
      */
      agxIO::FilePathContainer& getFilePath( Type t = RESOURCE_PATH );

      /**
      \return a const reference to the FilePathContainer for finding files given the current setup filepath:s.
      */
      const agxIO::FilePathContainer& getFilePath( Type t = RESOURCE_PATH ) const;

      DOXYGEN_START_INTERNAL_BLOCK()


      agx::String findComponent(const agx::String& path) const;

      /**
      This method will read settings from the registry and setup the runtime environment based on that.
      It will only be operative when agx is "packaged".
      \return true if reading from registry is successful, otherwise false (AGX not properly installed).
      */
      bool setupFromRegistry();

      SINGLETON_CLASSNAME_METHOD();
      DOXYGEN_END_INTERNAL_BLOCK()


    protected:
      void shutdown() override;

      agxIO::FilePathContainer m_filePathContainer[NUM_TYPES];
      static Environment* s_instance;
  };

/// Macro for easier access to the Environment singleton
#define AGX_ENVIRONMENT() (*agxIO::Environment::instance())
}

#endif

