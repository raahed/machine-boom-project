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

#ifndef AGX_CONFIGSINGLETON_H
#define AGX_CONFIGSINGLETON_H


#include <agx/agxPhysics_export.h>
#include <agx/Singleton.h>
#include <agxCFG/ConfigScript.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif

namespace agx
{

  /// Class for holding a reference to the Config db for the core settings defined in the file "settings.cfg"
  class AGXPHYSICS_EXPORT ConfigSingleton : public agx::Singleton
  {
    public:
      static void loadDefaultConfiguration();

    public:

      /// Constructor
      ConfigSingleton();

      /**
      A ConfigScript db used for various system settings of solver/Simulation/Space etc.
      \return a pointer to the ConfigScript that contains the settings
      */
      static agxCFG::ConfigScript* settings( void );

      /// \return a pointer to the ConfigScript that contains various user settings found in the default.cfg
      static agxCFG::ConfigScript* cfg( void );

      SINGLETON_CLASSNAME_METHOD();

    protected:
      void shutdown() override;

      agxCFG::ConfigScript* getSettingsCFG();

      agxCFG::ConfigScript* getDefaultCFG();

    private:
      agx::ref_ptr<agxCFG::ConfigScript> m_settings;
      agx::ref_ptr<agxCFG::ConfigScript> m_default;

      virtual ~ConfigSingleton();
      static ConfigSingleton* s_instance;

  };
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
