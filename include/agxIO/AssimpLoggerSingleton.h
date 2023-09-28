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

#pragma once

#include <agx/config/AGX_USE_ASSIMP.h>
#if AGX_USE_ASSIMP()

#include <agx/Singleton.h>
namespace agxIO
{

  /**
  This class will initialize the logging for the Assimp library as a singleton.
  It should be accessed through the instance method, which will initialize the logging of Assimp through
  AGX Dynamics LOGGER system.
  */
  class AGXPHYSICS_EXPORT AssimpLoggerSingleton : public agx::Singleton
  {

  public:

    /// Constructor
    AssimpLoggerSingleton();

    static AssimpLoggerSingleton* instance();

    SINGLETON_CLASSNAME_METHOD();

  protected:
    void shutdown() override;

  private:

    virtual ~AssimpLoggerSingleton();

    static AssimpLoggerSingleton* s_instance;
    agx::Vector<void*> m_logStreams;

  };
}

// AGX_USE_ASSIMP()
#endif
