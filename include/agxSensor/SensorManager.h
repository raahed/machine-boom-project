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
#ifndef AGXSENSOR_SENSORMANAGER
#define AGXSENSOR_SENSORMANAGER

#include <agxSensor/export.h>
#include <agx/Singleton.h>

DOXYGEN_START_INTERNAL_BLOCK()
namespace OIS
{
  class InputManager;
}
DOXYGEN_END_INTERNAL_BLOCK()

namespace agxSensor
{

  /// Class which controls all sensors, initialization, destruction etc.
  class AGXSENSOR_EXPORT SensorManager : public agx::Singleton
  {
  public:
    SINGLETON_CLASSNAME_METHOD();

    SensorManager();
    virtual void shutdown() override;

    /**
    Initializes the SensorManager with a valid window pointer. A device will only
    react if this window is the current one (at least on windows).
    Under all other platforms mWin can be 0.
    */
    bool init( agx::Int mWin );

    /**
    \return a pointer to the internal OIS InputManager
    */
    OIS::InputManager* getInputManager();

    /// \return a pointer to the SensorManager singleton.
    static SensorManager* instance();

    static bool hasShutdown();

  protected:
    virtual ~SensorManager();

  private:
    OIS::InputManager* m_InputManager;
    static SensorManager* s_ois;
    static bool s_hasShutdown;
  };
}

/// Utility macro to access the input manager more easily.
#define OIS_MANAGER() agxSensor::SensorManager::instance()->getInputManager()

#endif
