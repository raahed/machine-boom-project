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

#ifndef AGXLUA_LUASCRIPTOPERATION_H
  #define AGXLUA_LUASCRIPTOPERATION_H

#include <agx/config/AGX_USE_LUA.h>
#include <fstream>
#include <agxLua/export.h>
#include <agxControl/SensorEvent.h>
#include <agxControl/EventSensor.h>

#if AGX_USE_LUA()

/*
This file defines some example sensor operations
*/
namespace agxLua
{
  /**
  * This sensor operation executes a specified lua script string to manipulate the colliders,
  * which allows the user to create custom operations through scripting.
  */
  AGX_DECLARE_POINTER_TYPES(LuaScriptOperation);
  class AGXLUA_EXPORT LuaScriptOperation : public agxControl::SensorOperation
  {
  public:

    LuaScriptOperation(const agx::Name& name = agx::Name());
    LuaScriptOperation(const agx::Name& name, const agx::String& script, const agx::String& functionIdentifier="");

    virtual void triggerParticleEvent(const agx::TimeStamp& t,
      agxSDK::Simulation* simulation,
      agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData,
      agx::Physics::GeometryData& geometryData,
      agxControl::EventSensor* sensor) override;

    virtual void triggerEvent(const agx::TimeStamp& t,
      agxSDK::Simulation * simulation,
      agxCollide::GeometryContact * cd,
      agxControl::EventSensor* sensor) override;

    virtual void postContactHandling(agxSDK::Simulation * simulation, agxControl::EventSensor *sensor) override;

    void doInit(agxSDK::Simulation * simulation);

    AGXSTREAM_DECLARE_SERIALIZABLE(agxLua::LuaScriptOperation);

  protected:
    virtual ~LuaScriptOperation();

    void doErrorHandling(const agx::String& function, const agx::String& lastError);

  protected:
    agx::String m_script;
    agx::String m_functionIdentifier;
    bool m_initiated;
  };
}

#endif

#endif
