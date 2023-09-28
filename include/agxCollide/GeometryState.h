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

#include <agx/Math.h>
#include <agx/BitSet_small.h>

#include <agxData/Type.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxCollide
{

  class GeometryState : public agx::BitSet_small<agx::UInt32>
  {
  public:
    enum
    {
      ENABLE_BIT         = 0,
      REMOVED_BIT        = 1,
      CAN_COLLIDE_BIT    = 2,
      SENSOR_BIT         = 3,
      BOOLEAN_SENSOR_BIT = 4,
      ENABLE_MASSPROPERTIES = 5
    };

  public:
    GeometryState();
    GeometryState(agx::UInt32 mask);
    bool enabled() const;
    bool shouldCollide() const;
    bool removed() const;
    bool sensor() const;
    bool enableMassProperties() const;
    bool booleanSensor() const;
    bool sensorAndBooleanSensor() const;

  private:
    friend class Geometry;
    friend class Space;
    void setEnabled(bool flag);
    void setRemoved(bool flag);
    void setEnableMassProperties(bool flag);
    void setShouldCollide(bool flag);
    void setSensor(bool flag);
    void setBooleanSensor(bool flag);
  };


  /* Implementation */
  AGX_FORCE_INLINE GeometryState::GeometryState()
  {
  }

  AGX_FORCE_INLINE GeometryState::GeometryState(agx::UInt32 mask) : agx::BitSet_small<agx::UInt32>(mask)
  {
  }

  AGX_FORCE_INLINE bool GeometryState::enabled() const
  {
    return this->test(ENABLE_BIT);
  }

  AGX_FORCE_INLINE bool GeometryState::removed() const
  {
    return this->test(REMOVED_BIT);
  }

  AGX_FORCE_INLINE bool GeometryState::shouldCollide() const
  {
    return this->test(CAN_COLLIDE_BIT);
  }

  inline bool GeometryState::sensor() const
  {
    return this->test(SENSOR_BIT);
  }

  inline bool GeometryState::booleanSensor() const
  {
    return this->test(BOOLEAN_SENSOR_BIT);
  }

  inline bool GeometryState::enableMassProperties() const
  {
    return this->test(ENABLE_MASSPROPERTIES);
  }

  inline bool GeometryState::sensorAndBooleanSensor() const
  {
    return this->sensor() && this->booleanSensor();
  }

  AGX_FORCE_INLINE void GeometryState::setEnabled(bool flag)
  {
    this->set(ENABLE_BIT,flag);
  }

  AGX_FORCE_INLINE void GeometryState::setRemoved(bool flag)
  {
    this->set(REMOVED_BIT,flag);
  }

  AGX_FORCE_INLINE void GeometryState::setEnableMassProperties(bool flag)
  {
    this->set(ENABLE_MASSPROPERTIES, flag);
  }

  AGX_FORCE_INLINE void GeometryState::setShouldCollide(bool flag)
  {
    this->set(CAN_COLLIDE_BIT,flag);
  }

  inline void GeometryState::setSensor(bool flag)
  {
    this->set(SENSOR_BIT, flag);
  }

  inline void GeometryState::setBooleanSensor(bool flag)
  {
    this->set(BOOLEAN_SENSOR_BIT, flag);
  }
}

AGX_TYPE_BINDING(agxCollide::GeometryState, "GeometryState")

DOXYGEN_END_INTERNAL_BLOCK()

