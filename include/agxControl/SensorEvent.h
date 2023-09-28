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

#ifndef AGXCONTROL_SENSOREVENT_H
#define AGXCONTROL_SENSOREVENT_H

#include <agx/config.h>

#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agx/Name.h>
#include <agx/TimeStamp.h>
#include <agxCollide/Contacts.h>
#include <agxStream/Serializable.h>
#include <agxCollide/Geometry.h>
#include <agx/Physics/ParticleGeometryContactEntity.h>
#include <agx/Physics/ParticleEntity.h>

namespace agxSDK
{
  class Simulation;
}

namespace agxControl
{
  class EventSensor;

  AGX_DECLARE_POINTER_TYPES(SensorEvent);
  AGX_DECLARE_VECTOR_TYPES(SensorEvent);

  /**
  A sensor event is an event triggered when the sensor is activated by a contact. The sensor passes contact information
  to the sensor event that executes operations on the colliding geometry.
  */
  class AGXPHYSICS_EXPORT SensorEvent : public agx::Referenced, public agxStream::Serializable
  {

  public:

    /// Default Constructor
    SensorEvent( const agx::Name& name = agx::Name());

    /**
    \return The name of the sensorEvent.
    */
    const agx::Name& getName() const;

    /**
     Triggers the sensor event by passing contact information
    */
    virtual void triggerEvent(const agx::TimeStamp& t,
      agxSDK::Simulation* simulation,
      agxCollide::GeometryContact* cd,
      EventSensor* sensor);

    /**
     Triggers the sensor event by passing particle contact information
    */
    virtual void triggerParticleEvent(const agx::TimeStamp& t,
      agxSDK::Simulation* simulation,
      agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData,
      agx::Physics::GeometryData& geometryData,
      EventSensor* sensor);

    /**
    * Post function that will be triggered for the event after all the contacts have handled in the operation
    */
    virtual void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor);

    /**
    * Function will be executed when the sensor that the operations is added to is added to a simulation. If the sensor
    * is already added, this function will be triggered when the operation is added.
    */
    virtual void addNotification(EventSensor * sensor);

    /**
    * Used to update the sensor event each time step, if they have time dependent components
    */
    virtual void update(const agx::TimeStamp& t);

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::SensorEvent);

    friend class EventSensor;

  protected:
    /// Destructor
    virtual ~SensorEvent();

  protected:
    /// Name of the sensor event
    agx::Name m_name;
    /// The priority of the sensor event
    int priority;

  };

  AGX_DECLARE_POINTER_TYPES(SensorOperation);
  AGX_DECLARE_VECTOR_TYPES(SensorOperation);
  /**
  A specific type of sensor event that triggers a change in the simulation when activated by
  a contact.
  */
  class AGXPHYSICS_EXPORT SensorOperation : public SensorEvent
  {
  public:
    /// Default constructor
    SensorOperation( const agx::Name& name = agx::Name());

    /**
     Triggers the sensor event by passing contact information
    */
    virtual void triggerEvent(const agx::TimeStamp& t, agxSDK::Simulation * simulation,  agxCollide::GeometryContact *cd, EventSensor * sensor) override;

    /**
     Triggers the sensor event by passing particle contact information
    */
    virtual void triggerParticleEvent(const agx::TimeStamp& t,
      agxSDK::Simulation* simulation,
      agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData,
      agx::Physics::GeometryData& geometryData,
      EventSensor* sensor) override;

    /**
    * Post function that will be triggered for the event after all the contacts have handled in the operation
    */
    virtual void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor) override;

    /**
    * Function will be executed when the sensor that the operations is added to is added to a simulation. If the sensor
    * is already added, this function will be triggered when the operation is added.
    */
    virtual void addNotification(EventSensor * sensor) override;

    /**
    * Used to update the sensor operation each time step, if they have time dependent components
    */
    virtual void update(const agx::TimeStamp& t) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::SensorOperation);

  protected:
    /// Destructor
    virtual ~SensorOperation();

  };
}

#endif /*SENSOREVENT_H_*/

