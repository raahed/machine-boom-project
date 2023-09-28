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

#ifndef AGXCONTROL_GRANULARMODELREDUCTIONOPERATIONS_H
#define AGXCONTROL_GRANULARMODELREDUCTIONOPERATIONS_H

#include <agxControl/SensorEvent.h>
#include <agx/Physics/GranularBodySystem.h>

/*
This file defines some operations used in conjunction with the granular model reduction
*/
namespace agxControl
{
  class AGXPHYSICS_EXPORT SplitSensorOperation : public agxControl::SensorOperation
  {
  public:
    SplitSensorOperation(const agx::Name& name = agx::Name());

    void triggerEvent(const agx::TimeStamp& /*t*/, agxSDK::Simulation * simulation, agxCollide::GeometryContact *cd, agxControl::EventSensor * sensor) override;

    void triggerParticleEvent(const agx::TimeStamp& /*t*/,
      agxSDK::Simulation* /*simulation*/,
      agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData,
      agx::Physics::GeometryData& /*geometryData*/,
      agxControl::EventSensor* /*sensor*/) override;

    void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::SplitSensorOperation);

  protected:
    class ParticleStorageListener : public agxData::EntityStorage::EventListener
    {
      friend class SplitSensorOperation;
    public:
      ParticleStorageListener(SplitSensorOperation * op);

      virtual void destroyInstanceCallback(agxData::EntityStorage* storage, agxData::EntityPtr instance) override;
      virtual void destroyInstancesCallback(agxData::EntityStorage* storage, agxData::Array<agxData::EntityPtr> instances) override;
      virtual void destroyInstancesCallback(agxData::EntityStorage* storage, agxData::EntityRange instance) override;

      //////////////////////////////////////////////////////////////////////////
      // Variables
      //////////////////////////////////////////////////////////////////////////
    protected:
      SplitSensorOperation * m_splitOperation;
    };
    friend class ParticleStorageListener;

  protected:
    virtual ~SplitSensorOperation();

    void connectToParticleStorage(agx::ParticleSystem* system);

    void singalParticleDestroyed(agx::Index id);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::HashSet<agx::Index> m_previousParticleIdsInSensor;
    ParticleStorageListener  m_particleStorageListener;
    agx::ParticleSystem*     m_particleSystem;
  };
}

#endif
