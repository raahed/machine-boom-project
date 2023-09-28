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

#ifndef AGX_PARTICLECONTACTSENSOR_H
#define AGX_PARTICLECONTACTSENSOR_H

#include <agxCollide/Geometry.h>
#include <agx/Physics/ParticleGeometryContactEntity.h>
#include <agx/Physics/ParticleEntity.h>
#include <agx/Physics/GeometryEntity.h>
#include <agxStream/Serializable.h>

namespace agxSDK
{
  class Simulation;
}


namespace agx
{
  class ParticleSystem;

  AGX_DECLARE_POINTER_TYPES(ParticleContactSensor);
  AGX_DECLARE_VECTOR_TYPES(ParticleContactSensor) ;

  /**
  This class is for detecting and handling particle collisions with geometries in the simulation
  */
  class AGXPHYSICS_EXPORT ParticleContactSensor : public Object, public agxStream::Serializable
  {
  public:
    ParticleContactSensor();
    ParticleContactSensor(agxCollide::Geometry* geometry);

    void setGeometry(agxCollide::Geometry* geometry);

    agxCollide::Geometry* getGeometry();
    const agxCollide::Geometry* getGeometry() const;

    agxSDK::Simulation* getSimulation();
    const agxSDK::Simulation* getSimulation() const;

    virtual void contactCallback(agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData, agx::Physics::GeometryData& geometryData);

    AGXSTREAM_DECLARE_SERIALIZABLE( agx::ParticleContactSensor );

  protected:
    virtual ~ParticleContactSensor();

  private:
    friend class agxSDK::Simulation;
    void setSimulation(agxSDK::Simulation* simulation);

  private:
    agxCollide::GeometryRef m_geometry;
    agxSDK::Simulation* m_simulation;
  };

  /* Implementation */
  AGX_FORCE_INLINE agxCollide::Geometry* ParticleContactSensor::getGeometry()
  {
    return m_geometry;
  }

  AGX_FORCE_INLINE const agxCollide::Geometry* ParticleContactSensor::getGeometry() const
  {
    return m_geometry;
  }

  AGX_FORCE_INLINE agxSDK::Simulation* ParticleContactSensor::getSimulation()
  {
    return m_simulation;
  }

  AGX_FORCE_INLINE const agxSDK::Simulation* ParticleContactSensor::getSimulation() const
  {
    return m_simulation;
  }

}


#endif /* AGX_PARTICLECONTACTSENSOR_H */
