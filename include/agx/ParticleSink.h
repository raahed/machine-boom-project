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

#ifndef AGX_PARTICLESINK_H
#define AGX_PARTICLESINK_H

#include <agx/ParticleContactSensor.h>

namespace agxSDK
{
  class Simulation;
}


namespace agx
{
  class ParticleSystem;

  AGX_DECLARE_POINTER_TYPES(ParticleSink);
  AGX_DECLARE_VECTOR_TYPES(ParticleSink) ;

  /// A sink which destroys particles.
  class AGXPHYSICS_EXPORT ParticleSink : public agx::ParticleContactSensor
  {
  public:
    ParticleSink();
    /** Creates particle sink with a geometry for destruction on contact.
    \param geometry The particles in contact with this geometry will be destroyed.
    */
    ParticleSink(agxCollide::Geometry* geometry);

    virtual void contactCallback(agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData, agx::Physics::GeometryData& geometryData) override;

    AGXSTREAM_DECLARE_SERIALIZABLE( agx::ParticleSink );

  protected:
    virtual ~ParticleSink();
  };
}


#endif /* AGX_PARTICLESINK_H */
