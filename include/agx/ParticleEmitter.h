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


#ifndef AGX_PARTICLE_EMITTER_H
#define AGX_PARTICLE_EMITTER_H

#include <agx/Emitter.h>
#include <agx/ParticleSystem.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(ParticleEmitter);
  AGX_DECLARE_VECTOR_TYPES(ParticleEmitter);

  /**
   * Spawns new particles inside a given volume given to the Emitter.
   */
  class CALLABLE AGXPHYSICS_EXPORT ParticleEmitter : public agx::Emitter
  {
  public:
    typedef Event2<ParticleEmitter *, agxData::EntityRange> Event;
    Event emitEvent;

    AGX_DECLARE_POINTER_TYPES(DistributionModel);

  public:

    /**
    * Create a particle emitter coupled with a particle system
    */
    ParticleEmitter( ParticleSystem *target, Quantity quantity = QUANTITY_COUNT );

    /**
    Set the initial life-length of the particles.
    \param life - initial life length
    */
    void setLife( Real life );

    /**
    \return the initial life length given to particles by this emitter.
    */
    Real getLife() const;

    /**
    Set the initial color for the spawned particles
    \param color - Initial color
    */
    void setColor( const Vec4& color );

    /**
    \return the initial color of the spawned particles
    */
    const Vec4& getColor( ) const;

    /**
    Set if the emitter should set a moisture property for the created particles
    */
    void setEnableMoisture(bool enable);

    /*
    Get if the emitter should set a moisture property for the created particles
    */
    bool getEnableMoisture() const;

    /**
    Set the default value for the moisture property of created particles
    */
    void setDefaultMoisture(agx::Real defaultMoisture);

    /**
    Return the default value of the moisture property for created particles
    */
    agx::Real getDefaultMoisture() const;


    DOXYGEN_START_INTERNAL_BLOCK()

    AGXSTREAM_DECLARE_SERIALIZABLE( agx::ParticleEmitter );

    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    /// Default constructor
    ParticleEmitter();

    /// Default destructor
    virtual ~ParticleEmitter();

    virtual void emitBody(const Vec3& position,
                          const Quat& rotation,
                          const Vec3& velocity,
                          Emitter::DistributionModel *model) override;
    virtual bool preEmit() override;
    virtual void postEmit() override;
    virtual Real getDefaultElementQuantity(Quantity quantity) override;

  private:
    // Creates and initializes the parameters.
    void init(ParticleSystem *target);

  private:
    ParticleSystemRef m_particleSystem;
    bool m_isGranularSystem;
    ScalarParameterRef m_life;

    ScalarParameterRef m_particleColor;
    size_t m_startSize;

    bool m_enableMoistureProperty = false;
    agx::Real m_defaultMoisture   = 0;
  };


  class AGXPHYSICS_EXPORT ParticleEmitter::DistributionModel : public Emitter::DistributionModel
  {
  public:
    DistributionModel();
    DistributionModel(agx::Real particleRadius, agx::Material *particleMaterial, agx::Real probabilityWeight);

    /**
    Set the particle radius.
    */
    // void setParticleRadius(agx::Real radius);

    /**
    \return The particle radius.
    */
    agx::Real getParticleRadius() const;

    /**
    Set the particle material.
    */
    // void setParticleMaterial(agx::Material *material);

    /**
    \return The particle material.
    */
    agx::Material *getParticleMaterial();
    const agx::Material *getParticleMaterial() const;

    DOXYGEN_START_INTERNAL_BLOCK()

    AGXSTREAM_DECLARE_SERIALIZABLE( agx::ParticleEmitter::DistributionModel );

    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    virtual ~DistributionModel();
    virtual agx::Real calculateVolume() override;
    virtual agx::Real calculateMass() override;

  private:
    agx::Real m_particleRadius;
    agx::MaterialRef m_particleMaterial;
  };
}

#endif
