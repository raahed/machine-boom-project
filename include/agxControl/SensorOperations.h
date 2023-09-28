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

#ifndef AGXCONTROL_SENSOROPERATIONS_H
#define AGXCONTROL_SENSOROPERATIONS_H

#include <fstream>
#include <agx/Date.h>
#include <agx/String.h>
#include <agxData/Track.h>
#include <agxControl/SensorEvent.h>
#include <agxControl/EventSensor.h>
#include <agx/Physics/GranularBodySystem.h>
#include <agx/Journal.h>
#include <agxRender/Color.h>

/*
This file defines some example sensor operations
*/
namespace agxControl
{
  /// This operation removes the rigid body given by the geometry in the contact supplied by the sensor
  class AGXPHYSICS_EXPORT RemoveRigidBody : public agxControl::SensorOperation
  {
  public:
    RemoveRigidBody(const agx::Name& name = agx::Name(), bool onlyRemoveEmittedBodies = false);

    void triggerEvent(const agx::TimeStamp& /*t*/, agxSDK::Simulation * simulation, agxCollide::GeometryContact *cd, agxControl::EventSensor * sensor) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::RemoveRigidBody);

  protected:
    virtual ~RemoveRigidBody();

  protected:
    bool m_onlyRemoveEmittedBodies;
  };

  /// This operations removes all the particles in the contact supplied by the sensor
  class AGXPHYSICS_EXPORT RemoveParticles : public agxControl::SensorOperation
  {
  public:
    RemoveParticles(const agx::Name& name = agx::Name());

    void triggerParticleEvent(const agx::TimeStamp& /*t*/,
      agxSDK::Simulation* /*simulation*/,
      agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData,
      agx::Physics::GeometryData& /*geometryData*/,
      agxControl::EventSensor* /*sensor*/) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::RemoveParticles);

  protected:
    virtual ~RemoveParticles();

  };

  /// This operation can be used to know when a sensor has been activated by a contact
  class CALLABLE AGXPHYSICS_EXPORT TagOperation : public agxControl::SensorOperation
  {
  public:
    TagOperation(const agx::Name& name = agx::Name());


    virtual void triggerEvent(const agx::TimeStamp& /*t*/, agxSDK::Simulation * /*simulation*/, agxCollide::GeometryContact * /*cd*/, agxControl::EventSensor* /*sensor*/) override
    {
      // Tag this operation as executed
      m_isExecuted = true;
    }

    virtual void triggerParticleEvent(const agx::TimeStamp& /*t*/,
      agxSDK::Simulation* /*simulation*/,
      agx::Physics::ParticleGeometryContactInstance /*contact*/,
      agx::Physics::ParticleData& /*particleData*/,
      agx::Physics::GeometryData& /*geometryData*/,
      agxControl::EventSensor* /*sensor*/) override
    {
      m_isExecuted = true;
    }

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::TagOperation);

  public:
    bool isExecuted() { return m_isExecuted; }

    void setIsExecuted(bool executed) { m_isExecuted = executed; }

  protected:
    bool m_isExecuted;

  protected:
    virtual ~TagOperation();

  };

  /// This operation changes the color of the particles given in the contact
  class AGXPHYSICS_EXPORT ColorParticles : public agxControl::SensorOperation
  {
  private:

    static agx::HashSet<agx::UInt32> s_coloredParticles;
    static agx::TimeStamp s_lastTimeStep;

  public:
    ColorParticles(const agx::Name& name = agx::Name(), const agx::Vec3& color = agx::Vec3(0.5, 0.5, 0.5), bool mixColors = false);

    virtual void triggerParticleEvent(const agx::TimeStamp& t,
      agxSDK::Simulation* simulation,
      agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData,
      agx::Physics::GeometryData& geometryData,
      agxControl::EventSensor* sensor) override;

    bool getEnableMixColors(){ return m_mixColors; }
    void setEnableMixColors(bool mixColors){ m_mixColors = mixColors; }

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::ColorParticles);

  public:
    agx::Vec3& getColor(){ return m_color; }

  protected:
    agx::Vec3 m_color;
    bool m_mixColors;

  protected:
    virtual ~ColorParticles();

  };

  /// This operation dumps particle contact information to a file
  class AGXPHYSICS_EXPORT ParticleContactDumper : public agxControl::SensorOperation
  {
  public:

    ParticleContactDumper(const agx::Name& name = agx::Name());
    ParticleContactDumper(const agx::Name& name, const agx::Name& filepath, agx::UInt32 samplingStride = 1, bool exportOnce = false);

    void createFileName(const agx::Name& filePath);

    // This function appends a date stamp at the end of the filename
    agx::String generateDateStamp();

    virtual void triggerParticleEvent(const agx::TimeStamp& t,
      agxSDK::Simulation* /*simulation*/,
      agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData,
      agx::Physics::GeometryData& /*geometryData*/,
      agxControl::EventSensor* /*sensor*/) override;

    agx::Name& getFilePath(){ return m_filepath; }

    virtual void update(const agx::TimeStamp& t) override;

    void postContactHandling(agxSDK::Simulation* simulation, EventSensor* sensor) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::ParticleContactDumper);

  protected:
    agx::HashTable<agx::Index, agx::Bool> m_handledParticleIds;
    std::ofstream m_outStream;
    agx::Name     m_filepath;
    agx::Name     m_completeFilePath;
    agx::Name     m_fileFormat;
    agx::UInt32   m_samplingStride;
    agx::UInt32   m_framesSinceLastSampling;
    bool          m_recordFrame;
    bool          m_exportOnlyOnce;

  protected:
    virtual ~ParticleContactDumper();

  };

  // Switches the material of the particle in the supplied contact
  class AGXPHYSICS_EXPORT ReplaceParticleMaterialOperation : public agxControl::SensorOperation
  {
  public:

    ReplaceParticleMaterialOperation(const agx::Name& name = agx::Name());
    ReplaceParticleMaterialOperation(const agx::Name& name, const agx::Name& materialName);

    virtual void triggerParticleEvent(const agx::TimeStamp& t,
      agxSDK::Simulation* simulation,
      agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData,
      agx::Physics::GeometryData& geometryData,
      agxControl::EventSensor* sensor) override;

    void triggerEvent( const agx::TimeStamp& t,
      agxSDK::Simulation * simulation,
      agxCollide::GeometryContact * cd,
      agxControl::EventSensor* sensor ) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::ReplaceParticleMaterialOperation);

    void setMaterialName(const agx::String& material);
    agx::Name getMaterialName();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Name m_materialName;

  protected:
    virtual ~ReplaceParticleMaterialOperation();

  };

  AGX_FORCE_INLINE void ReplaceParticleMaterialOperation::setMaterialName(const agx::String& material)
  {
    m_materialName = material;
  }

  AGX_FORCE_INLINE agx::Name ReplaceParticleMaterialOperation::getMaterialName()
  {
    return m_materialName;
  }


  /**
  * This operation applies a "kinematic state" to particles that collide with the sensor. Is also able to "unfreeze" kinematic particles
  */
  class AGXPHYSICS_EXPORT MakeParticleKinematicOperation : public agxControl::SensorOperation
  {
    friend class ReplaceParticleMaterialOperation;
    friend class RemoveParticles;
  public:

    /// Enum describing what mode the operation should be in. If it should freeze or unfreeze particles.
    enum ParticleFreezeMode
    {
      FREEZE_PARTILCES,
      UNFREEZE_PARTICLES
    };

    MakeParticleKinematicOperation(const agx::Name& name = agx::Name(), ParticleFreezeMode mode = FREEZE_PARTILCES);

    virtual void triggerParticleEvent(const agx::TimeStamp& t,
      agxSDK::Simulation* simulation,
      agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData,
      agx::Physics::GeometryData& geometryData,
      agxControl::EventSensor* sensor) override;

    void triggerEvent( const agx::TimeStamp& /*t*/,
                       agxSDK::Simulation * simulation,
                       agxCollide::GeometryContact *cd,
                       agxControl::EventSensor * sensor ) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::MakeParticleKinematicOperation);

  protected:
    virtual ~MakeParticleKinematicOperation();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
    int m_mode;
  };

  /**
  This operation applies a kinematic "sleep state" to particles inside the sensor that are
  under a specified velocity threshold. Optional particle coloring when sleep is triggered.
  */
  class AGXPHYSICS_EXPORT ParticleKinematicSleepOperation : public agxControl::SensorOperation
  {
    friend class ReplaceParticleMaterialOperation;
    friend class RemoveParticles;
  public:

    ParticleKinematicSleepOperation( const agx::Name& name = agx::Name(),
                                     agx::Real sleepSpeedThreshold = 1.0,
                                     bool colorOnSleep = false,
                                     agx::Vec4f sleepColor = agxRender::Color::Yellow() );

    virtual void triggerParticleEvent( const agx::TimeStamp& t,
      agxSDK::Simulation* simulation,
      agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData,
      agx::Physics::GeometryData& geometryData,
      agxControl::EventSensor* sensor ) override;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::ParticleKinematicSleepOperation );

  protected:
    virtual ~ParticleKinematicSleepOperation();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
    agx::Real  m_sleepSpeedThreshold;
    bool       m_colorOnSleep;
    agx::Vec4f m_sleepColor;
  };

  /**
  * This sensor operation class sets a speed to the object that collides with the sensor, in both linear and angular direction
  */
  class AGXPHYSICS_EXPORT SetVelocityOperation : public agxControl::SensorOperation
  {
  public:

    SetVelocityOperation(const agx::Name& name = agx::Name());
    SetVelocityOperation(const agx::Name& name, const agx::Vec3& velocity, const agx::Vec3& angularVelocity, bool setInLocalFrame = false);

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

    void setVelocity(const agx::Vec3& velocity);
    const agx::Vec3& getVelocity() const;

    void setAngularVelocity(const agx::Vec3& angularVelocity);
    const agx::Vec3& getAngularVelocity() const;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::SetVelocityOperation);

  protected:
    virtual ~SetVelocityOperation();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Vec3 m_velocity;
    agx::Vec3 m_angularVelocity;
    bool      m_setInLocalFrame;
    agx::FrameRef m_localFrame;
  };

  AGX_FORCE_INLINE void SetVelocityOperation::setVelocity(const agx::Vec3& velocity)
  {
    m_velocity = velocity;
  }

  AGX_FORCE_INLINE const agx::Vec3& SetVelocityOperation::getVelocity() const
  {
    return m_velocity;
  }

  AGX_FORCE_INLINE void SetVelocityOperation::setAngularVelocity(const agx::Vec3& angularVelocity)
  {
    m_angularVelocity = angularVelocity;
  }

  AGX_FORCE_INLINE const agx::Vec3& SetVelocityOperation::getAngularVelocity() const
  {
    return m_angularVelocity;
  }

  /**
  * This sensor operation class store particle id's of particles that are in contact with the sensor
  */
  class AGXPHYSICS_EXPORT StoreParticlesOperation : public agxControl::SensorOperation
  {
  public:

    StoreParticlesOperation(const agx::Name& name = agx::Name());

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

    virtual void update(const agx::TimeStamp& t) override;

    agx::Physics::ParticlePtrVector getParticlesInContact() const;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::StoreParticlesOperation);

  protected:
    virtual ~StoreParticlesOperation();

  protected:
    agx::Physics::ParticlePtrVector m_particlesInContact;
  };

  AGX_FORCE_INLINE agx::Physics::ParticlePtrVector StoreParticlesOperation::getParticlesInContact() const { return m_particlesInContact; }

  /**
  * This sensor operation can teleport contacting particles to a target sensor geometry with arbitrary time dilation
  */
  class AGXPHYSICS_EXPORT TeleportToSensorOperation : public agxControl::SensorOperation
  {
  public:

    typedef std::pair<agx::TimeStamp, agx::Index> TimeParticleIdPair;
    typedef agx::List<TimeParticleIdPair>         ParticleTeleportQueue;

    TeleportToSensorOperation(const agx::Name& name,
                              EventSensor * target,
                              agx::Real teleportTimeDilation = 0,
                              bool cloneTeleportation = false,
                              agx::Frame * sourceoffset = nullptr,
                              agx::Frame * targetoffset = nullptr);

    void addNotification(EventSensor * sensor) override;

    void triggerEvent(const agx::TimeStamp& t,
      agxSDK::Simulation * simulation,
      agxCollide::GeometryContact * cd,
      agxControl::EventSensor* sensor) override;

    void triggerParticleEvent(const agx::TimeStamp& t,
      agxSDK::Simulation* /*simulation*/,
      agx::Physics::ParticleGeometryContactInstance contact,
      agx::Physics::ParticleData& particleData,
      agx::Physics::GeometryData& /*geometryData*/,
      agxControl::EventSensor* sensor) override;

    void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::TeleportToSensorOperation);

  protected:
    TeleportToSensorOperation(const agx::Name& name = agx::Name());
    void init();

    void teleportParticles(const agx::TimeStamp& t, agxControl::EventSensor * sensor, agx::Physics::GranularBodySystem * system);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Real                 m_teleportTimeDilation;
    agxControl::EventSensor * m_targetSensor;
    ParticleTeleportQueue     m_particleSpawnQueue;
    agxData::EntityStorageRef m_particleTeleportStorage;
    bool                      m_shouldCloneTeleport;
    agx::FrameRef             m_targetOffsetFrame;
    agx::FrameRef             m_sourceOffsetFrame;
    agx::HashSet<agx::UInt32> m_previousContactingParticleIds;
  };
}

#endif
