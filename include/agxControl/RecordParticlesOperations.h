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

#ifndef AGXCONTROL_RECORDPARTICLESOPERATIONS_H
#define AGXCONTROL_RECORDPARTICLESOPERATIONS_H

#include <agx/String.h>
#include <agxControl/EventSensor.h>
#include <agx/Journal.h>

#ifndef SWIG
namespace agx
{
  class Uuid;
}
#endif

/*
This file contains SensorOperation classes that gives the sensor functionality to record colliding particles to separate journals.
The operations can be used for recording continuous flow of particles in a sensor or store single frame snapshots/states. The
created journals can later be loaded into the simulation by other operations.
This creates an easy way to create and load different types of precomputed particle states into other simulations.
*/
namespace agxControl
{

  /**
  * Base virtual class for storing particle states to an external journal file. When inserted into a sensor, the class will create
  * a storage node in the active simulation for particle states in the sensor, which is then recorded to an attached journal.
  * Derived classes implemented features for either continuous particles flows or discrete snapshots.
  */
  class AGXPHYSICS_EXPORT BaseRecordParticleJournalOperation : public agxControl::SensorOperation
  {
  public:
    BaseRecordParticleJournalOperation(const agx::Name& name, const agx::String& journalPath, agx::Frame * sourceOffset = nullptr);

    virtual void update(const agx::TimeStamp&) override;

    virtual void addNotification(EventSensor * sensor) override;

    virtual void triggerEvent(const agx::TimeStamp&,
      agxSDK::Simulation * ,
      agxCollide::GeometryContact *,
      agxControl::EventSensor*) override {}

    virtual void triggerParticleEvent(const agx::TimeStamp& ,
      agxSDK::Simulation* ,
      agx::Physics::ParticleGeometryContactInstance ,
      agx::Physics::ParticleData& ,
      agx::Physics::GeometryData& ,
      agxControl::EventSensor*) override {}

      virtual void postContactHandling(agxSDK::Simulation *, EventSensor *) override {}

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::BaseRecordParticleJournalOperation);

  protected:
    BaseRecordParticleJournalOperation(const agx::Name& name = agx::Name());
    virtual ~BaseRecordParticleJournalOperation();

    void attachJournal(agxSDK::Simulation *simulation);

    void detachJournal();

    void initDataStructure(agxSDK::Simulation * simulation);

    void copyContactingParticlesToStorage(EventSensor * sensor);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Uuid                 m_journalUuid;
    agx::JournalRef           m_journal;
    agx::String               m_journalPath;
    agxData::EntityStorageRef m_particleStorage;
    agx::HashSet<agx::UInt32> m_previousContactingParticleIds;
    agx::FrameRef             m_sourceOffsetFrame;
  };



  /**
  * Class that stores particle data from a continuous flow in the coupled sensor to a separate specified journal file.
  */
  class AGXPHYSICS_EXPORT RecordParticleJournalOperation : public agxControl::BaseRecordParticleJournalOperation
  {
  public:
    RecordParticleJournalOperation(const agx::Name& name, const agx::String& journalPath, bool shouldAttachJournalOnContact = true, agx::Frame * sourceOffset = nullptr);

    void addNotification(EventSensor * sensor) override;

    void triggerEvent(const agx::TimeStamp& /*t*/,
      agxSDK::Simulation * /*simulation*/,
      agxCollide::GeometryContact * /*cd*/,
      agxControl::EventSensor* /*sensor*/) override;

    void triggerParticleEvent(const agx::TimeStamp& /*t*/,
      agxSDK::Simulation* simulation,
      agx::Physics::ParticleGeometryContactInstance /*contact*/,
      agx::Physics::ParticleData& /*particleData*/,
      agx::Physics::GeometryData& /*geometryData*/,
      agxControl::EventSensor* /*sensor*/) override;

    void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::RecordParticleJournalOperation);

  protected:
    RecordParticleJournalOperation(const agx::Name& name = agx::Name());
    virtual ~RecordParticleJournalOperation();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    bool                      m_shouldAttachJournalOnContact;
  };



  /**
  * Class that stores a single frame of the current particle configuration in the coupled sensor
  */
  class AGXPHYSICS_EXPORT RecordParticleShapshotJournalOperation : public BaseRecordParticleJournalOperation
  {
  public:
    RecordParticleShapshotJournalOperation(const agx::Name& name, const agx::String& journalPath, agx::Real snapShotTime = 0, agx::Frame * sourceOffset = nullptr);

    virtual void update(const agx::TimeStamp& /*t*/) override;

    virtual void addNotification(EventSensor * sensor) override;

    void triggerEvent(const agx::TimeStamp& /*t*/,
      agxSDK::Simulation * /*simulation*/,
      agxCollide::GeometryContact * /*cd*/,
      agxControl::EventSensor* /*sensor*/) override;

    void triggerParticleEvent(const agx::TimeStamp& /*t*/,
      agxSDK::Simulation* simulation,
      agx::Physics::ParticleGeometryContactInstance /*contact*/,
      agx::Physics::ParticleData& /*particleData*/,
      agx::Physics::GeometryData& /*geometryData*/,
      agxControl::EventSensor* /*sensor*/) override;

    void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::RecordParticleShapshotJournalOperation);

  protected:
    RecordParticleShapshotJournalOperation(agx::Name name = agx::Name());
    virtual ~RecordParticleShapshotJournalOperation();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Real                 m_snapShotTime;
    bool                      m_hasTakenSnapshot;
  };



  /**
  * Loads particle data from an external journal file into a simulation relative to the coupled sensor object. Can load journals
  * created by both the continuous- and snapshot operations.
  */
  class AGXPHYSICS_EXPORT LoadParticleJournalOperation : public agxControl::SensorOperation
  {
  public:
    LoadParticleJournalOperation( const agx::Name& name,
                                  const agx::String& journalPath,
                                  agx::Real startAfter = 0,
                                  agx::Material* startMaterial = nullptr,
                                  agx::Frame * targetOffsetFrame = nullptr);

    void addNotification(EventSensor * sensor) override;

    void triggerEvent(const agx::TimeStamp& /*t*/,
      agxSDK::Simulation * /*simulation*/,
      agxCollide::GeometryContact * /*cd*/,
      agxControl::EventSensor* /*sensor*/) override;

    void triggerParticleEvent(const agx::TimeStamp& /*t*/,
      agxSDK::Simulation* /*simulation*/,
      agx::Physics::ParticleGeometryContactInstance /*contact*/,
      agx::Physics::ParticleData& /*particleData*/,
      agx::Physics::GeometryData& /*geometryData*/,
      agxControl::EventSensor* /*sensor*/) override;

    void postContactHandling(agxSDK::Simulation * simulation, agxControl::EventSensor *sensor) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::LoadParticleJournalOperation);

  protected:
    LoadParticleJournalOperation(const agx::Name& name = agx::Name());
    virtual ~LoadParticleJournalOperation();

    void loadAndAttachJournal(EventSensor * sensor);

    bool loadParticleStorageFromSimulation(EventSensor * sensor);

    void loadParticlesFromStorage(EventSensor * sensor);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::JournalRef                 m_journal;
    agx::String                     m_journalPath;
    agxData::EntityStorageRef       m_particleStorage;
    bool                            m_hasLoadedJournal;
    agx::Real                       m_startAfter;
    agx::Material*                  m_startMaterial;
    agx::FrameRef                   m_targetOffsetFrame;
  };



  /**
  * Class that stores particles in contact with the sensor to an binary .agx file.
  */
  class AGXPHYSICS_EXPORT WriteParticleDataToAGXOperation : public agxControl::SensorOperation
  {
  public:
    WriteParticleDataToAGXOperation( const agx::Name& name,
                                     const agx::String& filepath,
                                     agx::Real snapShotTime,
                                     agx::Frame * sourceOffset = nullptr );

    virtual void update( const agx::TimeStamp& ) override;

    virtual void addNotification( EventSensor * sensor ) override;

    virtual void triggerEvent( const agx::TimeStamp&,
                               agxSDK::Simulation *,
                               agxCollide::GeometryContact *,
                               agxControl::EventSensor* ) override {}

    virtual void triggerParticleEvent( const agx::TimeStamp&,
                                       agxSDK::Simulation*,
                                       agx::Physics::ParticleGeometryContactInstance,
                                       agx::Physics::ParticleData&,
                                       agx::Physics::GeometryData&,
                                       agxControl::EventSensor* ) override {}

    void writeParticlesToFile( EventSensor * sensor );

    virtual void postContactHandling( agxSDK::Simulation *, EventSensor * ) override;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::WriteParticleDataToAGXOperation );

  protected:
    WriteParticleDataToAGXOperation( const agx::Name& name = agx::Name() );
    virtual ~WriteParticleDataToAGXOperation();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::String    m_filePath;
    agx::Real      m_snapShotTime;
    agx::FrameRef  m_sourceOffsetFrame;
    bool           m_hasTakenSnapshot;
  };



  /**
  * Loads particle data from an external agx file into a simulation relative to the coupled sensor object.
  */
  class AGXPHYSICS_EXPORT LoadParticleAGXFileOperation : public agxControl::SensorOperation
  {
  public:
    LoadParticleAGXFileOperation( const agx::Name& name,
                                  const agx::String& filePath,
                                  agx::Real startAfter = 0,
                                  agx::Material* startMaterial = nullptr,
                                  agx::Frame * targetOffsetFrame = nullptr );

    void addNotification( EventSensor * sensor ) override;

    void triggerEvent( const agx::TimeStamp& /*t*/,
                       agxSDK::Simulation * /*simulation*/,
                       agxCollide::GeometryContact * /*cd*/,
                       agxControl::EventSensor* /*sensor*/ ) override {}

    void triggerParticleEvent( const agx::TimeStamp& /*t*/,
                               agxSDK::Simulation* /*simulation*/,
                               agx::Physics::ParticleGeometryContactInstance /*contact*/,
                               agx::Physics::ParticleData& /*particleData*/,
                               agx::Physics::GeometryData& /*geometryData*/,
                               agxControl::EventSensor* /*sensor*/ ) override {}

    void postContactHandling( agxSDK::Simulation * simulation, agxControl::EventSensor *sensor ) override;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::LoadParticleAGXFileOperation );

  protected:
    LoadParticleAGXFileOperation( const agx::Name& name = agx::Name() );
    virtual ~LoadParticleAGXFileOperation();

    void loadParticlesFromFile( EventSensor * sensor );

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::String                     m_filePath;
    agx::Real                       m_startAfter;
    agx::Material*                  m_startMaterial;
    agx::FrameRef                   m_targetOffsetFrame;
    bool                            m_hasLoadedJournal;
  };

}
#endif /*SENSOROPERATIONS_H_*/
