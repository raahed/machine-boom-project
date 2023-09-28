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

#ifndef AGX_GRANULARIMPACTANALYSIS_H
  #define AGX_GRANULARIMPACTANALYSIS_H

#include <agx/agxPhysics_export.h>
#include <agx/Physics/GranularBodySystem.h>
#include <agx/Journal.h>
#include <agxSDK/StepEventListener.h>
#include <agxSDK/Simulation.h>
#include <agx/AnalysisBox.h>
#include <agx/BitState.h>
#include <iostream>

namespace agx
{
  // Detail namespace
  namespace detail
  {
    // Container for contact information
    struct ParticlePairContactInformation
    {
      agx::Physics::ParticlePairContactPtr     contactPtr;
    };

    // Container for contact information
    struct ParticleGeometryContactInformation
    {
      agx::Physics::ParticleGeometryContactPtr contactPtr;
    };

    // Container for information about the accumulated impacts experienced by a pellet.
    struct ParticleInformation
    {
      agx::Index      particleId;
      agx::Real       radius;
      agx::Real       accumulatedEnergy;
      agx::Vec3       position;
    };

    /**
    Particle EventListener that can be inherited and overridden in order to listen to particle events.
    */
    class AGXPHYSICS_EXPORT ParticleEventListener
    {
    public:
      // Constructor
      ParticleEventListener(agx::Physics::GranularBodySystem * system);

      /// Returns the currently destroyed particles registered from the coupled GranularBodySystem.
      IndexHashSet getDestroyedParticles() const;

    protected:
      virtual void onParticleDestroyed(agx::Index id) = 0;

      virtual void onParticleCreated(agx::Index id) = 0;

      bool particleIsDestroyed(agx::Index id);

      void clearDestroyedParticles();

    private:
      void triggerParticleDestroyed(agx::Index id);

      void triggerParticleCreated(agx::Index id);

      //////////////////////////////////////////////////////////////////////////
      // Internal classes
      //////////////////////////////////////////////////////////////////////////
      AGX_DECLARE_POINTER_TYPES(ParticleStorageListener);
      class ParticleStorageListener : public agxData::EntityStorage::EventListener, public agx::Referenced
      {
      public:
        ParticleStorageListener(ParticleEventListener* listener);

        virtual void destroyCallback(agxData::EntityStorage* storage) override;
        virtual void createInstanceCallback(agxData::EntityStorage* storage, agxData::EntityPtr instance) override;
        virtual void createInstancesCallback(agxData::EntityStorage* storage, agxData::EntityRange range) override;
        virtual void destroyInstanceCallback(agxData::EntityStorage* storage, agxData::EntityPtr instance) override;
        virtual void destroyInstancesCallback(agxData::EntityStorage* storage, agxData::Array<agxData::EntityPtr> instances) override;
        virtual void createInstancesCallback(agxData::EntityStorage* storage, agxData::Array<agxData::EntityPtr> instances) override;
        virtual void destroyInstancesCallback(agxData::EntityStorage* storage, agxData::EntityRange instances) override;

      protected:
        virtual ~ParticleStorageListener();

        //////////////////////////////////////////////////////////////////////////
        // Variables
        //////////////////////////////////////////////////////////////////////////
      protected:
        ParticleEventListener * m_listener;
      };

    protected:
      virtual ~ParticleEventListener();
      //////////////////////////////////////////////////////////////////////////
      // Variables
      //////////////////////////////////////////////////////////////////////////
    private:
      agx::Physics::GranularBodySystem*  m_system;
      ParticleStorageListenerRef         m_particleStorageListener;
      IndexHashSet                       m_destroyedIds;
    };

    AGX_FORCE_INLINE IndexHashSet agx::detail::ParticleEventListener::getDestroyedParticles() const { return m_destroyedIds; }
  }

  /**
  This class can be coupled to a simulation and is used to write contact information to a granular body system.
  It is both responsible for writing contact information from contacts exceeding a certain contact energy and
  also accumulating those energies on particles.
  */
  AGX_DECLARE_POINTER_TYPES(GranularImpactDataWriter);
  class AGXPHYSICS_EXPORT GranularImpactDataWriter : public agxSDK::StepEventListener
  {
  public:
#ifndef SWIG
    // Simulation buffer names for the data that the ImpactDataWriter records into the simulation
    static const char* const IMPACT_BUFFER;
    static const char* const ACCUMULATED_IMPACT_BUFFER;
    static const char* const PARTICLEPAIRCONTACT_BUFFER;
    static const char* const PARTICLEGEOMETRYCONTACT_BUFFER;
    static const char* const CONTACTENERGY_BUFFER;
#endif

  public:
    typedef agx::Physics::GranularBody::GranularGeometryPair                                 ParticleGeometryKey;
    typedef agx::Physics::GranularBody::GranularGranularPair                                 ParticlePairKey;
    typedef agx::HashTable<agx::Physics::GranularBody::GranularGeometryPair, agx::TimeStamp> ParticleGeometryTimeTable;
    typedef agx::HashTable<agx::Physics::GranularBody::GranularGranularPair, agx::TimeStamp> ParticlePairTimeTable;
    typedef agx::RealVector RealContainer;

    AGX_DECLARE_POINTER_TYPES(ImpactDataSimulationWriter);
  public:
    /**
    Constructor for the GranularImpactDataWriter.

    \param system The GranularBodySystem that the writer will write impact data too.
    \param recordThreshold The contact energy level threshold that the contact have to meet for the contact to be recorded.
    */
    GranularImpactDataWriter(agx::Physics::GranularBodySystem * system, agx::Real recordThreshold = 1E-3);

    /// Pre-step of the GranularImpactDataWriter
    virtual void pre(const agx::TimeStamp& /*t*/) override;

    /// Post-step of the GranularImpactDataWriter
    virtual void post(const agx::TimeStamp&) override;

    /**
    Called when this listener is added to the simulation.
    (given that it not already is in the simulation).
    */
    virtual void addNotification() override;

    /**
    Called when this listener is removed from the simulation.
    */
    virtual void removeNotification() override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agx::GranularImpactDataWriter);

  protected:
    virtual ~GranularImpactDataWriter();

  protected:


  private:
    GranularImpactDataWriter();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agxSDK::Simulation*                 m_simulation;
    agx::Physics::GranularBodySystem*   m_system;
    Real                                m_impactThreshold;
    ImpactDataSimulationWriterRef       m_impactDataSimulationWriter;
    IndexHashSet                        m_destroyedId;
    ParticleGeometryTimeTable           m_activeParticleGeometryPairs;
    ParticlePairTimeTable               m_activeParticlePairs;
  };

  /*
  Internal class that writes the contact and particle data to simulation under the specified GranularBodySystem.
  */
  class AGXPHYSICS_EXPORT GranularImpactDataWriter::ImpactDataSimulationWriter : public agx::Referenced
  {
  public:
    /**
    Constructor for the ImpactDataSimulationWriter.

    \param system The GranularBodySystem that the writer will write information to.
    */
    ImpactDataSimulationWriter(agx::Physics::GranularBodySystem * system);

    /// Function that will trigger when attached to a simulation
    void addNotification(agxSDK::Simulation * simulation);

    /// Function that will trigger when attached to a simulation
    void removeNotification(agxSDK::Simulation * simulation);

    /// Writes data from a contact to the granular body system
    void writeContactInformation(const detail::ParticlePairContactInformation& contact);

    /// Writes data from a contact to the granular body system
    void writeContactInformation(const detail::ParticleGeometryContactInformation& contact);

    /// Returns the particle pair impacts
    size_t getNumParticlePairImpacts() const;

    /// Returns the particle geometry impacts
    size_t getNumParticleGeometryImpacts() const;

    ImpactDataSimulationWriter& operator=(const ImpactDataSimulationWriter&) = delete;
  protected:
    virtual ~ImpactDataSimulationWriter();

    void initParticleBuffers();

    void journalAttachedCallback(agxSDK::Simulation * simulation, agx::Journal *journal);

    /// Initializes the journal bindings
    void createJournalBindings(agx::Journal * journal);

    void writeParticleContactInfo(agx::Index id, agx::Real contactEnergy);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Physics::GranularBodySystem*                    m_system;
    agxData::EntityStorageRef                            m_particlePairStorage;
    agxData::EntityStorageRef                            m_particleGeometryStorage;
    agxSDK::Simulation::JournalAttachEvent::CallbackType m_journalAttachedCallback;
  };
}

#endif
