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


#ifndef AGX_GRANULARDATAEXPORTER_H
#define AGX_GRANULARDATAEXPORTER_H

#include <agx/agxPhysics_export.h>
#include <agx/Physics/GranularBodySystem.h>
#include <agxSDK/StepEventListener.h>
#include <agxSDK/Simulation.h>
#include <agx/IndexSet.h>
#include <agx/AnalysisBox.h>
#include <agx/GranularImpactDataWriter.h>
#include <iostream>

namespace agx
{
  /**
    Class used to extract data from a granular journal, in the form of writing contact and particle data to file.
    This is currently used in the QT post process viewer to export contact impact data for post processing in external tools.

    Usage:

    Connect the GranularDataExporter to a simulation by adding it to a simulation.

    impactExporter = new agx::GranularDataExporter(system);
    impactExporter->setExporterType(exportType);
    impactExporter->writeToFile(writeFileName);
    simulation->add(impactExporter);

    impactExporter->init()

    <Do simulation stepping...>

    imapctExporter->finalizeAfterSimulation()

  */
  class AGXPHYSICS_EXPORT GranularDataExporter : public agxSDK::StepEventListener, public detail::ParticleEventListener
  {
  public:
    // Forward declarations
    AGX_DECLARE_POINTER_TYPES(IDataFileWriter);
    AGX_DECLARE_POINTER_TYPES(IDataExporter);
    AGX_DECLARE_POINTER_TYPES(ContactDataExporter);
    AGX_DECLARE_POINTER_TYPES(ParticleDataExporter);
    AGX_DECLARE_POINTER_TYPES(ContactDataWriter);
    AGX_DECLARE_POINTER_TYPES(ParticleDataWriter);

  public:
    /// Export Data Type
    enum ExporterDataType
    {
      PARTICLE_DATA,
      CONTACT_DATA
    };

    /// Convert export data type to filename
    static agx::String convertExportDataTypeToFilename(ExporterDataType exportDataType);

  public:
    /**
    Constructor of GranularDataExporter
    \param system the GranularBodySystem that the exporter will take information from.
    */
    GranularDataExporter(agx::Physics::GranularBodySystem* system);

    /**
    Initializes the exporter. Needs to be called before the exporter is run.
    */
    void init();

    /// Pre step function
    virtual void pre(const agx::TimeStamp& /*t*/) override;

    /// Post step function
    virtual void post(const agx::TimeStamp&) override;

    /**
    Sets and creates a file with the given path where the exporter writes information.
    \param file the complete path to the file that should be created.
    */
    void writeToFile(const agx::String& file);

    /**
    Sets the AnalysisBox that should be used by the exporter. If the box is enabled, only objects within the
    bound of the specified AnalysisBox will be exported.
    \param box The AnalysisBox object that should be used when determining what objects that should be exported.
    */
    void setAnalysisBox(agx::AnalysisBox* box);

    /**
    Sets the exporter type, i.e which type of data that should be exported. Currently supported data is particles and
    particle impact contacts stored in the journal under GranularBodySystem.
    \param type Determines the type of data that should be exported.
    */
    void setExporterType(ExporterDataType type);

    /**
    Function that is called manually after the simulation is done. Finalizes the data export.
    */
    void finalizeAfterSimulation();

  protected:
    virtual ~GranularDataExporter();

    virtual void onParticleCreated(agx::Index id) override;

    virtual void onParticleDestroyed(agx::Index id) override;

    void updateStartingContactIds();

    /// Returns the indices for the particle-geometry impact contacts currently stored in the storage.
    IndexHashSet getStartingParticleGeometryImpacts();

    /// Returns the indices for the particle-particle impact contacts currently stored in the storage.
    IndexHashSet getStartingParticlePairImpacts();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Physics::GranularBodySystem*  m_system;
    IDataExporterRef                   m_dataExporter;
    IndexHashSet                       m_startingParticleGeometryImpactContacts;
    IndexHashSet                       m_startingParticlePairImpactContacts;
  };

  AGX_FORCE_INLINE IndexHashSet agx::GranularDataExporter::getStartingParticleGeometryImpacts()
  {
    return m_startingParticleGeometryImpactContacts;
  }

  AGX_FORCE_INLINE IndexHashSet agx::GranularDataExporter::getStartingParticlePairImpacts()
  {
    return m_startingParticlePairImpactContacts;
  }

  /**
  Abstract class for data exporters used in the GranularDataExporter. Is inherited by child classes exporting
  a certain set of data. Every exporter writes their data to a file.
  */
  class AGXPHYSICS_EXPORT GranularDataExporter::IDataExporter : public agx::Referenced
  {
  public:
    /// Constructor
    IDataExporter(agx::Physics::GranularBodySystem * system);

    /**
    Sets the analysis box to be used in the exporter. If the analysis box is enabled, only data entities placed inside the
    box will be exported.
    \param box The analysis box to be used in the exporter.
    */
    void setAnalysisBox(agx::AnalysisBox* box);

    /**
    Manually called by the exporter parent when performing pre export step.
    \param destroyedParticles Particles destroyed in the current time step of the exporter.
    */
    void exportPreStep(const IndexHashSet& destroyedParticles);

    /**
    Manually called by the exporter parent when performing pre export step.
    \param destroyedParticles Particles destroyed in the current time step of the exporter.
    */
    void exportPostStep(const IndexHashSet& destroyedParticles);

    /**
    Callback that is called when a particle is destroyed
    \param particleId - Particle destroyed in the current time step of the exporter.
    */
    virtual void handleParticleDestroyed(agx::Index particleId);

    /**
    Called during the last steps of the export
    \param exporter The exporter object.
    */
    void finalizeExport(GranularDataExporter * exporter);

    /**
    Sets and creates a file with the given path where the exporter writes information.
    \param filename the complete path to the file that should be created.
    */
    virtual void createExportFile(const agx::String& filename) = 0;

    /**
    Returns the type of the exporter, which is overridden by child classes.
    Each exporter type exports different information.
    \return The exporter type.
    */
    virtual GranularDataExporter::ExporterDataType getType() const = 0;

  protected:
    virtual ~IDataExporter();

    virtual void _finalizeExport(GranularDataExporter * exporter) = 0;

    virtual void _exportPreStep(const IndexHashSet& destroyedParticles) = 0;

    virtual void _exportPostStep(const IndexHashSet& destroyedParticles) = 0;

    ///
    bool hasAnalysisBox() const;

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Physics::GranularBodySystem*  m_system;
    IDataFileWriterRef                 m_fileWriter;
    agx::AnalysisBox *                 m_analysisBox;
    IndexHashSet                       m_exportedParticles;
  };


  DOXYGEN_START_INTERNAL_BLOCK()
  AGX_FORCE_INLINE bool GranularDataExporter::IDataExporter::hasAnalysisBox() const { return m_analysisBox != nullptr; }
  DOXYGEN_END_INTERNAL_BLOCK()

  /**
  ContactDataExporter exports the impact contacts stored in the GranularBodySystem written by the GranularImpactDataWriter.
  It writes this contact information to a file.
  */
  class AGXPHYSICS_EXPORT GranularDataExporter::ContactDataExporter : public GranularDataExporter::IDataExporter
  {
  public:
    /// Default constructor
    ContactDataExporter(agx::Physics::GranularBodySystem * system);

    /// Is called when a particle is destroyed in the simulation.
    virtual void handleParticleDestroyed(agx::Index particleId) override;

    /**
    Creates the file which the contact data should be written to.
    \param filename the name of the that should be create for the impact data.
    */
    virtual void createExportFile(const agx::String& filename) override;

    /**
    Returns the type of the exporter, which is overridden by child classes.
    Each exporter type exports different information.
    \return The exporter type, which in this case is CONTACT_DATA.
    */
    virtual GranularDataExporter::ExporterDataType getType() const override;

  protected:
    void exportContactInformation(const detail::ParticlePairContactInformation& information);

    void exportContactInformation(const detail::ParticleGeometryContactInformation& information);

  protected:
    virtual ~ContactDataExporter();

    virtual void _finalizeExport(GranularDataExporter * exporter) override;

    virtual void _exportPreStep(const IndexHashSet& destroyedParticles) override;

    virtual void _exportPostStep(const IndexHashSet& destroyedParticles) override;
  };

  /**
  ParticleDataExporter that exports particle data to a file.
  */
  class AGXPHYSICS_EXPORT GranularDataExporter::ParticleDataExporter : public GranularDataExporter::IDataExporter
  {
  public:
    /// Constructor
    ParticleDataExporter(agx::Physics::GranularBodySystem * system);

    /**
    Called when a particle is destroyed.
    \param particleId ID of the particle that is destroyed.
    */
    virtual void handleParticleDestroyed(agx::Index particleId) override;

    /**
    Creates the file which the particle data should be written to.
    \param filename the name of the that should be create for the particle data.
    */
    virtual void createExportFile(const agx::String& filename) override;

    /**
    Returns the type of the exporter, which is overridden by child classes.
    Each exporter type exports different information.
    \return The exporter type, which in this case is PARTICLE_DATA.
    */
    virtual GranularDataExporter::ExporterDataType getType() const override;

  protected:
    virtual ~ParticleDataExporter();

    agx::Real getAccumulatedImpactEnergyFromParticle(agx::Index particleId);

    virtual void _finalizeExport(GranularDataExporter * exporter) override;

    virtual void _exportPreStep(const IndexHashSet& destroyedParticles) override;

    virtual void _exportPostStep(const IndexHashSet& destroyedParticles) override;

    void writeParticleInformation(const detail::ParticleInformation& information);

    void updateParticleEnergyTable();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    typedef agx::HashTable<agx::Index, detail::ParticleInformation> ParticleImpactDataTable;
  protected:
    ParticleImpactDataTable            m_particleImpactDataTable;
  };

  /**
  Base class for creating and writing export data to file. Is created and managed by the IDataExporter classes.
  */
  class AGXPHYSICS_EXPORT GranularDataExporter::IDataFileWriter : public agx::Referenced
  {
  public:
    /**
    Constructor for creating the file writer. File is created upon class construction.
    \param filePrefix The file prefix of the data file.
    \param postFix The file postfix for the data file.
    \param separator separator token that is used to separate the data entries in the export file.
    */
    IDataFileWriter(const agx::String& filePrefix, const agx::String& postFix, const agx::String& separator);

  protected:
    virtual ~IDataFileWriter();

    agx::String generateTimeStamp() const;

    void generateFileNames();

    void writeOutputString(const agx::String& output);

  protected:
    /// Each FileWriter implementation implements a different header, since different data is exported.
    virtual agx::String getHeader() = 0;

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::String     m_fileprefix;
    agx::String     m_postFix;
    agx::String     m_separator;
    agx::String     m_completeName;
    std::ofstream   m_outStream;
  };

  /**
  IDataFileWriter implementation for particle data.
  */
  class AGXPHYSICS_EXPORT GranularDataExporter::ParticleDataWriter : public agx::GranularDataExporter::IDataFileWriter
  {
  public:
    /**
    Constructor for creating the file writer. File is created upon class construction.
    \param filePrefix The file prefix of the data file.
    \param postFix The file postfix for the data file.
    \param separator separator token that is used to separate the data entries in the export file.
    */
    ParticleDataWriter(const agx::String& filePrefix, const agx::String& postFix, const agx::String& separator);

    /**
    Writes particle information to file.
    \param info Particle information that should be written to file.
    */
    void writeParticleInformation(const detail::ParticleInformation& info);

  protected:
    virtual ~ParticleDataWriter();

  protected:
    /// Each FileWriter implementation implements a different header, since different data is exported.
    virtual agx::String getHeader() override;
  };

  /**
  IDataFileWriter implementation for contact data.
  */
  class AGXPHYSICS_EXPORT GranularDataExporter::ContactDataWriter : public agx::GranularDataExporter::IDataFileWriter
  {
  public:
    /**
    Constructor for creating the file writer. File is created upon class construction.
    \param filePrefix The file prefix of the data file.
    \param postFix The file postfix for the data file.
    \param separator separator token that is used to separate the data entries in the export file.
    */
    ContactDataWriter(const agx::String& filePrefix, const agx::String& postFix, const agx::String& separator);

    /**
    Writes particle information to file.
    \param contact Particle information that should be written to file.
    */
    void writeContact(const detail::ParticleGeometryContactInformation& contact);

    /**
    Writes particle information to file.
    \param contact Particle information that should be written to file.
    */
    void writeContact(const detail::ParticlePairContactInformation& contact);

  protected:
    virtual ~ContactDataWriter();

  protected:
    virtual agx::String getHeader() override;

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
  };

  AGXPHYSICS_EXPORT void addGranularImpactCustomBuffers(agx::ParticleSystem * system);

  AGXPHYSICS_EXPORT agx::Physics::ParticlePairContactData * getParticlePairImpacts(agx::Physics::GranularBodySystem * system);

  AGXPHYSICS_EXPORT agx::Physics::ParticleGeometryContactData * getParticleGeometryImpacts(agx::Physics::GranularBodySystem * system);

  AGXPHYSICS_EXPORT agxData::EntityStorage* getParticlePairImpactsStorage(agx::Physics::GranularBodySystem * system);

  AGXPHYSICS_EXPORT agxData::EntityStorage* getParticleGeometryImpactsStorage(agx::Physics::GranularBodySystem * system);
}

#endif
