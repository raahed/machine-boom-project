/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_JOURNAL_H
#define AGX_JOURNAL_H

#include <agx/Component.h>

#include <agx/Clock.h>
#include <agx/Pointer.h>
#include <agx/Range.h>
#include <agxData/Track.h>
#include <agxData/FileTrack.h>
#include <agxData/Hdf5Track.h>
#include <agxData/FileJournal.h>
#include <agxSDK/Simulation.h>
#include <agxSDK/SimulationTrack.FrameReader.h>
#include <agxSDK/SimulationTrack.FrameWriter.h>

namespace agxSDK
{
  class Assembly;
}


namespace agx
{

  /**
  The journal is used to record/playback data from a simulation.
  */
  AGX_DECLARE_POINTER_TYPES(Journal);
  class AGXPHYSICS_EXPORT Journal : public Referenced
  {
  public:

    enum Mode
    {
      INVALID = -1,
      RECORD,
      PLAYBACK
    };

    enum ArchiveFormat
    {
      AUTO,
      HDF5, /// AGX has to be built with HDF5-support to use this.
      CUSTOM
    };

  public:
    typedef agx::Callback2<Journal *, std::exception&> ExceptionHandler;

    class Converter;

  public:
    Journal(const String& diskPath = "Journal.agxJournal", ArchiveFormat archiveFormat = AUTO, bool loadPlotFromJournal = true, bool readOnlyIfExists=false);

    /**
    \return The name of the journal.
    */
    const Name& getName() const;


    /**
    Set the disk path.
    */
    // void setDiskPath(const String& path);

    /**
    \return The disk path to the archive where the journal data is located.
    */
    const String& getDiskPath() const;


    /**
    Convert journal from one format to another.
    */
    static void convert(const String& targetPath, const String& sourcePath, ArchiveFormat targetFormat, bool expand = false);


    /**
    \return The journal archive format.
    */
    ArchiveFormat getArchiveFormat() const;

    /**
    Jump to a specified frame during playback.
    \return True if successful
    */
    bool jumpToFrame(agx::UInt frameIndex);


    /**
    Jump to a specified time during playback.
    \return True if successful
    */
    bool jumpToTime(agx::Real time);


    bool sanityCheck(const agxSDK::Simulation* simulation, agxData::JournalArchive::Session* session) const;

    /**
    Attach the journal to a simulation.
    \param simulation The simulation
    \param mode RECORD or PLAYBACK
    \param frameCache An existing track to load cached frames from
    \param passive True if simulation should not be made aware of this journal, must be true if more than one journal is attached
    */
    void attach(agxSDK::Simulation *simulation, Mode mode, agxData::Track *frameCache = nullptr, bool passive = false);

    /**
    Detach from the currently attached simulation.
    */
    void detach();

    ///\return the attached simulation
    agxSDK::Simulation *getSimulation();

    ///\return the attached simulation
    const agxSDK::Simulation *getSimulation() const;

    /**
    Set the journal mode, RECORD or PLAYBACK.
    */
    void setMode(Mode mode);

    /**
    \return The current journal mode.
    */
    Mode getMode() const;

    /**
    Enable/disable incremental simulation structure recording. Enabled by default.
    */
    void setEnableIncrementalStructure(bool flag);

    /**
    \return True if incremental structure recording is enabled.
    */
    bool getEnableIncrementalStructure() const;

    /**
    \return True if a jump is in progress.
    */
    bool isJumping() const;

    /**
    Load data bindings from a configuration file.
    \returns true for successful load, otherwise false
    */
    bool loadConfiguration(const String& configurationFilePath);

    /**
    Add a data binding to the journal, eg Particle.position will register the buffer to be stored/restored by journal.
    */
    void addDataBinding(const agx::Path& path, bool recursive = false);
    void addDataBinding(const agx::Object *object, bool recursive = false);

    /**
    Remove a data binding from the journal.
    */
    bool removeDataBinding(const agx::Path& path);
    bool removeDataBinding(const agx::Object *object);

    /**
    Remove all data bindings.
    */
    void clearDataBindings();

    /**
    Specify time stamp when journal recording should begin.
    */
    void startRecordingAtTime(agx::Real time);

    /**
    Flush the journal.
    \param force Does not wait for output thread to completed if true.
    */
    void flush(bool force = false);


    /**
    Specify an exception handler.
    */
    void setExceptionHandler(const ExceptionHandler& handler);

    /**
    \return The current exception handler.
    */
    const ExceptionHandler& getExceptionHandler() const;


    /// \return The JournalArchive that frames are being read from or written to.
    agxData::JournalArchive* getFileJournal() const;

    /**
    \return All sessions in the journal archive.
    */
    const agxData::JournalArchive::SessionTable& getSessions() const;

    /**
    Load the most recent session.
    \return nullptr if journal does not exist or does not contain any sessions.
    */
    agxData::JournalArchive::Session* loadNewestSession();




    /// \return The session within the JournalArchive that frames are being read from or written to.
    agxData::JournalArchive::Session* getSession( const agx::Name& name );

    /// Create a new session within the current Journal archive. Will make the
    /// newly created session the active session.
    agxData::JournalArchive::Session* createNewSession();
    agxData::JournalArchive::Session* createNewSession(const Name& name);

    /// Change the active session to the session with the given name. The session must be
    /// part of the current JournalArchive.
    /// Will return nullptr if it it didn't exist
    agxData::JournalArchive::Session* setCurrentSession( const agx::Name& sessionName, bool saveSceneOnNextPreTick=true );

    /// Change the active session to the given session. The session must be
    /// part of the current JournalArchive.
    void setCurrentSession( agxData::JournalArchive::Session* session, bool saveSceneOnNextPreTick=true );

    agxData::JournalArchive::Session *getCurrentSession();

    /**
    Change the name of the current session.
    */
    bool renameCurrentSession( const agx::String& newName );

    /**
    Remove a session from the journal.
    */
    bool removeSession( const agx::Name& sessionName );


    /**
    Explicitly read a frame from the journal.
    \param index The frame index
    */
    agxData::FrameRef readFrameWithIndex(agx::UInt index);

    /**
    Explicitly read a frame from the journal.
    \param time The point in time
    */
    agxData::FrameRef readFrameAtTime(agx::Real time);
    /**
    Find what serialization version a Journal was stored with.
    */
    agx::UInt16 getSerializationVersion() const;


    agxData::Track *getTrack();
    agxSDK::SimulationFrameReader *getSimulationReader();
    agxSDK::SimulationFrameWriter *getSimulationWriter();
    agxData::DiskFrameReader *getFileReader();
    agxData::DiskFrameWriter *getFileWriter();

    const agxData::Track *getTrack() const;
    const agxSDK::SimulationFrameReader *getSimulationReader() const;
    const agxSDK::SimulationFrameWriter *getSimulationWriter() const;
    const agxData::DiskFrameReader *getFileReader() const;
    const agxData::DiskFrameWriter *getFileWriter() const;


    /// A callback that will be called when starting playback from a recording.
    typedef Callback1<const agx::String&> SceneLoaderCallback;
    void setSceneLoader(SceneLoaderCallback loader);

    typedef Callback2<agxSDK::Assembly *, bool> RenderLoaderCallback;
    void setRenderLoader(RenderLoaderCallback loader);

    RenderLoaderCallback getRenderLoader();
    void triggerFrame();


    /// \return The most recent session
    agxData::JournalArchive::Session *findNewestSession();

    /// \return The name of the sessions with the highest date in its header. The
    /// empty string is returned if there are no sessions in the journal.
    static bool findNewestSession( const agx::String& diskPath, agx::JournalRef& journal, agxData::JournalArchive::SessionRef& session);

    DOXYGEN_START_INTERNAL_BLOCK()

    /**
     A form of attachment that doesn't use a Simulation. Frames are simply sent from the input to the output.
     Used by the JournalReader.
     */
    void attachStandalone( agxData::FrameReader* input, agxData::FrameWriter* output );

    DOXYGEN_END_INTERNAL_BLOCK()

    /// Save the scene state, normally done automatically
    void saveScene();

    /**
    Specifies if the simulation serialization should be temporary stored to disk or to memory when creating the journal.
    Storing to memory is default. But this might cause problem, as a large continuous memory need to be allocated.
    For 32bit applications, it might be problematic to allocate 100Mb or more continuous memory, hence storing to disk might be better. And slower.

    DEFAULT: store to memory (false)

    \param flag - if true, scene will be stored to disk, otherwise to memory before written to journal.
    */
    void setSaveSceneToDisk( bool flag );

    /**
    \return true if simulation scene will be temporarily stored to disk when writing a journal.
    */
    bool getSaveSceneToDisk() const;

  protected:
    virtual ~Journal();

  private:

    void preTickCallback(Clock *clock);
    void initPlayback();
    void defaultSceneLoader(const String& path);
    // void postTickCallback(Clock *clock);
    void trackExceptionCallback(agxData::Track *track, std::exception& e);
    void createApiTrack();


  private:
    agxData::JournalArchiveRef m_diskJournal;

    ExceptionHandler m_exceptionHandler;

    Mode m_mode;
    agxData::JournalArchive::SessionRef m_currentSession;
    agxSDK::SimulationObserver m_simulation;

    agxData::TrackRef m_track;
    agxSDK::SimulationFrameReaderRef m_simulationReader;
    agxSDK::SimulationFrameWriterRef m_simulationWriter;
    agxData::DiskFrameWriterRef m_fileWriter;
    agxData::DiskFrameReaderRef m_fileReader;

    SceneLoaderCallback m_sceneLoader;
    RenderLoaderCallback m_renderLoader;


    agxData::TrackRef m_apiTrack;
    // agxData::DiskFrameReaderRef m_apiFileReader;
    // agxSDK::SimulationFrameReaderRef m_apiSimulationReader;

    // ComponentRef m_context;
    // ClockRef m_clock;
    Clock::TickEvent::CallbackType m_preTickCallback;
    // Clock::TickEvent::CallbackType m_postTickCallback;


    Mutex m_mutex;
    Block m_block;

    bool m_hasSavedScene;
    Callback m_loadStartFrameCallback;
    ArchiveFormat m_archiveFormat;
    bool m_saveSimulationToDisk;
    bool m_passive;
    bool m_loadPlot;
    Real m_recordStartTime;
    bool m_isRecording;
    agx::UInt16 m_serializationVersion;
    bool m_enableIncrementalStructure;
    bool m_isJumping;
  };

  class AGXPHYSICS_EXPORT Journal::Converter
  {
  public:
    struct SessionHeader
    {
      agx::String name;
      agx::UInt startFrame;
      agx::UInt endFrame;
      agx::Real startTime;
      agx::Real endTime;
      agx::UInt stride;
      agx::String versionName;
      agx::UInt32 version;
    };

    typedef agx::Event1<UInt> ProgressEvent;
    ProgressEvent progressEvent;

    typedef agx::Event1<SessionHeader> SessionEvent;
    SessionEvent sessionStartEvent;
    SessionEvent sessionEndEvent;

  public:
    Converter();

    void execute();

    void setSourcePath(const agx::String& path);
    void setTargetPath(const agx::String& path);
    void setTargetFormat(ArchiveFormat format);
    void setExpand(agx::Bool expand);
    void setStartTime(agx::Real startTime);
    void setEndTime(agx::Real endTime);
    void setFrameStride(agx::UInt stride);

    void extractScene(const agx::String& sessionName);


    const agx::String& getSourcePath() const;
    const agx::String& getTargetPath() const;
    ArchiveFormat getTargetFormat() const;
    agx::Bool getExpand() const;
    agx::Real getStartTime() const;
    agx::Real getEndTime() const;
    agx::UInt getFrameStride() const;

  private:
    void writeCallback(const agxData::Frame *frame);
    void convertSession(
      agxData::JournalArchive::Session* targetSession, agxData::JournalArchive::Session* sourceSession);

  private:
    agx::String m_sourcePath;
    agx::String m_targetPath;
    ArchiveFormat m_format;
    agx::Bool m_expand;
    agx::Real m_startTime;
    agx::Real m_endTime;
    agx::UInt m_stride;
    agxData::FrameWriter::WriteEvent::CallbackType m_writeCallback;
    agxData::JournalArchiveRef m_sourceArchive;
    agxData::JournalArchiveRef m_targetArchive;
    SessionHeader m_currentSession;
    agx::UInt m_numFramesWritten;
  };


  /* Implementation */

  AGX_FORCE_INLINE const Name& Journal::getName() const { return m_diskJournal->getName(); }
  AGX_FORCE_INLINE const String& Journal::getDiskPath() const { return m_diskJournal->getPath(); }
  AGX_FORCE_INLINE Journal::Mode Journal::getMode() const { return m_mode; }
  AGX_FORCE_INLINE agxSDK::Simulation *Journal::getSimulation() { return m_simulation; }
  AGX_FORCE_INLINE const agxSDK::Simulation *Journal::getSimulation() const { return m_simulation; }

  AGX_FORCE_INLINE agxData::JournalArchive::Session *Journal::getCurrentSession() { return m_currentSession; }

  AGX_FORCE_INLINE agxData::Track *Journal::getTrack() { return m_track; }
  AGX_FORCE_INLINE agxSDK::SimulationFrameReader *Journal::getSimulationReader() { return m_simulationReader; }
  AGX_FORCE_INLINE agxSDK::SimulationFrameWriter *Journal::getSimulationWriter() { return m_simulationWriter; }

  AGX_FORCE_INLINE agxData::DiskFrameReader *Journal::getFileReader() { return m_fileReader; }
  AGX_FORCE_INLINE agxData::DiskFrameWriter *Journal::getFileWriter() { return m_fileWriter; }

  AGX_FORCE_INLINE const agxData::Track *Journal::getTrack() const { return m_track; }
  AGX_FORCE_INLINE const agxSDK::SimulationFrameReader *Journal::getSimulationReader() const { return m_simulationReader; }
  AGX_FORCE_INLINE const agxSDK::SimulationFrameWriter *Journal::getSimulationWriter() const { return m_simulationWriter; }

  AGX_FORCE_INLINE const agxData::DiskFrameReader *Journal::getFileReader() const { return m_fileReader; }
  AGX_FORCE_INLINE const agxData::DiskFrameWriter *Journal::getFileWriter() const { return m_fileWriter; }

  /////////////////////////////////////////////////////////////////////

}


#endif /* AGX_JOURNAL_H */
