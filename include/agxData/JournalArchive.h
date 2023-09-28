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

#ifndef AGXDATA_JOURNALARCHIVE_H
#define AGXDATA_JOURNALARCHIVE_H

#include <agx/Device.h>
#include <agx/Component.h>
#include <agx/Range.h>
#include <agx/Date.h>
#include <agxNet/FramePacket.h>
#include <agxData/SerializedFrame.h>
#include <agxData/FrameIO.h>


namespace agxData
{
  class DiskFrameReader;
  class DiskFrameWriter;



  /**
  The JournalArchive provides an abstract class for inspecting and
  manipulating the contents of a created journal. Support for the
  various saved journal formats is provided by classes deriving from this one.
  */
  class AGXCORE_EXPORT JournalArchive : public agx::Referenced
  {
  public:
    AGX_DECLARE_POINTER_TYPES( Session );


    typedef agx::HashTable<agx::Name, agxData::JournalArchive::SessionRef> SessionTable;


    /// \return The name of the journal. This is often the same as the name of a file on disk.
    const agx::Name& getName() const;

    /// \return A search path to the saved journal. The semantics of the string is storage format dependent, but is often a
    /// path to a file on disk.
    const agx::String& getPath() const;

    /// Add to the end of the given Vector the names of all the sessions in the Journal.
    void listSessionNames( agx::Vector<agx::Name>& result ) const;

    /// \return The session with the given name, or nullptr if no such session exists.
    agxData::JournalArchive::Session* getSession( const agx::Name& name );

    /// \return The session with the given name, or nullptr if no such session exists.
    const agxData::JournalArchive::Session* getSession( const agx::Name& name ) const;

    /// \return A table containing all sessions stored in the journal, mapping session names to sessions themselves.
    const SessionTable& getSessions() const;

    /**
     * Create a new session with the given name.
     * \return The session just created.
     */
    virtual agxData::JournalArchive::Session* createNewSession( const agx::Name& name ) = 0;

    /// Remove the session with the given name from the journal.
    bool removeSession( const agx::Name& name );

    /// Remove the given session from the journal.
    virtual bool removeSession( agxData::JournalArchive::Session* session ) = 0;

    /**
    Rename the given session to the given new name.

    \param session The session to rename.
    \param newName The name that the session should be renamed to.

    \internal
    The implementing class is expected to perform the renaming on the underlying storage and then call the
    protected updateSessionAndTableWithNewName method to update the in-memory session table and the name held
    by the Session object.
    */
    virtual bool renameSession( agxData::JournalArchive::Session* session, const agx::Name& newName ) = 0;


    /**
    Copy the given session from this journal into the given target journal. The session must be owned by the
    current JournalArchive and the target must be of the same journal type, i.e, use the same disk format.

    \param session The session to copy.
    \param target The journal that should receive the copy.
    \return True if the copy is successful, false otherwise.
    */
    bool copySession( agxData::JournalArchive::Session* session, agxData::JournalArchive* target );

    /**
    Transfer the given session from this journal into the given target. The
    session must be owned by the current JournalArchive and the target must be
    of the same journal type, i.e., use the same disk format. It is safe to
    continue using pointers to the transfered session, then in-memory object
    is updated rather than recreated.

    \param session The session to transfer.
    \param target The journal that should take over ownership of the session.
    \return True of the transfer is successful, false otherwise.
    */
    bool transferSession( agxData::JournalArchive::Session* session, agxData::JournalArchive* target );



    /**
    Returns the path to a disk directory where custom files related to the
    journal may be stored. The directory, including parents, is created if
    necessary if the 'create' argument is true.

    \param create If true then missing directories in the returned path is created.
    \return Path to custom files directory on disk.
    */
    virtual agx::String getCustomFilesPath(bool create = true) const = 0;


    /**
    Returns the path to a disk directory where custom files related to the
    given session may be stored. The directory, including parents, is created
    if necessary if the 'create' parameter is true. The given session name
    does not need to denote a session that is part of the journal. If not,
    then the returned path is the path that a sessions with the given name
    would have if such a session existed.

    \param sessionName The name of the session whose custom files path is needed.
    \return Path to custom files directory on disk for the given session.
    */
    virtual agx::String getCustomFilesPath( const agx::Name& sessionName, bool create = true ) const = 0;

    /// \return True if this journal is the owner of the given session. False otherwise.
    bool isOwnerOf( const agxData::JournalArchive::Session* session) const;

    /// \return The number of sessions currently stored in the journal.
    size_t getNumSessions() const;


    virtual DiskFrameReader* createFrameReader(agxData::JournalArchive::Session* session) = 0;
    virtual DiskFrameWriter* createFrameWriter(agxData::JournalArchive::Session* session) = 0;

  protected:
    JournalArchive( const agx::Name& name, const agx::String& path );
    JournalArchive();

    virtual ~JournalArchive();

    void setName( const agx::Name& name );
    void setPath( const agx::String& path );

    /// Create the Journal <-> Session coupling. Will fail if there already is a session with the given name
    /// in the session table.
    bool registerSession( agxData::JournalArchive::Session* newSession );

    /// Remove the Journal <-> Session coupling. Note that this will do an unreference of the session and may
    /// therefore delete the session.
    bool unregisterSession( agxData::JournalArchive::Session* removedSession );

    /// Change the name of the given session. Will ensure that the session table is kept consistent.
    bool updateSessionAndTableWithNewName( agxData::JournalArchive::Session* session, const agx::Name& newName );

    /// Create a copy of the given session. The copy will have its disk data stored in the current JournalArchive.
    virtual agxData::JournalArchive::Session* makeCopy( agxData::JournalArchive::Session* session ) = 0;

  private:
    agx::Name m_name;
    agx::String m_path;
    SessionTable m_sessions;

  };
  AGX_DECLARE_POINTER_TYPES(JournalArchive);



  /*
  The JournalArchive::Session class is a handle to a particular session in a journal. Every journal storage format
  that has a JournalArchive subclass should also provide a JournalArchive::Session derivation.
  */
  class AGXCORE_EXPORT JournalArchive::Session : public agx::Referenced
  {
  public:
    static agx::String generateNameFromCurrentDate();

  public:

    /// \return The name of the session.
    const agx::Name& getName() const;

    /**
    Returns a format specific string that identifies the location of this session within a journal.
    This may be, for example, a disk or HDF5 path.
    */
    virtual agx::String getPath() const = 0;


    /// \return The JournalArchive that this session is part of.
    agxData::JournalArchive* getJournal();
    /// \return The JournalArchive that this session is part of.
    const agxData::JournalArchive* getJournal() const;

    /**
    Some journal formats support more than one format for frame metadata. It
    is the responsibility of the agxData::Track reading or writing from/to the
    session to ensure that the set header format matches the actual format
    used on disk. In particular, setHeaderFormat cannot be used to change the
    header format of a session.
    */
    void setHeaderFormat(SerializedFrame::HeaderFormat format);

    SerializedFrame::HeaderFormat getHeaderFormat() const;

    /// Remove all frames from the given index and forward.
    virtual bool truncate( agx::UInt firstFrameToRemove ) = 0;


    /// Copy the session to another JournalArchive. The target JournalArchive must be of the same correct type.
    /// The result is the same as if "session->getJournal()->copySession(session, target)" had been called.
    bool copyTo( agxData::JournalArchive* target );

    /// Transfer the session to another JournalArchive. The target JournalArchive must be of the same correct type.
    bool transferTo( agxData::JournalArchive* target );


    /**
    Get the path on disk where the .agx file, called the scene file,
    containing the initial state of the simulation may be stored. Actual
    presence of the .agx file depends on if it has been written yet.

    If the journal stores the scene data in some internal format, then the
    unpackSceneFile may be used to copy the initial state from the internal
    format into a .agx file located at the path returned by this function.

    \return The path to the .agx file containing the initial state of the simulation.
    */
    virtual agx::String getScenePath(bool create = true) const = 0;


    /**
    Ensure that the scene file for the session is available at the location
    returned by getScenePath. Will overwrite the scene file with the packed
    version if a file is already present at the path returned by getScenePath.
    */
    virtual bool unpackSceneFile() = 0;

    /**
    Copy the scene data found in the given stringstream into the journal in
    some format specific way. The method is intended to be used with a
    stringstream filled using the agxSDK::Simulation::write method.
    */
    virtual bool packSceneData(std::stringstream& sceneData) = 0;

    /**
    Copy scene data from the archive, stored in some format specific way, into
    the given stringstream. The stringstream can then be passed on to the
    agxSDK::Simulation::read method.
    */
    virtual bool unpackSceneData(std::stringstream& sceneData) = 0;

    /**
    Copy the scene data found in the given stringstream into the journal in
    some format specific way. The method is intended to be used with a
    stringstream filled using the agxSDK::Simulation::write method.
    */
    virtual bool packSceneData(std::istream& sceneData) = 0;

    /**
    Copy scene data from the archive, stored in some format specific way, into
    the given stringstream. The stringstream can then be passed on to the
    agxSDK::Simulation::read method.
    */
    virtual bool unpackSceneData(std::ostream& sceneData) = 0;


    /**
    Get the path a disk directory where additional files related to the
    session may be stored. Can create the folder structure if it does not yet
    exist.

    \param create If true then the returned folder and its parents are created.
    \return Path to a folder owned by this session.
    */
    virtual agx::String getCustomFilesPath(bool create = true) const = 0;

    /** Store a string key-value pair to the session on disk. */
    virtual bool recordExtraData( const agx::String& key, const agx::String& value ) = 0;

    /** Read back the value associated with the key from disk. The value, if found, is stored in 'value'*/
    virtual bool retrieveExtraData( const agx::String& key, agx::String& value ) = 0;
    agx::String retrieveExtraData(const agx::String& key);

    /** Store a string key-value pair to the session on disk, without ruining the . */
    bool recordExtraDataAsJSon(const agx::String& key, const agx::String& value);

    /** Read back the value associated with the key from disk. The value, if found, is stored in 'value'*/
    bool retrieveExtraDataAsJSon(const agx::String& key, agx::String& value);
    agx::String retrieveExtraDataAsJSon(const agx::String& key);

    /** Load the header data for a particular frame. */
    virtual agxData::Frame::Header getFrameHeader( agx::UInt frameIndex ) const = 0;

    void setFrameIO(agxData::FrameIO* frameIO);
    agxData::FrameIO* getFrameIO();

    typedef agxNet::StructuredMessage PlotData;
    typedef agx::ref_ptr<PlotData> PlotDataRef;
    typedef agx::Vector<PlotDataRef> PlotDataRefVector;

    virtual void savePlot(const PlotData* plot) = 0;
    virtual void removePlot(const agx::String& plotName) = 0;
    virtual PlotData* getPlot(const agx::String& plotName) = 0;
    virtual void getAllPlots(PlotDataRefVector& result) = 0;
    virtual void getPlotList(agx::StringVector& result) = 0;

    virtual agx::UInt getNextFrameIndex(agx::UInt currentFrameIndex, agx::Int offset) const = 0;

  public:
    virtual void writeHeader() = 0;
    virtual void finalizeHeader() = 0;

    void setDate(agx::Date* date);
    const agx::Date* getDate() const;

    void setFirstFrameIndex(agx::UInt64 startFrame);
    agx::UInt64 getFirstFrameIndex() const;

    void setFirstFrameTimeStamp(agx::Real64 startTime);
    agx::Real64 getFirstFrameTimeStamp() const;

    void setLastFrameIndex(agx::UInt64 endFrame);
    agx::UInt64 getLastFrameIndex() const;

    void setLastFrameTimeStamp(agx::Real64 endTime);
    agx::Real64 getLastFrameTimeStamp() const;

    void setTimeStep(agx::Real64 timeStep);
    agx::Real64 getTimeStep() const;

    void setFrameStride(agx::UInt64 stride);
    agx::UInt64 getFrameStride() const;

    void setNumFrames(agx::UInt64 numFrames);
    agx::UInt64 getNumFrames() const;

    void setComputationTime(agx::Real64 computationTime);
    agx::Real64 getComputationTime() const;

    void setAgxVersionString(const agx::String& version);
    const agx::String& getAgxVersionString() const;
    void setAgxVersionNumber( const agx::UInt32 version);
    agx::UInt32 getAgxVersionNumber() const;

    bool is64BitReal() const;
    bool is64BitArchitecture() const;
    bool isLittleEndian() const;

    void set64BitReal(bool flag);
    void set64BitArchitecture(bool flag);
    void setLittleEndian(bool flag);

    void setSimulationHeader(agx::Component* header);
    agx::Component* getSimulationHeader();

  protected:
#ifndef SWIG
    Session( const agx::Name& name );
#endif
    virtual ~Session();

    bool rename( const agx::Name& name );

  private:
    friend class JournalArchive;
    void setName( const agx::Name& newName );
    void setJournal( agxData::JournalArchive* journal );

    /// Perform the disk operations required in order to move a session between journals.
    virtual bool transferDiskData( agxData::JournalArchive* target ) = 0;


  private:
    agx::Name m_name;
    agxData::JournalArchive* m_journal;
    agx::DateRef m_date;
    bool m_is64BitReal;
    bool m_is64BitArchitecture;
    bool m_isLittleEndian;
    agx::ComponentRef m_simulationHeader;
    agx::UInt64 m_numFrames;
    agx::UInt64 m_frameStride;
    agx::UInt64 m_startFrame;
    agx::Real64 m_startTime;
    agx::UInt64 m_endFrame;
    agx::Real64 m_endTime;
    agx::Real64 m_timeStep;
    agx::Real64 m_computationTime;
    agx::String m_agxVersionString;
    agx::UInt32 m_agxVersionNumber;
    SerializedFrame::HeaderFormat m_headerFormat;
    agxData::FrameIOObserver m_frameIO;
  };




  /* Implementation */

  DOXYGEN_START_INTERNAL_BLOCK()

  AGX_FORCE_INLINE bool JournalArchive::Session::is64BitReal() const { return m_is64BitReal; }
  AGX_FORCE_INLINE bool JournalArchive::Session::is64BitArchitecture() const { return m_is64BitArchitecture; }
  AGX_FORCE_INLINE bool JournalArchive::Session::isLittleEndian() const { return m_isLittleEndian; }

  AGX_FORCE_INLINE const agx::Date* JournalArchive::Session::getDate() const { return m_date; }
  AGX_FORCE_INLINE agx::UInt64 JournalArchive::Session::getFirstFrameIndex() const { return m_startFrame; }
  AGX_FORCE_INLINE agx::Real64 JournalArchive::Session::getFirstFrameTimeStamp() const { return m_startTime; }
  AGX_FORCE_INLINE agx::UInt64 JournalArchive::Session::getLastFrameIndex() const { return m_endFrame; }
  AGX_FORCE_INLINE agx::Real64 JournalArchive::Session::getLastFrameTimeStamp() const { return m_endTime; }
  AGX_FORCE_INLINE agx::Real64 JournalArchive::Session::getTimeStep() const { return m_timeStep; }
  AGX_FORCE_INLINE agx::UInt64 JournalArchive::Session::getFrameStride() const { return m_frameStride; }
  AGX_FORCE_INLINE const agx::String& JournalArchive::Session::getAgxVersionString() const { return m_agxVersionString; }
  AGX_FORCE_INLINE agx::UInt32 JournalArchive::Session::getAgxVersionNumber() const { return m_agxVersionNumber; }
  // AGX_FORCE_INLINE agx::UInt JournalArchive::Session::getNumFrames() const { return m_numFrames; }
  AGX_FORCE_INLINE agx::UInt64 JournalArchive::Session::getNumFrames() const { return m_numFrames; }
  AGX_FORCE_INLINE SerializedFrame::HeaderFormat JournalArchive::Session::getHeaderFormat() const { return m_headerFormat; }

  DOXYGEN_END_INTERNAL_BLOCK()

}


#endif /* AGXDATA_JOURNALARCHIVE_H */
