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


#ifndef AGXDATA_HDF5JOURNAL_H
#define AGXDATA_HDF5JOURNAL_H


#include <agx/config/AGX_USE_HDF5.h>
#include <agx/config.h>
#if AGX_USE_HDF5()


#include <agxData/JournalArchive.h>
#include <agx/AgxH5.h>

namespace agxData
{
  class Hdf5FrameReader;
  class Hdf5FrameWriter;

  AGX_DECLARE_POINTER_TYPES( Hdf5Journal );


  /**
  The Hdf5Journal provides a way to inspect and manipulate a journal created by
  the Hdf5FrameWriter and stored on disk.
  */
  class AGXCORE_EXPORT Hdf5Journal : public agxData::JournalArchive
  {
  public:
    static bool isHdf5Journal(const agx::String& path);

  public:
    AGX_DECLARE_POINTER_TYPES( Session );

    /**
    Open a journal on disk. The given disk path should be the path to a HDF5
    archive, which will be created if it does not exist already.
    */
    Hdf5Journal( const agx::String& diskPath, bool readOnlyIfExists=false);

    /**
    Create a new session within the journal. Will return nullptr if a session
    with the given name already exists.

    \param name The name of the new session.
    \return The newly created session, or nullptr on error.
    */
    virtual JournalArchive::Session* createNewSession( const agx::Name& name ) override;

    /**
    Variant of createNewSession that returns a derived type Session pointer
    rather than the abstract JournalArchive::Session type. The created session
    is exactly the same.
    */
    Hdf5Journal::Session* createNewHdf5Session( const agx::Name& name );

    /// \return The session with the given name.
    agxData::Hdf5Journal::Session* getHdf5Session( const agx::Name& name );

    /// Remove the given session from the journal.
    virtual bool removeSession(agxData::JournalArchive::Session* session) override;

    /// Rename the given session.
    virtual bool renameSession(agxData::JournalArchive::Session* session, const agx::Name& newName) override;

    /// \return A path where additional files related to the journal can be stored.
    virtual agx::String getCustomFilesPath(bool create = true) const override;

    /// \return A path where additional files related to the given session can be stored.
    virtual agx::String getCustomFilesPath( const agx::Name& sessionName, bool create = true ) const override;

    /// \return The path to the HDF5 file on disk.
    agx::String getHdf5ArchivePath() const;

    virtual DiskFrameReader* createFrameReader(JournalArchive::Session* session) override;
    virtual DiskFrameWriter* createFrameWriter(JournalArchive::Session* session) override;

  protected:
    virtual ~Hdf5Journal();

    virtual agxData::JournalArchive::Session* makeCopy(agxData::JournalArchive::Session* session) override;

  private:
    /// Find the sessions store on disk and create their in-memory representations.
    void readJournalContent();

    agxData::Hdf5Journal::Session* downcastSession( agxData::JournalArchive::Session* abstractSession ) const;

  private:
    friend class Session;
    void unlinkSessionData( const agx::Name& name );
    void renameSessionData( const agx::Name& name, const agx::Name& newName );
    H5::Group& getSessionsGroup();
    H5::H5File* getFile();

  private:
    H5::H5File* m_file;
    H5::Group m_sessions;
    agx::String m_baseName;
    agx::String m_parentFolder;
  };


  /**
  Representation of a session on disk created by the Hdf5FrameWriter.
  */
  class AGXCORE_EXPORT agxData::Hdf5Journal::Session : public agxData::JournalArchive::Session
  {
  public:

    /**
    Returns the disk/HDF5 path where the session is stored. Use
    getHdf5Journal()->getHdf5ArchivePath() to determine the part of the
    returned path which points to the HDF5 archive, the rest of the path
    identifies the node within the HDF5 archive that contains the session
    data. A Linux example:

    /some/folder/NameOfJournal.agxJournal/Sessions/NameOfSession

    In this example, the "/some/folder/NameOfJournal.agxJournal" part of the
    path identifies a HDF5 archive and the "/Sessions/NameOfSession" part
    identifies a path within the HDF5 archive.

    \return Filesystem/HDF5 path to the session data.
    */
    virtual agx::String getPath() const override;


    /*
    Get the path to the unpack location for the scene file for this session.
    The location will only contain a file if the unpackSceneData method has
    been called, or if a write to the file has been done explicitly from
    somewhere else.

    \return The path to the .agx file containing the initial state of the simulation.
    */
    virtual agx::String getScenePath(bool create = true) const override;



    /*
    Make a copy of the initial scene configuration data stored in the archive
    to the location pointed to by getScenePath. Returns false if there is no
    .agx file in the HDF5 archive, or if the write to the location pointed to
    by getScenePath fails.
    */
    virtual bool unpackSceneFile() override;


    /*
    Copy the data found in the stringstream into the initial scene
    configuration stored in the HDF5 archive.
    */
    virtual bool packSceneData(std::stringstream& sceneData) override;

    /*
    Copy the initial scene data stored in the HDF5 archive into the given string stream.
    */
    virtual bool unpackSceneData(std::stringstream& sceneData) override;

    /*
    Copy the data found in the stringstream into the initial scene
    configuration stored in the HDF5 archive.
    */
    virtual bool packSceneData(std::istream& sceneData) override;

    /*
    Copy the initial scene data stored in the HDF5 archive into the given string stream.
    */
    virtual bool unpackSceneData(std::ostream& sceneData) override;

    /*
    Get the path to a disk directory where custom files related to the session
    may be stored. The folder is created, if necessary, if the 'crate'
    argument is true.
    */
    virtual agx::String getCustomFilesPath(bool create = true) const override;

    /**
    Remove all frames from the given index and forward.
    */
    virtual bool truncate( agx::UInt firstFrameToRemove ) override;

    /**
    Load the header data for a particular frame.
    */
    virtual agxData::Frame::Header getFrameHeader( agx::UInt frameIndex ) const override;

    /// \return The HDF5 journal that the session is part of.
    Hdf5Journal* getHdf5Journal();
    /// \return The HDF5 journal that the session is part of.
    const Hdf5Journal* getHdf5Journal() const;

    virtual bool recordExtraData( const agx::String& key, const agx::String& value ) override;
    virtual bool retrieveExtraData( const agx::String& key, agx::String& value ) override;

    virtual void savePlot( const PlotData* plot) override;
    virtual void removePlot( const agx::String& plotName ) override;
    virtual PlotData* getPlot( const agx::String& plotName ) override;
    virtual void getAllPlots( PlotDataRefVector& result ) override;
    virtual void getPlotList( agx::StringVector& result ) override;

    virtual agx::UInt getNextFrameIndex(agx::UInt currentFrameIndex, agx::Int offset) const override;

    virtual void writeHeader() override;
    virtual void finalizeHeader() override;

  public:
    /// Create a new empty session to which a recording can be made. Will return nullptr on error.
    static agxData::Hdf5Journal::Session* create( const agx::Name& name, H5::Group& sessions );

    /// Load a session from disk. Will return nullptr on error.
    static agxData::Hdf5Journal::Session* load( const agx::Name& name, H5::Group& sessionRoot );

  private:
    friend class agxData::Hdf5Journal;

    // Rename the data owned by this session in the HDF5 archive.
    bool renameDiskData( const agx::Name& newName );

    bool copySessionFolder( agx::String targetFolderPath, bool deleteSource = false );

    /// Move the HDF5 group for this session into the HDF5 archive for the given target JournalArchive,
    /// which must be a HDF5 journal.
    virtual bool transferDiskData( agxData::JournalArchive* target ) override;

    bool copyDiskData( agxData::Hdf5Journal* target );


    /// Create a new session. Will only be done by the Hdf5Journal.
    Session( const agx::Name& name, H5::Group& session );


    void fixHeader(H5::Group& sessionRoot, H5::Group& header);
    void finalizeHeader(
      agx::UInt64 numFrames, agx::UInt64 startFrameIndex, agx::UInt64 endFrameIndex, double startFrameTime, double endFrameTime);

  private:
    friend class agxData::Hdf5FrameReader;
    friend class agxData::Hdf5FrameWriter;

    H5::Group& getHdf5Node();
    H5::H5File* getHdf5File();

    /// Will flush the entire HDF5 file to disk, including buffers belonging
    /// to groups other then this sessions root.
    void flushToFile();


  private:
    /// \return The path to the custom files for the session with the given name.
    agx::String getCustomFilesPath( const agx::Name& sessionName, bool create = true ) const;

    // Assumes that the h5::mutex is held by the current thread.
    bool unpackSceneDataToStream(std::ostream& sceneData);

    // Open the extra data HDF5 group. If 'create' is false, then the HDF5
    // library will Throw an exception if the group doesn't exist. The obvious
    // action is taken if 'create' is true in this case.
    H5::Group openExtraData(bool create);


    struct FrameInfo : public agxData::Frame::Header
    {
      agx::String nodeName;
      agx::UInt sequenceIndex;
    };

    void createFrameTable();

    static herr_t iterateSession(hid_t group_id, const char * member_name, const H5L_info_t *, void* operator_data);
    static bool compareFrames(const FrameInfo& lhs, const FrameInfo& rhs);

  protected:
    virtual ~Session();


  private:
    H5::Group m_session;
    typedef agx::HashVector<agx::UInt64, FrameInfo> FrameTable;
    FrameTable m_frameTable;
    agx::Vector<FrameInfo> m_tempFrames;
  };

}

/* AGX_USE_HDF5 */
#endif

/* AGXDATA_HDF5JOURNAL_H */
#endif
