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

#ifndef AGXDATA_FILEJOURNAL_H
#define AGXDATA_FILEJOURNAL_H

#include <agxData/JournalArchive.h>
#include <agxData/SerializedFrame.h>
#include <agx/HashVector.h>

namespace agx
{
  class TiXmlElement;
  class TiXmlText;
}


namespace agxData
{
  AGX_DECLARE_POINTER_TYPES(FileJournal);


  /**
  The FileJournal provides a way to inspect and manipulate a journal created by the FileFrameWriter and stored on disk.
  */
  class AGXCORE_EXPORT FileJournal : public agxData::JournalArchive
  {
  public:

    static bool isFileJournal(const agx::String& path);

  public:
    AGX_DECLARE_POINTER_TYPES(Session);

    /**
     * Open a journal on disk.
     * \param diskPath The disk path to a directory which is a journal.
     */
    FileJournal( const agx::String& diskPath );

    /**
     Create a new session within the journal. Will return nullptr if a session with the given name already exists.
     \param name The name of the new session.
     \return The newly created session, or nullptr on error.
     */
    virtual JournalArchive::Session* createNewSession( const agx::Name& name ) override;


    /// \return The session with the given name.
    agxData::FileJournal::Session* getFileSession( const agx::Name& name );


    /// Remove the given session from the journal.
    virtual bool removeSession( agxData::JournalArchive::Session* session ) override;

    /// Rename the given session, including data movement on disk.
    virtual bool renameSession( agxData::JournalArchive::Session* session, const agx::Name& newName ) override;

    /// \return Path to a directory on disk where the current FileJournal stores its files.
    virtual agx::String getCustomFilesPath(bool create = true) const override;

    /// \return Path to a directory on disk that belong to the session with the given name. Will return a
    /// path even if the FileJournal doesn't contain a session with the given name.
    virtual agx::String getCustomFilesPath( const agx::Name& sessionName, bool create = true ) const override;


    /**
    Split the given session path into the path to the path to the journal and
    the name of the session.
    \param fullPath Disk path to a session.
    \param[out] journalPath Will be filled with the path of the journal, or the empty string on error.
    \param[out] sessionName Will be filled with the name of the session, or the empty string on error.
    */
    static bool parseSessionPath(const agx::String& fullPath, agx::String& journalPath, agx::String& sessionName);


    virtual DiskFrameReader* createFrameReader(JournalArchive::Session* session) override;
    virtual DiskFrameWriter* createFrameWriter(JournalArchive::Session* session) override;

  protected:
    virtual ~FileJournal();

    virtual agxData::JournalArchive::Session* makeCopy( agxData::JournalArchive::Session* session ) override;

  private:
    /// Find the sessions stored on disk and create their in-memory representation.
    void readJournalContent();

    /// Variant of createNewSession that returns a derived typed Session pointer rather than the abstract
    /// JournalArchive::Session type.
    FileJournal::Session* createNewFileSession( const agx::Name& name );


    /// Dynamic cast an JournalArchive::Session to a FileJournal::Session.
    agxData::FileJournal::Session* downcastSession( agxData::JournalArchive::Session* abstractSession ) const;


    // agxData::JournalArchive::Session* makeCopy( agxData::JournalArchive::Session* session );
  };


  /**
   Representation of a session on disk created by the FileFrameWriter.
   */
  class AGXCORE_EXPORT agxData::FileJournal::Session : public agxData::JournalArchive::Session
  {
  public:

    /// \return Disk path to the folder containing all the frames.
    agx::String getFramesPath() const;


    /// \return The disk path where the session is stored.
    virtual agx::String getPath() const override;

    /// \return The disk path to the .agx file for this session.
    virtual agx::String getScenePath(bool create = true) const override;


    /// The file journal does not support packing, so this does nothing.
    /// \return True if a scene file exists at the location returned by getScenePath. False otherwise.
    virtual bool unpackSceneFile() override;

    /// Copy the scene data found in the given string stream into the scene
    /// file pointed to by getScenePath.
    virtual bool packSceneData(std::stringstream& sceneData) override;

    /// Copy the scene data from the file pointed to by getScenePath into the given string stream.
    virtual bool unpackSceneData(std::stringstream& sceneData) override;

    /// Copy the scene data found in the given input stream into the scene
    /// file pointed to by getScenePath.
    virtual bool packSceneData(std::istream& sceneData) override;

    /// Copy the scene data from the file pointed to by getScenePath into the given output stream.
    virtual bool unpackSceneData(std::ostream& sceneData) override;


    /// Remove all frames from the given index and forward.
    virtual bool truncate( agx::UInt firstFrameToRemove ) override;

    /// \return Disk path to the folder where the given frame is stored. If the given frame isn't part
    /// of the stored session data, then the returned path will point to a folder that doesn't exist.
    agx::String buildFramePath( agx::UInt frameIndex ) const;


    /** Load the header data for a particular frame. */
    virtual agxData::Frame::Header getFrameHeader( agx::UInt frameIndex ) const override;

    /// \return The file journal to which the session have been registered.
    FileJournal* getFileJournal();
    /// \return The file journal to which the session have been registered.
    const FileJournal* getFileJournal() const;




    virtual agx::String getCustomFilesPath(bool create = true) const override;
    agx::String getPlotDirectoryPath() const;
    agx::String getSessionHeaderPath() const;

    virtual bool recordExtraData( const agx::String& key, const agx::String& value ) override;
    virtual bool retrieveExtraData( const agx::String& key, agx::String& value  ) override;

    virtual void savePlot(const PlotData* plot) override;
    virtual void removePlot(const agx::String& plotName) override;
    virtual PlotData* getPlot(const agx::String& plotName) override;
    virtual void getAllPlots(PlotDataRefVector& result) override;
    virtual void getPlotList(agx::StringVector& result) override;

    virtual void writeHeader() override;
    virtual void finalizeHeader() override;

    virtual agx::UInt getNextFrameIndex(agx::UInt currentFrameIndex, agx::Int offset) const override;

  public:
    /// Create a new empty session to which a recording can be made. Will return nullptr on error.

    static agxData::FileJournal::Session* create( const agx::Name& name, const agx::String& path );

    /// Load a session from disk. Will return nullptr on error.
    static FileJournal::Session* load( const agx::TiXmlElement* eSession );


    struct FrameInfo : public SerializedFrame::BinaryHeader::FrameMeta
    {
      agx::UInt64 headerOffset;
      agx::UInt64 dataSegmentOffset;
      agx::UInt64 sequenceIndex;
    };


    const FrameInfo* getFrameInfo(agx::UInt frameIndex) const;

    agx::UInt64 getFrameHead();
    void setFrameHead(agx::UInt64 head);

  private:
    friend class agxData::FileJournal;

    /// Copy the session folder for this session into the "Sessions" folder of the target JournalArchive,
    /// which much be an instance of the FileJournal class.
    bool copyDiskData( agxData::FileJournal* target );

    /// Move the session folder for this session into the "Sessions" folder of the target JournalArchive,
    /// which much be an instance of the FileJournal class.
    virtual bool transferDiskData( agxData::JournalArchive* target ) override;


    agx::TiXmlElement* generateHeader();

  private:

    /// Create a new session. Should not be called directly.
    Session( const agx::Name& name );

    agx::TiXmlText* getTextNode(agx::TiXmlElement* root, const agx::String& key, bool create);

    bool unpackSceneDataToStream(std::ostream& sceneData);
    void createFrameTable();
    void printFileStatistics();
    void readExtraData(agx::TiXmlElement* eExtraData);
    void readPlotData(agx::TiXmlElement* ePlotData);

  protected:
    virtual ~Session();

  private:
    // std::ofstream m_file;
    agx::UInt64 m_sceneDataOffset;
    agx::UInt64 m_sceneDataSize;
    agx::UInt64 m_frameDataOffset;
    agx::UInt64 m_frameHead;

    typedef agx::HashVector<agx::UInt64, FrameInfo> FrameTable;
    FrameTable m_frameTable;

    typedef agx::HashTable<agx::String, agx::String> ExtraDataTable;
    ExtraDataTable m_extraDataTable;


    struct PlotEntry
    {
      agx::UInt64 offset;
      agx::UInt64 headerSize;
      agx::UInt64 dataSize;
    };

    class PlotSort;

    typedef agx::HashTable<agx::Name, PlotEntry> PlotTable;
    PlotTable m_plotTable;
  };

}


/* AGXDATA_FILEJOURNAL_H */
#endif
