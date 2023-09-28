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


#ifndef AGXDATA_FILE_TRACK_H
#define AGXDATA_FILE_TRACK_H

#include <agx/config/AGX_USE_AIO.h>
#include <agx/agx.h>
#include <agx/Logger.h>
#include <agx/List.h>
#include <agxData/DiskTrack.h>
#include <agxData/SerializedFrame.h>
#include <agxData/FileJournal.h>
#include <fstream>


#if AGX_USE_AIO()
#include <aio.h>
#endif

namespace agx
{
  class TiXmlDocument;
  class TiXmlElement;
}

namespace agxData
{
  AGX_DECLARE_POINTER_TYPES(FileFrameReader);

  /**
   * Frame reader that supports the files-and-foldes prototype format.
   */
  class AGXCORE_EXPORT FileFrameReader : public agxData::DiskFrameReader
  {
  public:
    FileFrameReader( agxData::JournalArchive::Session* session );

    virtual const agxData::JournalArchive::Session* getSession() const override;

    virtual bool jumpToFrame(agx::UInt frameIndex) override;
    virtual bool jumpToTime(agx::Real time) override;

    /// \return The index of the first frame in the track
    agx::UInt getFrameOffset() const;

    /**
      Controls how the binary data of a frame is read from disk.
       - FSTREAM Files are read using std::ifstream.
       - CACHED Files are read using open/read with default settings.
       - DIRECT Files are read using open/read with flags to disable OS buffering of the file content.
     */
    enum BinaryReadStrategy { FSTREAM, CACHED, DIRECT };

    void setBinaryReadStrategy( BinaryReadStrategy strategy );
    BinaryReadStrategy getBinaryReadStrategy() const;

#if AGX_USE_AIO()
    /**
     When enabled, then reads of binary files will done by the AIO library.
     Only supported when using a open/read strategy.
     */
    void setEnableUseAio( bool useAio );
    bool getEnableUseAio() const;
#endif

  protected:
    virtual ~FileFrameReader();

    virtual void prepareSequence(agx::UInt startFrame, agx::UInt lastFrame) override;
    virtual void endSequence() override;

  private:


    bool isValidFrame(agx::UInt frameIndex);
    virtual Frame* readFrame(agx::UInt frameIndex) override;
    using DiskFrameReader::readFrame;

    // void readComponent(agx::Component* parent, agx::TiXmlElement* eElement, std::istream& binaryFile);
    // void readBuffer(agx::Component* parent, agx::TiXmlElement* eElement, std::istream& binaryFile);
    // void readValue(agx::Component* parent, agx::TiXmlElement* eElement, std::istream& binaryFile);


    void removeUnwantedBuffers( agx::TiXmlElement* eElement, const agxData::Frame::DataBindingRefVector& bindings, const agx::String& path );


  private:
    BinaryReadStrategy m_readStrategy;

#if AGX_USE_AIO()
    bool m_useAio;
    PreloadedFrameRef m_preloadedFrame;
#endif

    FileJournal::SessionRef m_session;

    std::ifstream m_journalFile;
    SerializedFrame::ImportNameTable m_nameTable;
    bool m_activeMergeSequence;
  };



#if AGX_USE_AIO()
  /**
   Handle for asynchronous disk I/O operations.
   */
  class FileFrameReader::PreloadedFrame : public agx::Referenced
  {
  public:
    PreloadedFrame();

    /**
     Start a asynchronous disk read.
     \param filePath Disk path to the file to read.
     \param strategy Controls how the read will be performed. Currently only supports CACHED and DIRECT.
     \return True if the operations was successfully started. False otherwise.
     */
    bool queueFrameLoad( const agx::String& filePath, agxData::FileFrameReader::BinaryReadStrategy strategy );

    /**
     Accessor for the data read by the previously enqueued frame load. Ownership of the data is
     transfered from PreloadedFrame to the caller, so it is recommended to store the returned
     pointer in a BinaryDataRef. Calling 'getFrameData' without first calling 'queueFrameLoad',
     or calling 'getFrameData' multiple times with no 'queueFrameLoad' in between will result in
     nullptr being returned.
     */
    BinaryData* getFrameData();

    /**
     \return True if there is a queued I/O operation.
     */
    bool hasQueuedRead() const;

    /**
     If a read operation is currently queued, then the disk path to the file being read is returned. The
     empty string is otherwise returned.
     */
    const agx::String& getQueuedFilePath() const;

  private:
    bool waitForCurrentReadToComplete();
    void clear();
    virtual ~PreloadedFrame();
  private:
    struct aiocb m_aioHandle;
    agx::UInt8* m_frameData;
    size_t m_frameDataSize;
    size_t m_alignmentRequirement;
    agx::String m_filePath;
  };
#endif

  //---------------------------------------------------------------


  AGX_DECLARE_POINTER_TYPES(FileFrameWriter);
  class AGXCORE_EXPORT FileFrameWriter : public agxData::DiskFrameWriter
  {
  public:
    /**
    NOTE: Buffer compression is default disabled due to decompression being slow for large datasets, 70k transformations takes > 5sec to decompress :/
    */
    FileFrameWriter( agxData::JournalArchive::Session* session);


    virtual void writeFrame(const agxData::Frame* frame) override;
    virtual const JournalArchive::Session* getSession() const override;
    virtual void endOfStream() override;

  protected:
    virtual ~FileFrameWriter();

  private:
    agx::String getFilePath() const;

  private:
    agxData::FileJournal::SessionRef m_session;
    // bool m_storeAsciiValues;

    SerializedFrame::ExportNameTable m_nameTable;
  };




  /* Implementation */
  AGX_FORCE_INLINE agx::UInt FileFrameReader::getFrameOffset() const { return (agx::UInt)m_session->getFirstFrameIndex(); }
}



/* AGXDATA_FILE_TRACK_H */
#endif
