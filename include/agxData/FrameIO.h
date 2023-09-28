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

#ifndef AGXDATA_FRAMEIO_H
#define AGXDATA_FRAMEIO_H

#include <agxData/Frame.h>
#include <agxData/SerializedFrame.h>

namespace agxData
{
  AGX_DECLARE_POINTER_TYPES(FrameIO);

  AGX_DECLARE_POINTER_TYPES(FrameReader);
  AGX_DECLARE_VECTOR_TYPES(FrameReader);

  AGX_DECLARE_POINTER_TYPES(FrameWriter);
  AGX_DECLARE_VECTOR_TYPES(FrameWriter);

  AGX_DECLARE_POINTER_TYPES(FrameChannel);

  class AGXCORE_EXPORT FrameIO : public agx::Object
  {
  public:
    /**
    \return True if the reader is active. Otherwise a separate thread will pull frames from the reader.
    */
    bool isActive() const;

    /**
    \return The track which the IO implementation is connected to.
    */
    Track *getTrack();
    const Track *getTrack() const;

    /**
    Register a data binding.
    The external path is implementation specific.
    The internal path is where the data element is stored in the frame.

    Eg. a data buffer may be imported from disk using the external path "RigidBody.velocity", and
    then pushed to a data track with the internal path "foo". The corresponding outputs connected to the
    track must then bind to "foo" to retrieve the data.
    */
    void addDataBinding(const agx::Path& internalPath, const agx::Path& externalPath, bool recursive = false);

    /**
    Alias for addDataBinding(path, path)
    */
    void addDataBinding(const agx::Path& path, bool recursive = false);


    bool removeDataBinding( const agx::Path& path );
    bool removeDataBinding( const agx::Path& internalPath, const agx::Path& externalPath );

    // void addDataBinding(const agx::Path& storagePath, agx::UInt32 instanceId, const agx::String& attributeName);

    /**
    Load bindings from an xml file.

    Syntax example:
    \verbatim <DataFrame>
      <DataBinding internal="foo" external="bar"/>
      <DataBinding path="baz"/>
    </DataFrame>
    \endverbatim
    */
    void loadDataBindings(const agx::String& xmlFilePath);

    bool loadDataBindingsJSON(const agx::String& jsonFilePath);



    /**
    \return The list of registered data bindings.
    If no data binding is registered the Reader/Writer implementation is supposed to transfer *everything*
    */
    const Frame::DataBindingRefVector& getDataBindings() const;

    /**
    Remove all data bindings.
    */
    void clearDataBindings();


    void setHeaderFormat(SerializedFrame::HeaderFormat format);
    SerializedFrame::HeaderFormat getHeaderFormat() const;

  protected:
    FrameIO(bool active);
    virtual ~FrameIO();

    void addDataBinding(Frame::DataBinding *binding);

  private:
    friend class Track;
    void setTrack(Track *track);

    // Implementations may override this
    virtual Frame::DataBinding *createDataBinding(const agx::Path& internalPath, const agx::Path& externalPath);

  private:
    bool m_active;
    Track *m_track;
    Frame::DataBindingRefVector m_dataBindings;
    SerializedFrame::HeaderFormat m_headerFormat;
  };


  /**
  The FrameReader reads frames from an external source.
  */
  class AGXCORE_EXPORT FrameReader : public FrameIO
  {
  public:
    FrameReader(bool active);


    // This was removed due to thread synchronizing issues
    // virtual agx::UInt getNextFrameSize() = 0;

    /**
    Reads the next frame from the stream and moves the stream forward one frame. The Stream
    will perform memory allocations for the buffer contents, but the data is owned by the
    returned Frame instance.

    \return The next frame in the stream. nullptr if end-of-stream.
    */
    virtual Frame* readFrame() = 0;

    virtual agx::UInt getNextFrameIndex();
    virtual void advanceToNextFrame();

    /**
    Select which frame that should be returned by the next call to 'readFrame'.
    \return false if jumping is not allowed
    */
    virtual bool jumpToFrame(agx::UInt frameIndex);
    virtual bool jumpToTime(agx::Real time);


    virtual bool shouldExtract(const agx::Path& path);

    virtual agx::Real getTimeStep();
    virtual agx::UInt getFrameStride();

  protected:
    virtual ~FrameReader();
  };


  /**
  The FrameWriter writes frames to an external target.
  */
  class AGXCORE_EXPORT FrameWriter : public FrameIO
  {
  public:
    typedef agx::Event1<const Frame*> WriteEvent;
    WriteEvent writeEvent;

  public:
    FrameWriter(bool active);

    /**
    Export a frame the implementation-specific format.
    */
    virtual void writeFrame(const Frame *frame) = 0;

    // TODO: Clean redesign, this is ugly
    virtual void forceFlush() {}


    virtual void endOfStream() {}

    agx::UInt getNumWrittenFrames() const;

    bool isEOF() const;

  protected:
    virtual ~FrameWriter();

    agx::UInt m_numWrittenFrames;
  private:
    friend class Track;
    void setEOF(bool flag);

  private:
    bool m_eof;
  };


  /**
  The DebugFrameWriter prints the contents processed frames.
  */
  class AGXCORE_EXPORT DebugFrameWriter : public FrameWriter
  {
  public:
    DebugFrameWriter();

    virtual void writeFrame(const agxData::Frame *frame) override;

  protected:
    virtual ~DebugFrameWriter();
  };




  /**
  Convenience class that store a FrameReader and a FrameWriter.
  */
  class AGXCORE_EXPORT FrameChannel : public agx::Referenced
  {
  public:
    FrameChannel(FrameReader* reader, FrameWriter* writer) : m_reader(reader), m_writer(writer) {};

    FrameReader* getReader() { return m_reader; }
    FrameWriter* getWriter() { return m_writer; }

  protected:
    virtual ~FrameChannel() {};

  private:
    FrameReaderRef m_reader;
    FrameWriterRef m_writer;
  };



  /* Implementation */

  //---------------------------------------------------------------

  AGX_FORCE_INLINE bool FrameIO::isActive() const { return m_active; }
  AGX_FORCE_INLINE Track *FrameIO::getTrack() { return m_track; }
  AGX_FORCE_INLINE const Track *FrameIO::getTrack() const { return m_track; }

  AGX_FORCE_INLINE const Frame::DataBindingRefVector& FrameIO::getDataBindings() const { return m_dataBindings; }
  AGX_FORCE_INLINE SerializedFrame::HeaderFormat FrameIO::getHeaderFormat() const { return m_headerFormat; }

  //---------------------------------------------------------------


  //---------------------------------------------------------------

  AGX_FORCE_INLINE bool FrameWriter::isEOF() const { return m_eof; }
  AGX_FORCE_INLINE agx::UInt FrameWriter::getNumWrittenFrames() const { return m_numWrittenFrames; }

  //---------------------------------------------------------------

}


#endif /* AGXDATA_FRAMEIO_H */
