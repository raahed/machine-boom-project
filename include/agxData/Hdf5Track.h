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



#ifndef AGXDATA_HDF5_TRACK_H
#define AGXDATA_HDF5_TRACK_H

#include <agx/config/AGX_USE_HDF5.h>
#include <agx/config.h>

#if AGX_USE_HDF5()

#include <agx/agx.h>
#include <agxData/DiskTrack.h>
#include <agxData/Hdf5Journal.h>

#include <agxData/SerializedFrame.h>

#include <agx/AgxH5.h>

namespace agxData
{


  AGX_DECLARE_POINTER_TYPES( Hdf5FrameReader );

  /**
  Frame reader that reads frames from journals stored in the HDF5 format.
  */
  class AGXCORE_EXPORT Hdf5FrameReader : public agxData::DiskFrameReader
  {
  public:
    Hdf5FrameReader( agxData::JournalArchive::Session* session );


    virtual const agxData::JournalArchive::Session* getSession() const override;


    /**
    Change which frame that will be read next. If the given frame index is
    before the start of the session then the first frame will be the next one
    read and jumptoFrame returns false. If the given frame index is after the
    last frame in the session then the next call to 'readFrame()' will fail
    unless the missing frame is created before then. jumpToFrame return false
    in this case.

    \return True if the jump was successful, false otherwise.
    */
    virtual bool jumpToFrame( agx::UInt frameIndex ) override;

    /// Jump to the frame closest to the given time. If the given time is
    /// outside of the range of the session, then the next call to
    /// 'readFrame()' will fail.
    virtual bool jumpToTime( agx::Real time ) override;

    /// \return The index of the first frame in the track.
    agx::UInt getFrameOffset() const;

    using agxData::DiskFrameReader::readFrame;
  protected:
    virtual ~Hdf5FrameReader();

    virtual agxData::Frame* readFrame( agx::UInt frameIndex ) override;

  private:
    bool isValidFrame( agx::UInt frameIndex ) const;

    void extractEvents( H5::Group& frameNode, agx::Object::EventRefVector& events );

    SerializedFrame::Component* createFrame( H5::Group& frameNode );
    SerializedFrame::Component* createComponent( H5::Group& componentNode, const H5std_string& name );
    SerializedFrame::EntityStorage* createEntityStorage( H5::Group& storageNode, const H5std_string& name );
    SerializedFrame::Node* createBuffer( H5::Group& bufferNode, const H5std_string& name );
    SerializedFrame::Node* createValue( H5::Group& valueNode, const H5std_string& name );
    void addChildren( agxData::SerializedFrame::Component* frameNode, H5::Group& hdf5Node );
    void addBuffers( agxData::SerializedFrame::Component* frameNode, H5::Group& hdf5Node );
    void addValues( agxData::SerializedFrame::Component* frameNode, H5::Group& hdf5Node );

    void readComponent();
    void readBuffer();
    void readValue();


  private:
    agxData::Hdf5Journal::Session* m_session;
    agx::Path m_nodePath;
  };

  AGX_DECLARE_POINTER_TYPES( Hdf5FrameWriter );

  class AGXCORE_EXPORT Hdf5FrameWriter : public agxData::DiskFrameWriter
  {
  public:
    Hdf5FrameWriter( agxData::JournalArchive::Session* session );

    virtual void writeFrame( const agxData::Frame* frame ) override;

    virtual const JournalArchive::Session* getSession() const override;

    agx::UInt getNumFramesWritten() const;

  protected:
    virtual ~Hdf5FrameWriter();

  private:
    static void write( H5::CommonFG& parent, SerializedFrame::EntityStorage* storage );
    static void write( H5::CommonFG& parent, SerializedFrame::CustomBuffer* buffer );
    static void write( H5::CommonFG& parent, SerializedFrame::PartialBuffer *buffer );
    static void write( H5::CommonFG& parent, SerializedFrame::Buffer* buffer );
    static void write( H5::CommonFG& parent, SerializedFrame::Value* value );
    static void write( H5::CommonFG& parent, SerializedFrame::Component* component );
    static void write( H5::CommonFG& parent, SerializedFrame::Node* node );
    static void writeEvents( H5::CommonFG& parent, const agx::Object::EventRefVector& events );

  public:
    static void writeChildren( H5::CommonFG& parent, const agxData::SerializedFrame::Component* component );


  private:
    agx::UInt m_numFramesWritten;
    agxData::Hdf5Journal::SessionRef m_session;
  };


}

// AGX_USE_HDF5
#endif


// Include guard
#endif
