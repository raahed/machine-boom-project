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

#ifndef AGXDATA_FRAMETRANSPOSE_H
#define AGXDATA_FRAMETRANSPOSE_H

#include <agxData/FrameIO.h>
#include <agxData/FileTrack.h>
#include <agxData/Track.h>
#include <agx/Range.h>

namespace agxData
{
  AGX_DECLARE_POINTER_TYPES(FrameTranspose);
  class AGXCORE_EXPORT FrameTranspose : public FrameWriter
  {
  public:
    AGX_DECLARE_POINTER_TYPES(Track);
    AGX_DECLARE_VECTOR_TYPES(Track);

    AGX_DECLARE_POINTER_TYPES(AttributeTrack);
    AGX_DECLARE_POINTER_TYPES(ValueTrack);

  public:

    FrameTranspose(bool active = true);

    FrameTranspose(agxData::JournalArchive::Session* session);

    /**
    Register a track to be filled with transpose data.
    */
    void registerTrack(Track *track);
    AttributeTrack *registerTrack(const agx::Path& storagePath, const agx::Name& attributeName, agx::Index instanceId);
    ValueTrack *registerTrack(const agx::Path& valuePath);

    /**
    \return The registered track
    */
    const TrackRefVector& getTracks() const;

    /**
    \return The time track, which is automatically registered
    */
    agxData::Buffer *getTimeTrack();
    const agxData::Buffer *getTimeTrack() const;

    /**
    \return The index track, which is automatically registered
    */
    agxData::Buffer *getIndexTrack();
    const agxData::Buffer *getIndexTrack() const;


    /**
    Extract all frames from track. Assumes EOF is reached at some point.
    Must not be called if FrameTranspose is created as non-active.
    */
    void extractFrames();

    /**
    Extract all frames from track within a time range. Assumes EOF is reached at some point.
    Must not be called if FrameTranspose is created as non-active.
    \param timeRange - Range of simulation time (in seconds)
    */
    void extractFrames(agx::RangeReal timeRange);


    void setFrameStride(agx::UInt stride);

  protected:
    virtual ~FrameTranspose();

    virtual void writeFrame(const Frame *frame);
    virtual Frame::DataBinding *createDataBinding(const agx::Path& internalPath, const agx::Path& externalPath);

  private:
    void updateTrack(Track *track, const Frame *frame);
    void updateAttributeTrack(AttributeTrack *track, const Frame *frame);
    void updateValueTrack(ValueTrack *track, const Frame *frame);


  private:
    TrackRefVector m_tracks;
    agxData::BufferRef m_timeTrack;
    agxData::BufferRef m_indexTrack;
    agxData::TrackRef m_ownedTrack;
    agxData::DiskFrameReaderRef m_fileReader;
  };


  class AGXCORE_EXPORT FrameTranspose::Track : public Referenced
  {
  public:
    Track();
    Track(const agx::Path& dataPath);

    /**
    \return The time range for the track.
    */
    agx::RangeReal getTimeRange() const;

    /**
    \return The data path.
    */
    const agx::Path& getDataPath() const;

    /**
    \return The track data values.
    */
    agxData::Buffer *getData();

    /**
    \return The track data values.
    */
    const agxData::Buffer *getData() const;

  protected:
    virtual ~Track();

  private:
    friend class FrameTranspose;
    agxData::BufferRef m_idToIndexBuffer;

    agx::RangeReal m_timeRange;
    BufferRef m_data;
    agx::Path m_dataPath;
  };


  class AGXCORE_EXPORT FrameTranspose::AttributeTrack : public Track
  {
  public:
    AttributeTrack(const agx::Path& storagePath, const agx::Name& attributeName, agx::Index instanceId);

    const agx::Path& getStoragePath() const;
    const agx::Name& getAttributeName() const;
    agx::Index getInstanceId() const;

  protected:
    virtual ~AttributeTrack();

  private:
    friend class FrameTranspose;
    agx::Path m_storagePath;
    agx::Name m_attributeName;
    agx::Index m_instanceId;
    agxData::BufferRef m_currentFrameAttributeBuffer;
  };

  class AGXCORE_EXPORT FrameTranspose::ValueTrack : public Track
  {
  public:
    ValueTrack(const agx::Path& valuePath);
  protected:
    virtual ~ValueTrack();

  private:
    friend class FrameTranspose;

    agxData::ValueRef m_currentFrameValue;
  };


  /* Implementation */

  DOXYGEN_START_INTERNAL_BLOCK()
  AGX_FORCE_INLINE const FrameTranspose::TrackRefVector& FrameTranspose::getTracks() const { return m_tracks; }
  AGX_FORCE_INLINE agxData::Buffer *FrameTranspose::getTimeTrack() { return m_timeTrack; }
  AGX_FORCE_INLINE agxData::Buffer *FrameTranspose::getIndexTrack() { return m_indexTrack; }
  AGX_FORCE_INLINE const agxData::Buffer *FrameTranspose::getTimeTrack() const { return m_timeTrack; }
  AGX_FORCE_INLINE const agxData::Buffer *FrameTranspose::getIndexTrack() const { return m_indexTrack; }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE agx::RangeReal FrameTranspose::Track::getTimeRange() const { return m_timeRange; }
  AGX_FORCE_INLINE agxData::Buffer *FrameTranspose::Track::getData() { return m_data; }
  AGX_FORCE_INLINE const agxData::Buffer *FrameTranspose::Track::getData() const { return m_data; }
  AGX_FORCE_INLINE const agx::Path& FrameTranspose::Track::getDataPath() const { return m_dataPath; }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE const agx::Path& FrameTranspose::AttributeTrack::getStoragePath() const { return m_storagePath; }
  AGX_FORCE_INLINE const agx::Name& FrameTranspose::AttributeTrack::getAttributeName() const { return m_attributeName; }
  AGX_FORCE_INLINE agx::Index FrameTranspose::AttributeTrack::getInstanceId() const { return m_instanceId; }

  //---------------------------------------------------------------
  DOXYGEN_END_INTERNAL_BLOCK()
}


#endif /* AGXDATA_FRAMETRANSPOSE_H */
