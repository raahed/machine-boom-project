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

#ifndef AGXDATA_DISKTRACK_H
#define AGXDATA_DISKTRACK_H


#include <agxData/FrameIO.h>
#include <agxData/JournalArchive.h>


namespace agxData
{
  AGX_DECLARE_POINTER_TYPES( DiskFrameReader );

  /**
  Helper class for the FrameReaders that are backed by a file on disk.
  Provides a few algorithms that uses subclass dependent operations realized
  through virtual methods.
  */
  class AGXCORE_EXPORT DiskFrameReader : public agxData::FrameReader
  {
  public:

    /**
    Reads the next frame from disk using the virtual readFrame(UInt) method.
    Does frame merging from previous key frame when needed.
    */
    virtual Frame* readFrame() override;

    virtual agx::UInt getNextFrameIndex() override;
    virtual void advanceToNextFrame() override;

    /**
    \return The session that frames are being read from. Is always a subclass
            of Journal::Session suitable for the type of the current
            DiskFrameReader instance.
    */
    virtual const JournalArchive::Session* getSession() const = 0;


    /// \return The index of the frame that will be read on the next call to
    /// \'readFrame()'.
    agx::UInt getFrameIndexToRead() const;

    virtual bool shouldExtract(const agx::Path& path) override;

    virtual agx::Real getTimeStep() override;
    virtual agx::UInt getFrameStride() override;


    /**
    \return The frame index of the frame closes to the given time.
    */
    agx::UInt findFrameAtTime(agx::Real time) const;

  protected:
    DiskFrameReader();

    /**
    Read a frame from disk. Implemented by the subclasses.
    */
    virtual agxData::Frame* readFrame( agx::UInt frameIndex ) = 0;

    virtual void prepareSequence(agx::UInt startFrame, agx::UInt lastFrame);
    virtual void endSequence();

  private:
    void initBindTables();
    void updateExclusions(const Frame *frame, const Object *frameNode);
    agxData::Frame* mergeFrameSequence(agx::UInt startFrame);
    agx::UInt getSequenceStartFrame(agx::UInt lowerBound = 0) const;

  protected:
    agx::Mutex m_mutex;
    agx::UInt m_frameToRead;
    FrameRef m_currentFrame;

    FrameRefVector m_frameMergeSequence;

    typedef agx::HashSet<agx::Path> BindingTable;


    BindingTable m_bindingTable;
    BindingTable m_exclusions;
  };


  AGX_DECLARE_POINTER_TYPES( DiskFrameWriter );

  /**
  Base class for the FrameReaders that are backed by a file on disk.
  */
  class AGXCORE_EXPORT DiskFrameWriter : public agxData::FrameWriter
  {
  public:
    DiskFrameWriter();

    /**
    \return The session that frames are being read from. Is always a subclass
            of Journal::Session suitable for the type of the current
            DiskFrameReader instance.
    */
    virtual const JournalArchive::Session* getSession() const = 0;

  protected:
  };
}

#endif /* AGXDATA_DISKTRACK_H */
