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

#ifndef AGXDATA_TRACK_H
#define AGXDATA_TRACK_H

#include <agx/Component.h>
#include <agx/Clock.h>
#include <agxData/Frame.h>
#include <agxData/FrameIO.h>
#include <agx/List.h>
#include <agx/Thread.h>

#include <atomic>
#include <condition_variable>
#include <mutex>


namespace agxData
{
  AGX_DECLARE_POINTER_TYPES(Track);
  class AGXCORE_EXPORT Track : public agx::Object
  {
  public:
    static agx::Model *ClassModel();

    AGX_DECLARE_POINTER_TYPES(Thread);
    AGX_DECLARE_POINTER_TYPES(InputThread);
    AGX_DECLARE_POINTER_TYPES(OutputThread);

  public:
    typedef agx::Event1<Frame *> FrameEvent;
    FrameEvent pushFrameEvent;
    FrameEvent popFrameEvent;

  public:
    typedef agx::Callback2<Track *, std::exception&> ExceptionHandler;

  public:
    Track(const agx::Name& name = "Track");
    Track(FrameReader *input, FrameWriter *output);

    /**
    Set the input stream.
    */
    void setInput(FrameReader *input);

    /**
    \return The input stream.
    */
    FrameReader *getInput();
    const FrameReader *getInput() const;

    /**
    Specify the output stream.
    */
    void setOutput(FrameWriter *output);


    /**
    Start the track, producing frames from the input and consuming frames at the
    outputs. Should be called once the input and all outputs are set up. Outputs
    added after the track is started are not guaranteed to receive all frames.
    */
    void start();

    /**
     Stop frame production. Will block until both the input- and output thread have ceased to run.
     */
    void stop();

    /**
    \return The output streams.
    */
    // const FrameWriterRefVector& getOutputs() const;
    FrameWriter *getOutput();
    const FrameWriter *getOutput() const;

    /**
    Specify an exception handler.
    */
    void setExceptionHandler(const ExceptionHandler& handler);

    /**
    \return The current exception handler.
    */
    const ExceptionHandler& getExceptionHandler() const;

    /**
    Set the maximum number of bytes in the frame cache.
    */
    void setMaxNumBytes(agx::UInt numBytes);

    /**
    \return the maximum number of bytes in the frame cache.
    */
    agx::UInt getMaxNumBytes() const;

    /**
    Set the real mode.
    */
    void setRealMode(agx::RealModeEnum mode);

    /**
    \return The real mode.
    */
    agx::RealModeEnum getRealMode() const;

    /**
    \return Data rate statistics in KB/s
    */
    agx::Real getDataRate() const;

    /**
    \return the number of bytes used by active frames in the track queue/cache.
    */
    agx::UInt getNumQueuedBytes() const;
    agx::UInt getNumCachedBytes() const;


    /**
     \return the number of active frames in the track queue/cache.
     */
    agx::UInt getNumQueuedFrames() const;
    agx::UInt getNumCachedFrames() const;

    /**
    \return The total number of bytes transferred through the track.
    */
    agx::UInt getAccumulatedNumBytes() const;

    /**
    \return The total number of frames transferred through the track.
    */
    agx::UInt getAccumulatedNumFrames() const;

    /**
    Flush all frames to the registered outputs. Blocking.
    \param force Wait for all outputs to consume currently cached frames if false
    */
    void flush(bool force = false);

    /**
    Determine if the track is empty. In other words, if the last frame has
    been read by the output. May give false negatives since the input may
    later realize that it doesn't have any more frames to add and at that time
    mark the track as empty.

    \return true if the input has reached EOF and the queue is empty.
    */
    bool isQueueEmpty() const;

    /**
     \return true if the last frame has been read by the output.
     */
    bool isEOF() const;

    /**
    \return true if the input has signaled EOF.
    */
    bool isInputEOF() const;

    /**
    \return true if either input or output part is running
    */
    bool isRunning() const;

    /**
    Set to true to enable looping from the input.
    */
    void setEnableLooping(bool flag);

    /**
    \return True if looping is enabled.
    */
    bool getEnableLooping() const;

    /**
    Set to true to expand all partial frames to keyframes.
    */
    void setExpandFrames(bool flag);

    /**
    \return true if partial frames are expanded.
    */
    bool getExpandFrames() const;

    /**
    Select which frame that should be returned by the next call to 'readFrame'.
    If the jump fails for a track that uses a input frame reader that supports jumping,  /// \todo This needs to be fixed.
    then the caller should jump to some other frame that can be jumped to in order to
    restore correct behavior.
    \return false if jumping is not allowed or if the jump failed.
    */
    bool jumpToFrame(agx::UInt frameIndex);
    bool jumpToTime(agx::Real time);

    /**
    \return True if the track has jumped since last frame was read from input
    */
    bool hasJumped() const;

    /**
    Set frame stride. Default is 1.
    */
    void setFrameStride(agx::UInt stride);

    /**
    Set the frame stride using a frequency, which is converted to a stride.
    \return The actual frequency, an even multiple of the source frequency
    */
    agx::Real setFrameFrequency(agx::Real frequency);

    /**
    \return The frame stride.
    */
    agx::UInt getFrameStride() const;


    void setActiveTimeRangeStart(agx::Real time);
    void setActiveTimeRangeEnd(agx::Real time);

    agx::Real getActiveTimeRangeStart() const;
    agx::Real getActiveTimeRangeEnd() const;

    /**
    Load cached frames from another track.
    */
    void loadCache(Track *other);


    // Should only be called by the input
    void pushFrame(Frame *frame);

    // Should only be called by the output.
    FrameRef popFrame(bool blocking = true);

  protected:
    virtual ~Track();

  private:
    // friend class Thread;

    // The mutex must be held when calling this method.
    FrameRef _popFrame();
    void finalizePop(Frame *frame);
    void flush(bool force, bool needLock);

    void signalEOF();
    void resetEOF();
    void updateDataRate(Frame *frame);

    // Waits on the hasFrame condition variable until the frame buffer has
    // frames or the condition has been forced open.
    //
    // \return True if there is a frame in the queue and the condition
    // variable hasn't been forced open. False if the condition variable
    // has been forced open.
    bool waitForFrame();

    // Waits on the hasFreeSlots condition variable until the frame buffer has
    // free slots of the condition has been forced open.
    //
    // \return True if there is a free slot in the queue and the condition
    // variable hasn't been force open. False if the condition variable has been
    // forced open.
    bool waitForFreeSlot();

    void clearQueue();
    void _setFrameStride(agx::UInt stride);

  private:
    typedef agx::List<FrameRef> FrameList;
    typedef std::pair<FrameRef, FrameList::iterator> FrameEntry;
    typedef agx::HashTable<agx::UInt, FrameEntry> FrameTable;


    class FrameCache
    {
    public:
      FrameCache();
      FrameCache(const FrameCache& other);

      FrameCache& operator= (const FrameCache& other);

      void addFrame(Frame *frame);
      Frame *getFrame(agx::UInt frameIndex);
      bool evictFrame();
      void evictFrame(Frame *frame);

      agx::Real getHitRate() const;
      void clear();

      bool isFull() const;

      agx::UInt getNumFrames() const;
      agx::UInt getNumBytes() const;

      void setMaxNumBytes(agx::UInt numBytes);

      agx::UInt getMaxNumBytes() const;

    private:
      FrameList m_frameList;
      FrameTable m_frameTable;

      agx::UInt m_numRequests;
      agx::UInt m_numHits;
      agx::UInt m_numBytes;
      agx::UInt m_maxNumBytes;
    };


    FrameCache m_frameCache;
    FrameList m_frameQueue;

    FrameReaderRef m_input;
    FrameWriterRef m_output;

    ExceptionHandler m_exceptionHandler;

    mutable std::mutex m_mutex;
    std::mutex m_outputVectorMutex;
    std::mutex m_jumpMutex; // Locked when jumping between frames is forbidden.

    std::condition_variable_any m_hasFramesCondition;
    std::atomic<bool> m_hasFramesForceOpen;
    std::condition_variable_any m_hasFreeSlotsCondition;
    std::atomic<bool> m_hasFreeSlotsForceOpen;

    InputThreadRef m_inputThread;
    OutputThreadRef m_outputThread;

    agx::UInt m_numBytes;
    agx::UInt m_accumulatedNumBytes;
    agx::UInt m_accumulatedNumFrames;
    bool m_inputEOF;
    bool m_enableLooping;
    bool m_hasJumped;
    bool m_isRunning;
    agx::RealModeEnum m_realMode;
    agx::Real m_dataRate;
    agx::UInt m_dataRateCounter;
    agx::Timer m_dataRateTimer;
    agx::UInt m_frameStride;
    agx::Real m_activeTimeStart;
    agx::Real m_activeTimeEnd;

    ThreadRef m_exceptionThread;

    bool m_expandFrames;
    FrameRef m_expandedFrame;
  };


  class Track::Thread : public agx::BasicThread, public agx::Component
  {
  public:
    Thread(Track *track, agx::Callback entryPoint, const agx::Name& = "Thread");

    /**
    Start the thread, and optionally activate it.
    */
    void start(bool activate = true);

    /**
    Stop the thread.
    */
    void stop();

    /**
    \return True if the thread is running.
    */
    bool isRunning() const;

    /**
    Activate the thread.
    Thread state progression: stopped -> started -> [active -> inactive]* -> stopped
    In the sleeping state the OS thread is still running but inactivated by a
    semaphore. This to enable fast activation irregardless of OS implementation.
    */
    void activate();

    /**
    Deactivate the thread.
    */
    void deactivate();

    /**
    \return True if active.
    */
    bool isActive() const;

    Track *getTrack() { return m_track; }

  protected:
    virtual ~Thread();

  private:
    virtual void run(); // Do not override this

  private:
    Track *m_track;
    agx::Callback m_callback;
    std::atomic<bool> m_running;
    std::atomic<bool> m_active;
    agx::Block m_startBlock;
    agx::Block m_activationBlock;
  };

  class Track::InputThread : public Track::Thread
  {
  public:
    InputThread(Track *track);

  protected:
    virtual ~InputThread();

  private:
    void execute();
    // Frame *readFrame();
  };

  class Track::OutputThread : public Track::Thread
  {
  public:
    OutputThread(Track *track);

  protected:
    virtual ~OutputThread();


  private:
    void execute();
  };


  /* Implementation */

  AGX_FORCE_INLINE FrameReader *Track::getInput() { return m_input; }
  AGX_FORCE_INLINE const FrameReader *Track::getInput() const { return m_input; }
  // AGX_FORCE_INLINE const FrameWriterRefVector& Track::getOutputs() const { return m_outputs; }
  AGX_FORCE_INLINE FrameWriter *Track::getOutput() { return m_output; }
  AGX_FORCE_INLINE const FrameWriter *Track::getOutput() const { return m_output; }

  AGX_FORCE_INLINE bool Track::isInputEOF() const { return m_inputEOF; }
  AGX_FORCE_INLINE bool Track::isEOF() const { return m_output->isEOF(); }
  AGX_FORCE_INLINE bool Track::getEnableLooping() const { return m_enableLooping; }

  AGX_FORCE_INLINE agx::UInt Track::getFrameStride() const { return m_frameStride; }
  AGX_FORCE_INLINE bool Track::hasJumped() const { return m_hasJumped; }


  AGX_FORCE_INLINE agx::UInt Track::getAccumulatedNumBytes() const { return m_accumulatedNumBytes; }
  AGX_FORCE_INLINE agx::UInt Track::getAccumulatedNumFrames() const { return m_accumulatedNumFrames; }

  AGX_FORCE_INLINE agx::Real Track::getDataRate() const
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_dataRate;
  }

  AGX_FORCE_INLINE bool Track::Thread::isRunning() const { return m_running; }
  AGX_FORCE_INLINE bool Track::Thread::isActive() const { return m_active; }

}


#endif /* AGXDATA_TRACK_H */
