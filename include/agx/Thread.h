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

#pragma once

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable: 4275 )  //  warning C4275: non dll-interface class
#endif


#include <agx/agx.h>
#include <agx/Vector.h>
#include <agx/HashTable.h>
#include <agx/Timer.h>
#include <agx/Task.h>
#include <agx/ThreadSynchronization.h>

#include <agxData/AttributeContainer.h>
#include <agxData/EntityStorage.h>

#include <agx/AtomicValue.h>
#include <agx/Job.h>
#include <agx/Uuid.h>
#include <agx/Notify.h>

#include <queue>
#include <thread>

// Set to 1 to enable thread timeline reporting of internal steps
#define ENABLE_VERBOSE_THREAD_TIMELINE 0


#define AGX_MAX_NUM_THREADS 1024


extern "C"
{
  void agxFlushThreadLogs();
}

namespace agxData
{
  class EntityStorage;
  class EntityModel;
}





namespace agx
{
  class TiXmlElement;
  class Task;


  // AGX_DECLARE_POINTER_TYPES(Thread);
  // AGX_DECLARE_VECTOR_TYPES(Thread);



  /**
  Basic wrapper class aroud std::thread. The work will be done in the run method.

  All threads that are started should either be joined or detached.
  */
  class AGXCORE_EXPORT BasicThread
  {
    public:

      /**
      States for the thread.
      */
      enum ThreadState {
        THREAD_NEW       = 0, // When a BasicThread is created
        THREAD_RUNNING   = 1, // start has been called
        THREAD_DETACHED  = 2, // detach has been called
        THREAD_DONE      = 3, // After run() is done executing
        THREAD_JOINED    = 4, // After join() has been called
        THREAD_CANCELLED = 5  // After cancel but before join
      };

      /**
      Default constructor
      */
      BasicThread();

      BasicThread( const BasicThread& other ) = delete;

      BasicThread& operator=(const BasicThread& rhs ) = delete;

      /**
      Destructor
      */
      virtual ~BasicThread() = default;

      /**
      This method is invoked by start. Override in child classes to
      perform the work.
      */
      virtual void run() { }

      /**
      Launches the thread. Will execute the run method.
      If a thread is already started and we have a joinable handle,
      start will not launch another thread and instead return false.

      Returns true on success.
      */
      bool start();

      /**
      Threads should normally not need to be killed.
      Killing a thread can have bad consequences and leak resources.
      This will forcefully terminate the thread.

      The thread should still be joined even after cancel has been called.
      */
      void cancel();

      /**
      Joins the thread. Will block until the thread is done executing.
      */
      bool join();


      /**
      Detaches the thread to the background. Will no longer be joinable.
      */
      void detach();

      /**
      True if thread is joinable
      */
      inline bool joinable();

      /**
      Returns the current thread state
      */
      inline unsigned int getThreadState();




      /**
      Thread Affinity can be used to influence on which logical cores threads
      are scheduled and allowed to run.

      On Windows and Linux, the input is a bitmask where the bits that are set
      are maps to logical cores.

      On OS X, the input value is an affinity tag and threads with the same tag
      are scheduled so they should share and L2 cache.

      Note: If the BasicThread has not yet been launched via start(), then this
            method will affect the current executing thread.
      */
      bool setThreadAffinity(agx::UInt64 cpumask);


      /**
      Return a native_handle for the current executing thread
      */
      static std::thread::native_handle_type getCurrentThreadHandle();


   private:
      void launchThread();

    protected:
      std::thread m_handle;
      std::atomic< unsigned int> m_state;
    private:
      std::mutex m_handleMutex;
  };



  /**
  agx::Thread is a representation of an OS specific implementation of a computational thread. Threads
  can execute generic tasks (agx::Task) and jobs (agx::Job).
  */
  class AGXCORE_EXPORT Thread : public BasicThread, public agxData::AttributeContainer
  {
  public:


    // The types used when communicating with the thread local storage facilities.
    // The key type is an integer type and the data type is a pointer type.
    #ifdef _WIN32
      typedef unsigned long ThreadStorageKey; // Should be DWORD
      typedef void* ThreadStorageData; // Should be LPVOID
    #else
      typedef pthread_key_t ThreadStorageKey;
      typedef void* ThreadStorageData;
    #endif

      typedef std::mt19937 RandomGenerator;

    /**
    Start the thread.
    */
    bool start();

    /**
    Stop the thread.
    */
    void stop();

    /**
    Returns the thread's AGX thread ID, a value between 0 and N-1, for
    AGX's internal threads, i.e. threads created in response to a call to
    agx::setNumThreads. For threads created using Thread::registerAsAgxThread
    agx::InvalidIndex is returned.
    */
    Index getId() const;

    /**
    Get an index suitable for use when storing per-thread data in e.g. a Vector.
    The returned index is the thread ID for regular AGX worker threads and a
    value larger than the largest thread ID for non-AGX threads that has called
    registerAsAgxThread.

    The value returned may change when the number of AGX threads is changed, and
    when other threads are registered or unregistered as AGX threads. It is therefor
    not recommended to store this value for longer than a single time step.
    */
    Index getIndex() const;

    /**
    \return the current thread, or nullptr if not a registered AgxThread (see registerAsAgxThread).
    */
    static Thread *getCurrentThread();

    /**
    \return a string describing the current thread.
    The format is "os, agx" where 'os' is an OS specific identifier (typically a number)
    and 'agx' is the agx thread id, or null if this is not an agx thread
    */
    static std::string getCurrentThreadDescription();

    /**
    \return a specific thread.
    */
    static Thread *getThread(size_t id);

    /**
    \return the main thread.
    */
    static Thread *getMainThread();

    /**
    Register current thread as main thread.
    NOTE: The old main thread will also think it is the main thread!
    */
    static void makeCurrentThreadMainThread();

    /**
    Register the current thread as an AGX thread. Necessary to execute
    agx::Task instances. This is normally handled automatically.
    */
    static Thread *registerAsAgxThread();

    /**
    Remove agx attributes from current thread. This is normally
    handled automatically.
    */
    static void unregisterAsAgxThread();

    /**
    Return a reference to the mersienne twister used for generating random numbers.
    This is local to the thread and can be used safely from the current thread
    \returns reference to a mt19937 random generator
    */
    RandomGenerator& getRandomGenerator();


    /**
    \return True if main thread.
    */
    static bool isMainThread();


    /**
    Shutdown the threading system. After this no threads will be available.
    */
    static void shutdown();

    static bool isShuttingDown();


    /** Enable or disable job timeline statistics */
    static void setEnableJobTimeline(bool flag);
    static bool getEnableJobTimeline();

    /**
     * Add an entry to the job duration log. Intended to be used when the system
     * performs some lengthy operation without using the job system.
     *
     * This will NOT copy the content of 'description' and 'datatitle', so
     * don't pass pointers to anything that will be deallocated before the next
     * disk writing.
     */
    void reportSystemJob(
       UInt64 startTick, UInt64 endTick,
       const char* description,
       const char* extraDataTitle = nullptr,
       agx::Real64 extraData = 0.0);

    /**
    \return The overhead time for the thread, including jobpool stalls and job administration.
    */
    Real getOverheadTime() const;

    /**
    Reset the thread overhead time.
    */
    void resetOverheadTime();

    // Allocations
    // void *allocateBytes(size_t numBytes);
    // void *deallocateBytes(void *ptr);

    /**
    Generates a unique universal identifier
    \return a Unique Universal identifier
    */
    agx::Uuid generateUuid();








    // Main thread is normally allocated automatically, special usage before main is entered
    static void initThreadSystem();

    // Dump profiling logs, called by Simulation
    static void exportAllTimelines();
    static void resetStartTick();
    static void flushTimelineLogs();


    /*
     * Methods used for manipulating the thread local storage.
     */

    /** Allocate a storage location that is unique for each thread. This key will be usable from all threads. */
    static ThreadStorageKey allocatePerThreadStorage();
    /** Read the thread-local value for the currently executing thread associated with the given key. */
    static ThreadStorageData readPerThreadStorage( ThreadStorageKey key );
    /** Write to the thread-local location owned by the currently executing thread. */
    static void writePerThreadStorage( ThreadStorageKey key, ThreadStorageData data);
    /** Deallocate the storage location. No thread can use 'key' after this method has been called. */
    static void freePerThreadStorage( ThreadStorageKey key );

    static bool immediateLogging;
    static int log(const char *format, ...);
    static void flushLogs();
    agxData::EntityStorage *getDefaultStorage(agxData::EntityModel *entity);

    agxData::EntityStorageRef popTimelineEntryStorage();


    static void addTask(Task *task);

    // Threads that aren't "real" AGX threads, i.e., registered threads, get
    // a pseudo-ID in this table. These IDs are unrelated from the IDs of the
    // "real" AGX threads.
    typedef HashTable<Thread*, Index> ThreadIdTable;
    static Thread::ThreadIdTable* getPromotedThreads();

  protected:
    Thread(Index id);
    Thread(const Thread&) = delete;
    virtual ~Thread();

  private:



    friend void AGXCORE_EXPORT setNumThreads(size_t numThreads);
    friend void AGXPHYSICS_EXPORT shutdown();

    class MainThreadSingleton;
    static Thread *initMainThread();

    static void performNumThreadsChange();
    void startWaitTimer();
    void stopWaitTimer();

    virtual void run();

    void freeDefaultStorages();
    void spawn(Job *job);
    void sortInsertJob(Job *job);

    void activate();
    void completeFrame(Task *task);

    bool isActive() const;


    template <bool THREAD_TIMELINE_STATISTICS>
    void doWork();

    void doWork();
    void blockingDoWork();

    void stealWork();
    void wakeupThreads();
    void pushTargetJob(Thread *target, Job *job, bool activateTarget = true);
    static void taskCompleted(Task *task);

    void initialize();
    void sleep();

    Job *getExecutionJob();

    Index getRandomOtherThreadId(Index excludeIndex = InvalidIndex);

    friend class Block;

    friend class Job;
    friend class Notify;

  private:
    char *m_currentPathString;

    Notify::ThreadData *getNotifyData();
    agx::ref_ptr<Notify::ThreadData> m_notifyData;

    friend class Task;

    // Simple linear congruential generator
    class FastRandom
    {
    public:
      FastRandom(UInt32 seed);
      FastRandom();

      UInt32 operator() ();

    private:
      static const UInt32 mod = ((1ULL << 32) - 5);
      static const UInt32 mul = 69070U;

      void init(UInt32 seed);

    private:
      UInt32 m_state;
    };

    FastRandom m_fastRandom;

    struct Frame
    {
      inline Frame(Task *t = nullptr) : task(t), done(false) {}
      inline Frame& operator=(const Frame& other) {task = other.task; done = other.done.load(); return *this;}
      Task *task;
      std::atomic<bool> done;
    };

    #define AGX_THREAD_FRAME_STACK_MAX_DEPTH 64
    Frame m_frames[AGX_THREAD_FRAME_STACK_MAX_DEPTH];
    size_t m_numFrames;
    size_t m_activationDepth;
    std::atomic<Int32> m_activationCount;

    Thread *m_listNodeNext;

    agx::SpinMutex m_jobMutex;
    JobPtrVector m_localJobs;

    struct JobCompare
    {
      AGX_FORCE_INLINE bool operator() (const Job *lhs, const Job *rhs) const
      {
        return lhs->getCostEstimate() < rhs->getCostEstimate();
      }
    };

    typedef std::priority_queue<Job *, agx::Vector<Job *>, JobCompare> JobQueue;
    JobQueue m_sharedJobs;
    JobQueue m_tmpJobs;


    Timer m_wakeupTimer;
    Timer m_sleepTimer;
    size_t m_numSleep;

    // static THREAD_LOCAL_STORAGE Thread *s_currentThread;
    bool m_isSleeping;
    Block m_startBlock;
    Block m_runBlock;
    Block m_waitBlock;
    agx::SpinMutex m_mutex;
    std::atomic<bool> m_running;

    agx::SpinMutex m_pushMutex;
    JobPtrVector m_pushedJobs;
    std::atomic<Int32> m_pushCounter;


    Index m_id;
    Timer m_timer;
    Timer m_overheadTimer;
    Real m_waitTime;
    Real m_totJobProcessTime;
    Real m_totJobOverheadTime;
    int m_totJobElements;
    Real m_startupTimestamp;
    Real m_shutdownTimestamp;


    // std::ostringstream m_log;
    static Thread *s_threads[AGX_MAX_NUM_THREADS];
    static Thread *s_mainThread;
    static Thread *s_mainWorkThread;
    static Thread *s_ioThread;
    static Callback1<Task *> s_taskCompletionCallback;

    agx::UuidGenerator m_uuidGenerator;
    std::random_device m_randomDevice;
    std::mt19937 m_mersienneTwister;
    // Vector<AbstractMemoryPool *> m_pools;

    #if 0
    template <typename T>
    T *allocateLocalVector(LocalVector<T> *buffer);

    template <typename T>
    void deallocateLocalVector(LocalVector<T> *buffer);
    #endif

  private:

    friend class ThreadLocalAllocator;
    void registerContainerAllocation(Container *container);
    void unregisterContainerAllocation(Container *container);
    void *allocateScratchPadBuffer(size_t numBytes);
    void deallocateScratchPadBuffer(void *buffer, size_t numBytes);

    struct ScratchPadArea
    {
      ScratchPadArea() : buffer(nullptr), end(nullptr), head(nullptr), m_allocator("ScratchPad")
      {}

      ~ScratchPadArea() { m_allocator.deallocateBytes(buffer); }

      char *buffer;
      char *end;
      char *head;
      ByteAllocator m_allocator;
      VectorPOD<Container *> m_activeAllocations;
    };

    ScratchPadArea m_scratchPad;


    static bool s_enableJobTimeline;
    agxData::EntityStorageRef m_timelineEntries;

    static std::exception_ptr s_unhandledException;


    struct LocalTimelineEntry
    {
      LocalTimelineEntry()
         : jobType(UNKNOWN)
         , startTick(0)
         , endTick(0)
         , poolSize(0)
         , costEstimate(0)
         , message(nullptr)
         , job(nullptr)
         , extraDataTitle(nullptr)
         , extraData(0.0)
      {}

      enum JobType { PRE, DISPATCH, POST, UNKNOWN };

      agx::TaskRef task;
      JobType jobType;
      agx::UInt64 startTick;
      agx::UInt64 endTick;
      agx::UInt32 poolSize;
      agx::UInt32 costEstimate;
      const char* message;
      void* job;

      const char* extraDataTitle;
      agx::Real64 extraData;
    };

    Vector<LocalTimelineEntry> m_localTimelineEntries;
    agx::SpinMutex m_timelineMutex;
    AGX_DECLARE_POINTER_TYPES(TimelineExportKernel);


    void exportTimelineEntries();
    void reportTimelineJob(Job *job);
    void pushTimelineEntry(const LocalTimelineEntry& entry);

    AGX_DECLARE_POINTER_TYPES(LogChunk);
    class LogChunk : public Referenced
    {
    public:
      LogChunk(size_t numBytes);
      char* buffer;
      char* head;
      char* end;
      LogChunkRef next;
      size_t numAvailable();

    protected:
      virtual ~LogChunk();
    };

    struct LogEntry
    {
      LogEntry();
      LogEntry(agx::UInt64 timestamp_, const char* message_);
      agx::UInt64 timestamp;
      const char* message;
    };

    int logImplementation(const char* format, va_list ap);
    static int cmpLogEntry(const LogEntry& entry1, const LogEntry& entry2);

    LogChunkRef m_logChunk;
    VectorPOD<LogEntry> m_logEntries;

    typedef HashTable<agxData::EntityModel*, ref_ptr<Referenced>> DefaultStorageTable;
    DefaultStorageTable m_defaultStorageTable;


    /** This is a key into thread local storage where a pointer to the agx::Thread
     * object allocated for the executing thread.*/
    static ThreadStorageKey s_threadObject_key;
  };

  typedef agx::VectorPOD<Thread *> ThreadPtrVector;







  /* Implementation */
  ////////////////////////////////////////////////////////////////////////////////

  inline bool BasicThread::joinable()
  {
    return m_handle.joinable();
  }



  inline unsigned int BasicThread::getThreadState()
  {
    return m_state.load();
  }



  AGX_FORCE_INLINE Index Thread::getId() const
  {
    return m_id;
  }

  AGX_FORCE_INLINE Thread *Thread::getThread(size_t id)
  {
    agxAssert(s_threads[id]);
    return s_threads[id];
  }

  AGX_FORCE_INLINE Thread *Thread::getCurrentThread()
  {
    // The AGX threads store a pointer to their thread object in the thread local storage
    // pointed to by 's_threadObject_key'. If the currently executing thread isn't a AgX
    // thread, then nothing will have been written to 's_threadObject_key' and 0 is returned.
    return static_cast<Thread*>( Thread::readPerThreadStorage(s_threadObject_key) );
  }

  AGX_FORCE_INLINE Thread *Thread::getMainThread()
  {
    agxAssert(s_mainThread);
    return s_mainThread;
  }

  #if 0
  AGX_FORCE_INLINE Thread *Thread::getMainWorkThread()
  {
    return s_mainWorkThread;
  }
  #endif

  AGX_FORCE_INLINE bool Thread::isMainThread() { return Thread::getCurrentThread() == s_mainThread; }
  AGX_FORCE_INLINE bool Thread::isActive() const { return m_activationCount.load() > 0; }

  AGX_FORCE_INLINE void Thread::startWaitTimer()
  {
    m_timer.reset(true);
  }

  AGX_FORCE_INLINE void Thread::stopWaitTimer()
  {
    m_timer.stop();
    m_waitTime += (Real)m_timer.getTime();
  }

  AGX_FORCE_INLINE Thread::RandomGenerator& Thread::getRandomGenerator()
  {
    return m_mersienneTwister;
  }

  AGX_FORCE_INLINE Real Thread::getOverheadTime() const { return Real(m_overheadTimer.getTime()); }


  inline int Thread::log(const char *format, ...)
  {
    va_list arguments;
    va_start(arguments, format);
    int result = Thread::getCurrentThread()->logImplementation(format, arguments);
    va_end(arguments);
    return result;
  }


  #if 0
  AGX_FORCE_INLINE void *Thread::allocateBytes(size_t numBytes)
  {
    return m_byteAllocator.allocate(numBytes);
  }

  AGX_FORCE_INLINE void *Thread::deallocateBytes(void *ptr)
  {
    m_byteAllocator.deallocateBytes(ptr);
  }

  template <typename T>
  AGX_FORCE_INLINE T *Thread::allocate()
  {
    return this->getPool<T>()->allocate();
  }

  template <typename T>
  AGX_FORCE_INLINE void Thread::deallocate(T *ptr)
  {
    this->getPool<T>()->deallocate(ptr);
  }


  template <typename T>
  AGX_FORCE_INLINE T *Thread::create()
  {
    return this->getPool<T>()->create();
  }

  template <typename T>
  AGX_FORCE_INLINE void Thread::destroy(T *ptr)
  {
    this->getPool<T>()->destroy(ptr);
  }

  template <typename T>
  AGX_FORCE_INLINE MemoryPool<T> *Thread::getPool()
  {
    uint32_t id = agxData::getType<T>()->getId();
    if (id >= m_pools.size())
      m_pools.resize(id+1, 0);

    if (!m_pools[id])
      m_pools[id] = new MemoryPool<T>;

    return static_cast<MemoryPool<T> *>(m_pools[id]);
  }

  void setCurrentConstructionObject(void *ptr);
  void *getCurrentConstructionObject();
  #endif





  AGX_FORCE_INLINE void *Thread::allocateScratchPadBuffer(size_t numBytes)
  {
    /* Check if scratch pad is too small, which requires reallocation and updates to all existing container allocations */
    if (m_scratchPad.head + numBytes > m_scratchPad.end)
    {
      size_t currentSize = m_scratchPad.end - m_scratchPad.buffer;
      const Real growFactor = 1.5;
      size_t newSize = (size_t)( Real(currentSize + numBytes) * growFactor );
      // printf("Reallocating scratch pad for thread %d from %u to %u KB\n", this->getId(), (unsigned)(currentSize/1024), (unsigned)(newSize/1024));

      char *newBuffer = (char *)m_scratchPad.m_allocator.allocateBytes(newSize, 64);
      agxAssertN(newBuffer, "Thread %d could not allocate %u bytes for job scratch pad!", this->getId(), (unsigned)newSize);
      if (!newBuffer)
        return nullptr;

      /* Copy active allocations */
      size_t numUsed = m_scratchPad.head - m_scratchPad.buffer;

      if ( numUsed > 0 )
        memcpy(newBuffer, m_scratchPad.buffer, numUsed);

      m_scratchPad.m_allocator.deallocateBytes(m_scratchPad.buffer);

      /* Update buffer pointers for active allocations */
      for (size_t i = 0; i < m_scratchPad.m_activeAllocations.size(); ++i)
      {
        Container *container = m_scratchPad.m_activeAllocations[i];

        if (container->m_buffer)
        {
          ptrdiff_t offset = (char *)container->m_buffer - (char *)m_scratchPad.buffer;
          container->m_buffer = newBuffer + offset;
        }
      }

      m_scratchPad.buffer = newBuffer;
      m_scratchPad.end = m_scratchPad.buffer + newSize;
      m_scratchPad.head = m_scratchPad.buffer + numUsed;
    }

    void *mem = m_scratchPad.head;
    m_scratchPad.head += numBytes;

    return mem;
  }

  AGX_FORCE_INLINE void Thread::deallocateScratchPadBuffer(void *buffer, size_t numBytes)
  {
    agxAssert(!buffer || (buffer >= m_scratchPad.buffer && buffer < m_scratchPad.end));

    if ((char *)buffer + numBytes == m_scratchPad.head)
      m_scratchPad.head = (char *)buffer;
  }

  AGX_FORCE_INLINE void Thread::registerContainerAllocation(Container *container)
  {
    m_scratchPad.m_activeAllocations.push_back(container);
  }

  #ifdef AGX_DEBUG
  AGX_FORCE_INLINE void Thread::unregisterContainerAllocation(Container *container)
  #else
  AGX_FORCE_INLINE void Thread::unregisterContainerAllocation(Container * /* container */)
  #endif
  {
    agxAssert1(!m_scratchPad.m_activeAllocations.empty() && container == m_scratchPad.m_activeAllocations.back(), "LIFO order required!");
    m_scratchPad.m_activeAllocations.pop_back();

    /* Reset scratch pad, removing any holes from reallocations */
    if (m_scratchPad.m_activeAllocations.empty())
      m_scratchPad.head = m_scratchPad.buffer;
  }

}


#if ENABLE_VERBOSE_THREAD_TIMELINE
  #define AGX_BEGIN_TIMELINE_REPORT(variable)                             \
      auto variable ## _begin_time = agx::Timer::getCurrentTick()

  #define AGX_END_TIMELINE_REPORT(variable, title)                      \
      auto variable ## _end_time = agx::Timer::getCurrentTick();        \
      agx::Thread::getCurrentThread()->reportSystemJob( variable ## _begin_time , variable ## _end_time, title);

  #define AGX_END_TIMELINE_REPORT_DATA(variable, title, title2, data)   \
      auto variable ## _end_time = agx::Timer::getCurrentTick();        \
      agx::Thread::getCurrentThread()->reportSystemJob( variable ## _begin_time , variable ## _end_time, title, title2, data)
#else
  #define AGX_BEGIN_TIMELINE_REPORT(variable)
  #define AGX_END_TIMELINE_REPORT(variable, title)
  #define AGX_END_TIMELINE_REPORT_DATA(variable, title, title2, data)
#endif


#ifdef _MSC_VER
# pragma warning(pop)
#endif
