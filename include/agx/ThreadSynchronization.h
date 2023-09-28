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

#ifndef AGX_THREADSYNCHRONIZATION_H
#define AGX_THREADSYNCHRONIZATION_H

#include <agx/config/AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING.h>
#include <agx/config/AGX_USE_SSE.h>

#include <agx/agx.h>
#include <agx/debug.h>
#include <agx/SpinMutex.h>
#include <agx/AtomicValue.h>

#include <agxData/Type.h>

#include <chrono>
#include <mutex>
#include <condition_variable>

#if AGX_USE_SSE()
#include <xmmintrin.h>
#include <emmintrin.h>
#endif

#ifndef _MSC_VER
#include <errno.h>
#endif

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4275 ) //  warning C4275: non dll-interface class
# pragma warning( disable : 4702)  // warning C4702: unreachable code
#endif

/**
Work threads will measure the time spent in locks
*/
#define AGX_MEASURE_LOCK_TIME 0

namespace agx
{

  using Mutex = std::mutex;
  using ReentrantMutex = std::recursive_mutex;
  using Condition = std::condition_variable;



  //----------------------------------------------------------------------------------

  /**
  Barrier synchronization primitive.
  */
  class AGXCORE_EXPORT Barrier
  {
  public:

    Barrier( int numThreads );

    void block(); // Use numThreads from constructor

    void block(int numThreads); // Override numThreads to wait for

    Barrier( const Barrier& ) = delete;
    Barrier& operator=(const Barrier& other) = delete;

  private:
    std::mutex m_mutex;
    std::condition_variable m_condition;

    int m_wave;
    int m_numWaiting;
    int m_barrierCount;
  };

  //----------------------------------------------------------------------------------


  /**
  Block synchronization primitive.
  */
  class AGXCORE_EXPORT Block
  {
  public:

    /// Default constructor
    Block();

    /// No copy-constructor
    Block(const Block&) = delete;

    /// No assignment operator
    Block& operator=( const Block& other ) = delete;

    /**
    Block until released from another thread.
    */
    void block();

    /**
    Block until released from another thread, or
    until timer expires.
    return false if timer expired
    */
    bool block(unsigned long timeout_ms);

    /**
    Release the block.
    */
    void release();

    /**
    Reset the block.
    */
    void reset();


  protected:
  private:
    std::mutex m_mutex;
    std::condition_variable m_condition;
    bool m_released;
  };




  /**
  ReaderWriterLock provides shared access for readers and exclusive
  access for writers.

  A thread is not allowed to upgrade or downgrade their lock type.
  For example, if a shared access is taken via `lock_shared`, it
  must then be released via `unlock_shared` before exclusive access
  can be requested.

  If c++17 is available, std::shared_mutex is recommended instead
  of this class.
  */
  class AGXCORE_EXPORT ReaderWriterLock {

    public:
      /**
      Constructor
      */
      ReaderWriterLock();

      // No copy-constructor
      ReaderWriterLock( const ReaderWriterLock& ) = delete;

      ReaderWriterLock& operator=(const ReaderWriterLock&) = delete;

      /**
      Acquire the shared lock.
      Multiple threads can have shared access at the same time.
      For example used by multiple readers.
      */
      void lock_shared();

      /**
      Releases the shared access.
      */
      void unlock_shared();

      /**
      Acquire exclusive access.
      */
      void lock();

      /**
      Relese the exclusive access
      */
      void unlock();

    private:
      unsigned int   m_numReaders;
      unsigned int   m_numWriters;

      agx::Mutex     m_mutex;
      agx::Condition m_rwCond;
  };





  //----------------------------------------------------------------------------------

  inline Barrier::Barrier(int numThreads)
    : m_mutex(), m_condition(), m_wave(0), m_numWaiting(0), m_barrierCount( numThreads )
  {
  }

  AGX_FORCE_INLINE void Barrier::block()
  {
    this->block(0);
  }

  AGX_FORCE_INLINE void Barrier::block(int numThreads)
  {
    #if AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING()
    return;
    #endif

    #if AGX_MEASURE_LOCK_TIME
    Thread *thread = Thread::getCurrentThread();
    thread->startWaitTimer();
    #endif


    std::unique_lock<std::mutex> lock(m_mutex);

    int wave = m_wave;

    if ( numThreads != 0 )
      m_barrierCount = numThreads;

    if ( ++m_numWaiting == m_barrierCount ) {
      // Last one here.

      // Prepare for next barrier use
      m_numWaiting = 0;
      m_wave = 1 - wave;
      m_condition.notify_all();
    }
    else {
      m_condition.wait( lock, [this,wave](){ return m_wave != wave; } );
    }


    #if AGX_MEASURE_LOCK_TIME
    thread->stopWaitTimer();
    #endif
  }

  //----------------------------------------------------------------------------------

  inline Block::Block() :
    m_mutex(), m_condition(), m_released( false )
  {
  }

  AGX_FORCE_INLINE void Block::block()
  {
    #if AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING()
    return;
    #endif

    #if AGX_MEASURE_LOCK_TIME
    Thread *thread = Thread::getCurrentThread();
    thread->startWaitTimer();
    #endif

    std::unique_lock<std::mutex> lock( m_mutex );
    m_condition.wait( lock, [this]{return m_released;} );


    #if AGX_MEASURE_LOCK_TIME
    thread->stopWaitTimer();
    #endif
  }


  AGX_FORCE_INLINE bool Block::block(unsigned long timeout_ms)
  {
    #if AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING()
    return true;
    #endif

    #if AGX_MEASURE_LOCK_TIME
    Thread *thread = Thread::getCurrentThread();
    thread->startWaitTimer();
    #endif

    std::unique_lock<std::mutex> lock(m_mutex);
    bool status = m_condition.wait_for( lock, std::chrono::milliseconds( timeout_ms ), [this]{return m_released;} );

    // done waiting == true
    // timeout == false

    #if AGX_MEASURE_LOCK_TIME
    thread->stopWaitTimer();
    #endif

    return status;
  }


  AGX_FORCE_INLINE void Block::release()
  {
    #if AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING()
    return;
    #endif

    std::lock_guard<std::mutex> lock(m_mutex);

    if ( !m_released ) {
      m_released = true;
      m_condition.notify_all();
    }
  }

  AGX_FORCE_INLINE void Block::reset()
  {
    #if AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING()
    return;
    #endif

    std::lock_guard<std::mutex> lock(m_mutex);

    m_released = false;
  }



}


#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif /* AGX_THREADSYNCHRONIZATION_H */
