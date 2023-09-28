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

#ifndef AGX_SPINMUTEX_H
#define AGX_SPINMUTEX_H

#include <agx/config/AGX_USE_SSE.h>
#include <agx/config/AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING.h>
#include <agx/macros.h>
#include <agx/debug.h>

#include <agx/Integer.h>
#include <atomic>

#if AGX_USE_SSE()
#include <xmmintrin.h>
#include <emmintrin.h>
#endif


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4275 ) //  warning C4275: non dll-interface class
#  pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agx
{
  /**
  Spin-lock mutex
  */
  class AGXCORE_EXPORT SpinMutex
  {
  public:
    SpinMutex();
    ~SpinMutex();

    // Odd to have copy constructor on a mutex, but existing code requries it.
    // Try to prevent FormatT<T>::initializeElements on mutexes.
    SpinMutex(const SpinMutex& other);

    // Odd to have assignment operator on a mutex, but existing code requires it.
    // Try to remove ReferencedEntity::setObserverMutex.
    SpinMutex& operator=(const SpinMutex& other);

    /**
    Lock the mutex.
    */
    void lock();

    /**
    Unlock the mutex.
    */
    void unlock();

    /**
    Deprecated. Prefer the standard try_lock.
    Try to lock the mutex.
    \return false if lock failed
    */
    bool trylock();

    bool try_lock();

    /**
    Debug
    */
    bool isLocked() const;

  private:
    std::atomic<int> m_lock;
  };



  /* Implementation */
  AGXCORE_EXPORT extern bool forcedShutdown();

  AGX_FORCE_INLINE SpinMutex::SpinMutex()
      : m_lock{0}
  {
  }

  AGX_FORCE_INLINE SpinMutex::~SpinMutex()
  {
    agxAssert(forcedShutdown() || !this->isLocked());
  }

  AGX_FORCE_INLINE void SpinMutex::lock()
  {
    #if AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING()
    return;
    #endif

    agx::Int32 prev = m_lock.exchange(1, std::memory_order_acquire);

    while(prev != 0)
    {
      #if AGX_USE_SSE()
      _mm_pause();
      #endif
      prev = m_lock.exchange(1, std::memory_order_acquire);
    }
  }

  AGX_FORCE_INLINE void SpinMutex::unlock()
  {
    #if AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING()
    return;
    #endif

    m_lock.store(0, std::memory_order_release);
  }

  AGX_FORCE_INLINE bool SpinMutex::trylock()
  {
    #if AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING()
    return true;
    #endif

    int32_t prev = m_lock.exchange(1, std::memory_order_acquire);
    return (prev == 0);
  }

  AGX_FORCE_INLINE bool SpinMutex::try_lock()
  {
#if AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING()
    return true;
#endif

    int32_t prev = m_lock.exchange(1, std::memory_order_acquire);
    return (prev == 0);
  }

  AGX_FORCE_INLINE bool SpinMutex::isLocked() const
  {
    return m_lock.load( std::memory_order_relaxed ) != 0;
  }
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif /* AGX_SPINMUTEX_H */
