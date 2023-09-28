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

#ifndef AGX_HIGHACCURACYTIMER_H
  #define AGX_HIGHACCURACYTIMER_H

#include <agx/config/AGX_USE_CPP11_TIMERS.h>
#include <agx/agxCore_export.h>
#include <agx/Integer.h>

#include <agx/agx.h>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach_time.h>
#endif

// Still supporting compilers without full support for C++11, so cannot use chrono yet.
#if AGX_USE_CPP11_TIMERS()
#include <chrono>
#endif

#ifdef AGX_APPLE_IOS
#include <mach/mach_time.h>
#endif

#ifdef _WIN32

#else
#include <unistd.h>
#endif




namespace agx {

  /**
  The HighAccurayTimer class can replace the regular Timer class in high level
  applications where the need for accuracy is greater than performance.
  This timer should ONLY be used with low calling frequency, such as when
  measuring total computation time, where the regular timer accuracy can fall short
  on certain platforms (Windows).

  As of now, this HighAccuracyTimer has identical accuracy and performance
  for MAC and Linux as the regular Timer class.
  (For Windows, QueryPerformanceFrequency/QueryPerformanceTimer is used).
  */
  class AGXCORE_EXPORT HighAccuracyTimer
  {
  public:

    /**
    Creates a HighAccuracyTimer.
    \param startImmediately Should the HighAccuracyTimer start immediately or start in stopped mode
    (now, but also after consequent calls to reset())?
    */
    HighAccuracyTimer(bool startImmediately = false);

    /**
    Starts the HighAccuracyTimer.
    Calling start on a already started HighAccuracyTimer will have no effect.
    */
    void start();

    /**
    Stops the HighAccuracyTimer.
    Calling stop on a already stopped HighAccuracyTimer will have no effect.
    Upon stopping, the amount of cycles and time ran will be updated.
    */
    void stop();

    /**
    Clear all data.
    */
    void reset(bool startAfterReset = false);

    /**
    Report total elapsed time since start in milliseconds, excluding
    intervals when the timer was suspended.
    */
    Real64 getTime() const;

    /// Report execution time in CPU cycles.
    UInt64 getNumTicks() const;

    /// Return the start tick.
    UInt64 getStartTick() const;

    /// Return true if running.
    bool isRunning() const;

    /// Return raw value of the TSC register (in Windows: QueryPerformanceTimer).
    static UInt64 getCurrentTick();

    /// Return the given number of ticks in ms.
    static Real64 convertToMs(UInt64 ticks);

    /// For internal use only.
    static void initClockSpeed();

  private:
    static agx::Real clockSpeed();

    /// Report execution time in CPU cycles.
    static UInt64 getCurrentTickWindows();


  private:

#if AGX_USE_CPP11_TIMERS()
    std::high_resolution_clock::time_point m_start;
    std::duration<agx::Real64> m_total;
#else
    UInt64 m_start;
    UInt64 m_total;
#endif

    bool m_running;

    static agx::Real s_highAccuracyClockSpeed;
  };


  /* Implementation */
  inline HighAccuracyTimer::HighAccuracyTimer(bool startImmediately)
  {
    reset();
    if (startImmediately)
      this->start();
  }


  AGX_FORCE_INLINE void HighAccuracyTimer::start()
  {
    if (m_running)
      return;

    m_running = true;

#if AGX_USE_CPP11_TIMERS()
    m_start = std::high_resolution_clock::now();
#else
    m_start = getCurrentTick();
#endif
  }


  AGX_FORCE_INLINE void HighAccuracyTimer::stop()
  {
    if (m_running) {
      m_running = false;

#if AGX_USE_CPP11_TIMERS()
      std::high_resolution_clock::time_point now = std::high_resolution_clock::now();
      std::duration<Real64> duration = duration_cast<std::duration<Real64>>(now - m_start);
      m_total += duration;
#else
      UInt64 now = getCurrentTick();
      m_total += (now - m_start);
#endif
    }
  }


  AGX_FORCE_INLINE void HighAccuracyTimer::reset(bool startAfterReset)
  {
    m_start = 0L;
    m_total = 0L;

    m_running = false;
    if (startAfterReset)
      this->start();
  }


  AGX_FORCE_INLINE UInt64 HighAccuracyTimer::getNumTicks() const
  {
    return m_total;
  }


  AGX_FORCE_INLINE UInt64 HighAccuracyTimer::getStartTick() const
  {
    return m_start;
  }


  AGX_FORCE_INLINE bool HighAccuracyTimer::isRunning() const
  {
    return m_running;
  }


  AGX_FORCE_INLINE Real64 HighAccuracyTimer::getTime() const
  {
#if AGX_USE_CPP11_TIMERS()
    if (m_running)
    {
      std::high_resolution_clock::time_point now = std::high_resolution_clock::now();
      std::duration<Real64> duration = duration_cast<std::duration<Real64>>(now - m_start);
      std::duration<Real64> total = m_total + duration;
      return total.count() * Real64(1000);
    }
    else
    {
      return m_total.count() * Real64(1000);;
    }

#else
    return m_running ? (clockSpeed() * (Real64)(getCurrentTick() - m_start + m_total)) : clockSpeed() * (Real64)m_total;
#endif
  }


  AGX_FORCE_INLINE agx::Real HighAccuracyTimer::clockSpeed()
  {
#ifdef AGX_DEBUG
    agxAssert1(s_highAccuracyClockSpeed >= 0, "HighAccuracyTimer class not properly initialized, forgot to call agx::init() ?");
#endif

    return s_highAccuracyClockSpeed;
  }


  AGX_FORCE_INLINE UInt64 HighAccuracyTimer::getCurrentTick()
  {
#if defined(_MSC_VER)
    const UInt64 retVal = getCurrentTickWindows();
    return retVal;
#elif defined(AGX_APPLE_IOS) || defined(__APPLE__)
    return mach_absolute_time();
#elif defined (__GNUC__)  && ! defined(__INTEL_COMPILER) && defined(__ia64__)
    unsigned long long int val;
    __asm__ __volatile__("mov %0=ar.itc" : "=r"(val) :: "memory");
    return(val);
#elif defined(__INTEL_COMPILER) && defined(__ia64__) && !defined(HAVE_TICK_COUNTER)
    return __getReg(_IA64_REG_AR_ITC);
#elif defined(__ECC) && defined(__ia64__) && !defined(HAVE_TICK_COUNTER)
    return __getReg(_IA64_REG_AR_ITC);
#elif (defined(__GNUC__) || defined(__ICC)) && (defined(__i386__) || defined(__x86_64__))  && !defined(HAVE_TICK_COUNTER)

    // "The CPUID instruction serializes the processor pipeline so that all of the preceding
    // instructions must retire before it begins execution.  ...  This is thought to provide
    //a more accurate cycle count on the code being measured."  [intel person at forum on intel.com]

    // cpuid writes to eax,ebx,ecx, edx, tell g++ about registers being clobbered
    // in 64bits ebx is 32bits of the larger rbx register so should not require
    // any ifdef's in 64bit
    // would be better to use pushad, cpuid, popad but for some unknownn reason gcc
    // doesn't recognize pushad/popad
    // Update: gcc has different name for pushad/popad, l postfix instead of d,
    // but those instructions doesn't work in 64bit

#ifdef __LP64__
    unsigned long long int lowbits, highbits;
    __asm__ __volatile__("rdtsc": "=a" (lowbits), "=d"(highbits) : : "ebx", "ecx");
#else
    unsigned long int lowbits, highbits;
    __asm__ __volatile__("pushal\n\t" "cpuid\n\t" "popal\n\t" "rdtsc": "=a" (lowbits), "=d"(highbits));
    //__asm__ __volatile__("push %%ebx\n" "push %%ecx\n" "cpuid\n" "rdtsc\n" "pop %%ebx\n" "pop %%ecx": "=a" (lowbits), "=d"(highbits));
#endif

    return (((unsigned long long int)highbits << 32) | (unsigned long long int)lowbits);

#else
    agxAbort1("Timer::getCurrentTick failed!");
    return -1;
#endif

  }


  AGX_FORCE_INLINE Real64 HighAccuracyTimer::convertToMs(UInt64 ticks)
  {
    return clockSpeed() * (Real64)ticks;
  }



} // namespace agx
#endif
