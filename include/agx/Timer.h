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

#ifndef AGX_TIMER_H
#define AGX_TIMER_H

#include <agx/config/AGX_USE_CPP11_TIMERS.h>
#include <agx/agxCore_export.h>
#include <agx/Integer.h>

#include <agx/agx.h>

#if defined(_MSC_VER)
//#define AGX_USE_HPET // Internal. Comment to use HTC instead. Change only when compiling AgX.
#ifndef AGX_USE_HPET
#include <intrin.h>
#pragma intrinsic(__rdtsc)
#endif
#endif


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
  The Timer class permits timing execution speed with the same
  refinement as the built in hardware clock of the CPU.

  Profiling tools typically have sampling rates which are too coarse for
  detailed timing analysis. The Timer class uses the hardware clock to
  record beginning and end times of given code sections.

  A Timer with better accuracy, but lower performance ( that uses QueryPerformanceFrequency/QueryPerformanceTimer in Windows),
  can be found in HighAccuracyTimer. Use this for high level application where performance is not as important.

  (For Windows, QueryPerformanceFrequency/QueryPerformanceTimer is used if AGX_USE_HPET is set).
  */
  class AGXCORE_EXPORT Timer
  {
  public:

    /**
    Creates a timer.
    \param startImmediately Should the timer start immediately or start in stopped mode
    (now, but also after consequent calls to reset())?
    */
    Timer(bool startImmediately = false);

    /**
      Starts the timer.
      Calling start on a already started timer will have no effect.
    */
    void start();

    /**
      Stops the timer.
      Calling stop on a already stopped timer will have no effect.
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

    /// \return the start tick.
    UInt64 getStartTick() const;

    /// \return true if running.
    bool isRunning() const;

    /// \return raw value of the TSC register (in Windows: QueryPerformanceTimer if AGX_USE_HPET is set).
    static UInt64 getCurrentTick();

#ifndef SWIG
    /// \return time in ms relative to a start tick
    static Real64 getTime( UInt64 startTick );
#endif

    /// \return the given number of ticks in ms.
    static Real64 convertToMs( UInt64 ticks );

    /// For internal use only.
    static void initClockSpeed();

  private:
    static agx::Real clockSpeed();
#if defined(_MSC_VER)
    /// Report execution time in CPU cycles.
    static UInt64 getCurrentTickWindows();
#endif

  private:

    #if AGX_USE_CPP11_TIMERS()
    std::high_resolution_clock::time_point m_start;
    std::duration<agx::Real64> m_total;
    #else
    UInt64 m_start;
    UInt64 m_total;
    #endif

    bool m_running;

    static agx::Real s_clockSpeed;
  };


  /* Implementation */
  inline Timer::Timer(bool startImmediately)
  {
    reset();
    if (startImmediately)
      this->start();
  }


  AGX_FORCE_INLINE void Timer::start()
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


  AGX_FORCE_INLINE void Timer::stop()
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


  AGX_FORCE_INLINE void Timer::reset(bool startAfterReset)
  {
    m_start = 0L;
    m_total = 0L;

    m_running = false;
    if (startAfterReset)
      this->start();
  }


  AGX_FORCE_INLINE UInt64 Timer::getNumTicks() const
  {
    return m_total;
  }


  AGX_FORCE_INLINE UInt64 Timer::getStartTick() const
  {
    return m_start;
  }


  AGX_FORCE_INLINE bool Timer::isRunning() const
  {
    return m_running;
  }


  AGX_FORCE_INLINE Real64 Timer::getTime() const
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
    return m_running ? (clockSpeed() * (Real64)(getCurrentTick() - m_start  + m_total)) : clockSpeed() * (Real64)m_total;
    #endif
  }


  AGX_FORCE_INLINE agx::Real Timer::clockSpeed()
  {
    #ifdef AGX_DEBUG
    agxAssert1(s_clockSpeed >= 0, "Timer class not properly initialized, forgot to call agx::init() ?");
    #endif

    return s_clockSpeed;
  }


  AGX_FORCE_INLINE UInt64 Timer::getCurrentTick()
  {
#if defined(_MSC_VER)
#ifdef AGX_USE_HPET
    const UInt64 retVal = getCurrentTickWindows();
#else
    const UInt64 retVal = __rdtsc();
#endif
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
    // would be better to use pushad, cpuid, popad but for some unknown reason gcc
    // doesn't recognize pushad/popad
    // Update: gcc has different name for pushad/popad, l postfix instead of d,
    // but those instructions doesn't work in 64bit

#ifdef __LP64__
    unsigned long long int lowbits, highbits;
    __asm__ __volatile__("rdtsc": "=a" (lowbits), "=d"(highbits) : : );
#else
    unsigned long int lowbits, highbits;
    __asm__ __volatile__("pushal\n\t" "cpuid\n\t" "popal\n\t" "rdtsc": "=a" (lowbits), "=d"(highbits) );
    //__asm__ __volatile__("push %%ebx\n" "push %%ecx\n" "cpuid\n" "rdtsc\n" "pop %%ebx\n" "pop %%ecx": "=a" (lowbits), "=d"(highbits));
#endif

    return (((unsigned long long int)highbits<<32) | (unsigned long long int)lowbits);

#else
    agxAbort1("Timer::getCurrentTick failed!");
    return -1;
#endif

  }

  AGX_FORCE_INLINE Real64 Timer::getTime( UInt64 startTick )
  {
    return Timer::convertToMs(Timer::getCurrentTick() - startTick);
  }

  AGX_FORCE_INLINE Real64 Timer::convertToMs( UInt64 ticks )
  {
    return clockSpeed() * (Real64)ticks;
  }



} // namespace agx
#endif
