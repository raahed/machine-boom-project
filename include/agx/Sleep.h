/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_SLEEP_H
#define AGX_SLEEP_H

#include <agx/SystemInformation.h>
#include <time.h>

//#include <xmmintrin.h>

#ifdef _WIN64
#include <intrin.h>
#endif
// #include <emmintrin.h>

namespace agx
{
  void nanoSleep( long );

  inline void noopLoop(UInt64 numIterations)
  {
    for (UInt64 i = 0; i < numIterations; ++i)
    {
    #if defined(_MSC_VER)
      #if defined(_WIN64)
      __nop();
      #else
      __asm  nop;
      #endif
    #else
      asm volatile("nop\n\t");
      // _mm_pause();
    #endif
    }
  }

  /**
  Sleep the current thread. Timing is not exact, it is just a rough estimate.
  */
  inline void microSleep(Real microSeconds)
  {
#if defined(__APPLE__) && defined(__aarch64__)
    // getCpuFrequency is not supported on Apple Silicon yet.
    nanoSleep(static_cast<long>(microSeconds * 1000.0));
#else
    noopLoop((UInt64)(microSeconds * Real(SystemInformation::getCpuFrequency()))); // NOTE: getCpuFrequency returns MHz
#endif
  }


  inline void nanoSleep( long nanoSeconds )
  {
#if defined(_MSC_VER)
    agx::microSleep( nanoSeconds / agx::Real(1000.0) );
#else
    timespec sleepTime;
    sleepTime.tv_sec  = nanoSeconds / 1000000000;
    sleepTime.tv_nsec = nanoSeconds % 1000000000;

    nanosleep( &sleepTime, 0 );
#endif
  }
}


#endif /* AGX_SLEEP_H */
