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

#ifndef AGX_PREFETCH_H
#define AGX_PREFETCH_H

#include <agx/config/AGX_USE_PREFETCHING.h>
#include <agx/config/AGX_USE_SSE.h>

#include <agx/config.h>
#include <agx/macros.h>
#include <cstddef> // mostly for size_t

#if AGX_USE_PREFETCHING() && ( defined(_MSC_VER) || AGX_USE_SSE() )

#include <xmmintrin.h>

#if defined(__clang__) || defined(__INTEL_COMPILER)
// Clang use defines instead of enum
#define _mm_hint int
#endif

namespace agx
{

  enum PrefetchType
  {
    L1 = _MM_HINT_T0,
    L2 = _MM_HINT_T1,
    L3 = _MM_HINT_T2,
    NTA = _MM_HINT_NTA
  };

  // _MM_HINT_T0  - All cache levels
  // _MM_HINT_T1  - All except 0
  // _MM_HINT_T2  - All except 0&1
  // _MM_HINT_NTA - Non-temporal data prefetch


  template< int hint, typename T >
  AGX_FORCE_INLINE void prefetch( const T* ptr )
  {
    #ifdef __GNUC__
    _mm_prefetch( (const char*) ptr, (_mm_hint) hint );
    #else
    _mm_prefetch( (const char*) ptr, hint );
    #endif
  }


  template< int hint, typename T >
  AGX_FORCE_INLINE void prefetch_64( const T* ptr )
  {
    prefetch<hint>( ptr );
  }

  template< int hint, typename T >
  AGX_FORCE_INLINE void prefetch_128( const T* ptr )
  {
    prefetch<hint>( ptr );
    prefetch<hint>( ((const char*)ptr) + 64 );
  }

  template< int hint, typename T >
  AGX_FORCE_INLINE void prefetch_256( const T* ptr )
  {
    prefetch<hint>( ptr );
    prefetch<hint>( ((const char*)ptr) +  64 );
    prefetch<hint>( ((const char*)ptr) + 128 );
    prefetch<hint>( ((const char*)ptr) + 192 );
  }




  const size_t PREFETCH_SIZE = 64;

  template <PrefetchType PrefetchT>
  AGX_FORCE_INLINE void prefetchBytes(const void *buffer, size_t numBytes)
  {
    const size_t numBlocks = (numBytes + PREFETCH_SIZE-1) / PREFETCH_SIZE;
    // agxVerify(agx::isAligned(buffer, PREFETCH_SIZE));

    for (size_t i = 0; i < numBlocks; ++i)
    {
      const void *ptr = (const char *)buffer + i * PREFETCH_SIZE;
      prefetch<PrefetchT>(ptr);
    }
  }

  template <typename T>
  AGX_FORCE_INLINE void prefetchNTA(const T *buffer, size_t numElements)
  {
    const size_t numBytes = numElements * sizeof(T);
    prefetchBytes<agx::NTA>(buffer, numBytes);
  }

  template <typename T>
  AGX_FORCE_INLINE void prefetchL3(const T *buffer, size_t numElements)
  {
    const size_t numBytes = numElements * sizeof(T);
    prefetchBytes<agx::L3>(buffer, numBytes);
  }

} // namespace agx

#else
  // prefetching disabled

namespace agx
{

  enum PrefetchType
  {
    L1 = 3,
    L2 = 2,
    L3 = 1,
    NTA = 0
  };

  template< int hint, typename T >
  AGX_FORCE_INLINE void prefetch( const T* ) {}

  template< int hint, typename T >
  AGX_FORCE_INLINE void prefetch_64( const T* ) {}

  template< int hint, typename T >
  AGX_FORCE_INLINE void prefetch_128( const T* ) {}

  template< int hint, typename T >
  AGX_FORCE_INLINE void prefetch_256( const T* ) {}


  template <int PrefetchT>
  AGX_FORCE_INLINE void prefetchBytes(const void *, size_t )
  {}

  template <typename T>
  AGX_FORCE_INLINE void prefetchNTA(const T *, size_t )
  {}

  template <typename T>
  AGX_FORCE_INLINE void prefetchL3(const T *, size_t )
  {}

}
#endif // AGX_USE_SSE etc.

#endif // Include guard.

