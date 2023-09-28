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

#ifndef AGX_MACROS_H
#define AGX_MACROS_H

#include <agx/config.h>
#include <external/hedley/hedley.h>

#define AGX_STATIC_ASSERT(X) static_assert(X, #X)

#if !defined(AGX_DEBUG) && !defined(NDEBUG)
#define AGX_DEBUG
#endif

#define AGX_STRINGIFY(A) #A
#define AGX_TOSTRING(x) AGX_STRINGIFY(x)

/**
Macro string concatenation, using extra level of indirection
Example:
Foo ## __LINE__             ==>  Foo__LINE__
AGX_CONCAT(Foo, __LINE__)   ==>  Foo124
*/
#define _AGX_CONCAT(x,y) x ## y
#define AGX_CONCAT(x,y) _AGX_CONCAT(x,y)


#ifdef _MSC_VER
#  define AGX_NOEXCEPT
#else

#define AGX_NOEXCEPT noexcept

#endif

// Index set iterator for kernels.
#define AGX_ITERATE_INDEX_SET(set, i) typename SetT::iterator _agx_set_iterator = set.begin(), _agx_set_end = set.end(); for (typename SetT::Type i; (_agx_set_iterator != _agx_set_end) && ((i = *_agx_set_iterator), true); ++_agx_set_iterator)

#if defined(_MSC_VER)
# define AGX_FORCE_INLINE __forceinline
#elif defined(__GNUC__) && !defined(AGX_DEBUG)
# define AGX_FORCE_INLINE inline __attribute__((always_inline))
#else
# define AGX_FORCE_INLINE inline
#endif

#ifdef _MSC_VER
#define AGX_ALIGNED( t, a ) __declspec( align( a ) ) t
#else
#define AGX_ALIGNED( t, a ) t __attribute__ ((aligned (a)))
#endif

#ifdef CALLABLE_GENERATOR
#define CALLABLE __attribute__((annotate("CALLABLE")))
#define CALLABLE_UNIT(param) __attribute__((annotate("CALLABLE_UNIT " param)))
#define CALLABLE_IGNORE __attribute__((annotate("CALLABLE_IGNORE")))
#else
#define CALLABLE
#define CALLABLE_UNIT(param)
#define CALLABLE_IGNORE
#endif


/// \todo Remove this once all supported platforms support C++17.
#if defined(__GNUC__) && ( __GNUC__ >= 7) || defined(__clang__) && ( __clang_major__ >= 10 )
  #define AGX_MAYBE_UNUSED [[maybe_unused]]
#else
  #define AGX_MAYBE_UNUSED
#endif

#ifdef SWIG
  #define AGX_DEPRECATED(since)
  #define AGX_DEPRECATED_FOR(since, replacement)
#else
  #define AGX_DEPRECATED(since) HEDLEY_DEPRECATED(since)
  #define AGX_DEPRECATED_FOR(since, replacement) HEDLEY_DEPRECATED_FOR(since, replacement)
#endif


//
#ifdef AGX_HIDE_INTERNAL_DOCUMENTATION
  #define DOXYGEN_START_INTERNAL_BLOCK() /** \cond INTERNAL_DOCUMENTATION */
  #define DOXYGEN_END_INTERNAL_BLOCK()   /** \endcond */
#else
  #define DOXYGEN_START_INTERNAL_BLOCK()
  #define DOXYGEN_END_INTERNAL_BLOCK()
#endif


#endif
