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

/*
This source code has been taken and modified by Algoryx Simulation AB
from the source and under the license given below.
*/

// Copyright 2007-2010 Baptiste Lepilleur
// Distributed under MIT license, or public domain if desired and
// recognized in your jurisdiction.
// See file LICENSE for detail or copy at http://jsoncpp.sourceforge.net/LICENSE

#ifndef AGX_JSON_CONFIG_H_INCLUDED
# define AGX_JSON_CONFIG_H_INCLUDED

/// If defined, indicates that json library is embedded in CppTL library.
//# define JSON_IN_CPPTL 1

/// If defined, indicates that json may leverage CppTL library
//#  define JSON_USE_CPPTL 1
/// If defined, indicates that cpptl vector based map should be used instead of std::map
/// as Value container.
//#  define JSON_USE_CPPTL_SMALLMAP 1
/// If defined, indicates that Json specific container should be used
/// (hash table & simple deque container with customizable allocator).
/// THIS FEATURE IS STILL EXPERIMENTAL! There is know bugs: See #3177332
//#  define JSON_VALUE_USE_INTERNAL_MAP 1
/// Force usage of standard new/malloc based allocator instead of memory pool based allocator.
/// The memory pools allocator used optimization (initializing Value and ValueInternalLink
/// as if it was a POD) that may cause some validation tool to report errors.
/// Only has effects if JSON_VALUE_USE_INTERNAL_MAP is defined.
//#  define JSON_USE_SIMPLE_INTERNAL_ALLOCATOR 1

// If non-zero, the library uses exceptions to report bad input instead of C
// assertion macros. The default is to use exceptions.
# ifndef JSON_USE_EXCEPTION
# define JSON_USE_EXCEPTION 1
# endif

/// If defined, indicates that the source file is amalgamated
/// to prevent private header inclusion.
/// Remarks: it is automatically defined in the generated amalgamated header.
// #define JSON_IS_AMALGAMATION


# ifdef JSON_IN_CPPTL
#  include <cpptl/config.h>
#  ifndef JSON_USE_CPPTL
#   define JSON_USE_CPPTL 1
#  endif
# endif

#ifdef _MSC_VER
  # ifdef JSON_IN_CPPTL
  #  define JSON_API CPPTL_API
  # elif defined(JSON_DLL_BUILD_STATIC)
  #  define JSON_API
  # elif defined(JSON_DLL_BUILD)
  #  define JSON_API __declspec(dllexport)
  # else
  #  define JSON_API __declspec(dllimport)
  # endif
#else
  #  define JSON_API
  #  include <stdint.h>
#endif

// If JSON_NO_INT64 is defined, then Json only support C++ "int" type for integer
// Storages, and 64 bits integer support is disabled.
// #define JSON_NO_INT64 1

#if defined(_MSC_VER)  &&  _MSC_VER <= 1200 // MSVC 6
// Microsoft Visual Studio 6 only support conversion from __int64 to double
// (no conversion from unsigned __int64).
#define JSON_USE_INT64_DOUBLE_CONVERSION 1
#endif // if defined(_MSC_VER)  &&  _MSC_VER < 1200 // MSVC 6

#if defined(_MSC_VER)  &&  _MSC_VER >= 1500 // MSVC 2008
/// Indicates that the following function is deprecated.
# define JSONCPP_DEPRECATED(message) __declspec(deprecated(message))
#endif

#if !defined(JSONCPP_DEPRECATED)
# define JSONCPP_DEPRECATED(message)
#endif // if !defined(JSONCPP_DEPRECATED)

namespace agxJson {
   typedef int Int;
   typedef unsigned int UInt;
# if defined(JSON_NO_INT64)
   typedef int LargestInt;
   typedef unsigned int LargestUInt;
#  undef JSON_HAS_INT64
# else // if defined(JSON_NO_INT64)
   // For Microsoft Visual use specific types as long long is not supported
#  if defined(_MSC_VER) // Microsoft Visual Studio
   typedef __int64 Int64;
   typedef unsigned __int64 UInt64;
#  else // if defined(_MSC_VER) // Other platforms, use long long
   typedef int64_t Int64;
   typedef uint64_t UInt64;
#  endif // if defined(_MSC_VER)
   typedef Int64 LargestInt;
   typedef UInt64 LargestUInt;
#  define JSON_HAS_INT64
# endif // if defined(JSON_NO_INT64)
} // end namespace agxJson


#endif // AGX_JSON_CONFIG_H_INCLUDED
