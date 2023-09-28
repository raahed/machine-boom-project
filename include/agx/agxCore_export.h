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

#ifndef AGXCORE_EXPORT_H
#define AGXCORE_EXPORT_H


#include <agx/config.h>
#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()

#if defined(_WIN32) && !defined(CALLABLE_GENERATOR)

#if AGX_DYNAMIC() &&  defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
#  if defined( AGXCORE_LIBRARY_STATIC )
#    define AGXCORE_EXPORT
#  elif defined( AGXCORE_LIBRARY )
#    define AGXCORE_EXPORT   __declspec(dllexport)
#  else
#    define AGXCORE_EXPORT   __declspec(dllimport)
#  endif
#else
#  define AGXCORE_EXPORT
#endif

#elif defined(CALLABLE_GENERATOR)
  #define AGXCORE_EXPORT
#else
  // Non Win32
  #if __GNUC__ >= 4
    #define AGXCORE_EXPORT __attribute__ ((visibility("default")))
  #else
    #define AGXCORE_EXPORT
  #endif
#endif


/*
#ifndef __LOC_CORE__
#define __STR2_CORE__(x) #x
#define __STR1_CORE__(x) __STR2_CORE__(x)
#define __LOC_CORE__ __FILE__ " have export mode "__STR1_CORE__(AGXCORE_EXPORT)"\n"
#endif
#pragma message(__LOC_CORE__"")
*/


DOXYGEN_END_INTERNAL_BLOCK()

#endif

