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

#ifndef AGXPHYSICS_EXPORT_H
#define AGXPHYSICS_EXPORT_H


#include <agx/config.h>
#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()

#if defined(_WIN32) && !defined(CALLABLE_GENERATOR)

#if AGX_DYNAMIC() &&  defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
#  if defined( AGXPHYSICS_LIBRARY_STATIC )
#    define AGXPHYSICS_EXPORT
#  elif defined( AGXPHYSICS_LIBRARY )
#    define AGXPHYSICS_EXPORT   __declspec(dllexport)
#  else
#    define AGXPHYSICS_EXPORT   __declspec(dllimport)
#  endif
#else
#  define AGXPHYSICS_EXPORT
#endif

#elif defined(CALLABLE_GENERATOR)
  #define AGXPHYSICS_EXPORT
#else
  // Non Win32
  #if __GNUC__ >= 4
    #define AGXPHYSICS_EXPORT __attribute__ ((visibility("default")))
  #else
    #define AGXPHYSICS_EXPORT
  #endif
#endif


/*
#ifndef __LOC_PHYSICS__
#define __STR2_PHYSICS__(x) #x
#define __STR1_PHYSICS__(x) __STR2_PHYSICS__(x)
#define __LOC_PHYSICS__ __FILE__ " have export mode "__STR1_PHYSICS__(AGXPHYSICS_EXPORT)"\n"
#endif
#pragma message(__LOC_PHYSICS__"")
*/


DOXYGEN_END_INTERNAL_BLOCK()

#endif

