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

#ifndef AGXREMOTESOLVERCLIENT_EXPORT_H
#define AGXREMOTESOLVERCLIENT_EXPORT_H


#include <agx/config.h>
#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN    // Exclude rarely-used stuff from Windows headers
#endif

// Windows Header Files:

#if defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
#  if defined( AGXSENSOR_LIBRARY_STATIC )
#    define AGXREMOTESOLVERCLIENT_EXPORT
#  elif defined( AGXSENSOR_LIBRARY )
#    define AGXREMOTESOLVERCLIENT_EXPORT   __declspec(dllexport)
#  else
#    define AGXREMOTESOLVERCLIENT_EXPORT   __declspec(dllimport)
#  endif
#else
#  define AGXREMOTESOLVERCLIENT_EXPORT
#endif

#else
#define AGXREMOTESOLVERCLIENT_EXPORT
#endif

DOXYGEN_END_INTERNAL_BLOCK()

#endif

