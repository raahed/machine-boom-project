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

#ifndef AGXOSG_EXPORT_H
#define AGXOSG_EXPORT_H


#include <agx/config.h>
#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()

#ifdef _WIN32

#if AGX_DYNAMIC() &&  defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
#  if defined( AGXOSG_LIBRARY_STATIC )
#    define AGXOSG_EXPORT
#  elif defined( AGXOSG_LIBRARY )
#    define AGXOSG_EXPORT   __declspec(dllexport)
#  else
#    define AGXOSG_EXPORT   __declspec(dllimport)
#  endif
#else
#  define AGXOSG_EXPORT
#endif

#else
  // Non Win32
  #if __GNUC__ >= 4
    #define AGXOSG_EXPORT __attribute__ ((visibility("default")))
  #else
    #define AGXOSG_EXPORT
  #endif
#endif

#define OSG_VEC2_TO_AGX( X ) ( agx::Vec2( (X)[0], (X)[1] ))
#define OSG_VEC3_TO_AGX( X ) ( agx::Vec3( (X)[0], (X)[1], (X)[2] ))
#define OSG_VEC4_TO_AGX( X ) ( agx::Vec4( (X)[0], (X)[1], (X)[2], (X)[3] ))
#define OSG_VEC4F_TO_AGX( X ) ( agx::Vec4f( (X)[0], (X)[1], (X)[2], (X)[3] ))
#define OSG_QUAT_TO_AGX( X ) ( agx::Quat( (X).asVec4()[0], (X).asVec4()[1], (X).asVec4()[2], (X).asVec4()[3] ))


#define AGX_VEC2_TO_OSG( X ) ( osg::Vec2( (float)(X)[0], (float)(X)[1] ))
#define AGX_VEC3_TO_OSG( X ) ( osg::Vec3( (float)(X)[0], (float)(X)[1], (float)(X)[2] ))
#define AGX_VEC4_TO_OSG( X ) ( osg::Vec4( (float)(X)[0], (float)(X)[1], (float)(X)[2], (float)(X)[3] ))
#define AGX_QUAT_TO_OSG( X ) ( osg::Quat( (X).asVec4()[0], (X).asVec4()[1], (X).asVec4()[2], (X).asVec4()[3] ))

#  define AGX_MAT4X4_TO_OSG( X ) ( osg::Matrixd( (X).ptr() ))
#  define OSG_MAT4X4_TO_AGX( X ) ( agx::AffineMatrix4x4( (X).ptr() ))


DOXYGEN_END_INTERNAL_BLOCK()

/// Contains classes/utility functions for connecting AGX to OpenSceneGraph
namespace agxOSG
{

}

#endif

