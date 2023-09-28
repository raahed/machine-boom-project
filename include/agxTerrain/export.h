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

#ifndef AGXTERRAIN_EXPORT_H
#define AGXTERRAIN_EXPORT_H

#include <agx/config.h>

#ifdef _WIN32

#if AGX_DYNAMIC() &&  defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
#  if defined( AGX_LIBRARY_STATIC )
#    define AGXTERRAIN_EXPORT
#  elif defined( AGXTERRAIN_LIBRARY )
#    define AGXTERRAIN_EXPORT   __declspec(dllexport)
#  else
#    define AGXTERRAIN_EXPORT   __declspec(dllimport)
#  endif
#else
#  define AGXTERRAIN_EXPORT
#endif

#else
  // Non Win32
  #if __GNUC__ >= 4
    #define AGXTERRAIN_EXPORT __attribute__ ((visibility("default")))
  #else
    #define AGXTERRAIN_EXPORT
  #endif
#endif




#define AGX_VEC3_TO_OPENVDB_VEC3D(v) openvdb::Vec3d(v.x(), v.y(), v.z())
#define AGX_VEC3_TO_OPENVDB_VEC3R(v) openvdb::Vec3R(v.x(), v.y(), v.z())
#define AGX_VEC3_TO_OPENVDB_VEC3F(v) openvdb::Vec3f((float)v.x(), (float)v.y(), (float)v.z())
#define AGX_VEC3_TO_OPENVDB_VEC3I(v) openvdb::Vec3i(v.x(), v.y(), v.z())
#define AGX_VEC3_TO_OPENVDB_COORD(v) openvdb::Coord((int)v.x(), (int)v.y(), (int)v.z())

#define OPENVDB_VEC3_TO_AGX_VEC3I(v) agx::Vec3i(v.x(), v.y(), v.z())
#define OPENVDB_VEC3_TO_AGX(v) agx::Vec3(v.x(), v.y(), v.z())
#define OPENVDB_VEC3_TO_AGX_VEC3F(v) agx::Vec3f((float)v.x(), (float)v.y(), (float)v.z())

namespace agxStream
{
  class OutputArchive;
  class InputArchive;
}

#define AGXTERRAIN_STORE_RESTORE_INTERFACE \
    virtual void store( const class Terrain* terrain, agxStream::OutputArchive& out ) const; \
    virtual void restore( class Terrain* terrain, agxStream::InputArchive& in )

namespace agxTerrain
{

}

#endif
