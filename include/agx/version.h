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

#ifndef AGX_VERSION_H
#define AGX_VERSION_H

#include <agx/agxCore_export.h>
#include <agx/stdint.h>
#include <agx/agx_version.h>
#include <agx/Integer.h>

extern "C" {

  /// \return a size_t value from a given version information
  extern AGXCORE_EXPORT agx::UInt32 AGX_CALC_VERSION( agx::UInt8 generation, agx::UInt8 major, agx::UInt8 minor, agx::UInt8 patch);

  /// \return the current version in unsigned 32 bits
  extern AGXCORE_EXPORT agx::UInt32 AGX_GET_VERSION();

  /// Convert a 32 bit unsigned int with version encoding into generation, major, minor, patch
  extern AGXCORE_EXPORT void AGX_CONVERT_VERSION( agx::UInt32 version, agx::UInt8& generation, agx::UInt8& major, agx::UInt8& minor, agx::UInt8& patch);

  /** \return the python version used when build AGX */
  extern AGXCORE_EXPORT const char* agxGetPythonVersion();

  /** \return the version in the form GENERATION.MAJOR.MINOR.PATCH-rREVISION */
  extern AGXCORE_EXPORT const char* agxGetVersion(bool includeRevision=true);

  /** \return the generation version */
  extern AGXCORE_EXPORT agx::UInt8 agxGetGenerationVersion();

  /** \return the major version */
  extern AGXCORE_EXPORT agx::UInt8 agxGetMajorVersion();

  /** \return the minor version */
  extern AGXCORE_EXPORT agx::UInt8 agxGetMinorVersion();

  /** \return the patch version */
  extern AGXCORE_EXPORT agx::UInt8 agxGetPatchVersion();

  /** \return the last committed file version to the revision system */
  extern AGXCORE_EXPORT const char* agxGetRevision();

  /** Return the name of the library */
  extern AGXCORE_EXPORT const char* agxGetLibraryName();

  /** \return the company name */
  extern AGXCORE_EXPORT const char* agxGetCompanyName();

  /**
  \param inIsoFormat - If true the format of the date string will be: YYYY-MM-DD otherwise in the standard __DATE__
  format: mmm d yyyy
  \return the date of build of the file version.cpp
  */
  extern AGXCORE_EXPORT const char* agxGetBuildDate( bool inIsoFormat = false );

  /** \return the time of build of the file version.cpp */
  extern AGXCORE_EXPORT const char* agxGetBuildTime();

  /** \return the build type (e.g. DEBUG, RELWITHDEBINFO, RELEASE, ...).*/
  extern AGXCORE_EXPORT const char* agxGetBuildType();

  /** \return the build flags used for C++(e.g. Od, O2, ...).*/
  extern AGXCORE_EXPORT const char* agxGetCppBuildFlags();

  /** \return the build flags used for C(e.g. Od, O2, ...).*/
  extern AGXCORE_EXPORT const char* agxGetCBuildFlags();

  /** \return the current serialization version.*/
  extern AGXCORE_EXPORT agx::UInt16 agxGetSerializationVersion();

  /** \return the current storage stream serialization version. */
  extern AGXCORE_EXPORT agx::UInt16 agxGetStorageStreamVersion();
}


/// Macros below can be used as: AGX_VERSION_GREATER_OR_EQUAL(1,8,0,0)
/// Use to check if the build version of AGX is equal or higher

/// Use to check if AGX version is strictly lower than the specified version
#define AGX_VERSION_LESS_THAN(GENERATION, MAJOR, MINOR, PATCH) \
  ((AGX_GENERATION_VERSION<GENERATION) || (AGX_GENERATION_VERSION==GENERATION && (AGX_MAJOR_VERSION<MAJOR || (AGX_MAJOR_VERSION==MAJOR && (AGX_MINOR_VERSION<MINOR || (AGX_MINOR_VERSION==MINOR && AGX_PATCH_VERSION<PATCH) )))))

/// Use to check if AGX version is lower or equal than the specified version
#define AGX_VERSION_LESS_OR_EQUAL(GENERATION, MAJOR, MINOR, PATCH) \
  ((AGX_GENERATION_VERSION<GENERATION) || (AGX_GENERATION_VERSION==GENERATION && (AGX_MAJOR_VERSION<MAJOR || (AGX_MAJOR_VERSION==MAJOR && (AGX_MINOR_VERSION<MINOR || (AGX_MINOR_VERSION==MINOR && AGX_PATCH_VERSION<=PATCH))))))

/// Use to check if AGX version is higher or equal than the specified version
#define AGX_VERSION_GREATER_OR_EQUAL(GENERATION, MAJOR, MINOR, PATCH) \
  ((AGX_GENERATION_VERSION>GENERATION) || (AGX_GENERATION_VERSION==GENERATION && (AGX_MAJOR_VERSION>MAJOR) || (AGX_MAJOR_VERSION==MAJOR && (AGX_MINOR_VERSION>MINOR || (AGX_MINOR_VERSION==MINOR && AGX_PATCH_VERSION>=PATCH)))))

/// Use to check if AGX version is strictly higher than the specified version
#define AGX_VERSION_GREATER_THAN(GENERATION, MAJOR, MINOR, PATCH) \
  ((AGX_GENERATION_VERSION>GENERATION) || (AGX_GENERATION_VERSION==GENERATION && (AGX_MAJOR_VERSION>MAJOR) || (AGX_MAJOR_VERSION==MAJOR && (AGX_MINOR_VERSION>MINOR || (AGX_MINOR_VERSION==MINOR && AGX_PATCH_VERSION>PATCH)))))

#endif

