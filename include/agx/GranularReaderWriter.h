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

#pragma once

#include <agx/String.h>
#include <agx/Physics/GranularBodySystem.h>

namespace agx
{
  /**
  Utility class for reading and writing granular body data to .csv, .agx and .aagx files.
  */
  class AGXPHYSICS_EXPORT GranularReaderWriter
  {
  public:
    /**
    Write specified granular bodies to file with format depending on file extension. ( Supported: .csv, .agx, aagx )
    \param filename - The specified name of the file to save granular data to. File ending determines the
                      specified format to save.
    \param granularBodies - A vector of particles that should be stored to file.
    \param source - The source granular body system that contains the specified particles.
    \param transform - The offset transform that will be applied to particle position and velocity before writing.
    \return true if the granular bodies could be stored to a file, false otherwise.
    */
    static bool writeGranularFile( const agx::String& filename,
                                   const agx::Physics::GranularBodyPtrVector& granularBodies,
                                   agx::Physics::GranularBodySystem* source,
                                   const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4() );

    /**
    Write specified granular body data to a .agx or .aagx file.
    \param filename - The specified name of the file to save granular data to. File ending determines the
                      specified format to save.
    \param granularBodies - A vector of particles that should be stored to file.
    \param source - The source granular body system that contains the specified particles.
    \param transform - The offset transform that will be applied to particle position and velocity before writing.
    \return true if the granular bodies could be stored to an .agx or .aagx file, false otherwise.
    */
    static bool writeGranularFileAGX( const agx::String& filename,
                                      const agx::Physics::GranularBodyPtrVector& granularBodies,
                                      agx::Physics::GranularBodySystem* source,
                                      const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4() );

    /**
    Write specified granular body data to a .agx or .aagx file.
    \param filename - The specified name of the file to save granular data to. File ending determines the
                      specified format to save.
    \param granularBodies - A vector of particles that should be stored to file.
    \param source - The source granular body system that contains the specified particles.
    \param transform - The offset transform that will be applied to particle position and velocity before writing.
    \return true if the granular bodies could be stored to a .csv file, false otherwise.
    */
    static bool writeGranularFileCSV( const agx::String& filename,
                                      const agx::Physics::GranularBodyPtrVector& granularBodies,
                                      agx::Physics::GranularBodySystem* source,
                                      const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4() );

    /**
    Loads all granular bodies in a specified .agx or .aagx file for reading.
    \param filename - The specified name of the file to load granular body data from.
    \param granularBodies - The target granular body vector were loaded granulars will be inserted.
    \param target - The target granular body system the loaded particles should be inserted to.
    \param transform - The offset transform that will be applied to particle position and velocity after loading.
    \return true if the file was successfully loaded, false otherwise.
    */
    static bool loadGranularFileAGX( const agx::String& filename,
                                     agx::Physics::GranularBodyPtrVector& granularBodies,
                                     agx::Physics::GranularBodySystem* target,
                                     const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4() );

    /**
    Write specified granular and rigid bodies to an .AGX file.
    \param filename - The specified name of the file to save granular and rigid body data to.
    \param granularBodies - A vector of particles that should be stored to file.
    \param rigidBodies - A vector of rigid bodies that should be stored to file.
    \param source - The source granular body system that contains the specified particles.
    \param transform - The offset transform that will be applied to body position and velocity before writing.
    \return true if the bodies could be stored to a file, false otherwise.
    */
    static bool writeFile( const agx::String& filename,
      const agx::Physics::GranularBodyPtrVector& granularBodies,
      agx::RigidBodyPtrVector& rigidBodies,
      agx::Physics::GranularBodySystem* source,
      const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4() );

    /**
    Loads all granular and rigid bodies in a specified .agx or .aagx file for reading.
    \param filename - The specified name of the file to save granular data to. File ending determines the
                      specified format to save.
    \param granularBodies - The target granular body vector were loaded granulars will be inserted.
    \param rigidBodies - The target rigid body vector were loaded bodies will be inserted.
    \param target - The target granular body system where the loaded granular bodies should be placed.
    \param targetSimulation - The target simulation where the loaded rigid bodies should be added.
    \param initMaterial - The material that will be used to initialize the bodies.
    \param transform - The offset transform that will be applied to body position and velocity upon loading.
    \return true if the file was successfully loaded, false otherwise.
    */
    static bool loadFileAGX(
      const String& filename,
      agx::Physics::GranularBodyPtrVector& granularBodies,
      agx::RigidBodyPtrVector& rigidBodies,
      Physics::GranularBodySystem* target,
      agxSDK::Simulation* targetSimulation,
      agx::Material* initMaterial,
      const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4() );
  };
}
