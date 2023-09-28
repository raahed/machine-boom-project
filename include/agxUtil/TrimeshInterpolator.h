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

#ifndef AGXUTIL_TRIMESHINTERPOLATOR_H
#define AGXUTIL_TRIMESHINTERPOLATOR_H

#include <agx/agx.h>
#include <agxUtil/agxUtil.h>

namespace agxCollide {
  class Trimesh;
}

namespace agxUtil {

  /**
  Class for generating a Trimesh shape from an interpolation from two trimeshes
  */
  AGX_DECLARE_POINTER_TYPES(TrimeshInterpolator);
  class AGXPHYSICS_EXPORT TrimeshInterpolator
  {
    public:

      /// Default constructor
      TrimeshInterpolator() {}
      ~TrimeshInterpolator() {}

      /**
      Interpolate between two Trimeshes and create a Trimesh shape at an
      interpolation time
      \param startTrimesh - Mesh at t0
      \param endTrimesh - Mesh at t1
      \param subTime Decides what interpolation trimesh is created, ~0 = close to start trimesh,
      ~1 = close to end trimesh, 0.5 right between trimeshes
      \return interpolated trimesh
      */
      static agxCollide::Trimesh* createInterpolationTrimesh(
        const agxCollide::Trimesh* startTrimesh,
        const agxCollide::Trimesh* endTrimesh,
        agx::Real subTime);

      /**
      Extracts velocity values from interpolation of mesh and returns a
      velocity for each vertex on the mesh
      \param duration The time between the two meshes
      \param startTrimesh - Mesh at t0
      \param endTrimesh - Mesh at t1
      \return Velocity values in a Vec3Vector
      */
      static agx::Vec3Vector createVelocityVector(
        const agxCollide::Trimesh* startTrimesh,
        const agxCollide::Trimesh* endTrimesh,
        agx::Real duration);

      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      Used as a callback pre solve, replaces a body in the contact (from a mesh)
      to a dummy with velocity values from the contact triangle from the mesh
      \param newContactBody
      \param velocityVector Velocities for each vertex on mesh
      \param geometryContact GeometryContact to replace the body in
      \param geoToReplace Which body should be replaced (0 or 1)
      \return RigidBody that's inserted into the GeometryContact
       */
      static void replaceContactGeometry(
        agx::RigidBody* newContactBody,
        const agx::Vec3Vector* velocityVector,
        agxCollide::GeometryContact* geometryContact,
        size_t geoToReplace // Index 0 or 1
      );
      DOXYGEN_END_INTERNAL_BLOCK()

  };
}

#endif

