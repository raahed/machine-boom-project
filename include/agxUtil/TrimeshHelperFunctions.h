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

#include <agx/agx.h>
#include <agxUtil/agxUtil.h>

namespace agxCollide
{
  class Box;
  class Cylinder;
  class Capsule;
}

namespace agxUtil {
  /**
  Merged duplicate vertices (distance is <= precision).
  Will also remove triangles which have been collapsed by merging vertices,
  and thus 2 or 3 of its vertices have become identical.
  \param originalVertices The original vertices.
  \param originalIndices The original indices (always 3 per triangle, see agxCollide/Trimesh.h).
  \param remainingVertices The remaining vertices.
    All pre-existing content will be overwritten.
  \param remainingIndices The remaining indices (always 3 per triangle, see agxCollide/Trimesh.h).
    All pre-existing content will be overwritten.
  \param precision The maximum distance for merging.
  */
  void AGXPHYSICS_EXPORT mergeDuplicateVertices(
    const agx::Vec3Vector& originalVertices,
    const agx::UInt32Vector& originalIndices,
    agx::Vec3Vector& remainingVertices,
    agx::UInt32Vector& remainingIndices,
    agx::Real precision);

  /**
  Changes Winding from clockwise to counterclockwise.
  \param indices The indices (always 3 per triangle, see agxCollide/Trimesh.h).
  Expected to be in clockwise winding. Will reorder them to counterclockwise ordering.
  */
  void AGXPHYSICS_EXPORT changeWindingToCounterclockwise (agx::UInt32Vector& indices);


  /**
  Perform mesh reduction using Fast-Quadric-Mesh
  \param vertices - Vertices that will be target for reduction
  \param indices -  Indices that will be target for reduction. Must be a multiple of 3 (triangles) otherwise this method will return false
  \param[out] outVertices - The vertices for the reduced mesh will be written into this vector
  \param[out] outIndices - The indices for the reduced mesh will be written into this vector
  \param reductionRatio - The desired reduction of triangle count. 1 == no reduction, 0.5 == 50% reduction. Clamped between [0.01, 1.0]
  \param aggressiveness - (default = 7.0) Lower is faster and higher is better decimation. Valid range is [0.01,..]
  \return true if mesh reduction results in > 1 triangle. False if indata is invalid or reduction results in < 1 triangle.
  */
  AGXPHYSICS_EXPORT bool reduceMesh(
      const agx::Vec3Vector& vertices, const agx::UInt32Vector& indices,
      agx::Vec3Vector& outVertices, agx::UInt32Vector& outIndices,
      double reductionRatio=0.5,
      double aggressiveness = 7.0);


  /**
  Computes an oriented bounding box around the specified vertices. The result will a Vec3 with the half extents of the box and a
  transformation matrix that will orient/translate the principal axes of the box so that it encapsulates the vertices.

  \param vertices - The vertices from which an oriented bounding box will be created.
  \param[out] halfExtents - Half extensts of the bounding box
  \param[out] transform - Transformation that will translate and rotate a box with the size halfExtents so that it encapsulates the vertices.
  \return true if a valid bounding box was computed
  */
  AGXPHYSICS_EXPORT bool computeOrientedBox(
    const agx::Vec3Vector& vertices,
    agx::Vec3& halfExtents,
    agx::AffineMatrix4x4& transform);


    /// Specify the orientation of cylinder, capsule for computeOrientedCylinder and computeOrientedCapsule
  struct ShapeOrientation
  {
    enum Orientation {
      X_AXIS,         /**< Choose the principal axis along the X-axis of the bounding box */
      Y_AXIS,         /**< Choose the principal axis along the Y-axis of the bounding box */
      Z_AXIS,         /**< Choose the principal axis along the Z-axis of the bounding box */
      MINIMIZE_VOLUME /**< Automatically choose the direction which leads to the smallest volume */
    };
  };

  /**
  Compute the radius, height and rotation of a cylinder that encapsulates a specified box specified by \p halfExtents
  The \p orientation flag can be used to orient the cylinder along one of the principal axes of the box. Default is along Y.

  \param[in] vertices - The vertices from which an oriented cylinder will be created.
  \param[out] radiusHeight - 2D vector with the radius, height result from the calculation
  \param[out] localRotation - Transformation that will specify the local rotation of the cylinder relative to the box
  \param[in] orientation - Specifies which principal axis of the box that the cylinder will be aligned to
  \return true if a valid cylinder could be computed
  */
  AGXPHYSICS_EXPORT bool computeOrientedCylinder(
    const agx::Vec3Vector& vertices,
    agx::Vec2& radiusHeight,
    agx::AffineMatrix4x4& localRotation,
    agxUtil::ShapeOrientation::Orientation orientation = agxUtil::ShapeOrientation::MINIMIZE_VOLUME
    );

  /**
  Compute the radius, height and rotation of a capsule that encapsulates a specified box specified by \p halfExtents
  The \p orientation flag can be used to orient the cylinder along one of the principal axes of the box. Default is along Y.

  \param[in] vertices - The vertices from which an oriented capsule box will be created.
  \param[out] radiusHeight - 2D vector with the radius, height result from the calculation
  \param[out] localRotation - Transformation that will specify the local rotation of the capsule relative to the box
  \param[in] orientation - Specifies which principal axis of the box that the capsule will be aligned to
  \return true if a valid capsule could be computed
  */
  AGXPHYSICS_EXPORT bool computeOrientedCapsule(
    const agx::Vec3Vector& vertices,
    agx::Vec2& radiusHeight,
    agx::AffineMatrix4x4& localRotation,
    agxUtil::ShapeOrientation::Orientation orientation = agxUtil::ShapeOrientation::MINIMIZE_VOLUME
    );
}
