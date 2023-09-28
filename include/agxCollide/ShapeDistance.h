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

#include <cstdint>

#include <agx/agxPhysics_export.h>
#include <agx/Vec3.h>
#include <agx/AffineMatrix4x4.h>


namespace agxCollide {

  class Shape;
  class Geometry;

  /**
  Status values that can be returned when performing
  shape-shape distance computation.
  */
  enum class ShapeDistanceResult : std::uint8_t {
    SUCCESS,                 /**< Distance measurement successful */
    OVERLAPPING_SHAPES,      /**< Shapes overlap */
    UNSUPPORTED_SHAPE_TYPES, /**< One or more input shapes is of a non-supported type (Shape::hasSupportFunction() returns false) */
    FAILURE                  /**< Generic error, e.g supported shapes but invalid values such as some negative shape size */
  };


  /**
  Utility function to compute the minimum distance between two shapes.
  This method supports convex shapes for which Shape::hasSupportFunction() returns true.
  The function works directly on the shapes and does not need for the shapes to be part of
  agxCollide::Space or that e.g. BoundingBoxes are updated.

  When the shapes do not overlap, SUCCESS IS returned and the position on each shape in global
  coordinates are written to point1 and point2. The distance is the magnitude of the vector
  between point1 and point2.

  If the shapes do overlap, OVERLAPPING_SHAPES is returned and point1 and point2 are unchanged.
  More detailed contact information for this case, use the regular collision detection via Space.

  For unsupported shape types or an unspecified error, two other return codes are also possible.

  \param shape1 - First shape to be tested
  \param transform1 - Transform that transform the first shape to the world coordinate system
  \param shape2 - Second shape to be tested
  \param transform2 - Transform that transform the second shape to the world coordinate system
  \param point1 - The closest point on shape1
  \param point2 - The closest point on shape2
  \return The result of the computation. See ShapeDistanceResult.
  */
  AGXPHYSICS_EXPORT agxCollide::ShapeDistanceResult computeShapeDistance(
      const agxCollide::Shape* shape1, const agx::AffineMatrix4x4& transform1,
      const agxCollide::Shape* shape2, const agx::AffineMatrix4x4& transform2,
      agx::Vec3& point1, agx::Vec3& point2 );



  /**
  Utility method to perform shape-shape minimum distance test between all
  compatible shape pairs, where one shape is taken from each geometry.
  The tests will use computeShapeDistance and the smallest distance will be
  returned in point1 and point2 if the return status is SUCCESS.

  \note: The transformation of the Geometry (geometry->getTransform) will NOT be used in this method.
  Reason being that the geometry might not be enabled (or even part of a Simulation) and hence might
  not be updated correctly. So if you explicitly set the transformation of any of the geometries you
  can use geometry1->getTransform() as your transform argument.

  \param geometry1 - First geometry (with shapes) to be tested
  \param transform1 - Transform that transform the first geometry to the world coordinate system
  \param geometry2 - Second geometry (with shapes) to be tested
  \param transform2 - Transform that transform the second geometry to the world coordinate system
  \param point1 - The closest point on shape1
  \param point2 - The closest point on shape2
  \return The result of the computation. See ShapeDistanceResult.
  */
  AGXPHYSICS_EXPORT agxCollide::ShapeDistanceResult computeGeometryDistance(
      const agxCollide::Geometry* geometry1, const agx::AffineMatrix4x4& transform1,
      const agxCollide::Geometry* geometry2, const agx::AffineMatrix4x4& transform2,
      agx::Vec3& point1, agx::Vec3& point2 );


}

