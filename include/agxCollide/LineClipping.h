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

#ifndef AGXCOLLIDE_LINECLIPPING_H
#define AGXCOLLIDE_LINECLIPPING_H

#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()

#include <agx/agxPhysics_export.h>
#include <agx/Interval.h>

#ifndef _WIN32
#include <float.h>
#endif

#include <agx/agx.h>
#include <agx/Vec3.h>

namespace agxCollide
{
  /**
  * Clips an (infinite) line to a plane.
  * Gives only parameters on line as solution.
  * (Expect p_l = linePoint + lineDir * t for all points p_l on the line).
  * The calculation is done in the plane's coordinate system, expecting
  * the plane to encompass the origin,
  * and have the normal in one of the three standard axes.
  * \param planeDistance The positive distance of the plane to the origin of the plane's coordinate system in the plane normal (including sign).
  * \param planeDimension The standard axis which is the plane's normal.
  * \param planeNormalSign The sign of the plane normal (-1 or 1)
  * \param linePoint A point on the line (in plane's coordinate system).
  * \param lineDir The (not necessarily normalized) direction of the line (in plane's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineToPlane(
    agx::Real planeDistance,
    unsigned int planeDimension,
    int planeNormalSign,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an (infinite) line to a slab (two parallel planes).
  * Gives only parameters on line as solution.
  * (Expect p_l = linePoint + lineDir * t for all points p_l on the line).
  * The calculation is done in the slab's coordinate system, expecting
  * both parallel planes to be at equal distance to the origin and
  * having the normal in one of the three standard axes.
  * \param slabDistance The slab's end distance from origin
  * \param slabDimension The standard axis which is parallel to the slab's normal.
  * \param linePoint A point on the line (in slab's coordinate system).
  * \param lineDir The (not necessarily normalized) direction of the line (in slab's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineToSlab(
    agx::Real slabDistance,
    unsigned int slabDimension,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an (infinite) line to a double slab (two times two parallel planes).
  * Gives only parameters on line as solution.
  * (Expect p_l = linePoint + lineDir * t for all points p_l on the line).
  * The calculation is done in the double slab's coordinate system, expecting
  * both parallel planes each to be at equal distance to the origin and
  * having the normal in one of the three standard axes.
  * \param slab1Distance slab 1's end distance from origin
  * \param slab1Dimension The standard axis which is parallel to slab 1's normal.
  * \param slab2Distance slab 2's end distance from origin
  * \param slab2Dimension The standard axis which is parallel to the slab 2's normal.
  * \param linePoint A point on the line (in cylinder's coordinate system).
  * \param lineDir The (not necessarily normalized) direction of the line (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineToDoubleSlab(
    agx::Real slab1Distance,
    unsigned int slab1Dimension,
    agx::Real slab2Distance,
    unsigned int slab2Dimension,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an (infinite) line to an open box (a box which is open on one side)
  * Gives only parameters on line as solution.
  * (Expect p_l = linePoint + lineDir * t for all points p_l on the line).
  * The calculation is done in the open box's coordinate system
  * \param boxHalfExtents The box's half extents.
  * \param openDim The dimension in which the box is half open
  * \param openDimCutSign The sign against which shut be cut in the open dimension
  * \param lineDir The (not necessarily normalized) direction of the line (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineToOpenBox(
    const agx::Vec3& boxHalfExtents,
    int openDim,
    int openDimCutSign,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );



  /**
  * Clips an (infinite) line to a box.
  * Gives only parameters on line as solution.
  * (Expect p_l = linePoint + lineDir * t for all points p_l on the line).
  * The calculation is done in the double slab's coordinate system, expecting
  * both parallel planes each to be at equal distance to the origin and
  * having the normal in one of the three standard axes.
  * \param boxHalfExtents The box's half extents.
  * \param linePoint A point on the line (in box coordinate system).
  * \param lineDir The (not necessarily normalized) direction of the line (in box coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineToBox(
    const agx::Vec3& boxHalfExtents,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an (infinite) line to an infinite cylinder.
  * Gives only parameters on line as solution.
  * (Expect p_l = linePoint + lineDir * t for all points p_l on the line).
  * The calculation is done in the cylinder's coordinate system, expecting the cylinder
  * to be stretched in the y-direction.
  * \param cylRadius The cylinder's radius
  * \param linePoint A point on the line (in cylinder's coordinate system).
  * \param lineDir The (not necessarily normalized) direction of the line (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineToInfiniteCylinder(
    agx::Real cylRadius,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an (infinite) line to an infinite elliptic cylinder.
  * Gives only parameters on line as solution.
  * (Expect p_l = linePoint + lineDir * t for all points p_l on the line).
  * The calculation is done in the cylinder's coordinate system, expecting the cylinder
  * to be stretched in the y-direction.
  * \param cylRadiusX The cylinder's radius in x
  * \param cylRadiusZ The cylinder's radius in z
  * \param linePoint A point on the line (in cylinder's coordinate system).
  * \param lineDir The (not necessarily normalized) direction of the line (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineToInfiniteEllipticCylinder(
    agx::Real cylRadiusX,
    agx::Real cylRadiusZ,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an (infinite) line to a cylinder.
  * Gives only parameters on line as solution.
  * (Expect parameters p_l = linePoint + lineDir * t points p_l on the line)
  * The calculation is done in the cylinder's coordinate system, expecting the cylinder
  * to be stretched in the y-direction.
  * \param cylRadius The cylinder's radius
  * \param cylHeight The cylinder's height (in dimension y)
  * \param linePoint A point on the line (in cylinder's coordinate system).
  * \param lineDir The (not necessarily normalized) direction of the line (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineToCylinder(
    agx::Real cylRadius,
    agx::Real cylHeight,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );




  /**
  * Clips an (infinite) line to a sphere.
  * Gives only parameters on line as solution.
  * (Expect parameters p_l = linePoint + lineDir * t points p_l on the line)
  * The calculation is done in the sphere's coordinate system.
  * \param sphereRadius The sphere's radius
  * \param linePoint A point on the line (in sphere's coordinate system).
  * \param lineDir The (not necessarily normalized) direction of the line (in sphere's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineToSphere(
    agx::Real sphereRadius,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an (infinite) line to a capsule.
  * Gives only parameters on line as solution.
  * (Expect parameters p_l = linePoint + lineDir * t points p_l on the line)
  * The calculation is done in the capsule's coordinate system, expecting the capsule
  * to be stretched in the y-direction.
  * \param capsRadius The capsule's radius
  * \param capsHeight The capsule's height (in dimension y)
  * \param linePoint A point on the line (in capsule's coordinate system).
  * \param lineDir The (not necessarily normalized) direction of the line (in capsule's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineToCapsule(
    agx::Real capsRadius,
    agx::Real capsHeight,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an (infinite) ray to a plane.
  * Gives only parameters on ray as solution.
  * (Expect p_l = rayStartPoint + rayDir * t, t >= 0, for all points p_l on the ray).
  * The calculation is done in the plane's coordinate system, expecting
  * the plane to encompass the origin,
  * and have the normal in one of the three standard axes.
  * \param planeDistance The positive distance of the plane to the origin of the plane's coordinate system in the plane normal (including sign).
  * \param planeDimension The standard axis which is the plane's normal.
  * \param planeNormalSign The sign of the plane normal (-1 or 1)
  * \param rayStartPoint The ray's starting point (in plane's coordinate system).
  * \param rayDir The (not necessarily normalized) direction of the ray (in plane's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipRayToPlane(
    agx::Real planeDistance,
    unsigned int planeDimension,
    int planeNormalSign,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an (infinite) ray to a slab (two parallel planes).
  * Gives only parameters on ray as solution.
  * (Expect p_l = rayStartPoint + rayDir * t, t >= 0, for all points p_l on the ray).
  * The calculation is done in the slab's coordinate system, expecting
  * both parallel planes to be at equal distance to the origin and
  * having the normal in one of the three standard axes.
  * \param slabDistance The slab's end distance from origin
  * \param slabDimension The standard axis which is parallel to the slab's normal.
  * \param rayStartPoint The ray's starting point (in slab's coordinate system).
  * \param rayDir The (not necessarily normalized) direction of the ray (in slab's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipRayToSlab(
    agx::Real slabDistance,
    unsigned int slabDimension,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an (infinite) ray to a double slab (two times two parallel planes).
  * Gives only parameters on ray as solution.
  * (Expect p_l = rayStartPoint + rayDir * t, t >= 0, for all points p_l on the ray).
  * The calculation is done in the double slab's coordinate system, expecting
  * both parallel planes each to be at equal distance to the origin and
  * having the normal in one of the three standard axes.
  * \param slab1Distance slab 1's end distance from origin
  * \param slab1Dimension The standard axis which is parallel to slab 1's normal.
  * \param slab2Distance slab 2's end distance from origin
  * \param slab2Dimension The standard axis which is parallel to the slab 2's normal.
  * \param rayStartPoint The ray's starting point (in cylinder's coordinate system).
  * \param rayDir The (not necessarily normalized) direction of the ray (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipRayToDoubleSlab(
    agx::Real slab1Distance,
    unsigned int slab1Dimension,
    agx::Real slab2Distance,
    unsigned int slab2Dimension,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an (infinite) ray to an open box (a box which is open on one side)
  * Gives only parameters on ray as solution.
  * (Expect p_l = rayStartPoint + rayDir * t, t >= 0, for all points p_l on the ray).
  * The calculation is done in the open box's coordinate system
  * \param boxHalfExtents The box's half extent lengths.
  * \param openDim The dimension in which the box is half open
  * \param openDimCutSign The sign against which shut be cut in the open dimension
  * \param rayDir The (not necessarily normalized) direction of the ray (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipRayToOpenBox(
    const agx::Vec3& boxHalfExtents,
    int openDim,
    int openDimCutSign,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );



  /**
  * Clips an (infinite) ray to a box.
  * Gives only parameters on ray as solution.
  * (Expect p_l = rayStartPoint + rayDir * t, t >= 0, for all points p_l on the ray).
  * The calculation is done in the double slab's coordinate system, expecting
  * both parallel planes each to be at equal distance to the origin and
  * having the normal in one of the three standard axes.
  * \param boxHalfExtents The box's half extents.
  * \param rayStartPoint The ray's starting point (in cylinder's coordinate system).
  * \param rayDir The (not necessarily normalized) direction of the ray (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipRayToBox(
    const agx::Vec3& boxHalfExtents,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );



  /**
  * Clips an (infinite) ray to an infinite cylinder.
  * Gives only parameters on ray as solution.
  * (Expect p_l = rayStartPoint + rayDir * t, t >= 0, for all points p_l on the ray).
  * The calculation is done in the cylinder's coordinate system, expecting the cylinder
  * to be stretched in the y-direction.
  * \param cylRadius The cylinder's radius
  * \param rayStartPoint The ray's starting point (in cylinder's coordinate system).
  * \param rayDir The (not necessarily normalized) direction of the ray (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipRayToInfiniteCylinder(
    agx::Real cylRadius,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );



  /**
  * Clips an (infinite) ray to a cylinder.
  * Gives only parameters on ray as solution.
  * (Expect p_l = rayStartPoint + rayDir * t, t >= 0, for all points p_l on the ray).
  * The calculation is done in the cylinder's coordinate system, expecting the cylinder
  * to be stretched in the y-direction.
  * \param cylRadius The cylinder's radius
  * \param cylHeight The cylinder's height (in dimension y)
  * \param rayStartPoint The ray's starting point (in cylinder's coordinate system).
  * \param rayDir The (not necessarily normalized) direction of the ray (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipRayToCylinder(
    agx::Real cylRadius,
    agx::Real cylHeight,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );



  /**
  * Clips an (infinite) ray to a sphere.
  * Gives only parameters on ray as solution.
  * (Expect p_l = rayStartPoint + rayDir * t, t >= 0, for all points p_l on the ray).
  * The calculation is done in the sphere's coordinate system.
  * \param sphereRadius The sphere's radius
  * \param rayStartPoint The ray's starting point (in sphere's coordinate system).
  * \param rayDir The (not necessarily normalized) direction of the ray (in sphere's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipRayToSphere(
    agx::Real sphereRadius,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );



  /**
  * Clips an (infinite) ray to a capsule.
  * Gives only parameters on ray as solution.
  * (Expect p_l = rayStartPoint + rayDir * t, t >= 0, for all points p_l on the ray).
  * The calculation is done in the capsule's coordinate system, expecting the capsule
  * to be stretched in the y-direction.
  * \param capsRadius The capsule's radius
  * \param capsHeight The capsule's height (in dimension y)
  * \param rayStartPoint The ray's starting point (in capsule's coordinate system).
  * \param rayDir The (not necessarily normalized) direction of the ray (in capsule's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipRayToCapsule(
    agx::Real capsRadius,
    agx::Real capsHeight,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );



  /**
  * Clips an lineSegment to a plane.
  * Gives only parameters on lineSegment as solution.
    * (Expect p_l = lineSegmentPoint0 + (lineSegmentPoint0 - lineSegmentPoint1) * t, 0 <= t <= 1, for all points p_l on the lineSegment).
  * The calculation is done in the plane's coordinate system, expecting
  * the plane to encompass the origin,
  * and have the normal in one of the three standard axes.
  * \param planeDistance The positive distance of the plane to the origin of the plane's coordinate system in the plane normal (including sign).
  * \param planeDimension The standard axis which is the plane's normal.
  * \param planeNormalSign The sign of the plane normal (-1 or 1)
  * \param lineSegmentPoint0 The lineSegment's starting point (in plane's coordinate system).
  * \param lineSegmentPoint1 The (not necessarily normalized) direction of the lineSegment (in plane's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineSegmentToPlane(
    agx::Real planeDistance,
    unsigned int planeDimension,
    int planeNormalSign,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );



  /**
  * Clips an lineSegment to a slab (two parallel planes).
  * Gives only parameters on lineSegment as solution.
    * (Expect p_l = lineSegmentPoint0 + (lineSegmentPoint0 - lineSegmentPoint1) * t, 0 <= t <= 1, for all points p_l on the lineSegment).
  * The calculation is done in the slab's coordinate system, expecting
  * both parallel planes to be at equal distance to the origin and
  * having the normal in one of the three standard axes.
  * \param slabDistance The slab's end distance from origin
  * \param slabDimension The standard axis which is parallel to the slab's normal.
  * \param lineSegmentPoint0 The lineSegment's starting point (in slab's coordinate system).
  * \param lineSegmentPoint1 The (not necessarily normalized) direction of the lineSegment (in slab's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineSegmentToSlab(
    agx::Real slabDistance,
    unsigned int slabDimension,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );



  /**
  * Clips an lineSegment to a double slab (two times two parallel planes).
  * Gives only parameters on lineSegment as solution.
    * (Expect p_l = lineSegmentPoint0 + (lineSegmentPoint0 - lineSegmentPoint1) * t, 0 <= t <= 1, for all points p_l on the lineSegment).
  * The calculation is done in the double slab's coordinate system, expecting
  * both parallel planes each to be at equal distance to the origin and
  * having the normal in one of the three standard axes.
  * \param slab1Distance slab 1's end distance from origin
  * \param slab1Dimension The standard axis which is parallel to slab 1's normal.
  * \param slab2Distance slab 2's end distance from origin
  * \param slab2Dimension The standard axis which is parallel to the slab 2's normal.
  * \param lineSegmentPoint0 The lineSegment's starting point (in cylinder's coordinate system).
  * \param lineSegmentPoint1 The (not necessarily normalized) direction of the lineSegment (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineSegmentToDoubleSlab(
    agx::Real slab1Distance,
    unsigned int slab1Dimension,
    agx::Real slab2Distance,
    unsigned int slab2Dimension,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );



  /**
  * Clips an lineSegment to an open box (a box which is open on one side)
  * Gives only parameters on lineSegment as solution.
    * (Expect p_l = lineSegmentPoint0 + (lineSegmentPoint0 - lineSegmentPoint1) * t, 0 <= t <= 1, for all points p_l on the lineSegment).
  * The calculation is done in the open box's coordinate system
  * \param boxHalfExtents The box's half extents.
  * \param openDim The dimension in which the box is half open
  * \param openDimCutSign The sign against which shut be cut in the open dimension
  * \param lineSegmentPoint1 The (not necessarily normalized) direction of the lineSegment (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineSegmentToOpenBox(
    const agx::Vec3& boxHalfExtents,
    int openDim,
    int openDimCutSign,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an lineSegment to a box.
  * Gives only parameters on lineSegment as solution.
    * (Expect p_l = lineSegmentPoint0 + (lineSegmentPoint0 - lineSegmentPoint1) * t, 0 <= t <= 1, for all points p_l on the lineSegment).
  * The calculation is done in the double slab's coordinate system, expecting
  * both parallel planes each to be at equal distance to the origin and
  * having the normal in one of the three standard axes.
  * \param boxHalfExtents The box's half extents.
  * \param lineSegmentPoint0 The lineSegment's starting point (in cylinder's coordinate system).
  * \param lineSegmentPoint1 The (not necessarily normalized) direction of the lineSegment (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineSegmentToBox(
    const agx::Vec3& boxHalfExtents,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );



  /**
  * Clips an lineSegment to an infinite cylinder.
  * Gives only parameters on lineSegment as solution.
    * (Expect p_l = lineSegmentPoint0 + (lineSegmentPoint0 - lineSegmentPoint1) * t, 0 <= t <= 1, for all points p_l on the lineSegment).
  * The calculation is done in the cylinder's coordinate system, expecting the cylinder
  * to be stretched in the y-direction.
  * \param cylRadius The cylinder's radius
  * \param lineSegmentPoint0 The lineSegment's starting point (in cylinder's coordinate system).
  * \param lineSegmentPoint1 The (not necessarily normalized) direction of the lineSegment (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineSegmentToInfiniteCylinder(
    agx::Real cylRadius,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an lineSegment to a cylinder.
  * Gives only parameters on lineSegment as solution.
  * (Expect p_l = lineSegmentPoint0 + (lineSegmentPoint0 - lineSegmentPoint1) * t, 0 <= t <= 1, for all points p_l on the lineSegment).
  * The calculation is done in the cylinder's coordinate system, expecting the cylinder
  * to be stretched in the y-direction.
  * \param cylRadius The cylinder's radius
  * \param cylHeight The cylinder's height (in dimension y)
  * \param lineSegmentPoint0 The lineSegment's starting point (in cylinder's coordinate system).
  * \param lineSegmentPoint1 The (not necessarily normalized) direction of the lineSegment (in cylinder's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.\r\n
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineSegmentToCylinder(
    agx::Real cylRadius,
    agx::Real cylHeight,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );



  /**
  Clips a line segment to a cone.
  Returns true if segment intersected with cone. If return value is true,
  parametric values are stored to interval.
  The inputs are specified in the cones coordinate system where
  the cone base is at the xz-plane at y=0 and the cone top at the xz-plane y=coneHeight.

  \param bottomRadius Bottom radius of the cone.
  \param topRadius Top radius of the cone, must be smaller than the bottom radius
  \param coneHeight Height of the cone.
  \param lineSegmentPoint0 Start of line in cones coordinate system
  \param lineSegmentPoint1 End of line in cones coordinate system
  \param interval Solution if an overlap was found
  \retur True if overlap was found
  */
  AGXPHYSICS_EXPORT bool clipLineSegmentToCone(
    agx::Real bottomRadius,
    agx::Real topRadius,
    agx::Real coneHeight,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an lineSegment to a sphere.
  * Gives only parameters on lineSegment as solution.
  * (Expect p_l = lineSegmentPoint0 + (lineSegmentPoint0 - lineSegmentPoint1) * t, 0 <= t <= 1, for all points p_l on the lineSegment).
  * The calculation is done in the sphere's coordinate system.
  * \param sphereRadius The sphere's radius
  * \param lineSegmentPoint0 The lineSegment's starting point (in sphere's coordinate system).
  * \param lineSegmentPoint1 The (not necessarily normalized) direction of the lineSegment (in sphere's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.\r\n
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineSegmentToSphere(
    agx::Real sphereRadius,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );


  /**
  * Clips an lineSegment to a capsule.
  * Gives only parameters on lineSegment as solution.
  * (Expect p_l = lineSegmentPoint0 + (lineSegmentPoint0 - lineSegmentPoint1) * t, 0 <= t <= 1, for all points p_l on the lineSegment).
  * The calculation is done in the capsule's coordinate system, expecting the capsule
  * to be stretched in the y-direction.
  * \param capsRadius The capsule's radius
  * \param capsHeight The capsule's height (in dimension y)
  * \param lineSegmentPoint0 The lineSegment's starting point (in capsule's coordinate system).
  * \param lineSegmentPoint1 The (not necessarily normalized) direction of the lineSegment (in capsule's coordinate system).
  * \param interval: The solution of tMin and tMax as the line parameters in the equation given above.\r\n
  * \retval: Any intersection found? If false, the values in the interval have undefined values.
  */
  AGXPHYSICS_EXPORT bool clipLineSegmentToCapsule(
    agx::Real capsRadius,
    agx::Real capsHeight,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON );




  /* Implementation */

  AGX_FORCE_INLINE bool clipLineToDoubleSlab(
    agx::Real slab1Distance,
    unsigned int slab1Dimension,
    agx::Real slab2Distance,
    unsigned int slab2Dimension,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (clipLineToSlab( slab1Distance, slab1Dimension, linePoint, lineDir, interval, epsilon )) {
      agx::Interval tmpInt;
      if (clipLineToSlab( slab2Distance, slab2Dimension, linePoint, lineDir, tmpInt, epsilon )) {
        interval.intersectWith( tmpInt );
        return interval.hasElements();
      }
    }
    return false;
  }

  AGX_FORCE_INLINE bool clipLineToOpenBox(
    const agx::Vec3& boxHalfExtents,
    int openDim,
    int openDimCutSign,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    {
      int dim1 = (openDim + 1) % 3;
      if (!clipLineToSlab( boxHalfExtents[dim1], dim1, linePoint, lineDir, interval, epsilon ))
        return false;
    }
    {
      agx::Interval tmpInterval0;
      int dim2 = (openDim + 2) % 3;
      if (!clipLineToSlab( boxHalfExtents[dim2], dim2, linePoint, lineDir, tmpInterval0, epsilon ))
        return false;
      interval.intersectWith( tmpInterval0 );
      if (interval.isEmpty())
        return false;

    }
    {
      agx::Interval tmpInterval1;
      if (!clipLineToPlane( boxHalfExtents[openDim], openDim, openDimCutSign, linePoint, lineDir, tmpInterval1, epsilon ))
        return false;

      interval.intersectWith( tmpInterval1 );
      return interval.hasElements();
    }
  }

  AGX_FORCE_INLINE bool clipLineToBox(
    const agx::Vec3& boxHalfExtents,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (!clipLineToSlab( boxHalfExtents[0], 0, linePoint, lineDir, interval, epsilon ))
      return false;
    {
      agx::Interval tmpInterval0;
      if (!clipLineToSlab( boxHalfExtents[1], 1, linePoint, lineDir, tmpInterval0, epsilon ))
        return false;
      interval.intersectWith( tmpInterval0 );
      if (interval.isEmpty())
        return false;
    }
    {
      agx::Interval tmpInterval1;
      if (!clipLineToSlab( boxHalfExtents[2], 2, linePoint, lineDir, tmpInterval1, epsilon ))
        return false;
      interval.intersectWith( tmpInterval1 );
      return interval.hasElements();
    }
  }

  AGX_FORCE_INLINE bool clipLineToCylinder(
    agx::Real cylRadius,
    agx::Real cylHeight,
    const agx::Vec3& linePoint,
    const agx::Vec3& lineDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (!clipLineToSlab( agx::Real( 0.5 ) * cylHeight, 1, linePoint, lineDir, interval, epsilon ))
      return false;
    agx::Interval tmpInterval;
    if (!clipLineToInfiniteCylinder( cylRadius, linePoint, lineDir, tmpInterval, epsilon ))
      return false;
    interval.intersectWith( tmpInterval );
    return interval.hasElements();
  }

  AGX_FORCE_INLINE bool clipRayToPlane(
    agx::Real planeDistance,
    unsigned int planeDimension,
    int planeNormalSign,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (!clipLineToPlane( planeDistance, planeDimension, planeNormalSign,
      rayStartPoint, rayDir, interval, epsilon ))
      return false;
    interval.intersectWithRightUnbounded( 0 );
    return interval.hasElements();
  }

  AGX_FORCE_INLINE bool clipRayToSlab(
    agx::Real slabDistance,
    unsigned int slabDimension,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (clipLineToSlab( slabDistance, slabDimension, rayStartPoint, rayDir, interval, epsilon )) {
      interval.intersectWithRightUnbounded( 0 );
      return interval.hasElements();
    }
    return false;
  }

  AGX_FORCE_INLINE bool clipRayToDoubleSlab(
    agx::Real slab1Distance,
    unsigned int slab1Dimension,
    agx::Real slab2Distance,
    unsigned int slab2Dimension,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (clipRayToSlab( slab1Distance, slab1Dimension, rayStartPoint, rayDir, interval, epsilon )) {
      agx::Interval tmpInt;
      if (clipRayToSlab( slab2Distance, slab2Dimension, rayStartPoint, rayDir, tmpInt, epsilon )) {
        interval.intersectWith( tmpInt );
        return interval.hasElements();
      }
    }
    return false;
  }


  AGX_FORCE_INLINE bool clipRayToOpenBox(
    const agx::Vec3& boxHalfExtents,
    int openDim,
    int openDimCutSign,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    {
      int dim1 = (openDim + 1) % 3;
      if (!clipRayToSlab( boxHalfExtents[dim1], dim1, rayStartPoint, rayDir, interval, epsilon ))
        return false;
    }
    {
      agx::Interval tmpInterval0;
      int dim2 = (openDim + 2) % 3;
      if (!clipRayToSlab( boxHalfExtents[dim2], dim2, rayStartPoint, rayDir, tmpInterval0, epsilon ))
        return false;
      interval.intersectWith( tmpInterval0 );
      if (interval.isEmpty())
        return false;
    }
    {
      agx::Interval tmpInterval1;
      if (!clipRayToPlane( boxHalfExtents[openDim], openDim, openDimCutSign, rayStartPoint, rayDir, tmpInterval1, epsilon ))
        return false;

      interval.intersectWith( tmpInterval1 );
      return interval.hasElements();
    }
  }

  AGX_FORCE_INLINE bool clipRayToBox(
    const agx::Vec3& boxHalfExtents,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (!clipRayToSlab( boxHalfExtents[0], 0, rayStartPoint, rayDir, interval, epsilon ))
      return false;
    {
      agx::Interval tmpInterval0;
      if (!clipRayToSlab( boxHalfExtents[1], 1, rayStartPoint, rayDir, tmpInterval0, epsilon ))
        return false;
      interval.intersectWith( tmpInterval0 );
      if (interval.isEmpty())
        return false;
    }
    {
      agx::Interval tmpInterval1;
      if (!clipRayToSlab( boxHalfExtents[2], 2, rayStartPoint, rayDir, tmpInterval1, epsilon ))
        return false;
      interval.intersectWith( tmpInterval1 );
      return interval.hasElements();
    }
  }


  AGX_FORCE_INLINE bool clipRayToInfiniteCylinder(
    agx::Real cylRadius,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (clipLineToInfiniteCylinder( cylRadius, rayStartPoint, rayDir, interval, epsilon )) {
      interval.intersectWithRightUnbounded( 0 );
      return interval.hasElements();
    }
    return false;
  }

  AGX_FORCE_INLINE bool clipRayToCylinder(
    agx::Real cylRadius,
    agx::Real cylHeight,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (!clipRayToSlab( agx::Real( 0.5 ) * cylHeight, 1, rayStartPoint, rayDir, interval, epsilon ))
      return false;
    agx::Interval tmpInterval;
    if (!clipRayToInfiniteCylinder( cylRadius, rayStartPoint, rayDir, tmpInterval, epsilon ))
      return false;
    interval.intersectWith( tmpInterval );
    return interval.hasElements();
  }

  AGX_FORCE_INLINE bool clipRayToSphere(
    agx::Real sphereRadius,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (clipLineToSphere( sphereRadius, rayStartPoint, rayDir, interval, epsilon )) {
      interval.intersectWithRightUnbounded( 0 );
      return interval.hasElements();
    }
    return false;
  }

  AGX_FORCE_INLINE bool clipRayToCapsule(
    agx::Real capsRadius,
    agx::Real capsHeight,
    const agx::Vec3& rayStartPoint,
    const agx::Vec3& rayDir,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (clipLineToCapsule( capsRadius, capsHeight, rayStartPoint, rayDir, interval, epsilon )) {
      interval.intersectWithRightUnbounded( 0 );
      return interval.hasElements();
    }
    return false;
  }

  AGX_FORCE_INLINE bool clipLineSegmentToPlane(
    agx::Real planeDistance,
    unsigned int planeDimension,
    int planeNormalSign,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (!clipLineToPlane( planeDistance, planeDimension, planeNormalSign,
      lineSegmentPoint0, lineSegmentPoint1 - lineSegmentPoint0, interval, epsilon )) {
      return false;
    }
    interval.intersectWith( agx::Interval( -agx::RealEpsilon, agx::Real(1)+agx::RealEpsilon ));
    return interval.hasElements();
  }

  AGX_FORCE_INLINE bool clipLineSegmentToSlab(
    agx::Real slabDistance,
    unsigned int slabDimension,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (clipLineToSlab( slabDistance, slabDimension, lineSegmentPoint0, lineSegmentPoint1 - lineSegmentPoint0, interval, epsilon )) {
      interval.intersectWith( agx::Interval( -agx::RealEpsilon, agx::Real(1)+agx::RealEpsilon ));
      return interval.hasElements();
    }
    return false;
  }

  AGX_FORCE_INLINE bool clipLineSegmentToDoubleSlab(
    agx::Real slab1Distance,
    unsigned int slab1Dimension,
    agx::Real slab2Distance,
    unsigned int slab2Dimension,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (clipLineSegmentToSlab( slab1Distance, slab1Dimension, lineSegmentPoint0, lineSegmentPoint1, interval, epsilon )) {
      agx::Interval tmpInt;
      if (clipLineSegmentToSlab( slab2Distance, slab2Dimension, lineSegmentPoint0, lineSegmentPoint1, tmpInt, epsilon )) {
        interval.intersectWith( tmpInt );
        return interval.hasElements();
      }
    }
    return false;
  }

  AGX_FORCE_INLINE bool clipLineSegmentToOpenBox(
    const agx::Vec3& boxHalfExtents,
    int openDim,
    int openDimCutSign,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    {
      int dim1 = (openDim + 1) % 3;
      if (!clipLineSegmentToSlab( boxHalfExtents[dim1], dim1, lineSegmentPoint0, lineSegmentPoint1, interval, epsilon ))
        return false;
    }
    {
      agx::Interval tmpInterval0;
      int dim2 = (openDim + 2) % 3;
      if (!clipLineSegmentToSlab( boxHalfExtents[dim2], dim2, lineSegmentPoint0, lineSegmentPoint1, tmpInterval0, epsilon ))
        return false;
      interval.intersectWith( tmpInterval0 );
      if (interval.isEmpty())
        return false;
    }
    {
      agx::Interval tmpInterval1;
      if (!clipLineSegmentToPlane( boxHalfExtents[openDim], openDim, openDimCutSign, lineSegmentPoint0, lineSegmentPoint1, tmpInterval1, epsilon ))
        return false;

      interval.intersectWith( tmpInterval1 );
      return interval.hasElements();
    }
  }

  AGX_FORCE_INLINE bool clipLineSegmentToBox(
    const agx::Vec3& boxHalfExtents,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (!clipLineSegmentToSlab( boxHalfExtents[0], 0, lineSegmentPoint0, lineSegmentPoint1, interval, epsilon ))
      return false;
    {
      agx::Interval tmpInterval0;
      if (!clipLineSegmentToSlab( boxHalfExtents[1], 1, lineSegmentPoint0, lineSegmentPoint1, tmpInterval0, epsilon ))
        return false;
      interval.intersectWith( tmpInterval0 );
      if (interval.isEmpty())
        return false;
    }
    {
      agx::Interval tmpInterval1;
      if (!clipLineSegmentToSlab( boxHalfExtents[2], 2, lineSegmentPoint0, lineSegmentPoint1, tmpInterval1, epsilon ))
        return false;
      interval.intersectWith( tmpInterval1 );
      return interval.hasElements();
    }
  }

  AGX_FORCE_INLINE bool clipLineSegmentToInfiniteCylinder(
    agx::Real cylRadius,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (clipLineToInfiniteCylinder( cylRadius, lineSegmentPoint0, lineSegmentPoint1 - lineSegmentPoint0, interval, epsilon )) {
      interval.intersectWith( agx::Interval( -agx::RealEpsilon, agx::Real(1)+agx::RealEpsilon ));
      return interval.hasElements();
    }
    return false;
  }

  AGX_FORCE_INLINE bool clipLineSegmentToCylinder(
    agx::Real cylRadius,
    agx::Real cylHeight,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (!clipLineSegmentToSlab( agx::Real( 0.5 ) * cylHeight, 1, lineSegmentPoint0, lineSegmentPoint1, interval, epsilon ))
      return false;
    agx::Interval tmpInterval;
    if (!clipLineSegmentToInfiniteCylinder( cylRadius, lineSegmentPoint0, lineSegmentPoint1, tmpInterval, epsilon ))
      return false;
    interval.intersectWith( tmpInterval );
    return interval.hasElements();
  }

  AGX_FORCE_INLINE bool clipLineSegmentToSphere(
    agx::Real sphereRadius,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (clipLineToSphere( sphereRadius, lineSegmentPoint0, lineSegmentPoint1 - lineSegmentPoint0, interval, epsilon )) {
      interval.intersectWith( agx::Interval( -agx::RealEpsilon, agx::Real(1)+agx::RealEpsilon ));
      return interval.hasElements();
    }
    return false;
  }

  AGX_FORCE_INLINE bool clipLineSegmentToCapsule(
    agx::Real capsRadius,
    agx::Real capsHeight,
    const agx::Vec3& lineSegmentPoint0,
    const agx::Vec3& lineSegmentPoint1,
    agx::Interval& interval,
    const agx::Real epsilon)
  {
    if (clipLineToCapsule( capsRadius, capsHeight, lineSegmentPoint0, lineSegmentPoint1 - lineSegmentPoint0, interval, epsilon )) {
      interval.intersectWith( agx::Interval( -agx::RealEpsilon, agx::Real(1)+agx::RealEpsilon ));
      return interval.hasElements();
    }
    return false;
  }

}

DOXYGEN_END_INTERNAL_BLOCK()
#endif
