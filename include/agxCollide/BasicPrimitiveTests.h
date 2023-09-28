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

#ifndef AGXCOLLIDE_BASICPRIMITIVETESTS_H
#define AGXCOLLIDE_BASICPRIMITIVETESTS_H

#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()

#include <agx/agx_vector_types.h>
#include <agx/agxPhysics_export.h>
#include <agx/StackArray.h>
#include <agx/Plane.h>
#include <agx/Integer.h>

#ifndef _WIN32
#include <float.h>
#endif

#include <agx/agx.h>
#include <agx/Vec3.h>

namespace agxCollide
{
  class Mesh;


  /// When are two normalized directions parallel enough for a second contact point?
  const agx::Real parallelityThreshold = agx::Real(0.86602540378444); // sin(60°) and cos(30°); choice is motivated by testing.


  /**
  Are two (normalized) directions parallel to each other?
  Calculated by measuring cosine between the directions.
  \param dir0 The first normalized direction.
  \param dir1 The second normalized direction.
  \param cosEpsilon The threshold for the cosine of the angle between plane normal and direction.
  \retval Are the directions parallel (the cosine of their relative angle is larger than the epsilon)?
  */
  bool areDirectionsParallel(const agx::Vec3& dir0, const agx::Vec3& dir1,
    const agx::Real cosEpsilon = parallelityThreshold);


  /**
  Are two (normalized) directions orthogonal to each other?
  Calculated by measuring cosine between the directions.
  \param dir0 The first normalized direction.
  \param dir1 The second normalized direction.
  \param sinEpsilon The threshold for the sine of the angle between plane the directions.
  \retval Are the directions orthogonal (the sine of the angle is smaller than the epsilon)?
  */
  bool areDirectionsOrthogonal(const agx::Vec3& dir0, const agx::Vec3& dir1,
    const agx::Real sinEpsilon = parallelityThreshold);


  /**
  * Find intersection point for line segment and hyperplane.
  * This works in all dimensions (as long as the inner product is defined),
  * so T can be a scalar, an agx::Vec2, an agx::Vec3, agx::Vec4...
  *
  *\param lineP1 First point of line
  *\param lineP2 Second point of line
  *\param normal hyperplane normal
  *\param d plane distance
  *\param t return value: parameter along segment (0 to 1). Makes only sense if function returns true.
  *\param result Result: Q = lineP1 + (lineP2 - lineP1) * t
  *\param epsilon The epsilon value for comparisons.
  *\retval True if the line segment intersects the plane
  */
  template <typename T>
  bool intersectLineSegmentHyperPlane(
    const T& lineP1,
    const T& lineP2,
    const T& normal,
    agx::Real d,
    agx::Real& t,
    T& result,
    const agx::Real epsilon );


  /**
  * Find intersection point for line and hyperplane.
  * This works in all dimensions (as long as the inner product is defined),
  * so T can be a scalar, an agx::Vec2, an agx::Vec3, agx::Vec4...
  *
  *\param linePoint First point of line
  *\param lineDir Direction of line (not necessary normalized)
  *\param normal hyperplane normal
  *\param d plane distance
  *\param t return value: parameter along segment (0 to 1). Makes only sense if function returns true.
  *\param result Result: Q = lineP1 + (lineP2 - lineP1) * t
  *\param epsilon The epsilon value for comparisons.
  *\retval True if the line segment intersects the plane
  */
  template <typename T>
  bool intersectLineHyperPlane(
    const T& linePoint,
    const T& lineDir,
    const T& normal,
    agx::Real d,
    agx::Real& t,
    T& result,
    const agx::Real epsilon );


  /**
  *Returns the closest point on the triangle to point point P.
  *
  *\param p  The point P
  *\param a  first vertex of triangle
  *\param b  second vertex of triangle
  *\param c  third vertex of triangle
  *\param voronoiRegion  In which of the triangle's Voronoi regions is the point?
  *           0 to 2: point a to c, 3 to 5: edge AB to CA, 6: face
  *\return  the closest point on the triangle
  */
  AGXPHYSICS_EXPORT agx::Vec3 closestPointPointTriangle( const agx::Vec3& p, const agx::Vec3& a,
    const agx::Vec3& b, const agx::Vec3& c, uint8_t& voronoiRegion );



  AGX_FORCE_INLINE bool pointOnLine( const agx::Vec3& point,
    const agx::Vec3& lineP0, const agx::Vec3& lineP1, agx::Real epsilon )
  {
    const agx::Vec3 dir = lineP1 - lineP0;
    const agx::Real dirLen2 = dir.length2();
    if (agx::equalsZero(dirLen2, agx::RealEpsilon * agx::RealEpsilon)) {
      // Line segment degenerates to point, measure distance from point to point.
      return agx::equalsZero((lineP0 - point).length2(), epsilon * epsilon);
    }
    const agx::Vec3 pointP0 = lineP0 - point;
    const agx::Vec3 dist = pointP0 - dir * (dir * pointP0) / dirLen2;
    return dist.length2() < epsilon * epsilon;
  }


  /**
  * Test if point p on the triangle plane lies inside the counterclockwise triangle abc.
  *\param p Point on triangle plane
  *\param pointA First vertex of triangle
  *\param pointB Second vertex of triangle
  *\param pointC Third vertex of triangle
  *\param epsilon boundary around triangle to counter floating point errors
  *
  *\return True if p lies inside abc
  */
  AGXPHYSICS_EXPORT bool pointInTrianglePrism(
    const agx::Vec3& p,
    const agx::Vec3& pointA,
    const agx::Vec3& pointB,
    const agx::Vec3& pointC,
    const agx::Real epsilon = agx::RealEpsilon );


  /**
  * Test if point p lies inside the counterclockwise triangle abc.
  *\param p Point
  *\param a First vertex of triangle
  *\param b Second vertex of triangle
  *\param c Third vertex of triangle
  *\param n The triangle normal
  *\param epsilon boundary around triangle to counter floating point errors
  *
  *\return True if p lies inside abc
  */
  AGXPHYSICS_EXPORT bool pointInTriangle(
    const agx::Vec3& p,
    const agx::Vec3& a,
    const agx::Vec3& b,
    const agx::Vec3& c,
    const agx::Vec3& n,
    const agx::Real epsilon = agx::RealEpsilon );


#ifndef SWIG
  /**
  Computes barycentric coordinates u, v, w of point p on triangle a, b, c with normal n.
  Clamps if point is outside.
  \param u the weight of a.
  \param v the weight of b.
  \param w the weight of c.
  */
#endif
  void AGXPHYSICS_EXPORT computeBarycentricCoordinates(
    const agx::Vec3& p,
    const agx::Vec3& a,
    const agx::Vec3& b,
    const agx::Vec3& c,
    const agx::Vec3& /*n*/,
    agx::Real& u,
    agx::Real& v,
    agx::Real& w);

  /**
  * Test if point p lies inside the box.
  *\param p              - Point in box coordinate system.
  *\param boxHalfExtents - half extents of box.
  *\param epsilon        - boundary around box that give false positives.
  *
  *\return True if p lies inside the box
  */
  AGXPHYSICS_EXPORT bool pointInBox(
    const agx::Vec3& p,
    const agx::Vec3& boxHalfExtents,
    const agx::Real epsilon = agx::RealEpsilon );

  /**
  * Intersection test for line segment and triangle.
  * Here, the triangle normal is given and does not have
  * to be computed.
  *
  *\param lineP1  First point of line
  *\param lineP2  Second point of line
  *\param triangleP1 First vertex of triangle
  *\param triangleP2 Second vertex of triangle
  *\param triangleP3 Third vertex of triangle
  *\param normal The triangle normal
  *\param result  The intersection point, if any, is stored here
  *\param t Intersection parameter: Result = lineP1 + t * (lineP2 - lineP1)
  *\param isFrontFace True if intersects triangle front face, false if back face
  *\param epsilon boundary around triangle to counter floating point errors
  *@return True if the line intersects the triangle
  */
  AGXPHYSICS_EXPORT bool intersectLineSegmentTriangle(
    const agx::Vec3& lineP1,
    const agx::Vec3& lineP2,
    const agx::Vec3& triangleP1,
    const agx::Vec3& triangleP2,
    const agx::Vec3& triangleP3,
    const agx::Vec3& normal,
    agx::Vec3& result,
    agx::Real& t,
    bool& isFrontFace,
    const agx::Real epsilon = agx::RealEpsilon );


  //=============================================================================


  /**
  * Test if line p1p2 intersects the open cylinder pqr (no endcaps).
  *\param p1 First point of line
  *\param p2 Second point of line
  *\param p First point of cylinder
  *\param q Second point of cylinder
  *\param r Radius of cylinder
  *\param t Result: Intersection point: p1p2(t)
  *
  *\return True if intersection
  */
  AGXPHYSICS_EXPORT bool intersectLineOpenCylinder( const agx::Vec3& p1, const agx::Vec3& p2,
    const agx::Vec3& p, const agx::Vec3& q,
    agx::Real r, agx::Real& t );

  /**
  * Test if line p1p2 intersects the sphere center-r.
  *\param p1  First point of line
  *\param p2  Second point of line
  *\param center Sphere center
  *\param r  Sphere radius
  *\param q  Result: Intersection point
  *\param t  Result: Intersection point: p1p2(t)
  *
  *\return True if intersection
  */
  AGXPHYSICS_EXPORT bool intersectLineSphere( const agx::Vec3& p1, const agx::Vec3& p2, const agx::Vec3& center,
    agx::Real r, agx::Vec3& q, agx::Real& t );

  AGXPHYSICS_EXPORT bool sweptSphereTriangle( const agx::Vec3& p0, const agx::Vec3& p1, const
    agx::Real& radius, const agx::Vec3& v0, const agx::Vec3& v1,
    const agx::Vec3& v2, agx::Vec3& contactPoint,
    agx::Vec3& normal, agx::Real& time );

  AGXPHYSICS_EXPORT bool sphereTriangle( agx::Vec3 p0, agx::Vec3 p1, agx::Vec3 p2,
    const agx::Vec3& center, const agx::Real& radius,
    agx::Vec3& point, agx::Vec3& normal, agx::Real& depth );


  /**
  Finds the closest points between to line segments.
  \param start1 starting point first line segment
  \param end1 ending point first line segment
  \param start2 starting point first line segment
  \param end2 ending point first line segment
  \param pointOn1: returns closest point for first pair on first line segment
  \param pointOn2: returns closest point for first pair on second line segment
  \param t1: returns resulting parameter on line segment 1 (in interval [0, 1]).
  \param t2: returns resulting parameter on line segment 2 (in interval [0, 1]).
  \param isParallel: returns true if the lines are parallel. Two pairs of points are found then.
  \param parallelPointOn1: returns closest point for second pair (if existing) on first line segment
  \param parallelPointOn2: returns closest point for second pair (if existing) on second line segment
  \param parallelT1: returns parameter for first pair (if existing) on first line segment (in interval [0, 1]).
  \param parallelT2: returns parameter for second pair (if existing) on first line segment (in interval [0, 1]).
  */
  AGXPHYSICS_EXPORT void closestPointsSegmentSegment(
    const agx::Vec3& start1,
    const agx::Vec3& end1,
    const agx::Vec3& start2,
    const agx::Vec3& end2,
    agx::Vec3& pointOn1,
    agx::Vec3& pointOn2,
    agx::Real& t1,
    agx::Real& t2,
    bool& isParallel,
    agx::Vec3& parallelPointOn1,
    agx::Vec3& parallelPointOn2,
    agx::Real& parallelT1,
    agx::Real& parallelT2,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON);

  /// Small POD-struct holding results from closest point query.
  class ClosestPointsSolution
  {
    public:
      agx::Vec3 pointOn1; //Closest point for first pair on first line segment.
      agx::Vec3 pointOn2; //Closest point for first pair on second line segment.
      agx::Real t1;       //Parameter on first line segment (in interval [0, 1]).
      agx::Real t2;       //Parameter on second line segment (in interval [0, 1]).
  };


  /**
  Finds the closest points between to line segments.
  \param start1 starting point first line segment
  \param end1 ending point first line segment
  \param start2 starting point first line segment
  \param end2 ending point first line segment
  \param solution returns the found closest points.
  \param isParallel: returns true if the lines are parallel. Two pairs of points are found then.
  \param parallelSolution returns the second solution in case there the configuration is parallel.
  */
  void closestPointsSegmentSegment(
    const agx::Vec3& start1,
    const agx::Vec3& end1,
    const agx::Vec3& start2,
    const agx::Vec3& end2,
    ClosestPointsSolution& solution,
    bool& isParallel,
    ClosestPointsSolution& parallelSolution,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON
    );


  /**
  * Finds the closest point on a line segment to a given point.
  * \param p The given point.
  * \param s0 The segment starting point.
  * \param s1 The segment ending point.
  * \param t Returns the parameter on the line segment, from 0 (s0) to 1 (s1).
  * \retval The point on the line segment.
  */
  agx::Vec3 closestPointPointSegment(
    const agx::Vec3& p,
    const agx::Vec3& s0,
    const agx::Vec3& s1,
    agx::Real& t );


  // A convex polygon of n >= 3 vertices is clipped against slab
  // (box open in 2 dimensions).
  // The resulting polygon can have up to 2 more vertices than the input data.
  // The result array should be allocated outside the function.
  // \param cPolygon  the points of the convex polygon,
  //                  in counter clockwise or clockwise order
  // \param slabDistance  The distance in of the slab (in + and - in clipDim)
  // \param clipDim  the dimension which is clipped against. 0, 1 or 2
  // \param result  the resulting clipped points. Size N + 2
  // \param relativeEpsilon  epsilon for compensating errors due to
  //                         floating point arithmetics
  // \return found any vertices
  //          (if no, the polygon lies completely outside the slab)
  template <size_t N>
  bool clipConvexPolygonAgainstSlab(
    const agx::StackArray<agx::Vec3, N>& cPolygon,
    const agx::Real slabDistance,
    const uint8_t clipDim,
    agx::StackArray<agx::Vec3, N + 2>& result,
    const agx::Real relativeEpsilon );


  // A convex polygon of n >= 3 vertices is clipped against slab
  // (box open in 2 dimensions).
  // The resulting polygon can have up to 2 more vertices than the input data.
  // The result array should be allocated outside the function.
  // \param cPolygon  the points of the convex polygon,
  //                  in counter clockwise or clockwise order
  // \param boxHe The box half edge
  // \param tmp Some temporary memory. Given for optimization purposes.
  // \param result  the resulting clipped points. Size N + 6
  // \param relativeEpsilon  epsilon for compensating errors due to
  //                         floating point arithmetics
  // \return found any vertices
  //          (if no, the polygon lies completely outside the slab)
  template <size_t N>
  bool clipConvexPolygonAgainstBox(
    const agx::StackArray<agx::Vec3, N>& cPolygon,
    const agx::Vec3& boxHe,
    agx::StackArray<agx::Vec3, N + 6>& tmp,
    agx::StackArray<agx::Vec3, N + 6>& result,
    const agx::Real relativeEpsilon );


  // A convex polygon of n >= 3 vertices is clipped against slab
  // (box open in 2 dimensions).
  // The resulting polygon can have up to 2 more vertices than the input data.
  // The result array should be allocated outside the function.
  // \param cPolygon  the points of the convex polygon,
  //                  in counter clockwise or clockwise order
  // \param numVertsPolygon  number of vertices in the polygon
  // \param openDim  the dimension which is clipped against. 0, 1 or 2
  // \param result  the resulting clipped points. Size N + 2
  // \param relativeEpsilon  epsilon for compensating errors due to
  //                         floating point arithmetics
  // \return found any vertices
  //          (if no, the polygon lies completely outside the slab)
  AGXPHYSICS_EXPORT bool clipLineSegmentAgainstSlab(
    const agx::Vec3& lineP0,
    const agx::Vec3& lineP1,
    const agx::Real slabDistance,
    int clipDim,
    agx::StackArray<agx::Vec3, 2>& result,
    const agx::Real relativeEpsilon );


  // clips a line segment against a point in 1D.
  // The input data is three dimensional, all three dimensions are clipped
  // by the factor calculated from the 1d calculations.
  // \param a  segment's starting point
  // \param a  segment's ending point
  // \param dim  dimensional dim. 0, 1 or 2
  // \param clipBoundary  1d value that the line segment should be clipped against
  // \param result The clipped point. In parallel case, b.
  // \param relativeEpsilon
  // \retval Did the line segment get clipped? (No if both points on equal side)
  bool clipSegmentLineSemi1D(
    const agx::Vec3& a,
    const agx::Vec3& b,
    int dim,
    agx::Real clipBoundary,
    agx::Vec3& result,
    const agx::Real relativeEpsilon = agx::RealEpsilon);

  /**
  *   calculates how deep a point has come into the cylinder along a given normal
  *   everything has to be given in the Cylinder's frame
  */
  agx::Real AGXPHYSICS_EXPORT calculateDepthInCylinder ( agx::Real cylRadius,
    agx::Real cylHeight, const agx::Vec3& point, const agx::Vec3& normal,
    agx::Real relativeEpsilon );


  /// Clips line segment to the mantle of an (infinite) cylinder, stretching out in y-direction
  bool AGXPHYSICS_EXPORT clipLineSegmentToCylinderMantle( agx::Real cylRadius,
    const agx::Vec3& p0,
    const agx::Vec3& p1,
    agx::StackArray<agx::Vec3, 2>& pOut,
  const agx::Real epsilon = agx::REAL_SQRT_EPSILON );


  /**
  * Clips line segment to a) cylinder, stretching out in y-direction
  * Returns true if any overlap is found.
  */
  bool AGXPHYSICS_EXPORT clipLineSegmentToCylinder(
    agx::Real cylRadius, agx::Real cylHeight,
    const agx::Vec3& p0, const agx::Vec3& p1,
    agx::StackArray<agx::Vec3, 2>& pOut,
    const agx::Real epsilon = agx::RealEpsilon );

  /**
  * Computes the closest point on a circle to a 3d point.
  * The circle is around the origin in the xz-plane.
  * Only points on the circle will be found, not on its interior (the disk).
  * \param circleRadius The circle's radius.
  * \param point The 3d point
  * \retval The closest point on the circle.
  */
  agx::Vec3 AGXPHYSICS_EXPORT closestPointCirclePoint3D(
    agx::Real circleRadius, const agx::Vec3& point );

  /**
  * Computes the closest points on a circle to a 3d line segment.
  * The circle is around the origin in the xz-plane.
  * Only points on the circle will be found, not on its interior (the disk).
  * The line is defined as linePoint + lineDir * t.
  * In case several closest points are possible, one of the combinations will be returned.
  * \param circleRadius The circle's radius.
  * \param segmentP0 The line segment starting point.
  * \param segmentP1 The line segment ending point.
  * \param pointOnCircle The closest point on the circle.
  * \param pointOnSegment The closest point on the line segment.
  * \param lineT The line segment line parameter corresponding to the closest point.
  * \retval The closest point on the circle.
  */
  void AGXPHYSICS_EXPORT closestPointsCircleLineSegment3D(
    agx::Real circleRadius,
    const agx::Vec3& segmentP0, const agx::Vec3& segmentP1,
    agx::Vec3& pointOnCircle, agx::Vec3& pointOnSegment,
    agx::Real& lineT );

  /**
  * Clips a convex polygon (poly1) in a plane against a second one (poly2) in the same plane.
  * The plane is assumed to be the xy-plane, so that z-coordinates are ignored.
  * This method can be used for polygons whose sizes are not known at compile time.
  * The disadvantage is that dynamic allocation has to be used, which will most likely lead to
  * slower runtime than related methods which can used statically allocated data.
  * \param poly1 The convex polygon 1. Expects all sides to have positive lengths.
  *              The polygon should be given in counterclockwise order.
  * \param poly2 The convex polygon 2. Expects all sides to have positive lengths.
  *              The polygon should be given in counterclockwise order.
  * \param result The clipped result. Empty if no overlap. All sides will have positive lengths.
  *              The polygon will be given in counterclockwise order.
  * \param epsilon boundary around the polygon to counter floating point errors
  */
  template<size_t N1, size_t N2>
  void clipConvexPolygonAgainstConvexPolygon(
    const agx::StackArray<agx::Vec2, N1>& poly1,
    const agx::StackArray<agx::Vec2, N2>& poly2,
    agx::StackArray<agx::Vec2, N1+N2>& result,
    const agx::Real epsilon);


  /**
  * Clips a line segment in a plane against a polygon in the same plane.
  * The plane is assumed to be the xy-plane, so that z-coordinates are ignored.
  * \param segment The segment.
  * \param poly The convex polygon. Expects all sides to have positive lengths.
  *              The polygon should be given in counterclockwise order.
  * \param result The clipped result. Empty if no overlap.
  * \param epsilon boundary around the polygon to counter floating point errors
  */
  template<size_t N>
  void clipLineSegmentAgainstConvexPolygon(
    const agx::StackArray<agx::Vec2, 2>& segment,
    const agx::StackArray<agx::Vec2, N>& poly,
    agx::StackArray<agx::Vec2, 2>& result,
    const agx::Real epsilon);

    /**
  * Clips a line segment in a plane against a polygon in the same plane.
  * The plane is assumed to be the xy-plane, so that z-coordinates are ignored.
  * This method can be used for polygons whose sizes are not known at compile time.
  * The disadvantage is that dynamic allocation has to be used, which will most likely lead to
  * slower runtime than related methods which can used statically allocated data.
  * \param segment The segment.
  * \param poly The convex polygon. Expects all sides to have positive lengths.
  *              The polygon should be given in counterclockwise order.
  * \param result The clipped result. Empty if no overlap.
  * \param epsilon boundary around the polygon to counter floating point errors
  */
  void AGXPHYSICS_EXPORT clipLineSegmentAgainstConvexPolygon(
    const agx::Vec2Vector& segment,
    const agx::Vec2Vector& poly,
    agx::Vec2Vector& result,
    const agx::Real epsilon);

  /**
  Find the intersecting line between two planes
  \param n1 - normal of the first plane
  \param d1 - distance from the origin to the first plane along the first plane normal
  \param n2 - normal of the second plane
  \param d2 - distance from the origin to the second plane along the second plane normal
  \param point - resulting point on the intersecting line
  \param line direction - resulting direction of the intersecting line
  \returns false if the planes are parallel.
  */
  bool AGXPHYSICS_EXPORT intersectPlanePlane( const agx::Vec3 n1, const agx::Real& d1, const agx::Vec3& n2, const agx::Real& d2,
    agx::Vec3& point, agx::Vec3& lineDirection );


  /**
  * Clips a convex polygon (poly1) in a plane against a second one (poly2) in the same plane.
  * The plane is assumed to be the xy-plane, so that z-coordinates are ignored.
  * This method can be used for polygons whose sizes are not known at compile time.
  * The disadvantage is that dynamic allocation has to be used, which will most likely lead to
  * slower runtime than related methods which can used statically allocated data.
  * \param poly1 The convex polygon 1. Expects all sides to have positive lengths.
  *              The polygon should be given in counterclockwise order.
  * \param poly2 The convex polygon 2. Expects all sides to have positive lengths.
  *              The polygon should be given in counterclockwise order.
  * \param The clipped result. Empty if no overlap. All sides will have positive lengths.
  *              The polygon will be given in counterclockwise order.
  * \param epsilon boundary around the polygon to counter floating point errors
  */
  void AGXPHYSICS_EXPORT clipConvexPolygonAgainstConvexPolygon(
    const agx::Vec2Vector& poly1,
    const agx::Vec2Vector& poly2,
    agx::Vec2Vector& result,
    const agx::Real epsilon);

  /**
  * Intersection test for line segment and Mesh.
  * Finds earliest point of intersection.
  *\param lineP0  First point of line in mesh's coordinate system.
  *\param lineP1  Second point of line  in mesh's coordinate system.
  *\param mesh The Mesh.
  *\param lineT Intersection parameter: Result = lineP1 + t * (lineP2 - lineP1)
  *\param triangleIndex The index of the triangle realizing this contact.
  *@return True if the line intersects the Mesh.
  */
  AGXPHYSICS_EXPORT bool findIntersectionLineSegmentMesh(
    const agx::Vec3& lineP0,
    const agx::Vec3& lineP1,
    const agxCollide::Mesh* mesh,
    agx::Real& lineT,
    unsigned int& triangleIndex);

  /**
  Tests a box and a cylinder for overlap.
  Uses some of the most common separating axes directions.
  Not all cases are treated, so that false positive are
  possible (but no false negatives).
  Parameters have to be given in box's coordinate system.
  \param boxHe The box's half extents.
  \param cylinderRadius The cylinder's radius.
  \param cylinderHeight The cylinder's height.
  \param cylinderEnd0 The cylinder's axis' first end point (in box's coordinates).
  \param cylinderEnd1 The cylinder's axis' other end point (in box's coordinates).
  \retval Is a collision possible? (might give false positive).
  */
  AGXPHYSICS_EXPORT bool possibleOverlapBoxCylinder(
    const agx::Vec3& boxHe,
    agx::Real cylinderRadius,
    agx::Real cylinderHeight,
    const agx::Vec3& cylinderEnd0,
    const agx::Vec3& cylinderEnd1);


  /**
  Tests a box and a capsule for overlap.
  Uses some of the most common separating axes directions.
  Not all cases are treated, so that false positive are
  possible (but no false negatives).
  Parameters have to be given in box's coordinate system.
  \param boxHe The box's half extents.
  \param capsuleRadius The capsule's radius.
  \param capsuleEnd0 The capsule's axis' first end point (in box's coordinates).
  \param capsuleEnd1 The capsule's axis' other end point (in box's coordinates).
  \retval Is a collision possible? (might give false positive).
  */
  AGX_FORCE_INLINE bool possibleOverlapBoxCapsule(
    const agx::Vec3& boxHe,
    agx::Real capsRadius,
    const agx::Vec3& capsuleEnd0,
    const agx::Vec3& capsuleEnd1 );


  // Implementations

  template <size_t N>
  bool clipConvexPolygonAgainstSlab(
    const agx::StackArray<agx::Vec3, N>& cPolygon,
    const agx::Real slabDistance,
    const uint8_t clipDim,
    agx::StackArray<agx::Vec3, N + 2>& result,
    const agx::Real relativeEpsilon)
  {
    //
    // Sutherland-Hodgman polygon clipping.
    //

    agxAssert( clipDim < 3 );

    // The algorithm works only well for polygons (3 vertices or more).
    // If the input is no polygon, but a point or a line, make it a degenerate
    // polygon.
    result.clear();
    agxAssert( cPolygon.size() >= 3 && cPolygon.size() <= N );
    if (cPolygon.size() == 0)
      return false;

    for (unsigned int i = 0; i < cPolygon.size(); ++i)
      result.push_back(cPolygon[i]);

    while(result.size() < 3)
      result.push_back(cPolygon[0]);

    // clip cPolygon1 against each infinite edge of box
    for (int sign = -1; sign <=1; sign+=2) {
      agx::StackArray<agx::Vec3, N + 2> tmp;
      bool lastPointWasOutside = (result[0][clipDim] * sign > slabDistance + relativeEpsilon);
      // test each edge of the cPolygon1
      for (unsigned int j = 0; j < result.size(); ++j) {
        int j_next = (j + 1) % (unsigned int)result.size();
        bool nextPointIsOutside =
          (result[j_next][clipDim] * sign > slabDistance + relativeEpsilon);
        if (lastPointWasOutside != nextPointIsOutside) {
          // crossing line of edge, clip
          agx::Vec3 newPoint;
          if (clipSegmentLineSemi1D( result[j], result[j_next], clipDim,
            slabDistance * sign, newPoint, relativeEpsilon )) {
            tmp.push_back(newPoint);
          }
        }
        if (!nextPointIsOutside)
          tmp.push_back(result[j_next]);
        lastPointWasOutside = nextPointIsOutside;
      }

      result.clear();

      if (tmp.size() == 0)
        return false;

      // do not use almost identical points
      for (size_t j = 0; j < tmp.size(); ++j) {
        bool addPoint = true;
        if (result.size() > 0) {
          if ((result.back() - tmp[j]).length2() < relativeEpsilon)
            addPoint = false;
        }
        if (addPoint)
          result.push_back(tmp[j]);
      }
      // maintain polygon property by creating a degenerate polygon instead of a
      // point / line
      while(result.size() > 0 && result.size() < 3)
        result.push_back( result[0] );
    }

    return true;
  }


  template <size_t N>
  bool clipConvexPolygonAgainstBox(
    const agx::StackArray<agx::Vec3, N>& cPolygon,
    const agx::Vec3& boxHe,
    agx::StackArray<agx::Vec3, N + 6>& tmp,
    agx::StackArray<agx::Vec3, N + 6>& result,
    const agx::Real relativeEpsilon )
  {
    //
    // Sutherland-Hodgman polygon clipping.
    //
    // The algorithm works only well for polygons (3 vertices or more).
    // If the input is no polygon, but a point or a line, make it a degenerate
    // polygon.
    result.clear();
    agxAssert( cPolygon.size() >= 3 && cPolygon.size() <= N );
    if (cPolygon.size() == 0)
      return false;

    for (size_t i = 0; i < cPolygon.size(); ++i)
      result.push_back(cPolygon[i]);

    while (result.size() < 3)
      result.push_back(cPolygon[0]);

    // clip cPolygon1 against each infinite edge of box
    for (agx::UInt8 dim = 0; dim < 3; dim++) {
      agx::Real sign(1);
      for (int i = 0; i < 2; ++i) {
        sign *= agx::Real(-1);
        tmp.clear();
        bool lastPointWasOutside = (result[0][dim] * sign > boxHe[dim] + relativeEpsilon);
        // test each edge of the cPolygon1
        for (unsigned int j = 0; j < result.size(); ++j) {
          const int j_next = (int)((j + 1) % result.size());
          bool nextPointIsOutside = (result[j_next][dim] * sign > boxHe[dim] + relativeEpsilon);
          if (lastPointWasOutside != nextPointIsOutside) {
            // crossing line of edge, clip
            agx::Vec3 newPoint;
            if (clipSegmentLineSemi1D( result[j], result[j_next], dim, boxHe[dim]  * sign, newPoint, relativeEpsilon ))
              tmp.push_back(newPoint);
          }
          if (!nextPointIsOutside)
            tmp.push_back(result[j_next]);
          lastPointWasOutside = nextPointIsOutside;
        }

        result.clear();

        if (tmp.size() == 0)
          return false;

        // do not use almost identical points
        for (size_t j = 0; j < tmp.size(); ++j) {
          if (result.size() > 0 && (result.back() - tmp[j]).length2() < relativeEpsilon)
            continue;
          result.push_back(tmp[j]);
        }
        // maintain polygon property by creating a degenerate polygon instead of a
        // point / line
        while (result.size() > 0 && result.size() < 3)
          result.push_back( result[0] );
      }
    }
    return true;
  }


  template<size_t N>
  void clipLineSegmentAgainstConvexPolygon(
    const agx::StackArray<agx::Vec2, 2>& segment,
    const agx::StackArray<agx::Vec2, N>& poly,
    agx::StackArray<agx::Vec2, 2>& result,
    const agx::Real epsilon)
  {
    result.clear();
    if (segment.size() < 2 || poly.size() < 3)
      return;

    for (int i = 0; i < 2; ++i)
      result.push_back( segment[i] );

    // create plane normals
    agx::StackArray<agx::Vec2, N> planeNormals; // these planeNormals are not normalized. Pointing outside from poly2.
    agx::StackArray<agx::Real, N> planeDists;
    for (int i = 0; i < int(poly.size()) - 1; ++i) {
      agx::Vec2 tmp = (poly[(i + 1)] - poly[i]);
      planeNormals.push_back( agx::Vec2(tmp.y(), -tmp.x()) ); // cross product with (0, 0, 1)
      planeDists.push_back( poly[i] * planeNormals[i] );
    }
    // last element
    {
      agx::Vec2 tmp = poly[0] - poly.back();
      planeNormals.push_back( agx::Vec2(tmp.y(), -tmp.x()) ); // cross product with (0, 0, 1)
      planeDists.push_back( poly.back() * planeNormals.back() );
    }

    // clip convex polygon against each plane defined by polygon edges and normal
    for (unsigned int i = 0; i < poly.size(); ++i) {
      bool isInside[2];
      isInside[0] = result[0] * planeNormals[i] - planeDists[i] < epsilon;
      isInside[1] = result[1] * planeNormals[i] - planeDists[i] < epsilon;

      if (isInside[0] == isInside[1]) {
        if (!isInside[0]) {
          result.clear();
          return;
        }
      }
      else {
        agx::Vec2 secondPoint;
        int first = isInside[0] ? 0 : 1;
        int second = 1 - first;
        agx::Real tmpT;
        intersectLineSegmentHyperPlane( result[first], result[second], planeNormals[i],
          planeDists[i], tmpT, secondPoint, epsilon ); // ignore true/false, since we've had that analysis above

        result[second] = secondPoint;
      }
    }

    if (equivalent( result[0], result[1], epsilon ))
      result.pop_back();
  }


  template<size_t N1, size_t N2>
  void clipConvexPolygonAgainstConvexPolygon(
    const agx::StackArray<agx::Vec2, N1>& poly1,
    const agx::StackArray<agx::Vec2, N2>& poly2,
    agx::StackArray<agx::Vec2, N1+N2>& result,
    const agx::Real epsilon)
  {
    //
    // Modified Sutherland-Hodgman polygon clipping in 3D.
    //
    result.clear();

    if (poly1.size() < 2 || poly2.size() < 2)
      return;

    if (poly1.size() == 2) {
      agx::StackArray<agx::Vec2, 2> segment, segmentResult;
      segment.push_back(poly1[0]);
      segment.push_back(poly1[1]);
      clipLineSegmentAgainstConvexPolygon(segment, poly2, segmentResult, epsilon);
      for (size_t i = 0; i < segmentResult.size(); ++i)
        result.push_back(segmentResult[i]);
      return;
    }

    if (poly2.size() == 2) {
      agx::StackArray<agx::Vec2, 2> segment, segmentResult;
      segment.push_back(poly2[0]);
      segment.push_back(poly2[1]);
      clipLineSegmentAgainstConvexPolygon(segment, poly1, segmentResult, epsilon);
      for (size_t i = 0; i < segmentResult.size(); ++i)
        result.push_back(segmentResult[i]);
      return;
    }

    for (size_t i = 0; i < poly1.size(); ++i)
      result.push_back(poly1[i]);

    // create plane normals
    agx::StackArray<agx::Vec2, N2> planeNormals; // Pointing outside from poly2.
    agx::StackArray<agx::Real, N2> planeDists;
    for (size_t i = 0; i < poly2.size(); ++i) {
      size_t next = (i+1) % poly2.size();
      agx::Vec2 tmp = (poly2[next] - poly2[i]);
      planeNormals.push_back( agx::Vec2(tmp.y(), -tmp.x()) ); // cross product with (0, 0, 1)
      planeNormals.back().normalize();
      planeDists.push_back( poly2[i] * planeNormals[i] );
    }

    // clip convex polygon against each plane defined by triangle edges and normal
    for (size_t i = 0; i < poly2.size(); ++i) {
      if (planeNormals[i].x() * planeNormals[i].x() + planeNormals[i].y() * planeNormals[i].y() < epsilon)
        continue;

      agx::StackArray<agx::Vec2, N1 + N2> tmpPoints;
      bool lastPointWasOutside = result[0] * planeNormals[i] - planeDists[i] > epsilon;

      // test each edge of the polygon
      for (size_t j = 0; j < result.size(); ++j) {
        int j_next = int((j + 1) % result.size());
        bool nextPointIsOutside = result[j_next] * planeNormals[i] - planeDists[i] > epsilon;
        if (lastPointWasOutside != nextPointIsOutside) {
          // crossing plane of triangle, clip
          agx::Vec2 newPoint;
          agx::Real tmpT;
          if (intersectLineSegmentHyperPlane( result[j], result[j_next], planeNormals[i], planeDists[i], tmpT, newPoint, epsilon )) {
            tmpPoints.push_back(newPoint);
          }
        }
        if (!nextPointIsOutside) {
          tmpPoints.push_back(result[j_next]);
        }
        lastPointWasOutside = nextPointIsOutside;
      }
      result.clear();
      if (tmpPoints.size() == 0)
        return;

      for (size_t j = 0; j < tmpPoints.size(); ++j) {
        bool addPoint = true;
        if (result.size() > 0) {
          if ((result.back() - tmpPoints[j]).length2() < epsilon)
            addPoint = false; // do not keep nearly identical points
          if ((int)j == int(tmpPoints.size()) - 1) { // also check last against first
            if ((result[0] - tmpPoints[j]).length2() < epsilon)
              addPoint = false;
          }
        }
        if (addPoint)
          result.push_back(tmpPoints[j]);
      }
    }
    return;
  }


  AGX_FORCE_INLINE void closestPointsSegmentSegment(
    const agx::Vec3& start1,
    const agx::Vec3& end1,
    const agx::Vec3& start2,
    const agx::Vec3& end2,
    ClosestPointsSolution& solution,
    bool& isParallel,
    ClosestPointsSolution& parallelSolution,
    const agx::Real epsilon/* = agx::AGX_EQUIVALENT_EPSILON*/
    )
  {
    closestPointsSegmentSegment(start1, end1, start2, end2,
      solution.pointOn1, solution.pointOn2, solution.t1, solution.t2,
      isParallel,
      parallelSolution.pointOn1, parallelSolution.pointOn2, parallelSolution.t1, parallelSolution.t2, epsilon);
  }


  AGX_FORCE_INLINE bool clipSegmentLineSemi1D(
    const agx::Vec3& a,
    const agx::Vec3& b,
    int dim,
    agx::Real clipBoundary,
    agx::Vec3& result,
    const agx::Real relativeEpsilon)
  {

    agxAssert (dim >= 0 && dim  < 3);

    agx::Real divisor = b[dim] - a[dim];

    if (agx::equalsZero( divisor, relativeEpsilon )) {
      // The two lines are parallel.
      result = b;
      return true;
    }
    else {
      agx::Real t = (clipBoundary - a[dim]) / divisor;
      result = a * (1 - t) + b * t;
      return (t >= 0 && t <= 1);
    }
  }


  template <typename T>
  AGX_FORCE_INLINE bool intersectLineHyperPlane(
    const T& linePoint,
    const T& lineDir,
    const T& normal,
    agx::Real d,
    agx::Real& t,
    T& result,
    const agx::Real epsilon )
  {
    // Used segment-plane intersection from Ericson, Real Time Collision Detection, p. 175f.
    agx::Real divisor = normal * lineDir;

    if (agx::equalsZero( divisor, epsilon )) {
      // Segment coplanar to plane. Return second point if contained in plane.
      result = linePoint + lineDir;
      t = agx::Real(1);
      return agx::equalsZero( normal * result - d, epsilon );
    }

    t = (d - normal * linePoint) / divisor;

    result = linePoint + lineDir * t;

    return true;
  }


  template <typename T>
  AGX_FORCE_INLINE bool intersectLineSegmentHyperPlane(
    const T& lineP1,
    const T& lineP2,
    const T& normal,
    agx::Real d,
    agx::Real& t,
    T& result,
    const agx::Real epsilon )
  {
    if (intersectLineHyperPlane(lineP1, lineP2 - lineP1, normal, d, t, result, epsilon)) {
      const bool isInInterval = (t > -epsilon && t < agx::Real(1) + epsilon);
      return isInInterval;
    }
    else
      return false;
  }


  AGX_FORCE_INLINE bool areDirectionsParallel(const agx::Vec3& dir0, const agx::Vec3& dir1,
    const agx::Real cosEpsilon)
  {
    const agx::Real cosAngle = dir0 * dir1;
    bool areParallel = (std::abs(cosAngle) >= cosEpsilon);
    return areParallel;
  }


  AGX_FORCE_INLINE bool areDirectionsOrthogonal(const agx::Vec3& dir0, const agx::Vec3& dir1,
    const agx::Real sinEpsilon)
  {
    const agx::Real cosAngle = dir0 * dir1;
    const agx::Real cosAngleSquared = cosAngle * cosAngle;
    const agx::Real sinAngleSquared = agx::Real(1) - cosAngleSquared;
    bool areOrthogonal = (sinAngleSquared >= sinEpsilon * sinEpsilon);
    return areOrthogonal;
  }


  AGX_FORCE_INLINE agx::Vec3 closestPointPointSegment(
    const agx::Vec3& p,
    const agx::Vec3& s0,
    const agx::Vec3& s1,
    agx::Real& t)
  {
    agx::Vec3 sDir = s1 - s0;
    agx::Real divisor = sDir.length2();
    if (agx::equalsZero( divisor )) {
      t = 0;
      return s0;
    }

    t = ((p - s0) * sDir) / divisor;
    t = agx::clamp( t, agx::Real(0), agx::Real(1) );
    const agx::Vec3 point = s0 * (1 - t) + s1 * t;
    return point;
  }


  AGX_FORCE_INLINE bool possibleOverlapBoxCapsule(
    const agx::Vec3& boxHe,
    agx::Real capsRadius,
    const agx::Vec3& capsuleEnd0,
    const agx::Vec3& capsuleEnd1 )
  {
    // Approach: Use separating axis theorem.
    // We have 4 axes: The three from the box (local x, y and z),
    // and the one from closest point between local origin and capsule axis.

    const agx::Vec3 capsMid = (capsuleEnd0 + capsuleEnd1) * agx::Real(0.5);
    const agx::Vec3 capsAbsMid = agx::absolute(capsMid);

    const agx::Vec3 capsHalfExt = capsuleEnd1 - capsMid;
    const agx::Vec3 capsAbsHalfExt = agx::absolute(capsHalfExt);

    // The three box axes: Trying to reach caps mid point by he, radius and capsHe.
    for (size_t i = 0; i < 3; ++i) {
      if (capsAbsMid[i] > boxHe[i] + capsRadius + capsAbsHalfExt[i])
        return false;
    }

    // The fourth axis: find closest point to box origin.
    agx::Real t;
    agx::Vec3 closest = closestPointPointSegment(agx::Vec3(), capsuleEnd0, capsuleEnd1, t);

    agx::Vec3 closestAbsDir = agx::absolute(closest);
    agx::Real closestDist = closestAbsDir.normalize();

    // Try to reach point with radius and projection
    if (closestDist > capsRadius + closestAbsDir * boxHe )
      return false;

    return true;
  }


  /**
  Intersect a mesh with a plane, yielding a connected curve in the plane.
  If the intersected mesh would yield more than one connected curve in the plane, the result is undefined -
  a connected curve will be returned, but it is not defined which one.
  \param trimesh The mesh.
  \param plane The plane, given in the mesh's local coordinates.
  \param planeToWorld The transformation from the plane's local coordinate system to the world.
  (For the mesh, none is needed, since it is given from mesh->getTransform().)
  \retval The points in counterclockwise winding (given the plane's orientation). Empty if an error occurred.
  */
  AGXPHYSICS_EXPORT agx::Vec3Vector intersectTrimeshWithPlane(
    const agxCollide::Mesh* trimesh,
    const agx::Plane& plane,
    const agx::AffineMatrix4x4& planeToWorld);
}

DOXYGEN_END_INTERNAL_BLOCK()
#endif
