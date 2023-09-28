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

#include <agxUtil/PointCurve.h>

#include <agx/agx_vector_types.h>
#include <agx/Plane.h>

namespace agxUtil
{
  /**
  Utility class to create a convex hull in a plane given point cloud.
  */
  class AGXPHYSICS_EXPORT ConvexHull2D
  {
    public:
      /**
      Point in hull containing the point projected onto the plane
      and the original point.
      */
      struct Point
      {
        agx::Vec3 projected;  /**< Hull point projected onto the given plane. */
        agx::Vec3 original;   /**< Original point. */
        explicit operator const agx::Vec3&() const { return projected; }
      };

    public:
      using HullContainer  = agx::VectorPOD<Point>;
      using const_iterator = HullContainer::const_iterator;
      using Curve          = PointCurve<Point>;
      using CurveSegment   = PointCurve<Point>::Segment;

    public:
      /**
      Default constructor.
      */
      ConvexHull2D();

      /**
      Construct given point cloud and a plane. The hull will be calculated during
      construction using the Jarvis march algorithm.
      \param points - original point cloud
      \param plane - plane to project point cloud onto
      */
      ConvexHull2D( const agx::Vec3Vector& points, const agx::Plane& plane );

      /**
      Construct given point cloud, plane and number of sub-quadrangles. The hull
      will be calculated during construction using the Quadrangles algorithm.
      \param points - original point cloud
      \param plane - plane to project point cloud onto
      \param numSubQuadrangles - number of sub quadrangles defining the accuracy of this algorithm (3 is very low
                                 and 100 is large and probably accurate)
      */
      ConvexHull2D( const agx::Vec3Vector& points, const agx::Plane& plane, agx::UInt numSubQuadrangles );

      /**
      Access hull points of given index. No bounds checks make sure \p index < size().
      \param index - hull point index
      \return point with index \p index in the hull
      */
      const Point& operator[]( agx::UInt index ) const;

      /**
      \return first element of the hull assuming the hull isn't empty
      */
      const Point& front() const;

      /**
      \return last element of the hull assuming the hull isn't empty
      */
      const Point& back() const;

      /**
      \return number of points in the hull
      */
      agx::UInt size() const;

      /**
      \return begin iterator of this hull
      */
      const_iterator begin() const;

      /**
      \return begin iterator of this hull
      */
      const_iterator end() const;

      /**
      Computes the convex hull in the given plane using 2D gift wrapping algorithm - Jarvis march.
      \param points - original point cloud
      \param plane - plane to project point cloud onto
      \return hull points if successful - otherwise an empty array
      */
      const HullContainer& computeHullJarvisMarch( const agx::Vec3Vector& points, const agx::Plane& plane );

      /**
      Computes the convex hull in the given plane using a set of quadrangles. This approach can
      be much faster than the Jarvis march but points could be missed if \p numSubQuadrangles is too low.
      The lower number of sub-quadrangles the quicker this method will find a hull.
      \param points - original point cloud
      \param plane - plane to project point cloud onto
      \param numSubQuadrangles - number of sub quadrangles defining the accuracy of this algorithm (3 is very low
                                 and 100 is large and probably accurate)
      \return hull points if successful - otherwise an empty array
      */
      const HullContainer& computeHullQuadrangles( const agx::Vec3Vector& points, const agx::Plane& plane, agx::UInt numSubQuadrangles );

      /**
      Calculates the area of the hull - if the hull is present. If the hull is empty, i.e., number of points < 2, the area is 0.
      \return area of the hull in the plane
      */
      agx::Real calculateArea() const;

      /**
      Computes the perimeter of the hull - if the hull is present. If the hull is empty, i.e., number of points < 2, the
      perimeter is 0.
      \return the perimeter of the hull in the plane
      */
      agx::Real calculatePerimeter() const;

      /**
      \return the hull (empty array if not created)
      */
      const HullContainer& getHull() const;

      /**
      Current hull as a curve object.
      \param closed - true if the curve should be closed, i.e., first point == last point
      \return current hull points as a curve object
      */
      Curve asCurve( agx::Bool closed = true ) const;

    private:
      HullContainer m_hull;
  };

  inline const ConvexHull2D::Point& ConvexHull2D::operator[]( agx::UInt index ) const
  {
    return m_hull[ index ];
  }

  inline const ConvexHull2D::Point& ConvexHull2D::front() const
  {
    return m_hull.front();
  }

  inline const ConvexHull2D::Point& ConvexHull2D::back() const
  {
    return m_hull.back();
  }

  inline agx::UInt ConvexHull2D::size() const
  {
    return m_hull.size();
  }

  inline ConvexHull2D::const_iterator ConvexHull2D::begin() const
  {
    return m_hull.begin();
  }

  inline ConvexHull2D::const_iterator ConvexHull2D::end() const
  {
    return m_hull.end();
  }
}
