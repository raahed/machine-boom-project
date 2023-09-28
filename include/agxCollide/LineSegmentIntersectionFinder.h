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

#ifndef AGXCOLLIDE_LINESEGMENTINTERSECTIONFINDER_H
#define AGXCOLLIDE_LINESEGMENTINTERSECTIONFINDER_H


#include <agxCollide/agxCollide.h>
#include <agx/StackArray.h>
#include <agx/agxPhysics_export.h>
#include <agx/Interval.h>
#include <agx/Vec3.h>


namespace agxCollide
{

  namespace agxGeometryQueries {

    const agx::Real geometricEpsilon = agx::Real(1.0e-6);

    /**
    Tests if two points are identical given agxGeometryQueries::geometricEpsilon.
    \param point0 The first point.
    \param point1 The second point.
    \retval Are the two points are identical given agxGeometryQueries::geometricEpsilon?
    */
    bool AGXPHYSICS_EXPORT pointsAreIdentical(const agx::Vec3& point0, const agx::Vec3& point1);

    class AGXPHYSICS_EXPORT LineSegment;


    /// An infinite line, defined by a starting point and a (not necessarily normalized) direction.
    class Line {
    public:
      /// Note that the direction does not have to be normalized.
      Line(const agx::Vec3& point, const agx::Vec3& dir);

      /// Computes the scalar value of the projection of a point along the line.
      agx::Real projectPoint(const agx::Vec3& point) const;

      /// Does the line contain the point, given an epsilon threshold?
      bool containsPoint(const agx::Vec3& point, const agx::Real epsilon = agx::RealEpsilon) const;

      /// Is the line degenerated to a point?
      bool isDegeneratedToPoint() const;

      /// Computes the interval of the projection of a line segment onto this line.
      agx::Interval projectLineSegment(const LineSegment& segment) const;

      /// Gets the starting point of the line.
      const agx::Vec3& point() const;

      /// Note that the line dir does not have to be normalized.
      agx::Vec3 evaluateAt(const agx::Real lineParameter) const;

      /// Gets the direction; not necessarily normalized.
      const agx::Vec3& dir() const;

      /// Gets the squared length of the line direction.
      agx::Real dirLength2() const;

    private:
      agx::Vec3 m_point;
      agx::Vec3 m_dir;
      agx::Real m_dirLength2;
    };

    /// A line segment. Note that the direction does not have to be normalized.
    class AGXPHYSICS_EXPORT LineSegment {
    public:
      /// Creates a line segment given two end points.
      LineSegment(const agx::Vec3& point0, const agx::Vec3& point1);

      /// Creates an empty line segment, with both end points at origin.
      LineSegment();

      /// Computes the scalar value of the projection of a point along the segment's line.
      agx::Real projectPointToInfiniteLine(const agx::Vec3& point) const;

      /// Does the line segment contain the point, given an epsilon threshold?
      bool containsPoint(const agx::Vec3& point, const agx::Real epsilon = agx::RealEpsilon) const;

      /// Is the line segment degenerated to a point?
      bool isDegeneratedToPoint() const;

      /// Gets the line segment's line.
      agxGeometryQueries::Line getLine() const;

      /// Gets the line segment's first point.
      const agx::Vec3& point0() const;

      /// Gets the line segment's second point.
      const agx::Vec3& point1() const;

      /**
      Evaluates the line equation at the parameter.
      Note that no sanity check on the parameter is being done, only values between 0 and 1 make sense.
      \param lineParameter The line parameter (should be in [0,1]).
      \retval The point at the line segment given the parameter.
      */
      agx::Vec3 evaluateAt(const agx::Real lineParameter) const;

      /// Gets the direction; not necessarily normalized.
      const agx::Vec3& dir() const;

      /// Gets the squared length of the line direction.
      agx::Real dirLength2() const;

    private:
      agx::Vec3 m_point0;
      agx::Vec3 m_point1;
      agx::Vec3 m_dir;
      agx::Real m_dirLength2;
    };


    /**
    A line segment that is swept in time from t=0 to t=1.
    All sweeping is done by linear interpolation.
    */
    class AGXPHYSICS_EXPORT SweptLineSegment
    {
      public:
        /**
        Creates a SweptLineSegment, with both line segments degenaring to points at the origin.
        */
        SweptLineSegment();

        /**
        The SweptLineSegment constructor.
        \param start The segment at t=0.
        \param end The segment at t=1.
        */
        SweptLineSegment(const LineSegment& start, const LineSegment& end);

        /**
        Evaluates the swept line at a given time and line parameter and returns a point.
        \param time The time between 0 and 1.
        \param lineParameter The line parameter between 0 and 1.
        \retval The evaluated point.
        */
        agx::Vec3 evaluateAt(const agx::Real time, const agx::Real lineParameter) const;

        /**
        Evaluates the swept line at a given time.
        \param time The time between 0 and 1.
        \retval The evaluated line segment.
        */
        LineSegment evaluateAt(const agx::Real time) const;

        /// Gets the swept line segment's first line segment, i.e. at t=0.
        LineSegment& segment0();

        /// Gets the swept line segment's first line segment, i.e. at t=0.
        const LineSegment& segment0() const;

        /// Gets the swept line segment's second line segment, i.e. at t=1.
        LineSegment& segment1();

        /// Gets the swept line segment's second line segment, i.e. at t=1.
        const LineSegment& segment1() const;

        /// Is the swept line segment degenerated to a point?
        bool isDegeneratedToPoint() const;

        /// Is the swept line segment degenerated to a line?
        bool isDegeneratedToLine() const;

        /// Returns the non-normalized sweep direction of point0 over time.
        agx::Vec3 computeSweepDirectionPoint0() const;

        /// Returns the non-normalized sweep direction of point1 over time.
        agx::Vec3 computeSweepDirectionPoint1() const;

        /// Returns the line segment point0 over time.
        LineSegment computeSweepSegmentPoint0() const;

        /// Returns the line segment point1 over time.
        LineSegment computeSweepSegmentPoint1() const;

      private:
        LineSegment m_segment0;
        LineSegment m_segment1;
    };
  }


  class AGXPHYSICS_EXPORT LineSegmentIntersectionFinder
  {
    public:
      /**
      \param sweptSegment0 The first line segment (swept over time).
      \param sweptSegment1 The second line segment (swept over time).
      \param t Output value - found overlap time.
      \param lineParameters0 Output value - interval along line parameters for segment0 which overlaps with segment1 at time t.
      \param lineParameters1 Output value - interval along line parameters for segment1 which overlaps with segment0 at time t.
      \param points Output value - The contact points from the given time t, transferred to time 1.
      \param normal Output value - The contact normal from the given time t, transferred to time 1.
      \param distance Output value - The distance that the points from lineParameter0 and lineParameter1 moved from each other between times t and 1.
      \retval Was a collision found? All output value are only valid if method returns true.
      */
      static bool calculateLineSegmentCollision(
        const agxGeometryQueries::SweptLineSegment& sweptSegment0,
        const agxGeometryQueries::SweptLineSegment& sweptSegment1,
        agx::Real& t,
        agx::Interval& lineParameters0,
        agx::Interval& lineParameters1,
        agx::StackArray<agx::Vec3, 2>& points,
        agx::Vec3& normal,
        agx::Real& distance
        );
  };




/// Implementations


  namespace agxGeometryQueries {
    // Functions.
    AGX_FORCE_INLINE bool pointsAreIdentical(const agx::Vec3& point0, const agx::Vec3& point1)
    {
      return (point0 - point1).length2() <= agxGeometryQueries::geometricEpsilon * agxGeometryQueries::geometricEpsilon;
    }


    // Class Line.
    AGX_FORCE_INLINE agxGeometryQueries::Line::Line(const agx::Vec3& point, const agx::Vec3& dir) : m_point(point), m_dir(dir), m_dirLength2(dir*dir)
    {
    }


    AGX_FORCE_INLINE agx::Real agxGeometryQueries::Line::projectPoint(const agx::Vec3& point) const
    {
      if (isDegeneratedToPoint())
        return 0;
      else
        return (m_dir * (point - m_point)) / m_dirLength2;
    }


    AGX_FORCE_INLINE bool agxGeometryQueries::Line::containsPoint(const agx::Vec3& point, const agx::Real /*epsilon*/) const
    {
      const agx::Vec3 projection = m_point + m_dir * projectPoint(point);
      return pointsAreIdentical(projection, point);
    }


    AGX_FORCE_INLINE bool agxGeometryQueries::Line::isDegeneratedToPoint() const
    {
      return agx::equalsZero(m_dirLength2, agx::RealEpsilon * agx::RealEpsilon);
    }


    AGX_FORCE_INLINE agx::Interval agxGeometryQueries::Line::projectLineSegment(const LineSegment& segment) const
    {
      return agx::Interval(projectPoint(segment.point0()), projectPoint(segment.point1()), false);
    }


    AGX_FORCE_INLINE const agx::Vec3& agxGeometryQueries::Line::point() const
    {
      return m_point;
    }


    AGX_FORCE_INLINE agx::Vec3 agxGeometryQueries::Line::evaluateAt(const agx::Real lineParameter) const
    {
      return m_point + m_dir * lineParameter;
    }


    AGX_FORCE_INLINE const agx::Vec3& agxGeometryQueries::Line::dir() const
    {
      return m_dir;
    }


    AGX_FORCE_INLINE agx::Real agxGeometryQueries::Line::dirLength2() const
    {
      return m_dirLength2;
    }


    // Class LineSegment.
    AGX_FORCE_INLINE agxGeometryQueries::LineSegment::LineSegment(const agx::Vec3& point0, const agx::Vec3& point1) : m_point0(point0), m_point1(point1),
    m_dir(point1 - point0), m_dirLength2(m_dir * m_dir)
    {
    }


    AGX_FORCE_INLINE agxGeometryQueries::LineSegment::LineSegment() : m_dirLength2(0)
    {
    }


    AGX_FORCE_INLINE agx::Real agxGeometryQueries::LineSegment::projectPointToInfiniteLine(const agx::Vec3& point) const
    {
      if (isDegeneratedToPoint())
        return 0;
      else
        return (m_dir * (point - m_point0)) / m_dirLength2;
    }


    AGX_FORCE_INLINE bool agxGeometryQueries::LineSegment::containsPoint(const agx::Vec3& point, const agx::Real epsilon) const
    {
      const agx::Real projection = projectPointToInfiniteLine(point);
      if (projection < 0 || projection * projection > m_dirLength2)
        return false;
      const agx::Vec3 projectionPoint = m_point0 + m_dir * projection;
      const agx::Vec3 dist = point - projectionPoint;
      return dist.length2() < epsilon * epsilon;
    }


    AGX_FORCE_INLINE bool agxGeometryQueries::LineSegment::isDegeneratedToPoint() const
    {
      return agx::equalsZero(m_dirLength2, agx::RealEpsilon * agx::RealEpsilon);
    }


    AGX_FORCE_INLINE agxGeometryQueries::Line agxGeometryQueries::LineSegment::getLine() const
    {
      return agxGeometryQueries::Line(point0(), dir());
    }


    AGX_FORCE_INLINE const agx::Vec3& agxGeometryQueries::LineSegment::point0() const
    {
      return m_point0;
    }


    AGX_FORCE_INLINE const agx::Vec3& agxGeometryQueries::LineSegment::point1() const
    {
      return m_point1;
    }


    AGX_FORCE_INLINE const agx::Vec3& agxGeometryQueries::LineSegment::dir() const
    {
      return m_dir;
    }


    AGX_FORCE_INLINE agx::Real agxGeometryQueries::LineSegment::dirLength2() const
    {
      return m_dirLength2;
    }


    AGX_FORCE_INLINE agx::Vec3 agxGeometryQueries::LineSegment::evaluateAt(const agx::Real lineParameter) const
    {
      return m_point0 + m_dir * lineParameter;
    }


    // Class SweptLineSegment.
    AGX_FORCE_INLINE agxGeometryQueries::SweptLineSegment::SweptLineSegment()
    {
    }

    AGX_FORCE_INLINE agxGeometryQueries::SweptLineSegment::SweptLineSegment(const LineSegment& start, const LineSegment& end)
    :m_segment0(start), m_segment1(end)
    {
    }


    AGX_FORCE_INLINE agx::Vec3 agxGeometryQueries::SweptLineSegment::evaluateAt(const agx::Real time, const agx::Real lineParameter) const
    {
      const agx::Vec3 point = m_segment0.evaluateAt(lineParameter) * (agx::Real(1.0) - time) + m_segment1.evaluateAt(lineParameter) * time;
      return point;
    }


    AGX_FORCE_INLINE agxGeometryQueries::LineSegment agxGeometryQueries::SweptLineSegment::evaluateAt(const agx::Real time) const
    {
      const LineSegment segment(m_segment0.point0() * (agx::Real(1.0) - time) + m_segment1.point0() * time,
        m_segment0.point1() * (agx::Real(1.0) - time) + m_segment1.point1() * time);
      return segment;

    }


    AGX_FORCE_INLINE LineSegment& SweptLineSegment::segment0()
    {
      return m_segment0;
    }


    AGX_FORCE_INLINE const LineSegment& SweptLineSegment::segment0() const
    {
      return m_segment0;
    }


    AGX_FORCE_INLINE LineSegment& SweptLineSegment::segment1()
    {
      return m_segment1;
    }


    AGX_FORCE_INLINE const LineSegment& SweptLineSegment::segment1() const
    {
      return m_segment1;
    }


    AGX_FORCE_INLINE bool SweptLineSegment::isDegeneratedToPoint() const
    {
      return m_segment0.isDegeneratedToPoint() && m_segment1.isDegeneratedToPoint() && agx::equivalent(m_segment0.point0(), m_segment1.point0());

    }


    AGX_FORCE_INLINE bool SweptLineSegment::isDegeneratedToLine() const
    {
      return m_segment0.isDegeneratedToPoint() && m_segment1.isDegeneratedToPoint();
    }


    AGX_FORCE_INLINE agx::Vec3 SweptLineSegment::computeSweepDirectionPoint0() const
    {
      return m_segment1.point0() - m_segment0.point0();
    }


    AGX_FORCE_INLINE agx::Vec3 SweptLineSegment::computeSweepDirectionPoint1() const
    {
      return m_segment1.point1() - m_segment0.point1();
    }


    AGX_FORCE_INLINE LineSegment SweptLineSegment::computeSweepSegmentPoint0() const
    {
      return LineSegment(m_segment0.point0(), m_segment1.point0());
    }


    AGX_FORCE_INLINE LineSegment SweptLineSegment::computeSweepSegmentPoint1() const
    {
      return LineSegment(m_segment0.point1(), m_segment1.point1());
    }


  }
}

#endif // AGXCOLLIDE_LINESEGMENTINTERSECTIONFINDER_H
