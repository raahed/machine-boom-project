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


// Do not parse this file if we are running it through swig for the java bindings
#if (defined(__cplusplus) && !defined(SWIG)) || (defined(SWIG) && !defined(SWIGJAVA))

#include <agx/agx.h>
#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agxStream/Serializable.h>
#include <agx/Vec3.h>
#include <agx/AffineMatrix4x4.h>

namespace agxUtil {

  /**
  Base class for splines.
  */
  class AGXPHYSICS_EXPORT Spline : public agx::Referenced, public virtual agxStream::Serializable
  {
    /**
    Tangent
    */
    public:
      class Tangent : public agx::Vec3
      {
        public:
          Tangent()
            : agx::Vec3() {}
          Tangent( const agx::Vec3& v )
            : agx::Vec3( v ) {}
      };

      /**
      Point class for storing 3D point, tension, stretch and curvature
      */
      class Point : public agx::Vec3
      {
        public:
          AGXPHYSICS_EXPORT static const double DEFAULT_TENSION;
          AGXPHYSICS_EXPORT static const agx::Real DEFAULT_STRETCH;
          AGXPHYSICS_EXPORT static const agx::Real DEFAULT_CURVATURE;

          /**
          Create a spline point from a point \p p, tension \p t and stretch \p stretch
          \param p - point in 3D space
          \param t - tension at point
          \param stretch - stretch at point
          \param curvature - curvature at point
          */
          Point(const agx::Vec3& p=agx::Vec3(), agx::Real t= DEFAULT_TENSION, agx::Real stretch=DEFAULT_STRETCH, agx::Real curvature=DEFAULT_CURVATURE)
            : agx::Vec3(p), m_tension(t), m_stretch(stretch), m_curvature(curvature) {}

          /// \return the tension at the point
          inline agx::Real getTension() const { return m_tension; }

          /// Set the tension (N) in the point
          inline void setTension(agx::Real tension) { m_tension = tension; }

          /// \return the stretch of the point
          inline agx::Real getStretch() const { return m_stretch; }

          /// \return the curvature of the point
          inline agx::Real getCurvature() const { return m_curvature; }

        private:
          agx::Real m_tension;
          agx::Real m_stretch;
          agx::Real m_curvature;
      };

      typedef agx::Vector< Tangent > TangentVector;
      typedef agx::Vector< Point > PointVector;

    public:

      /// Default constructor
      Spline();

      /**
      Add point to the control points given tension (1 max, 0 min) and stretch.
      \param point - point to be added
      \param tension - tension at the specified point (clamped between 0 and 1)
      \param stretch - specified stretch at the point
      \param curvature - Specified curvature at the point
      */
      void add(const agx::Vec3& point, agx::Real tension = Point::DEFAULT_TENSION, agx::Real stretch = Point::DEFAULT_STRETCH, agx::Real curvature = Point::DEFAULT_CURVATURE);

      /**
      Interpolate from point with index \p index a time t forward (0 <= t <= 1)
      \return Interpolated point, at a specified segment, at a specified time t.
      */
      virtual Point evaluate( size_t index, agx::Real t ) const = 0;

      /**
      \return number of points added
      */
      inline size_t getNumPoints() const { return m_points.size(); }

      /**
      \return vector containing the points added (class Spline::Point)
      */
      inline PointVector& getPoints() { return m_points; }

#ifndef SWIG
      /**
      \return vector containing the points added (class Spline::Point)
      */
      inline const PointVector& getPoints() const { return m_points; }
#endif

      /**
      \return the ith tangent computed (updated after updateTangents call, class Spline::Tangent)
      */
      virtual Tangent getTangent(size_t /*index*/, agx::Real /*t*/) const { return Tangent(); }

      /**
      Linear interpolate of tension from point at index.
      \return interpolated tension at a given segment, at a specified time
      */
      virtual inline agx::Real getTension(size_t index, agx::Real time) const;

      /**
      All control points are added, computes the tangents.
      */
      virtual void updateTangents() = 0;

      /**
      Clears control points and tangents.
      */
      virtual void clear();

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxUtil::Spline);

      void store(agxStream::OutputArchive& out) const override;
      void restore(agxStream::InputArchive& in) override;

    protected:
      virtual ~Spline() {}

    protected:
      PointVector m_points;
      TangentVector m_tangents;
  };

  typedef agx::ref_ptr< Spline > SplineRef;


  inline agx::Real Spline::getTension(size_t index, agx::Real time) const
  {
    if (index + 1 == m_points.size())
      return m_points.back().getTension();

    time = agx::clamp(time, agx::Real(0), agx::Real(1));
    if (agx::equalsZero(time))
      return m_points[index].getTension();
    else if (agx::equivalent(time, agx::Real(1), agx::RealEpsilon))
      return m_points[index + 1].getTension();
    else
      return (1 - time) * m_points[index].getTension() + time * m_points[index + 1].getTension();
  }


  /**
  A Spline constructed of piecewise third-order polynomials which pass through a set of control points.
  https://en.wikipedia.org/wiki/Cubic_Hermite_spline
  */
  class AGXPHYSICS_EXPORT CubicSpline : public Spline
  {
    public:

      // Default constructor
      CubicSpline();

      /**
      \return point from \p index at time t
      */
      virtual Point evaluate( size_t index, agx::Real t ) const override;

      /**
      Compute tangents given current set of points.
      */
      void updateTangents() override;

      Tangent getTangent(size_t index, agx::Real t) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxUtil::CubicSpline);


    protected:
      virtual ~CubicSpline() {}

    private:
      agx::AffineMatrix4x4 m_matrix;
  };

  typedef agx::ref_ptr< CubicSpline > CubicSplineRef;

  /**
  A Spline constructed of piecewise third-order polynomials which pass through a set of control points.
  https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Cardinal_spline
  */
  class AGXPHYSICS_EXPORT CardinalSpline : public Spline
  {
    public:
      CardinalSpline();

      /**
      \return point from \p index at time t
      */
      virtual Point evaluate( size_t index, agx::Real t ) const override;

      /**
      Compute tangents given current set of points.
      */
      void updateTangents() override;

      Tangent getTangent(size_t index, agx::Real t) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxUtil::CardinalSpline);


    protected:
      virtual ~CardinalSpline() {}
  };

  typedef agx::ref_ptr< CardinalSpline > CardinalSplineRef;

  /**
  A Spline constructed of piecewise third-order polynomials which pass through a set of control points.
  https://codeplea.com/introduction-to-splines
  */
  class AGXPHYSICS_EXPORT NonUniformCardinalSpline : public Spline
  {
    public:
      NonUniformCardinalSpline();

      /**
      \return point from \p index at time t
      */
      virtual Point evaluate(size_t index, agx::Real t) const override;

      /**
      Compute tangents given current set of points.
      */
      void updateTangents() override;

      Tangent getTangent(size_t index, agx::Real t) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxUtil::NonUniformCardinalSpline);


    protected:
      virtual ~NonUniformCardinalSpline() {}
  };

  typedef agx::ref_ptr< NonUniformCardinalSpline > NonUniformCardinalSplineRef;

  /**
  Special case of a cardinal spline. Assumes uniform parameter spacing
  https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Catmull%E2%80%93Rom_spline
  */
  class AGXPHYSICS_EXPORT ParameterizedCatmullRomSpline : public Spline
  {
    public:
      ParameterizedCatmullRomSpline(agx::Real alpha);

      /**
      \return point from \p index at time t
      */
      virtual Point evaluate(size_t index, agx::Real t) const override;

      /**
      Compute tangents given current set of points.
      */
      void updateTangents() override;

      Tangent getTangent(size_t index, agx::Real t) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxUtil::ParameterizedCatmullRomSpline);


    protected:
      virtual ~ParameterizedCatmullRomSpline() {}

      /// Used only for restoring purposes
      ParameterizedCatmullRomSpline();

    private:
      agx::Real m_alpha;
  };

  typedef agx::ref_ptr< ParameterizedCatmullRomSpline > ParameterizedCatmullRomSplineRef;

  /**
  https://en.wikipedia.org/wiki/B-spline
  */
  class AGXPHYSICS_EXPORT BSpline : public Spline
  {
    public:
      BSpline();

      /**
      \return point from \p index at time t
      */
      virtual Point evaluate(size_t index, agx::Real t) const override;

      /**
      Compute tangents given current set of points.
      */
      void updateTangents() override;

      Tangent getTangent(size_t index, agx::Real t) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxUtil::BSpline);


    protected:
      virtual ~BSpline() {}
  };

  typedef agx::ref_ptr< BSpline > BSplineRef;

  /**
  https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline
  */
  class AGXPHYSICS_EXPORT CPCatmullRomBSpline : public Spline
  {
    public:
      CPCatmullRomBSpline();

      /**
      \return point from \p index at time t
      */
      virtual Point evaluate(size_t index, agx::Real t) const override;

      /**
      Compute tangents given current set of points.
      */
      virtual void updateTangents() override;

      Tangent getTangent(size_t index, agx::Real t) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxUtil::CPCatmullRomBSpline);


    protected:
      virtual ~CPCatmullRomBSpline() {}
      ParameterizedCatmullRomSplineRef   m_catmull;
      BSplineRef                         m_bspline;
  };

  typedef agx::ref_ptr< CPCatmullRomBSpline > CPCatmullRomBSplineRef;

  /**
  https://en.wikipedia.org/wiki/Hermite_spline
  */
  class AGXPHYSICS_EXPORT HermiteSpline : public Spline
  {
    public:
      HermiteSpline();

      /**
      \return point from \p index at time t
      */
      virtual Point evaluate(size_t index, agx::Real t) const override;

      /**
      Compute tangents given current set of points.
      */
      void updateTangents() override;

      Tangent getTangent(size_t index, agx::Real t) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxUtil::HermiteSpline);


    protected:
      virtual ~HermiteSpline() {}
      ParameterizedCatmullRomSplineRef   m_catmull;
      BSplineRef                         m_bspline;
  };

  typedef agx::ref_ptr< HermiteSpline > HermiteSplineRef;
}
#endif // SWIGJAVA
