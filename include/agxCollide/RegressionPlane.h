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

#ifndef AGX_REGRESSIONPLANE_H
#define AGX_REGRESSIONPLANE_H

#include <agx/agx_vector_types.h>
#include <agx/Referenced.h>

#include <agx/agxPhysics_export.h>
#include <agx/Plane.h>

#include <agxData/LocalVector.h>

namespace agx {
  class SPDMatrix3x3;
}

namespace agxCollide {

  /**
  A plane created from points using least squares minimization.
  If no valid plane can be created, regression to a line or point
  is attempted.
  */
  class AGXPHYSICS_EXPORT RegressionPlane
  {

    public:
      /// Dimension type of the sub-space resulted by the regression.
      enum DimensionType {
        EMPTY,
        POINT,
        LINE,
        PLANE
      };

      /**
      Creates a RegressionPlane from a number of 3D-points.
      If there a no points in the Vector, the RegressionPlane's
        DimensionType will be EMPTY.
      If all points are (roughly) identical, the RegressionPlane's
         DimensionType will be POINT.
      If all points lie (roughly) in a line with finite positive length,
       the RegressionPlane's DimensionType will be LINE.
      In all other cases, a plane can be constructed from the points, and
       the RegressionPlane's DimensionType will be PLANE.
      \param points A pointer to a vector of points. The RegressionPlane
             does not take over ownership of or store references to the pointer.
      */
      RegressionPlane(const agxData::LocalVector<agx::Vec3>& points);
      RegressionPlane(const agx::Vec3Vector& points);

      /**
      Creates a RegressionPlane from the coefficients precomputed from a number
      of 3D points.
      \param linearCoefficients sum of derivations of all points from
            averagePoint (x, y, z)
      \param quadraticCoefficients sum of component wise product
             of derivation of each point from
             averagePoint (xx, xy, xz, yx, yy, yz, zx, zy, zz)
      \param averagePoint Mean value of all points.
      \param numPoints The number of points.
      */
      RegressionPlane ( const agx::Vec3& linearCoefficients,
        const agx::SPDMatrix3x3& quadraticCoefficients,
        const agx::Vec3& averagePoint,
        const size_t numPoints);

      /// Gets the dimension type of the plane.
      DimensionType getDimensionType() const;

      /**
      Gets the average point of the plane.
      \note Only valid if dimension type is POINT, LINE or PLANE.
      */
      agx::Vec3 getAveragePoint() const;

      /**
      Gets the average point of the plane.
      \note Only valid if dimension type is LINE or PLANE.
      */
      agx::Vec3 getLineDirection() const;

      /**
      Gets the average point of the plane.
      \note Only valid if dimension type is PLANE.
      */
      agx::Plane getPlane() const;

      /// Is the regression plane a plane (and not empty, point or line)?
      bool isPlane() const;

      /// Returns the maximum determinant from the least squares minimization.
      agx::Real getMaxDeterminant() const;

  private:
      RegressionPlane();


      void computeRegressionPlane (
        agx::Vec3 linearCoefficients,
        agx::SPDMatrix3x3 quadraticCoefficients,
        size_t numPoints);

    private:
      agx::Plane m_plane;
      agx::Vec3 m_averagePoint;
      agx::Vec3 m_lineDirection;
      DimensionType m_dimensionType;

      agx::Real m_maxDeterminant;
  };



  // implementations
  AGX_FORCE_INLINE RegressionPlane::DimensionType RegressionPlane::getDimensionType() const
  {
    return m_dimensionType;
  }

  AGX_FORCE_INLINE agx::Real RegressionPlane::getMaxDeterminant() const
  {
    return m_maxDeterminant;
  }


  AGX_FORCE_INLINE agx::Vec3 RegressionPlane::getAveragePoint() const
  {
    agxAssert ( m_dimensionType != RegressionPlane::EMPTY );
    return m_averagePoint;
  }


  AGX_FORCE_INLINE agx::Vec3 RegressionPlane::getLineDirection() const
  {
    agxAssert ( m_dimensionType > RegressionPlane::POINT );
    return m_lineDirection;
  }


  AGX_FORCE_INLINE agx::Plane RegressionPlane::getPlane() const
  {
    agxAssert ( m_dimensionType == RegressionPlane::PLANE );
    return m_plane;
  }


  AGX_FORCE_INLINE bool RegressionPlane::isPlane() const
  {
    return m_dimensionType == PLANE;
  }
}


#endif
