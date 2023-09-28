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

#ifndef AGXCOLLIDE_LINE_H
#define AGXCOLLIDE_LINE_H

#include <agx/agxPhysics_export.h>
#include <agx/SPDMatrix3x3.h>
#include <agx/Physics/Geometry/LineEntity.h>
#include <agxCollide/Shape.h>
#include <agxStream/Serializable.h>

namespace agxCollide
{
  AGX_DECLARE_POINTER_TYPES(Line);
  /**
  A line shape for intersection tests, mostly for depth sensing and picking.
  The contacts with lines will therefore behave differently than all other shape contacts.
  \note Not to be used for modeling for general contacts, e.g. rigid body dynamics.

  Details about the contact behavior:

  The Line Shape is special in that its use is not in rigid body simulation,
  but in other fields as e.g. mouse picking or depth computation for unknown geometries.
  Thus, the computation of contact point, normal and depth differs from that of the other Shapes.
  The Line Shape represents mathematically a line segment, defined by
  two points \f$P_0\f$ and \f$P_1\f$, which define
  the normalized Line direction \f$d = \frac{P_1 - P_0}{||P_1 - P_0||}\f$
  The Line can thus be presented as
  \f$ P_0 + t d, 0 \le t \le ||P_1 - P_0|| \f$.

  If a Line does not intersect the other Shape, no contact is reported.

  If a Line intersects Shape, the first point in contact \f$P_c\f$
  along the infinite version of the line is reported as contact point.
  The surface normal of the Shape is reported as contact normal.
  The parameter \f$t_c\f$ is the value which realizes \f$P_c = P_0 + t_c d\f$.
  \f$1 - t_c\f$ is reported as length; this corresponds to how long
  the Line has to be pushed back in its direction in order to just touch the
  overlap point \f$P_c\f$.
  */
  class AGXPHYSICS_EXPORT Line : public Shape
  {
    public:
      /**
      Constructor. The endpoints of the line are in local coordinates.
      \param p1 First endpoint
      \param p2 Second endpoint - Has to be different than p1.
      */
      Line( const agx::Vec3 p1, const agx::Vec3 p2 );

      /**
      Volume of line-shape, will return 0.0 for line objects.
      \return agx::Real volume
      */
      virtual agx::Real getVolume() const override;

      /**
      Set first point (start point), in local coordinates.
      \param p First point. Has to be different than the second point.
      */
      void setFirstPoint( const agx::Vec3& p );

      /**
      Set second point (end point), in local coordinates.
      \param p Second point. Has to be different than the first point.
      */
      void setSecondPoint( const agx::Vec3& p );

      /**
      Set first and second point, in local coordinates.
      \param p1 First point
      \param p2 Second point - Has to be different than p1.
      */
      void set(const agx::Vec3& p1, const agx::Vec3& p2);

      /// Returns the first point, in local coordinates.
      agx::Vec3 getFirstPoint() const;

      /// Returns the second point, in local coordinates.
      agx::Vec3 getSecondPoint() const;

      /**
      Checks if line shape has support function, will return true for line object.
      \return true if it has a support function
      */
      virtual bool hasSupportFunction() const override;

      /**
      Gets support point.
      \param supportDirection
      \return support point
      */
      virtual agx::Vec3 getSupportPoint( const agx::Vec3& supportDirection ) const override;

      /**
      Calculate inertia for line shape, will return identity matrix for line object.
      \param mass Mass of shape
      \return inertia matrix
      */
      virtual agx::SPDMatrix3x3 calculateInertia(agx::Real mass) const override;

      /**
      Update the bounding volume for shape.
      \return Updated Axis Aligned Bounding Box
      */
      virtual const BoundingAABB& updateBoundingVolume() override;

      /**
      Calculate a local bounding volume for shape.
      \return Axis Aligned Bounding Box
      */
      virtual BoundingAABB calculateLocalBound() const override;

      /**
      Calculate a bounding volume for line shape from parameters.
      \param transform Transformation matrix
      \param firstPoint First point of line
      \param secondPoint Second point of line
      \return Axis Aligned Bounding Box
      */
      static BoundingAABB calculateBound(const agx::AffineMatrix4x4& transform, const agx::Vec3& firstPoint, const agx::Vec3& secondPoint);

      agx::Physics::Geometry::LinePtr getEntity() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxCollide::Line );

    protected:
      virtual ~Line( void );
      Line() : Shape(LINE, agx::Physics::Geometry::LineModel::createInstance()) {}
  };


  /* Implementation */
  inline agx::Physics::Geometry::LinePtr Line::getEntity() const
  {
    return m_entity;
  }

  AGX_FORCE_INLINE agx::Vec3 Line::getFirstPoint() const
  {
    return getEntity().line().p1;
  }

  AGX_FORCE_INLINE  agx::Vec3 Line::getSecondPoint() const
  {
    return getEntity().line().p2;
  }

  AGX_FORCE_INLINE BoundingAABB Line::calculateBound(const agx::AffineMatrix4x4& transform, const agx::Vec3& firstPoint, const agx::Vec3& secondPoint)
  {
    BoundingAABB bound;
    const agx::Vec3 p0 = firstPoint * transform;
    const agx::Vec3 p1 = secondPoint * transform;
    bound.min() = agx::Vec3::componentMin(p0, p1);
    bound.max() = agx::Vec3::componentMax(p0, p1);
    return bound;
  }
}

#endif
