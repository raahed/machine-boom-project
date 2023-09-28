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

#ifndef AGXCOLLIDE_PLANE_H
#define AGXCOLLIDE_PLANE_H

#include <agx/Vec4.h>
#include <agxCollide/Shape.h>
#include <agx/Physics/Geometry/PlaneEntity.h>

namespace agxCollide
{

  /**
  A plane shape for geometric intersection tests
  Default the normal is along Z-axis.
  Note: The distance d is defined in the plane equation ax+by+cz+d=0 for any point
  (x, y, z) in the plane and with the plane normal (a, b, c).
  */
  AGX_DECLARE_POINTER_TYPES(Plane);
  class AGXPHYSICS_EXPORT Plane : public Shape
  {
    public:

      /**
      Default constructor
      */
      Plane();

      /**
      Constructor, using plane equation ax+by+cz+d=0.
      */
      Plane( agx::Real const a, agx::Real const b, agx::Real const c, agx::Real d );

      /**
      Constructor
      \param normal plane normal (a,b,c) in plane equation
      \param d distance
      */
      Plane( const agx::Vec3 normal, agx::Real d );

      /**
      Constructor
      \param normal - Normal (a,b,c) in the plane equation
      \param point - A point on the plane
      */
      Plane( const agx::Vec3 normal, const agx::Vec3 point );

      Plane( const agx::Vec4 plane);

      /**
      Create a plane which is spanned using three distinct points.
      */
      Plane( const agx::Vec3 p1, const agx::Vec3 p2, const agx::Vec3 p3);

      void set( const agx::Vec4& plane );
      void set(const agx::Vec3& p1, const agx::Vec3& p2, const agx::Vec3& p3);

      virtual agx::Real getVolume() const override;

      agx::Vec4 asVec4() const;

      /// Get the normal (a, b, c) from plane equation ax+by+cz+d=0.
      const agx::Vec3 getNormal() const;

      /// Get the distance d (from plane equation ax+by+cz+d=0).
      agx::Real getDistance() const;

      /// Set the normal (a, b, c) from plane equation ax+by+cz+d=0.
      void setNormal( const agx::Vec3& normal );

      /// Set the distance d (from plane equation ax+by+cz+d=0).
      void setDistance( agx::Real distance );

      agx::Vec4 calculatePlane( const agx::AffineMatrix4x4& transform ) const;

      /**
      Calculates the distance from the plane to a point. Negative value if the point is behind the plane, otherwise positive.
      */
      agx::Real signedDistanceToPoint( const agx::Vec3& point ) const;

      virtual agx::SPDMatrix3x3 calculateInertia(agx::Real mass) const override;
      virtual const BoundingAABB& updateBoundingVolume() override;
      virtual BoundingAABB calculateLocalBound() const override;

      static BoundingAABB calculateBound(const agx::AffineMatrix4x4& transform, const agx::Plane& plane);

      /**
      Create a clone
      */
      virtual Shape *clone() const override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxCollide::Plane );

      agx::Physics::Geometry::PlanePtr getEntity() const;

  protected:
    Plane& operator=(const Plane&) { return *this; }
    virtual ~Plane();

  private:
    void init(const agx::Vec3& normal, agx::Real distance);
  };


  /* Implementation */
  inline agx::Physics::Geometry::PlanePtr Plane::getEntity() const { return m_entity; }

  AGX_FORCE_INLINE const agx::Vec3 Plane::getNormal() const { return getEntity().plane().getNormal(); }
  AGX_FORCE_INLINE agx::Real Plane::getDistance() const { return getEntity().plane().getDistance(); }

  AGX_FORCE_INLINE agx::Real Plane::signedDistanceToPoint( const agx::Vec3& point ) const
  {
    return ( getNormal() * point ) + getDistance();
  }


  AGX_FORCE_INLINE BoundingAABB Plane::calculateBound(const agx::AffineMatrix4x4& transform, const agx::Plane& /*plane*/ )
  {
    BoundingAABB localbound;

    localbound.max() = agx::Vec3( agx::RealMax - 1, agx::RealMax - 1, agx::RealMax - 1 );
    localbound.min() = agx::Vec3( -agx::RealMax + 1, -agx::RealMax + 1, -agx::RealMax + 1 );

    /*
    Disabled, until #1935 is fixed.

    // make bounding box smaller in case the normal is aligned to natural axis
    for (int i = 0; i < 3; ++i) {
      if (plane.getNormal()[i] == 1) {
        localbound.max[i] = -plane.getDistance();
        break;
      }
      else if (plane.getNormal()[i] == -1) {
        localbound.min[i] = -plane.getDistance();
        break;
      }
    }
    */

    return BoundingAABB(localbound, transform);
  }
}


#endif
