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

#include <agx/agxPhysics_export.h>
#include <agx/Vec3.h>
#include <agx/AffineMatrix4x4.h>
#include <agxCollide/LocalContactPoint.h>
#include <agxCollide/ShapeCollider.h>
#include <agx/Plane.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxCollide {
  class ClosestPointsSolution;

  /**
  Functionality used by different ShapeColliders to find overlaps between different shapes.
  */
  namespace ShapeColliders {

    /**
    Computes contact between two spheres.
    \param position1: The first sphere's position in the world coordinate system.
    \param radius1: The first sphere's radius.
    \param position2: The second sphere's position in the world coordinate system.
    \param radius2: The second sphere's radius.
    \param result: The result (0 to 1 contact point will be added).
    */
    void AGXPHYSICS_EXPORT collideSphereSphere(
      const agx::Vec3& position1,
      const agx::Real radius1,
      const agx::Vec3& position2,
      const agx::Real radius2,
      agxCollide::ShapeCollider::LocalContactPointVector& result);


    /**
    Computes contact between a sphere and a cylinder.
    \param spherePosition: The sphere's position in the world coordinate system.
    \param sphereRadius: The sphere's radius.
    \param cylinderTransform: The cylinder's transformation to the world coordinate system.
    \param cylinderRadius: The cylinder's radius.
    \param cylinderHeight: The cylinder's height.
    \param normalFlipSign: The flip factor for the normal.
           Should be 1.0 (no flip) or -1.0 (flip).
    \param result: The result (0 to 1 contact point will be added).
    */
    void AGXPHYSICS_EXPORT collideSphereCylinder(
      const agx::Vec3& spherePosition,
      const agx::Real sphereRadius,
      const agx::AffineMatrix4x4& cylinderTransform,
      const agx::Real cylinderRadius,
      const agx::Real cylinderHeight,
      const agx::Real normalFlipSign,
      agxCollide::ShapeCollider::LocalContactPointVector& result );


    /**
    Computes contact between a capsule and a plane.
    \param capsuleTransform: The capsule's transformation to the world coordinate system.
    \param capsuleRadius: The capsule's radius.
    \param capsuleHeight: The capsule's height.
    \param transformedPlane: The plane, transformed to the world coordinate system.
    \param normalFlipSign: The flip factor for the normal.
           Should be 1.0 (no flip) or -1.0 (flip).
    \param result: The result (0 to 2 contact point will be added).
    */
    void AGXPHYSICS_EXPORT collideCapsulePlane(
      const agx::AffineMatrix4x4& capsuleTransform,
      const agx::Real capsuleRadius,
      const agx::Real capsuleHeight,
      const agx::Plane& transformedPlane,
      const agx::Real normalFlipSign,
      agxCollide::ShapeCollider::LocalContactPointVector& result );


    /**
    Computes contact between a sphere and a plane.
    \param spherePosition: The sphere's position in the world coordinate system.
    \param sphereRadius: The sphere's radius.
    \param transformedPlane: The plane, transformed to the world coordinate system.
    \param result: The result (0 to 1 contact point will be added).
    */
    void AGXPHYSICS_EXPORT collideSpherePlane(
      const agx::Vec3& spherePosition,
      const agx::Real sphereRadius,
      const agx::Plane& transformedPlane,
      agxCollide::ShapeCollider::LocalContactPointVector& result);


    /**
    Computes contact between a box and a plane.
    \param boxTransform: The box's transformation to the world coordinate system.
    \param boxHalfExtents: The box's half extents.
    \param transformedPlane: The plane, transformed to the world coordinate system.
    \param result: The result (0 to 4 contact point will be added).
    */
    void AGXPHYSICS_EXPORT collideBoxPlane(
      const agx::AffineMatrix4x4& boxTransform,
      const agx::Vec3& boxHalfExtents,
      const agx::Plane& transformedPlane,
      agxCollide::ShapeCollider::LocalContactPointVector& result );


    /**
    Computes contact between a capsule and a plane.
    \param cylinderTransform: The cylinder's transformation to the world coordinate system.
    \param cylinderRadius: The cylinder's radius.
    \param cylinderHeight: The cylinder's height.
    \param transformedPlane: The plane, transformed to the world coordinate system.
    \param normalFlipSign: The flip factor for the normal.
           Should be 1.0 (no flip) or -1.0 (flip).
    \param result: The result (0 to 2 contact point will be added).
    */
    void AGXPHYSICS_EXPORT collideCylinderPlane(
      const agx::AffineMatrix4x4& cylinderTransform,
      const agx::Real cylinderRadius,
      const agx::Real cylinderHeight,
      const agx::Plane& transformedPlane,
      agxCollide::ShapeCollider::LocalContactPointVector& result );


    /**
    Creates support points along mantles of two cylinders (capsules, ...).
    \param closest Result of closest point query.
    \param cyl1Radius Radius of cylinder 1.
    \param cyl1Height Height of cylinder 1.
    \param cyl1World1 World position of point 1 of axis of cylinder 1.
    \param cyl1World2 World position of point 2 of axis of cylinder 1.
    \param transform1 Transformation from cylinder 1's local frame to world.
    \param cyl1Radius Radius of cylinder 2.
    \param cyl1Height Height of cylinder 2.
    \param cyl1World1 World position of point 1 of axis of cylinder 2.
    \param cyl1World2 World position of point 2 of axis of cylinder 2.
    \param transform1 Transformation from cylinder 2's local frame to world.
    \param normal The normal of the original, first contact point
           (pointing from cylinder 2 to cylinder 1).
    \param result Here, 0, 1 or 2 contact points will be added.
    \param createOnlyForCyl1InCyl2 Should only the contact point
           for cylinder 1 in cylinder 2 be created?
    */
    void AGXPHYSICS_EXPORT addSupportPointsForCylindersAlongMantle(
      const ClosestPointsSolution& closest,
      const agx::Real cyl1Radius,
      const agx::Real cyl1Height,
      const agx::Vec3& cyl1World1,
      const agx::Vec3& cyl1World2,
      const agx::AffineMatrix4x4& transform1,
      const agx::Real cyl2Radius,
      const agx::Real cyl2Height,
      const agx::Vec3& cyl2World1,
      const agx::Vec3& cyl2World2,
      const agx::AffineMatrix4x4& transform2,
      const agx::Vec3& normal,
      agxCollide::ShapeCollider::LocalContactPointVector& result,
      bool createOnlyForCyl1InCyl2 = false);

    /**
    Computes contacts between two cylinders (or cylindrical shapes such as HollowCylinder)
    Therefore we cannot assume anything about the ShapeInstances.

    \param radius1: Radius of the first cylinder
    \param radius2: Radius of the second cylinder
    \param length1: Length of the first cylinder
    \param length2: Length of the second cylinder
    \param shape1:  Shape of the first "cylinder" (can be a HollowCylinder)
    \param shape2:  Shape of the second "cylinder" (can be a HollowCylinder)
    \param linearVelocity1: Linear velocity of the first shape
    \param linearVelocity2: Linear velocity of the second shape
    \param angularVelocity1: Angular velocity of the first shape
    \param angularVelocity2: Angular velocity of the second shape
    \param earlyOut: Not being used.
    \param result: The result (added contact points)
    */
    AGXPHYSICS_EXPORT void collideCylinderCylinder(
      agx::Real radius1,
      agx::Real radius2,
      agx::Real height1,
      agx::Real height2,
      agx::Physics::Geometry::ShapeInstance shape1,
      agx::Physics::Geometry::ShapeInstance shape2,
      const agx::AffineMatrix4x4& transform1,
      const agx::AffineMatrix4x4& transform2,
      const agx::Vec3& linearVelocity1,
      const agx::Vec3& linearVelocity2,
      const agx::Vec3& angularVelocity1,
      const agx::Vec3& angularVelocity2,
      bool earlyOut,
      ShapeCollider::LocalContactPointVector& result);

  }



  // Implementations

  AGX_FORCE_INLINE void ShapeColliders::collideCapsulePlane(
    const agx::AffineMatrix4x4& capsuleTransform,
    const agx::Real capsuleRadius,
    const agx::Real capsuleHeight,
    const agx::Plane& transformedPlane,
    const agx::Real normalFlipSign,
    agxCollide::ShapeCollider::LocalContactPointVector& result )

  {
    // World coordinates for capsule points.
    const agx::Vec3 cap1 = agx::Vec3(0, capsuleHeight * agx::Real(0.5), 0);
    const agx::Vec3 cap2 = agx::Vec3(0, -capsuleHeight * agx::Real(0.5), 0);

    const agx::Vec3 p1 = cap1 * capsuleTransform;
    const agx::Vec3 p2 = cap2 * capsuleTransform;

    const agx::Vec3 normal = transformedPlane.getNormal();

    // Get the distance from the point to the plane.
    // If negative or distance < the radius of the sphere, a collision has occurred.
    const agx::Real d1 = transformedPlane.signedDistanceToPoint(p1);
    if (d1 <= capsuleRadius) {
      // The contact point is the capsule endpoint projected onto the plane.
      const agx::Real depth = capsuleRadius - d1;
      const agx::Vec3 point = p1 - normal * (d1 + depth * agx::Real(0.5));
      ShapeCollider::addContactPoint( result, point, normal * normalFlipSign, depth );
    }

    const agx::Real d2 = transformedPlane.signedDistanceToPoint(p2);
    if (d2 <= capsuleRadius) {
      // The contact point is the capsule endpoint projected onto the plane.
      const agx::Real depth = capsuleRadius - d2;
      const agx::Vec3 point = p2 - normal * (d2 + depth * agx::Real(0.5));
      ShapeCollider::addContactPoint( result, point, normal * normalFlipSign, depth );
    }
  }



  AGX_FORCE_INLINE void ShapeColliders::collideSphereSphere(
    const agx::Vec3& position1,
    const agx::Real radius1,
    const agx::Vec3& position2,
    const agx::Real radius2,
    agxCollide::ShapeCollider::LocalContactPointVector& result)
  {
    const agx::Real radiusSum = radius1 + radius2;
    agx::Vec3 normal = position1 - position2;

    const agx::Real len2 = normal.length2();
    if (radiusSum * radiusSum < len2) {
      // Mathematically no contact.
      // Make sure we don't lose contacts due to floating point precision.
      agx::Real epsilon = agx::RealEpsilon * (agx::Real(1.0) + radius1 + radius2);
      if ((radiusSum + epsilon) * (radiusSum + epsilon) < len2)
        return; // No contact.
      else {
        // Contact which might have been missed due to floating point precision.
        // We keep it (conservative on collision), but give it depth 0.
        normal.normalize();
        ShapeCollider::addContactPoint( result,
          position2 + normal * radius2, normal, agx::Real(0) );
      }
    }
    else {
      agx::Real distance;
      // Calculate the normalized contact normal pointing
      // from sphere2 towards sphere1.
      if ( len2 == 0 ) {
        // The sphere centers are at the exact same position.
        distance = agx::Real( 0 );
        normal[2] = 1;
      }
      else
        distance = normal.normalize();

      const agx::Real depth = radiusSum - distance;
      if (depth >= agx::Real(0)) {
        ShapeCollider::addContactPoint( result,
          position2 + normal * (radius2 - depth * agx::Real(0.5)),
          normal, depth );
      }
    }
  }


  AGX_FORCE_INLINE void ShapeColliders::collideSpherePlane(
    const agx::Vec3& spherePosition,
    const agx::Real sphereRadius,
    const agx::Plane& transformedPlane,
    agxCollide::ShapeCollider::LocalContactPointVector& result)
  {
    const agx::Real distance = transformedPlane.signedDistanceToPoint(spherePosition);
    if (distance > sphereRadius)
      return;
    const agx::Vec3 normal = transformedPlane.getNormal();
    const agx::Real depth = sphereRadius - distance;
    ShapeCollider::addContactPoint( result, spherePosition +
      normal * (depth * agx::Real(0.5) - sphereRadius), normal, depth );
  }
}

DOXYGEN_END_INTERNAL_BLOCK()


