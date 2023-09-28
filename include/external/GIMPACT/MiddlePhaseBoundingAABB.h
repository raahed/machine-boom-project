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

/*
This source code has been taken and modified by Algoryx Simulation AB
from the source and under the license given below.
*/

/*
This source file is part of GIMPACT Library.

For the latest info, see http://gimpact.sourceforge.net/

Copyright (c) 2007 Francisco Leon Najera. C.C. 80087371.
email: projectileman@yahoo.com


This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef AGXCOLLIDE_MIDDLE_PHASE_BOUNDING_AABB_H
#define AGXCOLLIDE_MIDDLE_PHASE_BOUNDING_AABB_H


#include <agx/agxPhysics_export.h>

#include <agx/Matrix3x3.h>
#include <agx/Vec3.h>
#include <agxCollide/BoundingAABB.h>
#include <agxCollide/BasicPrimitiveTests.h>

/// \cond INTERNAL_DOCUMENTATION


namespace agxCollide
{

  /**
  Class for caching transformation from one model space to the other, to use in all box-box tests in
  bounding volume tree traversal.
  */
  class BoxBoxTransformCache
  {

  public:
    /**
    Calculate transformation from one space to another
    \param trans0 - The transformation of the first object
    \param trans1 - The transformation of the second object, whose
          transform to the first is to be computed
    */
    AGX_FORCE_INLINE BoxBoxTransformCache( const agx::AffineMatrix4x4& trans0,
      const agx::AffineMatrix4x4& trans1 );

  public:
    agx::Matrix3x3 rotate1To0; //rotation of model1 to model 0
    agx::Matrix3x3 rotate0To1; //rotation of model0 to model 1
    agx::Matrix3x3 absoluteRotate1To0; //absolute value of rotation matrix 1 to 0
    agx::Matrix3x3 absoluteRotate0To1; //absolute value of rotation matrix 0 to 1
    agx::Vec3 translate1To0; //translation of model1 to model 0
    agx::Vec3 translate0To1; //translation of model0 to model 1
    int numParallels;

  private:
    BoxBoxTransformCache();
  };



  ///

  /**
  Representation of an AABB.
  Contains representation as well as methods to collide one AABB with another.
  This AABB is only axis-aligned if its rotation is the same as the world
  and because of that a transformation between the two objects are needed
  to compute collision.
  */
  class AGXPHYSICS_EXPORT MiddlePhaseBoundingAABB
  {

  public:
    /// Default constructor.
    MiddlePhaseBoundingAABB();

    /// Copy constructor.
    MiddlePhaseBoundingAABB( const BoundingAABB& boundingAABB );

    /**
    Constructs from center and half extents.
    \param setCenter The center.
    \param setHalfExtents The half extents.
    */
    MiddlePhaseBoundingAABB( const agx::Vec3& setCenter,
      const agx::Vec3& setHalfExtents );

    /**
    Fit an AABB around a triangle, also adding a small margin
    \param v1 - triangle vertex1
    \param v2 - triangle vertex2
    \param v3 - triangle vertex3
    \param margin - margin added to the AABB
    */
    void calculateFromTriangle( const agx::Vec3& v1,
      const agx::Vec3& v2, const agx::Vec3& v3, agx::Real margin );

    /**
    Merge this box with another
    \param box - other box
    */
    void merge( const MiddlePhaseBoundingAABB& box );

    /**
    Computes center and half extents from min and max values.
    \param min The min value.
    \param max The max value.
    */
    void setMinMax( const agx::Vec3& min, const agx::Vec3& max );


    /**
    Computes min and max values from center and half extents.
    \param min The min value. Return value.
    \param max The max value. Return value.
    */
    void getMinMax( agx::Vec3& min, agx::Vec3& max ) const;

    /**
    Collision test versus another (rotated) AABB.
    \param other - the other AABB
    \param transform - the transformation of the other box into this space
    \param fulltest - Specify whether to test the last 9 axes or not. Most
      collisions are rejected by testing the first 6 axes, and by not doing
      the full test time can be saved.
    \retval: if fulltest is true, then the return value answers if the boxes
            overlap. Otherwise only if all first 6 sat tests overlap (so they
             might still be separated even in case of a 'true' as answer).
    */
    bool collideBoxBox( const MiddlePhaseBoundingAABB& other,
      const BoxBoxTransformCache& transform, bool fulltest ) const;


    /**
    Collision test versus another (rotated) AABB.
    Optimized version which only does the first 6 SAT tests,
    and assumes the transformed half extents and center for the
    own box to be precomputed.
    \param other - the other AABB
    \param transform - the transformation of the other box into this space
    \param transformedHalfExtents0 The half extents of this box
           multiplied with the absolute rotation
           matrix to the other box's local coordinate system.
    \param transformedCenter0 The center of this box, transformed into
           the other box's coordinate system.
    \retval: if fulltest is true, then the return value answers if the boxes
            overlap. Otherwise only if all first 6 sat tests overlap (so they
             might still be separated even in case of a 'true' as answer).
    */
    bool collideBoxBox( const MiddlePhaseBoundingAABB& other,
      const BoxBoxTransformCache& transform,
      const agx::Vec3& transformedHalfExtents0,
      const agx::Vec3& transformedCenter0 ) const;

    /**
    Computes it there is overlap with a sphere.
    \param spereRadius The sphere's radius.
    \param sperePosition The sphere's center position.
    \retval True if any overlap, false otherwise.
    */
    bool collideBoxSphere( const agx::Real sphereRadius,
      const agx::Vec3& spherePosition ) const;

    /**
    Computes it there is overlap with a capsule.
    Does not do exact computation, but allows for false positives.
    \param capsulRadius The capsule's radius.
    \param capsuleEndPoint0 The capsule's axis' first end point.
    \param capsuleEndPoint1 The capsule's axis' other end point.
    \retval True if possible overlap, false otherwise.
    */
    bool collideBoxCapsule( agx::Real capsuleRadius,
      const agx::Vec3& capsuleEndPoint0,
      const agx::Vec3& capsuleEndPoint1 ) const;

    /**
    Computes it there is overlap with a cylinder.
    Does not do exact computation, but allows for false positives.
    \param capsulRadius The cylinder's radius.
    \param cylinderEndPoint0 The cylinder's axis' first end point.
    \param cylinderEndPoint1 The cylinder's axis' other end point.
    \retval True if possible overlap, false otherwise.
    */
    bool collideBoxCylinder(
      agx::Real cylinderRadius,
      agx::Real cylinderHeight,
      const agx::Vec3& cylinderEndPoint0,
      const agx::Vec3& cylinderEndPoint1 ) const;

    /// Returns box's center.
    const agx::Vec3& getCenter() const;

    /// Returns box's half extents.
    const agx::Vec3f& getHalfExtents() const;

  private:
    agx::Vec3 m_center;
    agx::Vec3f m_halfExtents;
  };


  // implementations:



  AGX_FORCE_INLINE float roundToLargerFloat( double d ) {
    union FloatUIntUnion {
      agx::Real32 f;
      agx::UInt32 u;
    };

    FloatUIntUnion h;
    h.f = (float)d;
    h.u =  ( h.u&0x80000000 ? ~h.u : h.u|0x80000000);
    h.u++;
    h.u = ( h.u&0x80000000 ? h.u&0x7fffffff : ~h.u );
    return h.f;
  }



  AGX_FORCE_INLINE BoxBoxTransformCache::BoxBoxTransformCache (
    const agx::AffineMatrix4x4& trans0,
    const agx::AffineMatrix4x4& trans1 )
  {
    numParallels = 0;
    agx::AffineMatrix4x4 transform1To0 = trans1 * trans0.inverse();
    agx::AffineMatrix4x4 transform0To1 = transform1To0.inverse();
    rotate1To0 = agx::Matrix3x3( transform1To0 );
    rotate0To1 = agx::Matrix3x3( transform0To1 );

    //Calculate rotation matrix to rotate from frame1 to frame0
    for (int i = 0; i < 3; i++){
      for (int j = 0; j < 3; j++) {
        absoluteRotate1To0( i, j ) = std::abs( rotate1To0( i, j ) );
        absoluteRotate0To1( i, j ) = std::abs( rotate0To1( i, j ) );
      }
      if (agx::equivalent( agx::Real( 1 ), std::abs( rotate1To0( i, i ) ), agx::Real( 0.001 ) )) {
        numParallels++;
      }
    }

    //Calculate translation vector to translate from frame1 to frame0
    translate1To0 = transform1To0.getTranslate();
    translate0To1 = transform0To1.getTranslate();
  }


  AGX_FORCE_INLINE const agx::Vec3& MiddlePhaseBoundingAABB::getCenter() const
  {
    return m_center;
  }


  AGX_FORCE_INLINE const agx::Vec3f& MiddlePhaseBoundingAABB::getHalfExtents() const
  {
    return m_halfExtents;
  }


  AGX_FORCE_INLINE MiddlePhaseBoundingAABB::MiddlePhaseBoundingAABB()
  {}


  AGX_FORCE_INLINE MiddlePhaseBoundingAABB::MiddlePhaseBoundingAABB(
    const agx::Vec3& setCenter, const agx::Vec3& setHalfExtents ) :
  m_center( setCenter ), m_halfExtents( setHalfExtents )
  {}


  AGX_FORCE_INLINE MiddlePhaseBoundingAABB::MiddlePhaseBoundingAABB(
    const BoundingAABB& boundingAABB )
  {
    setMinMax( boundingAABB.min(), boundingAABB.max() );
  }


  AGX_FORCE_INLINE void MiddlePhaseBoundingAABB::setMinMax( const agx::Vec3& min, const agx::Vec3& max)
  {
    m_center = (max + min) * agx::Real(0.5);
    m_halfExtents[0] = roundToLargerFloat( max[0] - m_center[0] );
    m_halfExtents[1] = roundToLargerFloat( max[1] - m_center[1] );
    m_halfExtents[2] = roundToLargerFloat( max[2] - m_center[2] );
  }



  AGX_FORCE_INLINE void MiddlePhaseBoundingAABB::getMinMax( agx::Vec3& min, agx::Vec3& max) const
  {
    agx::Vec3 he = agx::Vec3( getHalfExtents() );
    min = m_center - he;
    max = m_center + he;
  }


  AGX_FORCE_INLINE void MiddlePhaseBoundingAABB::calculateFromTriangle( const agx::Vec3& v1,
    const agx::Vec3& v2, const agx::Vec3& v3, agx::Real margin )
  {
    agx::Vec3 min = agx::Vec3::componentMin(v1, v2);
    min = agx::Vec3::componentMin( min, v3 );

    agx::Vec3 max = agx::Vec3::componentMax(v1, v2 );
    max = agx::Vec3::componentMax(max, v3 );

    min[0] -= margin;
    min[1] -= margin;
    min[2] -= margin;
    max[0] += margin;
    max[1] += margin;
    max[2] += margin;

    setMinMax( min, max );
  }


  AGX_FORCE_INLINE void MiddlePhaseBoundingAABB::merge( const MiddlePhaseBoundingAABB& box )
  {
    agx::Vec3 minNew, maxNew;
    box.getMinMax( minNew, maxNew );
    agx::Vec3 minThis, maxThis;
    getMinMax( minThis, maxThis );
    minNew = agx::Vec3::componentMin(minThis, minNew);
    maxNew = agx::Vec3::componentMax(maxThis, maxNew);

    setMinMax( minNew, maxNew );
  }


  AGX_FORCE_INLINE bool MiddlePhaseBoundingAABB::collideBoxSphere (const agx::Real sphereRadius,
    const agx::Vec3& spherePosition ) const
  {
    const agx::Vec3 sphereCenter = spherePosition - m_center;
    const agx::Vec3 diff = agx::absolute(sphereCenter) - agx::Vec3(m_halfExtents);
    if (diff[0] > 0 || diff[1] > 0 || diff[2] > 0) {
      agx::Vec3 pointOnBox = sphereCenter;
      agx::Vec3 he = agx::Vec3(m_halfExtents);
      pointOnBox.clamp(-he, he);
      const bool overlapsFromOutside = (sphereCenter - pointOnBox).length2() <= sphereRadius * sphereRadius;
      return overlapsFromOutside;
    }
    else
      return true;
  }



  AGX_FORCE_INLINE bool MiddlePhaseBoundingAABB::collideBoxBox( const MiddlePhaseBoundingAABB& other,
    const BoxBoxTransformCache& transform,
    const agx::Vec3& transformedHalfExtents0,
    const agx::Vec3& transformedCenter0 ) const
  {
    const agx::Real epsilon = (other.m_halfExtents[0] + other.m_halfExtents[1] + other.m_halfExtents[2] +
      m_halfExtents[0] + m_halfExtents[1] + m_halfExtents[2]) * 2 * agx::RealEpsilon;

    // First 3 axes: The other boxes' basis vectors
    if (std::abs( transformedCenter0[0] - other.getCenter()[0] ) > other.getHalfExtents()[0] + transformedHalfExtents0[0] + epsilon ||
      std::abs( transformedCenter0[1] - other.getCenter()[1] ) > other.getHalfExtents()[1] + transformedHalfExtents0[1] + epsilon ||
      std::abs( transformedCenter0[2] - other.getCenter()[2] ) > other.getHalfExtents()[2] + transformedHalfExtents0[2] + epsilon )
      return false;

    const agx::Vec3 transformedCenter1 = other.getCenter() * transform.rotate1To0 + transform.translate1To0;
    const agx::Vec3 transformedHalfExtents1 = agx::Vec3( other.getHalfExtents() ) * transform.absoluteRotate1To0;

    // Other 3 axes: This boxes' basis vectors
    if (std::abs( transformedCenter1[0] - m_center[0] ) > m_halfExtents[0] + transformedHalfExtents1[0] + epsilon ||
      std::abs( transformedCenter1[1] - m_center[1] ) > m_halfExtents[1] + transformedHalfExtents1[1] + epsilon ||
      std::abs( transformedCenter1[2] - m_center[2] ) > m_halfExtents[2] + transformedHalfExtents1[2] + epsilon )
      return false;

    return true;
  }


  AGX_FORCE_INLINE bool MiddlePhaseBoundingAABB::collideBoxCapsule( agx::Real capsuleRadius,
    const agx::Vec3& capsuleEndPoint0,
    const agx::Vec3& capsuleEndPoint1 ) const
  {
    const bool retVal = possibleOverlapBoxCapsule( agx::Vec3(m_halfExtents),
      capsuleRadius, capsuleEndPoint0 - m_center, capsuleEndPoint1 - m_center);
    return retVal;
  }


  AGX_FORCE_INLINE bool MiddlePhaseBoundingAABB::collideBoxCylinder(
    agx::Real cylinderRadius,
    agx::Real cylinderHeight,
    const agx::Vec3& cylinderEndPoint0,
    const agx::Vec3& cylinderEndPoint1 ) const
  {
    const bool retVal = possibleOverlapBoxCylinder(agx::Vec3(m_halfExtents),
      cylinderRadius, cylinderHeight,
      cylinderEndPoint0 - m_center, cylinderEndPoint1 - m_center);
    return retVal;
  }
}


#endif

/// \endcond
