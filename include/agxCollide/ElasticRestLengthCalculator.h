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

#ifndef AGXCOLLIDE_ELASTIC_REST_LENGTH_CALCULATOR_H
#define AGXCOLLIDE_ELASTIC_REST_LENGTH_CALCULATOR_H

#include <agx/agxPhysics_export.h>
#include <agx/Vec3.h>


namespace agxCollide
{
  class Shape;
  class Sphere;
  class Box;
  class Capsule;
  class Cylinder;
  class Mesh;

  namespace ElasticRestLengthCalculator
  {
    /**
    Computes elastic rest length for this shape, point and normal.
    It is defined as the intersection length of a line (defined by the point
    and the normal) with the shape.
    Note that the intersection can be on the line's positive side, negative side, or both.
    \param shape The shape. Can have transformation. Should be a 'concrete' shape,
      not an agxCollide::Group.
    \param point A point in world coordinates. Used for anchoring the line.
      Assumed to have come from collision detection code.
    \param normal A normal in world coordinates. Expected to have length 1.
      Used for giving direction to the line. Assumed to have come from collision detection code.
    \param depth The depth of the contact point. Only used for approximative calculations in mesh case.
    \retval The elastic rest length, defined as the intersection length between line and shape.
    */
    agx::Real AGXPHYSICS_EXPORT calculateElasticRestLength(
      const agxCollide::Shape* shape,
      const agx::Vec3& point,
      const agx::Vec3& normal,
      const agx::Real depth);

    /**
    Computes elastic rest length for this shape, point and normal.
    It is defined as the intersection length of a line (defined by the point
    and the normal) with the shape.
    Note that the intersection can be on the line's positive side, negative side, or both.
    Note that for the sphere case, we will assume that the line of point and normal goes
    through the sphere's mid point, since all colliders involving spheres give us contact points
    like that.
    \param sphere The shape. Can have transformation.
    \param point A point in world coordinates. Used for anchoring the line.
    Assumed to have come from collision detection code.
    \param normal A normal in world coordinates. Expected to have length 1.
    \param depth The depth of the contact point. Only used for approximative calculations in mesh case.
    Used for giving direction to the line. Assumed to have come from collision detection code.
    \retval The elastic rest length, defined as the intersection length between line and shape.
    */
    agx::Real AGXPHYSICS_EXPORT calculateElasticRestLength(
      const agxCollide::Sphere* sphere,
      const agx::Vec3& point,
      const agx::Vec3& normal,
      const agx::Real depth);

    /**
    Computes elastic rest length for this shape, point and normal.
    It is defined as the intersection length of a line (defined by the point
    and the normal) with the shape.
    Note that the intersection can be on the line's positive side, negative side, or both.
    Note that for the sphere case, we will assume that the line of point and normal goes
    through the sphere's mid point, since all colliders involving spheres give us contact points
    like that.
    \param box The shape. Can have transformation.
    \param point A point in world coordinates. Used for anchoring the line.
    Assumed to have come from collision detection code.
    \param normal A normal in world coordinates. Expected to have length 1.
    \param depth The depth of the contact point. Only used for approximative calculations in mesh case.
    Used for giving direction to the line. Assumed to have come from collision detection code.
    \retval The elastic rest length, defined as the intersection length between line and shape.
    */
    agx::Real AGXPHYSICS_EXPORT calculateElasticRestLength(
      const agxCollide::Box* box,
      const agx::Vec3& point,
      const agx::Vec3& normal,
      const agx::Real depth);

    /**
    Computes elastic rest length for this shape, point and normal.
    It is defined as the intersection length of a line (defined by the point
    and the normal) with the shape.
    Note that the intersection can be on the line's positive side, negative side, or both.
    Note that for the sphere case, we will assume that the line of point and normal goes
    through the sphere's mid point, since all colliders involving spheres give us contact points
    like that.
    \param capsule The shape. Can have transformation.
    \param point A point in world coordinates. Used for anchoring the line.
    Assumed to have come from collision detection code.
    \param normal A normal in world coordinates. Expected to have length 1.
    \param depth The depth of the contact point. Only used for approximative calculations in mesh case.
    Used for giving direction to the line. Assumed to have come from collision detection code.
    \retval The elastic rest length, defined as the intersection length between line and shape.
    */
    agx::Real AGXPHYSICS_EXPORT calculateElasticRestLength(
      const agxCollide::Capsule* capsule,
      const agx::Vec3& point,
      const agx::Vec3& normal,
      const agx::Real depth);

    /**
    Computes elastic rest length for this shape, point and normal.
    It is defined as the intersection length of a line (defined by the point
    and the normal) with the shape.
    Note that the intersection can be on the line's positive side, negative side, or both.
    Note that for the sphere case, we will assume that the line of point and normal goes
    through the sphere's mid point, since all colliders involving spheres give us contact points
    like that.
    \param cylinder The shape. Can have transformation.
    \param point A point in world coordinates. Used for anchoring the line.
    Assumed to have come from collision detection code.
    \param normal A normal in world coordinates. Expected to have length 1.
    \param depth The depth of the contact point. Only used for approximative calculations in mesh case.
    Used for giving direction to the line. Assumed to have come from collision detection code.
    \retval The elastic rest length, defined as the intersection length between line and shape.
    */
    agx::Real AGXPHYSICS_EXPORT calculateElasticRestLength(
      const agxCollide::Cylinder* cylinder,
      const agx::Vec3& point,
      const agx::Vec3& normal,
      const agx::Real depth);

    /**
    Computes elastic rest length for this shape, point and normal.
    It is defined as the intersection length of a line (defined by the point
    and the normal) with the shape.
    Note that the intersection can be on the line's positive side, negative side, or both.
    Note that for the sphere case, we will assume that the line of point and normal goes
    through the sphere's mid point, since all colliders involving spheres give us contact points
    like that.
    \param mesh The shape. Can have transformation.
    \param point A point in world coordinates. Used for anchoring the line.
    Assumed to have come from collision detection code.
    \param normal A normal in world coordinates. Expected to have length 1.
    \param depth The depth of the contact point. Only used for approximative calculations in mesh case.
    Used for giving direction to the line. Assumed to have come from collision detection code.
    \retval The elastic rest length, defined as the intersection length between line and shape.
    */
    agx::Real AGXPHYSICS_EXPORT calculateElasticRestLength(
      const agxCollide::Mesh* mesh,
      const agx::Vec3& point,
      const agx::Vec3& normal,
      const agx::Real depth);

  }
}

#endif
