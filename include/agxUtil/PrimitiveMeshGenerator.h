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

#include <agxCollide/Convex.h>

namespace agxCollide
{
  class Box;
  class Capsule;
  class Cylinder;
  class Sphere;
  class RenderData;
}

namespace agxUtil
{
  class AGXPHYSICS_EXPORT PrimitiveMeshGenerator
  {
    public:
      /**
      Creates vertices of a half-sphere of given radius.
      \param numSegments - number of circle segments
      \param numRows - number of rows in "height"
      \param radius - base radius of the half-sphere
      \param top - true if this sphere is the top of a complete sphere (top along y axis)
      \param yOffset - offset of this half-sphere along local y axis
      \param[out] vertices - container with vertices
      */
      static void createHalfSphereVertices( agx::UInt numSegments, agx::UInt numRows, agx::Real radius, agx::Bool top, agx::Real yOffset, agx::Vec3Vector& vertices );

      /**
      Creates a cone cap given radii and height
      \param numSegments - number of circle segments
      \param topRadius - radius of the top of the cone
      \param bottomRadius - radius of the bottom of the cone
      \param height - the height of the cone
      \param basey - the y value for the bottom of the cone
      \param[out] vertices - container with vertices
      */
      static void createConeCapVertices( agx::UInt numSegments, agx::Real topRadius, agx::Real bottomRadius,
                                         agx::Real height, agx::Real basey, agx::Vec3Vector& vertices );

      /**
      Creates cylinder cap of given radius and height.
      \param numSegments - number of segment about the cylinder axis
      \param radius - radius of the cylinder cap
      \param height - height of the cylinder cap
      \param[out] vertices - container with vertices
      */
      static void createCylinderCapVertices( agx::UInt numSegments, agx::Real radius, agx::Real height, agx::Vec3Vector& vertices );

      /**
      Creates circle of given radius.
      \param numSegments - number of circle segments
      \param radius - radius of the circle
      \param yOffset - offset along local y axis
      \param top - true if this is the top, i.e., normals pointing in positive y axis
      \param[out] vertices - container with vertices
      \param useCenter - creates a vertex in the middle of the circle if true (default). Otherwise it creates a ring.
      */
      static void createCircleVertices( agx::UInt numSegments, agx::Real radius, agx::Real yOffset, agx::Bool top, agx::Vec3Vector& vertices, agx::Bool useCenter = true );

      /**
      Creates capsule vertices.
      \param radius - radius of the capsule
      \param height - height of the capsule
      \param resolution - resolution scaler, higher value results in more triangles
      \return capsule vertices
      */
      static agx::Vec3Vector createCapsuleVertices( agx::Real radius, agx::Real height, agx::Real resolution = agx::Real( 1 ) );

      /**
      Creates a convex capsule.
      \param radius - radius of the capsule
      \param height - height of the capsule
      \param resolution - resolution scaler, higher value results in more triangles
      \return convex shape
      */
      static agxCollide::ConvexRef createCapsule( agx::Real radius, agx::Real height, agx::Real resolution = agx::Real( 1 ) );

      /**
      Creates cylinder vertices.
      \param radius - radius of the cylinder
      \param height - height of the cylinder
      \param resolution - resolution scaler, higher value results in more triangles
      \return cylinder vertices
      */
      static agx::Vec3Vector createCylinderVertices( agx::Real radius, agx::Real height, agx::Real resolution = agx::Real( 1 ) );

      /**
      Creates a convex cylinder.
      \param radius - radius of the cylinder
      \param height - height of the cylinder
      \param resolution - resolution scaler, higher value results in more triangles
      \return convex shape
      */
      static agxCollide::ConvexRef createCylinder( agx::Real radius, agx::Real height, agx::Real resolution = agx::Real( 1 ) );

      /**
      Creates sphere vertices.
      \param radius - radius of the sphere
      \param resolution - resolution scaler, higher value results in more triangles
      \return sphere vertices
      */
      static agx::Vec3Vector createSphereVertices( agx::Real radius, agx::Real resolution = agx::Real( 1 ) );

      /**
      Creates a convex sphere.
      \param radius - radius of the sphere
      \param resolution - resolution scaler, higher value results in more triangles
      \return convex shape
      */
      static agxCollide::ConvexRef createSphere( agx::Real radius, agx::Real resolution = agx::Real( 1 ) );

      /**
      Creates box of given half extents. 24 vertices.
      \param halfExtents - box half extents
      \return box vertices
      */
      static agx::Vec3Vector createBoxVertices( const agx::Vec3& halfExtents );

      /**
      Creates a convex box with 24 vertices.
      \param halfExtents - box half extents
      \return convex shape
      */
      static agxCollide::ConvexRef createBox( const agx::Vec3& halfExtents );

      /**
      Creates convex given shape of primitive type BOX, CAPSULE, CYLINDER or SPHERE. Any
      other type is ignored and nullptr is returned.
      \param shape - primitive shape of type BOX, CAPSULE, CYLINDER or SPHERE
      \param resolution - resolution scaler, higher value results in more triangles (ignored if BOX)
      \return convex shape if successful - otherwise nullptr
      */
      static agxCollide::ConvexRef createConvex( const agxCollide::Shape* shape, agx::Real resolution = agx::Real( 1 ) );

      /**
      Creates a mesh given a primitive shape.
      \param shape - primitive shape
      \param resolution - resolution scaler, higher value results in more triangles (ignored if BOX)
      \return a mesh if successful, otherwise nullptr
      */
      static agxCollide::TrimeshRef createMesh( const agxCollide::Shape* shape, agx::Real resolution = agx::Real( 1 ) );


      /**
      Creates convex vertices given shape of primitive type BOX, CAPSULE, CYLINDER or SPHERE. Any
      other type is ignored and an empty vector is returned.
      \param shape - primitive shape of type BOX, CAPSULE, CYLINDER or SPHERE
      \param resolution - resolution scaler, higher value results in more triangles (ignored if BOX)
      \return convex shape vertices if successful - otherwise empty vector
      */
      static agx::Vec3Vector createConvexVertices( const agxCollide::Shape* shape, agx::Real resolution = agx::Real( 1 ) );

      /**
      Creates a convex cone vertices.
      \param topRadius - the top radius of the cone
      \param bottomRadius - the bottom radius of the cone
      \param height - the height of the cone
      \param resolution - resolution scaler, higher value results in more triangles
      \return cone vertices
      */
      static agx::Vec3Vector createConeVertices( agx::Real topRadius, agx::Real bottomRadius, agx::Real height,
                                                 agx::Real resolution = agx::Real( 1 ) );

      /**
      Creates a convex cone.
      \param topRadius - radius on the top of the cone
      \param bottomRadius - radius on the bottom of the cone
      \param height - height of the cone
      \param resolution - resolution scaler, higher value results in more triangles
      \return convex shape of a cone
      */
      static agxCollide::ConvexRef createCone( agx::Real topRadius, agx::Real bottomRadius, agx::Real height,
                                               agx::Real resolution = agx::Real( 1 ) );

      /**
      Creates a trimesh of a hollow cone.
      \param topOuterRadius - radius of the outer top of the hollow cone
      \param bottomInnerRadius - radius of the inner bottom of the hollow cone
      \param outerHeight - height of the hollow cone
      \param thickness - thickness of the hollow cone
      \param resolution - resolution scaler, higher value results in more triangles
      \return trimesh shape of a hollow cone
      */
      static agxCollide::TrimeshRef createHollowCone( agx::Real topOuterRadius, agx::Real bottomInnerRadius,
                                                      agx::Real outerHeight, agx::Real thickness,
                                                      agx::Real resolution = agx::Real( 1 ) );

      /**
      Creates a trimesh of a hollow cylinder.
      \param innerRadius - radius of the inner surface of the cylinder
      \param height - height of the cylinder
      \param resolution - resolution scaler, higher value results in more triangles
      \return trimesh shape of a hollow cylinder
      */
      static agxCollide::TrimeshRef createHollowCylinder( agx::Real innerRadius, agx::Real height, agx::Real thickness,
                                                          agx::Real resolution = agx::Real( 1 ) );

      /**
      Creates a RenderData with per vertex normals and texture coordinates of a box.
      \param halfExtents - the half extents of the box
      \return RenderData of a box
      */
      static agxCollide::RenderDataRef createBoxRenderData(const agx::Vec3& halfExtents);

      /**
      Creates a RenderData with per vertex normals and texture coordinates of a sphere.
      \param radius - the radius of the sphere
      \param numSegments - the vertical and horizontal number of segments, determines the resolution
      \return RenderData of a sphere
      */
      static agxCollide::RenderDataRef createSphereRenderData(agx::Real radius,
        agx::UInt32 numSegments = 32);

      /**
      Creates a RenderData with per vertex normals and texture coordinates of a cylinder.
      \param radius - the radius of the cylinder
      \param height - the height of the cylinder
      \param numCircleSegments - number of circle segments, determines the resolution
      \return RenderData of a cylinder
      */
      static agxCollide::RenderDataRef createCylinderRenderData(agx::Real radius, agx::Real height,
        agx::UInt32 numCircleSegments = 32);

      /**
      Create a polyhedra mesh from a randomly generated Voronoi Diagram by using the method described in:
      https://www.sciencedirect.com/science/article/pii/S0032591014005191

      The polyhedra is created by taking the center cell of a Voronoi diagram generated from randomly placed
      equidistant point in a bound. The vertices of the center cell are used to generate a convex shape. The scale
      parameter is used to modify the generated points in the x, y and z dimensions to create shapes of different sizes
      and aspect ratios.

      \param seed - the seed used to generate the random points in the Vornoi Diagram.
      \param size - scale factors in the x,y and z dimensions used to modify the generated vertices.
      \param pointDistance - the minimum distance between the generated points in the Voronoi diagram.
      \param boundSize - The size of the Voronoi Diagram bound.
      \return a convex shape of the generated polyhedra.
      */
      static agxCollide::ConvexRef generateVoronoiPolyhedraMesh( agx::Vec3 size,
                                                                 unsigned int seed,
                                                                 agx::Real pointDistance = 0.75,
                                                                 agx::Real boundSize = 5 );

      /**
      Create polyhedra points from a randomly generated Voronoi Diagram by using the method described in:
      https://www.sciencedirect.com/science/article/pii/S0032591014005191

      The polyhedra is created by taking the center cell of a Voronoi diagram generated from randomly placed
      equidistant point in a bound. The vertices of the center cell of the diagram are extracted and returned.

      \param seed - the seed used to generate the random points in the Vornoi Diagram.
      \param pointDistance - the minimum distance between the generated points in the Voronoi diagram.
      \param boundSize - The size of the Voronoi Diagram bound.
      \return a vector of the vertices of the generated polyhedra cell.
      */
      static agx::Vec3Vector generateVoronoiPolyhedraVertices( unsigned int seed,
                                                               agx::Real pointDistance = 0.75,
                                                               agx::Real boundSize = 5.0 );

  };
}
