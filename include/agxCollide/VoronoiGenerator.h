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

#ifndef AGXCOLLIDE_VORONOIGENERATOR_H
#define AGXCOLLIDE_VORONOIGENERATOR_H

#include <agx/agx.h>
#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agx/debug.h>
#include <agx/Vec3.h>
#include <agx/Bound.h>
#include <agxCollide/Shape.h>

namespace agxCollide
{
  /**
  Class for generating 3D Voronoi diagram in various shapes. Trimesh it NOT supported.
  */
  AGX_DECLARE_POINTER_TYPES(VoronoiGenerator);
  class AGXPHYSICS_EXPORT VoronoiGenerator : public agx::Referenced
  {
  public:
    struct AGXPHYSICS_EXPORT VoronoiCell
    {
    public:
      agx::Vec3 center;
      agx::Vec3Vector voronoiCellVertices;
    };

    typedef agx::Vector<VoronoiCell> VoronoiDiagramData;

  public:
    /**
    * Default constructor
    */
    VoronoiGenerator();

    /**
    * Generate the 3D Voronoi diagram inside a 3 dimensional bound from a specified point,
    */
    VoronoiDiagramData generateVoronoiDiagramInBound(const agx::Bound3& bound, const agx::Vec3Vector& voronoiPoints);

    /**
    * Generate the 3D Voronoi diagram inside a 3 dimensional bound from a specified point,
    */
    VoronoiDiagramData generateVoronoiDiagramInBound( const agx::Bound3& bound, const agx::UInt numPoints );

    /**
    * Generate the 3D Voronoi diagram in a shape from a set of specified points, generated in the LOCAL coordinate system of the shape.
    */
    VoronoiDiagramData generateVoronoiDiagramInShape(const agxCollide::Shape* shape, const agx::Vec3Vector& voronoiPoints);

    /**
    * Generate the 3D Voronoi diagram in a shape from a set of specified points generated inside the specified shape.
    */
    VoronoiDiagramData generateVoronoiDiagramInShape(const agxCollide::Shape* shape, agx::UInt numPoints);

    /**
    Create a polyhedra from a randomly generated Voronoi Diagram by using the method described in:
    https://www.sciencedirect.com/science/article/pii/S0032591014005191

    The polyhedra is created by taking the center cell of a Voronoi diagaram generated from randomly placed
    equidistant point in a bound.

    \param seed - the seed used to generate the random points in the Vornoi Diagram.
    \param pointDistance - the minimum distance between the generated points in the Voronoi diagaram.
    \param boundSize - The size of the Voronoi Diagram bound.

    \return the center Voronoi cell that make up the polyhedra.
    */
    VoronoiCell generateVoronoiPolyhedra(unsigned int seed = 0, agx::Real pointDistance=0.75, agx::Real boundSize=5);

    /**
    * Set resolution for plane tessellation in Voronoi diagram of basic primitives such as Spheres, Cylinder and Capsules.
    */
    void setResolution(agx::Real resolution);

    /**
    * Get resolution for plane tessellation in Voronoi diagram of basic primitives such as Spheres, Cylinder and Capsules.
    */
    agx::Real getResolution() const;

  protected:
    virtual ~VoronoiGenerator();

    agx::Vec3Vector generateEquidistantPointsInBound( agx::Real distance, agx::Real boundHalfVec, unsigned int seed );

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    agx::Real m_resolution;
  };

  AGX_FORCE_INLINE void agxCollide::VoronoiGenerator::setResolution(agx::Real resolution)
  {
    m_resolution = resolution;
  }

  AGX_FORCE_INLINE agx::Real agxCollide::VoronoiGenerator::getResolution() const
  {
    return m_resolution;
  }
}

#endif