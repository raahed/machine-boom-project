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

#include <agxTerrain/export.h>
#include <agx/Referenced.h>
#include <agxCollide/Geometry.h>
#include <agx/HashTable.h>

namespace agxTerrain
{
  class TerrainContactGenerator;
  typedef agx::HashTable<agx::Vec2i, agx::Real> TerrainIndexToDepthTable;

  AGX_DECLARE_POINTER_TYPES(TerrainContact);
  AGX_DECLARE_VECTOR_TYPES(TerrainContact);

  /**
  Class that contains information about a terrain contact between an external object and the terrain geometry.
  The contact contains an estimated area calculated from grid points intersecting the colliding geometry as well as
  mean stress calculated from estimated area and contact forces. Stress distribution can also be estimated by scaling the
  surface stress with the depth in each grid point.

  This data container is used primarily in soil compaction calculation.
  */
  class AGXTERRAIN_EXPORT TerrainContact : public agx::Referenced
  {
  public:
    /**
    Struct holding information about a surface force of the Terrain contact. The surface force points are derived from the contact points
    between the geometry where the positions has been projected to the Terrain surface.
    */
    struct SurfaceForce {
      agx::Vec3 localPoint;
      agx::Real verticalForce;
    };

    typedef agx::Vector<SurfaceForce> SurfaceForceVector;

  public:

    /**
    Get the geometry colliding with the terrain.
    \return The geometry colliding with the terrain.
    */
    const agxCollide::Geometry* getGeometry() const;

    /**
    Get the table which holds information about the contact depth at each terrain index in the contact area.
    \return The geometry colliding with the terrain.
    */
    const TerrainIndexToDepthTable& getDepthTable() const;

    /**
    The 3D bound spanned by the contact points of the colliding geometry.
    \return The contact bound.
    */
    agx::Bound3 getContactBound() const;

    /**
    Get the contact area. The contact area is approximated by doing intersection tests between the colliding geometry and the terrain
    in grid points close to the contact points. All the grid points which intersect the collider are used when calculating the total area
    which is given by: number_of_grid_points * gridElementArea.
    \return The approximated contact area from intersected grid points.
    */
    agx::Real getArea() const;

    /**
    Get the mean depth of the intersected grid points in the TerrainContact.
    \return The mean depth of the Terrain contact.
    */
    agx::Real getMeanGridDepth() const;

    /**
    Get the time that the colliding geometry has been in contact with the terrain. Mainly used in compression calculation in order to include compression rate relaxation.
    \return The current contact time of the Terrain contact.
    */
    agx::Real getCurrentContactTime() const;

    /**
    Get the average surface stress of the contact which is calculated using the approximated contact area and the total surface force of the contact.
    \return The average surface stress of the contact.
    */
    agx::Real getSurfaceStress() const;

    /**
    Get the specific contact stress in a terrain index in the contact by scaling the average stress with the depth in the specific index.
    \return The specific surface stress in the contact terrain index.
    */
    agx::Real getSurfaceStressInIndex(const agx::Vec2i& terrainIndex) const;

    /**
    Get the total surface force of the Terrain contact from all the contact points.
    \note - This is only accessible in post-update of the simulation
    \return The total surface force of the Terrain contact.
    */
    agx::Real getSurfaceForce() const;

    /**
    Get the maximum force of the Terrain contact from the associated contact points
    \return The total surface force of the Terrain contact.
    */
    agx::Real getMaxSurfaceForce() const;

    /**
    Get the surface point forces active in the Terrain contact. The surface point forces consists of the contact points in the associated geometry contact, which
    has been projected onto the terrain surface.
    \return The surface points forces in the Terrain contact.
    */
    const SurfaceForceVector& getSurfacePointForces() const;

  public:
    // Internal, but public methods

    /**
    Internal method.

    Default constructor
    */
    TerrainContact();

    /**
    Internal method.

    Construct a Terrain contact with specified arguments.
    */
    TerrainContact(const agxCollide::Geometry* geometry,
      const TerrainIndexToDepthTable terrainIndicesToDepth,
      const agx::Real area,
      const agx::Real meanGridDepth,
      const agx::Real currentContactTime,
      const agx::Bound3 contactBound);

    /**
    Internal method.

    Set the surface stress of the contact.
    \param surfaceStress - The surface stress to set.
    */
    void setSurfaceStress(agx::Real surfaceStress);

    /**
    Internal method.

    Set the surface force of the contact.
    \param surfaceForce - The surface force to set on the contact
    */
    void setSurfaceForce(agx::Real surfaceForce);

    /**
    Internal method.

    Set the contact area of the contact.
    \param area - The surface area of the contact.
    */
    void setContactArea(agx::Real area);

    /**
    Internal method.

    Set the surface point forces of the contact.
    \param pointForces - The point forces to set on the contact.
    */
    void setSurfacePointForces(const SurfaceForceVector& pointForces);

  protected:
    /**
    Reference counted object - protected destructor.
    */
    virtual ~TerrainContact();

  public:
    agx::Real calculateTotalForce() const;

    private:
      const agxCollide::Geometry* m_geometry;
      TerrainIndexToDepthTable m_terrainIndicesToDepth;
      agx::Real m_area;
      agx::Real m_meanGridDepth;
      agx::Real m_currentContactTime;
      agx::Bound3 m_contactBound;
      agx::Real m_surfaceStress;
      agx::Real m_surfaceForce;
      SurfaceForceVector m_pointForces;
  };
}