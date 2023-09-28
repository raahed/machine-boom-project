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

#include <agx/agx.h>
#include <agx/Plane.h>

#include <agxCollide/Geometry.h>
#include <agxCollide/ShapePrimitives.h>
#include <agxCollide/Convex.h>

namespace agxTerrain
{
  class SoilWedge
  {
  public:
    enum Side
    {
      MID = 0,
      LEFT = 1,
      RIGHT = 2
    };

    enum SoilIndices : agx::UInt32
    {
      CUTTING_LEFT = 0,
      CUTTING_RIGHT = 1,
      FAILURE_LEFT = 2,
      FAILURE_RIGHT = 3,
      TOP_LEFT = 4,
      TOP_RIGHT = 5
    };

    static const std::array<Side, 3> SideIterable;

    SoilWedge();

    bool getEnable() const;

    void setEnable(bool enabled);

    agx::Vec3& getCuttingPoint(Side side);

    const agx::Vec3& getCuttingPoint(Side side) const;

    agx::Vec3& getTopPoint(Side side);

    const agx::Vec3& getTopPoint(Side side) const;

    agx::Vec3& getFailurePoint(Side side);

    const agx::Vec3& getFailurePoint(Side side) const;

    std::array<agx::Plane, 3>& getPlanes();

    const std::array<agx::Plane, 3>& getPlanes() const;

    agx::Real& getFailureAngle();

    agx::Real getFailureAngle() const;

    agx::Real& getTerrainInclination();

    agx::Real getTerrainInclination() const;

    agx::Real& getToolInclination();

    agx::Real getToolInclination() const;

    agxCollide::Convex* getMesh();

    const agxCollide::Convex* getMesh() const;

    void constructTriangleArrayAndIndices(agx::Vec3Vector& vertices, agx::UInt32Vector& indices) const;

    bool hasIntersectingSideNormals() const;

    void setCuttingPoint(const agx::Vec3& point, Side side);

    void setTopPoint(const agx::Vec3& point, Side side);

    void setFailurePoint(const agx::Vec3& point, Side side);

    void setPlanes(const std::array<agx::Plane, 3>& planes);

    void setFailureAngle(agx::Real angle);

    void setTerrainInclination(agx::Real angle);

    void setToolInclination(agx::Real angle);

    void setMesh(agxCollide::Convex* convex);

    void setBorderProjected(bool isBorderProjected);

    bool isBorderProjected() const;

  private:

  private:
    bool m_enabled;
    std::array<agx::Vec3, 3> m_cutPoints;
    std::array<agx::Vec3, 3> m_topPoints;
    std::array<agx::Vec3, 3> m_failPoints;
    std::array<agx::Plane, 3> m_planes;
    agx::Real m_failureAngle;
    agx::Real m_terrainInclination;
    agx::Real m_toolInclination;
    bool m_isBorderProjected = false;
    agxCollide::ConvexRef m_mesh;
  };

  typedef agx::Vector<SoilWedge> SoilWedgeVector;




  inline agx::Vec3& SoilWedge::getCuttingPoint(Side side)
  {
    return m_cutPoints[side];
  }



  inline const agx::Vec3& SoilWedge::getCuttingPoint(Side side) const
  {
    return m_cutPoints[side];
  }



  inline agx::Vec3& SoilWedge::getTopPoint(Side side)
  {
    return m_topPoints[side];
  }



  inline const agx::Vec3& SoilWedge::getTopPoint(Side side) const
  {
    return m_topPoints[side];
  }



  inline agx::Vec3& SoilWedge::getFailurePoint(Side side)
  {
    return m_failPoints[side];
  }



  inline const agx::Vec3& SoilWedge::getFailurePoint(Side side) const
  {
    return m_failPoints[side];
  }




}
