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



#define DEBUG_RENDER_SHAPE_CONTACTS 0

#include <agx/Referenced.h>
#include <agx/Vec3.h>
#include <agxWire/Node.h>
#include <agxCollide/Shape.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxCollide
{
  class GeometryContactPtr;
  class Geometry;
}

namespace agxWire
{
  class ShapeContactColliderUtils;

  struct ShapeCurvature
  {
    ShapeCurvature()
    {

    }
    ShapeCurvature(agx::Vec3 n, agx::Vec3 prev, agx::Vec3 next)
      : normal(n), prevNormal(prev), nextNormal(next)
    {
    }

    ShapeCurvature& operator=(const ShapeCurvature& rhs)
    {
      this->normal = rhs.normal;
      this->prevNormal = rhs.prevNormal;
      this->nextNormal = rhs.nextNormal;
      return *this;
    }

    agx::Bool isFlat() const
    {
      return agx::equivalent(prevNormal*nextNormal, agx::Real(1));
    }

    agx::Vec3 calculateCurvatureOrthogonalNormal() const
    {
      agx::Vec3 orthoVector = prevNormal ^ nextNormal;
      orthoVector.normalize();
      return orthoVector;
    }

    void normalize()
    {
      normal.normalize();
      prevNormal.normalize();
      nextNormal.normalize();
    }

    void setNormal(agx::Vec3 n)
    {
      normal = n;
    }

    agx::Real getAngle() const
    {
      agx::Real angle = std::acos(agx::clamp(nextNormal *prevNormal, agx::Real(-1), agx::Real(1)));
      return angle;
    }

    agx::Vec3 calculateMeanCurvature() const
    {
      agx::Vec3 totDir = prevNormal + nextNormal;
      totDir.normalize();
      return totDir;
    }
    agx::Vec3 normal;
    agx::Vec3 prevNormal;
    agx::Vec3 nextNormal;
  };

  class AGXPHYSICS_EXPORT ShapeContactEdge
  {
  public:

    enum ClampDirection
    {
      NOT_CLAMPED = 0,
      FORWARD = 1,
      BACKWARD = 2,
      BOTH = 3
    };

    ShapeContactEdge();

    ShapeContactEdge& operator=(const ShapeContactEdge& rhs)
    {
      this->m_shape = rhs.m_shape;
      this->m_edgeNormal = rhs.m_edgeNormal;
      this->m_edgeStart = rhs.m_edgeStart;
      this->m_edgeEnd = rhs.m_edgeEnd;
      this->m_localShapeSurfaceProjectedPosition = rhs.m_localShapeSurfaceProjectedPosition;
      this->m_previousLocalShapePosition = rhs.m_previousLocalShapePosition;
      this->m_worldVectorFromLastProjection = rhs.m_worldVectorFromLastProjection;
      this->getLocalCurvature() = rhs.m_localCurvature;
      return *this;
    }

    static agx::Vec3 calculateEdgeDir(const ShapeCurvature& worldCurvature, const agx::Vec3& posPrev, const agx::Vec3& pos, const agx::Vec3& posNext);

    static agx::Bool findLocalCurvature(
      const ShapeContactColliderUtils* colliderUtils, const agx::Vec3& normal, const agx::Vec3& wpOnSurface,
      const agx::Vec3& wp, const agx::Vec3& wpPrev, const agx::Vec3& wpNext,
      const agx::Real& acceptedWireOverlap, const agxCollide::Shape* shape, ShapeCurvature& worldCurvature);

    static agx::Vec3 transformPointToShape(const agxCollide::Shape* shape, const agx::Vec3& worldPoint);
    static agx::Vec3 transformVectorToShape(const agxCollide::Shape* shape, const agx::Vec3& worldVector);
    static agx::Vec3 transformPointToWorld(const agxCollide::Shape* shape, const agx::Vec3& shapePoint);
    static agx::Vec3 transformVectorToWorld(const agxCollide::Shape* shape, const agx::Vec3& shapeVector);

    void initialize(agxCollide::Shape* shape, const agx::Vec3& worldNormal, const agx::Vec3& surfacePos);

    agx::Real getAngleAroundEdge(const agx::Vec3& wireBack, const agx::Vec3& wireForward) const;
    agx::Bool findWorldPositionAndEdgeGivenLocalCurvature(const NodeConstIterator nIt, const agx::Real& wireRadius, const agx::Real& wirePrecision, agx::Vec3& newWireWorldPosition, agx::Vec3& newWireWorldEdge) const;
    agx::Bool updateCurvature(const ShapeContactColliderUtils* colliderUtils,const NodeConstIterator nIt, const agx::Real acceptedWireOverlap);
    ShapeCurvature transformCurvatureToWorld(const ShapeCurvature& localCurvature) const;
    ShapeCurvature transformCurvatureToShape(const ShapeCurvature& worldCurvature) const;

    ShapeCurvature getWorldCurvature() const;
    ShapeCurvature getShapeLocalCurvature() const;


    agx::Bool update(const NodeConstIterator nIt, const ShapeContactColliderUtils* cu, const agx::Real& wireRadius, const agx::Real& wirePrecision);

    void addClampShape(agxCollide::Shape* shape);
    void clearClampShapes();
    const agxCollide::ShapeObserverVector& getClampShapes() const;

    agx::Vec3 transformPointToShape(const agx::Vec3& worldPoint  ) const;
    agx::Vec3 transformVectorToShape(const agx::Vec3& worldVector) const;
    agx::Vec3 transformPointToWorld(const agx::Vec3& shapePoint  ) const;
    agx::Vec3 transformVectorToWorld(const agx::Vec3& shapeVector) const;

    void setWorldEdgeNormal(const agx::Vec3& worldEdgeDir);
    void setShapeEdgeNormal(const agx::Vec3& shapeEdgeDir);
    agx::Vec3 getShapeEdgeNormal() const;
    agx::Vec3 getWorldEdgeNormal() const;


    void resetEdgeLength(agx::Real edgeHalfLength);
    agx::Bool isClamped(const agx::Real& edgeDefaultLength) const;
    ShapeContactEdge::ClampDirection getClampDirection(agx::Real edgeDefaultLength) const;

    void renderDebug(const agx::Vec4& color) const;

    void setShape(agxCollide::Shape* shape);
    agxCollide::Shape* getShape() const;

    agx::Real getEdgeVelocity(const agx::Vec3& vel) const;

    void setLocalEdgeNormal(agx::Vec3 dir);
    agx::Vec3 getLocalEdgeNormal() const;

    void setEdgeStart(const agx::Real& lengthToStartAlongEdgeDir);
    agx::Real getEdgeStart() const;

    void setEdgeEnd(const agx::Real& lengthToEndAlongEdgeDir);
    agx::Real getEdgeEnd() const;

    void setLocalShapeSurfaceProjectedPosition(agx::Vec3 pos);
    agx::Vec3 getLocalShapeSurfaceProjectedPosition() const;

    void setPreviousLocalShapePosition(agx::Vec3 pos);
    agx::Vec3 getPreviousLocalShapePosition() const;

    void setWorldPositionIfThisEdgeIsUsed(agx::Vec3 worldPos);
    agx::Vec3 getWorldPositionIfThisEdgeIsUsed() const;

    void setWorldVectorFromLastProjection(agx::Vec3 vec);
    agx::Vec3 getWorldVectorFromLastProjection() const;

    ShapeCurvature& getLocalCurvature();
    const ShapeCurvature& getLocalCurvature() const;
    void setLocalCurvature(ShapeCurvature& newLocalCurvature);

    void setWorldNormal(const agx::Vec3& worldNormal);
    agx::Vec3 getWorldNormal() const;

    void setWorldSurfaceProjectedPosition(const agx::Vec3& worldPos);
    agx::Vec3 getWorldSurfaceProjectedPosition() const;

    agx::Bool hasValidShape() const;

  private:
    agxCollide::ShapeObserver m_shape;
    agx::Vec3 m_edgeNormal;
    agx::Real m_edgeStart;
    agx::Real m_edgeEnd;

    agx::Vec3 m_localShapeSurfaceProjectedPosition;
    agx::Vec3 m_previousLocalShapePosition;
    agx::Vec3 m_worldVectorFromLastProjection;

    ShapeCurvature m_localCurvature;

    agxCollide::ShapeObserverVector m_shapesUsedToClamp;
  };

  typedef agx::Vector<ShapeContactEdge> ShapeContactEdgeVector;
}

DOXYGEN_END_INTERNAL_BLOCK()
