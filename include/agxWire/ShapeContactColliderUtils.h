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

#include <agx/Referenced.h>
#include <agx/Vec3.h>
#include <agxWire/ShapeContactEdge.h>
#include <agxWire/ShapeContactNode.h>
#include <agxCollide/Convex.h>
#include <agxCollide/Sphere.h>
#include <agxCollide/Line.h>
#include <agxCollide/WireShape.h>
#include <agxCollide/ShapeCollider.h>
#include <agxCollide/ColliderTable.h>
#include <agxUtil/ConvexReaderWriter/ConvexReaderWriter.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxCollide
{
  class Shape;
  class GeometryContactPtr;
  class Geometry;
}

namespace agxSDK
{
  class Simulation;
}


namespace agxWire
{
  /**
  This class is used for various collision tests between a wire and shapes.
  */
  class AGXPHYSICS_EXPORT ShapeContactColliderUtils : public agx::Referenced
  {
  public:

    /**
    Constructor
    */
    ShapeContactColliderUtils();

    /**
    This method will replace any existance of the older agxWire::ContactNode to the new agxWire::ShapeContactNode
    \return true if a ContactNode is replaced by a ShapeContactNode, false if no node was converted
    */
    agx::Bool replaceContactNodesWithShapeContacts(agxWire::Wire* wire) const;

    static agx::Bool getLocalEdgeEnds(Node* node, agx::Vec3& edgeStart, agx::Vec3& edgeEnd);

    static agx::Material* getMaterial(agxCollide::Shape* shape);
    static agx::Vec3 getWorldEdgeNormal(Node* node);
    static agxCollide::Geometry* getGeometry(Node* node);
    static const agxCollide::Geometry* getGeometry(const Node* node);
    static agx::Vec3 getWorldSurfaceNormal(Node* node);
    static bool onSameGeometry(const Node* node, const Node* otherNode);
    static agx::Real getEdgeVelocity(const Node* node);
    static agx::Bool hasCylinderProperies(const agxCollide::Shape* shape);

    /**
    Find surface position and world normal, if the node is a cylinder contact
    \returns false if the shape is not a cylinder
    */
    static agx::Bool findCylinderSurfacePositionAndNormal(const agxCollide::Shape* shape, const agx::Vec3& nodeWp, agx::Vec3& cylCenterWorld, agx::Vec3& cylSurfaceVectorWorld, agx::Vec3& alongCylinderAxisVectorWorld);


    agxWire::ShapeContactNode* createShapeContact(const agxWire::Wire* wire, agxCollide::Shape* shape, const agx::Vec3& wp) const;
    agxWire::ShapeContactNode* convertToShapeContact(const agxWire::Wire* wire, agxWire::ContactNode* cn) const;
    agx::Real calculateCornerAngle(ShapeContactEdge* edge, const agx::Real& wireRadius, const agx::Real& wirePrecision) const;
    void generateSupportContact(agxSDK::Simulation* sim, ShapeContactNode* cn, agxCollide::Geometry* otherGeom) const;
    agx::Bool getDeepestContact(const agx::Vec3& wirePos, const agx::Real& radius, const agxCollide::Shape* myShape, agx::Vec3& point, agx::Vec3& normal, agx::Real& depth) const;
    agx::Bool collideSphereAndShape(const agx::Vec3 worldPos, const agx::Real sphereRadius, const agxCollide::Shape* shape, agxCollide::ShapeCollider::LocalContactPointVector& contactPoints) const;
    agx::Bool collideWireSegmentAndShape(const agx::Real& radius, const agx::Vec3& sStart, const agx::Vec3& sEnd, const agxCollide::Shape* shape, agxCollide::ShapeCollider::LocalContactPointVector& contactPoints) const;
    size_t collideLineAndShape(const agx::Vec3& sStart, const agx::Vec3& sEnd, const agxCollide::Shape* shape, agxCollide::ShapeCollider::LocalContactPointVector& contactPoints) const;
    agx::Bool collideWithShape(const Node* node, const agxCollide::Shape* shape, const agx::Real& wireRadius) const;
    agx::Bool collideEdgeParallelLinesNearShape(const ShapeContactEdge& edge, const agxCollide::Shape* shape, const agx::Real& allowedOverlap);
    size_t handleStuckInsideOtherShape(const size_t numToReconsider, const agx::Vec3& edgeStart, const agx::Vec3& edgeCenter, const agx::Vec3 edgeEnd,const agxCollide::Shape* shape, agxCollide::ShapeCollider::LocalContactPointVector& contactPoints) const;
    agx::Bool collectContactPointsForEdgeClamping(const ShapeContactEdge& edge, const agxCollide::Shape* shape, const agx::Real& allowedOverlap, agxCollide::ShapeCollider::LocalContactPointVector& contactPoints) const;

    private:

      agxWire::ShapeContactNode* createShapeContact(const agxWire::Wire* wire, agxCollide::Shape* shape, const agx::Vec3& wp,
        const NodeConstIterator beforeNodeIterator, const NodeConstIterator afterNodeIterator,
        const agx::Vec3& point, const agx::Vec3& normal, const agx::Real depth) const;
      agxCollide::Convex* getPossibleSweepedConvex(ShapeContactNode* scn, const agx::Vec3 neighborPosition) const;
      agxCollide::Sphere* getSphere(const agx::Real& radius) const;
      agxCollide::Line* getLine(const agx::Vec3& sStart, const agx::Vec3& sEnd) const;
      agxCollide::WireShape* getWireShape(const agx::Real& radius, const agx::Vec3& sStart, const agx::Vec3& sEnd, agx::AffineMatrix4x4& transform) const;

     agxCollide::SphereRef m_sphere;
     agxCollide::WireShapeRef m_wireSegment;
     agxCollide::LineRef m_line;
  };

  typedef agx::ref_ptr<ShapeContactColliderUtils> ShapeContactColliderUtilsRef;
}

DOXYGEN_END_INTERNAL_BLOCK()
