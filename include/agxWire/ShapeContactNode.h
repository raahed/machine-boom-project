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

#include <agxWire/Node.h>
#include <agxWire/ShapeContactEdge.h>

namespace agxWire
{
  class ShapeContactNode;
  typedef agx::ref_ptr< ShapeContactNode > ShapeContactNodeRef;
  class ShapeContactColliderUtils;

  class AGXPHYSICS_EXPORT ShapeContactNode : public agxWire::EyeNode
  {
  public:

    enum class NodeExistancePolicy
    {
      KEEP = 0, REMOVE, REPLACE, REMOVED
    };

    enum ContactState
    {
      NO_ISSUES = 0,
      PREV_IS_CLOSE = 1 << 0,
      NEXT_IS_CLOSE = 1 << 1,
      PREV_HAS_SAME_NORMAL = 1 << 2,
      NEXT_HAS_SAME_NORMAL = 1 << 3,
      ANGLE_NOT_VALID = 1 << 4,
      NEAR_CLAMP_POSITION = 1 << 5,
      WORTHLESS_MIDDLE_CONTACT = 1 << 6,
      REMOVE_IMEDIATELY = 1 << 7,
      LEAVING_GYPSY = 1 << 8

    };

    static ShapeContactNode* createFromWorld(agxCollide::Shape* shape, const agx::Real& wireRadius, const agx::Vec3& surfacePosition, const ShapeCurvature& worldCurvature);
    static ShapeContactNode* getAsShapeContact(agxWire::Node* node);
    static const ShapeContactNode* getAsShapeContact(const agxWire::Node* node);

  public:

    ShapeContactNode();
    ShapeContactNode(agxCollide::Shape* shape);
    ShapeContactNode(agxCollide::Shape* shape, const agx::Real& wireRadius, const agx::Vec3& shapeLocalPoint);

    /**
    Initialize the node:
    1. finding the local curvature
    given the surrounding nodes and a point in world
    that is supposed to be on the shape surface.
    2. Give its node material the correct friction coefficients.
    3. Find a position relative the surface, given the local curvature.

    \param wire - the wire of which it is inserted
    \param prevNode - the preceding node
    \param nextNode - the succeeding node
    \param surfacePoint - the world position of the nodes projection to a shape surface (usually found from collision detection)
    */
    agx::Bool initialize(const Wire* wire, const Node* prevNode, const Node* nextNode, const agx::Vec3& surfacePoint);


    /**
    Find out if the node is a cylinder contact.
    \returns true if the node has one or more of the following states set: GYPSY_CONTACT, NO_MOVEMENT_RANGE or NO_CIRCLE_CONTACT.
    */
    agx::Bool hasCylinderContactState() const;

    /**
    Treat special cases where the node shape is a cylinder
    and hasCylinderContactState() return true.
    This will force the node edge to either be of zero length,
    aline with the cylinder axis
    and/or exist only on the center plane of the cylinder.
    */
    agx::Bool handleCylinderContact();

    /**
    \returns true - if the node is near the end of its edge
    */
    agx::Bool isNearClampPosition(const agx::Real& edgeDefaultLength) const;

    /**
    \returns the current shape of the node
    */
    const agxCollide::Shape* getShape() const;

    /**
    \returns the current shape of the node
    */
    agxCollide::Shape* getShape();

    /**
    \returns the shape edge object of this node
    */
    ShapeContactEdge& getShapeContactEdge();
    const ShapeContactEdge& getShapeContactEdge() const;

    /**
    Position the node in world coordinates. The node will get an updated shape translate
    \param worldPosition - wanted world position of the node.
    */
    void setWorldPosition(const agx::Vec3& worldPosition);

    /**
    \returns reference to the velocity defined in world coordinates.
    */
    agx::Vec3& velocity();

    /**
    \returns the velocity of the node in world coordinates.
    */
    virtual agx::Vec3 getVelocity() const override;

    /**
    Set the velocity of the node in world coordinates.
    */
    void setVelocity( const agx::Vec3& velocity );

    /**
    \returns the veolcity of the node along it's edge.
    */
    agx::Real getEdgeVelocity() const;

    /**
    set the velocity of the node along it's edge.
    */
    void setEdgeVelocity(const agx::Real& vel);

    /**
    \returns the normal of the shape where the node is positioned in world coordinates.
    */
    agx::Vec3 getWorldSurfaceNormal() const;

    /**
    \returns the normalized node edge direction in world coordinates
    */
    agx::Vec3 getWorldEdgeNormal() const;

    /**
    \returns the edge start and end position in world coordinates
    */
    void getWorldEdgeEnds(agx::Vec3& edgeStart, agx::Vec3& edgeEnd);

    /**
    \returns the edge start position, in edge coordinates. Should be <= 0.
    */
    agx::Real getEdgeStart() const;

    /**
    \returns the edge end position, in edge coordinates. Should be >= 0.
    */
    agx::Real getEdgeEnd() const;

    /**
    Get data in shape coordinates
    */
    agx::Vec3 getShapeSurfaceNormal() const;
    agx::Vec3 getShapeEdgeNormal() const;
    agx::Vec3 getShapeTranslate() const;

    /**
    Update position using a local translate
    */
    void setShapeTranslate( const agx::Vec3& localPos);

    /**
    \returns extra length found from a solve of the WireContactDirectSolver.
    */
    agx::Real getExtraLength() const;

    /**
    Set the extra lenght of this node this timestep.
    - an estimate of how much room there is for the wire to be extended in length due to contact node movement.
    */
    void setExtraLength(const agx::Real extraLength);

    /**
    Set an edge to the node, containing shape etc.
    */
    void setShapeContactEdge(ShapeContactEdge& shapeContactEdge);

    /**
    Undo all clamping
    */
    void resetEdgeLength(agx::Real edgeHalfLength);

    AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::ShapeContactNode );

  protected:
    virtual ~ShapeContactNode();

  private:
    agx::Real m_extraLength;

    agx::Vec3 m_velocity;

    ShapeContactEdge m_shapeEdge;

  };


  struct ContactNodeData
  {
    ContactNodeData();

    ContactNodeData(ShapeContactNode* contactNode, NodeConstIterator contactNodeIt);

    agx::Real calculateMass(const Wire* wire) const;

    ShapeContactNode* contactNode;
    NodeConstIterator contactNodeIt;
    ShapeContactNode::NodeExistancePolicy existancePolicy;
    int state;

  };

  typedef agx::HashSet< ContactNodeData* > ContactNodeDataSet;
  typedef agx::List< ContactNodeData > ContactNodeDataList;
  typedef agx::Vector<ContactNodeDataList> ContactNodeDataListVector;

  typedef ContactNodeDataList::const_iterator ContactNodeDataListIterator;
  typedef agx::List< ContactNodeDataListIterator > ContactNodeDataListIteratorList;


}

