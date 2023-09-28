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

#include <agxWire/WireDistanceCompositeConstraint.h>
#include <agxWire/WireContactController.h>

#include <agxSDK/LineCollisionUtils.h>
#include <agxSDK/Simulation.h>

#include <agx/WireContactSolver.h>
#include <agxCollide/Line.h>

#include <agx/LinearProbingHashTable.h>
#include <agx/LinearProbingHashTable.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxSDK
{
  class WireContactEdge;
  class WireContactPosition;
}

namespace agxCollide
{
  class Box;
  class Cylinder;
  class Sphere;
}

namespace agxWire
{
  class Wire;
  class WireGeometryController;
  class WireContactController;
  // Typedefs
  typedef agx::Vector< std::pair< NodeRef, std::pair< NodePtrIterator, agx::Real > > > NodeRefNodePtrListIterRealVector;
  typedef agx::Vector<agxSDK::LineContactEdge> WireContactEdgeVector;

  class AGXPHYSICS_EXPORT WireSensorController : public agx::Referenced
  {
  public:
    WireSensorController(WireGeometryController* geometryController, WireContactController* contactController);

    /**
    Implicitly added to a simulation.
    */
    void addNotification();

    /**
    Implicitly removed from simulation.
    */
    void removeNotification();

    /**
    \return true if the contact should be removed asap
    */
    bool handleSensorContact(agxCollide::GeometryContact *gc);

    /**
    Position a sensor sphere where \p cln is located.
    \param cln - contact node where sensor sphere should be moved to
    */
    void setSensorPosition(agxWire::ContactNode* cln);

    /**
    \return the geometry controller
    */
    agxWire::WireGeometryController* getGeometryController() const { return m_geometryController; }

    /**
    \return the contact controller
    */
    agxWire::WireContactController* getActiveContactController()  const { return m_contactController; }

    /**
    \return true if this geometry belongs to this controller
    */
    bool match(agx::Physics::GeometryPtr geometry) const;

  protected:
    virtual ~WireSensorController() {}

  private:
    agx::observer_ptr< WireGeometryController > m_geometryController;
    agx::observer_ptr< WireContactController > m_contactController;
    agx::UInt32 m_wireId;
  };

  typedef agx::ref_ptr< WireSensorController > WireSensorControllerRef;

  class WireContactGeometryTransferController
  {
  public:
    WireContactGeometryTransferController(WireContactController *contactController);
    ~WireContactGeometryTransferController() {}

    //int handleBox(ContactNode* cln, agxCollide::Geometry* sensor, agxCollide::Geometry* oldGeometry, agxCollide::Geometry* newBoxGeometry);
    //int handleCylinder(ContactNode* cln, agxCollide::Geometry* sensor, agxCollide::Geometry* oldGeometry, agxCollide::Geometry* newCylinderGeometry);
    agx::Real getGeometryPenetrationDepth(agxCollide::Geometry* geometry, const agx::Vec3& contactPoint, const agx::Vec3& edge);
    WireContactController* getActiveContactController() { return m_contactController; }

  private:
    bool getWithinGeometryMovementRange(agxCollide::Geometry* geometry, const agx::Vec3& contactPoint, agxSDK::LineContactEdge* edge);
    bool getWithinBoxMovementRange(agxCollide::Box* boxShape, const agx::Vec3& contactPoint, agxSDK::LineContactEdge* edge);
    bool getWithinCylinderMovementRange(agxCollide::Cylinder* cylinderShape, const agx::Vec3& contactPoint, agxSDK::LineContactEdge* edge);

    agx::Real getBoxPenetrationDepth(agxCollide::Box* box, const agx::Vec3& contactPoint, agx::Vec3 edge);
    agx::Real getCylinderPenetrationDepth(agxCollide::Cylinder* cylinder, agx::Vec3 contactPoint, agx::Vec3 edge);

    agx::observer_ptr< WireContactController > m_contactController;
  };

  class ContactNodePlacementData
  {
  public:
    ContactNodePlacementData() : movingFromStartToEnd(false), openAngle(false), wireRadius(0), shortOffsetLength(0), triangleIndex(0), edgeIndex(0) {}
    agx::Vec3 contactPointInsideGeometry;
    agx::Vec3 contactPointOnSurface;
    agx::Vec3 newWP;
    agx::Vec3 edgeStartRelativeMovementWorld;
    agx::Vec3 offsetW;
    agx::Vec3 moveDir;
    bool movingFromStartToEnd;
    bool openAngle;
    agx::Real wireRadius;
    agx::Real shortOffsetLength;
    size_t triangleIndex;
    size_t edgeIndex;

    ~ContactNodePlacementData() {}
  };

  class WireRay// : public agx::Referenced
  {
  public:
    WireRay() {}
    WireRay(agx::Vec3 begin, agx::Vec3 end) : m_begin(begin), m_end(end) {}
    AGX_FORCE_INLINE agx::Vec3 getBegin() const { return m_begin; }
    AGX_FORCE_INLINE agx::Vec3 getEnd() const { return m_end; }
    AGX_FORCE_INLINE agx::Vec3 getDir() const { return m_end - m_begin; }
  private:
    agx::Vec3 m_begin;
    agx::Vec3 m_end;
  };
  //typedef agx::ref_ptr<WireRay> WireRayRef;

  class AGXPHYSICS_EXPORT WireOldContactController : public agxWire::WireContactController
  {
  public:
    WireOldContactController(Wire* wire);

    static BodyFixedNode* changeToBodyFixed(ContactNode* contactNode, WireDistanceCompositeConstraint* wdcc);

    /**
    Will call moveEdgeContactNodes.
    */
    virtual void preCollideUpdate();
    virtual void preUpdate();
    virtual void postUpdate();

    virtual agx::Bool handleGeometryContact(agxCollide::GeometryContact* geometryContact, agxCollide::Geometry* otherGeometry);

    WireContactGeometryTransferController* getContactGeometryTransferController() { return m_contactGeometryTransferController; }

    /**
    Check if contact nodes could be transfered between the two cylinders 1,2
    */
    bool almostPerpendicularCylinders(agxCollide::Geometry* cyl1, agxCollide::Geometry* cyl2);

    bool getIfStuckNode(Node* cln) { return m_stuckNodes.contains(cln); }
    bool getIfStuckNode(const Node* cln) const { return m_stuckNodes.contains(const_cast<Node*>(cln)); }
    bool getIfStuckWhereNextGeometryIsAvailable(Node* cln);
    int getClampDirection(ContactNode* cln, agxCollide::Geometry* collidingGeometry, const agx::Vec3& axis, const agx::Vec3& sensorPos, agx::Vec3& moveDirection, agxCollide::Geometry* sensor = nullptr, agxCollide::GeometryContact *gc = nullptr);

    /**
    Calculate edge offset for handling of nodes using this contact controller.
    */
    agx::Vec3 calculateShapeEdgeOffset(ContactNode* cln, agxCollide::Shape* shape, agx::Real wireRadius) const;

    bool removeContact(WireDistanceCompositeConstraint* ldcc, ContactNode* cln);
    WireSensorController* getSensorController() const { return m_sensorController; }

  protected:
    friend class WireContactGeometryTransferController;
    friend class WireSensorController;
    friend class WireImpactController;
    virtual ~WireOldContactController();


  protected:
    /**
    Small class that temporarily will follow new contact nodes around when they are being validated.
    */
    class ContactNodeCustomData : public agx::Referenced
    {
    public:
      ContactNodeCustomData(bool passedEdgePassTest = false) : m_passedEdgePassTest(passedEdgePassTest) {}
      ContactNodeCustomData(const agx::Vec3& worldDirTowardSurface, const agx::Vec3& orgShapeEdgeStart, const agx::Vec3& orgShapeEdgeEnd)
        : m_passedEdgePassTest(false), m_dirTowardSurface(worldDirTowardSurface), m_orgShapeEdgeStart(orgShapeEdgeStart),
        m_orgShapeEdgeEnd(orgShapeEdgeEnd) {}

      bool m_passedEdgePassTest;
      agx::Vec3 m_dirTowardSurface;               /**< Direction toward the surface of the wire geometry in world coordinates */
      agx::Vec3 m_orgShapeEdgeStart;              /**< Edge start given m_orgShapeCorrectedContactPoint in shape coordinates */
      agx::Vec3 m_orgShapeEdgeEnd;                /**< Edge end given m_orgShapeCorrectedContactPoint in shape coordinates */
    protected:
      virtual ~ContactNodeCustomData() {}
    };

    typedef agx::ref_ptr< ContactNodeCustomData > ContactNodeCustomDataRef;
    typedef agx::Vector< ContactNodeCustomDataRef > ContactNodeCustomDataRefVector;

  private:
    class EdgeTransferController
    {
    public:
      EdgeTransferController(WireOldContactController* contactController);
      ~EdgeTransferController() {}
      /**
      Will calculate the new edge on the geometry and move the contact to that
      */
      agx::Real transferToNextEdge(ContactNode* cln, agx::Real tPos);


    private:

      /**
      Handles edge transfer on a box
      */
      bool handleBoxEdgeTransfer(ContactNode* cln, agx::Real& tPos, agxSDK::LineContactEdge* oldEdge);
      /**
      Finds the two other edges of a box that start or end where the contact is about to slip of the box
      */
      bool getNeighboringBoxEdges(ContactNode* cln, agx::Real tPos, WireContactEdgeVector& edges);
      /**
      given two new edges on a box, we choose the correct one to jump upon to
      */
      void chooseCorrectBoxEdge(ContactNode* cln, agx::Real& tPos, WireContactEdgeVector& edges);


      /**
      Handles edge transfer on a circle end
      */
      bool handleCylinderEdgeTransfer(ContactNode* cln, agx::Real& tPos, agxSDK::LineContactEdge* oldEdge);

      /**
      Given previous and next node, decides if this node should fall off circle edge of cylinder (to cyl. axis edge).
      */
      bool getNeighbourAlsoCircleFallOff(ContactNode* cln, Node* otherNode, WireContactEdgeVector& edges);

      /**
      Finds the two other edges of a circle. The three possible are the ones along
      */
      bool getNeighboringCylinderEdges(ContactNode* cln, agx::Real tPos, WireContactEdgeVector& edges, bool onCircle);
      /**
      given two new edges on a box, we choose the correct one to jump upon to
      */
      void chooseCorrectCylinderEdge(ContactNode* cln, agx::Real& tPos, WireContactEdgeVector& edges, bool onCircle);

      /**
      Handles edge transfer on a box
      */
      bool handleMeshEdgeTransfer(ContactNode* cln, agx::Real& tPos, agxSDK::LineContactEdge* oldEdge);


      agx::observer_ptr< WireOldContactController > m_contactController;
    };

    /**
    Flip normals.
    //The normal is supposed to point FROM the non wire geometry
    */
    static void flipContactNormals(agxCollide::GeometryContact* geometryContact, agxCollide::Geometry* otherGeometry);

    agx::Real calculateMassForContact(WireDistanceCompositeConstraint* wdcc, Node* cln);

    bool validateDistanceBetweenWires(const agx::Vec3& eStartNewWorld, const agx::Vec3& eEndNewWorld, const agx::Vec3& eStartOldWorld, const agx::Vec3& eEndOldWorld, agx::Real sensorRadius);

    void getAllContactNodeTypes(WireDistanceCompositeConstraint* ldcc, NodePtrContainer& allMeshContacts, NodePtrContainer& allBoxContacts, NodePtrContainer& allCylinderContacts);
    void getAllContactNodes(WireDistanceCompositeConstraint* ldcc, agx::List<NodeIterator>& allContacts);
    void getMovableContactNodes(const agx::List<NodeIterator>& allContacts, agx::List<NodeIterator>& contacts, agx::List<NodeIterator>& noMovementContacts);
    bool analyzeIfBoxContactIsLoose(agx::Vec3 edgeStart, agx::Vec3 edgeEnd, agx::Vec3 cnPoint, agx::Vec3 prevPoint, agx::Vec3 nextPoint);
    bool nodeFallsOffCylinderWithoutCircleEdges(const ContactNode* cln);
    bool areContactsOnNeighboringEdges(ContactNode* cln1, ContactNode* cln2);
    bool checkIfPulledOffAlongEdge(ContactNode* cln, const agx::Vec3 vPrev, const agx::Vec3 vNext);
    int validateContact(ContactNode* cln);
    bool surroundingNodeInsideGeometry(const agx::Vec3& prev, const ContactNode* cln, const agx::Vec3& next, agxCollide::Geometry* geometry);
    bool surroundingNodeInsideBox(const agx::Vec3& prev, const ContactNode* cln, const agx::Vec3& next, agxCollide::Box* box);
    bool surroundingNodeInsideCylinder(agx::Vec3 prev, agx::Vec3 next, agxCollide::Cylinder* cylinder);
    bool considerTurningToFree(ContactNode* cln, const agx::Vec3& prev, const agx::Vec3& next, const agx::Vec3& edge, const agx::Vec3& center);
    bool checkIfFuturePositionIsBad(Node* middleContact);
    void fixFreeEndContactPassings(WireDistanceCompositeConstraint* ldcc);
    Node* getContactAtBadConfiguration(const NodeIterator it, const NodeIterator next, const NodeIterator nextNext, const agxCollide::Shape* shape, const WireDistanceCompositeConstraint* ldcc);
    void catchBadContactConfiguration(WireDistanceCompositeConstraint* ldcc, agx::List<NodeIterator>& allContacts);
    void moveEdgeContactNodes();
    bool testWireGeometryCollision(agxCollide::Geometry* collidedGeometry, agxCollide::Geometry* wire, agx::Real& depth, agx::Vec3& contactPointW, agx::Vec3& contactNormalW, bool allowStartInsideGeometry);
    bool clipLine(agxCollide::Geometry* collidedGeometry, const WireRay& wireRay, agx::Vec3& point);
    bool testWireGeometryCollision(agxCollide::Geometry* collidedGeometry, const WireRay& wireRay, agx::Real& depth, agx::Vec3& contactPointW, bool allowStartInsideGeometry);
    void removeTightCylinderContacts(WireDistanceCompositeConstraint* ldcc, agx::List<NodeIterator>& allContacts);
    agx::Real testEdgeTransfer(ContactNode* cn, agx::Real t);
    bool findNextTriangleAndEdge(agxCollide::Geometry* meshGeometry, agxCollide::Mesh* mesh, agx::Vec3& clnPosW,
      agx::Vec3& lineDirW, agx::Real& lineLength, size_t& triangleIndex, size_t& edgeIndex, bool& convexAngle,
      agx::Vec3 nodePosW, agx::Vec3 linePlaneNormal, bool useRelVel, size_t numIterations);
    bool analyzeMeshContact(ContactNode* cln, agxCollide::Mesh* mesh, bool forward, agx::Vec3Vector& newContactShapeTranslates, agx::Vector< size_t >& newTriangleIndices, agx::Vector< size_t >& newEdgeIndices);
    void fixPenetrationProblems();
    void fixMeshPenetrationProblems(NodePtrContainer& meshContacts, WireDistanceCompositeConstraint* ldcc);
    bool addBoxContactNearBoxContact(WireDistanceCompositeConstraint* ldcc, ContactNode* cln, bool beforeNode);
    void fixBoxPenetrationProblems(NodePtrContainer& boxContacts, WireDistanceCompositeConstraint* ldcc);
    void handleNewPossibleEdges();
    bool changeToNextGeometry(agxWire::ContactNode* cn, agx::Vec3 edgeStart, agx::Vec3 edgeEnd, agx::Vec3 pointWorld, const agx::Vec3& depthVector);
    agx::Vec3 moveNodeOutsideGeometry(agx::Vec3 contactPointInside, agx::Vec3 contactPointOnSurface, agx::Vec3 newWP, agx::Vec3 edgeStartRelativeMovementWorld, agx::Vec3 offsetW, bool openAngle, agx::Real wireRadius, agx::Real shortOffsetLength, agx::Real relativeMoveDist = 0.0);
    bool interestingGeometryForContact(ContactNode* cln, agxCollide::Geometry* otherGeometry);
    void getWireAndEdgeInfo(ContactNode* cln, agx::Real t, agx::Vec3& oldWP, agx::Vec3& newWP, agx::Vec3& edgeStartW, agx::Vec3& edgeEndW, agx::Vec3& wireStartW, agx::Vec3& wireEndW, agx::Vec3& moveDirW, agx::Vec3& offsetW, agx::Vec3& edgeW, agx::Vec3&  edgeStart, agx::Vec3& edgeEnd, agx::Vec3& wireStart, agx::Vec3& wireEnd, agx::Vec3& moveDir, agx::Vec3& offset, agx::Vec3& edge, agx::Real& wireRadius, agx::Real& edgeLength);
    bool analyzeGeometriesSimple(ContactNode* cln, const agxCollide::GeometryRefVector& geometries, agx::Real& t, ContactNodePlacementData& data);
    bool analyzeGeometries(ContactNode* cln, const agxCollide::GeometryRefVector& geometries, agx::Real& t, ContactNodePlacementData& data, bool allowStartOfWireInsideGeometry/* = false*/);

    bool rayCastToFindNextGeometries(ContactNode* cn, const agxCollide::GeometryRefVector& geometries, ContactNodePlacementData& data, agx::Real& t, bool advanced);
    agx::Vec3 contactBendForce(WireDistanceCompositeConstraint* ldcc, Node* node1, Node* node2, Node* node3, agx::Vec3 edgeW);

    void getData(const agx::Vector<NodeIterator>& movableContactsOnOneMassless, WireDistanceCompositeConstraint* ldcc, BodyFixedNode* nextBfn,
      agx::Vec3Vector& positions, agx::Vec3Vector& edgeStartPoints, agx::Vec3Vector& edgeEndPoints, agx::RealVector& velocities,
      agx::RealVector& normalForces, agx::Real& maxTension, const agx::Real wireRestLength, const agx::Vec3 beginPos);
    agx::Real calculateNormalForces(WireDistanceCompositeConstraint* ldcc, const agx::Real wireTension, const agx::Real restLength, const agx::Vec3Vector& positions, const agx::Vec3Vector& normalEdges, agx::RealVector& normalForces);

    void updateNormalForces(agxWire::WireDistanceCompositeConstraint* wireConstraint, agx::List<NodeIterator>& contactNodeIterators);

    bool moveEdgeContactNode(WireDistanceCompositeConstraint* ldcc, ContactNode* cln, agx::Real maxLengthDifference, bool updatePreviousPosition = false, bool ignoreFrictionAndMaxmovement = false, agx::Vec3 newShapePosition = agx::Vec3(0, 0, 0));
    agx::Real changeMovementSoThatTheWireIsNotShorterThanRestlength(agx::Real t, agx::Real to, agx::Real oldL, agx::Real newL, agx::Real midL, agx::Real violation, const agx::Vec3& av, const agx::Vec3& bv, const agx::Vec3& edge_start, const agx::Vec3& edge_vector);
    void moveContact(ContactNode* cln, const agx::Vec3& edgeStart, const agx::Vec3& edgeVector, agx::Real t);
    void clearStuckNodes() { m_stuckNodes.clear(); }
    void setIfStuckNode(ContactNode*cln, const agx::Vec3& eStartNewWorld, const agx::Vec3& eEndNewWorld, const agx::Vec3& eStartOldWorld, const agx::Vec3& eEndOldWorld, const agx::Vec3& clnPosWorld, const agx::Vec3& newGeomCenter, const agx::Vec3& oldGeomCenter, agx::Real sensorRadius, agxCollide::Geometry* nextGeometry);
    void unStuck(ContactNode*cln);
    void setStuckNode(ContactNode* cn);
    void setIfStuckNode(Node* cln, bool nextIsPossible) { m_stuckNodes.insert(cln, nextIsPossible); }
    void resetPreviousPositions(agx::List<NodeIterator>& contacts);
    bool saveContactNewPosition(ContactNode*cln, const agx::Vec3& eStart, const agx::Vec3& eEnd, agx::Real t);
    agxSDK::LineContactPosition& getSavedNewContactPosition(ContactNode* cln);
    void removeSavedNewContactPosition(ContactNode* cln);
    void clearContactsNewPosition() { m_contactToNewPosition.clear(); }
    bool hasUsableNextGeometry(ContactNode* cln);
    void removeContactsNearSourceDrain(WireDistanceCompositeConstraint* ldcc);

    /**
    Handles sensor - geometry collisions. Will call either sensorCylinder or sensorBox
    returns
    \param edge               - shapeEdge on geometry (including offset)
    \param shapeContactPoint  - contact point (including offset)
    \param startMovementRange - start of moving range (including offset)
    \param endMovementRange   - end of moving range (including offset)
    \param shapeEdgeOffset    - offset from geometry surface to edge
    */
    bool handleSensor(ContactNode* cln, agxCollide::Geometry* geometry, agxCollide::GeometryContact* geometryContact, agx::Vec3& edge, agx::Vec3& shapeContactPoint, agx::Vec3& startMovementRange, agx::Vec3& endMovementRange, agx::Vec3& shapeEdgeOffset);

    /**
    Handles sensor - geometry collisions. Will call either sensorCylinder or sensorBox
    returns
    \param edge               - shapeEdge on geometry (including offset)
    \param shapeContactPoint  - contact point (including offset)
    \param startMovementRange - start of moving range (including offset)
    \param endMovementRange   - end of moving range (including offset)
    \param shapeEdgeOffset    - offset from geometry surface to edge
    */
    bool handleSensor(ContactNode* cln, agxCollide::Geometry* geometry, agx::Vec3 contactPointWorld, agx::Vec3& edge, agx::Vec3& shapeContactPoint, agx::Vec3& startMovementRange, agx::Vec3& endMovementRange, agx::Vec3& shapeEdgeOffset);

    /**
    Internal method.
    */
    void addMeshInfoToContact(const agxCollide::Mesh* meshShape, ContactNode* cln, size_t triangleIndex, size_t edgeIndex);

    /**
    Creates a new contact wire node given the geometry, shape position and the edge.
    \param geometry - the geometry the contact wire nodes should belong to
    \param shapePosition - the position of the wire node relative the shape
    \param shapeEdge - the edge given in shape coordinates
    \return a new contact wire node
    */
    ContactNode* createContactNode(agxCollide::Geometry* geometry, const agx::Vec3& shapePosition, const agx::Vec3& shapeEdge, agx::Vec3 edgeStart, agx::Vec3 edgeEnd, agx::Vec3 shapeEdgeOffset = agx::Vec3(0, 0, 0), size_t triangleIndex = 0, size_t edgeIndex = 0) const;

    /**
    Goes from begin to end and collects the nodes in between.
    \param begin - start iterator
    \param end - end iterator
    \param nodes - container to fill with wire nodes
    */
    void getUnregistredNodes(NodeIterator begin, NodeIterator end, NodePtrContainer& nodes) const;

    /**
    \return true if the two contact wire nodes are on the same edge - otherwise false
    */
    bool isOnSameEdge(const ContactNode* cln1, const ContactNode* cln2) const;

    /**
    Check so that two contact nodes after each other has correct edge vector pattern. This method can in some cases add
    contact nodes to the lists.
    \return number of nodes added (ret_val > 0) or removed (ret_val < 0)
    */
    int validateEdges(const agx::AffineMatrix4x4& invShapeTransform, const agx::Vec3& halfExtent, const Node* geomBeginNode, const Node* geomEndNode, NodePtrContainer& newContacts, NodeRefNodePtrListIterRealVector& acceptedNodes) const;


    /**
    Handles wire geometry against cylinder contacts.
    \return true since geometry contact should be REMOVE_CONTACT_IMMEDIATELY
    */
    bool handleCylinder(agxCollide::Geometry* geometryCylinder, agxCollide::GeometryContact* geometryContact);


    /**
    Handles wire geometry against mesh contacts.
    \return true since geometry contact should be REMOVE_CONTACT_IMMEDIATELY
    */
    bool handleMesh(agxCollide::Geometry* geometryMesh, agxCollide::GeometryContact* geometryContact);

    /**
    Handles wire geometry against box contacts.
    \return true since geometry contact should be REMOVE_CONTACT_IMMEDIATELY
    */
    bool handleBox(agxCollide::Geometry* geometryBox, agxCollide::GeometryContact* geometryContact);


    /**
    Handle sensorSphere collision with box
    This will handle a geometryContact for a sensorSphere which is following the ContactWireNode cln.
    The purpose is to find a proper contact position for cln on a box geometry
    */
    bool handleSensorBox(ContactNode* cln, agxCollide::Geometry* geometryBox, agx::Vec3 contactPointWorld, agx::Vec3& edge, agx::Vec3& shapeContactPoint, agx::Vec3& startMovementRange, agx::Vec3& endMovementRange, agx::Vec3& shapeEdgeOffset);

    /**
    Handle sensorSphere collision with box
    This will handle a geometryContact for a sensorSphere which is following the ContactWireNode cln.
    The purpose is to find a proper contact position for cln on a box geometry
    */
    bool handleSensorCylinder(ContactNode* cln, agxCollide::Geometry* geometryCylinder, agx::Vec3 contactPointWorld, agx::Vec3& edge, agx::Vec3& shapeContactPoint, agx::Vec3& startMovementRange, agx::Vec3& endMovementRange, agx::Vec3& shapeEdgeOffset);


    bool findNextTriangleAndEdge(const ContactNode* cln, const agxCollide::Mesh* meshShape, const size_t triangleIndex, const size_t vertexIndex, const agx::Vec3& lineStartW, const agx::Vec3& lineEndW, agx::Vec3& edge1StartW, agx::Vec3& edge1EndW, size_t& tempNewTriangleIndex, size_t& tempEdgeLocalIndex);

    /**
    Will validate a contact point in world coordinates given the wire geometry and a threshold. If the contact point in
    wire shape/geometry coordinates is too close (threshold) to one of the ends, this method returns false.
    \param wireGeometry - current wire geometry (cylinder or capsule) with shape transform == nullptr
    \param point - contact point from a geometry contact
    \param threshold - some value k (bool valid = point_in_geometry_coordinates.length - wire_geometry_length/2 > threshold)
    \return true if the point is valid - otherwise false
    */
    bool validatePoint(const agxCollide::Geometry* wireGeometry, const agx::Vec3& point, agx::Real threshold) const;

    /**
    Will validate a contact point in world coordinates given a list of nodes. If the contact point is too close to another node
    in the list, this method will return false.
    \param nodes - list of nodes
    \param point - calculated contact point in world coordinates
    \param threshold - some value k, in meters, which defines the minimum distance two contact nodes may appear
    \return true if this point is valid - otherwise false
    */
    bool validatePoint(const NodePtrContainer& nodes, const agx::Vec3& point, agx::Real threshold) const;

    /**
    Check the angles for invalid new contact wire nodes. When in contact with an object/shape, the angle between a vector
    from the center to the contact point and the neighboring nodes should not exceed 180 degrees. If halfExtent != 0,0,0
    an additional wire<->box intersection test is made.
    \return the number of contact nodes removed from accepted nodes list
    */
    int validateAngles(const agx::AffineMatrix4x4& invShapeTransform, const agx::Vec3& shapeWireGeomBeginPos, const agx::Vec3& shapeWireGeomEndPos, NodePtrContainer& newContacts, NodeRefNodePtrListIterRealVector& acceptedNodes, const agx::Vec3& halfExtent = agx::Vec3(), agx::Real threshold = 0) const;

    /**
    Checks so that the edges from the new contacts have been passed by the wire segment defined by geomBeginNode and geomEndNode.
    \param boxGeometry - body that belongs to the box (may be null)
    \param invShapeTransform - transform world->shape
    \param geomBeginNode - begin wire node that defines the wire geometry currently handled
    \param geomEndNode - end wire node that defines the wire geometry currently handled
    \param numTimeStepsBackToEvaluateFrom - number of time steps back to where to start looking for edge passings (3 works)
    \param maxDepth - if the speed toward the edge is small, this edge passings test will not remove the contact node UNLESS the penetration depth is larger than this parameter
    \param timeStep - the time step
    \param newContacts - new valid contacts found
    \param acceptedNodes - new accepted contacts found
    \return the number of contact nodes removed from newContacts and acceptedNodes
    */
    int validateEdgePassings(const agxCollide::Geometry* boxGeometry, const agx::AffineMatrix4x4& invShapeTransform, const Node* geomBeginNode, const Node* geomEndNode, int numTimeStepsBackToEvaluateFrom, agx::Real maxDepth, agx::Real timeStep, NodePtrContainer& newContacts, NodeRefNodePtrListIterRealVector& acceptedNodes) const;

    /**
    Validate the contact normal (in shape coordinates) for cylinder contacts. The return vector is the correct normal to be used.
    \param normal - contact normal in shape coordinates
    \param invShapeTransform - shape transform to take a point from world to shape (p_world * invShapeTransform = p_shape)
    \param geomBeginNode - begin wire node that defines the wire geometry currently handled
    \param geomEndNode - end wire node that defines the wire geometry currently handled
    \param timeStep - current time step
    \return the correct normal, note that if the length of the normal is zero, the contact is invalid
    */
    agx::Vec3 validateNormal(const agx::Vec3& normal, const agx::AffineMatrix4x4& invShapeTransform, const Node* geomBeginNode, const Node* geomEndNode, agx::Real timeStep) const;

    int testCircleContacts(size_t i, const agxCollide::Geometry* geometryCylinder, const agxCollide::Cylinder* cylinder, const agx::Vec3& shapeWireGeomBeginPos, const agx::Vec3& shapeWireGeomEndPos, const agx::Vec3& worldWireGeomBeginPos, const agx::Vec3& worldWireGeomEndPos, agx::Real circlePos, NodePtrContainer& newContacts, NodeRefNodePtrListIterRealVector& acceptedNodes) const;

    /**
    Calculates the contact nodes local translate given a box shape, contact point in shape coordinates, start- and end point
    of the wire geometry in the box shape coordinates.
    \param box - box shape
    \param localContactPoint - the contact point given in shape coordinates
    \param localSegmentBegin - start point of the wire geometry in shape coordinates (box shape coordinates)
    \param localSegmentEnd - end point of the wire geometry in shape coordinates (box shape coordinates)
    \param retLocalCorrectedContactPoint - return value 1, the corrected contact point given in shape coordinates
    \param retEdge - return value 2, the edge, defined to go from minus to plus in shape coordinates
    \param retEdgeStart - return value 3, vector from the shapes center to the point where the edge starts
    \param retEdgeEnd - return value 4, vector from the shapes center to the point where the edge ends
    */
    void calculateContactPoint(const agxCollide::Box* box, const agx::Vec3& localContactPoint, const agx::Vec3& localSegmentBegin, const agx::Vec3& localSegmentEnd, agx::Vec3& retLocalCorrectedContactPoint, agx::Vec3& retEdge, agx::Vec3& retEdgeStart, agx::Vec3& retEdgeEnd) const;
    void calculateContactPoint(const agxCollide::Box* box, const agx::Vec3& localContactPoint, agx::Vec3& retLocalCorrectedContactPoint, agx::Vec3& retEdge, agx::Vec3& retEdgeStart, agx::Vec3& retEdgeEnd) const;

    /**
    Calculates the contact nodes local translate along the cylinder axis given a cylinder shape, contact point in shape coordinates, start- and end point
    of the wire geometry in the cylinders shape coordinates.
    \param cylinder - cylinder shape
    \param localContactPoint - the contact point given in shape coordinates
    \param localSegmentBegin - start point of the wire geometry in shape coordinates (cylinder shape coordinates)
    \param localSegmentEnd - end point of the wire geometry in shape coordinates (cylinder shape coordinates)
    \param retLocalCorrectedContactPoint - return value 1, the corrected contact point given in shape coordinates
    \param retEdge - return value 2, the edge, defined to go from minus to plus in shape coordinates
    \param retEdgeStart - return value 3, vector from the shapes center to the point where the edge starts
    \param retEdgeEnd - return value 4, vector from the shapes center to the point where the edge ends
    */
    void calculateContactPoint(const agxCollide::Cylinder* cylinder, const agx::Vec3& localContactPoint, const agx::Vec3& localContactNormal, const agx::Vec3& localSegmentBegin, const agx::Vec3& localSegmentEnd, agx::Vec3& retLocalCorrectedContactPoint, agx::Vec3& retEdge, agx::Vec3& retEdgeStart, agx::Vec3& retEdgeEnd) const;
    void calculateContactPoint(const agxCollide::Cylinder* cylinder, const agx::Vec3& localContactPoint, const agx::Vec3& localContactNormal, agx::Vec3& retLocalCorrectedContactPoint, agx::Vec3& retEdge, agx::Vec3& retEdgeStart, agx::Vec3& retEdgeEnd) const;

    void calculateCircleContactPointAndEdge(const agxCollide::Cylinder* cylinderShape, agx::Vec3& Q1, agx::Vec3& shapeLocalEdgeStart, agx::Vec3& shapeLocalEdgeEnd, agx::Vec3& shapeLocalEdge, agx::Vec3& shapeEdgeOffset, agx::Real planePos, bool validateAgainstEnds = false, const agx::Vec3& shapeGeomBeginPos = agx::Vec3(), const agx::Vec3& shapeGeomEndPos = agx::Vec3()) const;

    /**
    Inserts contactNode in the local node list nodes (incl. wireGeomBeginPos and wireGeomEndPos) given the shortest
    distance route.
    \param nodes - local node list
    \param contactNode - new contact node about to be inserted
    \param wireGeomBeginPos - wire geometry begin world position
    \param wireGeomEndPos - wire geometry end world position
    \param closeThreshold - the closest the new contact may be to another node
    \return iterator in nodes where contactNode were inserted
    */
    NodePtrContainer::iterator insertShortestDistance(NodePtrContainer& nodes, ContactNode* contactNode, const agx::Vec3& wireGeomBeginPos, const agx::Vec3& wireGeomEndPos, agx::Real closeThreshold) const;

    /**
    Creates a contact wire node and inserts it in newContacts and acceptedNodes. All points/vectors must be given in shape coordinates.
    \returns true on success
    */
    bool createAndInsertNode(NodePtrContainer& newContacts, size_t indexInAcceptedNodes, NodeRefNodePtrListIterRealVector& acceptedNodes, const agxCollide::Geometry* geometry, const agx::Vec3& worldWireGeomBeginPos, const agx::Vec3& worldWireGeomEndPos, const agx::Vec3& contactPoint, const agx::Vec3& edgeStart, const agx::Vec3& edgeEnd, const agx::Vec3& edge, const agx::Vec3& shapeEdgeOffset, agx::Real closeThreshold) const;

    /**
    Given new accepted contacts, if they are not in the wire, this method will insert the nodes.
    \param wireGeomBeginIt - wire node iterator to geometry begin
    \param newContacts - new contacts (contacts inserted earlier this time step)
    \param wire - the wire constraint
    \param acceptedNodes - new contact nodes that has been accepted for insert
    \param geometryContact - the current geometry contact
    */
    void insertValidNodesInWire(NodeIterator wireGeomBeginIt, NodePtrContainer& newContacts, WireDistanceCompositeConstraint* wire, const NodeRefNodePtrListIterRealVector& acceptedNodes, const agxCollide::GeometryContact* geometryContact);


    /**
    Will remove/erase the node of index 'index' (newContacts.erase and acceptedNodes[index].first = nullptr) with an invalid flag.
    All local removes MUST go through this method (convenient for debugging and debug rendering).
    \param newContacts - new valid contacts found
    \param acceptedNodes - new accepted contacts found
    \param index - index in acceptedNodes to the node that is rejected
    */
    void removeLocalNode(NodePtrContainer& newContacts, NodeRefNodePtrListIterRealVector& acceptedNodes, size_t index) const;

    inline bool hasSavedNewContactPosition(ContactNode* cln)
    {
      return m_contactToNewPosition.contains(cln);
    }


#include <HashImplementationSwitcher.h>
#if HASH_FOR_WIRE == HASH_NEW
    using GeometryContactWireNodeHash = agx::LinearProbingHashTable<agxCollide::Geometry*, ContactNode*>;
    using ContactWireNodePtrContactPositionTable = agx::LinearProbingHashTable<ContactNode*, agxSDK::LineContactPosition>;
#elif HASH_FOR_WIRE == HASH_OLD
    typedef agx::QuadraticProbingHashTable<agxCollide::Geometry*, ContactNode*> GeometryContactWireNodeHash;
    typedef agx::QuadraticProbingHashTable<ContactNode*, agxSDK::LineContactPosition> ContactWireNodePtrContactPositionTable;
#else
#error
#endif

    agx::ref_ptr< WireSensorController > m_sensorController;
    WireContactGeometryTransferController* m_contactGeometryTransferController;
    EdgeTransferController* m_edgeTransferController;

#if HASH_FOR_WIRE == HASH_NEW
    agx::LinearProbingHashTable<Node*, bool> m_stuckNodes;
#elif HASH_FOR_WIRE == HASH_OLD
    agx::QuadraticProbingHashTable<Node*, bool> m_stuckNodes; //bool is if it is possible to switch to nextGeometry if validateContact says we could try
#else
#error
#endif

    ContactWireNodePtrContactPositionTable m_contactToNewPosition;

    agxCollide::Geometry *initLineGeometry(const agx::Vec3& start, const agx::Vec3& end);

    agxCollide::GeometryRef m_lineGeometry;
    agxCollide::Line *m_lineShape;
  };

  typedef agx::ref_ptr< WireOldContactController > WireOldContactControllerRef;

  AGX_FORCE_INLINE bool WireSensorController::match(agx::Physics::GeometryPtr geometry) const
  {
    agx::Physics::Geometry::ShapePtr shape = geometry.shape();
    return shape && shape.type() == agxCollide::Shape::SPHERE && geometry.model()->hasGroup(m_wireId);
  }
}

DOXYGEN_END_INTERNAL_BLOCK()
