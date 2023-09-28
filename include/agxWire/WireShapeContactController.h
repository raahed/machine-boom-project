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

#include <agxWire/WireContactController.h>
#include <agx/Vec3.h>
#include <agxWire/Node.h>
#include <agxWire/ShapeContactNode.h>
#include <agxWire/WireUtils.h>
#include <agxWire/ShapeContactColliderUtils.h>
#include <agxWire/WireContactDirectSolver.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxCollide
{
  class Shape;
  class GeometryContactPtr;
  class Geometry;
}

namespace agxWire
{
  class Wire;
  class WireDistanceCompositeConstraint;

  typedef agx::Vector<agxCollide::GeometryContact> GeometryContactVector;
  typedef agx::HashVector< Node*, GeometryContactVector > NodePtrGeometryContactVectorTable;
  typedef agx::Vector< std::pair< ShapeContactNode*, agx::Real> > ShapeContactRealPairVector;
  typedef agx::HashVector<ShapeContactNodeRef, ShapeContactNodeRef> NodeMergePairHashVector;

  typedef agx::HashSet<agxCollide::Shape*> ShapeSet;
  typedef agx::HashTable<ShapeContactNode*, ShapeSet > NodeShapeSetVector;
  typedef agx::List< agxWire::NodeConstIterator > NodeIteratorList;

  class AGXPHYSICS_EXPORT WireShapeContactController : public agxWire::WireContactController
  {
    public:
      WireShapeContactController( Wire* wire );

      static BodyFixedNode* changeToBodyFixed(ShapeContactNode* cn, Wire* wire);
      static ShapeContactNode* changeToShapeContact(Node* node, Wire* wire, agxCollide::Shape* shape, const agx::Vec3& cPoint,
                                                    const agx::Vec3& cNormal, const agx::Real& cDepth, const agx::Bool removeOtherNode);

      void removeLumpedNodesTooClose();

      void findExtraLength();

      void analyzePossibleBodyFixReplacement();

      virtual agx::Bool handleGeometryContact(agxCollide::GeometryContact* geometryContact, agxCollide::Geometry* otherGeometry);

      virtual void preCollideUpdate();
      virtual void preUpdate();
      virtual void postUpdate();

      struct SurfacePointData
      {
        SurfacePointData( agx::Vec3 surfacePoint,
                          agx::Vec3 normal,
                          agxCollide::Shape* shape,
                          agxCollide::GeometryContact* geometryContact,
                          agx::UInt pointIndex,
                          agx::Bool isFaceContact )
          : point( surfacePoint ),
            normal( normal ),
            shape( shape ),
            geometryContact( geometryContact ),
            pointIndex( pointIndex ),
            isFaceContact( isFaceContact )
        {
        }

        agx::Vec3 point;
        agx::Vec3 normal;
        agxCollide::Shape* shape;
        agxCollide::GeometryContact* geometryContact;
        agx::UInt pointIndex;
        agx::Bool isFaceContact;
      };
      typedef agx::Vector< SurfacePointData > SurfacePointVector;
      typedef agx::Vector< agxCollide::MeshRef > MeshVector;

    protected:
      virtual ~WireShapeContactController();

      void movePenetratingContactNodesBackAlongEdges(NodeShapeSetVector& nodesToShapeList);
      void resetEdgeClamping(const ContactNodeDataList& nodes) const;
      /**
      \returns true, if the node can be kept.
      */
      agx::Bool findSurfacePosition( ShapeContactNode* cn ) const;
      void removeUnwantedNodes(const ContactNodeDataList& nodes);
      void findSurfacePositions(ContactNodeDataList& nodes) const;
      void removeNodesByAngle(ContactNodeDataList& nodes, const agx::Bool ignoreClampedPosition = false);
      void chooseAndClampNodeEdges(ContactNodeDataList& nodes, NodeShapeSetVector& nodesToShapes);


      agx::Real calculateAcceptedWireOverlap(const agx::Real& wireRadius) const;
      agx::Bool getWorldPositionNormalAndEdgeOfNeighbour(NodeConstIterator it, agx::Vec3& pos, agx::Vec3& normal, agx::Vec3& edge, const agx::Bool before) const;


      agx::Bool evaluateMiddleContact(const ShapeContactNode* prevContact, const ShapeContactNode* middleContact, const ShapeContactNode* nextContact, agx::Bool& dueToPrev) const;
      agx::Bool shouldContactBecomeBodyFixed(ContactNodeDataList::iterator nodeDataIterator) const;
      void analyzeContactState(ContactNodeDataList::iterator nodeDataIterator, ContactNodeDataList& nodeDataList) const;
      void validateContact(ContactNodeDataList::iterator nodeDataIterator, const agx::Bool ignoreClampedPosition = false) const;
      agx::Bool evaluateCloseContact( const ShapeContactNode* closeContact,
                                      const agx::Vec3& nodePositionWorld,
                                      const ShapeCurvature& worldCurvature,
                                      agx::Real wireRadius ) const;
      agx::Bool evaluateCloseContact( const ShapeContactNode* closeContact,
                                      const agx::Vec3& nodePositionWorld,
                                      const ShapeCurvature& worldCurvature,
                                      agx::Real wireRadius,
                                      agx::Real& dotNormals ) const;

      agxWire::ShapeContactNode* evaluateAndInsertNewContact( agxCollide::Geometry* wireGeom, const SurfacePointData& surfacePointData, const ContactNodePtrVector& newNodes);

      void createNewContactNodes(NodeShapeSetVector& nodesToShapeList);

      void collectPossibleEdges(ShapeContactNode* scn, agxCollide::Shape* shape, ShapeContactEdgeVector& possibleEdges);

      agx::Bool clampEdgeToShape(ShapeContactNode* scn, ShapeContactEdge& edge, agxCollide::Shape* shape);

      agx::Bool clampEdgeToPlane(ShapeContactNode* cn, ShapeContactEdge& edge, const agx::Vec3& point, const agx::Vec3& clampNormal, const agx::Bool moveNode = false);

      void chooseAndClampNodeEdge(ContactNodeData& cnData, const NodeShapeSetVector& nodesToShapes);

      agx::Bool mergeCrossingEdgeNodes(const agx::Bool forwardInTime);
      agx::Bool mergeCloseNodes();
      agx::Bool mergeNodes(NodeMergePairHashVector& nodesToMerge);

      agx::Bool findWhichNodeToRemoveByMerge(ShapeContactNode* firstContact, ShapeContactNode* secondContact, const Node* prevNode, const Node* nextNode) const;

      agx::Bool isClamped(ShapeContactNode* cn);

      agx::Bool merge(ShapeContactNode* a, ShapeContactNode* b, Node* prevNode, Node* nextNode);

      void fixFreeEndContactPassings(WireDistanceCompositeConstraint* ldcc);
      agx::Bool nearClampPosition(const ShapeContactNode* scn) const;
      agx::Bool atClampPosition(const ShapeContactNode* scn) const;

      /**
      \returns true if the edge is more suited than the current edge
      */
      agx::Bool considerNewEdge(const ShapeContactNode* scn, const ShapeContactEdge& edge) const;

      /**
      Solve for new node velocities.
      \param integrate - will move the nodes if true.
                         If false, the delta length of the node system will be stored in the
                         m_extraLength of the shape contact nodes.
      */
      void getNewNodeVelocities(agx::Bool integrate);

      void getAllContactNodes(WireDistanceCompositeConstraint* ldcc, agx::List<NodeIterator>& allContacts);
      void getMovableContactNodes(const agx::List<NodeIterator>& allContacts, agx::List<NodeIterator>& contacts, agx::List<NodeIterator>& noMovementContacts);

      void analyzeCollision(NodeShapeSetVector& nodesToShapeList, agxCollide::GeometryContact& gc, agxCollide::Geometry* wireGeom, agxCollide::Geometry* otherGeom, SurfacePointVector& contactInfo );
      void findRouteAroundMesh( agxCollide::Geometry* wireGeom, const agxCollide::GeometryContact& gc, const agx::Vector< agxCollide::MeshRef >& meshes);

      enum class EvaluationResult
      {
        IGNORE_POINT,
        CREATE_NODE,
        CREATE_GEOMETRY_CONTACT_PREV,
        CREATE_GEOMETRY_CONTACT_NEXT
      };

      bool verifyBothAreNeeded(const ShapeContactNode* otherContact,
                               const Node* behindOtherContact, 
                               const Node* behindNewNode, 
                               const agx::Vec3& newWorldPosition,
                               const agx::Real& wireRadius) const;

      EvaluationResult evaluate( const agxCollide::Shape* shape,
                                 const agx::Vec3& nodePositionWorld,
                                 const ShapeCurvature& worldCurvature,
                                 const NodeContainer& wireNodes,
                                 NodeIterator prevIt,
                                 NodeIterator nextIt,
                                 const agx::Real& wireRadius,
                                 const agx::Vector<ShapeContactNode*>& newNodes,
                                 const agx::Bool favorLumpSphereContacts = true ) const;
    protected:
      NodePtrGeometryContactVectorTable m_nodeGeometryContactTable;
      agx::ref_ptr<WireContactDirectSolver> m_wcds;
      agxCollide::LocalGeometryContactVector m_localGeometryContacts;
  };

  typedef agx::ref_ptr< WireShapeContactController > WireShapeContactControllerRef;
}

DOXYGEN_END_INTERNAL_BLOCK()
