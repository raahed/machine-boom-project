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

#include <agx/agxPhysics_export.h>
#include <agx/Math.h>
#include <agxCollide/Mesh.h>
#include <agxCollide/Contacts.h>
#include <agxSDK/LineCollisionUtils.h>
#include <agxCollide/BasicPrimitiveTests.h>
#include <agxCollide/TriangleVoronoiRegion.h>

namespace agxWire
{
  class WireDistanceCompositeConstraint;

#define shape_to_world_transform( geom ) \
  ((*geom->getShapes().begin())->getLocalTransform() * geom->getTransform())

#define world_to_shape_transform( geom ) \
  (shape_to_world_transform( geom ).inverse())

  // Temporary: I don't like this.

  /**
  Transforms a vector given in world coordinates to shape coordinates of geometry g.
  \param v - vector in world coordinates
  \param g - geometry with one shape
  \return vector in shape coordinates
  */
  static inline agx::Vec3 transformVectorToShape( const agx::Vec3& v, const agxCollide::Geometry* g )
  {
    agxAssert( g && g->getShapes().size() == 1 && v.isValid() );
    return g ? world_to_shape_transform( g ).transform3x3( v ) : v;
  }

  /**
  Transforms a vector given in shape coordinates to world coordinates given geometry g.
  \param v - vector in shape coordinates
  \param g - geometry with one shape
  \return vector in world coordinates
  */
  static inline agx::Vec3 transformVectorToWorld( const agx::Vec3& v, const agxCollide::Geometry* g )
  {
    agxAssert( g && g->getShapes().size() == 1 && v.isValid() );
    return g ? shape_to_world_transform( g ).transform3x3( v ) : v;
  }

  /**
  Transforms a point given in world coordinates to shape coordinates of geometry g.
  \param p - point in world
  \param g - geometry with one shape
  \return point in shape coordinates
  */
  static inline agx::Vec3 transformPointToShape( const agx::Vec3& p, const agxCollide::Geometry* g )
  {
    agxAssert( g && g->getShapes().size() == 1 && p.isValid() );
    return g? (p * world_to_shape_transform( g )) : p;
  }

  /**
  Transforms a point given in shape coordinates to world given geometry g.
  \param p - point in shape coordinate system
  \param g - geometry with one shape
  \return point in world coordinates
  */
  static inline agx::Vec3 transformPointToWorld( const agx::Vec3& p, const agxCollide::Geometry* g )
  {
    agxAssert( g && g->getShapes().size() == 1 && p.isValid() );
    return g ? (p * shape_to_world_transform( g )) : p;
  }

#undef shape_to_world_transform
#undef world_to_shape_transform

  DOXYGEN_START_INTERNAL_BLOCK()

  static inline bool overlappingRange( agx::Real range1Start,agx::Real range1End,agx::Real range2Start,agx::Real range2End )
  {
    return ( agx::leq(range1Start, range2End ) && agx::geq(range1End, range2Start ) );
  }

  DOXYGEN_END_INTERNAL_BLOCK()


  /**
  Primitive function to find begin iterator for utility functions below. Optimal solution is to use geometry iterators.
  \param container - container defining the wire
  \param nodeMask - nodes to look for
  \return iterator to the first hit on nodeMask from begin - container.end() if nothing was found
  */
  template < typename IteratorCompatibleContainer >
  typename IteratorCompatibleContainer::const_iterator findBeginIterator( const IteratorCompatibleContainer& container, int nodeMask )
  {
    if ( container.empty() )
      return container.end();

    typename IteratorCompatibleContainer::const_iterator i = container.begin();
    while ( i != container.end() && ((*i)->getType() & nodeMask) == 0 )
      ++i;
    return i;
  }

  /**
  Primitive function to find end iterator for utility functions below. Optimal solution is to use geometry iterators.
  \param container - container defining the wire
  \param nodeMask - nodes to look for
  \return iterator to the first hit on nodeMask from end - container.begin() if nothing was found
  */
  template < typename IteratorCompatibleContainer >
  typename IteratorCompatibleContainer::const_iterator findEndIterator( const IteratorCompatibleContainer& container, int nodeMask )
  {
    // Empty, begin == end.
    if ( container.empty() )
      return container.end();

    typename IteratorCompatibleContainer::const_iterator i = container.end();
    --i;
    while ( i != container.begin() && ((*i)->getType() & nodeMask) == 0 )
      --i;
    return i;
  }

  /**
  Search vector/list container defining a wire for the 3D point -> 1D wire projection. I.e., given any 3D point,
  this function will find the distance along the wire closest to \p point.
  \param pulledInBegin - amount of wire hauled in, in for example a winch at the beginning of the wire
  \param container - iterator compatible container (like std::list/vector, agx::Vector etc) with wire node pointers of reference pointers defining a wire
  \param point - any 3D point
  \param beginIt - where to begin in \p container
  \param endIt - where to end in \p container (like while (it != endIt) ...)
  \param shortestDistanceSquared - Returns the squared shortest distance.
  \return distance along wire closest to the 3D point
  */
  template < typename IteratorCompatibleContainer >
  agx::Real findDistanceFromStartGivenPoint( agx::Real pulledInBegin, const IteratorCompatibleContainer& container, const agx::Vec3& point, typename IteratorCompatibleContainer::const_iterator beginIt, typename IteratorCompatibleContainer::const_iterator endIt, agx::Real& shortestDistanceSquared )
  {
    agx::Real bestDistance = 0;
    agx::Real currentDistance = 0;
    shortestDistanceSquared = agx::Infinity;

    // Container is empty, undefined.
    if ( container.empty() || beginIt == endIt )
      return 0;

    typename IteratorCompatibleContainer::const_iterator i = beginIt;
    agxAssert( i != container.end() );
    typename IteratorCompatibleContainer::const_iterator end = endIt;
    agxAssert( end != container.begin() );

    agx::Vec3 startPoint = (*i)->getFrame()->getWorldPosition();
    agx::Vec3 endPoint, direction;
    ++i;
    for ( ; i != end; ++i ) {
      endPoint     = (*i)->getFrame()->getWorldPosition();

      direction    = endPoint - startPoint;
      agx::Real t;
      if (agx::equalsZero(direction.length2()))
        t = 0;
      else {
        t  = (( point - startPoint ) * direction) / direction.length2();
        t  = agx::clamp( t, agx::Real( 0 ), agx::Real( 1 ) );
      }

      startPoint  += direction * t;
      agx::Real d2 = ( startPoint - point ).length2();
      agx::Real dl = direction.length();

      if ( d2 < shortestDistanceSquared ) {
        shortestDistanceSquared = d2;
        bestDistance      = currentDistance + t * dl;
      }
      currentDistance += dl;
      startPoint       = endPoint;
    }

    return bestDistance + pulledInBegin;
  }

  /**
  Search vector/list container defining a wire for the 3D point -> 1D wire projection. I.e., given any 3D point,
  this function will find the distance along the wire closest to \p point.
  \param pulledInBegin - amount of wire hauled in, in for example a winch at the beginning of the wire
  \param container - iterator compatible container (like std::list/vector, agx::Vector etc) with wire node pointers of reference pointers defining a wire
  \param point - any 3D point
  \return distance along wire closest to the 3D point
  */
  template < typename IteratorCompatibleContainer >
  agx::Real findDistanceFromStartGivenPoint( agx::Real pulledInBegin, const IteratorCompatibleContainer& container, const agx::Vec3& point )
  {
    agx::Real dummySquaredDistance = 0;
    return findDistanceFromStartGivenPoint( pulledInBegin, container, point, container.begin(), container.end(),dummySquaredDistance );
  }

  /**
  Projects any 3D point onto wire. Finds point on wire (defined by vector/list container) given any point in 3D.
  \param container - iterator compatible container (like std::list/vector, agx::Vector etc) with wire node pointers of reference pointers defining a wire
  \param point - any 3D point
  \param beginIt - where to begin in \p container
  \param endIt - where to end in \p container (like while (it != endIt) ...)
  \param shortestDistanceSquared - The shortest distance squared returned.
  \return a point on the wire closest to \p point
  */
  template < typename IteratorCompatibleContainer >
  agx::Vec3 findPointOnWire( const IteratorCompatibleContainer& container, const agx::Vec3& point, typename IteratorCompatibleContainer::const_iterator beginIt, typename IteratorCompatibleContainer::const_iterator endIt, agx::Real& shortestDistanceSquared )
  {
    if ( container.empty() )
      return agx::Vec3();

    // Find the distance along the wire.
    agx::Real distanceAlongWire = findDistanceFromStartGivenPoint( 0, container, point, beginIt, endIt, shortestDistanceSquared );

    // If the distance is equal to the pulled in length the first point is the correct one.
    if ( agx::equalsZero( distanceAlongWire ) ) {
      return beginIt->get()->getFrame()->getWorldPosition();
    }

    typename IteratorCompatibleContainer::const_iterator i = beginIt;
    typename IteratorCompatibleContainer::const_iterator end = endIt;
    agxAssert( i != container.end() && end != container.begin() );

    agx::Vec3 currPos = (*i)->getFrame()->getWorldPosition();
    ++i;
    agx::Real currDistanceAlongWire = 0;
    for ( ; i != end; ++i ) {
      agx::Vec3 nextPos = (*i)->getFrame()->getWorldPosition();
      agx::Vec3 dir = nextPos - currPos;
      agx::Real length = dir.normalize();
      // If we're about pass distanceAlongWire, calculate the difference and the point wanted:
      // point_wanted = current_position_on_line + dir_to_next_node * ( distance_from_start_to_previous_node + distance_to_next_node - total_distance_to_wanted_point )
      if ( currDistanceAlongWire + length > distanceAlongWire || agx::equivalent( currDistanceAlongWire + length, distanceAlongWire, agx::RealEpsilon ) ) {
        agx::Real d = distanceAlongWire - currDistanceAlongWire;
        return currPos + dir * d;
      }
      currDistanceAlongWire += length;
      currPos = nextPos;
    }
    // Last position is the one wanted.
    return currPos;
  }

  template < typename IteratorCompatibleContainer >
  agx::Vec3 findPointOnWire( const IteratorCompatibleContainer& container, const agx::Vec3& point, typename IteratorCompatibleContainer::const_iterator beginIt, typename IteratorCompatibleContainer::const_iterator endIt, const agx::Vec3 n, agx::Real& shortestDistanceSquared )
  {
    shortestDistanceSquared = agx::RealMax;
    agx::Vec3 currentBest;

    agx::Vec3 normal;
    agx::Vec4 plane;

    // pp = p in plane
    // w1 & w2, two line node points in world coords
    // p1 & p2, same two line node points, but projected onto plane
    // pd = plane direction (p2-p1)
    agx::Vec3 pp, p1,p2, w1,w2, pd;

    agx::Real t, dist;

    normal = n;
    normal.normalize();

    plane = agx::Vec4( normal, 0 );

#define planeProject( pnt ) pnt - ( normal * ( plane * pnt ) )

    pp = planeProject(point);

    if ( container.empty() ) return agx::Vec3();
    // Collect the first node position from the first segments' node list
    w1 = (*(++container.begin()))->getFrame()->getWorldPosition(); //c->m_nodes.front()->getNodeFrame().getWorldPosition();
    p1 = planeProject( w1 );

    typename IteratorCompatibleContainer::const_iterator nodeIter = beginIt;
    typename IteratorCompatibleContainer::const_iterator end = endIt;

    for ( ; nodeIter != end; ++nodeIter ) {
      const Node* currentNode = static_cast< const Node* >( (*nodeIter).get() );
      w2 = currentNode->getFrame()->getWorldPosition();

      p2 = planeProject( w2 );

      pd = p2 - p1;

      t = (( pp - p1 ) * pd ) / pd.length2();
      t = agx::clamp( t, agx::Real(0.0), agx::Real(1.0) );

      // calculate closest point on this part of the line composite ... and distance to point
      p1 += pd * t;

      dist = (p1-pp).length2();

      if ( dist < shortestDistanceSquared )
      {
        shortestDistanceSquared = dist;
        currentBest = w1 + (w2-w1) * t;
      }

      p1 = p2;
      w1 = w2;
    }



#undef planeProject

    return currentBest;
  }

  /**
  Projects any 3D point onto wire. Finds point on wire (defined by vector/list container) given any point in 3D.
  \param container - iterator compatible container (like std::list/vector, agx::Vector etc) with wire node pointers of reference pointers defining a wire
  \param point - any 3D point
  \return a point on the wire closest to \p point
  */
  template < typename IteratorCompatibleContainer >
  agx::Vec3 findPointOnWire( const IteratorCompatibleContainer& container, const agx::Vec3& point )
  {
    agx::Real dummySquaredDistance = 0;
    return findPointOnWire( container, point, container.begin(), container.end(), dummySquaredDistance );
  }

    /**
  Projects any 3D point onto wire. Finds point on wire (defined by vector/list container) given any point in 3D.
  \param container - iterator compatible container (like std::list/vector, agx::Vector etc) with wire node pointers of reference pointers defining a wire
  \param point - any 3D point
  \param n - Normal
  \return a point on the wire closest to \p point
  */
  template < typename IteratorCompatibleContainer >
  agx::Vec3 findPointOnWire( const IteratorCompatibleContainer& container, const agx::Vec3& point, const agx::Vec3 n )
  {
    agx::Real dummySquaredDistance = 0;
    return findPointOnWire( container, point, container.begin(), container.end(), n , dummySquaredDistance );
  }

  /**
  This function is used by getNode, but can be used of some nodes in the beginning or end of the wire should be excluded.
  \param distanceFromStart - current distance from start
  \param begin - begin iterator
  \param end - end iterator
  \param containerEnd - iterator to end of the current container
  */
  template < typename It >
  It getNodeIterator( agx::Real distanceFromStart, It begin, It end, It containerEnd )
  {
    if ( begin == end || begin == containerEnd )
      return containerEnd;

    if ( agx::leq( distanceFromStart, agx::Real( 0 ) ) )
      return begin;

    It next = begin;
    It curr = next++;
    if ( next == containerEnd || next == end )
      return curr;

    agx::Real length = 0;
    while ( next != containerEnd && next != end ) {
      length += ( (*next)->getFrame()->getWorldPosition() - (*curr)->getFrame()->getWorldPosition() ).length();
      if ( length > distanceFromStart )
        return curr;

      curr = next;
      ++next;
    }

    return end;
  }

  /**
  Finds the wire node given current distance from the start of the wire (wire defined to be iterator compatible container).
  \param container - iterator compatible container
  \param distanceFromStart - current distance from start
  \return wire node in container before current_length_along_line > distance_from_start
  */
  template < typename IteratorCompatibleContainer >
  const Node* getNode( const IteratorCompatibleContainer& container, agx::Real distanceFromStart )
  {
    if ( container.empty() )
      return nullptr;

    typename IteratorCompatibleContainer::const_iterator i = getNodeIterator( distanceFromStart, container.begin(), container.end(), container.end() );
    if ( i != container.end() )
      return *i;
    return nullptr;
  }

  /**
  Finds the wire node given current distance from the start of the wire (wire defined to be iterator compatible container).
  \param container - iterator compatible container
  \param distanceFromStart - current distance from start
  \return wire node in container before current_length_along_line > distance_from_start
  */
  template < typename IteratorCompatibleContainer >
  Node* getNode( IteratorCompatibleContainer& container, agx::Real distanceFromStart )
  {
    if ( container.empty() )
      return nullptr;

    typename IteratorCompatibleContainer::iterator i = getNodeIterator( distanceFromStart, container.begin(), container.end(), container.end() );
    if ( i != container.end() )
      return *i;
    return nullptr;
  }

  DOXYGEN_START_INTERNAL_BLOCK()

  //return -1 if non was passed. otherwise 0,1 or 2 (index if edge passed first)
  int AGXPHYSICS_EXPORT validateEdgePassings( const agxCollide::Geometry* meshGeometry,const agxCollide::Mesh* mesh,const size_t triangleIndex, const agx::RigidBody* rb, const agx::Vec3 lineGeometryBegin, const agx::Vec3 lineGeometryEnd, const agx::Vec3 lineGeometryBeginLastNTimeSteps, const agx::Vec3 lineGeometryEndLastNTimeSteps, agx::Vec3 /*contactPointW*/,agx::Real timeStep,agx::Real wireRadius,const agx::Real numTimeStepsBackToEvaluateFrom = 2);

  int closestEdgeToContact(const agxCollide::Mesh* mesh,const size_t triangleIndex,const agx::Vec3 pointInTriangle);


  template <typename T>
  static inline bool findMeshTriangleAndEdge( const agx::Vec3 lineGeometryBegin, const agx::Vec3 lineGeometryEnd,
    const agx::Vec3 lineGeometryBeginLastNTimeSteps, const agx::Vec3 lineGeometryEndLastNTimeSteps ,
    size_t& triangleIndex,size_t& edgeIndex, T* /* geometryContact */,
    const typename T::PointType& contactPoint, const agxCollide::Mesh* meshShape,
    agxCollide::Geometry* geometryMesh, const agx::Real timeStep,
    const agx::Real wireRadius,const agx::Real numTimeStepsBackToEvaluateFrom = 2 )
  {
    agx::UInt32 faceIndex;
    agx::UInt8 faceFeature;
    if (meshShape->getEntity() == contactPoint.shape1()) {
      faceIndex = contactPoint.faceIndex1();
      faceFeature = contactPoint.faceFeature1();
    }
    else {
      agxAssert(meshShape->getEntity() == contactPoint.shape2());
      faceIndex = contactPoint.faceIndex2();
      faceFeature = contactPoint.faceFeature2();
    }


    triangleIndex = faceIndex;
    agxCollide::TriangleVoronoiRegion::Type edgeIndexTVR  = agxCollide::TriangleVoronoiRegion::calculateType(faceFeature);

    if ( triangleIndex == meshShape->getNumTriangles() )
      return false;

    if ( edgeIndexTVR == agxCollide::TriangleVoronoiRegion::EDGE )
      edgeIndex = faceFeature - 3;
    else if ( edgeIndexTVR == agxCollide::TriangleVoronoiRegion::FACE )
    {
      int closestEdgeIndex = validateEdgePassings(geometryMesh,meshShape,triangleIndex,geometryMesh->getRigidBody(),lineGeometryBegin,lineGeometryEnd,lineGeometryBeginLastNTimeSteps,lineGeometryEndLastNTimeSteps,contactPoint.point(),timeStep,wireRadius,numTimeStepsBackToEvaluateFrom );

      if ( closestEdgeIndex < 0 )
      {
        // Find if the wire segment points are on different sides of the face.
        // If they are NOT, we cant accept that the face contact to create an edge contact.
        agx::Vec3 triangleNormalWorld = transformVectorToWorld( meshShape->getTriangle(triangleIndex).getNormal(), geometryMesh );
        agx::Vec3 pointInTriangle = contactPoint.point() + agx::Vec3( contactPoint.normal() ) * contactPoint.depth() * agx::Real(0.5);
        agx::Vec3 toLineBegin = lineGeometryBegin - pointInTriangle;
        agx::Vec3 toLineEnd   = lineGeometryEnd   - pointInTriangle;

        //Both segment points are on the same side of the plane
        agx::Real distToPlaneFromBegin =  (toLineBegin * triangleNormalWorld );
        agx::Real distToPlaneFromEnd   =  (toLineEnd   * triangleNormalWorld );
        if ( distToPlaneFromBegin * distToPlaneFromEnd > 0 )
          return false;

        // The two points seem to be on the same side
        // Find which edge that is closest
        agx::Vec3 pointInTriangleLocal = transformPointToShape( pointInTriangle, geometryMesh );
        if ( !agx::equivalent( meshShape->getTriangle(triangleIndex).getNormal() * (pointInTriangleLocal - meshShape->getTriangle(triangleIndex).getVertex(0)),agx::Real(0) ) )
        {// point followed the wrong direction of the normal
          pointInTriangle = contactPoint.point() - agx::Vec3( contactPoint.normal() ) * contactPoint.depth() * agx::Real(0.5);
          pointInTriangleLocal = transformPointToShape( pointInTriangle, geometryMesh );
        }
        closestEdgeIndex = closestEdgeToContact( meshShape, triangleIndex, pointInTriangleLocal );

        // Now find if the path through the closest point of the edge is
        // increasing the wire length more than the distance to the plane
        // for any of the two wire segment points.
        // if the distance HAS increased more, then we assume that
        // the wire segment is more or less resting on the surface.
        // If the distance has increased less than the min distance to the plane,
        // we assume that the contact is valid, since then it is not near resting on the plane.
        // Distance from line nodes to edge must also be greater than the wire radius for the face contact to be interesting.
        agx::Vec3 edgeStart = transformPointToWorld(meshShape->getTriangle(triangleIndex).getVertex((uint_fast8_t)closestEdgeIndex),geometryMesh);
        agx::Vec3 edgeEnd   = transformPointToWorld(meshShape->getTriangle(triangleIndex).getVertex( uint_fast8_t(closestEdgeIndex+1)%3),geometryMesh);
        agxCollide::ClosestPointsSolution solution;
        agxCollide::ClosestPointsSolution parallelSolution;
        bool isParallel = false;
        const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON;
        closestPointsSegmentSegment(edgeStart, edgeEnd, lineGeometryBegin, lineGeometryEnd, solution, isParallel, parallelSolution, epsilon );
        agx::Real spDistanceBegin = solution.pointOn1.distance(lineGeometryBegin);
        agx::Real spDistanceEnd = solution.pointOn1.distance(lineGeometryEnd);
        // Find if the distance to the edge close point is less than radius
        if ( spDistanceBegin < wireRadius || spDistanceEnd < wireRadius )
          closestEdgeIndex = -1;
        else
        {
          agx::Real distanceDiff = spDistanceBegin + spDistanceEnd - lineGeometryEnd.distance(lineGeometryBegin);

          if ( std::min( std::abs(distToPlaneFromBegin), std::abs(distToPlaneFromEnd) ) < distanceDiff )
            closestEdgeIndex = -1;
        }

      }

      if ( closestEdgeIndex < 0 )
        return false;

      edgeIndex = (size_t)closestEdgeIndex;

    }
    else
      return false;

    if ( meshShape->getTriangle( triangleIndex ).isValid() && meshShape->getTriangle(triangleIndex).hasHalfEdgePartner((uint8_t)edgeIndex) )
    {
      agx::Vec3 n1 = meshShape->getTriangle(triangleIndex ).getNormal();
      agx::Vec3 n2 = meshShape->getTriangle(triangleIndex).getHalfEdgePartner((uint8_t)edgeIndex).getNormal();

      if ( agx::geq(n1*n2,agx::Real(1) ) )
        return false;
    }
    else
      return false;

    return true;
  }

  //Adjust edge given line radius
  static inline void calculateContactPoint( const agxCollide::Mesh* mesh, size_t triangleIndex, size_t edgeIndex, const agx::Vec3& localContactPoint, agx::Vec3& retLocalCorrectedContactPoint, agx::Vec3& retEdge, agx::Vec3& retEdgeStart, agx::Vec3& retEdgeEnd, const agx::Real /*radius*/ )
  {
    const agxCollide::Mesh::Triangle& triangle = mesh->getTriangle( triangleIndex );

    agx::Vec3 edgeStart = triangle.getEdgeStartVertex( ( uint8_t ) edgeIndex );
    agx::Vec3 edgeEnd = triangle.getEdgeEndVertex( ( uint8_t ) edgeIndex );

    retEdge = edgeEnd - edgeStart;
    agx::Real edgeL = retEdge.normalize();

    agx::Vec3 pRelStart = localContactPoint - edgeStart;
    pRelStart = retEdge*( pRelStart*retEdge );
    retLocalCorrectedContactPoint = edgeStart + pRelStart;

    retEdgeStart = edgeEnd - retEdge*( edgeL /*+ radius*/ );
    retEdgeEnd = edgeStart + retEdge*( edgeL /*+ radius*/ );
  }


  AGXPHYSICS_EXPORT agx::Real calculateNormalForceMagnitude( const agxWire::WireDistanceCompositeConstraint* wireConstraint, agxWire::NodeConstIterator nIt );

  AGXPHYSICS_EXPORT agx::Vec3 getMovementRangeDirection( agx::Vec3 contactNormal, agx::Vec3 wireDir );

  AGXPHYSICS_EXPORT agx::Vec3 getContactNormal( agx::Vec3 wireDir, agx::Vec3 movementRangeDir );

  AGXPHYSICS_EXPORT agx::Vec3 getWireDirection( agxWire::NodeConstIterator nIt );

  AGXPHYSICS_EXPORT agx::Real getAngleAroundMovementRange( const agx::Vec3& p1, const agx::Vec3& p2, const agx::Vec3& p3, const agx::Vec3& contactNormal, const agx::Vec3& movementDir  );

  AGXPHYSICS_EXPORT agx::Real getAngleAroundMovementRange( agxWire::NodeConstIterator nIt );

  DOXYGEN_END_INTERNAL_BLOCK()

}
