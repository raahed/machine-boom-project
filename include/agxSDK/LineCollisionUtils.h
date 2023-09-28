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

#ifndef AGXSDK_LINECOLLISIONUTILS_H
#define AGXSDK_LINECOLLISIONUTILS_H

#include <agx/Math.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>
#include <agx/Quat.h>
#include <agx/AffineMatrix4x4.h>
#include <agxCollide/Mesh.h>

namespace agxSDK
{
  // Return true if boundLower - threshold < val < boundUpper + threshold
  static inline bool inBound( agx::Real val, agx::Real boundLower, agx::Real boundUpper, agx::Real threshold = agx::Real( 1E-6 ) )
  {
    return val > boundLower - threshold && val < boundUpper + threshold;
  }

  // Returns the sign pattern in v
  static inline agx::Vec3 getSigned( const agx::Vec3& v )
  {
    return agx::Vec3( agx::sign( v.x() ), agx::sign( v.y() ), agx::sign( v.z() ) );
  }

  // Computes the scalar triple product: s = p * (q x r)
  static inline agx::Real scalarTripleProduct( const agx::Vec3& p, const agx::Vec3& q, const agx::Vec3& r )
  {
    return p * ( q ^ r );
  }

  static inline agx::Vec3 getProjection( const agx::Vec3 contactPoint, agx::Vec3 edge, bool normalizedEdge = false ) {

    if (!normalizedEdge)
      edge.normalize();
    return (edge*(contactPoint*edge));

  }

  static inline agx::Real getTotalAngleShapeCoordinates( agx::Vec3 edge, agx::Vec3 contactToCenter, const agx::Vec3& pos1, const agx::Vec3& pos2, const agx::Vec3& pos3, bool& onSameEdge )
  {
    // Calculate vectors
    agx::Vec3 v21 = pos1 - pos2;
    agx::Vec3 v23 = pos3 - pos2;
    // Project all vectors onto plane defined by edge
    contactToCenter -= edge * (edge * contactToCenter);
    v21             -= edge * (edge * v21);
    v23             -= edge * (edge * v23);
    agx::Real l1 = contactToCenter.normalize();
    agx::Real l2 = v21.normalize();
    agx::Real l3 = v23.normalize();
    onSameEdge = ( agx::equalsZero( l1 ) || agx::equalsZero( l2 ) || agx::equalsZero( l3 ) );
    return (agx::Real)(acos( agx::clamp( v21 * contactToCenter, agx::Real( -1 ), agx::Real( 1 ) ) ) + acos( agx::clamp( v23 * contactToCenter, agx::Real( -1 ), agx::Real( 1 ) ) ));
  }

  static inline agx::Real findDistanceFromPointToLine( const agx::Vec3& L1, const agx::Vec3& L2, const agx::Vec3& P )
  {
    agx::Vec3 v1 = L2 - L1;
    agx::Vec3 v2 = P  - L1;
    if ( agx::equalsZero(L2.distance2(P)) )
      return 0.0;

    if ( agx::equalsZero(L1.distance2(P)) )
      return 0.0;

    if ( agx::equalsZero(L2.distance2(L1)) )
      return P.distance(L1);

    //project v2 to v1
    agx::Vec3 v2Proj = v1*(v1*v2)/v1.length2();

    /************************************************************************/
    /*
    CASE 1:
    L1------------------------------------L2  return dist between P and L1
    P

    CASE 2:

    L1------------p-----------------------L2          return normal distance to line
    |
    |
    P
    CASE 3:

    L1------------------------------------L2         return dist between P1 and L2

    P

    */
    /************************************************************************/

    if (  agx::leq(v2Proj * v1,agx::Real(0) ) )
    {// CASE 1
      return P.distance(L1);
    }
    else
    {
      if ( agx::geq( v2Proj.length2(), v1.length2()) )
      {//CASE 3
        return P.distance(L2);
      }
      else
      {
        agx::Vec3 p = L1 + v2Proj;
        return P.distance(p);
      }

    }

  }

  // Given a vector with 12 elements, pair( Vec3, Vec3 ), this function will fill that what vector
  // with the twelve edges.
  static inline void fillEdgeVector( agx::Vector< std::pair< agx::Vec3, agx::Vec3 > >& v, const agx::Vec3& halfExtent )
  {
    agx::Real hex = halfExtent.x();
    agx::Real hey = halfExtent.y();
    agx::Real hez = halfExtent.z();

    // ---------------------------------------------

    v[ 0 ].first.set(  -hex, -hey, -hez );
    v[ 0 ].second.set( -hex,  hey, -hez );

    v[ 1 ].first.set(  -hex, -hey,  hez );
    v[ 1 ].second.set( -hex,  hey,  hez );

    v[ 2 ].first.set(   hex, -hey,  hez );
    v[ 2 ].second.set(  hex,  hey,  hez );

    v[ 3 ].first.set(   hex, -hey, -hez );
    v[ 3 ].second.set(  hex,  hey, -hez );

    // ---------------------------------------------

    v[ 4 ].first.set(  -hex, -hey, -hez );
    v[ 4 ].second.set( -hex, -hey,  hez );

    v[ 5 ].first.set(  -hex, -hey,  hez );
    v[ 5 ].second.set(  hex, -hey,  hez );

    v[ 6 ].first.set(   hex, -hey, -hez );
    v[ 6 ].second.set(  hex, -hey,  hez );

    v[ 7 ].first.set(  -hex, -hey, -hez );
    v[ 7 ].second.set(  hex, -hey, -hez );

    // ---------------------------------------------

    v[ 8 ].first.set(  -hex,  hey, -hez );
    v[ 8 ].second.set( -hex,  hey,  hez );

    v[ 9 ].first.set(  -hex,  hey,  hez );
    v[ 9 ].second.set(  hex,  hey,  hez );

    v[ 10].first.set(   hex,  hey, -hez );
    v[ 10].second.set(  hex,  hey,  hez );

    v[ 11].first.set(  -hex,  hey, -hez );
    v[ 11].second.set(  hex,  hey, -hez );

    // ---------------------------------------------
  }

  // Given A and B in the circle plane, this function will find Q1 and/or Q2 which is the points
  // where the line segment intersects the circle.
  static inline bool findSegmentCircleIntersections2D(
    const agx::Vec3& A,
    const agx::Vec3& B,
    const agx::Vec3& circlePos,
    agx::Real radius,
    agx::Vec3& Q1,
    agx::Vec3& Q2,
    bool& Q1Found,
    bool& Q2Found,
    const agx::Real epsilon = agx::AGX_EQUIVALENT_EPSILON )
  {
    Q1Found = false;
    Q2Found = false;

    agx::Vec3 R = A - circlePos;
    agx::Vec3 dir = B - A;
    agx::Real segmentLengthSq = dir.length2();

    if ( agx::equalsZero( segmentLengthSq ) )
      return false;

    agx::Real a = 2 * ( dir * R ) / segmentLengthSq;
    agx::Real b = ( R.length2() - radius * radius ) / segmentLengthSq;

    agx::Real roots = agx::Real(0.25) * a * a - b;
    if ( roots < 0 || agx::equalsZero( roots ) )
      return false;

    roots = std::sqrt( roots );

    agx::Real t1 = agx::Real(-0.5) * a - roots;
    agx::Real t2 = agx::Real(-0.5) * a + roots;

    agxAssert( t1 < t2 );

    if ( agx::geq( t1 , agx::Real(0),epsilon*radius*radius ) && agx::leq( t1 , 1,epsilon*radius*radius) )
    {
      if ( t1 < 0 )
        t1 = 0;
      if ( t1 > 1 )
        t1 = 1;
      Q1Found = true;
      Q1 = A + dir * t1;
    }
    if ( agx::geq( t2 , agx::Real(0),epsilon*radius*radius ) && agx::leq( t2 , 1, epsilon*radius*radius)  )
    {
      if ( t2 < 0 )
        t2 = 0;
      if ( t2 > 1 )
        t2 = 1;
      Q2Found = true;
      Q2 = A + dir * t2;
    }

    return Q1Found || Q2Found;
  }

  // Given line segment A -> B and a plane, this method finds the point in the plane where the line segment passes.
  static bool findPlaneLineSegmentIntersection( const agx::Vec3& A, const agx::Vec3& B, const agx::Vec3& planeOrigin, const agx::Vec3& planeNormal, agx::Real nearEndsThreshold, agx::Vec3& Q )
  {
    agxAssert( agx::equivalent( planeNormal.length2(), agx::Real( 1 ) ) );

    agx::Real lengthInPlane = planeNormal * ( B - A );
    // Segment is parallel to the plane, check if A is in the plane?
    if ( agx::equalsZero( lengthInPlane ) )
      return false;

    agx::Real s = ( planeNormal * ( planeOrigin - A ) ) / lengthInPlane;
    Q = A + ( B - A ) * s;
    return inBound( s, 0 - nearEndsThreshold, 1 + nearEndsThreshold, 0 );
  }

  // Given line segment A -> B and a circle, this method finds the intersection point in the circle.
  // If the return value is true, the segment intersects the circle.
  static inline bool findSegmentCircleIntersection3D( const agx::Vec3& A, const agx::Vec3& B, const agx::Vec3& circleOrigin, const agx::Vec3& circleNormal, agx::Real circleRadius, agx::Real nearEndsThreshold, agx::Vec3& intersectionPoint )
  {
    bool intersects = findPlaneLineSegmentIntersection( A, B, circleOrigin, circleNormal, nearEndsThreshold, intersectionPoint );
    if ( !intersects )
      return false;

    agx::Real l2 = ( intersectionPoint - circleOrigin ).length2();

    return l2 < circleRadius * circleRadius && !agx::equalsZero( l2-circleRadius*circleRadius, nearEndsThreshold*2 ) ;
  }

  static inline bool pointInsideCircle( const agx::Vec3& point, const agx::Vec3& circleOrigin, agx::Real radius )
  {
    agx::Real dist2 = point.distance2( circleOrigin );
    agx::Real r2 = radius * radius;
    return  agx::leq(dist2, r2) && !agx::equalsZero( dist2 - r2,agx::Real(1E-5) );
  }

  static inline agx::AffineMatrix4x4 getFuturePosition( const agx::Vec3& velocity, const agx::Vec3& angularVelocity, const agx::Vec3& position, const agx::Quat& rotation, agx::Real time )
  {
    agx::AffineMatrix4x4 ret;
    ret.setTranslate( position + velocity * time );
    agx::Vec3 ww = angularVelocity;
    agx::Real alpha = agx::Real(0.5) * time * ww.normalize();
    ret.setRotate( rotation * agx::Quat( agx::Vec4( ww * (agx::Real)sin( alpha ), (agx::Real)cos( alpha ) ) ) );
    return ret;
  }

  class LineContactEdge
  {
  public:
    LineContactEdge() : m_length( 0 ) {}
    LineContactEdge( const agx::Vec3& start, const agx::Vec3& end )
      : m_start( start ), m_end( end )
    {
      m_normalized = end-start;
      m_length = m_normalized.normalize();
    }
    ~LineContactEdge(){}
    const agx::Vec3& getNormalized() const { return m_normalized; }
    const agx::Vec3& getStartPoint() const { return m_start; }
    const agx::Vec3& getEndPoint() const { return m_end; }
    agx::Vec3 getVector() const { return m_end-m_start; }
    agx::Real getLength() const { return m_length; }

    int getMajorEdgeAxis() const
    {
      agx::Real max = 0.0;
      int maxIndex = -1;
      for (  int i = 0; i < 3; ++i ) {
        if (m_normalized[i] > max) {
          maxIndex = i;
          max = m_normalized[i];
        }
      }
      return maxIndex;

    }

    void mirror()
    {
      agx::Vec3 temp = m_start;
      m_start = m_end;
      m_end = temp;
      m_normalized = m_end-m_start;
      m_length = m_normalized.normalize();
    }

    LineContactEdge findOppositeEdge() const
    {
      agx::Vec3 edgeStart = m_start;
      agx::Vec3 edgeEnd = m_end;
      negateWhereNormalEntriesAreZero( edgeStart );
      negateWhereNormalEntriesAreZero( edgeEnd );
      return LineContactEdge( edgeStart, edgeEnd );
    }

    /**
    Sets the value where the edge entry is zero. edge = 1,0,0 => return = 0,value,value.
    */
    static inline agx::Vec3 setValueWhereZeroEntries( const agx::Vec3& edge, agx::Real value )
    {
      return agx::Vec3( edge[ 0 ] == 0 ? value : 0, edge[ 1 ] == 0 ? value : 0, edge[ 2 ] == 0 ? value : 0 );
    }

    static inline LineContactEdge findOppositeEdge( const agx::Vec3& edgeStart, const agx::Vec3& edgeEnd, const agx::Vec3& edge )
    {
      agx::Vec3 oppositeEdgeStart = edgeStart;
      agx::Vec3 oppositeEdgeEnd = edgeEnd;

      for ( int i = 0; i < 3; ++i ) {
        if ( edge[ i ] == 0 ) {
          oppositeEdgeStart[ i ] *= -1;
          oppositeEdgeEnd[ i ]   *= -1;
        }
        else {
          agxAssert( !agx::equalsZero( edge[ i ] ) );
        }
      }

      return LineContactEdge( oppositeEdgeStart, oppositeEdgeEnd );
    }
  private:
    void negateWhereNormalEntriesAreZero( agx::Vec3& v ) const
    {
      v[ 0 ] *= agx::Real(m_normalized[ 0 ] == 0 ? -1 : 1);
      v[ 1 ] *= agx::Real(m_normalized[ 1 ] == 0 ? -1 : 1);
      v[ 2 ] *= agx::Real(m_normalized[ 2 ] == 0 ? -1 : 1);
    }

    agx::Vec3 m_start;
    agx::Vec3 m_end;
    agx::Vec3 m_normalized;
    agx::Real m_length;
  };

  class LineContactPosition
  {
  public:
    LineContactPosition()
      :m_t(0), m_edge(LineContactEdge()){}

    LineContactPosition( agx::Vec3 start, agx::Vec3 end, agx::Real t ) : m_t(t)
    {
      m_edge = LineContactEdge(start,end);

    }

    ~LineContactPosition() {}

    ///\return local position
    agx::Vec3 getPosition()
    {
      return m_edge.getStartPoint()+(m_edge.getEndPoint()-m_edge.getStartPoint())*m_t;
    }

    agx::Real getT() const {return m_t;}
    const LineContactEdge& getEdge() const {return m_edge;}

  private:
    agx::Real m_t;
    LineContactEdge m_edge;
  };

  static inline agx::Vec3 calculateBoxOffset( const agx::Vec3& shapeEdge, const agx::Vec3& shapeContactPosition, agx::Real radius )
  {
    agx::Vec3 offset = LineContactEdge::setValueWhereZeroEntries( shapeEdge, radius );
    return agx::Vec3::mul( offset, getSigned( shapeContactPosition ) );
  }

  static inline agx::Vec3 calculateCylinderOffset( const agx::Vec3& shapeEdge, const agx::Vec3& shapeContactPosition, agx::Real radius )
  {

    agx::Vec3 offset;
    agx::Vec3 radialDir = shapeContactPosition;
    radialDir.y() = 0;
    radialDir.normalize();
    radialDir *= radius;
    offset = radialDir;
    if ( agx::equalsZero( shapeEdge.y() ) )
    {//circle contact

      offset += agx::Vec3(0,(agx::Real)agx::sign(shapeContactPosition.y())*radius,0);
    }
    return offset;
  }

  static inline agx::Vec3 calculateMeshOffset( const agxCollide::Mesh* mesh, const size_t triangleIndex, const uint8_t edgeIndex ,const agx::Vec3& /*shapeEdge*/, const agx::Real radius )
  {
    const agxCollide::Mesh::Triangle triangle1 = mesh->getTriangle(triangleIndex);
    if ( !triangle1.isValid() )
      return agx::Vec3(0,0,0);

    agx::Vec3 normal1 = triangle1.getNormal();

    const agxCollide::Mesh::Triangle triangle2 = triangle1.getHalfEdgePartner( edgeIndex );

    agx::Vec3 normal2;

    if ( !triangle2.isValid() )
      normal2 = agx::Vec3(0,0,0);
    else
      normal2 = triangle2.getNormal();

    agx::Vec3 offset = normal1 + normal2;
    offset.normalize();
    offset *= radius;
    return offset;

  }

  static inline uint8_t getEdgeStartLocalVertexIndex(const uint8_t localEdgeIndex,const bool atBegin)
  {
    uint8_t vertexIndex = localEdgeIndex;
    if ( !atBegin )
      vertexIndex = (uint8_t)((localEdgeIndex + 1)%3);
    return vertexIndex;
  }

  static inline uint8_t getEdgeEndLocalVertexIndex(const uint8_t localEdgeIndex,const bool atBegin)
  {
    //The other end of edge, must be compared with the result so that we don't get the same edge again, but other direction
    uint8_t startEdgeIndex = getEdgeStartLocalVertexIndex(localEdgeIndex, atBegin);

    if ( atBegin )
      startEdgeIndex = (uint8_t)((startEdgeIndex + 1)%3);
    else
      startEdgeIndex = (uint8_t)((startEdgeIndex + 2)%3);

    return startEdgeIndex;//really end =)

  }
  static inline uint8_t getLocalEdgeIndexFromVertexIndices( const size_t iteratedVertexIndex, const size_t oppositeVertexIndex )
  {
    uint8_t localEdgeIndex = 0;
    if ( oppositeVertexIndex > iteratedVertexIndex )
    {
      if ( oppositeVertexIndex == 2 )
      {
        if ( iteratedVertexIndex == 0 )
          localEdgeIndex = 2;
        else //iteratedVertexIndex == 1
          localEdgeIndex = 1;
      }
      else if ( oppositeVertexIndex == 1 )
        localEdgeIndex = 0;
    }
    else
    {
      if ( oppositeVertexIndex == 1 )
      {

        localEdgeIndex = 1;
      }
      else if ( oppositeVertexIndex == 0 )
      {
        if ( iteratedVertexIndex == 2 )
          localEdgeIndex = 2;
        else //iteratedVertexIndex == 1
          localEdgeIndex = 0;
      }
    }
    return localEdgeIndex;
  }

  static inline void findMostParallelEdgeAmongNeighboringTriangles( const agxCollide::Mesh* mesh,const agx::Vec3& /*shapeOffset*/,const size_t triangleIndex,const uint8_t edgeIndex, size_t& tempNewTriangleStartIndex,size_t& tempNewEdgeLocalStartIndex,const bool atBegin )
  {

    agxCollide::Mesh::Triangle triangle = mesh->getTriangle(triangleIndex);

    uint8_t vertexIndex = getEdgeStartLocalVertexIndex(edgeIndex,atBegin);

    agx::Real angle = agx::Real(1.1)*agx::PI;

    agxCollide::Mesh::VertexEdgeCirculator triangleIteratorVertex = mesh->createVertexEdgeCirculator( triangleIndex, vertexIndex);

    agx::Vec3 currentEdge = mesh->getTriangleVertex(triangleIndex, (uint_fast8_t)(edgeIndex + (uint8_t)1)%(uint8_t)3) - mesh->getTriangleVertex(triangleIndex,(uint_fast8_t)edgeIndex);
    currentEdge.normalize();

    if ( atBegin )
      currentEdge *= -1.0;

    size_t globalVertexIndex = triangleIteratorVertex.getTriangle().getGlobalVertexIndex(vertexIndex);
    agx::Vec3 vertexPos = triangleIteratorVertex.getTriangle().getVertex(vertexIndex);

    bool done = false;

    triangleIteratorVertex++; //++

    while ( triangleIteratorVertex.isValid() && !triangleIteratorVertex.atEnd() ) {

      size_t tempTriangleIndex = triangleIteratorVertex.getTriangle().getTriangleIndex();

      if ( tempTriangleIndex == triangleIndex )
      {
        if ( triangleIteratorVertex.getTriangle().hasHalfEdgePartner( triangleIteratorVertex.getLocalEdgeIndex() ) )
        {
          tempTriangleIndex = triangleIteratorVertex.getTriangle().getHalfEdgePartner( triangleIteratorVertex.getLocalEdgeIndex() ).getTriangleIndex();
        }
      }


      agxCollide::Mesh::Triangle tempTriangle = mesh->getTriangle( tempTriangleIndex );

      if ( tempTriangleIndex != triangleIndex )
      {
        agx::Vec3 end1,end2;
        uint8_t i1 = 0;
        uint8_t i2 = 0;
        size_t localVertexIndex = 0;
        if ( tempTriangle.getGlobalVertexIndex( 0 ) == globalVertexIndex )
        {
          i1 = 1;
          i2 = 2;
          localVertexIndex = 0;
        }
        else if ( tempTriangle.getGlobalVertexIndex( 1 ) == globalVertexIndex )
        {
          i1 = 0;
          i2 = 2;
          localVertexIndex = 1;
        }
        else if ( tempTriangle.getGlobalVertexIndex( 2 ) == globalVertexIndex )
        {
          i1 = 0;
          i2 = 1;
          localVertexIndex = 2;
        }
        else
        {
          triangleIteratorVertex++;
          continue;
        }

        end1 = tempTriangle.getVertex(i1);
        end2 = tempTriangle.getVertex(i2);

        end1 -= vertexPos;
        end1.normalize();

        end2 -= vertexPos;
        end2.normalize();

        agx::Real tempAngle1 = (agx::Real)acos( agx::clamp( currentEdge * end1,agx::Real(-1),agx::Real(1)  ));

        //if ( end1 * shapeOffset > 0 )
        //  tempAngle1 += agx::PI;

        agx::Real tempAngle2 = (agx::Real)acos( agx::clamp( currentEdge * end2 ,agx::Real(-1),agx::Real(1)) );

        //if ( end2 * shapeOffset > 0 )
        //  tempAngle2 += agx::PI;

        bool secondIsVerified = false;
        uint8_t localEdgeIndex = 0;
        if ( tempAngle2 < tempAngle1)
        {//only need to check the larger of the two

          //verify i2
          localEdgeIndex = getLocalEdgeIndexFromVertexIndices( localVertexIndex,i2 );

          if ( triangle.getTriangleIndex() != tempTriangle.getHalfEdgePartner( localEdgeIndex ).getTriangleIndex() )
          {
            tempAngle1 = tempAngle2;
            i1 = i2;
            secondIsVerified = true;
          }

        }

        if ( tempAngle1 < angle )
        {
          if ( !secondIsVerified )
          {
            localEdgeIndex = getLocalEdgeIndexFromVertexIndices( localVertexIndex,i1 );

            if ( triangle.getTriangleIndex() == tempTriangle.getHalfEdgePartner( localEdgeIndex ).getTriangleIndex() )
            {
              triangleIteratorVertex++;
              continue;
            }
          }

          tempNewTriangleStartIndex  = tempTriangleIndex;
          tempNewEdgeLocalStartIndex = localEdgeIndex;
          angle = tempAngle1;
          done = true;

        }
      }

      triangleIteratorVertex++; //++

    }

    if ( !done )
    {
      tempNewTriangleStartIndex = triangleIndex;
      tempNewEdgeLocalStartIndex = edgeIndex;
    }

  }

  /**
  Shorten the movement range due to neighboring triangles. This function does not consider if the actual edge is in start - end direction, all it assumes is that the two startpositions is at same position
  \param lineEndW - end position for the edge linecontact is located on
  \param lineStartW - start position for the edge linecontact is located on
  \param edgeStart  - start position for edge on next triangle ( should be same position as lineStartW )
  \param edgeEnd  - end position for edge on next triangle
  \return t - the clamp position on the parametrized edge between lineEndW (0) and lineStartW( 0)
  */
  static inline agx::Real findAdjustmentForMeshEdge( const agx::Vec3& lineEndW,  const agx::Vec3& lineStartW, const agx::Vec3& edgeStart,  const agx::Vec3& edgeEnd , agx::Real lineRadius)
  {
    agx::Real t = 0;

    agx::Vec3 v1 = lineEndW - lineStartW;
    agx::Vec3 v2 = edgeEnd - edgeStart;
    agx::Real dist = v1.normalize();
    v2.normalize();

    agx::Vec3 mean = v2+v1;
    mean.normalize();

    agx::Real angle = (agx::Real)acos( agx::clamp(-v1*v2,agx::Real(-1),agx::Real(1)) );
    agx::Real removedDist = lineRadius * (agx::Real)tan( angle * agx::Real(0.5) ) + lineRadius;

    if ( dist - removedDist < 0 )
      return 0;

    t = (dist - removedDist)/dist;

    return t;
  }

  static inline void copyVectors( agx::Vec3Vector& newContactShapeTranslates,const agx::Vec3Vector& newContactShapeTranslatesB,agx::Vector<size_t>& newTriangleIndices,const agx::Vector<size_t>& newTriangleIndicesB,agx::Vector<size_t>& newEdgeIndices,const agx::Vector<size_t>& newEdgeIndicesB )
  {
    for ( size_t i = 0; i < newContactShapeTranslatesB.size(); ++i )
    {
      newContactShapeTranslates.push_back( newContactShapeTranslatesB[i] );
      newTriangleIndices.push_back( newTriangleIndicesB[i] );
      newEdgeIndices.push_back( newEdgeIndicesB[i] );
    }
  }

}

#endif

