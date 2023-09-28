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

#include <agxRender/Color.h>

#include <agxCollide/Contacts.h>

namespace agxCollide
{
  class ContactPoint;
}

namespace agx
{
  typedef std::pair< agx::Real, agx::Real > RealPair;
}

namespace agxWire
{
  class WireDistanceCompositeConstraint;
  class WireGeometryController;
  class WireMaterialController;
  class WireHandler;

  typedef agx::SymmetricPair< agxWire::Node* > SymmetricNodePtrPair;

  DOXYGEN_START_INTERNAL_BLOCK()


  /**
  Internal class to store definition of a contact point.
  */
  struct ContactPointDef
  {
    enum OnSegment { BEFORE = -1, UNKNOWN, AFTER };
    ContactPointDef( const agx::Vec3& p, const agx::Vec3f& n, agx::Real d )
      : point( p ), normal( n ), depth( d ), primary( true ), onSegment( UNKNOWN ) {}
    ContactPointDef( const agx::Vec3& p, const agx::Vec3f& n, agx::Real d, agx::Bool isPrimaryPoint )
      : point( p ), normal( n ), depth( d ), primary( isPrimaryPoint ), onSegment( UNKNOWN ) {}
    agx::Vec3   point;
    agx::Vec3f  normal;
    agx::Real   depth;
    agx::Bool   primary;
    OnSegment   onSegment;

    typedef agx::VectorPOD< ContactPointDef > Container;
  };

  namespace WireContactStatus
  {
    enum Enum { KEEP, REMOVE, RELEASE };
  }

  //                      CONTACT SEGMENT
  //                  |++++++++++++++++++++|
  //   begin       *currIt     point    *nextIt          end
  // ----o------------o----------O+++++++++o---------------
  //                  ^----------^---------^
  //                      \/          \/
  //           dist to curr            dist to next
  class AGXPHYSICS_EXPORT WireContactSegment
  {
    public:
      /**
      Internal convenient class making it possible to create wire
      contact segments for any type of known data.
      */
      class AGXPHYSICS_EXPORT PointDef
      {
        public:
          /**
          Default constructor. All values set to 0.
          */
          PointDef();

          /**
          Construct given point, normal and depth.
          */
          PointDef( const agx::Vec3& p, const agx::Vec3f& n, agx::Real d );

          /**
          Construct given agxCollide::GeometryContact.
          */
          explicit PointDef( const agxCollide::ContactPoint& contactPoint );

          /**
          \return the contact point
          */
          agx::Vec3& point() { return m_point; }
          const agx::Vec3& point() const { return m_point; }

          /**
          \return the contact normal
          */
          agx::Vec3f& normal() { return m_normal; }
          const agx::Vec3f& normal() const { return m_normal; }

          /**
          \return the contact depth
          */
          agx::Real& depth() { return m_depth; }
          agx::Real depth() const { return m_depth; }

          operator agxWire::ContactPointDef() const { return ContactPointDef( m_point, m_normal, m_depth ); }

        private:
          agx::Vec3  m_point;
          agx::Vec3f m_normal;
          agx::Real  m_depth;
      };

    public:
      /**
      \return true if the node is a wire-wire contact node
      */
      static bool isContactNode( const agxWire::Node* node );

      /**
      Set (or unset) \p node to a (by definition) contact node.
      \param node - node to set
      \param enable - true to transform \p node to a contact node, false to transform \p node to a normal node
      */
      static void setContactNode( agxWire::Node* node, bool enable );

      /**
      Definition of the minimum and maximum rest length of a
      contact segment given the wire.
      \param wire - the wire
      \return pair where first element is the minimum length and the second the maximum (first <= second)
      */
      static agx::RealPair getMinMaxSegmentRestLength( const agxWire::WireDistanceCompositeConstraint* wire );

      /**
      Definition of the minimum and maximum rest length of a
      contact segment during wire-wire contacts.
      \param wire1 - first wire
      \param wire2 - second wire
      \return pair where first element is the minimum length and the second the maximum (first <= second)
      */
      static agx::RealPair getMinMaxSegmentRestLength( const agxWire::WireDistanceCompositeConstraint* wire1, const agxWire::WireDistanceCompositeConstraint* wire2 );

    public:
      /**
      Construct given a wire, contact point, the wire geometry and minimum distance for this segment to be valid.
      \param wire - the wire this contact segment is on
      \param geometryController
      \param p - contact point ON this segment
      \param wireGeometry - the wire geometry from the geometry contact
      */
      WireContactSegment( agxWire::WireDistanceCompositeConstraint* wire, agxWire::WireGeometryController* geometryController, const agxWire::WireContactSegment::PointDef& p, const agxCollide::Geometry* wireGeometry );

      /**
      Invalid if e.g., iterators or the wire is undefined.
      \return true if valid to use this contact segment - otherwise false
      */
      bool isValid() const;

      /**
      \return the length of this contact segment
      */
      agx::Real getLength() const;

      /**
      \return the rest length of this contact segment
      */
      agx::Real getRestLength() const;

      /**
      \return the minimum length this segment may be
      */
      agx::Real getMinimumLength() const;

      /**
      Assign minimum length of this segment.
      \note This method must be called before any nodes were created by
            this segment. Otherwise the new minimum length will have no affect.
      */
      void setMinimumLength( agx::Real minimumLength );

      /**
      \return the distance from the potential contact node to curr node (begin of this segment)
      */
      agx::Real getDistanceToCurr() const;

      /**
      \return the distance from the potential contact node to next node (end of this segment)
      */
      agx::Real getDistanceToNext() const;

      /**
      \return the radius of this contact segment
      */
      agx::Real getRadius() const;

      /**
      \return contact point projected onto this segment
      */
      agx::Vec3 getPointOnWire() const;

      /**
      \return the data of the original contact point
      */
      const agxWire::WireContactSegment::PointDef& getContactPoint() const;

      /**
      \return the data of the original contact point
      */
      agxWire::WireContactSegment::PointDef& getContactPoint();

      /**
      \param ret_newCurr - set to true by this method if new node has to be created at curr
      \param ret_newNext - set to true by this method if new node has to be created at next
      \return true if newCurr, newNext or both are true
      */
      bool requiresNewNodes( bool& ret_newCurr, bool& ret_newNext ) const;

      /**
      This method will fetch (create or find) nodes given minimum segment
      distance.
      \return true if this contact segment still is valid after this call
      */
      bool acquireNodes();

      /**
      \return the wire this contact segment is defined on
      */
      agxWire::WireDistanceCompositeConstraint* getWire() const;

      /**
      \return the wire geometry
      */
      const agxCollide::Geometry* getWireGeometry() const;

      /**
      \return the direction of this contact segment - in world coordinates
      */
      agx::Vec3 getWorldDir() const;

      /**
      \return iterator to begin node in this contact segment
      */
      agxWire::NodeIterator getCurrIt() const;

      /**
      \return iterator to begin node in this contact segment
      */
      agxWire::NodeIterator getNextIt() const;

      /**
      \return begin node in this contact segment (0 if invalid)
      */
      agxWire::Node* getCurrNode() const;

      /**
      \return end node in this contact segment (0 if invalid)
      */
      agxWire::Node* getNextNode() const;

      /**
      Set world transform of this contact segment. If this segment "is solid" both
      current and next node will be moved. If "not solid", only the contact node
      will be moved.

      The transform is the middle point, y-axis is pointing towards next node.
      The rotation is ignored if this segment isn't solid.
      */
      void setTransform( const agx::AffineMatrix4x4& transform );

    private:
      enum RealData { SEGMENT_LENGTH, DIST_TO_CURR, DIST_TO_NEXT, SEGMENT_MINIMUM_LENGTH, NUM_PARAMS };

    private:
      WireContactSegment();

      /**
      Initializes all parameters.
      */
      void initialize( agxWire::WireGeometryController* geometryController, const agxCollide::Geometry* wireGeometry );

    private:
      WireDistanceCompositeConstraint*  m_wire;
      const agxCollide::Geometry*       m_wireGeometry;
      PointDef                          m_point;
      agx::Vec3                         m_pointOnWire;
      NodeIterator                      m_currIt;
      NodeIterator                      m_nextIt;
      agx::Real                         m_params[ NUM_PARAMS ];
      bool                              m_initialized;
  };

  #define RENDER_DEBUG_POINT( p, color, reason, pos ) \
    ::render( p, color ); \
    agxRender::RenderSingleton::instance()->add( reason, pos, color )


  AGXPHYSICS_EXPORT void render( const agx::Vec3& point, const agx::Vec3f& normal, const agxRender::Color& color );

  template< typename T >
  void render( const T& point, const agxRender::Color& color ) { agxWire::render( point.point(), point.normal(), color ); }

  /**
  Render a wire shape given positions, radius and color.
  */
  AGXPHYSICS_EXPORT void renderWireShape( const agx::Vec3& prevStart, const agx::Vec3& prevEnd, const agx::Vec3& currStart, const agx::Vec3& currEnd, agx::Real radius, const agx::Vec4f& color );

  /**
  Calculates the angle nextIt wraps round the other wire, defined by normalIt.
  */
  AGXPHYSICS_EXPORT agx::Real calculateAngle( agxWire::NodeConstIterator normalIt, agxWire::NodeConstIterator nextIt );

  typedef std::pair< NodeConstIterator, NodeConstIterator > NodeConstIteratorPair;

  /**
  Calculates normalized vectors from it to previous node and from it to next node. Assumes ++it and --it is valid.
  \param it - iterator to node
  \param ret_toPrev - return value, vector from \p it to previous node
  \param ret_toNext - return value, vector from \p it to next node
  \return iterators to previous and next node
  */
  AGXPHYSICS_EXPORT NodeConstIteratorPair calculateToPrevToNextVectors( const agxWire::NodeConstIterator it, agx::Vec3& ret_toPrev, agx::Vec3& ret_toNext );

  /**
  Finds stability factor for lumped nodes. If \p nodeIt isn't a lumped node,
  the stability factor is defined to be zero.
  \param nodeIt - lumped node
  \param wire - wire with \p node
  \return a scale given the current numerical stability of the node (the bigger value the more unstable)
  */
  AGXPHYSICS_EXPORT agx::Real findStabilityFactor( const agxWire::NodeConstIterator nodeIt, const agxWire::WireDistanceCompositeConstraint* wire );

  /**
  Reduce wire contacts, return index with maximum depth of the remaining points.
  */
  AGXPHYSICS_EXPORT agx::UInt reduceContacts( agxWire::ContactPointDef::Container& points, agx::Real32 threshold = 0.99f, agx::Bool reduceNegativeDotProducts = false, agx::Real32 negThreshold = -0.0f );

  DOXYGEN_END_INTERNAL_BLOCK()

}

