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
#include <agxWire/WireDistanceCompositeConstraint.h>

#include <agxSDK/StepEventListener.h>

DOXYGEN_START_INTERNAL_BLOCK()


#define WIRE_IMPACT_EPSILON ((double)1E-6)


namespace agx
{
  class DebugRendererCache;
}

namespace agxCollide
{
  class Box;
  class Cylinder;
  class Mesh;
}

namespace agxWire
{
  // Forward declarations
  class WireGeometryController;

  DOXYGEN_START_INTERNAL_BLOCK()


  namespace NodeInvalidFlag
  {
    /// Enum used to debug why nodes are removed and used for rendering
    enum Enum { VALID = 0, VALID_CIRCLE_CONTACT, INVALID_PLANE_ANGLE_OR_SAME_EDGE_TEST, INVALID_PASSED_EDGE_TEST, INVALID_PENETRATION_DEPTH_TOO_LARGE, INVALID_SAME_EDGE_TEST, INVALID };
  }
  DOXYGEN_END_INTERNAL_BLOCK()

  // Small class used to render contact points handled by this controller
  class WireImpactRenderer : public agxSDK::StepEventListener
  {
    public:
      WireImpactRenderer();

      virtual void preCollide( const agx::TimeStamp& /*t*/ )
      {
        m_points.clear( agx::Vector< PointNormalDepth >::MAINTAIN_BUFFER );
        m_clnPND.clear( ContactWireNodeRefPointNormalDepthTable::MAINTAIN_BUFFER );
        m_wires.clear();
      }

      void add( const agxCollide::GeometryContact* geometryContact, bool used = true );
      void add( const agxCollide::ContactPoint& point, bool used = true );
      void add( const agxCollide::ContactPoint& point, NodeInvalidFlag::Enum renderFlag );
      void add( ContactNode* cln );
      void add( const agx::Vec3& p1, const agx::Vec3& p2, const agx::Vec3& color );
      void changeRenderFlag( ContactNode* cln, NodeInvalidFlag::Enum renderFlag );

      void render( agxRender::RenderManager *mgr ) const;

    protected:
      virtual ~WireImpactRenderer() {}

    private:
      struct PointNormalDepth
      {
        PointNormalDepth() : depth( 0 ), renderFlag( NodeInvalidFlag::VALID ) {}
        PointNormalDepth( agx::Real d, const agx::Vec3& n, const agx::Vec3& p, NodeInvalidFlag::Enum flag )
          : depth( d ), normal( n ), point( p ), renderFlag( flag ) {}
        agx::Real depth;
        agx::Vec3 normal;
        agx::Vec3 point;
        NodeInvalidFlag::Enum renderFlag;
      };
      typedef agx::HashTable< ContactNodeRef, PointNormalDepth > ContactWireNodeRefPointNormalDepthTable;

      ContactWireNodeRefPointNormalDepthTable m_clnPND;
      agx::Vector< PointNormalDepth > m_points;
      agx::Vector< agx::Vec3 > m_colors;
      agx::Vector< std::pair< std::pair< agx::Vec3, agx::Vec3 >, agx::Vec3 > > m_wires;
      agx::Vector< std::string > m_descriptions;
  };

  typedef agx::ref_ptr< WireImpactRenderer > WireImpactRendererRef;

  /**
  Handles impacts (first contact) between a wire geometry and other geometries. If the contact point
  is valid and interesting for the wire, a contact wire node is created. It is up to the wire or
  other controllers to remove this contact point.
  */
  class AGXPHYSICS_EXPORT WireImpactController : public agx::Referenced
  {
    public:
      WireImpactController();

      /**
      If collision debugging, adds the impact renderer.
      */
      virtual void addNotification();

      /**
      If collision debugging, removes the impact renderer.
      */
      virtual void removeNotification();

      /**
      Associate the wire constraint this impact controller handles.
      \param wire - wire distance composite constraint
      \param geometryController - pointer to a WireGeometryController
      */
      void setWire( WireDistanceCompositeConstraint* wire, WireGeometryController* geometryController );

      /**
      Handles the geometry contact (impact or contact).
      \return true if this geometry contact should be removed immediately
      */
      virtual bool handleGeometryContact( agxCollide::GeometryContact* geometryContact );


    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~WireImpactController();


      /**
      Handles wire-wire contacts.
      \return true if the geometry contact should be removed immediately
      */
      bool handleWire( agxCollide::GeometryContact* geometryContact, agxCollide::Geometry* otherGeometry );

      /**
      \return the simulation
      */
      agxSDK::SimulationProxy* getSimulationProxy() const;

    protected:
      friend class WireOldContactController;

    protected:
      agx::observer_ptr< WireDistanceCompositeConstraint > m_wire;
      WireGeometryController*                              m_geometryController;
  };

  typedef agx::ref_ptr< WireImpactController > WireImpactControllerRef;

  inline agxSDK::SimulationProxy* WireImpactController::getSimulationProxy() const
  {
    return m_wire != nullptr ? m_wire->getSimulationProxy() : nullptr;
  }
}

DOXYGEN_END_INTERNAL_BLOCK()

