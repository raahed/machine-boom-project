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

#include <agxWire/WireAttachmentController.h>
#include <agxWire/WireStabilityController.h>

#include <agxSDK/Assembly.h>
#include <agx/Hinge.h>

#include <agx/Range.h>

// Forward declarations
namespace agx
{
  class RigidBody;
  class Prismatic;
}

namespace agxWire
{
  /**
  Winch class for handling in and out hauling of wires.
  \note Spooling out all available wire is not recommended. Either the winch should be stopped before this happens, or the wire should be detached from the winch.
        Force reported by the brake/winch will not be correct if no wire is left in the winch.
  */
  class AGXPHYSICS_EXPORT WireWinchController : public WireAttachmentController, public virtual agxStream::Serializable
  {
    public:
      /**
      Construct this winch given position, normal and body this winch is attached to. Both position
      and normal are in body frame. The normal defines the direction out from the winch, i.e., the
      positive direction in which the wire will be hauled out.
      \param body - rigid body this winch is attached to
      \param positionInBodyFrame - position of this winch on \p body
      \param normalInBodyFrame - direction of this winch
      \param pulledInLength - pulled in length in this winch
      */
      WireWinchController( agx::RigidBody* body, const agx::Vec3& positionInBodyFrame, const agx::Vec3& normalInBodyFrame, agx::Real pulledInLength = agx::Real( 0 ) );

      /**
      Get pointer to Constraint1DOF
      */
      agx::Constraint1DOF* get1DOFConstraint();
      const agx::Constraint1DOF* get1DOFConstraint() const;

      /**
      \return the normal in rigid body frame
      */
      inline const agx::Vec3& getNormal() const;

      /**
      \return the normal in world frame
      */
      agx::Vec3 getWorldNormal() const;

      /**
      Set auto feed to this winch. This mode can be used to facilitate routing. E.g., put all wire
      in this winch (start of route), set mode to auto feed, and route. The wire will automatically
      decrease in this winch to fulfill the rout without initial tension. Default: false
      \param autoFeed - true for auto feed mode
      */
      void setAutoFeed( bool autoFeed );

      /**
      \return true if auto feed mode - otherwise false
      \sa setAutoFeed
      */
      inline bool getAutoFeed() const;

      /**
      Set winch motor speed.
      \param speed - speed of the winch motor (Default: 0)
      */
      void setSpeed( agx::Real speed );

      /**
      \return the desired speed of the winch motor
      */
      inline agx::Real getSpeed() const;

      /**
      \return the current speed of the winch motor
      */
      virtual agx::Real getCurrentSpeed() const;

      /**
      Set the desired winch force where range::lower is the force in which this winch
      may haul in wire, range::upper haul out. It's important that the lower value is
      smaller than zero and the upper larger than zero. Default: (-Infinity, Infinity)
      \param forceRange - force (Newton) range [haul_in, haul_out] where haul_in <= 0 and haul_out >= 0
      */
      void setForceRange( agx::RangeReal forceRange );

      /**
      Same as setForceRange( agx::RangeReal forceRange ) but this method will set range as [-force, force].
      \param force - force (Newton) to define the total force range
      */
      void setForceRange( agx::Real force );

      /**
      \return the winch force range
      */
      inline const agx::RangeReal& getForceRange() const;

      /**
      \return the current force the winch motor is applying
      */
      virtual agx::Real getCurrentForce() const;

      /**
      Set the desired brake force range where range::lower is the force in which this winch brake
      can hold the wire from hauling out and ranger::upper the brake force hold the wire from hauling in.
      It's important that the lower value is smaller than zero and the upper larger than zero.
      Default: (-0, 0)
      \param brakeForceRange - brake force range
      \sa setBrakeForce
      */
      void setBrakeForceRange( agx::RangeReal brakeForceRange );

      /**
      Same as setBrakeForceRange( agx::RangeReal brakeForceRange ) but this method will set range as [-brakeForce, brakeForce].
      \param brakeForce - brake force (Newton) to define the total brake force range
      */
      void setBrakeForceRange( agx::Real brakeForce );

      /**
      \return the brake force range
      */
      inline const agx::RangeReal& getBrakeForceRange() const;

      /**
      \note that if all the wire is spooled out, the force reported will not be correct. Avoid this situation.
      \return the current brake force the winch brake is applying
      */
      virtual agx::Real getCurrentBrakeForce() const;

      /**
      Set the pulled in length for this winch.
      \note Explicitly assigning this value will affect the total length of the
            wire this winch is attached to.
      \param pulledInLength - pulled in length
      */
      void setPulledInWireLength( agx::Real pulledInLength );

      /**
      \return the pulled in length of the wire attached to the winch controller.
      */
      virtual agx::Real getPulledInWireLength() const;

      /**
      \return the fixed node (i.e., first or last node depending on which side this winch is on)
      */
      agxWire::BodyFixedNode* getFixedNode() const;

      /**
      \return the stop node
      */
      agxWire::StopNode* getStopNode() const;

      /**
      \return the rigid body
      */
      agx::RigidBody* getRigidBody() const;

      /**
      \return true if the stop node has been inserted into the wire - otherwise false
      */
      bool getStopNodeInserted() const;

      /**
      \return the reference position of the lump node given normal and fixed node
      */
      agx::Vec3 getLumpNodeReferencePosition() const;

      /**
      \return distance between lump reposition position and the stop node.
      */
      agx::Real getStopNodeReferenceDistance( ) const;

      /**
      Sets the default compliance for the prismatic constraint.
      \param prismatic - a prismatic constraint
      */
      void setCompliance( agx::Prismatic* prismatic ) const;

      /**
      Associate a stability controller to this winch. This winch will use the stability controller
      to calculate and set masses to the neighboring lumped nodes when hauling in and out wire. It
      makes it also possible for this winch to merge two lumps if the haul in speed is large.
      \param stabilityController - new stability controller
      */
      void setStabilityController( agxWire::WireStabilityController* stabilityController );

      /**
      Associate a parameter controller to this winch.
      */
      void setParameterController( agxWire::WireParameterController* parameters );

      virtual bool getWithinEndRange( ) const;

      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      Set to true to prevent this winch to pull in any more wire. The state will be changed
      if speed is set to pull out
      \note Internal method
      \param enableForcedBrake - enable or disable forced brake
      */
      void setEnableForcedBrake(bool enableForcedBrake);

      /**
      \note Internal method
      \return the current forced brake state
      */
      bool getEnableForcedBrake() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::WireWinchController );

      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Default constructor for serialization.
      */
      WireWinchController();

      virtual ~WireWinchController();

      friend class WireInitializationController;
      friend class WireRouteController;
      friend class Wire;

      /**
      Called from Wire::addNotification.
      */
      virtual void addNotification( agxSDK::SimulationProxy* /*simulationProxy*/ ) {}

      /**
      Initializes this winch (inserts the stop node and move the lump for example).
      \param begin - true if this attachment is at the beginning of the wire
      \param parameters - pointer to the parameter object
      */
      virtual void initialize( bool begin, WireParameterController* parameters ) override;

      /**
      Forgets about all wire parameters and clears the constraint.
      */
      virtual void unitialize();

      /**
      Create new nodes for new wire
      */
      virtual bool attach( WireDistanceCompositeConstraint* constraint, WireParameterController* parameters, bool begin  );

      /**
      Search the wire, given begin true of false, for the parameters holding node.
      If this winch is located at the beginning of the wire, the parameters holding
      node will be the first lump after m_lumpedNode. If at end, the parameters
      holding node is equal to the lumped node.
      \param begin - if this winch is located at the beginning or at the end of the wire
      \return the node which holds the rest/current length of its segment
      */
      BodyFixedNode* findParametersHoldingNode( bool begin );

      /**
      Inserts the stop wire node in the wire.
      \return true if the operation was successful (or if the node is already inserted) - otherwise false
      */
      bool insertStopNode( bool begin );

      /**
      This method will set the position of the lumped node given the stop node and pulled in length.
      \return the distance the lumped node moved during this initialization (distance < reference distance means that the range should be enabled)
      */
      agx::Real configureInitialState();

      /**
      Updates mass for lumped nodes where the rest length is changed by restLengthChange
      */
      void massUpdate( agx::Real restLengthChange, bool begin );

      /**
      Pre-collide update call from the wire.
      */
      virtual void preCollideUpdate( agx::Bool /*begin*/, agxSDK::SimulationProxy* /* simulationProxy */ ) {}

      /**
      Pre update call from the wire.
      */
      virtual void preUpdate( agx::Bool /*begin*/, agxSDK::SimulationProxy* /* simulationProxy */ ) {}

      /**
      Called when it's time to update the constraint and reset the position of the lump.
      */
      virtual void update( Node* controlledNode, Node* lumpedNode, bool begin ) override;

      /**
      Post update call from the wire.
      */
      virtual void postUpdate( agx::Bool /*begin*/, agxSDK::SimulationProxy* /* simulationProxy */ ) {}

      /**
      For the wire to know if \p node belongs to this attachment.
      \param node - node
      \return true if \p node is part of this attachment
      */
      virtual bool owns( const Node* node ) const override
      {
        return m_stopNode == node;
      }

      /**
      \return true if our stop node is on the wire
      */
      virtual Node* getReferenceNode() override { return m_stopNode; }

      /**
      \return current length between lumped node and stop node
      */
      virtual agx::Real getReferenceCurrentLength( bool begin ) const override;

      /**
      Creates a prismatic constraint given the fixed-, lumped node and a normal.
      */
      agx::Prismatic* createPrismatic( Node* fixedNode, Node* lumpedNode, const agx::Vec3& worldNormal ) const;

      /**
      Updates motor, lock and range parameters.
      \param prismatic - the prismatic constraint
      \param desiredSpeed - desired speed of the prismatic motor
      \param forceRange - force range of the motor
      \param brakeRange - force range of the lock
      \param rangeEnable - if pulled in length < reference distance the range should be enabled
      \param forcedBrake - true for "emergency stop" of this winch, brake force will be infinite
      */
      void updateConstraintSettings( agx::Prismatic* prismatic, agx::Real desiredSpeed, const agx::RangeReal& forceRange, const agx::RangeReal& brakeRange, bool rangeEnable, bool forcedBrake = false ) const;

      /**
      Resets the lump position to reference.
      \param angle - the current angle of the constraint
      \return change in angle (i.e., the amount the rest and current (?) length should change
      */
      agx::Real resetLumpPosition( agx::Real angle );

      /**
      Finds the next (or previous) lumped node given a lumped node and the direction of the search.
      \param from - the search will start from this body fixed wire node
      \param begin - direction of the search, begin == true the search will be forward, false backward (in the list)
      \return the next lumped node - zero of no lump was found
      */
      BodyFixedNode* getNextLumpedNode( BodyFixedNode* from, bool begin );

      virtual void forcePulledInLength( agx::Real newLength );

      virtual void incrementPulledInLength( agx::Real dl );

    protected:
      bool                      m_autoFeed;
      agx::Real                 m_speed;
      agx::RangeReal            m_forceRange;
      agx::RangeReal            m_brakeRange;
      agx::Real                 m_pulledInLength;
      bool                      m_withinEndRange;
      bool                      m_forcedBrake;

      BodyFixedNodeRef                     m_fixedNode;
      StopNodeRef                          m_stopNode;
      agx::Vec3                            m_normal;
      agx::observer_ptr< BodyFixedNode >   m_lumpedNode;
      agx::observer_ptr< BodyFixedNode >   m_parametersHoldingNode; /**< To access and change rest/current length. If this winch is located at begin this will be the lumped node after m_lumpedNode, if at end m_lumpedNode == m_parametersHoldingNode. */

      agx::observer_ptr< WireStabilityController >  m_stabilityController; /**< If by the initialization controller, this winch can merge segments and change mass of the lumps via this controller. */

      WireParameterController*  m_parameters;
  };

  typedef agx::ref_ptr< WireWinchController > WireWinchControllerRef;
  typedef agx::Vector< WireWinchControllerRef > WireWinchControllerRefVector;
  typedef agx::Vector< agx::observer_ptr< WireWinchController > > WireWinchControllerObsVector;

  // Inline methods ***********************************************************************************

  inline const agx::Vec3& WireWinchController::getNormal() const
  {
    return m_normal;
  }

  inline bool WireWinchController::getAutoFeed() const
  {
    return m_autoFeed;
  }

  inline agx::Real WireWinchController::getSpeed() const
  {
    return m_speed;
  }

  inline const agx::RangeReal& WireWinchController::getForceRange() const
  {
    return m_forceRange;
  }

  inline const agx::RangeReal& WireWinchController::getBrakeForceRange() const
  {
    return m_brakeRange;
  }

  inline void WireWinchController::setEnableForcedBrake( bool enableForcedBrake )
  {
    m_forcedBrake = enableForcedBrake;
  }

  inline bool WireWinchController::getEnableForcedBrake() const
  {
    return m_forcedBrake;
  }

  // **************************************************************************************************

  class CylindricalCoordinateRPY
  {
    public:
      CylindricalCoordinateRPY()
      {
        set( 0, 0, 0 );
      }

      CylindricalCoordinateRPY( const agx::Vec3& v )
      {
        set( v.x(), v.y(), v.z() );
      }

      CylindricalCoordinateRPY( const CylindricalCoordinateRPY& c )
      {
        *this = c;
      }

      CylindricalCoordinateRPY& operator = ( const CylindricalCoordinateRPY& c )
      {
        _v[ 0 ] = c._v[ 0 ];
        _v[ 1 ] = c._v[ 1 ];
        _v[ 2 ] = c._v[ 2 ];

        return *this;
      }

      agx::Real r() const { return _v[ 0 ]; }
      agx::Real phi() const { return _v[ 1 ]; }
      agx::Real y() const { return _v[ 2 ]; }

      agx::Real& r() { return _v[ 0 ]; }
      agx::Real& phi() { return _v[ 1 ]; }
      agx::Real& y() { return _v[ 2 ]; }

      agx::Real x() const { return _v[ 0 ] * std::cos( convertAngle( _v[ 1 ] ) ); }
      agx::Real z() const { return _v[ 0 ] * std::sin( convertAngle( _v[ 1 ] ) ); }

      operator agx::Vec3 () const { return agx::Vec3( x(), y(), z() ); }

      void clampAngle()
      {
        if ( _v[ 1 ] < 0 )
          _v[ 1 ] += 2 * agx::PI;
        else if ( _v[ 1 ] > 2 * agx::PI )
          _v[ 1 ] -= 2 * agx::PI;
      }

    protected:
      void set( agx::Real x, agx::Real y, agx::Real z )
      {
        // Rho or radius.
        _v[ 0 ] = std::sqrt( x * x + z * z );
        agx::Real angle = std::atan2( z, x );
        _v[ 1 ] = angle < 0 ? 2 * agx::PI + angle : angle;
        _v[ 2 ] = y;
      }

      agx::Real convertAngle( agx::Real full ) const
      {
        agxAssert( !(full < 0) );
        return full > agx::PI ? full - 2 * agx::PI : full;
      }

      agx::Real _v[3];
  };

  struct WireSimpleDrumControlPoint
  {
    WireSimpleDrumControlPoint() : coordinate(), restLength( 0 ), currentLength( 0 ) {}
    WireSimpleDrumControlPoint( const agx::Vec3& position, agx::Real rest, agx::Real current )
      : coordinate( position ), restLength( rest ), currentLength( current ) {}
    WireSimpleDrumControlPoint( const WireSimpleDrumControlPoint& p )
    {
      *this = p;
    }
    WireSimpleDrumControlPoint& operator = ( const WireSimpleDrumControlPoint& p )
    {
      coordinate = p.coordinate;
      restLength = p.restLength;
      currentLength = p.currentLength;

      return *this;
    }

    void store( agxStream::OutputArchive& out ) const;
    void restore( agxStream::InputArchive& in );

    CylindricalCoordinateRPY coordinate; // Cylindrical coordinate system (r, phi, y), to fit with cylinder shape axis.
    agx::Real restLength;
    agx::Real currentLength;
  };

  typedef agx::List< WireSimpleDrumControlPoint > WireSimpleDrumControlPointList;

  class AGXPHYSICS_EXPORT WireSimpleDrumController : public WireWinchController, public virtual agxSDK::Assembly
  {
    public:
      WireSimpleDrumController( agx::Real radius, agx::Real length, size_t maxNumActiveContactNodes = 5, bool upsideDown = false );

      /**
      Attach this simple drum to a rigid body using an attachment frame.
      \param rb - rigid body to attach to (if 0 this drum will be attached to the world)
      \param attachmentFrame - the attachment frame defining the hinge axis and center
      \return true if the attach was successful
      */
      bool attach( agx::RigidBody* rb, agx::Frame* attachmentFrame );

      virtual agx::Real getCurrentSpeed() const override;
      virtual agx::Real getCurrentForce() const override;
      virtual agx::Real getCurrentBrakeForce() const override;
      const WireSimpleDrumControlPointList& getControlPoints() const { return m_controlPoints; }
      agx::RigidBody* getRigidBody() const { return m_rb; }
      agxCollide::Geometry* getGeometry() const { return m_geometry; }
      agx::Hinge* getHinge() const { return m_hinge; }
      using WireWinchController::getConstraint;


      AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::WireSimpleDrumController );

    protected:
      WireSimpleDrumController();
      virtual ~WireSimpleDrumController() {}

      virtual void initialize( bool begin, WireParameterController* parameters ) override;
      void initializeControlPoints( WireSimpleDrumControlPointList& controlPoints ) const;
      virtual void update( Node* controlledNode, Node* lumpedNode, bool begin ) override;

      void updateHinge( agx::Hinge* h ) const;

      size_t getNumContacts( bool begin, WireDistanceCompositeConstraint* wire, NodeIterator& lastContact, NodeIterator& firstLumpIter ) const;
      void createControlPoints( NodeConstIterator from, NodeConstIterator to, size_t numNodes, agx::Real& restLength, agx::Real& currentLength );
      ContactNode* rollOut( agx::Real& dRestLength, agx::Real& dCurrentLength );
      void rollIn( bool begin, NodeIterator from, size_t numContacts, agx::Real& dRestLength, agx::Real& dCurrentLength );
      bool changeBodyFixedToContactNode( BodyFixedNode* bfn );
      void rebind( const Node* to );
      bool removeContactNodes( NodeIterator from, NodeIterator to );

      agx::Real getControlPointRadius() const;

    private:
      using WireWinchController::attach;

    protected:
      friend class WireSimpleDrumRenderer;

      size_t m_maxNumActiveContactNodes;
      agxCollide::GeometryRef m_geometry;
      agx::RigidBodyRef m_rb;
      agx::HingeRef m_hinge;
      WireSimpleDrumControlPointList m_controlPoints;
  };

  typedef agx::ref_ptr< WireSimpleDrumController > WireSimpleDrumControllerRef;


  DOXYGEN_START_INTERNAL_BLOCK()
  class WireSimpleDrumRenderer : public ConstraintRenderer
  {
    public:
      WireSimpleDrumRenderer( WireSimpleDrumController* drum )
        : m_drum( drum ) {}

      virtual void render( agxRender::RenderManager *mgr, float scale );

    private:
      void createCylinder( agxRender::RenderManager *mgr , const agx::Vec3& p1, const agx::Vec3& p2, float radius );

    private:
      agx::observer_ptr< WireSimpleDrumController > m_drum;
  };
  DOXYGEN_END_INTERNAL_BLOCK()

}

