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

#include <agxWire/Wire.h>
#include <agxWire/Link.h>

namespace agxWire
{
  /**
  Winch object which works like agxWire::WireWinchController but enables the
  features of having stored, inactive, wire segments inside of the winch which
  can be pulled out given a link. It also features the possibility of completely
  pulling in a wire segment with a link connection.
  */
  class AGXPHYSICS_EXPORT Winch : public agxWire::WireWinchController
  {
    public:
      /**
      Construct given rigid body, relative translate (rigid body frame), relative normal (rigid body frame) and
      initial pulled in length. If the rigid body is null the winch will be attached in world and hence the
      relative translate and normal is given in world coordinates.
      */
      Winch( agx::RigidBody* rb, const agx::Vec3& relativeTranslate, const agx::Vec3& relativeNormal, agx::Real pulledInLength = agx::Real( 0 ) );

      /**
      Add inactive wire segment with a given length.
      \note Remember to connect this wire to a link!
      \param wire - inactive wire that is fully pulled in to this winch
      \param length - length of the wire
      \return true if added, false if already added, null or length less than or equal to zero.
      */
      agx::Bool add( agxWire::Wire* wire, agx::Real length );

      /**
      Remove an inactive wire segment.
      \return true if removed, otherwise false
      */
      agx::Bool remove( const agxWire::Wire* wire );

      /**
      \return the number of pulled in wires
      */
      agx::UInt getNumPulledInWires() const;

      /**
      Calculates the total pulled in length, including the pulled in length of the
      currently active wire (i.e., getPulledInlength()).
      \note getPulledInLength will only return the pulled in length of the currently
            active wire, handled by the agxWire::WireWinchController.
      \return the total pulled in length including all inactive wires and the current
              amount pulled in of the active wire (i.e., getPulledInLength()).
      */
      agx::Real getTotalPulledInLength() const;

      /**
      Finds rest length of \p wire if this winch is either attached to \p wire
      or when \p wire is completely pulled in to this winch. If the wire isn't
      associated to this winch in any way this method returns -1.
      \param wire - wire associated to this winch
      \return rest length of the wire or -1 if the wire isn't associated to this winch.
      */
      agx::Real findRestLength( const agxWire::Wire* wire ) const;

    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::Winch );

    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Internal data to store the inactive wire and the length of that wire.
      */
      struct AGXPHYSICS_EXPORT WireLength
      {
        /**
        \return an invalid version of this struct
        */
        static WireLength invalid() { return WireLength( nullptr, agx::Real( -1 ) ); }

        /**
        Construct given a wire and a length.
        */
        WireLength( agxWire::Wire* w, agx::Real len ) : wire( w ), length( len ) {}

        /**
        \return true if valid (wire != 0 and length >= 0).
        */
        agx::Bool isValid() const { return wire != nullptr && length >= agx::Real( 0 ); }

        agxWire::WireRef wire;
        agx::Real        length;
      };

    protected:
      typedef agx::Vector< WireLength > PulledInWiresContainer;

    protected:
      /**
      Protected default constructor, used during restore.
      */
      Winch();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Winch();

      /**
      \return true if the near-wire shouldn't have any geometries active
      */
      virtual bool getForceGeometryListEmpty( bool begin ) const override;

      /**
      Initializes the inactive links.
      */
      virtual void addNotification( agxSDK::SimulationProxy* simulationProxy ) override;

      /**
      Checks for 'forced brake condition' where the winch is automatically stopped due
      to an unsupported approaching node. E.g., not a link.
      */
      virtual void preUpdate( agx::Bool begin, agxSDK::SimulationProxy* simulationProxy ) override;

    protected:
      friend class LinkWinchSliderAlgorithm;
      /**
      Pull in the near wire completely. Moves this winch to the far wire, disables the link
      and stores the near wire and the given length it has.
      */
      virtual agx::Bool pullInWire( agxWire::Winch::WireLength nearWireAndLength, agxWire::LinkRef link, agxWire::WireRef farWire, agxSDK::SimulationProxy* simulationProxy );

      friend class LinkWinchIdleAlgorithm;
      /**
      Pulls out a new wire segment, if any inactive wire is connected to the link.
      */
      virtual agxWire::Winch::WireLength pullOutWire( agxWire::LinkRef link, agxWire::WireRef farWire, agxSDK::SimulationProxy* simulationProxy );

      friend class WinchWireInitializationController;
      /**
      Transfers this winch from far wire to a newly created near wire.
      */
      virtual void transfer( agxWire::ILinkNode* linkNode, agxWire::WireDistanceCompositeConstraint* wireConstraint, agxWire::WireParameterController* parameters, agxWire::WireStabilityController* stabilityController, agx::Bool begin );

      /**
      Other friend objects have the possibility to manipulate the pulled in length of
      the current segment. Use with caution.
      */
      virtual void manipulatePulledInLength( agx::Real newPulledInLength );

    protected:
      /**
      Pulling in a wire, replaces link's far node with this winch.
      */
      agx::Bool replaceLink( agxWire::Wire* farWire, agxWire::Link* link );

      /**
      Finds the wire the connection node matches and returns that wire including its length.
      It also erases the completely pulled in wire from the list of inactive wires.
      */
      agxWire::Winch::WireLength getAndEraseFromPulledInWires( const agxWire::ConnectingNode* connectingNode );

      /**
      Returns that wire including its length. It also erases the completely pulled
      in wire from the list of inactive wires.
      */
      agxWire::Winch::WireLength getAndEraseFromPulledInWires( const agxWire::Wire* wire );

      /**
      Initializes the inactive links when this winch is added to the simulation.
      */
      void initializePulledInLinks( agxSDK::SimulationProxy* simulationProxy );

      /**
      Handles the approaching nodes during preUpdate.
      */
      void handleApproachingNodes( agx::Bool begin, agx::Real dt );

    protected:
      PulledInWiresContainer m_pulledInWires;
  };

  typedef agx::ref_ptr< Winch > WinchRef;
}


