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
#include <agxSDK/Simulation.h>
#include <agx/Referenced.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agx
{
  class MergedBody;
}

namespace agxWire
{
  class Wire;

  typedef agx::ref_ptr< class WireSimplifyInterface > WireSimplifyInterfaceRef;

  /**
  Interface class for wire simplify.
  */
  class AGXPHYSICS_EXPORT WireSimplifyInterface : public agx::Referenced
  {
    public:
      typedef agx::Vector< WireSimplifyInterfaceRef > SimplifyCallbackContainer; /**< Container for listeners. */

    public:
      /**
      Base implementation of simplify. Will store parent frame, update
      current last transform and fire simplify event to all listeners.
      \param parentFrame - parent frame that is reference for node position update (world if null)
      \return true if successful
      */
      virtual bool simplify( const agx::Frame* parentFrame );

      /**
      Base implementation of unsimplify. Will fire unsimplify event to all listeners.
      \return true if successful
      */
      virtual bool unsimplify();

      /**
      Base implementation of reset. Will fire reset event to all listeners,
      remove reference to parent frame and simulation, and reset the
      last stored transform.
      */
      virtual void reset();

    public:
      /**
      Base implementation of pre-collide. Will fire preCollide event to all listeners
      and save the current parent frame matrix.

      Pre-collide event fired when the wire is simplified. I.e., the wire will
      not receive this event.
      \param simulation - simulation simplified from
      */
      virtual void preCollide( const agxSDK::Simulation* simulation );

      /**
      Base implementation of pre. Will fire pre event to all listeners.

      Pre event fired when the wire is simplified. I.e., the wire will
      not receive this event.
      \param simulation - simulation simplified from
      */
      virtual void pre( const agxSDK::Simulation* simulation );

      /**
      Base implementation of post. Will fire post event to all listeners.

      Post event fired when the wire is simplified. I.e., the wire will
      not receive this event.
      \param simulation - simulation simplified from
      */
      virtual void post( const agxSDK::Simulation* simulation );

    public:
      /**
      Add child/callback/listener to this simplify algorithm.
      \param simplifyCallback - simplify callback to add
      */
      void add( agxWire::WireSimplifyInterfaceRef simplifyCallback );

      /**
      Remove child/callback/listener to this simplify algorithm.
      \param simplifyCallback - simplify callback to remove
      */
      void remove( agxWire::WireSimplifyInterfaceRef simplifyCallback );

      /**
      Removes all callbacks.
      */
      void removeAllCallbacks();

      /**
      \return the list of all callbacks associated to this simplify algorithm
      */
      const SimplifyCallbackContainer& getCallbacks() const;

      /**
      \return the simulation simplified from
      */
      agxSDK::Simulation* getSimplifiedFrom() const;

    protected:
      /**
      Protected default constructor.
      */
      WireSimplifyInterface();

      /**
      Reference counted object - protected destructor.
      */
      virtual ~WireSimplifyInterface();

      /**
      Assign last transform.
      \param matrix - last transform matrix
      */
      void setLastTransform( const agx::AffineMatrix4x4& matrix );

      /**
      \return the last transform
      */
      const agx::AffineMatrix4x4& getLastTransform() const;

      /**
      Assign simulation simplified from (i.e., before remove). When unsimplified
      the wire will be added to this simulation.
      \param simulation - simulation simplified from
      */
      void setSimplifiedFrom( agxSDK::Simulation* simulation );

    protected:
      agx::FrameConstRef         m_parentFrame;

    private:
      agx::AffineMatrix4x4       m_lastTransform;
      SimplifyCallbackContainer  m_callbacks;
      agxSDK::SimulationObserver m_simplifiedFrom;
  };

  /*
  Wire simplify.

  A wire will keep its current state when removed from a simulation:

  simulation->remove( wire );
  ...
  // Keep rendering.
  for ( auto i = wire->getRenderBeginIterator()... )

  Adding the wire to the simulation again will 'enable' it again (unsimplify).

  So, it's possible to simply remove a wire from the simulation, and for
  example keep rendering it, for easiest and most effective simplification. Note
  that the nodes wont be updated automatically.

  If one want the wire to support all types of actions (cut, merge, contacts etc etc),
  and move/follow a parent frame, this class can be used. Note that this class is
  member of agxWire::Wire (accessed agxWire::Wire::getSimplifyController()). Example:

  const agx::Frame* parentFrame = ship->getFrame();
  wire->simplify( parentFrame ); // Will remove the wire from the simulation but keep updating geometry and node positions.
  bool simplified = wire->isSimplified(); // True!
  wire->unsimplify(); // Will add the wire again. Wire will start to interact again.

  When the wire is simplified it will unsimplify itself if:
    * A new node is inserted (e.g., valid contact).
    * An unsimplified wire is merged with it.
  */
  class AGXPHYSICS_EXPORT WireSimplifyController : public agxWire::WireSimplifyInterface
  {
    public:
      /**
      Garbage collects simplified wires. Since the wires aren't part of the simulation
      the simulation will call this method when it's being cleared.
      */
      static void garbageCollect( const agxSDK::Simulation* originalSimulation );

      /**
      If a wire is simplified then explicitly removed from the simulation, a call to this
      method will validate and change the state of the simplify controller to a valid one.
      \param wire - wire to check
      */
      static void validateRemoveState( agxWire::Wire* wire );

    public:
      enum State
      {
        SIMPLIFYING     = ( 1 << 0 ), /**< Simplifying state - when the wire is removed. */
        SIMPLIFIED      = ( 1 << 1 ), /**< Simplifying was successful - fully simplified wire. */
        UNSIMPLIFYING   = ( 1 << 2 ), /**< When a previously simplified wire is added again. */
        NONE            = ( 1 << 3 ), /**< Default state. Interacting and in a simulation. */
        COLLIDING       = ( 1 << 4 ), /**< When a node has been inserted, this state is added. */
        HAS_MERGED_BODY = ( 1 << 5 ), /**< When merged to an agx::MergedBody (doesn't say if the agx::MergedBody is in a simulation or not, see MERGED_ACTIVE). */
        MERGED_ACTIVE   = ( 1 << 6 )  /**< When the wire is considered completely merged and the agx::MergedBody is in a simulation. */
      };

    public:
      /**
      Construct given wire.
      \param wire - wire this controller is part of
      */
      WireSimplifyController( agxWire::Wire* wire );

      /**
      Utility method to check if this controller has a given state.
      */
      bool is( agx::Int state ) const;

      /**
      Initialized the geometry controller to handle geometries when simplified.
      */
      void initializeGeometryController();

      /**
      If state is MERGED, this method finds and returns the merged body. If not merged
      or only partially merged, this method returns null.
      \param onlyIfActive - true to only return an agx::MergedBody if the agx::MergedBody is active in a simulation,
                            false to access a merged body that even if the merged body isn't in a simulation
      \return merged body if state is MERGED
      */
      agx::MergedBody* getMergedBody( agx::Bool onlyIfActive ) const;

      /**
      Finds the parent body passed to the merge method. This method returns null if
      the body isn't found or if the wire isn't completely merged (state == MERGED).
      \param onlyIfActive - true to only return the parent body if the agx::MergedBody is active in a simulation,
                            false to access the parent body even if the merged body isn't in a simulation
      \return the parent body if state is MERGED, null if not found or not merged
      */
      agx::RigidBody* getMergedParentBody( agx::Bool onlyIfActive ) const;

    public:
      /**
      Clone given current state, parent transform, simulation.
      \param wireForClone - new wire for the clone
      \return clone of this simplify controller
      */
      virtual WireSimplifyController* clone( agxWire::Wire* wireForClone ) const;

      /**
      Simplify given parent frame (if null - world).
      \return true if successfully simplified
      */
      virtual bool simplify( const agx::Frame* parentFrame ) override;

      /**
      Unsimplify.
      \return true if successfully unsimplified
      */
      virtual bool unsimplify() override;

      /**
      Merge all lumped nodes to \p parentBody. After this the state will be
      merged and some functionality may be limited.
      */
      virtual bool merge( agx::RigidBody* parentBody, agx::MergedBody* mergedBody );

      /**
      If merged, all lumped nodes will be split from the parent.
      */
      virtual bool split();

      /**
      Explicit reset of current state.
      */
      virtual void reset() override;

      /**
      Save transform, position geometries for collision detection.
      */
      virtual void preCollide( const agxSDK::Simulation* simulation ) override;

      /**
      Called when a node is inserted.
      \param node - node that has been inserted
      */
      virtual void collide( const agxWire::Node* node );

      /**
      Change state from simplified to unsimplified if collide has been called.
      */
      virtual void pre( const agxSDK::Simulation* simulation ) override;

      /**
      Calculates relative transform and position the wire nodes given that.
      */
      virtual void post( const agxSDK::Simulation* simulation ) override;

      /**
      Stores internal states.
      */
      virtual void store( agxStream::OutputArchive& out ) const;

      /**
      Restores internal states.
      */
      virtual void restore( agxStream::InputArchive& in );

    protected:
      /**
      Reference counted object - protected destructor.
      */
      virtual ~WireSimplifyController();

    protected:
      /**
      Assign whole state.
      */
      void setState( agx::Int state );

      /**
      Add state to the current state.
      */
      void addState( State state );

      /**
      \return the current state
      */
      agx::Int getState() const;

      /**
      \return the wire
      */
      agxWire::Wire* getWire() const;

      /**
      If state HAS_MERGED_BODY but the parent body has been split/removed, this
      method will change state from HAS_MERGED_BODY to NONE.
      */
      void updateMergedState();

    protected:
      typedef agx::observer_ptr< agx::Referenced > WireType;

    protected:
      WireType                  m_wire;
      agx::Int                  m_state;
  };

  typedef agx::ref_ptr< WireSimplifyController > WireSimplifyControllerRef;
}

DOXYGEN_END_INTERNAL_BLOCK()
