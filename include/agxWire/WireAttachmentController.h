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

#include <agx/Referenced.h>
#include <agx/Vector.h>


DOXYGEN_START_INTERNAL_BLOCK()

// Forward declarations
namespace agx
{
  class Constraint;
}

namespace agxWire
{
  // Forward declarations
  class WireParameterController;

  /**
  Base class for wire attachments.
  */
  class AGXPHYSICS_EXPORT WireAttachmentController : public virtual agx::Referenced
  {
    public:
      /**
      Default constructor.
      */
      WireAttachmentController();

      /**
      Update method called at any time this attachment must update its parameters and constraint.
      \param controlledNode - the route node, i.e., the node that defines this attachment
      \param lumpedNode - the lumped node this attachment should attach to
      \param begin - true if this attachment is located at the beginning of the wire
      */
      virtual void update( Node* controlledNode, Node* lumpedNode, bool begin );

      /**
      In order to find the begin and end geometry/render iterators, each attachment has to be
      able to report if \p node is part of the attachment. Default attachment doesn't have
      these kinds of nodes, but for example a winch of some sort may control a stop wire node
      and in this case \p node will be the stop wire node.
      \param node - the wire node
      \return true if this attachment owns \p node - otherwise false
      */
      virtual bool owns( const Node* node ) const;

      /**
      Only for custom attachments, like winches.
      \return node on the wire that defines where geometry/render iterator should start or end (for example)
      */
      virtual Node* getReferenceNode() { return nullptr; }

      /**
      It's possible for an attachment to force the geometry list to be empty.
      \return true to force the geometry list to be empty (equally render)
      */
      virtual bool getForceGeometryListEmpty( bool begin ) const;

      /**
      \return current length between any reference node and its lump
      */
      virtual agx::Real getReferenceCurrentLength( bool begin ) const;

      /**
      \return the current constraint this wire is attached with
      \note This constraint is zero if the wire isn't attached! (For example if this end is free)
      */
      inline agx::Constraint* getConstraint() const;

      /**
      \return the object able to debug render this attachment
      */
      inline ConstraintRenderer* getRenderer() const;

    protected:
      virtual ~WireAttachmentController();

      /**
      Set the constraint for this attachment (used by child class).
      \param constraint - a constraint
      */
      void setConstraint( agx::Constraint* constraint );

      friend class WireInitializationController;

      /**
      Initialization call before the composite constraint is initialized with all the constraints.
      \param begin - true if this attachment is at the beginning of the wire
      */
      virtual void initialize( bool begin, WireParameterController* parameters );

      /**
      Set the wire distance composite constraint for this attachment controller. Default
      attachment controllers do not have this set.
      \param wire - the wire distance composite constraint
      */
      void setWire( WireDistanceCompositeConstraint* wire );

      /**
      \return the distance composite constraint
      */
      WireDistanceCompositeConstraint* getWire() const;

      /**
      Finds the first (begin == true) or last (begin == false) on the wire.
      \return the first or last lumped node
      */
      BodyFixedNode* getLumpedNode( bool begin ) const;

      /**
      Create attachment constraint given two bodies. Default parameters will be assigned.
      */
      agx::Constraint* createConstraint( agx::RigidBody* rb1, agx::RigidBody* rb2 ) const;

      /**
      Delete current attachment constraint and set m_constraint to null.
      */
      void deleteConstraint();

    private:
      agx::ConstraintRef                       m_constraint;
      agx::observer_ptr< WireDistanceCompositeConstraint > m_wire;
      ConstraintRendererRef                                m_renderer;
  };

  typedef agx::ref_ptr< WireAttachmentController > WireAttachmentControllerRef;
  typedef agx::Vector< WireAttachmentControllerRef > WireAttachmentControllerRefVector;

  // Inline methods ************************************************************************

  inline agx::Constraint* WireAttachmentController::getConstraint() const
  {
    return m_constraint;
  }

  inline ConstraintRenderer* WireAttachmentController::getRenderer() const
  {
    return m_renderer;
  }

  inline WireDistanceCompositeConstraint* WireAttachmentController::getWire() const
  {
    return m_wire;
  }

  // ***************************************************************************************
}

DOXYGEN_END_INTERNAL_BLOCK()
