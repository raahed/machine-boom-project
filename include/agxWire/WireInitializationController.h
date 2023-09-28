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
#include <agx/Referenced.h>
#include <agx/Vec3.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agx
{
  class Frame;
}

namespace agxWire
{
  // Forward declarations
  class Wire;
  class WireRouteController;
  class WireDistanceCompositeConstraint;
  class BodyFixedNode;

  /**
  Reference implementation of initialization of a wire given a wire route controller.
  */
  class AGXPHYSICS_EXPORT WireInitializationController : public agx::Referenced
  {
    public:
      /**
      Given the current route, this method will calculate the correct rolled in length for the winches and always returns the rest length
      of the wire.
      \param route - current route
      \return rest length of the wire
      */
      static agx::Real calculateWinchesPulledInLength( agxWire::WireRouteController* route );

      /**
      Create default lumped node for \p constraint.
      \param position - position of the lumped node
      \param restLength - initial rest length of the constraint segment
      \param currentLength - initial current length of the constraint segment
      \param constraint - the wire constraint
      \return a lumped node
      */
      static agxWire::BodyFixedNode* createLump( const agx::Vec3& position, agx::Real restLength, agx::Real currentLength, agx::Bool particle, agxWire::WireDistanceCompositeConstraint* constraint );

    public:
      WireInitializationController();

      /**
      Call to this method is made when the \p wire is being initialized.
      \param wire - wire to initialize
      \return the WireDistanceCompositeConstraint (constraint) configured and ready to be simulated. Returns 0
              of the initialization fails.
      */
      virtual agxWire::WireDistanceCompositeConstraint* initialize( agxWire::Wire* wire );

    protected:
      virtual ~WireInitializationController();

  };

  typedef agx::ref_ptr< WireInitializationController > WireInitializationControllerRef;
}

DOXYGEN_END_INTERNAL_BLOCK()
