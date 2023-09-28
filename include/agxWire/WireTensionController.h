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

#include <agxWire/WireDistanceCompositeConstraint.h>
#include <agxWire/WireTensionData.h>

#include <agxUtil/SmoothingFilter.h>

namespace agxWire
{
  class WireDistanceCompositeConstraintImplementation;
  class WireDistanceConstraintImplementation;

  class AGXPHYSICS_EXPORT WireTensionController : public agxWire::WireListener
  {
    public:
      /**
      \return the tension in the wire given a node in this wire
      */
      agxWire::WireNodeTensionData getTension( const agxWire::Node* node ) const;

      /**
      \return the tension in the wire after distance
      */
      agxWire::WireSegmentTensionData getTension( agx::Real distanceFromStart ) const;

      /**
      Assign smoothing filter for smoothing of tension values.
      \param smoothingFilter - smoothing filter
      */
      void setSmoothingFilter( agxUtil::SmoothingFilter* smoothingFilter );

      /**
      \return the smoothing filter used to smooth tension values
      */
      agxUtil::SmoothingFilter* getSmoothingFilter() const;

    protected:
      /**
      Reference counted object - protected destructor.
      */
      virtual ~WireTensionController();

      virtual void onAdd( agxWire::NodeIterator nodeIt ) override;
      virtual void onInsert( agxWire::NodeIterator nodeIt ) override;
      virtual void onRemove( agxWire::NodeIterator nodeIt ) override;
      virtual void postReverse() override;
      virtual void preSolve() override;
      virtual void postSolve() override;

    private:
      friend class WireDistanceCompositeConstraintImplementation;
      /**
      Default hidden constructor. May only be constructed by WireDistanceCompositeConstraintImplementation.
      */
      WireTensionController( agxWire::WireDistanceCompositeConstraintImplementation* wire );

      /**
      \return the tension in the wire given a node iterator in this wire
      */
      agxWire::WireNodeTensionData getTension( agxWire::NodeConstIterator node ) const;

      friend class WireDistanceConstraintImplementation;
      /**
      From postSolveCallback, assigns raw tension from the distance constraint.
      */
      static void updateTension( const agxWire::WireDistanceConstraintImplementation* constraint );

    private:
      WireDistanceCompositeConstraintImplementation* m_wire;
      agxUtil::SmoothingFilterRef m_smoothingFilter;
  };

  typedef agx::ref_ptr< WireTensionController > WireTensionControllerRef;
}

