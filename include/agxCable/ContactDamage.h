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


#include <agxCable/SegmentDamageState.h>
#include <agxCable/export.h>

namespace agxCable
{
  class Cable;
  class SegmentDamageState;

  /**
  Compute the contact forces part of the SegmentDamageState for each cable
  segment of the given cable.

  \param cable - The cable from which the current state should be computed.
  \param damages - The SegmentDamageStates in which the computed state should be stored.
  */
  AGXCABLE_EXPORT void computeContactForceState(const Cable& cable, SegmentDamageStateVector& damages);

  /**
  Compute the impact speed part of the SegmentDamageState for each cable segment
  of the given cable.

  \param cable - The cable from which the current state should be computed.
  \param damages - The SegmentDamageStates in which the computed state should be stored.
  */
  AGXCABLE_EXPORT void computeImpactSpeedState(const Cable& cable, SegmentDamageStateVector& damages);
}
