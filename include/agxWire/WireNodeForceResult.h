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

namespace agxWire
{
  /**
  Resulting force data from wire distance and bend
  calculateNodeForces calls.
  */
  struct WireNodeForceResult
  {
    enum Source
    {
      UNKNOWN,
      DISTANCE,
      FRICTION,
      BEND
    };

    WireNodeForceResult()
      : node( nullptr ), source( UNKNOWN ) {}

    Node* node;
    agx::Vec3 force;
    Source source;
  };

  typedef agx::VectorPOD<WireNodeForceResult> WireNodeForceResultVector;
}
