/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
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

#include <agx/agxCore_export.h>
#include <agx/Integer.h>

namespace agx
{
  namespace GraphNode
  {
    enum Type
    {
      BODY = 0x0,
      INTERACTION = 0x1,
      PARTICLE = 0x2,
      GEOMETRY_CONTACT = 0x10 | INTERACTION,
      PARTICLE_CONTACT = 0x20 | INTERACTION,
      PARTICLE_GEOMETRY_CONTACT = 0x30 | INTERACTION,
      BINARY_CONSTRAINT = 0x40 | INTERACTION,
      MANY_BODY_CONSTRAINT = 0x50 | INTERACTION,
      STRONG_INTERACTION = 0x60 | INTERACTION
    };

    enum State
    {
      UNVISITED,
      SEEN,
      VISITED
    };

    enum LocalSolveComplexity
    {
      UNKNOWN,
      SPARSE,   /* Cheap part of the matrix (few or no fills), like a wire or a constrained row of bodies.*/
      DENSE     /* Expensive part of the matrix (many fills), like a pile of rocks with many contacts. */
    };

    extern const agx::Index NO_COLOR;
  }
}
