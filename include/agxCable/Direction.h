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

namespace agxCable
{
  /**
  The different directions in which a cable can be deformed.

  This is not a bit field, so the values cannot be or-ed together.

  \cond INTERNAL_DOCUMENTATION
  A bunch of code assumes that the enum literal values start at zero and
  are consecutive.
  \endcond
  */
  enum Direction
  {
    BEND,
    TWIST,
    STRETCH,
    NUM_DIRECTIONS,
    ALL_DIRECTIONS = NUM_DIRECTIONS
  };
}
