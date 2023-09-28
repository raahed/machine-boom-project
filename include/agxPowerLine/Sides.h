/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/


#ifndef AGXPOWERLINE_SIDES_H
#define AGXPOWERLINE_SIDES_H

#include <agxModel/export.h>

#include <cstddef>

namespace agxPowerLine
{
  class Unit;
  class Connector;

  enum Side
  {
    INPUT,
    OUTPUT,
    NO_SIDE
  };

  AGXMODEL_EXPORT agxPowerLine::Side opposite(agxPowerLine::Side side);
  AGXMODEL_EXPORT agxPowerLine::Side facing(agxPowerLine::Side side);
  AGXMODEL_EXPORT bool isValidSide(agxPowerLine::Side side, bool allowNoSide = false);

  AGXMODEL_EXPORT void getConnectedSides(
    agxPowerLine::Unit* unit, agxPowerLine::Connector* connector,
    agxPowerLine::Side& unitSide, agxPowerLine::Side& connectorSide);
}



#endif
