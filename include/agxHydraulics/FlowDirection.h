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

#ifndef AGXHYDRAULICS_FLOW_DIRECTION_H
#define AGXHYDRAULICS_FLOW_DIRECTION_H

#include <agxHydraulics/export.h>

#define AGXHYDRAULICS_FLOWDIRECTION_NONE 0
#define AGXHYDRAULICS_FLOWDIRECTION_FORWARD 0x1
#define AGXHYDRAULICS_FLOWDIRECTION_BACKWARD 0x2
#define AGXHYDRAULICS_FLOWDIRECTION_BOTH (AGXHYDRAULICS_FLOWDIRECTION_FORWARD|AGXHYDRAULICS_FLOWDIRECTION_BACKWARD)


namespace agxHydraulics
{
  /**
   * Enum describing the possible flow directions. Used, for example, when
   * configuring check and stop valves.
   */
  struct AGXHYDRAULICS_EXPORT FlowDirection
  {

    enum Literal
    {
      NONE = AGXHYDRAULICS_FLOWDIRECTION_NONE,
      FORWARD = AGXHYDRAULICS_FLOWDIRECTION_FORWARD,
      BACKWARD = AGXHYDRAULICS_FLOWDIRECTION_BACKWARD,
      BOTH = AGXHYDRAULICS_FLOWDIRECTION_BOTH
    };

    static const char* getName(Literal direction)
    {
      switch (direction) {
      case NONE:     return "NONE";     break;
      case FORWARD:  return "FORWARD";  break;
      case BACKWARD: return "BACKWARD"; break;
      case BOTH:     return "BOTH";     break;
      default:       return "INVALID";  break;
      }
    }
  };

}

#endif

