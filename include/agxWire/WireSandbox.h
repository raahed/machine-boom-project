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

#include <agxWire/Wire.h>


DOXYGEN_START_INTERNAL_BLOCK()

namespace agxWire
{
  /// Implements a Wire model with adaptive resolution and wire<->wire collisions enabled.
  class AGXPHYSICS_EXPORT WireSandbox : public agxWire::Wire
  {
    public:
      enum Mode
      {
        WIRE_WIRE = (1<<0),
        NEW_CONTACT_HANDLING = (1<<1),
        DEFAULT_MODE = WIRE_WIRE | NEW_CONTACT_HANDLING
      };

      WireSandbox( agx::Real radius, agx::Real resolutionPerUnitLength, Mode mode = DEFAULT_MODE );

      /**
      Clone this wire object and return an empty, uninitialized wire. Used during cut.
      \return a new empty wire
      */
      virtual Wire* clone();

    protected:
      virtual ~WireSandbox();

      virtual void addNotification();
      virtual void removeNotification();

      virtual void preCollide( const agx::TimeStamp& );
      virtual void pre( const agx::TimeStamp& );

    private:
      using agxWire::Wire::addNotification;
      using agxWire::Wire::removeNotification;
      using agxWire::Wire::clone;

    protected:
      Mode m_mode;
  };

  typedef agx::ref_ptr< WireSandbox > WireSandboxRef;

}

DOXYGEN_END_INTERNAL_BLOCK()
