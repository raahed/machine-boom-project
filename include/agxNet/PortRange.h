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

#include <agx/Referenced.h>
#include <agx/Vector.h>

namespace agxNet
{
  AGX_DECLARE_POINTER_TYPES(PortRange);

  /**
  Handles testing of ports in a range. Return ports in the range until the range is depleted.
  */
  class AGXCORE_EXPORT PortRange : public agx::Referenced
  {
  public:
    typedef agx::UInt16 Port;

  public:
    PortRange(Port startPort, Port endPort);

    /// \return The start port
    Port getStartPort() const;

    /// \return The end port
    Port getEndPort() const;


    /**
    \return The next port to test. Return InvalidIndex when all ports have been iterated.
    */
    Port getPort();

    /**
    Reset the range.
    */
    void reset();

  protected:
    ~PortRange();

  private:
    Port m_startPort;
    Port m_endPort;
    agx::Vector<Port> m_availablePorts;
  };
}

