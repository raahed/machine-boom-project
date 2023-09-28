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

#ifndef AGXPLOT_DESCRIPTIONPACKET_H
#define AGXPLOT_DESCRIPTIONPACKET_H

#include <agxPlot/Packet.h>
#include <external/json/value.h>

namespace agxPlot
{

  class DescriptionPacket : public Packet
  {
  public:
    DescriptionPacket(const agxJson::Value& description);

    agxJson::Value& getDescription();
  protected:
    virtual ~DescriptionPacket();

  private:
    agxJson::Value m_description;
  };
  AGX_DECLARE_POINTER_TYPES(DescriptionPacket);
}

#endif
