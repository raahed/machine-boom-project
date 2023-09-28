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

#ifndef AGXPLOT_QUEUE_H
#define AGXPLOT_QUEUE_H

#include <agxPlot/Packet.h>

#include <agx/List.h>
#include <agx/Callback.h>

#include <agx/ThreadSynchronization.h>

namespace agxPlot
{
  class Queue : public agx::Referenced
  {
    public:
      Queue();

      void push(Packet *packet);
      PacketRef pop();
      bool empty() const;
      // TODO: Max queue size + blocking

    private:
      agx::List<PacketRef> m_packetQueue;
      agx::Mutex m_mutex;
  };

  AGX_DECLARE_POINTER_TYPES(Queue);
  AGX_DECLARE_VECTOR_TYPES(Queue);
}

#endif
