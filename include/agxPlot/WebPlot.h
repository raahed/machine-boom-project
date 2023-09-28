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

#pragma once

#include <agx/config/AGX_USE_WEBPLOT.h>

#if AGX_USE_WEBPLOT()

#include <agxPlot/Output.h>
#include <agxNet/WebSocket.h>

namespace agxPlot
{
  AGX_DECLARE_POINTER_TYPES(WebPlot);

  class AGXPHYSICS_EXPORT WebPlot : public agxPlot::Output
  {
    public:
      WebPlot(bool openBrowser = false, bool useDefaultPorts = true);

      agx::UInt16 getPort() const;

      virtual void handlePacket(DataPacket *packet) override;
      virtual void handlePacket(DescriptionPacket *packet) override;
      virtual void handlePacket(TimePacket *packet) override;
      virtual bool isReady() override;
      virtual void closeOutput()  override;

      void removeCurve(CurveID curveID);

    protected:
      virtual ~WebPlot();

    private:
      void clientConnected(agxNet::WebSocket* client);
      void clientDisconnected(agxNet::WebSocket* client);

      agxNet::StructuredMessage *createMessage(DataPacket *packet);
      agxNet::StructuredMessage *createMessage(DescriptionPacket *packet);
      agxNet::StructuredMessage *createMessage(TimePacket *packet);
    private:
      agxNet::WebSocketServerRef m_socketServer;
      DescriptionPacketRef m_currentDescription;
      agx::Vector<DataPacketRef> m_dataHistory;
  };
}


// End AGX_USE_WEBPLOT()
#endif
