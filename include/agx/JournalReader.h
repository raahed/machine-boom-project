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

#ifndef AGX_JOURNAL_READER_H
#define AGX_JOURNAL_READER_H

#include <agx/config/AGX_USE_WEBSOCKETS.h>

#if AGX_USE_WEBSOCKETS()

#include <agx/Journal.h>
#include <agx/ConfigSingleton.h>
#include <agxData/FrameTranspose.h>
#include <agxData/FileTrack.h>
#include <agxIO/ArgumentParser.h>
#include <agxIO/RemoteCommandServer.h>
#include <agxNet/WebSocketTrack.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(JournalReader);
  class AGXPHYSICS_EXPORT JournalReader : public Referenced
  {
    public:
      JournalReader( const agx::String& journalSearchPath, agxNet::WebSocket::ControlChannel *controlChannel, agx::JournalRef& journal);

      // bool httpListJournals(agxIO::RemoteCommandServer* server, mg_connection* connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
      // bool httpLoadJournal(agxIO::RemoteCommandServer* server, mg_connection* connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
      // bool httpListSessions(agxIO::RemoteCommandServer* server, mg_connection* connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
      // bool httpInspectSession(agxIO::RemoteCommandServer* server, mg_connection* connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
      // bool httpAttachRemoteViewer(agxIO::RemoteCommandServer* server, mg_connection* connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
      // bool httpSelectBuffer(agxIO::RemoteCommandServer* server, mg_connection* connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
      //
      // bool httpQuit(agxIO::RemoteCommandServer* server, mg_connection* connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
      // bool httpGetControlChannel(agxIO::RemoteCommandServer* server, mg_connection* connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

      void blockUntilDone();

      bool loadSession(const agx::String& journalPath, const agx::String& sessionName);
      // void setJournal(agx::Journal *journal) { m_journal = journal; }

      bool checkAndResetPlotSavedState();

    protected:
      virtual ~JournalReader() {}
      JournalReader& operator=(const JournalReader&) { agxAbort(); return *this; }

    private:
      void mirrorFrame( agxData::Frame* source, agxJson::Value& destination );
      void addChildren( agx::Component* source, agxJson::Value& destination, const agx::String& path );
      void addDispatch( agx::Object* source, agxJson::Value& destination, const agx::String& path );
      void add( agx::Component* source, agxJson::Value& destination, const agx::String& path );
      void add( agxData::Buffer* source, agxJson::Value& destination, const agx::String& path );
      void add( agxData::Value* source, agxJson::Value& destination, const agx::String& path );
      void add( agxData::Frame::EntityStorage* source, agxJson::Value& destination, const agx::String& path );
      agxJson::Value& createArrayElement( agxJson::Value& eParent, const char* groupName );
      void ccGetPlotData( agxNet::WebSocket::ControlChannel* /*channel*/, agxNet::WebSocket* /*socket*/, agxNet::StructuredMessage* message);
      void ccSavePlot( agxNet::WebSocket::ControlChannel* /*channel*/, agxNet::WebSocket* /*socket*/, agxNet::StructuredMessage* message);
      void ccAllPlotsSaved( agxNet::WebSocket::ControlChannel* /*channel*/, agxNet::WebSocket* /*socket*/, agxNet::StructuredMessage* message);
      void ccRemovePlot( agxNet::WebSocket::ControlChannel* /*channel*/, agxNet::WebSocket* /*socket*/, agxNet::StructuredMessage* message);
      void ccGetSavedPlots( agxNet::WebSocket::ControlChannel* /*channel*/, agxNet::WebSocket* /*socket*/, agxNet::StructuredMessage* message);
      void ccGetPlotList( agxNet::WebSocket::ControlChannel* /*channel*/, agxNet::WebSocket* /*socket*/, agxNet::StructuredMessage* message);
      void ccRequestSimulationTrack( agxNet::WebSocket::ControlChannel* /*channel*/, agxNet::WebSocket* /*socket*/, agxNet::StructuredMessage* message);
      void ccGetTreeStructure( agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
      void ccGetBuffer( agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);

    private:
      agx::String m_journalSearchPath;
      agx::JournalRef& m_journal;
      bool m_plotSaved;

      agxNet::WebSocketFrameWriterRef m_frameSocket;
      agxNet::WebSocket::ControlChannelRef m_controlChannel;
      bool m_running;
  };

}

#endif

#endif
