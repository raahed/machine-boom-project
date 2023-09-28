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

#ifndef AGXIO_REMOTE_COMMAND_SERVER_H
#define AGXIO_REMOTE_COMMAND_SERVER_H

#include <agx/config/AGX_USE_WEBSOCKETS.h>

#if AGX_USE_WEBSOCKETS()

#include <agx/Component.h>
#include <external/civetweb/civetweb.h>
#include <agxNet/PortRange.h>

#include <agx/Json.h>

namespace agxIO
{
  AGX_DECLARE_POINTER_TYPES(RemoteCommandServer);
  class AGXCORE_EXPORT RemoteCommandServer : public agx::Component
  {
    public:
      // Explicit/temporary etc. to handle e.g., Lua.
      class AGXCORE_EXPORT ScriptListener : public agx::Referenced
      {
        public:
          ScriptListener() {}

          /**
          Send \p command and get \p result and \p error back (given script and listener implementation).
          */
          virtual bool handle( const agx::String& command, agx::String& result, agx::String& error ) = 0;

          /**
          Fetch list of methods or similar used for auto complete.
          */
          virtual bool autoComplete( const agx::String& /*objOrType*/, agx::StringVector& /*result*/, agx::String& /*error*/ ) { return true; }

        protected:
          virtual ~ScriptListener() {}
      };

      typedef agx::ref_ptr< ScriptListener > ScriptListenerRef;
      typedef agx::Vector< ScriptListenerRef > ScriptListenerContainer;

      // typedef agx::HashTable<agx::String, agx::String> QueryArgumentTable;
      class QueryArgumentTable;

    public:
      static RemoteCommandServer *instance();
      static bool hasInstance();

    public:
      bool start(agxNet::PortRange *range, bool allowExternalRequests = false);
      bool start(agx::UInt16 port = 5656, bool allowExternalRequests = false);
      void stop();

      void service();

      agx::UInt16 getPort() const;

      bool isStarted() const;
      bool isUsingSSL() const;

      void setAuthenticationToken(const agx::String& token);
      const agx::String& getAuthenticationToken() const;

      bool add( agxIO::RemoteCommandServer::ScriptListener* scriptListener );
      bool remove( agxIO::RemoteCommandServer::ScriptListener* scriptListener );


      class AGXCORE_EXPORT RequestHandler
      {
      public:
        typedef std::function<bool (RemoteCommandServer *, mg_connection *, const QueryArgumentTable&)> CallbackFunction;

        RequestHandler() {}
        RequestHandler(const CallbackFunction& func) : m_callback(func) {}

        template<class ClassT>
        RequestHandler(bool (ClassT::*fun)(RemoteCommandServer *, mg_connection *, const QueryArgumentTable&), ClassT* obj) : m_callback(std::bind(fun, obj,std::placeholders:: _1,std::placeholders:: _2,std::placeholders:: _3)) {}

        bool isValid() const
        {
          return m_callback ? true : false;
        }

        bool operator() (RemoteCommandServer *server, mg_connection *connection, const QueryArgumentTable& arguments) const
        {
          return m_callback(server, connection, arguments);
        }

      private:
        CallbackFunction m_callback;
      };

      void registerHandler(const agx::String& uri, const RequestHandler& handler);
      void unregisterHandler(const agx::String& uri);

      static agx::String get_var(const char* buf, size_t buf_len, const char* name);
      static agx::String get_qsvar(const mg_connection* connection, const char* name);

      static void writeResponse(const agxJson::Value& eResponse, mg_connection* connection);

      // Safari ajax breaks if we do not write any response, use this if no response is sent from a handler
      static void writeDummyResponse(mg_connection *connection);

      void log(const char *format, ...);

    protected:
      RemoteCommandServer(const agx::Name& name = "RemoteCommandServer");
      virtual ~RemoteCommandServer();

      friend class RemoteCommandServerSingleton;

    private:
      bool handleAutoComplete(RemoteCommandServer *server, mg_connection *connection, const QueryArgumentTable& arguments);
      bool handleCommand(RemoteCommandServer *server, mg_connection *connection, const QueryArgumentTable& arguments);
      bool handleScript(RemoteCommandServer *server, mg_connection *connection, const QueryArgumentTable& arguments);
      bool handleScriptAutoComplete(RemoteCommandServer *server, mg_connection *connection, const QueryArgumentTable& arguments);
      bool handleJournalNavigationTree(RemoteCommandServer *server, mg_connection *connection, const QueryArgumentTable& arguments);
      bool handleJournalConfigurationList(RemoteCommandServer *server, mg_connection *connection, const QueryArgumentTable& arguments);
      bool readComponent(RemoteCommandServer *server, mg_connection *connection, const QueryArgumentTable& arguments);
      bool writeComponent(RemoteCommandServer *server, mg_connection *connection, const QueryArgumentTable& arguments);
      // bool handleComponentRequest(mg_connection *connection, const agx::String& path);



      agx::Object* findMatchingNode(agx::Path path, agx::Component* root, agx::Path& matchedPath, agx::StringVector& matchList);
      void writeNodeDescription(agx::Object* object, agxJson::Value& eNode);
      int handleHttpEvent(mg_connection* connection);
      int handleHttpAuth(mg_connection* connection);
      static int mgAuthCallback(mg_connection* connection, void* server);
      static int mgRequestCallback(mg_connection* connection, void* server);
      static bool handle_jsonp(mg_connection* connection, const mg_connection* request_info);
      void parseArguments(QueryArgumentTable& result, const mg_connection* connection);

      void writeTreeStructureJson(const agx::Object* node, agxJson::Value& eNode);


    private:
      agx::UInt16 m_port;
      mg_context *m_mongoose;
      agx::String m_htmlRoot;
      agx::String m_token;
      agx::ComponentRef m_activeContext;
      ScriptListenerContainer m_scriptListeners;
      bool m_allowExternalRequests;
      bool m_started;
      bool m_useSSL;


      typedef agx::HashTable<agx::String, RequestHandler> RequestHandlerTable;
      RequestHandlerTable m_requestHandlerTable;

      typedef agx::HashSet<agx::String> StringSet;
      StringSet m_allowedComponentFileTypes;
  };

  class RemoteCommandServer::QueryArgumentTable : public agx::HashTable<agx::String, agx::String>
  {
  public:

    using agx::HashTable<agx::String, agx::String>::operator[];

    // Overrider []-operator to return empty string if not inserted
    AGX_FORCE_INLINE const agx::String& operator[](const agx::String& key ) const
    {
      const_iterator it = find( key );
      static agx::String emptyString;
      return it != end() ? it->second : emptyString;
    }

  };

}

// End AGX_USE_WEBSOCKETS.
#endif

#endif /* AGXIO_REMOTE_COMMAND_SERVER_H */
