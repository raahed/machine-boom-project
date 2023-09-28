/* Copyright 2007-2023. Algoryx Simulation AB.

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


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4355) // No warnings about using this in member initializer list
#endif

#include <agx/agx.h>
#include <iostream>
#include <agx/Vector.h>
#include <sstream>
#include <agx/HashTable.h>
#include <agx/Singleton.h>
#include <agx/LogStream.h>

#define THREAD_SAFE_NOTIFY 1

#if THREAD_SAFE_NOTIFY
#include <agx/ThreadSynchronization.h>
#endif

namespace agx
{

  DOXYGEN_START_INTERNAL_BLOCK()
  class Thread;
  class Notify;
  class ThreadDataHolder;
  DOXYGEN_END_INTERNAL_BLOCK()

  /// Callback class, derive from this and Register to the notify singleton to get callbacks
  class AGXCORE_EXPORT NotifyCallback : public agx::Referenced
  {
  public:
    NotifyCallback();
    virtual ~NotifyCallback();

    /// Virtual method called whenever there is a message ready
    virtual void message( const agx::String& msg, int notifyLevel) = 0;

  protected:

  };

  /// Class for handling logging of messages
  class AGXCORE_EXPORT Notify : public agx::Singleton
  {
    public:

      enum NotifyLevel { NOTIFY_DEBUG = 1 << 0, /**< DEBUG  (lowest) level. */
                         NOTIFY_INFO = 1 << 1,   /**< INFO level */
                         NOTIFY_WARNING = 1 << 2, /**< WARNING level */
                         NOTIFY_ERROR = 1 << 3, /**< ERROR (highest) level */
                         NOTIFY_END = 1 << 4,
                         NOTIFY_CLEAR = 1 << 5,
                         NOTIFY_PUSH = 1 << 6,
                         NOTIFY_LOGONLY = 1 << 7
                       };

      enum NotifySelection {
        PRINT_NONE = 0,    // Print nothing
        PRINT_FILE = 1,    // Print the filename
        PRINT_TAG = 2,     // Print the tag (ERROR, WARNING, INFO, DEBUG
        PRINT_LINE = 4,    // Print the linenumber
        PRINT_FUNCTION = 8,  // Print the functionname
        PRINT_SELECTION_END = 8,
        PRINT_DEFAULT = PRINT_TAG | PRINT_FUNCTION,
        PRINT_ALL = PRINT_FILE | PRINT_TAG | PRINT_LINE | PRINT_FUNCTION
      };

      /**
      Constructor
      \param out_stream - Connect the default output to a specified stream, default is std::cerr
      */
      Notify( std::ostream& out_stream = std::cerr);

      /// \return pointer to the singleton instance of a Notify
      static Notify* instance();

      /// \return a reference to the LogStream used for streaming messages to function callbacks, files..
      agx::LogStream& logStream();

      /// \return a reference to the LogStream used for streaming messages to function callbacks, files..
      const agx::LogStream& logStream() const;

      /// \return the current notify level
      NotifyLevel getNotifyLevel() const;

      /// Set the current notify level
      void setNotifyLevel( NotifyLevel level );

      /// Open a log file (in the LogStream)
      bool openLog( const agx::String& filename );

      /// Set the notify level for the LogStream
      void setLogNotifyLevel( NotifyLevel l );

      /// \return true if an exception should be thrown upon calling error?
      bool getThrowOnError() const;

      /// Specify whether we should Throw an exception upon calling error?
      void setThrowOnError( bool v );

      /**
      WIN32 specific: If set to true, a breakpoint will be set in debug build.
      This makes it easier to debug problems in VisualStudio.
      A call to this method has effect only if:
       - Built in VisualStudio
       - Build in DEBUG mode

       Observe, if this is set to true, NO exception will be generated if the above conditions are met.
      */
      void setBreakOnError( bool f );

      /// \return true if an a breakpoint should be set upon calling error
      bool getBreakOnError() const;


      /// Add a callback class that will be printed to if the notify level less or equal to l.
      void addCallback( NotifyCallback *callback, NotifyLevel l );

      /**
      Remove a callback from list of callbacks to report messages to.
      \return true if the stream was removed
      */
      bool removeCallback( NotifyCallback *callback );

      /**
      Set the notify level for a specific callback
      \return true if the stream exists
      */
      bool setNotifyLevel( NotifyCallback* callback, NotifyLevel l );

      /// Add a stream that will be printed to if the notify level less or equal to l, the memory will NOT be managed by this class.
      void addStream( std::ostream* stream, NotifyLevel l );

      /**
      Remove a stream from list of streams to report messages to. NO memory will be deallocated.
      \return true if the stream was removed
      */
      bool removeStream( std::ostream* stream );

      /**
      Set the notify level for a specific stream
      \return true if the stream exists
      */
      bool setNotifyLevel( std::ostream* stream, NotifyLevel l );

      /// Set the current output mode, do not use! (used internally)
      void setOutputMode( NotifyLevel l );

      /// \return the current notify level
      NotifyLevel getOutputMode() const;

      /// \return true if logging occurs to the LogStream ONLY.
      bool getToLogOnly() const;

      /// If \p f==true, logging will occur only to the LogStream, not to any of the associated ostreams.
      void setToLogOnly( bool f );

      std::ostream& outStream();

  private:

    template <class A, A val >
      class Mode
      {
        public:
#ifdef __APPLE__
          // Hm, some weird bug in Apples gcc doesn't like the original constructor
          Mode( Notify* notify ) : m_notify( notify ), m_level( val ) {}
#else
          Mode( Notify* notify, A l = val ) : m_notify( notify ), m_level( l ) {}
#endif
          Mode( const Mode& m ) : m_notify( m.m_notify ), m_level( m.m_level ) {}

          Notify::NotifyLevel getLevel() const {
            return m_level;
          }
          Notify* getNotify() {
            return m_notify;
          }
          const Notify* getNotify() const {
            return m_notify;
          }
        private:
          Notify* m_notify;
          Notify::NotifyLevel m_level;
          Mode() {}
      };

      DOXYGEN_START_INTERNAL_BLOCK()


      template <class A>
      class Modifier
      {
        public:
          Modifier( Notify* notify ) : m_notify( notify ), m_value( A( 0 ) ) {}
          Modifier( const Modifier& m ) : m_notify( m.m_notify ), m_value( m.m_value ) {}
          A getValue() {
            return m_value;
          }
          Modifier& setValue( A val ) {
            m_value = val;
            return *this;
          }

          Notify* getNotify() {
            return m_notify;
          }
          const Notify* getNotify() const {
            return m_notify;
          }

        private:
          Notify* m_notify;
          A m_value;
          Modifier() {}
      };
      DOXYGEN_END_INTERNAL_BLOCK()

      /// Class to store the fields that will be printed during a message
      class AGXCORE_EXPORT PrintSelection
      {
        public:
          PrintSelection( int mask );

          PrintSelection();

          PrintSelection( const PrintSelection& p );
          PrintSelection& operator=( const PrintSelection& p );

          ~PrintSelection();

          void clear();

          ///
          void setMask( int f );

          /// Set the given field to true
          void setMask( int f, bool val );

          bool isSet( int s ) const;

          bool isNoneSet() const;

        private:
          int m_selection_mask;
      };
      public:

        DOXYGEN_START_INTERNAL_BLOCK()

      typedef Mode<Notify::NotifyLevel, Notify::NOTIFY_END> End;
      typedef Mode<Notify::NotifyLevel, Notify::NOTIFY_CLEAR> Clear;
      typedef Mode<Notify::NotifyLevel, Notify::NOTIFY_LOGONLY> LogOnly;
      typedef Modifier<int> Push;

      SINGLETON_CLASSNAME_METHOD();

      template<typename T>
      Notify& operator<<(const T& rhs) {
        getThreadStringStream() << rhs;
        return *this;
      }

      typedef std::basic_ostream<char, std::char_traits<char> > CoutType;
      typedef CoutType& (*StandardEndLine)(CoutType&);

      /// define an operator<< to take in std::endl
      Notify& operator<<(StandardEndLine manip);

      operator std::ostream&() { return getThreadStringStream(); }

      Notify& operator<<( Notify::Clear& /*mode*/ ) {
        std::ostringstream& str = getThreadStringStream();
        str.str( "" );
        str.clear();
        return *this;
      }

      Notify& operator<<( Notify::End& modifier );
      Notify& operator<<( Notify::Push& modifier );
      Notify& operator<<( Notify::LogOnly& modifier );

      std::ios_base::fmtflags setf(std::ios_base::fmtflags newFormatFlags);


      Clear& clearString() ;

      End& end();

      Push& push();

      LogOnly& logOnly();


      void debug( const char* function, int line, const char* file, const agx::String& msg );
      void info( const char* function, int line, const char* file, const agx::String& msg  );
      void warning( const char* function, int line, const char* file, const agx::String& msg  );
      void error( const char* function, int line, const char* file, const agx::String& msg  );

      void debug( const agx::String& msg );
      void info( const agx::String& msg );
      void warning( const agx::String& msg );
      void error( const agx::String& msg );

      /// Push a new print selection stack
      void pushPrintSelection( const PrintSelection& p, NotifyLevel l );

      void pushPrintSelection( const PrintSelection& p );

      // Get the current print selection stack
      PrintSelection& getPrintSelection( NotifyLevel l );

      PrintSelection& getPrintSelection();

      /// Pop the print selection stack (always leave the last though
      bool popPrintSelection( NotifyLevel l );

      bool popPrintSelection();

      DOXYGEN_END_INTERNAL_BLOCK()


      agx::String buildString( );
      agx::String setDebugStrings( NotifyLevel l, const char* function, int line, const char* file );

      virtual ~Notify();

      void shutdown() override;

    protected:

      friend void agxCore::init();
      friend class agx::Thread;
      friend void agx::setNumThreads(size_t numThreads);
      friend class ThreadDataHolder;

      std::string str() const;

      void str( const std::string& text );

      void clear();

      End m_end;
      Clear m_clear;
      Push m_push;
      LogOnly m_log;

    private:

      typedef agx::HashTable<std::ostream*, NotifyLevel > StreamMap;
      StreamMap m_streams;

      typedef agx::HashTable<agx::ref_ptr<NotifyCallback>, NotifyLevel > CallbackMap;
      CallbackMap m_callbacks;

      std::ostringstream& getThreadStringStream();
      const std::ostringstream& getThreadStringStream() const;

      typedef Vector<PrintSelection> PrintSelectionStack;
      AGX_DECLARE_POINTER_TYPES(ThreadData);
      class ThreadData : public agx::Referenced {
      public:
          ThreadData();
          NotifyLevel output_mode;
          agx::String tag;
          int line;
          agx::String function;
          agx::String file;
          std::ostringstream stream;
          NotifyLevel notify_level;

          Vector<PrintSelectionStack> selection_stack;

        protected:
          virtual ~ThreadData();
      };

      ThreadData* getThreadData();
      const ThreadData* getThreadData() const;

      agx::LogStream m_logStream;

      bool m_to_log_only;
      bool m_throw_on_error;
      bool m_break_on_error;

      static Notify* s_instance;
      std::ostream& m_outStream;
      mutable agx::ReentrantMutex m_mutex;

  };


} // agx


#define ADD_NOTIFY_FUNCTIONALITY_TO_CLASS() \
  public:\
  void debug( const agx::String& msg ) { Notify::instance()->debug(msg); };\
  void info( const agx::String& msg ) { Notify::instance()->info(msg); };\
  void warning( const agx::String& msg ) { Notify::instance()->warning(msg); };\
  void error( const agx::String& msg ) { Notify::instance()->error(msg); };\
  protected:\
  void debug( const char* function, int line, const char* file, const agx::String& msg ) { Notify::instance()->debug(function,line,file,msg); };\
  void info( const char* function, int line, const char* file, const agx::String& msg ) { Notify::instance()->info(function,line,file,msg); };\
  void warning( const char* function, int line, const char* file, const agx::String& msg ) { Notify::instance()->warning(function,line,file,msg); };\
  void error( const char* function, int line, const char* file, const agx::String& msg ) { Notify::instance()->error(function,line,file,msg); };\
  ::agx::Notify::Push push() { return Notify::instance()->push(); };\
  ::agx::Notify::End end() { return Notify::instance()->end(); };\
  agx::String setDebugStrings( ::agx::Notify::NotifyLevel l, const char* function, int line, const char* file ) { return Notify::instance()->setDebugStrings(l,function,line,file); };\
  public:\
  ::agx::Notify& getNotify() {\
    ::agx::Notify* n = Notify::instance();\
    return *n;\
  } \
  const ::agx::Notify& getNotify() const {\
  ::agx::Notify* n = Notify::instance();\
  return *n;\
}


#ifdef AGX_DEBUG
#define NOTIFY_POS_ARGS __FUNCTION__,__LINE__,__FILE__
#else
#define NOTIFY_POS_ARGS "",-1,""
#endif

#define NOTIFY() getNotify()
#define NOTIFY_DEBUG() setDebugStrings(::agx::Notify::NOTIFY_DEBUG, NOTIFY_POS_ARGS)
#define NOTIFY_INFO() setDebugStrings(::agx::Notify::NOTIFY_INFO, NOTIFY_POS_ARGS)
#define NOTIFY_WARNING() setDebugStrings(::agx::Notify::NOTIFY_WARNING, NOTIFY_POS_ARGS)
#define NOTIFY_ERROR() setDebugStrings(::agx::Notify::NOTIFY_ERROR, NOTIFY_POS_ARGS)
#define NOTIFY_LOG() logOnly()
#define NOTIFY_END()  end()
#define NOTIFY_CLEAR()  clear()
#define NOTIFY_STATE(X)  push().setValue(X)


#define NOTIFY_WARNING_IF_nullptr(X,Y)  if ( Y == nullptr ) { X << X.NOTIFY_WARNING() << " nullptr pointer: " << #Y << std::endl <<  X.NOTIFY_END(); }
#define NOTIFY_WARNING_IF_FALSE(X,Y) if ( !Y ) { X << X.NOTIFY_WARNING() << " Test false: " << #Y <<std::endl <<  X.NOTIFY_END(); }
#define NOTIFY_WARNING_IF_TRUE(X,Y)  if ( Y ) { X <<  X.NOTIFY_WARNING() << " Test true: " << #Y <<std::endl <<  X.NOTIFY_END(); }

#define NOTIFY_ERROR_IF_nullptr(X,Y)  if ( Y == nullptr ) { X <<  X.NOTIFY_STATE(::agx::Notify::PRINT_ALL) << X.NOTIFY_ERROR() << " nullptr pointer: " << #Y <<std::endl <<  X.NOTIFY_END(); }
#define NOTIFY_ERROR_IF_FALSE(X,Y) if ( !Y ) { X <<  X.NOTIFY_STATE(::agx::Notify::PRINT_ALL) << X.NOTIFY_ERROR() << " Test false: " << #Y <<std::endl <<  X.NOTIFY_END(); }
#define NOTIFY_ERROR_IF_TRUE(X,Y)  if ( Y ) {  X <<  X.NOTIFY_STATE(::agx::Notify::PRINT_ALL) << X.NOTIFY_ERROR() << " Test true: " << #Y <<std::endl <<  X.NOTIFY_END(); }


#define NOTIFY_INSTANCE() ::agx::Notify::instance()



#ifdef _MSC_VER
# pragma warning(pop)
#endif

