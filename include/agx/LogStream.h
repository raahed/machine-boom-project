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

#ifndef AGX_LOGSTREAM_H
#define AGX_LOGSTREAM_H

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4355) // No warnings about using this in member initializer list
# pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
#endif

#include <agx/agxCore_export.h>

#include <time.h>

#include <iostream>
#include <string>
#include <sstream>
#include <streambuf>
#include <agx/String.h>



namespace agx
{

  /**
  Class for printing out messages to console, file, virtual method and or a callback function.
  */
  class AGXCORE_EXPORT LogStream : public std::ostream
  {
    public:

      /// Constructor
      LogStream();

    public:

      typedef void ( *CallbackFunction_t )( const char*, unsigned int );

      /**
      \param dateFormat - Format in date (if logTimeWithMilliseconds==false) as being used by the method strftime()
      \param logTimeWithMilliseconds - If true (default) the date string will be reported in milliseconds
      \return the current time in the specified time format
      */
      static agx::String getTimeString( const agx::String& dateFormat="", bool logTimeWithMilliseconds=true );

      /**
       Specify the format of the date/time string that getTimeString() will return
       Look at strftime for examples of how to specify the date.
       Default is "%X" (hh:mm:ss)
      */
      void setDateFormat( const agx::String& date_format );

      /// If \p flag == true, the date in the specified date format will be added prior to each message.
      void setEnableDate( bool flag );

      bool getEnableDate() const;

      /**
      If set to true, the messages will be prefixed with the id of the thread that generates the message.
      */
      void setEnableThreadId(bool flag);

      bool getEnableThreadId() const;

      /// \return the date format used for specifying the output format of the getTimeString()
      const agx::String& getDateFormat() const;

      /// Enums used to specify the output mode used. These can be OR:ed together
      enum OutputMode { CERR = 1, COUT = 2, FILE = 4, FUNCTION = 8 };

      /// OR:s together the active outputmode with modemask
      void appendOutputMode( short modemask );

      /// Set the active outputmode to modemask
      void setOutputMode( short modemask );

      /// Resets the active output modes to none
      void clearOutputMode( void );

      /// Returns the active outputmode
      int getOutputMode( void );

      /// Returns true if the modemask is set as the active output mode.
      bool isActive( OutputMode modemask );

      /// Open a specified file as a log
      bool open( const agx::String& name );

      void close( void );

      /// \return true if a file is open
      bool haveOpenFile() const;

      agx::String getFilename() const;

      void flush();

      /**
      * Flush and empty all buffers, and then remove the allocated
      * buffers.
      */
      virtual ~LogStream();

      /// Sets the function that will be called if FUNCTION is set as outputmode
      void setCallback( CallbackFunction_t f );

      //std::ofstream& getStream();

      CallbackFunction_t getCallback();

      /**
      This method will be called whenever there is a message ready for output.
      Derive from this class and implement this method.
      */
      virtual void message( const char* msg, unsigned int length );

  protected:

      size_t write( const char *msg, std::streamsize count);

      CallbackFunction_t m_callback;

      int m_mode;
      ::FILE *m_file;
      bool m_isOpen;
      agx::String m_filename;

      // Copy & assignment are undefined in iostreams
#ifndef SWIG
      LogStream( const LogStream& );
      LogStream& operator=( const LogStream& );
#endif

    private:

      agx::String m_date_format;
      bool m_date_enabled;
      bool m_threadId_enabled;


      DOXYGEN_START_INTERNAL_BLOCK()
public:
      template < class logStreamT, class charT, class traits = std::char_traits<charT> >
      class AGXCORE_EXPORT LogBuf : public std::basic_streambuf<charT, traits>
      {
        public:

          LogBuf( logStreamT* stream ) : m_stream( stream ) {}
          virtual ~LogBuf(){};

        protected:

          typedef LogBuf<logStreamT, charT> classT;
          typedef std::basic_streambuf<charT, traits> BaseClassT;

          std::streamsize output( const char* pStr, std::streamsize n );

          int overflow ( int ch );

          int sync ();

          LogBuf( const LogBuf& );
          LogBuf& operator=( const LogBuf& );
          logStreamT* m_stream;
          std::ostringstream m_strStream;

      };

      typedef LogBuf<LogStream, char> LogBufT;
      LogBufT* m_logBuf;

      template < class logStreamT, class charT, class traits > friend class LogBuf;
      DOXYGEN_END_INTERNAL_BLOCK()

  };
} // namespace agx

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
