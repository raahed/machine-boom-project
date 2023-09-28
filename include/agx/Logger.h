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

#define g_LogMgr agx::Logger::instance()

#define LOGGER() agx::Logger::instance()->getNotify()
#define LOGGER_ERROR() LOGGER() << LOGGER().NOTIFY_ERROR()
#define LOGGER_WARNING() LOGGER() << LOGGER().NOTIFY_WARNING()
#define LOGGER_INFO() LOGGER() << LOGGER().NOTIFY_INFO()
#define LOGGER_DEBUG() LOGGER() << LOGGER().NOTIFY_DEBUG()
#define LOGGER_END() LOGGER().NOTIFY_END()
#define LOGGER_ENDL() std::endl << LOGGER().NOTIFY_END()

#define LOGGER_ERROR_IF_nullptr(Y) NOTIFY_ERROR_IF_nullptr(LOGGER(), Y)
#define LOGGER_ERROR_IF_TRUE(Y) NOTIFY_ERROR_IF_TRUE(LOGGER(), Y)
#define LOGGER_ERROR_IF_FALSE(Y) NOTIFY_ERROR_IF_FALSE(LOGGER(), Y)

#define LOGGER_WARNING_IF_nullptr(Y) NOTIFY_ERROR_IF_nullptr(LOGGER(), Y)
#define LOGGER_ERROR_IF_TRUE(Y) NOTIFY_ERROR_IF_TRUE(LOGGER(), Y)
#define LOGGER_ERROR_IF_FALSE(Y) NOTIFY_ERROR_IF_FALSE(LOGGER(), Y)
#define LOGGER_STATE(X) LOGGER().NOTIFY_STATE(X)

#include <agx/agx.h>
#include <agx/Notify.h>
#include <agx/Singleton.h>
namespace agx
{

  /**
  Logger is a class for writing information, debug info, warnings but also throwing exceptions in a stream-alike way.
  Always use the macros LOGGER_*()
  This class is thread safe. It should ONLY be called from AGX threads. If you create a thread of your own, and call LOGGER
  it will certainly crash. AGX has a "mainThread", which is the thread from which agx::init() was called. This is
  also part of the "safe zone".
  */
  class AGXCORE_EXPORT Logger : public agx::Singleton
  {
  public:

    Logger();
    /// Return the singleton object
    static Logger *instance( void );

    ADD_NOTIFY_FUNCTIONALITY_TO_CLASS();
    SINGLETON_CLASSNAME_METHOD();

    /**
    Open a new log file.
    \param path - Path to a new log file
    \param overwrite - if false, a version number will be added to the filename: test.log.1, test.log.2 etc.
    \param closeAndOpen - if true, an opened log file will be closed and the new path will be opened.
    If false, nothing will happen, false will be returned
    \return true if log file is successfully opened.
    */
    bool openLogfile( const agx::String& path, bool overwrite, bool closeAndOpen );

    /**
    \return the path of the file where log messages will be written
    */
    agx::String getLogFilename() const;

    /**
    Set the lowest level of messages that will be written to the logger system
    */
    void setLogNotifyLevel( Notify::NotifyLevel l );

  protected:
    void shutdown() override;

  private:
    virtual ~Logger();
    static Logger *s_instance;

  };

  typedef ref_ptr<Logger> LoggerRef;

} // namespace agx


// Only used by lua bindings
namespace agx
{
  void AGXCORE_EXPORT logDebug(std::string str);
  void AGXCORE_EXPORT logInfo(std::string str);
  void AGXCORE_EXPORT logWarning(std::string str);
  void AGXCORE_EXPORT logError(std::string str);
}


