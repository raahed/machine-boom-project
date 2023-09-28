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

#include <agx/Notify.h>
#include <deque>

namespace agx {

  /**
  Class for polling messages from the LOGGER system in a Subscriber pattern.
  Messages is stored until the message pop() is called.
  The class is reentrant thread safe.

  There are a message queue for each NotifyLevel.
  So all methods can use a bitmask to filter out which queue that is requested.

  */
  class AGXPHYSICS_EXPORT LoggerSubscriber : public agx::Referenced
  {
  public:

    // agx::Notify::NOTIFY_DEBUG | agx::Notify::NOTIFY_INFO | agx::Notify::NOTIFY_WARNING | agx::Notify::NOTIFY_ERROR;
    static int ALL_LEVELS;


    /**
    Constructor for a logger mailbox
    \param levelMask - Specifies which levels of messages that will be captured.
                       Default is all messages with equal or higher level that NOTIFY_INFO.
    */
    LoggerSubscriber(int levelMask = agx::Notify::NOTIFY_INFO | agx::Notify::NOTIFY_WARNING | agx::Notify::NOTIFY_ERROR);

    /**
    Get all available messages and then remove them.
    \param levelMask - Specifies which levels of messages that should be checked for.
    \return all messages concatenated into a string. Messages will be separated with a newline
    */
    std::string pop(int levelMask = ALL_LEVELS);

    /**
    Get all available messages and then remove them.
    \param[out] outMessages - Where messages will be stored
    \param levelMask - Specifies which levels of messages that should be captured.
    \return the number of available messages
    */
    size_t pop(agx::StringVector& outMessages, int levelMask = ALL_LEVELS);

    /**
    \param levelMask - Specifies which levels of messages that should be checked for, by default all levels
    \return true if no messages are present
    */
    bool empty(int levelMask = ALL_LEVELS) const;

    /**
    Remove all messages
    \param levelMask - Specifies which levels of messages that should be cleared, by default all levels
    */
    void clear(int levelMask = ALL_LEVELS);

    /**
    Enable/disable the capturing of messages
    \param flag - If true messages will be captured
    */
    void setEnable(bool flag);

    /**
    \return true if messages are being captured
    */
    bool getEnable() const;


    typedef std::pair<int, agx::String> Message;
    typedef agx::Vector<Message> MessageVector;
    /**
    Get a copy of all messages matching the specified notify level \p levelMask in the message queue.
    Each message contains a pair of: NOTIFY level (INFO, WARNING, ERROR) and the message string in chronological order

    \param messages - Reference to a message queue that will be populated with all the matching messages
    \param clearMessages - If true, the message queue will be cleared from the matching messages
    \return true if any messages were awailable.
    */
    bool getMessages(MessageVector& messages, bool clearMessages, int levelMask = ALL_LEVELS);

  protected:

    /**
    Callback that will be called when a new message is present
    */
    void message(const agx::String& msg, int notifyLevel);

    /// Destructor
    virtual ~LoggerSubscriber();

    class LogCallback : public agx::NotifyCallback
    {
    public:
      LogCallback(LoggerSubscriber* subscriber);
      ~LogCallback();

    protected:
      LoggerSubscriber *m_subscriber;      
      
      void message(const agx::String& msg, int notifyLevel) override;
    };

    friend class LogCallback;

  private:
    typedef std::deque<Message> MessageQueue;
    MessageQueue m_messageQueue;

    agx::ref_ptr<LogCallback> m_callback;

    int m_levelMask;
    bool m_enabled;
    mutable agx::ReentrantMutex m_mutex;
  };
}
