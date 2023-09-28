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

#ifndef AGX_EVENT_H
#define AGX_EVENT_H

#include <iosfwd>
#include <agx/Name.h>
#include <agx/Callback.h>
#include <agx/Vector.h>

namespace agx
{
  class Thread;

  /**
  An event has a list of listeners (agx::Callback), which are
  called when the event is triggered.
  */
  template <typename T>
  class EventT
  {
  public:
    typedef T CallbackType;
    typedef VectorPOD< const CallbackType * > CallbackPtrVector;

  public:
    EventT( const agx::Name& name = agx::Name());

    /**
    \return The name of the event.
    */
    const agx::Name& getName() const;

    /**
    Add a callback to the event.
    */
    void addCallback( const T* callback );

    /**
    Add a callback to the front of the callback list.
    */
    void addCallbackFirst( const T* callback );

    /*
    Remove a callback.
    \return false if the callback was not previously added
    **/
    bool removeCallback( const T* callback );

    /**
    Remove all callbacks.
    */
    void removeAllCallbacks();

    /**
    \return True if the callback is registered with this event.
    */
    bool containsCallback( const T* callback) const;

    /**
    \return The list of registered callbacks.
    */
    const CallbackPtrVector& getCallbacks() const;

  protected:
    CallbackPtrVector m_callbacks;
    Name m_name;
    Thread *m_activeThread;
    agx::UInt32 m_iterationIndex;
  };

  //---------------------------------------------------------------

  /// An event with no arguments
  class AGXCORE_EXPORT Event : public EventT< Callback >
  {
  public:
    Event( const agx::Name& name = agx::Name());

    /// Trigger the event
    void trigger();
  };

  //---------------------------------------------------------------

  /// An event with one argument
  template <typename T1>
  class Event1 : public EventT< Callback1<T1> >
  {
  public:
    Event1( const agx::Name& name = agx::Name(), const T1& defaultArg = T1());

    /// Trigger the event with the default argument
    void trigger();

    /// Trigger the event
    void trigger( const T1& arg );

  private:
    T1 m_defaultArg;
  };

  //---------------------------------------------------------------

  /// An event with two arguments
  template <typename T1, typename T2>
  class Event2 : public EventT< Callback2<T1, T2> >
  {
  public:
    Event2( const agx::Name& name = agx::Name(), const T1& defaultArg = T1());

    /// Trigger the event with default first argument
    void trigger( const T2& arg2 );

    /// Trigger the event
    void trigger( const T1& arg1, const T2& arg2 );

  private:
    T1 m_defaultArg;
  };

  //---------------------------------------------------------------

  /// An event with three arguments
  template <typename T1, typename T2, typename T3>
  class Event3 : public EventT< Callback3<T1, T2, T3> >
  {
  public:
    Event3( const agx::Name& name = agx::Name(), const T1& defaultArg = T1());

    /// Trigger the event with default first argument
    void trigger( const T2& arg2, const T3& arg3 );

    /// Trigger the event
    void trigger( const T1& arg1, const T2& arg2, const T3& arg3 );

  private:
    T1 m_defaultArg;
  };


  //---------------------------------------------------------------


  // implementation
  #define INDEX this->m_iterationIndex
  AGXCORE_EXPORT Thread *__agx_Event_getActiveThread();

  template <typename T>
  AGX_FORCE_INLINE EventT<T>::EventT( const Name& name ) : m_name(name), m_activeThread(nullptr), m_iterationIndex(0)
  {
  }

  template <typename T>
  AGX_FORCE_INLINE const Name& EventT<T>::getName() const { return m_name; }

  template <typename T>
  AGX_FORCE_INLINE void EventT<T>::addCallback( const T* callback )
  {
    agxAssert(!m_activeThread || m_activeThread == __agx_Event_getActiveThread());
    m_callbacks.push_back( callback );

    #if 0
    /* Move callback past those with lower prio */
    for (size_t i = m_callbacks.size() - 1; i > 0 ; --i)
    {
      if (m_callbacks[i].priority <= m_callbacks[i-1].priority)
        break;

      std::swap(m_callbacks[i], m_callbacks[i-1]);
    }
    #endif
  }

  template <typename T>
  AGX_FORCE_INLINE void EventT<T>::addCallbackFirst( const T* callback )
  {
    agxAssert(!m_activeThread || m_activeThread == __agx_Event_getActiveThread());
    m_callbacks.insert( (size_t)0, callback );
    m_iterationIndex++;
  }


  template <typename T>
  AGX_FORCE_INLINE bool EventT<T>::removeCallback( const T* callback )
  {
    agxAssert(!m_activeThread || m_activeThread == __agx_Event_getActiveThread());
    size_t index = m_callbacks.find(callback);

    if (index == m_callbacks.size())
      return false;


    m_callbacks.erase(index);

    if (index <= m_iterationIndex)
      m_iterationIndex--;

    return true;
  }

  template <typename T>
  AGX_FORCE_INLINE void EventT<T>::removeAllCallbacks() { m_callbacks.clear(); }

  template <typename T>
  AGX_FORCE_INLINE const typename EventT<T>::CallbackPtrVector& EventT<T>::getCallbacks() const { return m_callbacks; }

  template <typename T>
  AGX_FORCE_INLINE bool EventT<T>::containsCallback(const T* callback) const { return m_callbacks.contains(callback); }

  AGX_FORCE_INLINE Event::Event(const Name& name) : EventT<Callback>(name) {}

  #ifdef AGX_DEBUG
  #define AGX_EVENT_TRIGGER_START(); this->m_activeThread = __agx_Event_getActiveThread();
  #define AGX_EVENT_TRIGGER_END(); this->m_activeThread = nullptr
  #else
  #define AGX_EVENT_TRIGGER_START();
  #define AGX_EVENT_TRIGGER_END();
  #endif

  AGX_FORCE_INLINE void Event::trigger()
  {
    AGX_EVENT_TRIGGER_START();
    for (INDEX = 0; INDEX < m_callbacks.size(); ++INDEX)
    {
      const Callback *callback = m_callbacks[INDEX];
      if (!callback->isRepeating())
      {
        m_callbacks[INDEX] = m_callbacks.back();
        m_callbacks.pop_back();
        INDEX--;
      }
      callback->run();
    }
    AGX_EVENT_TRIGGER_END();
  }

  template< typename T1 >
  AGX_FORCE_INLINE Event1<T1>::Event1(const Name& name, const T1& defaultArg) : EventT<Callback1<T1> >(name), m_defaultArg(defaultArg) {}

  template< typename T1 >
  AGX_FORCE_INLINE void Event1<T1>::trigger()
  {
    this->trigger(m_defaultArg);
  }

  template< typename T1 >
  AGX_FORCE_INLINE void Event1<T1>::trigger(const T1& arg)
  {
    AGX_EVENT_TRIGGER_START();

    for (INDEX = 0; INDEX < this->m_callbacks.size(); ++INDEX)
    {
      const Callback1<T1> *callback = this->m_callbacks[INDEX];
      if (!callback->isRepeating())
      {
        this->m_callbacks[INDEX] = this->m_callbacks.back();
        this->m_callbacks.pop_back();
        INDEX--;
      }
      callback->run(arg);
    }

    AGX_EVENT_TRIGGER_END();
  }


  template< typename T1, typename T2 >
  AGX_FORCE_INLINE Event2<T1, T2>::Event2(const Name& name, const T1& defaultArg) : EventT<Callback2<T1, T2> >(name), m_defaultArg(defaultArg) {}

  template< typename T1, typename T2 >
  AGX_FORCE_INLINE void Event2<T1,T2>::trigger(const T2& arg2)
  {
    this->trigger(m_defaultArg, arg2);
  }


  template< typename T1, typename T2 >
  AGX_FORCE_INLINE void Event2<T1,T2>::trigger(const T1& arg1, const T2& arg2)
  {
    AGX_EVENT_TRIGGER_START();

    for (INDEX = 0; INDEX < this->m_callbacks.size(); ++INDEX)
    {
      const Callback2<T1,T2> *callback = this->m_callbacks[INDEX];
      if (!callback->isRepeating())
      {
        this->m_callbacks[INDEX] = this->m_callbacks.back();
        this->m_callbacks.pop_back();
        INDEX--;
      }
      callback->run(arg1, arg2);
    }

    AGX_EVENT_TRIGGER_END();
  }


  template< typename T1, typename T2, typename T3 >
  AGX_FORCE_INLINE Event3<T1, T2, T3>::Event3(const Name& name, const T1& defaultArg) : EventT<Callback3<T1, T2, T3> >(name), m_defaultArg(defaultArg) {}

  template< typename T1, typename T2, typename T3 >
  AGX_FORCE_INLINE void Event3<T1,T2,T3>::trigger(const T2& arg2, const T3& arg3)
  {
    this->trigger(m_defaultArg, arg2, arg3);
  }

  template< typename T1, typename T2, typename T3 >
  AGX_FORCE_INLINE void Event3<T1,T2,T3>::trigger(const T1& arg1, const T2& arg2, const T3& arg3)
  {
    AGX_EVENT_TRIGGER_START();

    for (INDEX = 0; INDEX < this->m_callbacks.size(); ++INDEX)
    {
      const Callback3<T1,T2,T3> *callback = this->m_callbacks[INDEX];
      if (!callback->isRepeating())
      {
        this->m_callbacks[INDEX] = this->m_callbacks.back();
        this->m_callbacks.pop_back();
        INDEX--;
      }

      callback->run(arg1, arg2, arg3);
    }

    AGX_EVENT_TRIGGER_END();
  }


  template <typename T>
  AGX_FORCE_INLINE std::ostream& operator<<(std::ostream& stream, const agx::EventT<T>& e)
  {
    stream << e.getName();
    return stream;
  }

  #undef INDEX
}

#endif

