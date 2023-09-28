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

#ifndef AGX_ATOMICVALUE_H
#define AGX_ATOMICVALUE_H

#include <agx/config/AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING.h>
#include <agx/Integer.h>
#include <ostream>
#include <atomic>

namespace agx
{

  /**
  AtomicValueT template for integral types, all operations are atomic.
  */
  template< class T >
  class AtomicValueT
  {
  public:
    explicit AtomicValueT(T initialValue = 0);

    AtomicValueT( const agx::AtomicValueT<T>& av );

    AtomicValueT<T>& operator=(const agx::AtomicValueT<T>& rhs );

    /**
    Get the current value.
    */
    inline T get() const;

    /**
    Set the value.
    \return The previous value
    */
    inline T set(T value);

    /**
    Set the value ignoring the previous value,
    thus avoiding synchronization.
    */
    inline AtomicValueT<T>& operator= (T value);

    /**
    Add a value.
    \return The previous value
    */
    inline T add(T value);

    /**
    Subtract a value.
    \return The previous value
    */
    inline T sub(T value);

    /**
    Increment the value.
    \return The previous value
    */
    inline T inc();

    /**
    Decrement the value.
    \return The previous value
    */
    inline T dec();

    /**
    Compares the current value with a test value and if they are equal the value is
    replaced by a new value.
    \param test The value to compare with
    \param newValue The new value
    \return The previous value
    */
    inline T compareAndSet(T test, T newValue);

  private:
    #if !AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING()
    std::atomic< T > m_value;
    #else
    T m_value;
    #endif
  };


  /**
  agx::AtomicValue is an alias for agx::AtomicValueT<agx::Int32>
  */
  using AtomicValue = AtomicValueT<agx::Int32>;



  // -----------------------------------------------------------------------------------------------
  template<class T>
  inline std::ostream& operator << ( std::ostream& stream, const AtomicValueT<T>& value )
  {
    stream << value.get();
    return stream;
  }

  /* Implementation */

  #if !(defined(AGX_APPLE_IOS) || (AGX_DEBUG_SYNCHRONIZATION_OVERHEAD_TESTING()))

  template<class T>
  inline AtomicValueT<T>::AtomicValueT(T initialValue) : m_value(initialValue)
  {
  }

  template<class T>
  inline AtomicValueT<T>::AtomicValueT( const agx::AtomicValueT<T>& av )
  {
    m_value.store( av.m_value.load() );
  }

  template<class T>
  inline AtomicValueT<T>& AtomicValueT<T>::operator=(const agx::AtomicValueT<T>& rhs )
  {
    m_value.store( rhs.m_value.load() );
    return *this;
  }

  template<class T>
  inline T AtomicValueT<T>::get() const
  {
    return m_value;
  }

  template<class T>
  inline T AtomicValueT<T>::set(T value)
  {
    return m_value.exchange( value );
  }

  template<class T>
  inline AtomicValueT<T>& AtomicValueT<T>::operator=(T value)
  {
    m_value = value;
    return *this;
  }


  template<class T>
  inline T AtomicValueT<T>::add(T value)
  {
    return m_value.fetch_add( value );
  }

  template<class T>
  inline T AtomicValueT<T>::sub(T value)
  {
    return m_value.fetch_sub( value );
  }

  template<class T>
  inline T AtomicValueT<T>::inc()
  {
    return m_value.fetch_add( 1 );
  }

  template<class T>
  inline T AtomicValueT<T>::dec()
  {
    return m_value.fetch_sub( 1 );
  }

  template<class T>
  inline T AtomicValueT<T>::compareAndSet(T test, T newValue)
  {
    return m_value.compare_exchange_strong( test, newValue );
  }

  #else
  template<class T>
  inline AtomicValueT<T>::AtomicValueT(T initialValue) : m_value(initialValue)
  {
  }

  template<class T>
  inline AtomicValueT<T>::AtomicValueT( const agx::AtomicValueT<T>& av )
  {
    m_value = av.m_value;
  }

  template<class T>
  AtomicValueT<T>& AtomicValueT<T>::operator=(const agx::AtomicValueT<T>& rhs )
  {
    m_value = rhs.m_value;
    return *this;
  }

  template<class T>
  inline T AtomicValueT<T>::get() const
  {
    return m_value;
  }

  template<class T>
  inline T AtomicValueT<T>::set(T value)
  {
    T prev = m_value;
    m_value = value;

    return prev;
  }

  template<class T>
  inline AtomicValueT<T>& AtomicValueT<T>::operator=(T value)
  {
    m_value = value;
    return *this;
  }


  template<class T>
  inline T AtomicValueT<T>::add(T value)
  {
    T prev = m_value;
    m_value += value;

    return prev;
  }

  template<class T>
  inline T AtomicValueT<T>::sub(T value)
  {
    T prev = m_value;
    m_value -= value;


    return prev;
  }

  template<class T>
  inline T AtomicValueT<T>::inc()
  {
    return this->add(1);
  }

  template<class T>
  inline T AtomicValueT<T>::dec()
  {
    return this->add(-1);
  }

  template<class T>
  inline T AtomicValueT<T>::compareAndSet(T test, T newValue)
  {
    T prev = m_value;
    if (m_value == test)
      m_value = newValue;

    return prev;
  }

  #endif
}


#endif /* _AGX_ATOMICVALUE_H_ */
