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

#ifndef AGX_CALLBACK_H
#define AGX_CALLBACK_H

#include <agx/agx.h>
#include <functional>


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agx
{
  /**
  Generalized callback, using std::function.

  Example:

  class Foo
  {
  public:
    void Foo()
    {
      Callback callback(&Foo::bar, this);
      callback(); // Will print 'bar called'
    }

    void bar()
    {
      printf("bar called\n");
    }
  };

  */
  class AGXCORE_EXPORT Callback
  {
  public:
    typedef std::function<void ()> CallbackFunction;

    Callback();

    /**
    \param func - Callback function
    \param repeatEnable - Is repeat enabled or not.
    */
    Callback(const CallbackFunction& func, bool repeatEnable = true);

    /**
    \param obj -
    \param repeatEnable - Is repeat enabled or not.
    */
    template<class ClassT>
    Callback(void (ClassT::*fun)(), ClassT* obj, bool repeatEnable = true);


  public:
    /// \return True if the callback is non-void
    bool isValid() const;

    operator bool() const;

    /// Execute the callback
    void operator() () const;
    void run() const;

    /// \return True if the callback is repeating (when added to an agx::Event)
    bool isRepeating() const;

    /// Return the std representation
    const CallbackFunction& implementation() const;
    operator const CallbackFunction& () const;

  private:
    CallbackFunction m_callback;
    bool m_repeating;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////

  /// Templated callback with one argument
  template< typename T >
  class Callback1
  {
  public:
    typedef std::function<void (T)> CallbackFunction;

    Callback1();
    Callback1(const CallbackFunction& func, bool repeatEnable = true);

    template<class ClassT>
    Callback1(void (ClassT::*fun)(T), ClassT* obj, bool repeatEnable = true);

    template<class ClassT>
    Callback1(void (ClassT::*fun)(), ClassT* obj, bool repeatEnable = true);

  public:

    /// \return True if the callback is non-void
    bool isValid() const;

    /// Execute the callback
    void operator() ( T arg1 ) const;
    void run( T arg1 ) const;

    /// \return True if the callback is repeating (when added to an agx::Event)
    bool isRepeating() const;

    /// Return the std representation
    const CallbackFunction& implementation() const;
    operator const CallbackFunction& () const;

  private:
    CallbackFunction m_callback;
    bool m_repeating;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////

  /// Templated callback with two arguments
  template< typename T1, typename T2 >
  class Callback2
  {
  public:
    typedef std::function<void (T1, T2)> CallbackFunction;

    Callback2();
    Callback2(const CallbackFunction& func, bool repeatEnable = true);

    template<class ClassT>
    Callback2(void (ClassT::*fun)(T1, T2), ClassT* obj, bool repeatEnable = true);

    template<class ClassT>
    Callback2(void (ClassT::*fun)(T1), ClassT* obj, bool repeatEnable = true);

    template<class ClassT>
    Callback2(void (ClassT::*fun)(), ClassT* obj, bool repeatEnable = true);

  public:

    /// \return True if the callback is non-void
    bool isValid() const;

    /// Execute the callback
    void operator() ( T1 arg1, T2 arg2 ) const;
    void run( T1 arg1, T2 arg2 ) const;

    /// \return True if the callback is repeating (when added to an agx::Event)
    bool isRepeating() const;

    /// Return the std representation
    const CallbackFunction& implementation() const;
    operator const CallbackFunction& () const;

  private:
    CallbackFunction m_callback;
    bool m_repeating;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////

  /// Templated callback with three arguments
  template< typename T1, typename T2, typename T3 >
  class Callback3
  {
  public:
    typedef std::function<void (T1, T2, T3)> CallbackFunction;

    Callback3();
    Callback3(const CallbackFunction& func, bool repeatEnable = true);

    template<class ClassT>
    Callback3(void (ClassT::*fun)(T1, T2, T3), ClassT* obj, bool repeatEnable = true);

    template<class ClassT>
    Callback3(void (ClassT::*fun)(T1, T2), ClassT* obj, bool repeatEnable = true);

    template<class ClassT>
    Callback3(void (ClassT::*fun)(T1), ClassT* obj, bool repeatEnable = true);

    template<class ClassT>
    Callback3(void (ClassT::*fun)(), ClassT* obj, bool repeatEnable = true);

  public:

    /// \return True if the callback is non-void
    bool isValid() const;

    /// Execute the callback
    void operator() ( T1 arg1, T2 arg2, T3 arg3 ) const;
    void run( T1 arg1, T2 arg2, T3 arg3 ) const;

    /// \return True if the callback is repeating (when added to an agx::Event)
    bool isRepeating() const;

    /// Return the std representation
    const CallbackFunction& implementation() const;
    operator const CallbackFunction& () const;

  private:
    CallbackFunction m_callback;
    bool m_repeating;
  };


  template< typename T1, typename T2, typename T3, typename T4 >
  class Callback4
  {
  public:
    typedef std::function<void (T1, T2, T3, T4)> CallbackFunction;

    Callback4();
    Callback4(const CallbackFunction& func, bool repeatEnable = true);

    template<class ClassT>
    Callback4(void (ClassT::*fun)(T1, T2, T3, T4), ClassT* obj, bool repeatEnable = true);

    template<class ClassT>
    Callback4(void (ClassT::*fun)(T1, T2, T3), ClassT* obj, bool repeatEnable = true);

    template<class ClassT>
    Callback4(void (ClassT::*fun)(T1, T2), ClassT* obj, bool repeatEnable = true);

    template<class ClassT>
    Callback4(void (ClassT::*fun)(T1), ClassT* obj, bool repeatEnable = true);

    template<class ClassT>
    Callback4(void (ClassT::*fun)(), ClassT* obj, bool repeatEnable = true);

  public:
    operator const CallbackFunction& () const;
    bool isValid() const;

    void operator() ( T1 arg1, T2 arg2, T3 arg3, T4 arg4 ) const;
    void run( T1 arg1, T2 arg2, T3 arg3, T4 arg4 ) const;

    const CallbackFunction& implementation() const;
    bool isRepeating() const;

  private:
    bool m_repeating;
    CallbackFunction m_callback;
  };

  /* Implementation */
  template<class ClassT>
  AGX_FORCE_INLINE Callback::Callback(void (ClassT::*fun)(), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj)), m_repeating(repeatEnable)
  {
  }

  AGX_FORCE_INLINE Callback::Callback() : m_repeating(true) {}
  AGX_FORCE_INLINE Callback::Callback( const CallbackFunction& func, bool repeatEnable ) : m_callback(func), m_repeating(repeatEnable) {}

  AGX_FORCE_INLINE const Callback::CallbackFunction& Callback::implementation() const { return m_callback; }
  AGX_FORCE_INLINE bool Callback::isRepeating() const { return m_repeating; }


  AGX_FORCE_INLINE void Callback::operator() () const { this->run(); }
  AGX_FORCE_INLINE void Callback::run() const
  {
    agxAssert(m_callback);
    m_callback();
  }

  AGX_FORCE_INLINE Callback::operator const Callback::CallbackFunction& () const { return m_callback; }
  AGX_FORCE_INLINE bool Callback::isValid () const { return m_callback != nullptr; }
  AGX_FORCE_INLINE Callback::operator bool() const { return this->isValid(); }


  //// CALLBACK 1 ////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename T> template<class ClassT>
  inline Callback1<T>::Callback1(void (ClassT::*fun)(T), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj,std::placeholders:: _1)), m_repeating(repeatEnable)  {}

  template <typename T> template<class ClassT>
  AGX_FORCE_INLINE Callback1<T>::Callback1(void (ClassT::*fun)(), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj)), m_repeating(repeatEnable) {}

  template <typename T>
  AGX_FORCE_INLINE Callback1<T>::Callback1() : m_repeating(true) {}

  template <typename T>
  AGX_FORCE_INLINE Callback1<T>::Callback1( const CallbackFunction& func, bool repeatEnable ) : m_callback(func), m_repeating(repeatEnable)  {}

  template <typename T>
  AGX_FORCE_INLINE void Callback1<T>::operator() (T arg1) const { this->run(arg1); }

  template< typename T >
  AGX_FORCE_INLINE void Callback1<T>::run(T arg1) const
  {
    agxAssert(m_callback);
    m_callback(arg1);
  }

  template <typename T>
  AGX_FORCE_INLINE const typename Callback1<T>::CallbackFunction& Callback1<T>::implementation() const { return m_callback; }

  template <typename T>
  AGX_FORCE_INLINE bool Callback1<T>::isRepeating() const { return m_repeating; }

  template <typename T>
  AGX_FORCE_INLINE Callback1<T>::operator const typename Callback1<T>::CallbackFunction& () const { return m_callback; }

  template <typename T>
  AGX_FORCE_INLINE bool Callback1<T>::isValid() const { return m_callback != nullptr; }

  //// CALLBACK 2 ////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename T1, typename T2> template<class ClassT>
#if defined( __GNUC__ ) && __GNUC__ == 4 && __GNUC_MINOR__ >= 4
  inline Callback2<T1, T2>::Callback2(void (ClassT::*fun)(T1, T2), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj,std::placeholders:: _1,std::placeholders:: _2)), m_repeating(repeatEnable)  {}
#else
  AGX_FORCE_INLINE Callback2<T1, T2>::Callback2(void (ClassT::*fun)(T1, T2), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj,std::placeholders:: _1,std::placeholders:: _2)), m_repeating(repeatEnable) {}
#endif

  template <typename T1, typename T2> template<class ClassT>
  AGX_FORCE_INLINE Callback2<T1, T2>::Callback2(void (ClassT::*fun)(T1), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj,std::placeholders:: _1)), m_repeating(repeatEnable) {}

  template <typename T1, typename T2> template<class ClassT>
  AGX_FORCE_INLINE Callback2<T1, T2>::Callback2(void (ClassT::*fun)(), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj)), m_repeating(repeatEnable) {}

  template <typename T1, typename T2>
  AGX_FORCE_INLINE Callback2<T1, T2>::Callback2() : m_repeating(true) {}
  template <typename T1, typename T2>
  AGX_FORCE_INLINE Callback2<T1, T2>::Callback2( const CallbackFunction& func, bool repeatEnable ) : m_callback(func), m_repeating(repeatEnable) {}

  template <typename T1, typename T2>
  AGX_FORCE_INLINE void Callback2<T1,T2>::operator() (T1 arg1, T2 arg2) const { this->run(arg1, arg2); }

  template< typename T1, typename T2 >
  AGX_FORCE_INLINE void Callback2<T1,T2>::run(T1 arg1, T2 arg2 ) const
  {
    agxAssert(m_callback);
    m_callback(arg1,arg2);
  }

  template <typename T1, typename T2>
  AGX_FORCE_INLINE const typename Callback2<T1,T2>::CallbackFunction& Callback2<T1,T2>::implementation() const { return m_callback; }

  template <typename T1, typename T2>
  AGX_FORCE_INLINE bool Callback2<T1, T2>::isRepeating() const { return m_repeating; }

  template <typename T1, typename T2>
  AGX_FORCE_INLINE Callback2<T1, T2>::operator const typename Callback2<T1, T2>::CallbackFunction& () const { return m_callback; }

  template <typename T1, typename T2>
  AGX_FORCE_INLINE bool Callback2<T1, T2>::isValid() const { return m_callback != nullptr; }


  //// CALLBACK 3 ////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3> template<class ClassT>
#if defined( __GNUC__ ) && __GNUC__ == 4 && __GNUC_MINOR__ >= 4
inline Callback3<T1, T2, T3>::Callback3(void (ClassT::*fun)(T1, T2, T3), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj,std::placeholders:: _1,std::placeholders:: _2,std::placeholders:: _3)), m_repeating(repeatEnable) {}
#else
  AGX_FORCE_INLINE Callback3<T1, T2, T3>::Callback3(void (ClassT::*fun)(T1, T2, T3), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj,std::placeholders:: _1,std::placeholders:: _2,std::placeholders:: _3)), m_repeating(repeatEnable) {}
#endif

  template <typename T1, typename T2, typename T3> template<class ClassT>
  AGX_FORCE_INLINE Callback3<T1, T2, T3>::Callback3(void (ClassT::*fun)(T1, T2), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj,std::placeholders:: _1,std::placeholders:: _2)), m_repeating(repeatEnable) {}

  template <typename T1, typename T2, typename T3> template<class ClassT>
  AGX_FORCE_INLINE Callback3<T1, T2, T3>::Callback3(void (ClassT::*fun)(T1), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj,std::placeholders:: _1)), m_repeating(repeatEnable) {}

  template <typename T1, typename T2, typename T3> template<class ClassT>
  AGX_FORCE_INLINE Callback3<T1, T2, T3>::Callback3(void (ClassT::*fun)(), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj)), m_repeating(repeatEnable) {}

  template <typename T1, typename T2, typename T3>
  AGX_FORCE_INLINE Callback3<T1, T2, T3>::Callback3() : m_repeating(true) {}

  template <typename T1, typename T2, typename T3>
  AGX_FORCE_INLINE Callback3<T1, T2, T3>::Callback3( const CallbackFunction& func, bool repeatEnable ) : m_callback(func), m_repeating(repeatEnable) {}

  template <typename T1, typename T2, typename T3>
  AGX_FORCE_INLINE void Callback3<T1,T2,T3>::operator() (T1 arg1, T2 arg2, T3 arg3) const { this->run(arg1, arg2, arg3); }

  template< typename T1, typename T2, typename T3 >
  AGX_FORCE_INLINE void Callback3<T1,T2,T3>::run( T1 arg1, T2 arg2, T3 arg3 ) const
  {
    agxAssert(m_callback);
    m_callback(arg1,arg2,arg3);
  }

  template <typename T1, typename T2, typename T3>
  AGX_FORCE_INLINE const typename Callback3<T1,T2,T3>::CallbackFunction& Callback3<T1,T2,T3>::implementation() const { return m_callback; }

  template <typename T1, typename T2, typename T3>
  AGX_FORCE_INLINE bool Callback3<T1, T2, T3>::isRepeating() const { return m_repeating; }

  template <typename T1, typename T2, typename T3>
  AGX_FORCE_INLINE Callback3<T1, T2, T3>::operator const typename Callback3<T1, T2, T3>::CallbackFunction& () const { return m_callback; }

  template <typename T1, typename T2, typename T3>
  AGX_FORCE_INLINE bool Callback3<T1, T2, T3>::isValid() const { return m_callback != nullptr; }

  //// CALLBACK 4 ////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename T1, typename T2, typename T3, typename T4> template<class ClassT>
  AGX_FORCE_INLINE Callback4<T1, T2, T3, T4>::Callback4(void (ClassT::*fun)(T1, T2, T3, T4), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj,std::placeholders:: _1,std::placeholders:: _2,std::placeholders:: _3,std::placeholders:: _4)), m_repeating(repeatEnable) {}

  template <typename T1, typename T2, typename T3, typename T4> template<class ClassT>
  AGX_FORCE_INLINE Callback4<T1, T2, T3, T4>::Callback4(void (ClassT::*fun)(T1, T2, T3), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj,std::placeholders:: _1,std::placeholders:: _2,std::placeholders:: _3)), m_repeating(repeatEnable) {}

  template <typename T1, typename T2, typename T3, typename T4> template<class ClassT>
  AGX_FORCE_INLINE Callback4<T1, T2, T3, T4>::Callback4(void (ClassT::*fun)(T1, T2), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj,std::placeholders:: _1,std::placeholders:: _2)), m_repeating(repeatEnable) {}

  template <typename T1, typename T2, typename T3, typename T4> template<class ClassT>
  AGX_FORCE_INLINE Callback4<T1, T2, T3, T4>::Callback4(void (ClassT::*fun)(T1), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj,std::placeholders:: _1)), m_repeating(repeatEnable) {}

  template <typename T1, typename T2, typename T3, typename T4> template<class ClassT>
  AGX_FORCE_INLINE Callback4<T1, T2, T3, T4>::Callback4(void (ClassT::*fun)(), ClassT* obj, bool repeatEnable) : m_callback(std::bind(fun, obj)), m_repeating(repeatEnable) {}

  template <typename T1, typename T2, typename T3, typename T4>
  AGX_FORCE_INLINE Callback4<T1, T2, T3, T4>::Callback4() : m_repeating(true) {}

  template <typename T1, typename T2, typename T3, typename T4>
  AGX_FORCE_INLINE Callback4<T1, T2, T3, T4>::Callback4( const CallbackFunction& func, bool repeatEnable ) : m_callback(func), m_repeating(repeatEnable) {}

  template <typename T1, typename T2, typename T3, typename T4>
  AGX_FORCE_INLINE void Callback4<T1, T2, T3, T4>::operator() (T1 arg1, T2 arg2, T3 arg3, T4 arg4) const { this->run(arg1, arg2, arg3, arg4); }

  template <typename T1, typename T2, typename T3, typename T4>
  AGX_FORCE_INLINE void Callback4<T1, T2, T3, T4>::run( T1 arg1, T2 arg2, T3 arg3, T4 arg4 ) const
  {
    agxAssert(m_callback);
    m_callback(arg1, arg2, arg3, arg4);
  }

  template <typename T1, typename T2, typename T3, typename T4>
  AGX_FORCE_INLINE const typename Callback4<T1, T2, T3, T4>::CallbackFunction& Callback4<T1, T2, T3, T4>::implementation() const { return m_callback; }

  template <typename T1, typename T2, typename T3, typename T4>
  AGX_FORCE_INLINE bool Callback4<T1, T2, T3, T4>::isRepeating() const { return m_repeating; }

  template <typename T1, typename T2, typename T3, typename T4>
  AGX_FORCE_INLINE Callback4<T1, T2, T3, T4>::operator const typename Callback4<T1, T2, T3, T4>::CallbackFunction& () const { return m_callback; }

  template <typename T1, typename T2, typename T3, typename T4>
  AGX_FORCE_INLINE bool Callback4<T1, T2, T3, T4>::isValid() const { return m_callback != nullptr; }
}


#ifdef _MSC_VER
#  pragma warning(pop)
#endif


#endif
