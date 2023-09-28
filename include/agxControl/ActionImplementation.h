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

/// \cond CONTROL



#ifndef AGXCONTROL_ACTION_IMPLEMENTATION_H
#define AGXCONTROL_ACTION_IMPLEMENTATION_H

#include <agx/config.h>


#include <agx/Referenced.h>
#include <agxStream/Serializable.h>
#include <agxStream/InputArchive.h>
#include <agxStream/OutputArchive.h>

#include <agxControl/ActionImplementationVisitor.h>
#include <functional>

namespace agxControl
{

  class CallbackActionImplementation;
  template<typename T> class Callback1ActionImplementation;
  template<typename R, typename T> class CallbackR1ActionImplementation;
  template<typename T1, typename T2> class Callback2ActionImplementation;

  AGX_DECLARE_POINTER_TYPES(ActionImplementation);

  /*
  The ActionImplementation is the base class for work that an Action can perform.
  It inherits from Serializable, but many of the subclasses cannot be serialized
  completely due to the use of function pointers. They do however store as much
  as they can.
  */
  class AGXPHYSICS_EXPORT ActionImplementation : public agx::Referenced, public agxStream::Serializable
  {
  public:
    /// \param time The time when the action should be triggered.
    ActionImplementation(agx::Real time);

    /**
    Called when it's time to perform the action. The time parameter can be
    different from the time passed to the constructor due to the discrete nature
    of the time stepper used in AGX.
    \param time The current time of the simulation.
    */
    virtual void trigger(agx::Real time);

    /// Equivalent to actionImpl->trigger(actionImpl->getTime());
    void trigger();

    /// \return The time when the action should be triggered.
    agx::Real getTime() const;

    /**
    Visitor pattern. Each subclass should implement this visit method and call operation->update(this, time).
    */
    virtual void visit(agxControl::ActionImplementationVisitor* operation, agx::Real time);

    /**
    Double dispatch visitor. Each subclass should implement this visit method
    and call end->dispatchVisit(operation, this, time).
    */
    virtual void visit(agxControl::ActionImplementationVisitor* operation, ActionImplementation* end, agx::Real time);

    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::ActionImplementation );

  protected:
    virtual ~ActionImplementation() {};

    /// Used only for store/restore.
    ActionImplementation() {}

  public: /// \todo Should not be public. Only called by subclasses, why isn't protected working?

    /**
    /cond internal.

    Second state of the double dispatch. The base class supplies implementations
    of these that cast an exception. Subclasses should implement the method that
    has the second argument, begin, being the type of the subclass and call
    operation->update(begin, this, time).

    Note that we can't have templated virtual methods, so every supported template
    combination for all supported subclasses of ActionImplementation must be listed
    here.
    */
    virtual void dispatchVisit(agxControl::ActionImplementationVisitor* operation, agxControl::CallbackActionImplementation* begin, agx::Real time);
    virtual void dispatchVisit(agxControl::ActionImplementationVisitor* operation, agxControl::Callback1ActionImplementation<agx::Real>* begin, agx::Real time);
    virtual void dispatchVisit(agxControl::ActionImplementationVisitor* operation, agxControl::CallbackR1ActionImplementation<void, agx::Real>* begin, agx::Real time);
    virtual void dispatchVisit(agxControl::ActionImplementationVisitor* operation, agxControl::Callback2ActionImplementation<agx::Real, agx::Int>* begin, agx::Real time);

  protected:
    static std::string m_className;

  private:
    agx::Real m_time;
  };


  /** Action that does nothing. */
  class AGXPHYSICS_EXPORT VoidActionImplementation : public ActionImplementation
  {
  public:
    VoidActionImplementation(agx::Real time);
  protected:
    virtual ~VoidActionImplementation() {}
  };

  /**
  An ActionImplementation that calls a member function when triggered, passing
  along the current time.
  */
  class AGXPHYSICS_EXPORT CallbackActionImplementation : public ActionImplementation
  {
  public:
    typedef std::function<void (agx::Real)> Callback;

    template<typename ClassT>
    CallbackActionImplementation(agx::Real time, void (ClassT::*function)(agx::Real), ClassT* object);

    virtual void trigger(agx::Real currentTime);
    using ActionImplementation::trigger;

    virtual void visit(agxControl::ActionImplementationVisitor* operation, agx::Real time);
    using ActionImplementation::visit;

  protected:
    virtual ~CallbackActionImplementation() {}

  private:
    Callback m_callback;
  };


  /**
  An ActionImplementation that calls a member function when triggered, passing
  along a configurable value. The function's signature must be "void (T)", that
  is, a function that takes a single argument and returns void.

  Serialization will store the currently configured value.
  */
  template<typename T>
  class Callback1ActionImplementation : public ActionImplementation
  {
  public:
    typedef std::function<void (T)> Callback;

    /**
    \param time The time at which the action should be triggered.
    \param function The member function to be called when the action is triggered.
    \param object The object on which to call the function.
    \param value The value passed as the sole parameter to the function.
    */
    template<typename ClassT>
    Callback1ActionImplementation(agx::Real time, void(ClassT::*function)(T), ClassT* object, const T& value );

    template<typename ClassT>
    Callback1ActionImplementation(agx::Real time, void(ClassT::*function)(const T&), ClassT* object, const T& value );

    /// Change the value that will be passed to the function when the action is triggered.
    void setValue( const T& newValue);

    /// \return The value that will be passed to the function when the action is triggered.
    const T& getValue() const;


    virtual void trigger(agx::Real currentTime);
    using ActionImplementation::trigger;

    virtual void visit(agxControl::ActionImplementationVisitor* operation, agx::Real time);
    virtual void visit(agxControl::ActionImplementationVisitor* operation, ActionImplementation* end, agx::Real time);
    virtual void dispatchVisit(agxControl::ActionImplementationVisitor* operation, agxControl::Callback1ActionImplementation<agx::Real>* begin, agx::Real time);

    using ActionImplementation::visit;
    using ActionImplementation::dispatchVisit;

    // Serialization code. Cannot use AGXSTREAM_DECLARE_SERIALIZABLE macro because
    // of templating. For example, the getConstructClassId() and getClassName()
    // methods have to translate the <T> part of the name into a substring containing
    // the actual type of T.
    agxStream::StorageAgent* getStorageAgent() const { return agxStream::StorageManager::instance()->find( getConstructClassId() );  }
    static const char* getConstructClassId();
    friend class agxStream::DefStorageAgent<Callback1ActionImplementation>;
    const char* getClassName() const;
    static agxStream::Serializable* create() { return new Callback1ActionImplementation(); }
    static agxStream::Serializable* create(agxStream::InputArchive&) { return new Callback1ActionImplementation(); }
    void store( agxStream::OutputArchive& out ) const;
    void restore( agxStream::InputArchive& in );

    // Used only by store/restore.
    Callback1ActionImplementation();

  protected:
    virtual ~Callback1ActionImplementation() {}
  private:
    Callback m_callback;
    T m_value;
  };


  /*
  An ActionImplementation that works the same way as the Callback1ActionImplementation,
  but supports member functions with a return value. The returned value is ignored.
  */
  template<typename R, typename T>
  class CallbackR1ActionImplementation : public ActionImplementation
  {
  public:
    typedef std::function<R (T)> Callback;


    template<typename ClassT>
    CallbackR1ActionImplementation(agx::Real time, R (ClassT::*function)(T), ClassT* object, const T& value );

    void setValue( const T& newValue);
    const T& getValue() const;

    virtual void trigger(agx::Real currentTime);
    using ActionImplementation::trigger;

    virtual void visit(agxControl::ActionImplementationVisitor* operation, agx::Real time);
    virtual void visit(agxControl::ActionImplementationVisitor *operation, ActionImplementation* end, agx::Real time);
    virtual void dispatchVisit(agxControl::ActionImplementationVisitor* operation, agxControl::CallbackR1ActionImplementation<void, agx::Real>* begin, agx::Real time);

    using ActionImplementation::visit;
    using ActionImplementation::dispatchVisit;

    // Serialization code. Cannot use AGXSTREAM_DECLARE_SERIALIZABLE macro because
    // of templating. For example, the getConstructClassId() and getClassName()
    // methods have to translate the <R, T> part of the name into a substring
    // containing the actual types for R and T.
    agxStream::StorageAgent* getStorageAgent() const { return agxStream::StorageManager::instance()->find( getConstructClassId() );  }
    static const char* getConstructClassId();
    friend class agxStream::DefStorageAgent<CallbackR1ActionImplementation>;
    const char* getClassName() const;
    static agxStream::Serializable* create() { return new CallbackR1ActionImplementation(); }
    static agxStream::Serializable* create(agxStream::InputArchive&) { return new CallbackR1ActionImplementation(); }
    void store( agxStream::OutputArchive& out ) const;
    void restore( agxStream::InputArchive& in );

    // Used only by store/restore.
    CallbackR1ActionImplementation();

  protected:
    virtual ~CallbackR1ActionImplementation() {}
  private:
    Callback m_callback;
    T m_value;
  };


  /*
  An ActionImplementation that works the same way as the Callback1ActionImplementation,
  but supports member functions with two arguments.
  */
  template<typename T1, typename T2>
  class Callback2ActionImplementation : public ActionImplementation
  {
  public:
    typedef std::function<void (T1, T2)> Callback;

    template<typename ClassT>
    Callback2ActionImplementation(agx::Real, void (ClassT::*function)(T1, T2), ClassT* object, const T1& value1, const T2& value2);

    void setValue1( const T1& newValue);
    void setValue2( const T2& newValue);

    const T1& getValue1() const;
    const T2& getValue2() const;

    virtual void trigger(agx::Real currentTime);
    using ActionImplementation::trigger;

    virtual void visit(agxControl::ActionImplementationVisitor* operation, agx::Real time);
    virtual void visit(agxControl::ActionImplementationVisitor *operation, ActionImplementation* end, agx::Real time);
    virtual void dispatchVisit(agxControl::ActionImplementationVisitor* operation, agxControl::Callback2ActionImplementation<agx::Real, agx::Int>* begin, agx::Real time);

    using ActionImplementation::visit;
    using ActionImplementation::dispatchVisit;

    // Serialization code. Cannot use AGXSTREAM_DECLARE_SERIALIZABLE macro because of templating.
    agxStream::StorageAgent* getStorageAgent() const { return agxStream::StorageManager::instance()->find( getConstructClassId() );  }
    static const char* getConstructClassId();
    friend class agxStream::DefStorageAgent<Callback2ActionImplementation>;
    const char* getClassName() const;
    static agxStream::Serializable* create() { return new Callback2ActionImplementation(); }
    static agxStream::Serializable* create(agxStream::InputArchive&) { return new Callback2ActionImplementation(); }
    void store( agxStream::OutputArchive& out ) const;
    void restore( agxStream::InputArchive& in );

    // Used only by store/restore.
    Callback2ActionImplementation();

  protected:
    virtual ~Callback2ActionImplementation() {}
  private:
    Callback m_callback;
    T1 m_value1;
    T2 m_value2;
  };





  /* Implementation, CallbackActionImplementation */

  template<typename ClassT>
  CallbackActionImplementation::CallbackActionImplementation(
      agx::Real time, void (ClassT::*function)(agx::Real), ClassT* object
  ):
      ActionImplementation(time), m_callback(std::bind(function, object,std::placeholders:: _1))
  {
  }


  /* Implementation, Callback1ActionImplementation */

  template<typename T>
  Callback1ActionImplementation<T>::Callback1ActionImplementation():
      ActionImplementation(agx::Real(-1.0)), m_value(T())
  {
  }


  template<typename T> template<typename ClassT>
  Callback1ActionImplementation<T>::Callback1ActionImplementation(
      agx::Real time, void (ClassT::*function)(T), ClassT* object, const T& value
  ):
      ActionImplementation(time), m_callback(std::bind(function, object,std::placeholders:: _1)), m_value(value)
  {
  }

  template<typename T> template<typename ClassT>
  Callback1ActionImplementation<T>::Callback1ActionImplementation(
      agx::Real time, void (ClassT::*function)(const T&), ClassT* object, const T& value
  ):
      ActionImplementation(time), m_callback(std::bind(function, object,std::placeholders:: _1)), m_value(value)
  {
  }

  template<typename T>
  inline void Callback1ActionImplementation<T>::setValue(const T& newValue)
  {
    m_value = newValue;
  }


  template<typename T>
  inline const T& Callback1ActionImplementation<T>::getValue() const
  {
    return m_value;
  }


  template<typename T>
  inline void Callback1ActionImplementation<T>::trigger(agx::Real /*currentTime*/)
  {
    m_callback(m_value);
  }


  template<typename T>
  inline void Callback1ActionImplementation<T>::visit(agxControl::ActionImplementationVisitor* operation, agx::Real time)
  {
    operation->update(this, time);
  }

  template<typename T>
  inline void Callback1ActionImplementation<T>::visit(
      agxControl::ActionImplementationVisitor* /*operation*/, ActionImplementation* /*end*/, agx::Real /*time*/)
  {
    agxAbort1("Action: Asked to visit Callback1Actions with unsupported template parameters.");
  }

  template<>
  inline void Callback1ActionImplementation<agx::Real>::visit(
      agxControl::ActionImplementationVisitor* operation, ActionImplementation* end, agx::Real time)
  {
    end->dispatchVisit(operation, this, time);
  }

  template<typename T>
  inline void Callback1ActionImplementation<T>::dispatchVisit(agxControl::ActionImplementationVisitor* /*operation*/,
      agxControl::Callback1ActionImplementation<agx::Real>* /*begin*/, agx::Real /*time*/)
  {
    agxAbort1("Action: Asked to dispatch visit Callback1Actions with unsupported template parameters.");
  }


  template<>
  inline void Callback1ActionImplementation<agx::Real>::dispatchVisit(agxControl::ActionImplementationVisitor* operation,
      agxControl::Callback1ActionImplementation<agx::Real> *begin, agx::Real time)
  {
    operation->update(begin, this, time);
  }

  template<typename T>
  inline void Callback1ActionImplementation<T>::store(agxStream::OutputArchive& out) const
  {
    this->ActionImplementation::store(out);
    out << agxStream::out("value", m_value);
  }


  template<typename T>
  inline void Callback1ActionImplementation<T>::restore(agxStream::InputArchive& in)
  {
    this->ActionImplementation::restore(in);
    in >> agxStream::in("value", m_value);
  }





  /* Implementation, CallbackR1ActionImplementation */

  template<typename R, typename T>
  CallbackR1ActionImplementation<R,T>::CallbackR1ActionImplementation():
      ActionImplementation(agx::Real(-1.0)), m_value(T(0))
  {
  }


  template<typename R, typename T> template<typename ClassT>
  CallbackR1ActionImplementation<R,T>::CallbackR1ActionImplementation(
      agx::Real time, R (ClassT::*function)(T), ClassT* object, const T &value
  ):
      ActionImplementation(time), m_callback(std::bind(function, object,std::placeholders:: _1)), m_value(value)
  {
  }


  template<typename R, typename T>
  inline void CallbackR1ActionImplementation<R, T>::setValue(const T& newValue)
  {
    m_value = newValue;
  }


  template<typename R, typename T>
  inline const T& CallbackR1ActionImplementation<R, T>::getValue() const
  {
    return m_value;
  }


  template<typename R, typename T>
  inline void CallbackR1ActionImplementation<R, T>::trigger(agx::Real /*currentTime*/)
  {
    m_callback(m_value);
  }


  template<typename R, typename T>
  inline void CallbackR1ActionImplementation<R,T>::visit(agxControl::ActionImplementationVisitor* operation, agx::Real time)
  {
    operation->update(this, time);
  }


  template<typename R, typename T>
  inline void CallbackR1ActionImplementation<R, T>::visit(
      agxControl::ActionImplementationVisitor* /*operation*/, ActionImplementation* /*end*/, agx::Real /*time*/)
  {
    agxAbort1("Action: Asked to visit CallbackR1Actions with unsupported template parameters.");
  }

  template<>
  inline void CallbackR1ActionImplementation<void, agx::Real>::visit(
      agxControl::ActionImplementationVisitor* operation, ActionImplementation* end, agx::Real time)
  {
    end->dispatchVisit(operation, this, time);
  }

  template<typename R, typename T>
  inline void CallbackR1ActionImplementation<R, T>::dispatchVisit(agxControl::ActionImplementationVisitor* /*operation*/,
      agxControl::CallbackR1ActionImplementation<void, agx::Real>* /*begin*/, agx::Real /*time*/)
  {
    agxAbort1("Action: Asked to mix CallbackR1Actions with unsupported template parameters.");
  }

  template<>
  inline void CallbackR1ActionImplementation<void, agx::Real>::dispatchVisit(
      agxControl::ActionImplementationVisitor* operation,
      agxControl::CallbackR1ActionImplementation<void, agx::Real> *begin, agx::Real time)
  {
    operation->update(begin, this, time);
  }

  template<typename R, typename T>
  inline void CallbackR1ActionImplementation<R, T>::store(agxStream::OutputArchive& out) const
  {
    this->ActionImplementation::store(out);
    out << agxStream::out("value", m_value);
  }


  template<typename R, typename T>
  inline void CallbackR1ActionImplementation<R, T>::restore(agxStream::InputArchive& in)
  {
    this->ActionImplementation::restore(in);
    in >> agxStream::in("value", m_value);
  }



  /* Implementation, Callback2ActionImplementation */

  template<typename T1, typename T2>
  Callback2ActionImplementation<T1, T2>::Callback2ActionImplementation():
      ActionImplementation(agx::Real(-1.0)), m_value1(T1(0)), m_value2(T2(0))
  {
  }

  template<typename T1, typename T2> template<typename ClassT>
  Callback2ActionImplementation<T1, T2>::Callback2ActionImplementation(
      agx::Real time, void (ClassT::*function)(T1, T2), ClassT *object, const T1 &value1, const T2 &value2
  ):
      ActionImplementation(time), m_callback(std::bind(function, object,std::placeholders:: _1,std::placeholders:: _2)),
      m_value1(value1), m_value2(value2)
  {
  }


  template<typename T1, typename T2>
  inline void Callback2ActionImplementation<T1, T2>::setValue1(const T1& newValue)
  {
    m_value1 = newValue;
  }


  template<typename T1, typename T2>
  inline void Callback2ActionImplementation<T1, T2>::setValue2(const T2& newValue)
  {
    m_value2 = newValue;
  }


  template<typename T1, typename T2>
  inline const T1& Callback2ActionImplementation<T1, T2>::getValue1() const
  {
    return m_value1;
  }


  template<typename T1, typename T2>
  inline const T2& Callback2ActionImplementation<T1, T2>::getValue2() const
  {
    return m_value2;
  }


  template<typename T1, typename T2>
  inline void Callback2ActionImplementation<T1, T2>::trigger(agx::Real /*currentTime*/)
  {
    m_callback(m_value1, m_value2);
  }


  template<typename T1, typename T2>
  inline void Callback2ActionImplementation<T1, T2>::visit(agxControl::ActionImplementationVisitor* operation, agx::Real time)
  {
    operation->update(this, time);
  }

  template<typename T1, typename T2>
  inline void Callback2ActionImplementation<T1, T2>::visit(
      agxControl::ActionImplementationVisitor* /*operation*/, ActionImplementation* /*end*/, agx::Real /*time*/)
  {
    agxAbort1("Action: Asked to visit Callback2Actions with unsupported template parameters.");
  }

  template<>
  inline void Callback2ActionImplementation<agx::Real, agx::Int>::visit(
      agxControl::ActionImplementationVisitor* operation, ActionImplementation* end, agx::Real time)
  {
    end->dispatchVisit(operation, this, time);
  }

  template<typename T1, typename T2>
  inline void Callback2ActionImplementation<T1, T2>::dispatchVisit(agxControl::ActionImplementationVisitor* /*operation*/,
      agxControl::Callback2ActionImplementation<agx::Real, agx::Int>* /*begin*/, agx::Real /*time*/)
  {
    agxAbort1("Action: Asked to mix Callbac2Actions with unsupported template parameters.");
  }

  template<>
  inline void Callback2ActionImplementation<agx::Real, agx::Int>::dispatchVisit(
      agxControl::ActionImplementationVisitor*operation,
      agxControl::Callback2ActionImplementation<agx::Real, agx::Int> *begin,
      agx::Real time)
  {
    operation->update(begin, this, time);
  }


  template<typename T1, typename T2>
  inline void Callback2ActionImplementation<T1, T2>::store(agxStream::OutputArchive& out) const
  {
    this->ActionImplementation::store(out);
    out << agxStream::out("Value1", m_value1);
    out << agxStream::out("Value2", m_value2);
  }


  template<typename T1, typename T2>
  inline void Callback2ActionImplementation<T1, T2>::restore(agxStream::InputArchive& in)
  {
    this->ActionImplementation::restore(in);
    in >> agxStream::in("Value1", m_value1);
    in >> agxStream::in("Value2", m_value2);
  }



  /* Serialization helper code and type-to-string translations. */

  template<typename T> inline  std::string templateToFormatName() { return std::string("UnknownFormat: \'") + typeid(T).name() + "\'"; }
  template<> std::string inline templateToFormatName<bool>() { return std::string("Bool:8bit");  }
  template<> std::string inline templateToFormatName<agx::Int32>()  { return std::string("Int:32bit");  }
  template<> std::string inline templateToFormatName<agx::Int64>()  { return std::string("Int:64bit");  }
  template<> std::string inline templateToFormatName<agx::Vec3f>()  { return std::string("Vec3:32bit"); }
  template<> std::string inline templateToFormatName<agx::Vec3d>()  { return std::string("Vec3:64bit"); }

  template<typename T>
  inline const char* Callback1ActionImplementation<T>::getConstructClassId()
  {
    // Cannot use stripClassName because it doesn't handle the '::' in the template
    // parameters properly. For example when T=agx::Vec3T<double>.
    m_className = "agxControl::Callback1ActionImplementation";
    m_className = m_className + std::string("<") + templateToFormatName<T>() + ">";
    return  m_className.c_str();
  }

  template<>
  inline const char* Callback1ActionImplementation<agx::Real>::getConstructClassId()
  {
    return "agxControl::Callback1ActionImplementation<Real>";
  }

  template<typename T>
  inline const char* Callback1ActionImplementation<T>::getClassName() const
  {
    // Cannot use stripClassName because it doesn't handle the '::' in the template
    // parameters properly. For example when T=agx::Vec3T<double>.
    m_className = "agxControl::Callback1ActionImplementation";
    m_className = m_className + std::string("<") + templateToFormatName<T>() + ">";
    return  m_className.c_str();
  }

  template<>
  inline const char* Callback1ActionImplementation<agx::Real>::getClassName() const
  {
    return "agxControl::Callback1ActionImplementation<Real>";
  }


  template<typename R, typename T>
  inline const char* CallbackR1ActionImplementation<R, T>::getConstructClassId()
  {
    // Cannot use stripClassName because it doesn't handle the space in "<R, T>" properly,
    //std::string className = agxStream::StorageAgent::stripClassName( "getConstructClassId", AGX_FUNCTION );
    m_className = "agxControl::CallbackR1ActionImplementation<R, T>";
    m_className = m_className.substr(0, m_className.length() - 6); // Strip away "<R, T>".
    m_className += std::string("<") + templateToFormatName<R>() + ", " + templateToFormatName<T>() + ">";
    return  m_className.c_str();
  }

  template<>
  inline const char* CallbackR1ActionImplementation<agx::Real, agx::Real>::getConstructClassId()
  {
    return "agxControl::CallbackR1ActionImplementation<Real, Real>";
  }

  template<typename R, typename T>
  inline const char* CallbackR1ActionImplementation<R, T>::getClassName() const
  {
    // Cannot use stripClassName because it doesn't handle the space in "<R, T>" properly,
    m_className = "agxControl::CallbackR1ActionImplementation<R, T>";
    m_className = m_className.substr(0, m_className.length() - 6); // Strip away "<R, T>".
    m_className += std::string("<") + templateToFormatName<R>() + ", " + templateToFormatName<T>() + ">";
    return  m_className.c_str();
  }

  template<>
  inline const char* CallbackR1ActionImplementation<agx::Real, agx::Real>::getClassName() const
  {
    return "agxControl::CallbackR1ActionImplementation<Real, Real>";
  }


  template<typename T1, typename T2>
  inline const char* Callback2ActionImplementation<T1, T2>::getConstructClassId()
  {
    // Cannot use stripClassName because it doesn't handle the space in "<T1, T2>" properly,
    m_className = "agxControl::Callback2ActionImplementation<T1, T2>";
    m_className = m_className.substr(0, m_className.length() - 8); // Strip away "<T1, T2>"
    m_className += std::string("<") + templateToFormatName<T1>() + ", " + templateToFormatName<T2>() + ">";
    return m_className.c_str();
  }

  template<>
  inline const char* Callback2ActionImplementation<agx::Real, agx::Real>::getConstructClassId()
  {
    return "agxControl::Callback2ActionImplementation<Real, Real>";
  }

  template<>
  inline const char* Callback2ActionImplementation<agx::Real, agx::Int32>::getConstructClassId()
  {
    return "agxControl::Callback2ActionImplementation<Real, Int32>";
  }

  template<>
  inline const char* Callback2ActionImplementation<agx::Real, agx::Int64>::getConstructClassId()
  {
    return "agxControl::Callback2ActionImplementation<Real, Int64>";
  }

  template<typename T1, typename T2>
  inline const char* Callback2ActionImplementation<T1, T2>::getClassName() const
  {
    // Cannot use stripClassName because it doesn't handle the space in "<T1, T2>" properly,
    m_className = "agxControl::Callback2ActionImplementation<T1, T2>";
    m_className = m_className.substr(0, m_className.length() - 8); // Strip away "<T1, T2>"
    m_className += std::string("<") + templateToFormatName<T1>() + ", " + templateToFormatName<T2>() + ">";
    return m_className.c_str();
  }

  template<>
  inline const char* Callback2ActionImplementation<agx::Real, agx::Real>::getClassName() const
  {
    return "agxControl::Callback2ActionImplementation<Real, Real>";
  }

  template<>
  inline const char* Callback2ActionImplementation<agx::Real, agx::Int32>::getClassName() const
  {
    return "agxControl::Callback2ActionImplementation<Real, Int32>";
  }

  template<>
  inline const char* Callback2ActionImplementation<agx::Real, agx::Int64>::getClassName() const
  {
    return "agxControl::Callback2ActionImplementation<Real, Int64>";
  }
}

// Include guard
#endif

/// \endcond
