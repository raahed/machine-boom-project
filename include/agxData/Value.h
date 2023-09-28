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

#ifndef AGXDATA_VALUE_H
#define AGXDATA_VALUE_H

#include <agx/Object.h>
#include <agx/Event.h>
#include <agx/ref_ptr.h>
#include <agxData/Scalar.h>

namespace agxNet { class Device; }

namespace agxData
{
  class Type;

  AGX_DECLARE_POINTER_TYPES(Value);
  AGX_DECLARE_VECTOR_TYPES(Value);

  /**
  Abstract representation of a value. The value has a name and specified format,
  and since it is an agx::Object it can be inserted as a node in the system
  hierarchy.

  Values can bind to other values, and is automatically synchronized when
  the reference value is updated.
  */
  class AGXCORE_EXPORT Value : public agx::Object
  {
  public:
    static agx::Model *ClassModel();

    static Value *load(agx::TiXmlElement *eValue, agx::Device *device);
    virtual void configure(agx::TiXmlElement *eValue) override;
    virtual void save(agx::TiXmlElement *eParent) const override;
    virtual void snapshot(agx::TiXmlNode *eParent, const agx::String& directory) const override;

  public:

    typedef agx::Event1<Value *> Event;

    /// Triggered when the value is updated
    Event updateEvent; // TODO Remove!

    // Generated event interface, defined in Value.agxEvent
    #include <agx/ValueEvent.Declaration.h>

#ifndef SWIG
    void addListener(EventListener *listener);
    void removeListener(EventListener *listener);
    bool hasListener(EventListener *listener);
#endif

  public:
    Value(const agx::Name& name, const agxData::Type *type);
    Value(const agx::Name& name, const agxData::Format *format);

    Value(const agxData::Type *type);
    Value(const agxData::Format *format);

    /**
    \return The value
    */
    const void *ptr() const;

    template <typename T>
    const T& get() const;

    template <typename T>
    T transform() const;

    /**
    Explicitly set the value, removes active binding and disables autobinding.
    */
    void set(const Value *other);


    template <typename T>
    void set(const T& value);


    /**
    Wrapper for this->getRootBinding()->set()
    */
    template <typename T>
    void signal(const T& value);


    /**
    Set the default value of the format.
    */
    void setDefaultValue();

    /**
    Configure value using an expression.
    Internal use only, deprecated syntax.
    */
    void setExpression(const agx::String& expression);

    /**
    \return The value type.
    */
    agxData::Type *getType();
    const agxData::Type *getType() const;

    /**
    \return The value format.
    */
    agxData::Format *getFormat();
    const agxData::Format *getFormat() const;

    /**
    \return The parent value binding.
    */
    Value *getBinding();
    const Value *getBinding() const;

    /**
    \return The root value binding.
    */
    Value *getRootBinding();
    const Value *getRootBinding() const;

    /**
    \return The auto-bind path.
    */
    const agx::Path& getBindPath() const;


    /**
    Set bind path which will auto-bind when resolved.
    */
    void bind(const agx::Path& path);

    /**
    Explicit binding, also disables auto-binding.
    */
    void bind(Value *binding);

    /**
    Unbind, also disables auto-binding.
    */
    void unbind(bool removeAutoBindPath = true);

    /**
    Rebind with the current auto-bind path.
    */
    virtual void rebind() override;

    /**
    \return true if value is valid
    */
    bool isValid() const;

    virtual agxData::Data *getData();

    void print() const;
    agx::String toString() const;
    bool isTypeCompatible(const Value *other);

    Value *clone() const;
    static void copy(Value *target, const Value *source);

    // Needed so template does not capture these cases
    void set(Value *other);
    void set(ValueRef& other);
    void set(const ValueRef& other);

    void *ptr();


    void setFormat(agxData::Format *format);

    virtual void buildNavigationTree(agxJson::Value& eNode) const override;
  protected:
    virtual ~Value();

    virtual bool bind(Object *object);

    // void setType(const agxData::Type *type);
    void forceRemove();
  private:
    friend class agxNet::Device;
    // void updateCallback(Value *other);
    void set(const void *value);
    void setStorage(void *ptr);
    void propagateUpdate();

    static void allocate(Value *value);
    static void deallocate(Value *value);

  private:
    friend class agxData::Format;
    void forceInvalidateDefaultFormatValue();

  protected:
    agx::Path m_bindPath;
    ValueRef m_binding;
    ValuePtrVector m_children;
    AbstractScalar m_value;
    FormatRef m_format;
    agx::UInt32 m_allocationIndex;
    bool m_restartIteration;
    EventDispatch m_eventDispatch;
    EventListener m_bindingListener;
  };

  typedef agx::HashTable<agx::String, ValueRef> ValueTable;

  AGXCORE_EXPORT std::ostream& operator << ( std::ostream& output, const Value& value );

  //----------------------------------------------------------------------------------

  /// Templated value
  template <typename T>
  class ValueT : public Value
  {
  public:
    template <typename T2>
    static bool ValidateCast(const agx::Referenced *object);

  public:
    ValueT(const agx::Name& name);
    ValueT(const T& value = T());
    ValueT(const agx::Name& name, const T& value);


    virtual ~ValueT();

    operator const T& () const;

    void set(const T& value);
    void signal(const T& value);
    const T& get() const;

    ValueT<T>& operator= (const ValueT<T>& value);
    ValueT<T>& operator= (const T& value);
  };

  /// Templated value-reference
  template <typename T>
  class ValueRefT : public agx::ref_ptr< ValueT<T> >
  {
  public:
    typedef agx::ref_ptr< ValueT<T> > RefT;

  public:
    ValueRefT() {}
    ValueRefT(ValueT<T>* val) : RefT(val) {}

    AGX_FORCE_INLINE operator const ValueT<T>*() const { return this->get(); }
    AGX_FORCE_INLINE ValueT<T>& operator*() const { return *this->get(); }
    AGX_FORCE_INLINE ValueT<T>* operator->() const { return this->get(); }
  };


  //----------------------------------------------------------------------------------


  /**
  Only use as member allocated variable.
  */
  template <typename T>
  class Val : public ValueT<T>
  {
  public:
    Val(const agx::Name& name) : ValueT<T>(name)
    {
      this->reference();
    }

    Val(const agx::Name& name, const T& value) : ValueT<T>(name, value)
    {
      this->reference();
    }

    Val(const T& value = T()) : ValueT<T>(value)
    {
      this->reference();
    }

    virtual ~Val()
    {
      this->forceRemove();
    }


    const T *operator-> () const { return &this->get(); }
    Val<T>& operator= (const Val<T>& value)
    {
      this->operator=(value.get());
      return *this;
    }

    Val<T>& operator= (const T& value)
    {
      this->set(value);
      return *this;
    }
  };


  //----------------------------------------------------------------------------------


  /* Implementation */
  AGX_FORCE_INLINE void *Value::ptr() { return m_value.ptr(); }
  AGX_FORCE_INLINE const void *Value::ptr() const { return m_value.ptr(); }

  template <typename T>
  AGX_FORCE_INLINE const T& Value::get() const
  {
    agxAssert(agxCore::isShutdown() || (m_format && m_format->is(agxData::getFormat<T>())));
    agxAssertN(this->isValid(), "Can not dereference an invalid value! Value \'%s\' in component \'%s\'", this->getName().c_str(), this->getContext() ? this->getContext()->getName().c_str() : "nullptr");
    return *(const T *)m_value.ptr();
  }

  template <typename T>
  inline T Value::transform() const
  {
    T result;
    agxData::transform(&result, agxData::getFormat<T>(), m_value.ptr(), m_format, 1);
    return result;
  }

  AGX_FORCE_INLINE void Value::set(Value *other)
  {
    this->set(const_cast<const Value *>(other));
  }

  AGX_FORCE_INLINE void Value::set(ValueRef& other)
  {
    this->set(const_cast<const Value *>(other.get()));
  }

  AGX_FORCE_INLINE void Value::set(const ValueRef& other)
  {
    this->set(const_cast<const Value *>(other.get()));
  }

  template <typename T>
  AGX_FORCE_INLINE void Value::set(const T& value)
  {
    if (!m_format)
      this->setFormat(agxData::getFormat<T>());

    agxAssert(agxCore::isShutdown() || m_format->is(agxData::getFormat<T>()));
    this->set((const void *)&value);
  }

  template <typename T>
  AGX_FORCE_INLINE void Value::signal(const T& value)
  {
    this->getRootBinding()->set(value);
  }

  AGX_FORCE_INLINE agxData::Type *Value::getType()
  {
    return m_format ? m_format->getType() : nullptr;
  }

  AGX_FORCE_INLINE const agxData::Type *Value::getType() const
  {
    return m_format ? m_format->getType() : nullptr;
  }

  AGX_FORCE_INLINE agxData::Format *Value::getFormat() { return m_format; }
  AGX_FORCE_INLINE const agxData::Format *Value::getFormat() const { return m_format; }
  AGX_FORCE_INLINE Value *Value::getBinding() { return m_binding; }
  AGX_FORCE_INLINE const Value *Value::getBinding() const { return m_binding; }

  AGX_FORCE_INLINE bool Value::isValid() const
  {
    return m_value.ptr() != nullptr;
  }

  AGX_FORCE_INLINE const agx::Path& Value::getBindPath() const
  {
    return m_bindPath;
  }

  //----------------------------------------------------------------------------------/////////


  template <typename T> template <typename T2>
  bool ValueT<T>::ValidateCast(const agx::Referenced *object)
  {
    const Value *value = dynamic_cast<const Value *>(object);
    return value ? value->getFormat() == agxData::getFormat<T>() : false;
  }


  template <typename T>
  ValueT<T>::ValueT(const agx::Name& name) : Value(name, agxData::getFormat<T>())
  {
  }


  template <typename T>
  ValueT<T>::ValueT(const agx::Name& name, const T& value) : Value(name, agxData::getFormat<T>())
  {
    this->set(value);
  }

  template <typename T>
  ValueT<T>::ValueT(const T& value) : Value(agxData::getFormat<T>())
  {
    this->set(value);
  }

  template <typename T>
  ValueT<T>::~ValueT()
  {
  }


  template <typename T>
  AGX_FORCE_INLINE ValueT<T>::operator const T& () const { return this->get(); }

  template <typename T>
  AGX_FORCE_INLINE void ValueT<T>::set(const T& value)
  {
    Value::set(value);
  }

  template <typename T>
  AGX_FORCE_INLINE void ValueT<T>::signal(const T& value)
  {
    Value::signal(value);
  }


  template <typename T>
  AGX_FORCE_INLINE const T& ValueT<T>::get() const
  {
    agxAssertN(this->isValid(), "Can not dereference an invalid value (%s)!", this->getPath().c_str());
    return *(const T*)Value::ptr();
  }

  template <typename T>
  AGX_FORCE_INLINE ValueT<T>& ValueT<T>::operator= (const ValueT<T>& value)
  {
    this->operator=(value.get());
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE ValueT<T>& ValueT<T>::operator= (const T& value)
  {
    this->set(value);
    return *this;
  }


}

#include <agx/ValueEvent.Implementation.h>

#endif /* _AGXDATA_VALUE_H_ */
