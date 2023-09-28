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

#ifndef AGX_PARAMETER_H
#define AGX_PARAMETER_H

#ifdef _MSC_VER
# pragma warning( disable: 4714 )// const' marked as __forceinline not inlined
#endif


#include <agx/Object.h>
#include <agxData/Value.h>
#include <agx/Pointer.h>
#include <agxData/Buffer.h>
#include <agxData/EntityModel.h>
#include <agxData/EntityStorage.h>

namespace agxData { class FunctionValue; }

namespace agx
{
  class Task;
  AGX_DECLARE_POINTER_TYPES(Parameter);
  AGX_DECLARE_POINTER_TYPES(UnresolvedParameter);
  AGX_DECLARE_POINTER_TYPES(ScalarParameter);
  AGX_DECLARE_POINTER_TYPES(ArrayParameter);
  AGX_DECLARE_POINTER_TYPES(PointerParameter);
  AGX_DECLARE_POINTER_TYPES(EntityDataParameter);
  AGX_DECLARE_POINTER_TYPES(EntityPtrParameter);
  AGX_DECLARE_POINTER_TYPES(EntityInstanceParameter);

  AGX_DECLARE_VECTOR_TYPES(Parameter);
  AGX_DECLARE_POINTER_TYPES(UnresolvedParameter);
  AGX_DECLARE_VECTOR_TYPES(ScalarParameter);
  AGX_DECLARE_VECTOR_TYPES(ArrayParameter);
  AGX_DECLARE_VECTOR_TYPES(PointerParameter);
  AGX_DECLARE_VECTOR_TYPES(EntityDataParameter);
  AGX_DECLARE_VECTOR_TYPES(EntityPtrParameter);
  AGX_DECLARE_VECTOR_TYPES(EntityInstanceParameter);

  typedef HashTable<String, UnresolvedParameterRef> ParameterTable;

  /**
  A representation of a task/kernel parameter, which can be bound using
  a string path or explicitly to a data node.
  */
  class AGXCORE_EXPORT Parameter : public agx::Object
  {
  public:
    static agx::Model *ClassModel();

    typedef Event1<Parameter *> Event;
    Event bindEvent;
    Event unbindEvent;
    Event updateEvent;

    /// Specify which kind of parameter
    enum MetaType
    {
      SCALAR,
      ARRAY,
      POINTER,
      ENTITY_DATA,
      ENTITY_PTR,
      ENTITY_INSTANCE,
      UNRESOLVED
    };

  public:
    /**
    \return The task in which the parameter list contain this parameter.
    */
    agx::Task *getTask();
    const agx::Task *getTask() const;

    /**
    \return The parameter index.
    */
    size_t getIndex() const;

    /**
    \return The parameter meta type.
    */
    MetaType getMetaType() const;

    /**
     \return The parameter meta type as a string.
     */
    const agx::String& getMetaTypeName() const;

    /**
    \return True if the parameter is required.
    */
    bool isRequired() const;

    /**
    Set if the parameter is required, default is true.
    */
    void setRequired(bool flag);

    /**
    \return The parameter format.
    */
    agxData::Format *getFormat();
    const agxData::Format *getFormat() const;

    /**
    \return The parameter type.
    */
    virtual agxData::Type* getType();
    virtual const agxData::Type* getType() const;

    /**
    \return The access mode of the parameter.
    */
    UInt8 getAccessMode() const;

    /**
    \return The bind path of the parameter.
    */
    const agx::Path& getBindPath() const;

    /**
    \return True if the parameter has a valid binding.
    */
    // bool isValid() const;
    virtual bool isValid() const = 0;

    /**
    \return The current parameter binding.
    */
    agx::Object *getBinding();
    const agx::Object *getBinding() const;


    /**
    Set explicit binding.
    */
    void bind(agx::Object *binding);

    /**
    Set parameter bind path.
    */
    void bind(const agx::Path& path);

    /**
    Set the parameter, alias for bind.
    */
    void set(agx::Object *binding);

    /**
    Remove current binding and bind path.
    */
    void unbind();

    ///\return True if the parameter has a global result
    virtual bool hasGlobalResult() const;


    virtual void rebind() override;
    virtual Object *getResourceImpl(const Path& path, agx::Model *model) override;

    virtual void update() = 0;
    virtual void commit() = 0;

    bool needUpdate();
    bool needCommit();

  protected:
    Parameter(Model *model, MetaType type, const Name& name, agxData::Format *format, UInt8 accessMode, const Path& bindPath = Path());
    virtual ~Parameter();
    virtual void _bind(Object *binding) = 0;

  private:
    Parameter& operator=(const Parameter&) = delete;
    friend class Task;
    friend class agxData::FunctionValue;
    friend class EntityDataParameter;
    void setTask(Task *task, size_t index);

    virtual void prepareGlobalResultSorting(size_t numJobs);

  protected:
    Task *m_task;
    size_t m_index;
    agxData::FormatRef m_format;
    Path m_bindPath;
    ObjectObserver m_binding;
    const UInt8 m_accessMode;
    bool m_required;
    UInt8 m_metaType;
    bool m_needUpdate;
    bool m_needCommit;
  };

  //---------------------------------------------------------------

  /**
  Unresolved parameter.
  */
  class AGXCORE_EXPORT UnresolvedParameter : public Parameter
  {
  public:
    static agx::Model *ClassModel();

    static UnresolvedParameter *load(TiXmlElement *eParameter, Device *device);
    virtual void configure(TiXmlElement *eParameter) override;

  public:
    UnresolvedParameter(const agx::Name& name, const agx::String& expression);
    UnresolvedParameter(const agx::Name& name, MetaType type, const agx::String& expression);

    virtual bool isValid() const override;

    agxData::Type *getType() override;
    const agxData::Type *getType() const override;

  protected:
    virtual ~UnresolvedParameter();
    virtual void _bind(Object *binding) override;
    virtual void update() override;
    virtual void commit() override;
    void triggerUpdateCallback(agxData::Value *value);
    void init(const String& expression);

  private:
    UnresolvedParameter& operator=(const UnresolvedParameter&) = delete;

  private:
    agxData::TypeRef m_type;
    agxData::Value::Event::CallbackType m_updateCallback;
    agxData::ValueRef m_localValue;
  };


  //---------------------------------------------------------------

  /**
  Scalar parameter.
  */
  class AGXCORE_EXPORT ScalarParameter : public Parameter
  {
  public:
    static agx::Model *ClassModel();

    static ScalarParameter *load(TiXmlElement *eParameter, Device *device);
    virtual void configure(TiXmlElement *eParameter) override;

  public:
    ScalarParameter(const agx::Name& name, agxData::Format *format, UInt8 accessMode = agxData::READ, const String& expression = "");
    ScalarParameter(const agx::Name& name, agxData::Format *format, UInt8 accessMode, agxData::Value *binding);

    /**
    \return The current value.
    */
    agxData::Value *getValue();
    const agxData::Value *getValue() const;

    template <typename T>
    const T& get() const;

    /**
    Set the parameter value.
    */
    void set(agxData::Value *value);

    template <typename T>
    void set(const T& value);

    /** Only for internal use from the task containing this Parameter. */
    template <typename T>
    void write(const T& value, const Task* owner);

    /**
    Set the parameter using an expression.
    */
    void setExpression(const String& expression);


    // Only used by kernel dispatching
    void *ptr();

    virtual bool isValid() const override;
    virtual Object *getResourceImpl(const Path& path, agx::Model *model) override;


    void setReal( Real value );
    Real getReal();

    virtual void update() override;
    virtual void commit() override;

  protected:
    virtual ~ScalarParameter();
    virtual void _bind(Object *binding) override;

  private:
    ScalarParameter& operator=(const ScalarParameter&) = delete;
    void init();
    void triggerUpdateCallback(agxData::Value *value);

  private:
    // Parameter value is never replaced, binding the parameter will bind the value.
    agxData::ValueRef m_value;

    agxData::Value::Event::CallbackType m_updateCallback;
  };

  //---------------------------------------------------------------

  /**
  Array parameter.
  */
  class AGXCORE_EXPORT ArrayParameter : public Parameter
  {
  public:
    static agx::Model *ClassModel();


    static ArrayParameter *load(TiXmlElement *eParameter, Device *device);
    virtual void configure(TiXmlElement *eParameter) override;

  public:
    ArrayParameter(const agx::Name& name, agxData::Type *type, UInt8 accessMode = agxData::READ, const Path& bindPath = Path());
    ArrayParameter(const agx::Name& name, agxData::Type *type, UInt8 accessMode, agxData::Buffer *binding);

    ArrayParameter(const agx::Name& name, agxData::Format *format, UInt8 accessMode = agxData::READ, const Path& bindPath = Path());
    ArrayParameter(const agx::Name& name, agxData::Format *format, UInt8 accessMode, agxData::Buffer *binding);

    /**
    \return The parameter type.
    */
    agxData::Type *getType() override;
    const agxData::Type *getType() const override;

    /**
    \return The current buffer binding.
    */
    agxData::Buffer *getBuffer();
    const agxData::Buffer *getBuffer() const;

    /**
    Access the data of the current buffer binding.
    */
    template <typename T>
    agxData::Array<T>& get();

    template <typename T>
    const agxData::Array<T>& get() const;

    /**
    Create a local memory allocation. Eg. __local in OpenCL.
    */
    void setLocalSize(size_t numElements);

    /**
    \return The local memory size, number of elements (not bytes).
    */
    size_t getLocalSize() const;

    /**
    \return The number of elements of the currently bound buffer.
    */
    size_t getNumElements() const;


    /**
    Set the global result mode
    */
    void setGlobalResultMode(UInt mode);

    /**
    \return The global result mode.
    */
    UInt getGlobalResultMode() const;

    /**
    \return The global result.
    */
    GlobalResultBuffer *getGlobalResult();
    const GlobalResultBuffer *getGlobalResult() const;

    virtual bool hasGlobalResult() const override;


    virtual bool isValid() const override;
    virtual Object *getResourceImpl(const Path& path, agx::Model *model) override;

    virtual void update() override;
    virtual void commit() override;

  protected:
    virtual ~ArrayParameter();
    virtual void _bind(Object *binding) override;
    virtual void prepareGlobalResultSorting(size_t numJobs) override;

  private:
    ArrayParameter& operator=(const ArrayParameter&) = delete;
    friend class Task;
    void parentBindCallback(Parameter *parent);
    void parentUnbindCallback(Parameter *parent);
    void setBuffer(agxData::Buffer *buffer);

  private:
    static agxData::AbstractArray s_dummyArray;
    agxData::TypeRef m_type;
    agxData::BufferObserver m_buffer;
    size_t m_localMemorySize;
    UInt m_globalResultMode;
    GlobalResultBufferRef m_globalResult;
    Parameter::Event::CallbackType m_parentBindCallback;
    Parameter::Event::CallbackType m_parentUnbindCallback;
  };

  //---------------------------------------------------------------

  /**
  Pointer parameter.
  */
  class AGXCORE_EXPORT PointerParameter : public Parameter, public Object::EventListener
  {
  public:
    static agx::Model *ClassModel();

    static PointerParameter *load(TiXmlElement *eParameter, Device *device);
    virtual void configure(TiXmlElement *eParameter) override;

  public:
    PointerParameter(const agx::Name& name, const Path& bindPath = Path());
    PointerParameter(const agx::Name& name, Object *binding);

    Object* get();
    const Object* get() const;

    template <typename T>
    T* get();

    template <typename T>
    const T* get() const;

    virtual bool isValid() const override;
    virtual Object *getResourceImpl(const Path& path, agx::Model *model) override;

    virtual void update() override;
    virtual void commit() override;

  protected:
    virtual ~PointerParameter();
    virtual void _bind(Object *binding) override;

    // From Object::EventListener
    virtual void destroyCallback(agx::Object* object) override;

  private:
    PointerParameter& operator=(const PointerParameter&) = delete;
    void init();
    void triggerUpdateCallback(agxData::Value *value);

  private:
    agxData::Value::Event::CallbackType m_updateCallback;
  };

  //---------------------------------------------------------------

  /**
  Entity parameter.
  */
  class AGXCORE_EXPORT EntityDataParameter : public Parameter
  {
  public:
    static agx::Model *ClassModel();


    static EntityDataParameter *load(TiXmlElement *eParameter, Device *device);
    virtual void configure(TiXmlElement *eParameter) override;

  public:
    EntityDataParameter(const agx::Name& name, agxData::EntityModel *entity, const Path& bindPath);

    void addChildParameter(ArrayParameter *parameter);
    void addChildParameter(ScalarParameter* parameter);

    /**
    \return The entity model
    */
    agxData::EntityModel *getEntityModel();

    /**
    \return The storage binding.
    */
    agxData::EntityStorage *getStorage();

    /**
    \return The data set binding for this parameter.
    */
    agxData::EntityData *getData();

    template <typename T>
    T& getData();

    /**
    Set the global result mode
    */
    void setGlobalResultMode(UInt mode);

    /**
    \return The global result mode.
    */
    UInt getGlobalResultMode() const;

    /**
    \return The global result.
    */
    GlobalResultStorage *getGlobalResult();
    const GlobalResultStorage *getGlobalResult() const;

    // Only used by KernelGenerator for non-required parameters to set up a dummy binding
    void setData(agxData::EntityData *dataSet);

    const ParameterRefVector& getChildParameters() const;

    virtual bool isValid() const override;
    virtual Object *getResourceImpl(const Path& path, agx::Model *model) override;

    virtual void update() override;
    virtual void commit() override;

    virtual bool hasGlobalResult() const override;

  protected:
    virtual ~EntityDataParameter();
    virtual void _bind(Object *binding) override;
    virtual void prepareGlobalResultSorting(size_t numJobs) override;

    // Protected because only Scalar- and ArrayParameters may be added.
    void addChildParameter(Parameter* parameter);

  private:
    EntityDataParameter& operator=(const EntityDataParameter&) = delete;

  private:
    agxData::EntityModelRef m_entityModel;
    agxData::EntityStorageRef m_storage;
    agxData::EntityDataRef m_dataSet;
    ParameterRefVector m_children;
    UInt m_globalResultMode;
    GlobalResultStorageRef m_globalResult;
    bool m_childrenNeedCommit;
  };


  /**
  EntityPtr parameter.
  */
  class AGXCORE_EXPORT EntityPtrParameter : public Parameter
  {
  public:
    static agx::Model *ClassModel();

    static EntityPtrParameter *load(TiXmlElement *eParameter, Device *device);
    virtual void configure(TiXmlElement *eParameter) override;

  public:
    EntityPtrParameter(const agx::Name& name, agxData::EntityModel *entity);

    void registerAttributeAccess(const agxData::Attribute *attribute, agx::UInt8 mode);

    void bind(agxData::EntityPtr ptr);

    /**
    \return The entity model
    */
    agxData::EntityModel *getEntityModel();

    /**
    \return The current binding.
    */
    agxData::EntityPtr getBinding();

    template <typename T>
    T getBinding();


    virtual bool isValid() const override;

    virtual void update() override;
    virtual void commit() override;

  protected:
    virtual ~EntityPtrParameter();
    virtual void _bind(Object *binding) override;

  private:
    EntityPtrParameter& operator=(EntityPtrParameter&) = delete;

  private:
    agxData::EntityModelRef m_entityModel;
    agxData::EntityPtr m_ptr;

    struct AttributeAccess
    {
      AttributeAccess() : attribute(nullptr), buffer(nullptr), mode(agx::InvalidIndex) {}
      AttributeAccess(const agxData::Attribute *_attribute, agx::UInt8 _mode) : attribute(_attribute), buffer(nullptr), mode(_mode) {}

      const agxData::Attribute *attribute;
      agxData::Buffer *buffer;
      agx::UInt8 mode;
    };

    VectorPOD<AttributeAccess> m_attributeAccessVector;
  };


  /**
  EntityInstance parameter.
  */
  class AGXCORE_EXPORT EntityInstanceParameter : public Parameter
  {
  public:
    static agx::Model *ClassModel();

    static EntityInstanceParameter *load(TiXmlElement *eParameter, Device *device);
    virtual void configure(TiXmlElement *eParameter) override;

  public:
    EntityInstanceParameter(const agx::Name& name, agxData::EntityModel *entity);

    void registerAttributeAccess(const agxData::Attribute *attribute, agx::UInt8 mode);

    void bind(agxData::EntityInstance instance);

    /**
    \return The entity model
    */
    agxData::EntityModel *getEntityModel();

    /**
    \return The current binding.
    */
    agxData::EntityInstance getBinding();

    template <typename T>
    T getBinding();


    virtual bool isValid() const override;

    virtual void update() override;
    virtual void commit() override;

  protected:
    virtual ~EntityInstanceParameter();
    virtual void _bind(Object *binding) override;

  private:
    EntityInstanceParameter& operator=(const EntityInstanceParameter&) = delete;

  private:
    agxData::EntityModelRef m_entityModel;
    agxData::EntityInstance m_instance;

    struct AttributeAccess
    {
      AttributeAccess() : attribute(nullptr), buffer(nullptr), mode(agx::InvalidIndex) {}
      AttributeAccess(const agxData::Attribute *_attribute, agx::UInt8 _mode) : attribute(_attribute), buffer(nullptr), mode(_mode) {}

      const agxData::Attribute *attribute;
      agxData::Buffer *buffer;
      agx::UInt8 mode;
    };

    VectorPOD<AttributeAccess> m_attributeAccessVector;
  };


  //---------------------------------------------------------------


  /* Implementation */

  AGX_FORCE_INLINE Task *Parameter::getTask() { return m_task; }
  AGX_FORCE_INLINE const Task *Parameter::getTask() const { return m_task; }
  AGX_FORCE_INLINE size_t Parameter::getIndex() const { return m_index; }
  AGX_FORCE_INLINE Parameter::MetaType Parameter::getMetaType() const { return (MetaType)m_metaType; }
  AGX_FORCE_INLINE bool Parameter::isRequired() const { return m_required; }
  AGX_FORCE_INLINE agxData::Format *Parameter::getFormat() { return m_format; }
  AGX_FORCE_INLINE const agxData::Format *Parameter::getFormat() const { return m_format; }
  AGX_FORCE_INLINE UInt8 Parameter::getAccessMode() const { return m_accessMode; }
  AGX_FORCE_INLINE const Path& Parameter::getBindPath() const { return m_bindPath; }
  AGX_FORCE_INLINE Object *Parameter::getBinding() { return m_binding; }
  AGX_FORCE_INLINE const Object *Parameter::getBinding() const { return m_binding; }
  AGX_FORCE_INLINE agxData::Type *Parameter::getType() { return m_format ? m_format->getType() : nullptr; }
  AGX_FORCE_INLINE const agxData::Type *Parameter::getType() const { return m_format ? m_format->getType() : nullptr; }
  AGX_FORCE_INLINE bool Parameter::needUpdate() { return m_needUpdate; }
  AGX_FORCE_INLINE bool Parameter::needCommit() { return m_needCommit; }

  ////////////////////////////////////////////////////////////////////////////////////////

  AGX_FORCE_INLINE agxData::Value *ScalarParameter::getValue() { return m_value; }
  AGX_FORCE_INLINE const agxData::Value *ScalarParameter::getValue() const { return m_value; }

  template <typename T>
  AGX_FORCE_INLINE const T& ScalarParameter::get() const
  {
    agxAssertN(agxData::getFormat<T>() == m_format, "%s: Type mismatch, parameter is type %s, dereferencing using type %s", this->getPath().c_str(), m_format->fullName().c_str(), agxData::getFormat<T>()->fullName().c_str());
    agxAssertN(m_value && m_value->isValid(), "%s: Can not dereference an unbound parameter", this->getPath().c_str());
    return m_value->get<T>();
  }

  template <typename T>
  AGX_FORCE_INLINE void ScalarParameter::set(const T& value)
  {
    if (m_binding.get() != m_value.get())
    {
      this->unbind();
      m_binding = m_value;
    }
    m_value->set(value);
  }

  template <typename T>
#ifdef AGX_DEBUG
AGX_FORCE_INLINE void ScalarParameter::write(const T& value, const Task* owner)
#else
AGX_FORCE_INLINE void ScalarParameter::write(const T& value, const Task* /*owner*/)
#endif
  {
    agxAssert(owner == this->getTask());
    agxAssert(m_format->is(agxData::getFormat<T>()));
    *(T*)this->ptr() = value;
  }

  AGX_FORCE_INLINE void *ScalarParameter::ptr()
  {
    return const_cast<void *>(m_value->ptr());
  }


  AGX_FORCE_INLINE void ScalarParameter::setReal( Real value )
  {
    this->set( value );
  }

  AGX_FORCE_INLINE Real ScalarParameter::getReal()
  {
    return this->get<Real>();
  }

  ////////////////////////////////////////////////////////////////////////////////////////

  AGX_FORCE_INLINE agxData::Buffer *ArrayParameter::getBuffer() { return m_buffer; }
  AGX_FORCE_INLINE const agxData::Buffer *ArrayParameter::getBuffer() const { return m_buffer; }
  AGX_FORCE_INLINE agxData::Type *ArrayParameter::getType() { return m_type; }
  AGX_FORCE_INLINE const agxData::Type *ArrayParameter::getType() const { return m_type; }

  template <typename T>
  AGX_FORCE_INLINE agxData::Array<T>& ArrayParameter::get()
  {
    agxAssertN(!m_required || m_buffer, "%s: Can not dereference an unbound array parameter", this->getPath().c_str());
    agxAssertN(agxData::getFormat<T>() == m_format, "%s: Type mismatch, parameter is type %s, dereferencing using type %s", this->getPath().c_str(), m_format->fullName().c_str(), agxData::getFormat<T>()->fullName().c_str());

    return m_buffer ? m_buffer->getArray<T>() : static_cast<agxData::Array<T>&>(s_dummyArray);
  }

  template <typename T>
  AGX_FORCE_INLINE const agxData::Array<T>& ArrayParameter::get() const { return const_cast<ArrayParameter *>(this)->get<T>(); }

  AGX_FORCE_INLINE size_t ArrayParameter::getLocalSize() const { return m_localMemorySize; }

  AGX_FORCE_INLINE size_t ArrayParameter::getNumElements() const
  {
    agxAssert(this->getBuffer());
    return this->getBuffer()->size() / this->getBuffer()->getElementArraySize();
  }


  AGX_FORCE_INLINE GlobalResultBuffer *ArrayParameter::getGlobalResult() { return m_globalResult; }
  AGX_FORCE_INLINE const GlobalResultBuffer *ArrayParameter::getGlobalResult() const { return m_globalResult; }

  AGX_FORCE_INLINE UInt ArrayParameter::getGlobalResultMode() const { return m_globalResultMode; }

  ////////////////////////////////////////////////////////////////////////////////////////

  template <typename T>
  AGX_FORCE_INLINE T* PointerParameter::get()
  {
    agxAssertN(!this->isRequired() || dynamic_cast<T *>(m_binding.get()), "%s: Pointer parameter does not have a binding!", getPath().c_str());
    return static_cast<T *>(m_binding.get());
  }

  AGX_FORCE_INLINE Object* PointerParameter::get() { return m_binding; }
  AGX_FORCE_INLINE const Object* PointerParameter::get() const { return m_binding; }

  template <typename T>
  AGX_FORCE_INLINE const T* PointerParameter::get() const { return const_cast<PointerParameter *>(this)->get<T>(); }


  ////////////////////////////////////////////////////////////////////////////////////////

  AGX_FORCE_INLINE agxData::EntityModel *EntityDataParameter::getEntityModel() { return m_entityModel; }
  AGX_FORCE_INLINE agxData::EntityStorage *EntityDataParameter::getStorage() { return m_storage; }
  AGX_FORCE_INLINE agxData::EntityData *EntityDataParameter::getData() { return m_dataSet; }

  template <typename T>
  AGX_FORCE_INLINE T& EntityDataParameter::getData() { agxAssert(m_dataSet); return *static_cast<T *>(m_dataSet.get()); }

  AGX_FORCE_INLINE const ParameterRefVector& EntityDataParameter::getChildParameters() const { return m_children; }


  AGX_FORCE_INLINE GlobalResultStorage *EntityDataParameter::getGlobalResult() { return m_globalResult; }
  AGX_FORCE_INLINE const GlobalResultStorage *EntityDataParameter::getGlobalResult() const { return m_globalResult; }

  AGX_FORCE_INLINE UInt EntityDataParameter::getGlobalResultMode() const { return m_globalResultMode; }

  ////////////////////////////////////////////////////////////////////////////////////////

  AGX_FORCE_INLINE agxData::EntityModel *EntityPtrParameter::getEntityModel() { return m_entityModel; }
  inline agxData::EntityPtr EntityPtrParameter::getBinding() { return m_ptr; }

  template <typename T>
  AGX_FORCE_INLINE T EntityPtrParameter::getBinding() { return T(m_ptr); }

  ////////////////////////////////////////////////////////////////////////////////////////

  inline agxData::EntityModel *EntityInstanceParameter::getEntityModel() { return m_entityModel; }
  inline agxData::EntityInstance EntityInstanceParameter::getBinding() { return m_instance; }

  template <typename T>
  AGX_FORCE_INLINE T EntityInstanceParameter::getBinding() { return T(m_instance); }

}


#endif
