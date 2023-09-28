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

#ifndef AGXDATA_BUFFER_H
#define AGXDATA_BUFFER_H

#include <agx/agx.h>
#include <agx/Component.h>
#include <agx/Event.h>

#include <agxData/Array.h>
#include <agxData/Value.h>
#include <agxData/ShareHandle.h>

#include <agx/Vector.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 6011) // Disable warningC6011: dereferencing nullptr pointer
#endif


namespace agx
{
  typedef VectorPOD<Index>   IndexVector;
  class GlobalResultBuffer;
}

namespace agxData
{
  class Attribute;
  class EntityStorage;

  AGX_DECLARE_POINTER_TYPES(Buffer);
  AGX_DECLARE_VECTOR_TYPES(Buffer);

  /**
  Abstract representation of a data buffer. The buffer has a name and
  data format/type. The buffer is an agx::Object and can be inserted
  in the system hierarchy.
  */
  class AGXCORE_EXPORT Buffer : public agx::Component
  {
  public:
    static agx::Model *ClassModel();

    static Buffer *load(agx::TiXmlElement *eBuffer, agx::Device *device);
    virtual void configure(agx::TiXmlElement *eBuffer) override;
    virtual void snapshot(agx::TiXmlNode *eParent, const agx::String& directory) const override;

  public:
    static agx::Real CAPACITY_MULTIPLIER;
    static agx::UInt MIN_BUFFER_CAPACITY;
    static agx::UInt ALIGNMENT;

    // Generated event interface, defined in Buffer.agxEvent
    #include <agx/BufferEvent.Declaration.h>

#ifndef SWIG
    void addListener(EventListener *listener);
    void removeListener(EventListener *listener);
    bool hasListener(EventListener *listener);
#endif

    agxData::ValueRefT<agx::UInt> sizeParameter;
    agxData::ValueRefT<agx::UInt> capacityParameter;


    typedef agx::Event1<Buffer *> Event;

    /// Triggered when buffer is reallocated
    Event reallocationEvent;

  public:
    Buffer(const agx::Name& name, const agxData::Format *format);
    Buffer(const agxData::Format *format);

    /**
    \return The type of the buffer.
    */
    agxData::Type *getType();
    const agxData::Type *getType() const;

    /**
    \return The format of the buffer.
    */
    agxData::Format *getFormat();
    const agxData::Format *getFormat() const;


    /** Convenience method to get an direct data array */
    template <typename T>
    agxData::Array<T>& getArray();

    template <typename T>
    const agxData::Array<T>& getArray() const;


    agxData::AbstractArray& getArray();
    const agxData::AbstractArray& getArray() const;

    /**
    \return The size of the buffer.
    */
    size_t size() const;

    /**
    \return The capacity of the buffer.
    */
    size_t capacity() const;

    /**
    \return True if the buffer is empty.
    */
    bool empty() const;

    /**
    Resize the buffer.
    */
    void resize(size_t size, bool initializeElements = true);
    void presize(size_t size, size_t initialSize); // Resize without generating events, use with caution

    /**
    Clear the buffer.
    */
    void clear();

    /**
    Reserve space for future buffer expansion.
    */
    void reserve(size_t size);

    /**
    \return The revision counter
    */
    agx::UInt64 getRevision() const;

    /**
    Get access to the raw data buffer.
    */
    void* ptr();
    const void* ptr() const;

    template <typename T>
    T *ptr();

    template <typename T>
    const T *ptr() const;

    /**
    Initialize a range of elements.
    */
    void initialize(const agx::IndexRange& range);

    /**
    \return A specific element in the buffer.
    */
    template <typename T>
    T& getElement(size_t index);

    template <typename T>
    const T& getElement(size_t index) const;


    void commit();
    void commit(agx::IndexRange32 range);
    void commit(agx::Index index);
    void commit(const agxData::IndexArray& indexList);


    /**
    \return The raw data of a specific element in the buffer.
    */
    void *getElement(size_t index);
    const void *getElement(size_t index) const;

    void *kernelOverflowReallocation(size_t size);
    void deleteOverflowBuffer(void *buffer, size_t numElements);

    void *allocateBackBuffer();
    void swapBackBuffer(void *backBuffer);

    /**
    \return The array size. (used for ArrayAttribute in an EntityStorage)
    */
    size_t getElementArraySize() const;


    /**
    Clone the buffer.
    */
    Buffer *clone() const;

    /**
    Clone a subset of the buffer.
    */
    Buffer *clone(const agx::IndexRange& range) const;
    Buffer *clone(const agxData::IndexArray& indices) const;
    Buffer *clone(const agx::IndexVector& indices) const;

    /**
    Copy one buffer to another.
    */
    static void copy(Buffer *target, const Buffer *source);
    static void copy(Buffer *target, size_t targetOffset, const Buffer *source, size_t sourceOffset, size_t numElements);
    static void copyGather(Buffer *target, const Buffer *source, const agxData::IndexArray& indices);
    static void copyScatter(Buffer *target, const Buffer *source, const agxData::IndexArray& indices);


    /**
    Print the buffer.
    */
    void print() const;
    void print(agx::IndexRange range) const;

    void print(std::ostream& stream) const;
    void print(std::ostream& stream, agx::IndexRange range) const;

    /**
    \return The storage which the buffer is part of.
    */
    agxData::EntityStorage *getStorage();
    const agxData::EntityStorage *getStorage() const;

    /**
    \return The attribute which the buffer holds data, nullptr if not part of a storage.
    */
    agxData::Attribute *getAttribute();
    const agxData::Attribute *getAttribute() const;

    /**
    \return The global result bound to this buffer.
    */
    agx::GlobalResultBuffer *getGlobalResult();
    const agx::GlobalResultBuffer *getGlobalResult() const;


    /**
    If the buffer is used as a global result, we provide a permutation buffer for parallel determinism.
    */
    Buffer *getGlobalOrderBuffer();
    const Buffer *getGlobalOrderBuffer() const;

    Buffer *getOrCreateGlobalOrderBuffer();


    /**
    If the buffer contains arrays/vectors, the actual elements are stored in another buffer.
    */
    Buffer *getElementBuffer();
    const Buffer *getElementBuffer() const;

    /**
    Set the element buffer, internal use only.
    */
    void setElementBuffer(Buffer *buffer);

    /**
    Set the default element value.
    */
    void setDefaultValue(agxData::Value *value);

    /**
    \return The default element value.
    */
    agxData::Value *getDefaultValue();
    const agxData::Value *getDefaultValue() const;

    /**
    Specify a custom element initializer callback.
    */
    typedef agx::Callback2<Buffer *, const agx::IndexRange&> RangeCallback;
    void setElementInitializer(const RangeCallback& callback);
    void setElementDestructor(const RangeCallback& callback);

    Buffer *getOriginalBuffer();
    const Buffer *getOriginalBuffer() const;

    const agxData::Value::Event::CallbackType& getResizeCallback() const;
    const agxData::Value::Event::CallbackType& getReserveCallback() const;

    virtual void buildNavigationTree(agxJson::Value& eNode) const override;

  public:
    void registerSharedContainer(agx::Container *container);
    void unregisterSharedContainer(agx::Container *container);

    void setRevision(agx::UInt64 revision);

  protected:
    virtual ~Buffer();

    virtual void reallocate(size_t size);
    static void initializeElements(Buffer *buffer, const agx::IndexRange& range);
    static void destroyElements(Buffer *buffer, const agx::IndexRange& range);

  private:
    friend class agx::GlobalResultBuffer;
    void setGlobalResult(agx::GlobalResultBuffer *globalResult);
    void signalReallocation();

  private:
    friend class EntityStorage;
    friend class BufferProxyAllocator;
    void keyFrameReserve(size_t capacity);

    void init();
    void resizeCallback(agxData::Value *value);
    void reserveCallback(agxData::Value *value);
    void readBinaryDump(const agx::String& path);
    void writeBinaryDump(const agx::String& path);
    void connect(EntityStorage *storage, Attribute *attribute);
    void setElementArraySize(const agxData::Value *value);


    agxData::ValueRefT<agx::UInt> m_elementArraySize;

  private:
    friend class agx::Device;
    void setOriginalBuffer(Buffer *buffer);

  protected:
    size_t m_size;
    size_t m_capacity;
    agx::UInt64 m_revision;

    void *m_ptr;
    agx::ByteAllocator m_allocator;
    AbstractArray m_array;
    bool m_initElements;
    bool m_usedResizeMethod;
    EntityStorage *m_storage;
    Attribute *m_attribute;
    FormatRef m_format;
    agxData::ValueRef m_defaultElementValue;
    agxData::BufferRef m_elementBuffer;
    agxData::BufferRef m_globalOrderBuffer;
    agx::Container *m_sharedContainer;
    agxData::Value::Event::CallbackType m_resizeCallback;
    agxData::Value::Event::CallbackType m_reserveCallback;
    agx::GlobalResultBuffer *m_globalResult;
    EventDispatch m_eventDispatch;

    RangeCallback m_initializer;
    RangeCallback m_destructor;

    friend class ShareHandle;
    agxData::ShareHandleRef m_shareHandle;
    agxData::Buffer *m_originalBuffer;

    template <typename T>
    friend class Vector;
  };

  // Container allocator for Vector, HashTable, etc.
  class AGXCORE_EXPORT BufferProxyAllocator
  {
  public:
    BufferProxyAllocator(Buffer *buffer = nullptr);
    ~BufferProxyAllocator();

    void setBuffer(Buffer *buffer);

    void update() const;
    void commit();

    // Automatically called by container
    void setContainer(agx::Container *container);

    void *allocateBytes(size_t numBytes);
    void deallocateBytes(void *buffer, size_t numBytes);

    BufferProxyAllocator& operator=(const BufferProxyAllocator& other);
  private:
    BufferRef m_buffer;
    agx::Container *m_container;
    int m_reallocations;
  };


  // Templated buffer
  template <typename T>
  class BufferT : public Buffer
  {
  public:
    template <typename T2>
    static bool ValidateCast(const agx::Referenced *object);

  public:
    typedef T        Type;
    typedef T        value_type;
    typedef T*       iterator;
    typedef const T* const_iterator;

  public:
    BufferT();
    BufferT(const agx::Name& name);


    void push_back(const T& value);
    void pop_back();

    void clear();

    template <typename T2>
    bool contains(const T2& value) const;

    template <typename T2>
    size_t find(const T2& value) const;

    T& operator[] (size_t index);
    const T& operator[] (size_t index) const;

    T& at(size_t index);
    const T& at(size_t index) const;

    T& front();
    const T& front() const;

    T& back();
    const T& back() const;


    iterator begin();
    const_iterator begin() const;

    iterator end();
    const_iterator end() const;

    void commit();
    void commit(agx::IndexRange32 range);
    void commit(agx::Index index);
    void commit(const agxData::IndexArray& indexList);
    void commit(const agx::IndexVector& indexList);

  protected:
    virtual ~BufferT();

  private:
    Array<T>& array();
    const Array<T>& array() const;
  };


  /* Implementation */
  AGX_FORCE_INLINE size_t Buffer::size() const { return m_size; }
  AGX_FORCE_INLINE size_t Buffer::capacity() const { return m_capacity; }
  AGX_FORCE_INLINE bool Buffer::empty() const { return m_size == 0; }


  AGX_FORCE_INLINE agx::UInt64 Buffer::getRevision() const { return m_revision; }

  AGX_FORCE_INLINE size_t Buffer::getElementArraySize() const { return m_elementArraySize.get() ? m_elementArraySize->get() : 1; }

  AGX_FORCE_INLINE EntityStorage *Buffer::getStorage() { return m_storage; }
  AGX_FORCE_INLINE const EntityStorage *Buffer::getStorage() const { return m_storage; }

  AGX_FORCE_INLINE Attribute *Buffer::getAttribute() { return m_attribute; }
  AGX_FORCE_INLINE const Attribute *Buffer::getAttribute() const { return m_attribute; }

  AGX_FORCE_INLINE Type *Buffer::getType() { return m_format->getType(); }
  AGX_FORCE_INLINE const Type *Buffer::getType() const { return m_format->getType(); }
  AGX_FORCE_INLINE Format *Buffer::getFormat() { return m_format; }
  AGX_FORCE_INLINE const Format *Buffer::getFormat() const { return m_format; }

  AGX_FORCE_INLINE agxData::Value *Buffer::getDefaultValue() { return m_defaultElementValue; }
  AGX_FORCE_INLINE const agxData::Value *Buffer::getDefaultValue() const { return m_defaultElementValue; }

  template <typename T>
  AGX_FORCE_INLINE Array<T>& Buffer::getArray()
  {
    agxAssertN(m_format->is(agxData::getFormat<T>()) && m_format->getSize() == agxData::getFormat<T>()->getSize(), "Type mismatch! Buffer has type %s, requested array of type %s", m_format->fullName().c_str(), agxData::getFormat<T>()->fullName().c_str());
    return reinterpret_cast<agxData::Array<T>&>(m_array);
  }

  template <typename T>
  AGX_FORCE_INLINE const Array<T>& Buffer::getArray() const
  {
    return const_cast<Buffer *>(this)->getArray<T>();
  }

  AGX_FORCE_INLINE AbstractArray& Buffer::getArray() { return m_array; }
  AGX_FORCE_INLINE const AbstractArray& Buffer::getArray() const { return m_array; }


  AGX_FORCE_INLINE void *Buffer::ptr() { return m_ptr; }
  AGX_FORCE_INLINE const void *Buffer::ptr() const { return m_ptr; }

  template <typename T>
  AGX_FORCE_INLINE T *Buffer::ptr()
  {
    agxAssertN(agxCore::isShutdown() || this->getFormat()->is(agxData::getFormat<T>()), "%s not the same as %s", agxData::getFormat<T>()->fullName().c_str(), this->getFormat()->fullName().c_str());
    return static_cast<T *>(m_ptr);
  }

  template <typename T>
  AGX_FORCE_INLINE const T *Buffer::ptr() const
  {
    return const_cast<Buffer *>(this)->ptr<T>();
  }


  template <typename T>
  AGX_FORCE_INLINE T& Buffer::getElement(size_t index)
  {
    agxAssert(m_ptr);
    agxAssert(index < this->capacity());
    agxAssertN(agxCore::isShutdown() || this->getFormat()->is(agxData::getFormat<T>()), "%s not the same as %s", agxData::getFormat<T>()->fullName().c_str(), this->getFormat()->fullName().c_str());
    agxAssert(this->getFormat()->getSize() == sizeof(T));

    T *tmp = (T *)m_ptr;
    return tmp[index];
  }

  template <typename T>
  AGX_FORCE_INLINE const T& Buffer::getElement(size_t index) const
  {
    return const_cast<Buffer *>(this)->getElement<T>(index);
  }


  AGX_FORCE_INLINE void *Buffer::getElement(size_t index)
  {
    agxAssert(m_ptr);
    agxAssert(index < this->capacity());
    return (void *)((char *)m_ptr + index * this->getFormat()->getSize());
  }

  AGX_FORCE_INLINE const void *Buffer::getElement(size_t index) const
  {
    return const_cast<Buffer *>(this)->getElement(index);
  }


  AGX_FORCE_INLINE agx::GlobalResultBuffer *Buffer::getGlobalResult() { return m_globalResult; }
  AGX_FORCE_INLINE const agx::GlobalResultBuffer *Buffer::getGlobalResult() const { return m_globalResult; }

  AGX_FORCE_INLINE void Buffer::setGlobalResult(agx::GlobalResultBuffer *globalResult) { m_globalResult = globalResult; }


  AGX_FORCE_INLINE Buffer *Buffer::getElementBuffer() { return m_elementBuffer; }
  AGX_FORCE_INLINE const Buffer *Buffer::getElementBuffer() const { return m_elementBuffer; }

  AGX_FORCE_INLINE Buffer *Buffer::getGlobalOrderBuffer() { return m_globalOrderBuffer; }
  AGX_FORCE_INLINE const Buffer *Buffer::getGlobalOrderBuffer() const { return m_globalOrderBuffer; }

  //--------------------------------------------


  template <typename T>
  AGX_FORCE_INLINE agx::GlobalResult *Array<T>::getGlobalResult()
  {
    agxAssert(m_buffer);
    agxAssert(m_buffer->getGlobalResult());
    return m_buffer->getGlobalResult();
  }

  template <typename T>
  AGX_FORCE_INLINE agx::GlobalResult::Transaction Array<T>::allocateResult(size_t numElements)
  {
    return this->getGlobalResult()->allocateResult(numElements);
  }

  template <typename T>
  AGX_FORCE_INLINE size_t Array<T>::commitResult(size_t numElements, const void *localResult, agx::Index sortIndex)
  {
    if (numElements == 0)
      return agx::InvalidIndex;

    return this->getGlobalResult()->commit(*this, numElements, localResult, sortIndex);
  }

  template <typename T>
  AGX_FORCE_INLINE size_t Array<T>::commitResult(size_t numElements, const void *localResult)
  {
    if (numElements == 0)
      return agx::InvalidIndex;

    return this->getGlobalResult()->commit(*this, numElements, localResult);
  }

  template <typename T> template <typename T2>
  AGX_FORCE_INLINE size_t Array<T>::commitResult(const T2& vector, agx::Index sortIndex)
  {
    if (vector.empty())
      return agx::InvalidIndex;

    return this->getGlobalResult()->commit(*this, vector, sortIndex);
  }


  template <typename T> template <typename T2>
  AGX_FORCE_INLINE size_t Array<T>::commitResult(const T2& vector)
  {
    if (vector.empty())
      return agx::InvalidIndex;

    return this->getGlobalResult()->commit(*this, vector);
  }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE AbstractArray& Data::asArray() { return *static_cast<AbstractArray *>(this); }
  AGX_FORCE_INLINE const AbstractArray& Data::asArray() const { return *static_cast<const AbstractArray *>(this); }

  template <typename T>
  AGX_FORCE_INLINE Array<T>& Data::asArray() { return *static_cast<Array<T> *>(this); }

  template <typename T>
  AGX_FORCE_INLINE const Array<T>& Data::asArray() const { return *static_cast<const Array<T> *>(this); }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE AbstractArray::AbstractArray() : m_buffer(nullptr)
  {}

  AGX_FORCE_INLINE AbstractArray::AbstractArray(Buffer *buffer)
  {
    if (buffer != nullptr)
    {
      this->init(buffer);
    }
    else
    {
      this->init();
    }
  }

  AGX_FORCE_INLINE AbstractArray::AbstractArray(Buffer *buffer, agx::IndexRange32 range)
  {
    this->init(buffer, range);
  }

  AGX_FORCE_INLINE AbstractArray::AbstractArray(const AbstractArray& other, agx::IndexRange32 localRange)
  {
    agxAssert(localRange.end() <= other.m_range.size());
    agx::IndexRange32 range(other.m_range.begin() + localRange.begin(), other.m_range.begin() + localRange.end());
    this->init(other.m_buffer, range);
  }

  AGX_FORCE_INLINE void AbstractArray::init()
  {
    m_ptr = nullptr;
    m_buffer = nullptr;
    m_range = agx::IndexRange32();
  }

  AGX_FORCE_INLINE void AbstractArray::init(Buffer* buffer, agx::IndexRange32 range)
  {
    agxAssert(buffer);
    // agxAssert(buffer->getType() == this->getType());
    agxAssert(range.end() <= std::max(buffer->size(), (size_t)buffer->getArray().range().end()));

    m_buffer = buffer;
    m_range = range;
    this->sync();
  }


  AGX_FORCE_INLINE void AbstractArray::init(Buffer* buffer)
  {
    agxAssert(buffer);
    this->init(buffer, agx::IndexRange32(0, agx::UInt32(buffer->size())));
  }

  AGX_FORCE_INLINE void AbstractArray::_setBuffer(agxData::Buffer *buffer)
  {
    agxAssert(buffer);
    m_buffer = buffer;
    this->sync();
  }


  AGX_FORCE_INLINE void AbstractArray::setRange(agx::IndexRange32 range)
  {
    agxAssert(m_buffer);
    agxAssert(range.end() <= std::max(m_buffer->size(), (size_t)m_buffer->getArray().range().end()));
    m_range = range;
    this->sync();
  }


  AGX_FORCE_INLINE void AbstractArray::setRange(size_t startIndex, size_t endIndex)
  {
    this->setRange(agx::IndexRange32(agx::UInt32(startIndex), agx::UInt32(endIndex)));
  }


  AGX_FORCE_INLINE void AbstractArray::sync()
  {
    agxAssert(m_buffer);
    m_ptr = ((char*)m_buffer->ptr()) + m_range.begin() * m_buffer->getFormat()->getSize();
  }

  AGX_FORCE_INLINE size_t AbstractArray::size() const
  {
    return m_range.size();
  }

  AGX_FORCE_INLINE bool AbstractArray::empty() const
  {
    return m_range.size() == 0;
  }


  AGX_FORCE_INLINE Buffer* AbstractArray::buffer() { return m_buffer; }
  AGX_FORCE_INLINE const Buffer* AbstractArray::buffer() const { return m_buffer; }
  AGX_FORCE_INLINE agx::IndexRange32& AbstractArray::range() { return m_range; }
  AGX_FORCE_INLINE const agx::IndexRange32& AbstractArray::range() const { return m_range; }

  AGX_FORCE_INLINE bool AbstractArray::isValid() const { return m_buffer != nullptr; }

  AGX_FORCE_INLINE Format* AbstractArray::getFormat() { return m_buffer != nullptr ? m_buffer->getFormat() : nullptr; }
  AGX_FORCE_INLINE const Format* AbstractArray::getFormat() const { return m_buffer != nullptr ? m_buffer->getFormat() : nullptr; }


  //---------------------------------------------------------------

  template <typename T>
  AGX_FORCE_INLINE Array<T> Array<T>::raw(T* buffer, agx::Index size)
  {
    Array<T> result;
    #include <agx/PushDisableWarnings.h>
    result.m_ptr = buffer;
    result.m_range = agx::IndexRange32(0, size);
    #include <agx/PopDisableWarnings.h>
    return result;
  }

  template <typename T>
  AGX_FORCE_INLINE const Array<T> Array<T>::raw(const T* buffer, agx::Index size)
  {
    Array<T> result;
    #include <agx/PushDisableWarnings.h>
    result.m_ptr = (T *)buffer;
    result.m_range = agx::IndexRange32(0, size);
    #include <agx/PopDisableWarnings.h>
    return result;
  }

  template <typename T>
  AGX_FORCE_INLINE Array<T>::Array() : AbstractArray()
  {
  }

  template <typename T>
  AGX_FORCE_INLINE Array<T>::Array(Buffer* buffer) : AbstractArray(buffer)
  {
    #ifndef AGX_ENTITY_WRAPPER
    agxAssert( agxCore::isShutdown() || (buffer->getFormat()->is(agxData::getFormat<T>()) && buffer->getFormat()->getSize() == agxData::getFormat<T>()->getSize()));
    #endif
  }


  template <typename T>
  AGX_FORCE_INLINE Array<T>::Array(Buffer* buffer, agx::IndexRange32 range) : AbstractArray(buffer, range)
  {
    #ifndef AGX_ENTITY_WRAPPER
    agxAssert( agxCore::isShutdown() || (buffer->getFormat()->is(agxData::getFormat<T>()) && buffer->getFormat()->getSize() == agxData::getFormat<T>()->getSize()));
    #endif
  }

  template <typename T>
  AGX_FORCE_INLINE Array<T>::Array(const Array<T>& other, agx::IndexRange32 localRange) : AbstractArray(other, localRange)
  {
  }

  template <typename T>
  AGX_FORCE_INLINE Array<T>::Array(agx::Vector<T>& vec)
  {
    this->m_ptr = vec.ptr();
    this->m_range = agx::IndexRange32(0, (agx::Index)vec.size());
  }

  template <typename T>
  AGX_FORCE_INLINE Array<T>::Array(agx::Vector<T>& vec, agx::IndexRange32 range)
  {
    agxAssert(range.begin() < range.end() && range.end() <= vec.size());
    this->m_ptr = vec.ptr();
    this->m_range = range;
  }

  template <typename T>
  AGX_FORCE_INLINE Array<T>::Array(agx::VectorPOD<T>& vec)
  {
    this->m_ptr = vec.ptr();
    this->m_range = agx::IndexRange32(0, (agx::Index)vec.size());
  }

  template <typename T>
  AGX_FORCE_INLINE Array<T>::Array(agx::VectorPOD<T>& vec, agx::IndexRange32 range)
  {
    agxAssert(range.begin() < range.end() && range.end() <= vec.size());
    this->m_ptr = vec.ptr();
    this->m_range = range;
  }

  template <typename T>
  AGX_FORCE_INLINE const T& Array<T>::back() const
  {
    agxAssert(m_ptr);
    agxAssert1(!m_range.empty(), "Can not take back element of an empty array!");
    return ((T*)m_ptr)[m_range.size()-1];
  }

  template <typename T>
  AGX_FORCE_INLINE T& Array<T>::back()
  {
    agxAssert(m_ptr);
    agxAssert1(!m_range.empty(), "Can not take back element of an empty array!");
    return ((T*)m_ptr)[m_range.size()-1];
  }

  template <typename T>
  AGX_FORCE_INLINE const T& Array<T>::front() const
  {
    agxAssert(m_ptr);
    agxAssert1(!m_range.empty(), "Can not take front element of an empty array!");
    return ((T*)m_ptr)[0];
  }

  template <typename T>
  AGX_FORCE_INLINE T& Array<T>::front()
  {
    agxAssert(m_ptr);
    agxAssert1(!m_range.empty(), "Can not take front element of an empty array!");
    return ((T*)m_ptr)[0];
  }


  template <typename T>
  AGX_FORCE_INLINE T& Array<T>::operator[] (size_t index)
  {
    agxAssert1(m_ptr, "Array does not have an active buffer!");
    agxAssertN(index < m_range.size(), "Array index %llu is out of bounds, size is %llu", (long long unsigned)index, (long long unsigned)m_range.size());
    return ((T*)m_ptr)[index];
  }


  template <typename T>
  AGX_FORCE_INLINE const T& Array<T>::operator[] (size_t index) const
  {
    return const_cast<Array<T>*>(this)->operator[](index);
  }

  template <typename T>
  AGX_FORCE_INLINE T& Array<T>::at(size_t index)
  {
    agxAssert(m_ptr);
    agxVerifyN(index < m_range.size(), "Array index %llu is out of bounds, size is %llu", (long long unsigned)index, (long long unsigned)m_range.size());
    return ((T*)m_ptr)[index];
  }

  template <typename T>
  AGX_FORCE_INLINE const T& Array<T>::at(size_t index) const
  {
    return const_cast<Array<T>*>(this)->at(index);
  }


  template <typename T>
  AGX_FORCE_INLINE Array<T> Array<T>::slice(agx::IndexRange32 subRange) const
  {
    return Array<T>(m_buffer, agx::IndexRange32(m_range.begin() + subRange.begin(), m_range.begin() + subRange.end()));
  }


  template <typename T>
  AGX_FORCE_INLINE Array<T> Array<T>::operator[] (agx::IndexRange32 subRange) const
  {
    return this->slice(subRange);
  }

  template <typename T>
  AGX_FORCE_INLINE typename Array<T>::iterator Array<T>::begin() { return ptr(); }

  template <typename T>
  AGX_FORCE_INLINE typename Array<T>::const_iterator Array<T>::begin() const { return ptr(); }

  template <typename T>
  AGX_FORCE_INLINE typename Array<T>::iterator Array<T>::end() { return ptr() + size(); }

  template <typename T>
  AGX_FORCE_INLINE typename Array<T>::const_iterator Array<T>::end() const { return ptr() + size(); }

  template <typename T>
  AGX_FORCE_INLINE T* Array<T>::ptr() { return (T*)m_ptr; }


  template <typename T>
  AGX_FORCE_INLINE const T* Array<T>::ptr() const { return (const T*)m_ptr; }


  template <typename T> template <typename T2>
  AGX_FORCE_INLINE size_t Array<T>::find(const T2& element) const
  {
    for (size_t i = 0, numElements = size(); i < numElements; ++i)
    {
      if ((*this)[i] == element)
        return i;
    }

    return agx::InvalidIndex;
  }

  template <typename T> template <typename T2>
  AGX_FORCE_INLINE bool Array<T>::contains(const T2& element) const
  {
    return this->find(element) != agx::InvalidIndex;
  }


  template <typename T>
  std::ostream& operator << ( std::ostream& output, const Array<T>& array)
  {
    for (size_t i = 0; i < array.size(); i++)
    {
      output << array[i];

      if (i < array.size()-1)
        output << ", ";
    }

    return output;
  }


  //--------------------------------------------


  template <typename T> template <typename T2>
  bool BufferT<T>::ValidateCast(const agx::Referenced *object)
  {
    const Buffer *buffer = dynamic_cast<const Buffer *>(object);
    return buffer ? buffer->getFormat() == agxData::getFormat<T>() : false;
  }

  template <typename T>
  BufferT<T>::BufferT() : Buffer(agxData::getFormat<T>())
  {
  }

  template <typename T>
  BufferT<T>::BufferT(const agx::Name& name) : Buffer(name, agxData::getFormat<T>())
  {
  }

  template <typename T>
  BufferT<T>::~BufferT()
  {
  }


  template <typename T>
  AGX_FORCE_INLINE Array<T>& BufferT<T>::array()
  {
    agxAssert(agxData::getFormat<T>() == this->getFormat());
    return reinterpret_cast<Array<T>&>(m_array);
  }

  template <typename T>
  AGX_FORCE_INLINE const Array<T>& BufferT<T>::array() const
  {
    return const_cast<BufferT<T> *>(this)->array();
  }



  template <typename T>
  AGX_FORCE_INLINE void BufferT<T>::push_back(const T& value)
  {
    if (m_size == m_capacity)
      this->reserve(std::max((size_t)4, (size_t)((agx::Real)m_size * Buffer::CAPACITY_MULTIPLIER)));

    m_size++;
    array().range().end()++;
    ::new((void *)&array().back()) T(value);
  }

  template <typename T>
  AGX_FORCE_INLINE void BufferT<T>::pop_back()
  {
    agxAssert(!array().empty());

    array().back().~T();
    array().range().end()--;
    m_size--;
  }

  template <typename T>
  AGX_FORCE_INLINE void BufferT<T>::clear()
  {
    for (size_t i = 0; i < m_size; ++i)
      array()[i].~T();

    m_size = 0;
    array().range().end() = 0;
  }

  template <typename T> template <typename T2>
  AGX_FORCE_INLINE bool BufferT<T>::contains(const T2& value) const
  {
    return array().contains(value);
  }

  template <typename T> template <typename T2>
  AGX_FORCE_INLINE size_t BufferT<T>::find(const T2& value) const
  {
    return array().find(value);
  }

  template <typename T>
  AGX_FORCE_INLINE T& BufferT<T>::operator[] (size_t index)
  {
    return array()[index];
  }

  template <typename T>
  AGX_FORCE_INLINE const T& BufferT<T>::operator[] (size_t index) const
  {
    return const_cast<BufferT<T> *>(this)->operator[](index);
  }

  template <typename T>
  AGX_FORCE_INLINE T& BufferT<T>::at(size_t index)
  {
    return array().at(index);
  }

  template <typename T>
  AGX_FORCE_INLINE const T& BufferT<T>::at(size_t index) const
  {
    return const_cast<BufferT<T> *>(this)->at(index);
  }

  template <typename T>
  AGX_FORCE_INLINE T& BufferT<T>::front()
  {
    return array().front();
  }

  template <typename T>
  AGX_FORCE_INLINE const T& BufferT<T>::front() const
  {
    return const_cast<BufferT<T> *>(this)->front();
  }

  template <typename T>
  AGX_FORCE_INLINE T& BufferT<T>::back()
  {
    return array().back();
  }

  template <typename T>
  AGX_FORCE_INLINE const T& BufferT<T>::back() const
  {
    return const_cast<BufferT<T> *>(this)->back();
  }



  template <typename T>
  AGX_FORCE_INLINE typename BufferT<T>::iterator BufferT<T>::begin()
  {
    return array().begin();
  }

  template <typename T>
  AGX_FORCE_INLINE typename BufferT<T>::const_iterator BufferT<T>::begin() const
  {
    return array().begin();
  }

  template <typename T>
  AGX_FORCE_INLINE typename BufferT<T>::iterator BufferT<T>::end()
  {
    return array().end();
  }

  template <typename T>
  AGX_FORCE_INLINE typename BufferT<T>::const_iterator BufferT<T>::end() const
  {
    return array().end();
  }

  template <typename T>
  AGX_FORCE_INLINE void BufferT<T>::commit()
  {
    Buffer::commit();
  }

  template <typename T>
  AGX_FORCE_INLINE void BufferT<T>::commit(agx::IndexRange32 range)
  {
    Buffer::commit(range);
  }

  template <typename T>
  AGX_FORCE_INLINE void BufferT<T>::commit(agx::Index index)
  {
    Buffer::commit(index);
  }

  template <typename T>
  AGX_FORCE_INLINE void BufferT<T>::commit(const agxData::IndexArray& indexList)
  {
    Buffer::commit(indexList);
  }

  template <typename T>
  AGX_FORCE_INLINE void BufferT<T>::commit(const agx::IndexVector& indexList)
  {
    Buffer::commit(agxData::IndexArray::raw(indexList.ptr(), indexList.size()));
  }


}

#include <agx/BufferEvent.Implementation.h>

#ifdef _MSC_VER
# pragma warning(pop)
#endif


#endif /* AGXDATA_BUFFER_H */
