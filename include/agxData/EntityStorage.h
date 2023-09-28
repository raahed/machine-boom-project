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

#include <agx/agx.h>
#include <agx/agxCore_export.h>
#include <agxData/Buffer.h>
#include <agxData/EntityModel.h>
#include <agxData/EntityData.h>
#include <agxData/AttributeContainer.h>
#include <agxData/LocalVector.h>
#include <agx/Vector.h>
#include <agx/HashTable.h>
#include <agx/Referenced.h>
#include <agx/Component.h>


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 6011) // Disable warningC6011: dereferencing nullptr pointer
#endif


namespace agx
{
  class Device;
  class TiXmlElement;
  class GlobalResultStorage;
  class AliasStorage;
}

namespace agxSDK
{
  class SimulationFrameWriter;
}

namespace agxStream
{
  class InputArchive;
  class OutputArchive;
}

namespace agxData
{
  typedef agx::HashTable<Attribute *, BufferRef> BufferTable;


  AGX_DECLARE_POINTER_TYPES(EntityStorage);
  AGX_DECLARE_VECTOR_TYPES(EntityStorage);

  /**
  Data storage for a collection of entity instances of a specified EntityModel. Each
  attribute is stored in a separate buffer. The storage is always packed, when a entity
  instance is destroyed it is overwritten by the last instance in the storage. The use
  of a id lookup table allow entity instances to use indirect addressing and remain valid
  when the data is permuted/reallocated.
  */
  class AGXCORE_EXPORT EntityStorage : public agx::Component, public AttributeContainer
  {
  public:
    static agx::Model *ClassModel();

    static agxData::EntityStorage *load(agx::TiXmlElement *eStorage, agx::Device *device);
    virtual void configure(agx::TiXmlElement *eStorage) override;

  public:
    // Generated event interface, defined in EntityStorage.agxEvent
    #include <agx/EntityStorageEvent.Declaration.h>

#ifndef SWIG
    void addListener(EventListener *listener);
    void removeListener(EventListener *listener);
    bool hasListener(EventListener *listener);
#endif
    agxData::ValueRefT<agx::UInt> sizeParameter;
    agxData::ValueRefT<agx::UInt> capacityParameter;

  public:
    EntityStorage(agxData::EntityModel *entity, size_t initialCapacity = 32);
    EntityStorage(agxData::EntityModel *entity, const agx::Name& name, size_t initialCapacity = 32);

    /**
    \return True if the storage is empty.
    */
    bool empty() const;

    /**
    \return The active size of the storage.
    */
    size_t size() const;

    /**
    \return The capacity of the storage.
    */
    size_t capacity() const;

    /**
    \return The number of allocated bytes.
    */
    size_t getNumBytes() const;

    /**
    \return The abstract entity for which the storage holds data.
    */
    agxData::EntityModel *getEntityModel();
    const agxData::EntityModel *getEntityModel() const;

    /**
    Reserve capacity in the storage.
    */
    void reserve(size_t size);

    /**
    \return An array of all active instances (only valid until storage is modified, e.g. by deleting an instance).
    */
    const agxData::Array<EntityPtr> getInstances() const;
    agxData::Array<EntityPtr> getInstances();

    template <typename T>
    const agxData::Array<T> getInstances() const;

    template <typename T>
    agxData::Array<T> getInstances();

    /**
    Creates a new instance and initializes it with default values.
    \return The instance
    */
    agxData::EntityPtr createInstance();

    /**
    Create a range of entity instances
    \return The range (only valid until storage is modified, e.g. by deleting an instance).
    */
    agxData::EntityRange createInstances(size_t numInstances);

    void resize(size_t size, bool initializeElements = true);

    size_t presize(size_t size, size_t initialSize = 0); // Resize without generating events, use with caution

    void destroyInstance(agxData::EntityPtr instance);
    void destroyInstance(size_t id);

    template <typename VectorT>
    void destroyInstances(const VectorT& instances);

    void destroyInstances(agxData::EntityRange range);


    typedef agx::VectorPOD< std::pair<size_t, size_t> > InstanceSwapVector;
    void swapInstances(const InstanceSwapVector &swaps);

    typedef agxData::LocalVector< std::pair<size_t, size_t> > InstanceSwapLocalVector;
    void swapInstances(const InstanceSwapLocalVector& swaps);

    // Must be followed by a matching resize...
    void preDestroy(agxData::EntityPtr instance);

    agxData::EntityPtr transferInstance(size_t id, agxData::EntityStorage* target);
    agxData::EntityPtr copyInstance(size_t id, agxData::EntityStorage* target);
    agxData::EntityPtr copyInstance(size_t id, agxData::EntityStorage* target, size_t targetId);

    agxData::EntityStorage *clone() const;
    static void copy(agxData::EntityStorage *target, const agxData::EntityStorage *source);

    /**
    Clears the storage.
    */
    void clear();

    /**
    Clear the storage, and reset index-id tables.
    */
    void reset();

    /**
    Swap buffers with another storage.
    */
    // void swap(EntityStorage *other);


    /**
    Create a shared storage with the same buffer instances.
    */
    agxData::EntityStorage *createSharedStorage();

    /**
    \return the buffer for a specific attribute.
    */
    agxData::Buffer *getBuffer(const agx::Name& name);
    agxData::Buffer *getBuffer(const agxData::Attribute *attribute);


    /**
    \return all registered proxy buffers.
    */
    const agxData::BufferTable& getBuffers() const;

    /// \return The idToIndex buffer
    agxData::Buffer *getIdToIndexBuffer();

    /// \return The idToIndex buffer
    const agxData::Buffer *getIdToIndexBuffer() const;

    /// \return The instance buffer
    agxData::Buffer *getInstanceBuffer();

    /// \return The instance buffer
    const agxData::Buffer *getInstanceBuffer() const;

    /**
    Convert between instance id and data index.
    */
    size_t idToIndex(size_t id) const;

    /**
    Convert between instance id and data index.
    */
    size_t indexToId(size_t index) const;


    /**
    \return The entity data accessor.
    */
    agxData::EntityData *getData();

    template <typename T>
    T* getData();

    template <typename T>
    const T* getData() const;

    /**
    Set to false to disable instance pointer handling. Saves memory for
    instance buffer and idToIndex table, but invalidates any usage of EntityPtr
    references to this storage.
    */
    void setUseInstanceTable(bool flag);

    /**
    \return true if instance pointers are active (default).
    */
    bool useInstanceTable() const;

    /**
    Add a custom attribute buffer.
    */
    void addCustomAttribute(agxData::Attribute *attribute);

    /**
    If the storage is used as a global result, we provide a permutation buffer for parallel determinism.
    */
    agxData::Buffer *getGlobalOrderBuffer();

    /**
    If the storage is used as a global result, we provide a permutation buffer for parallel determinism.
    */
    const agxData::Buffer *getGlobalOrderBuffer() const;

    /**
    Create the global order buffer if not already allocated.
    */
    agxData::Buffer *getOrCreateGlobalOrderBuffer();

    /**
    Reorder the storage using a specified permutation.
    */
    void reorder(const agxData::IndexArray& permutation);


    /**
    Reorder a subrange.
    */
    // void reorder(const IndexArray& permutation, agx::IndexRange range);


    void print(std::ostream& stream) const;
    void print() const;

    void verifyConsistency();


    /**
    Slow but convenient accessor.
    */
    template <typename T>
    T& get(const agx::Name& attributeName, size_t index);



    void preparePermutation();
    void commitPermutation(const agxData::IndexArray& permutation);
    void reorderAttributes(agx::IndexRange32 range, const agxData::IndexArray& permutation);

    agxData::Byte *getPermutationBuffer(agxData::Attribute *attribute);
    agxData::Byte *getPermutationBuffer(size_t index);


    void unregisterBuffer(agxData::Buffer *buffer);

    agxData::IndexArray& getIdToIndexTable();
    const agxData::IndexArray& getIdToIndexTable() const;


    // Moved here from EntityPtr because of include loop, see EntityPtr.h
    template <typename T>
    agxData::Array<T> dereferenceMultipliedAttribute(const agxData::EntityPtr& entity, const agxData::ScalarAttributeT<T> *attribute);

    template <typename T>
    const agxData::Array<T> dereferenceMultipliedAttribute(const agxData::EntityPtr& entity, const agxData::ScalarAttributeT<T> *attribute) const;


    static const agx::Name IdToIndexName;
    static const agx::Name IndexToIdName;

    // Internal
    void keyFrameResize(size_t size, size_t capacity);

    void setEnableEvents(bool flag);


    void storePermutation(agxStream::OutputArchive& archive) const;
    void restorePermutation(agxStream::InputArchive& archive, bool skip = false);
    static void extractPermutation(agx::IndexVector& result, agxStream::InputArchive& archive);
    void printInstanceTable() const;

  protected:
    EntityStorage(const agx::Name& name);
    EntityStorage(EntityModel *entity, const agx::Name& name, bool isFrameStorage, size_t capacity, size_t size);

    virtual ~EntityStorage();
    virtual agx::Object *getResourceImpl(const agx::Path& path, agx::Model *model) override;

    void explicityRegisterBuffer(agxData::Buffer *buffer, agxData::Attribute *attribute);

  private:
    friend class EntityPtr;
    void swapInstanceIds(agx::Index id1, agx::Index id2);

  private:
    friend class EntityData;
    void beginBatch();
    void commitBatch(size_t size);

  private:
    friend class Buffer;
    friend class EntityModel;
    friend class agx::GlobalResultStorage;
    friend class agxSDK::SimulationFrameWriter;
    void initializeInstances(const agx::IndexRange& range);
    void swapInstances(size_t index1, size_t index2);
    void init(size_t initialCapacity);


    // Buffer *tryGetBuffer(const Attribute *attribute, Device *device);
    // Buffer *tryGetBuffer(const String& name, Device* device);
    void resizeCallback(agxData::Value* value);
    void reserveCallback(agxData::Value* value);
    void elementBufferReallocationCallback(Buffer* elementBuffer);
    agxData::Buffer* createBuffer(const agxData::Attribute* attribute);
    void registerBuffer(agxData::Buffer *buffer);

    void sharedResizeCallback(agxData::Value* value);
    void sharedReserveCallback(agxData::Value* value);


    void destroyInstances(const EntityPtr* instances, size_t numInstances, bool triggerEvent = true);

    /**
    Access an attribute value for a specific entity instance.
    \param attribute The attribute.
    \param index The index of the instance.
    \return The value.
    */
    template <typename T>
    T& get(const ScalarAttributeT<T> *attribute, size_t index);



  private:

    size_t m_size;
    size_t m_capacity;
    bool m_useInstanceTable;
    bool m_implicitResize;
    bool m_initElements;
    bool m_usedResizeMethod;

    EntityModelRef m_entity;
    BufferTable m_bufferTable;
    BufferRefVector m_buffers;
    BufferRef m_instanceBuffer;
    EntityDataRef m_dataSet;

    typedef agx::HashTable<agx::Name, agxData::AttributeRef> AttributeTable;
    AttributeTable m_customAttributes;

    agx::VectorPOD<Byte *> m_permutationBuffers;

    BufferRef m_globalOrderBuffer;

    agx::Vector<agx::ref_ptr<agx::Referenced> > m_arrayAllocators;

    BufferRef m_idToIndexBuffer;
    IndexArray m_idToIndexArray;

    agxData::Value::Event::CallbackType m_resizeCallback;
    agxData::Value::Event::CallbackType m_reserveCallback;
    Buffer::Event::CallbackType m_elementBufferReallocationCallback;
    EventDispatch m_eventDispatch;
    size_t m_batchIndex;
    bool m_enableEvents;
  };

  AGXCORE_EXPORT std::ostream& operator << (std::ostream& output, const EntityStorage& storage);









  /* Implementation */

  //// ENTITY STORAGE //////////////////////////////////////////////////////
  AGX_FORCE_INLINE size_t EntityStorage::size() const { return m_size; }
  AGX_FORCE_INLINE bool EntityStorage::empty() const { return m_size == 0; }
  AGX_FORCE_INLINE size_t EntityStorage::capacity() const { agxAssert(capacityParameter->get() == m_capacity); return m_capacity; }
  AGX_FORCE_INLINE bool EntityStorage::useInstanceTable() const { return m_useInstanceTable; }
  AGX_FORCE_INLINE EntityModel *EntityStorage::getEntityModel() { return m_entity; }
  AGX_FORCE_INLINE const EntityModel *EntityStorage::getEntityModel() const { return m_entity; }
  AGX_FORCE_INLINE const BufferTable& EntityStorage::getBuffers() const { return m_bufferTable; }
  AGX_FORCE_INLINE const IndexArray& EntityStorage::getIdToIndexTable() const { return m_idToIndexArray; }
  AGX_FORCE_INLINE IndexArray& EntityStorage::getIdToIndexTable() { return m_idToIndexArray; }
  AGX_FORCE_INLINE Buffer *EntityStorage::getGlobalOrderBuffer() { return m_globalOrderBuffer; }
  AGX_FORCE_INLINE const Buffer *EntityStorage::getGlobalOrderBuffer() const { return m_globalOrderBuffer; }

  AGX_FORCE_INLINE Array<EntityPtr> EntityStorage::getInstances() { return Array<EntityPtr>(m_instanceBuffer); }
  AGX_FORCE_INLINE const Array<EntityPtr> EntityStorage::getInstances() const { return Array<EntityPtr>(m_instanceBuffer); }

  template <typename T>
  AGX_FORCE_INLINE Array<T> EntityStorage::getInstances()
  {
    agxAssert(agxData::getFormat<T>()->is(agxData::getFormat<EntityPtr>()));
    return Array<T>::raw((T *)m_instanceBuffer->ptr(), (agx::Index)m_instanceBuffer->size());
  }

  template <typename T>
  AGX_FORCE_INLINE const Array<T> EntityStorage::getInstances() const { return const_cast<EntityStorage *>(this)->getInstances<T>(); }

  AGX_FORCE_INLINE Buffer *EntityStorage::getBuffer(const Attribute *attribute)
  {
    Buffer *buffer = m_buffers[attribute->getId()];

    if (!buffer)
      buffer = this->createBuffer(attribute);

    return buffer;
  }


  template <typename T>
  AGX_FORCE_INLINE T& EntityStorage::get(const ScalarAttributeT<T> *attribute, size_t index)
  {
    Buffer *buffer = this->getBuffer(attribute);
    return buffer->getElement<T>(index);
  }

  template <typename T>
  AGX_FORCE_INLINE T& EntityStorage::get(const agx::Name& attributeName, size_t index)
  {
    for (size_t i = 0; i < m_buffers.size(); ++i)
    {
      Buffer *buffer = m_buffers[i];
      if (buffer->getName() == attributeName)
        return buffer->getElement<T>(index);
    }

    agxAbort();
    return m_buffers[0]->getElement<T>(0); // Just to make compiler happy
  }


  AGX_FORCE_INLINE size_t EntityStorage::idToIndex(size_t id) const
  {
    return m_idToIndexArray[id];
  }

  AGX_FORCE_INLINE size_t EntityStorage::indexToId(size_t index) const
  {
    return m_instanceBuffer->getElement<agxData::EntityPtr>(index).getId();
  }


  AGX_FORCE_INLINE void EntityStorage::destroyInstance(EntityPtr instance)
  {
    destroyInstance(instance.getId());
  }

  // inline EntityIterator EntityStorage::iterator() { return EntityIterator(this); }
  // inline EntityIterator EntityStorage::iterator(IndexRange range) { return EntityIterator(this, range); }


  AGX_FORCE_INLINE EntityData *EntityStorage::getData() { return m_dataSet; }

  template <typename T>
  AGX_FORCE_INLINE T *EntityStorage::getData() { return m_dataSet->as<T>(); }

  template <typename T>
  AGX_FORCE_INLINE const T *EntityStorage::getData() const { return m_dataSet->as<const T>(); }

  AGX_FORCE_INLINE Buffer *EntityStorage::getIdToIndexBuffer() { return m_idToIndexBuffer; }
  AGX_FORCE_INLINE const Buffer *EntityStorage::getIdToIndexBuffer() const { return m_idToIndexBuffer; }

  AGX_FORCE_INLINE Buffer *EntityStorage::getInstanceBuffer() { return m_instanceBuffer; }
  AGX_FORCE_INLINE const Buffer *EntityStorage::getInstanceBuffer() const { return m_instanceBuffer; }


  //// ENTITY DATA ///////////////////////////////////////////////////

  AGX_FORCE_INLINE size_t EntityData::idToIndex(size_t id) const
  {
    agxAssert(m_storage);
    return m_storage->idToIndex(id);
  }

  AGX_FORCE_INLINE size_t EntityData::indexToId(size_t index) const
  {
    agxAssert(m_storage);
    return m_storage->indexToId(index);
  }

  AGX_FORCE_INLINE agx::Index EntityData::createInstance()
  {
    return this->createInstances(1);
  }

  AGX_FORCE_INLINE agx::Index EntityData::createInstances(agx::Index numInstances)
  {
    if (m_numElements + numInstances >= m_storage->size())
    {
      if (!m_hasBatchInstances)
      {
        m_storage->beginBatch();
        m_hasBatchInstances = true;
      }

      agx::UInt head = m_numElements; // Save current size as resize will implicitly update m_numElements
      m_storage->resize(agx::align_ceil((agx::Index)(m_numElements + numInstances), (agx::Index)m_instanceBatchSize));
      m_numElements = head;
    }

    agx::Index instanceIndex = (agx::Index)m_numElements;
    m_numElements += numInstances;

    return instanceIndex;
  }


  AGX_FORCE_INLINE void EntityData::commitInstanceBatch()
  {
    if (m_hasBatchInstances)
    {
      m_storage->commitBatch(m_numElements);
      m_hasBatchInstances = false;
    }
  }



  //// ENTITY POINTER ///////////////////////////////////////////////////
  AGX_FORCE_INLINE void EntityPtr::init(EntityStorage *storage, agx::Index id)
  {
    m_storage = storage;
    m_id = id;
  }

  AGX_FORCE_INLINE EntityData *EntityPtr::getData()
  {
    agxAssert(m_storage);
    return m_storage->getData();
  }

  AGX_FORCE_INLINE const EntityData *EntityPtr::getData() const { return const_cast<EntityPtr *>(this)->getData(); }

  template <typename T>
  AGX_FORCE_INLINE T& EntityPtr::dataSet()
  {
    agxAssert(dynamic_cast<T *>(this->getData()));
    return *static_cast<T *>(this->getData());
  }

  template <typename T>
  AGX_FORCE_INLINE const T& EntityPtr::dataSet() const { return const_cast<EntityPtr *>(this)->dataSet<T>(); }


  AGX_FORCE_INLINE agx::Index EntityPtr::calculateIndex() const
  {
    agxAssert1(m_storage, "This is a null ptr");
    return (agx::Index)m_storage->idToIndex(m_id);
  }


  template <typename T>
  AGX_FORCE_INLINE T& EntityPtr::getElement(const Attribute *attribute) const
  {
    agxAssert(attribute);
    agxAssertN(m_storage, "Can not access attribute \'%s\' from an invalid pointer.", attribute->fullName().c_str());
    Buffer *buffer = m_storage->getBuffer(attribute);
    agx::Index index = this->calculateIndex();
    agxAssertN(index < m_storage->size(), "%s: Entity (%p, %u) claims to have index %u, but storage size is %u.", m_storage->getPath().c_str(), (void *)m_storage, (unsigned)m_id, (unsigned)index, (unsigned)m_storage->size());
    return *(T *)buffer->getElement(index);
  }

  AGX_FORCE_INLINE void EntityPtr::verifyIndex() const
  {
#ifdef AGX_DEBUG
    agx::Index index = this->calculateIndex();
    agxAssertN(index < m_storage->size(), "%s: Entity (%p, %u) claims to have index %u, but storage size is %u.", m_storage->getPath().c_str(), (void *)m_storage, (unsigned)m_id, (unsigned)index, (unsigned)m_storage->size());
#endif
  }


  template <typename T>
  AGX_FORCE_INLINE T& EntityPtr::operator() (const ScalarAttributeT<T> *attribute)
  {
    return this->getElement<T>(attribute);
  }

  template <typename T>
  AGX_FORCE_INLINE const T& EntityPtr::operator() (const ScalarAttributeT<T> *attribute) const
  {
    return this->getElement<const T>(attribute);
  }

  template <typename T>
  AGX_FORCE_INLINE T& EntityPtr::operator() (const PointerAttributeT<T> *attribute)
  {
    return this->getElement<T>(attribute);
  }

  template <typename T>
  AGX_FORCE_INLINE const T& EntityPtr::operator() (const PointerAttributeT<T> *attribute) const
  {
    return this->getElement<const T>(attribute);
  }


  template <typename T>
  AGX_FORCE_INLINE typename ArrayAttributeT<T>::Type& EntityPtr::operator() (const ArrayAttributeT<T> *attribute)
  {
    return this->getElement<typename ArrayAttributeT<T>::Type>(attribute);
  }

  template <typename T>
  AGX_FORCE_INLINE const typename ArrayAttributeT<T>::Type& EntityPtr::operator() (const ArrayAttributeT<T> *attribute) const
  {
    return this->getElement<const typename ArrayAttributeT<T>::Type>(attribute);
  }



  template <typename T>
  AGX_FORCE_INLINE typename VectorAttributeT<T>::Type& EntityPtr::operator() (const VectorAttributeT<T> *attribute)
  {
    return this->getElement<typename VectorAttributeT<T>::Type>(attribute);
  }

  template <typename T>
  AGX_FORCE_INLINE const typename VectorAttributeT<T>::Type& EntityPtr::operator() (const VectorAttributeT<T> *attribute) const
  {
    return this->getElement<const typename VectorAttributeT<T>::Type>(attribute);
  }

  template <typename T>
  AGX_FORCE_INLINE typename HashSetAttributeT<T>::Type& EntityPtr::operator() (const HashSetAttributeT<T> *attribute)
  {
    return this->getElement<typename HashSetAttributeT<T>::Type>(attribute);
  }

  template <typename T>
  AGX_FORCE_INLINE const typename HashSetAttributeT<T>::Type& EntityPtr::operator() (const HashSetAttributeT<T> *attribute) const
  {
    return this->getElement<const typename HashSetAttributeT<T>::Type>(attribute);
  }

  template <typename T>
  AGX_FORCE_INLINE Array<T> EntityStorage::dereferenceMultipliedAttribute(const EntityPtr& entity, const ScalarAttributeT<T> *attribute)
  {
    agxAssert(entity.getStorage() == this);
    Buffer *buffer = this->getBuffer(attribute);
    // agxAssert1(buffer->getFormat()->getSize() >= sizeof(T), "Only down-cast allowed, eg int64 -> int32");
    agxAssertN(entity.calculateIndex() < this->size(), "%s: Entity with id %u claims to have index %u, but storage size is %u.", this->getPath().c_str(), (unsigned)entity.getId(), (unsigned)entity.calculateIndex(), (unsigned)this->size());
    size_t arraySize = buffer->getElementArraySize();
    size_t offset = entity.calculateIndex() * arraySize;
    return Array<T>(buffer, agx::IndexRange(offset, offset + arraySize));
  }

  template <typename T>
  AGX_FORCE_INLINE const Array<T> EntityStorage::dereferenceMultipliedAttribute(const EntityPtr& entity, const ScalarAttributeT<T> *attribute) const
  {
    return const_cast<EntityStorage *>(this)->dereferenceMultipliedAttribute(entity, attribute);
  }



  template <typename VectorT>
  void EntityStorage::destroyInstances(const VectorT& instances)
  {
    this->destroyInstances(instances.begin(), instances.size());
  }


  AGX_FORCE_INLINE Byte *EntityStorage::getPermutationBuffer(agxData::Attribute *attribute)
  {
    return this->getPermutationBuffer(attribute->getId());
  }

  AGX_FORCE_INLINE Byte *EntityStorage::getPermutationBuffer(size_t index)
  {
    return m_permutationBuffers[index];
  }

  //---------------------------------------------------------------

  template <typename T>
  AGX_FORCE_INLINE T& EntityPtr::operator() (const SharedAttributeT<T> *attribute)
  {
    agxAssert(m_storage);
    agxData::Value *value = m_storage->getObject<agxData::Value>(attribute->getName());
    return const_cast<T&>(value->get<T>());
  }

  template <typename T>
  AGX_FORCE_INLINE const T& EntityPtr::operator() (const SharedAttributeT<T> *attribute) const
  {
    return const_cast<EntityPtr *>(this)->operator()(attribute);
  }


  AGX_FORCE_INLINE EntityModel* EntityPtr::getModel()
  {
    return m_storage ? m_storage->getEntityModel() : nullptr;
  }

  AGX_FORCE_INLINE const EntityModel* EntityPtr::getModel() const
  {
    return m_storage ? m_storage->getEntityModel() : nullptr;
  }

  AGX_FORCE_INLINE void EntityPtr::swapId(EntityPtr& other)
  {
    agxAssert(m_storage);
    agxAssert(m_storage == other.m_storage);

    m_storage->swapInstanceIds(m_id, other.m_id);
  }

  AGX_FORCE_INLINE void EntityPtr::swapIndex(EntityPtr& other)
  {
    agxAssert(m_storage);
    agxAssert(m_storage == other.m_storage);

    m_storage->swapInstances(calculateIndex(), other.calculateIndex());
  }


  AGX_FORCE_INLINE bool EntityPtr::isInstanceOf(agxData::EntityModel *model) const
  {
    return m_storage && m_storage->getEntityModel()->is(model);
  }


  inline EntityPtr EntityPtr::createWithIndex(EntityStorage *storage, agx::UInt32 index)
  {
    agxAssert(storage);
    return EntityPtr(storage, (agx::Index)storage->indexToId(index));
  }

  AGX_FORCE_INLINE EntityPtr::EntityPtr(const EntityInstance& instance) : m_storage(const_cast<EntityStorage *>(instance.getStorage())), m_id(agx::InvalidIndex)
  {
    if (m_storage)
      m_id = (agx::Index)m_storage->indexToId(instance.getIndex());
  }

  //// ENTITY INSTANCE ///////////////////////////////////////////////////

  AGX_FORCE_INLINE EntityInstance::EntityInstance(agxData::EntityStorage* storage, agx::Index index) : m_data(storage->getData()), m_index(index)
  {}

  AGX_FORCE_INLINE EntityInstance::EntityInstance(const agxData::EntityPtr& other)
  {
    agxAssert(other);

    m_data = const_cast<EntityData *>(other.getData());
    m_index = other.calculateIndex();
  }

  AGX_FORCE_INLINE agx::Index EntityInstance::calculateId() const
  {
    agxAssert(getStorage() && getStorage()->useInstanceTable());
    return (agx::Index)getStorage()->indexToId(m_index);
  }


  AGX_FORCE_INLINE agxData::EntityStorage *EntityInstance::getStorage() { return m_data ? m_data->getStorage() : nullptr; }
  AGX_FORCE_INLINE const agxData::EntityStorage *EntityInstance::getStorage() const { return const_cast<EntityInstance *>(this)->getStorage(); }

  AGX_FORCE_INLINE agxData::EntityModel *EntityInstance::getModel() { agxData::EntityStorage *storage = getStorage(); return storage ? storage->getEntityModel() : nullptr; }
  AGX_FORCE_INLINE const agxData::EntityModel *EntityInstance::getModel() const { return const_cast<EntityInstance *>(this)->getModel(); }


  AGX_FORCE_INLINE bool EntityInstance::isInstanceOf(agxData::EntityModel *model) const
  {
    const agxData::EntityModel *m = getModel();
    return m ? m->is(model) : false;
  }


  AGX_FORCE_INLINE void EntityInstance::verifyIndex() const
  {
#ifdef AGX_DEBUG
    const EntityStorage *storage = this->getStorage();
    agx::Index index = this->getIndex();
    agxAssertN(index < storage->size(), "%s: Entity instance (%p, %u) is out of bounds, storage size is %u.", storage->getPath().c_str(), (void *)storage, (unsigned)index, (unsigned)storage->size());
#endif
  }


  //// ENTITY RANGE ///////////////////////////////////////////////////

  AGX_FORCE_INLINE EntityRange::EntityRange() : m_storage(nullptr)
  {}

  AGX_FORCE_INLINE EntityRange::EntityRange(EntityStorage *storage, size_t start, size_t end) : agx::IndexRange(start, end), m_storage(storage)
  {
  }

  inline EntityPtr EntityRange::at(size_t index) { return operator[](index); }
  AGX_FORCE_INLINE EntityRange EntityRange::operator[] (agx::IndexRange subRange) { return EntityRange(m_storage, subRange.begin(), subRange.end()); }

  inline EntityPtr EntityRange::operator[] (size_t index)
  {
    agxAssert(m_storage);
    agxAssert(index < this->size());
    return EntityPtr(m_storage, (agx::UInt32)m_storage->indexToId(this->begin() + index));
  }



  //// ATTRIBUTE PTR //////////////////////////////////////////////////////
  template <typename T>
  inline AttributePtr<T>::AttributePtr() : m_attribute(nullptr)
  {
  }

  template <typename T>
  inline AttributePtr<T>::AttributePtr(EntityPtr entity, const Attribute *attribute) : m_entityPtr(entity), m_attribute(attribute)
  {
  }

  template <typename T>
  AGX_FORCE_INLINE T& AttributePtr<T>::get()
  {
    agxAssert(this->isValid());
    return m_entityPtr.getElement<T>(m_attribute);
  }

  template <typename T>
  AGX_FORCE_INLINE const T& AttributePtr<T>::get() const
  {
    agxAssert(this->isValid());
    return m_entityPtr.getElement<T>(m_attribute);
  }

  template <typename T>
  AGX_FORCE_INLINE bool AttributePtr<T>::isValid() const
  {
    return m_entityPtr.isValid() && m_attribute;
  }


  template <typename T>
  AGX_FORCE_INLINE bool AttributePtr<T>::operator!() const
  {
    return !this->isValid();
  }

  template <typename T>
  AGX_FORCE_INLINE AttributePtr<T>::operator bool() const
  {
    return this->isValid();
  }

}

#include <agx/EntityStorageEvent.Implementation.h>

#ifdef _MSC_VER
# pragma warning(pop)
#endif

