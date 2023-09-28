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





#ifndef AGX_HASHTABLECONTEXT_H
#define AGX_HASHTABLECONTEXT_H


#include <agx/Component.h>
#include <agx/QuadraticProbingHashTable.h>

#include <agxData/Buffer.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4251 ) //  needs to have dll-interface to be used by clients
#endif

DOXYGEN_START_INTERNAL_BLOCK()

namespace agx
{
  class AGXCORE_EXPORT AbstractHashTableComponent : public Component
  {
  public:
    static agx::Model *ClassModel();

    static AbstractHashTableComponent *load(TiXmlElement *eHashTable, Device *device);

  public:
    AbstractHashTableComponent(const agx::Name& name);

    // Access bucket buffer
    agxData::Buffer *getBucketBuffer();

  protected:
    virtual ~AbstractHashTableComponent();

  protected:
    agxData::BufferRef m_buckets;
    // agxData::Val<UInt> m_numBuckets; // Use "buckets.numElements" instead
    agxData::Val<UInt> m_numFilled;
    agxData::Val<UInt> m_maxProbeLength;
  };


  template <typename KeyT, typename DataT, typename HashT = agx::HashFn<KeyT> >
  class HashTableComponent : public AbstractHashTableComponent
  {
  public:

    // Cannot use implementation agnostic HashTable type alias because the
    // HashTableComponent make assumptions about the memory layout of its
    // underlying HashTable container. Must therefore use
    // QuadraticProbingHashTable.
    typedef QuadraticProbingHashTable<KeyT, DataT, HashT, agxData::BufferProxyAllocator> ImplementationType;
    // void configure(TiXmlElement *eHashTable);

  public:
    HashTableComponent(const agx::Name& name);

    // Synchronize implementation with buffer and values
    void commitImplementation();
    void updateImplementation();

    // Access the agx::HashTable implementation
    ImplementationType& impl();
    const ImplementationType& impl() const;


    // 'Generic' constructor
    HashTableComponent(const agx::Name& name, size_t numBytes);

  protected:
    virtual ~HashTableComponent();

  private:
    ImplementationType m_implementation;
  };



  /* Implementation */
  AGX_FORCE_INLINE agxData::Buffer *AbstractHashTableComponent::getBucketBuffer() { return m_buckets; }

  template <typename KeyT, typename DataT, typename HashT>
  HashTableComponent<KeyT, DataT, HashT>::HashTableComponent(const agx::Name& name) : AbstractHashTableComponent(name)
  {
    this->addObject(&m_numFilled);
    this->addObject(&m_maxProbeLength);

    m_buckets = new agxData::Buffer("buckets", agxData::getFormat<typename ImplementationType::Bucket>());
    this->addObject(m_buckets);

    m_implementation.allocator().setBuffer(m_buckets);
  }

  template <typename KeyT, typename DataT, typename HashT>
  HashTableComponent<KeyT, DataT, HashT>::HashTableComponent(const agx::Name& name, size_t numBytes) : AbstractHashTableComponent(name)
  {
    this->addObject(&m_numFilled);
    this->addObject(&m_maxProbeLength);

    m_buckets = new agxData::Buffer("buckets", agxData::getGenericStructFormat(numBytes));
    this->addObject(m_buckets);

    m_implementation.allocator().setBuffer(m_buckets);
  }


  template <typename KeyT, typename DataT, typename HashT>
  HashTableComponent<KeyT, DataT, HashT>::~HashTableComponent()
  {
    m_implementation.clear(Container::SHRINK_BUFFER);
    // m_implementation.setExplicitBuffer(nullptr, 0);
    this->removeAllObjects();
  }

  template <typename KeyT, typename DataT, typename HashT>
  AGX_FORCE_INLINE typename HashTableComponent<KeyT, DataT, HashT>::ImplementationType& HashTableComponent<KeyT, DataT, HashT>::impl() { return m_implementation; }

  template <typename KeyT, typename DataT, typename HashT>
  AGX_FORCE_INLINE const typename HashTableComponent<KeyT, DataT, HashT>::ImplementationType& HashTableComponent<KeyT, DataT, HashT>::impl() const { return m_implementation; }

  template <typename KeyT, typename DataT, typename HashT>
  void HashTableComponent<KeyT, DataT, HashT>::commitImplementation()
  {
    m_numFilled = m_implementation.m_size;
    m_maxProbeLength = m_implementation.m_maxProbeLength;
    // m_buckets->m_minSize = m_implementation.capacity();
    m_implementation.allocator().commit();
  }

  template <typename KeyT, typename DataT, typename HashT>
  void HashTableComponent<KeyT, DataT, HashT>::updateImplementation()
  {
    m_implementation.allocator().update();
    m_implementation.m_size = m_numFilled;
    m_implementation.m_maxProbeLength = m_maxProbeLength;
  }

}
DOXYGEN_END_INTERNAL_BLOCK()


#ifdef _MSC_VER
# pragma warning(pop)
#endif


#endif /* AGX_HASHTABLECONTEXT_H */
