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

#ifndef AGX_HIERARCHICAL_GRID_CELLTABLE_H
#define AGX_HIERARCHICAL_GRID_CELLTABLE_H

#include <agx/Vec3.h>
#include <agx/QuadraticProbingHashTable.h>
#include <agxData/Array.h>
#include <agxData/Buffer.h>

namespace agx
{
  template <typename CellIdT, typename HashT = HashFn<CellIdT> >
  class CellTable
  {
  public:
    class Bucket
    {
    public:
      enum State
      {
        INACTIVE,
        ACTIVE,
        VALID
      };
     
      UInt32 cellIndex;
      
      AGX_FORCE_INLINE Bucket() : cellIndex(InvalidIndex), state(INACTIVE)
      {
      }
      
      AGX_FORCE_INLINE Bucket& operator=(const Bucket& b )
      {
        cellIndex = b.cellIndex;
        state = b.state;

        return *this;
      }
      
      AGX_FORCE_INLINE bool isActive() const
      { return state >= ACTIVE; }

      AGX_FORCE_INLINE bool isValid() const
      {
        // prefetch( &state, agx::NTA );
        return state == VALID; 
      }
      
    private:
      friend class CellTable<CellIdT, HashT>;
      agx::UInt8 state;
    };
    
    
    template<typename T>
    class template_iterator
    {
    public:
      /** Constructors */
      AGX_FORCE_INLINE template_iterator() : m_cellTable(0), m_currentIndex(0) {}
      
      AGX_FORCE_INLINE template_iterator(T* hashSet, size_t startIndex = 0) : m_cellTable(hashSet), m_currentIndex(startIndex-1)
      { this->gotoNextElement(); }
      
      /** Cast operator from non-const to const */
      #ifndef __clang__ // TODO Make sure clang does not produce warnings for this
      AGX_FORCE_INLINE operator template_iterator<const T>() const
      {
        template_iterator<const T> constIt;
        constIt.m_cellTable = m_cellTable;
        constIt.m_currentIndex = m_currentIndex;
        return constIt;
      }
      #endif

      AGX_FORCE_INLINE template_iterator<T>& operator++()
      {
        this->gotoNextElement();
        return *this;
      }
      
      AGX_FORCE_INLINE template_iterator<T> operator++(int)
      {
        size_t oldIndex = m_currentIndex;
        this->gotoNextElement();
          
        return template_iterator<T>(m_cellTable, oldIndex);
      }


      AGX_FORCE_INLINE template_iterator<T> operator--()
      {
        this->gotoPreviousElement();
        return *this;
      }
      
      AGX_FORCE_INLINE template_iterator<T> operator--(int)
      {
        size_t oldIndex = m_currentIndex;
        this->gotoPreviousElement();
          
        return template_iterator<T>(m_cellTable, oldIndex);
      }
     
      AGX_FORCE_INLINE bool operator==(const template_iterator<T>& rhs)
      { return this->m_currentIndex == rhs.m_currentIndex && this->m_cellTable == rhs.m_cellTable; }
      
      AGX_FORCE_INLINE bool operator!=(const template_iterator<T>& rhs)
      { return this->m_currentIndex != rhs.m_currentIndex || this->m_cellTable != rhs.m_cellTable; }
      

      AGX_FORCE_INLINE UInt32& operator*() const
      {
        agxAssert1(m_cellTable && m_currentIndex != m_cellTable->m_capacity, "Can not dereference an invalid bucket!");
        return m_cellTable->m_buckets[m_currentIndex].cellIndex;
      }
      
      AGX_FORCE_INLINE UInt32 *operator->() const
      { return &(operator*()); }
      

      AGX_FORCE_INLINE template_iterator<T>& operator =( const template_iterator<T>& copy) 
      {
        m_cellTable = copy.m_cellTable;
        m_currentIndex = copy.m_currentIndex;
        return *this;
      }

    private:
      AGX_FORCE_INLINE void gotoNextElement()
      {
        m_currentIndex++;
        while (m_currentIndex < m_cellTable->m_capacity && !m_cellTable->m_buckets[m_currentIndex].isValid())
          m_currentIndex++;
      }
      
      AGX_FORCE_INLINE void gotoPreviousElement()
      {
        if (m_currentIndex > 0) {
          m_currentIndex--;
          while (m_currentIndex > 0 && !m_cellTable->buckets()[m_currentIndex].isValid())
            m_currentIndex--;
        }
      }
    private:
      friend class CellTable;
      T* m_cellTable;
      size_t m_currentIndex;
    };
    
    /** Iterators */
    typedef template_iterator< CellTable > iterator;
    typedef template_iterator< const CellTable > const_iterator;
    
    
  public:
    CellTable(agxData::Buffer *cellIdBuffer) :
      m_size(0), m_capacity(0), m_buckets(nullptr),
      m_smoothingAverage(0), m_maxProbeLength(0),
      m_cellIds(cellIdBuffer->ptr<CellIdT>()), m_cellIdBuffer(cellIdBuffer), m_cellIdReallocationCallback(&CellTable::cellIdReallocationCallback, this)
    {
      m_cellIdBuffer->reallocationEvent.addCallback(&m_cellIdReallocationCallback);
      this->rebuild(0);
    }
    
    ~CellTable()
    {
      m_cellIdBuffer->reallocationEvent.removeCallback(&m_cellIdReallocationCallback);
      m_allocator.deallocateBytes(m_buckets, m_capacity * sizeof(Bucket));
    }
    
    
    AGX_FORCE_INLINE UInt32 find(const CellIdT& cellId) const
    {
      UInt32 bucketIndex = findBucket(cellId);
      // std::cout << "Found " << cellId << " at " << bucketIndex << std::endl;
      return bucketIndex != InvalidIndex ? m_buckets[bucketIndex].cellIndex : InvalidIndex;
    }
    
    AGX_FORCE_INLINE bool contains(const CellIdT& cellId) const
    {
      return this->findBucket(cellId) != InvalidIndex;
    }
    
    void updateWithReordering(const agxData::Array<UInt32>& sortedIndices)
    {
      // std::cout << "numBuckets: " << m_capacity << ", size: " << m_size << ", maxProbeLength: " << m_maxProbeLength << std::endl;
      for (size_t i = 0; i < m_capacity; ++i)
      {
        if (m_buckets[i].isValid())
          m_buckets[i].cellIndex = sortedIndices[m_buckets[i].cellIndex];
      }
    }

    
    void insert(const CellIdT& cellId, UInt32 cellIndex)
    {
      UInt32 hashValue = m_hashFunctor(cellId);
      Bucket *bucket;
      Bucket *targetBucket = 0;
      size_t offset = 0;
      // std::cout << "Insert " << cellIndex << " with id " << cellId << std::endl;

      /* Check if we should grow and rebuild hash table */
      if (m_size >= (size_t)((float)m_capacity * AGX_HASH_TABLE_GROW_THRESHOLD))
        this->rebuild((size_t)((float)(m_capacity+1) * AGX_HASH_TABLE_GROW_FACTOR));

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127:  conditional expression is constant
#endif

      {
        /* Iterate until target bucket is found */
        const bool trueVal = true;
        do
        {
          bucket = &m_buckets[(hashValue + offset * offset) % m_capacity];

          /* Check if overwriting previous insertion with same key */
          if (bucket->isValid() && m_cellIds[bucket->cellIndex] == cellId)
          {
            targetBucket = bucket;
            break;
          }

          /* Tag first free slot */
          if (!bucket->isValid() && !targetBucket)
            targetBucket = bucket;

          /* If we have found a free bucket we can stop iterating when we are sure that the key was not previously inserted */
          if (targetBucket && offset >= m_maxProbeLength)
            break;

          /* Go to next bucket */
          offset++;

          /* Resize and rebuild table if it takes too long to find a free slot */
          if (offset > AGX_HASH_TABLE_MAX_PROBE_LENGTH)
          {
            this->rebuild((size_t)((float)m_capacity * AGX_HASH_TABLE_GROW_FACTOR));
            this->insert(cellId, cellIndex);
            return;
          }
        } while (trueVal);
      }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
      
      agxAssert(targetBucket);
      
      if (offset > m_maxProbeLength)
        m_maxProbeLength = (UInt16)offset;

      targetBucket->cellIndex = cellIndex;

      /* Check if inserting a new value rather than overwriting an old entry */
      if (!targetBucket->isValid())
      {
        m_size++;
        targetBucket->state = Bucket::VALID;
      }
      
    }
    
    void erase(const CellIdT& cellId)
    {
      UInt32 bucketIndex = findBucket(cellId);
      agxAssert(bucketIndex != InvalidIndex);
      
      Bucket *bucket = &m_buckets[bucketIndex];
      agxAssert(bucket->isValid() && m_cellIds[bucket->cellIndex] == cellId);
      
      m_size--;

      // Bucket does not become inactive since it might be in the middle of a probing chain
      bucket->state = Bucket::ACTIVE;
    }
    
    /**
    Iterator to first element in hash set.
    */
    AGX_FORCE_INLINE iterator begin()
    { return iterator(this); }
    
    AGX_FORCE_INLINE const_iterator begin() const
    { return const_iterator(this); }
    
    
    /**
    Iterator marking end of hash set.
    */
    AGX_FORCE_INLINE iterator end()
    { return iterator(this, m_capacity); }
    
    AGX_FORCE_INLINE const_iterator end() const
    { return const_iterator(this, m_capacity); }
    
  private:
    AGX_FORCE_INLINE UInt32 findBucket(const CellIdT& cellId) const
    {
      agxAssert(m_buckets);
        
      UInt32 hashValue = m_hashFunctor(cellId);
      size_t index = hashValue % m_capacity;
      const Bucket *bucket = &m_buckets[index];
      
      /* First test is outside for-loop, hopefully getting better branch prediction */
      if (bucket->isValid() && m_cellIds[bucket->cellIndex] == cellId)
        return (UInt32)index;
      
      
      for (size_t offset = 1; bucket->isActive() && offset <= m_maxProbeLength; offset++)
      {
        index  = (hashValue + offset * offset) % m_capacity;
        bucket = &m_buckets[index];
        
        if (bucket->isValid() && m_cellIds[bucket->cellIndex] == cellId)
          return (UInt32)index;
      }
      
      /* Not found */
      return InvalidIndex;
    }
    
    void rebuild(size_t size)
    {
      size = std::max( size, size_t(AGX_HASH_TABLE_MIN_SIZE) );
      // std::cout << "HashTable: Rebuilding and resizing from " << m_capacity << " to " << agx::nextPrime((int)(size)) << " buckets." << std::endl;
      agxAssert(size >= m_size);
      size_t oldNumBuckets = m_capacity;
      Bucket *oldBuckets = m_buckets;

      m_capacity = agx::nextPrime((int)size);
      m_buckets = (Bucket *)m_allocator.allocateBytes(m_capacity * sizeof(Bucket));
      for (size_t i = 0; i < m_capacity; ++i)
        ::new (&m_buckets[i]) Bucket();
      
      m_size = 0;
      m_maxProbeLength = 0;
      
      for (size_t i = 0; i < oldNumBuckets; i++)
      {
        Bucket& bucket = oldBuckets[i];

        if (bucket.isValid())
          this->insert(m_cellIds[bucket.cellIndex], bucket.cellIndex);
      }

      m_allocator.deallocateBytes((void *)oldBuckets, oldNumBuckets * sizeof(Bucket));
    }
    
    void cellIdReallocationCallback(agxData::Buffer *)
    {
      // std::cout << "cellIdReallocationCallback" << std::endl;
      m_cellIds = m_cellIdBuffer->ptr<CellIdT>();
    }
    
  private:
    size_t m_size;
    size_t m_capacity;
    Bucket *m_buckets;
    // UInt32 *m_bucketValues;
    // Bucket::State *m_bucketStates;
    
    Real32 m_smoothingAverage;
    UInt16 m_maxProbeLength;
    ByteAllocator m_allocator;
    HashT m_hashFunctor;
    
    
    const CellIdT *m_cellIds;
    agxData::BufferRef m_cellIdBuffer;
    agxData::Buffer::Event::CallbackType m_cellIdReallocationCallback;
  };
}


#endif /* AGX_HIERARCHICAL_GRID_CELLTABLE_H */
