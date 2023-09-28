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

#ifndef AGX_QUADRATIC_PROBING_HASH_TABLE_H
#define AGX_QUADRATIC_PROBING_HASH_TABLE_H


#include <agx/Math.h>
#include <agx/agx.h>
#include <agx/ref_ptr.h>

#include <agx/HashFunction.h>
#include <agx/Container.h>
#include <agx/Allocator.h>

#include <cstddef> // mostly for size_t

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 6011) // Disable warningC6011: dereferencing nullptr pointer
#endif


namespace agx
{

  /**
  A simple hash table using quadratic probing. Built-in support for keys of type int, agx::String,
  and pointers. To use other types of keys, a custom hash functor must be supplied.

  \\todo implement [] operator
  */
  template <typename KeyT, typename DataT, typename HashT = HashFn<KeyT>, typename AllocatorT = ByteAllocator>
  class QuadraticProbingHashTableImplementation : public Container
  {
  public:
    typedef DataT value_type;
    typedef KeyT key_type;
    typedef HashT hash_fn_type;

  private:
    #define AGX_HASH_TABLE_MIN_SIZE 4

    #define AGX_HASH_TABLE_GROW_THRESHOLD 0.8
    #define AGX_HASH_TABLE_SHRINK_THRESHOLD 0.25
    #define AGX_HASH_TABLE_GROW_FACTOR 2.0
    #define AGX_HASH_TABLE_SHRINK_FACTOR 0.5
    #define AGX_HASH_TABLE_SMOOTHING_FACTOR 0.8f
    #define AGX_HASH_TABLE_MAX_PROBE_LENGTH 10

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

      /**
      We used to wrap key,data in a std::pair. But doing so gives double
      padding by compiler as first the std::pair is aligned, and then the Bucket
      is aligned. Storing key,data,state directly in struct gives better compaction.
      */

      // std::pair<KeyT, DataT> pair;
      KeyT first;
      DataT second;

      // Cast operator to std::pair
      AGX_FORCE_INLINE operator std::pair<KeyT, DataT>() { return std::make_pair(first, second); }

      AGX_FORCE_INLINE Bucket() : state(INACTIVE)
      {
      }

      AGX_FORCE_INLINE Bucket& operator=(const Bucket& b )
      {
        if (&b == this)
          return *this;

        first = b.first;
        second = b.second;
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

    AGX_FORCE_INLINE agx::UInt8 getState() const
    {
      return state;
    }

      #if 0
      AGX_FORCE_INLINE bool isActive() const
      { return (state & (VALID | ACTIVE)) != 0; }

      AGX_FORCE_INLINE bool isValid() const
      { return (state & VALID) != 0; }
      #endif

    private:
      friend class QuadraticProbingHashTableImplementation<KeyT, DataT, HashT, AllocatorT>;
      agx::UInt8 state;
    };

  public:

    template<typename T>
    class template_iterator
    {
    public:
      typedef T QuadraticProbingHashTableT;

    public:
      /** Constructors */
      AGX_FORCE_INLINE template_iterator() : m_hashTable(0), m_currentIndex(0) {}

      AGX_FORCE_INLINE template_iterator(T* hashTable, size_t startIndex = 0) : m_hashTable(hashTable), m_currentIndex(startIndex-1)
      { this->gotoNextElement(); }

      /** Cast operator from non-const to const */
      AGX_FORCE_INLINE operator template_iterator<const T>() const
      {
        template_iterator<const T> constIt;
        constIt.m_hashTable = m_hashTable;
        constIt.m_currentIndex = m_currentIndex;
        return constIt;
      }

      AGX_FORCE_INLINE template_iterator<T>& operator++()
      {
        this->gotoNextElement();
        return *this;
      }

      AGX_FORCE_INLINE template_iterator<T> operator++(int)
      {
        size_t oldIndex = m_currentIndex;
        this->gotoNextElement();

        return template_iterator<T>(m_hashTable, oldIndex);
      }

      AGX_FORCE_INLINE template_iterator<T>& operator--()
      {
        this->gotoPreviousElement();
        return *this;
      }

      AGX_FORCE_INLINE template_iterator<T> operator--(int)
      {
        size_t oldIndex = m_currentIndex;
        this->gotoPreviousElement();

        return template_iterator<T>(m_hashTable, oldIndex);
      }

      AGX_FORCE_INLINE bool operator==(const template_iterator<T>& rhs)
      { return this->m_currentIndex == rhs.m_currentIndex && this->m_hashTable == rhs.m_hashTable; }

      AGX_FORCE_INLINE bool operator!=(const template_iterator<T>& rhs)
      { return this->m_currentIndex != rhs.m_currentIndex || this->m_hashTable != rhs.m_hashTable; }

      AGX_FORCE_INLINE Bucket& operator*() const
      {
        agxAssert1(m_hashTable && m_currentIndex != m_hashTable->numBuckets(), "Can not dereference an invalid bucket!");
        return m_hashTable->buckets()[m_currentIndex];
      }

      AGX_FORCE_INLINE Bucket *operator->() const
      { return &(operator*()); }


      AGX_FORCE_INLINE template_iterator<T>& operator =( const template_iterator<T>& copy)
      {
        m_hashTable = copy.m_hashTable;
        m_currentIndex = copy.m_currentIndex;
        return *this;
      }

    private:
      AGX_FORCE_INLINE void gotoNextElement()
      {
        m_currentIndex++;
        while (m_currentIndex < m_hashTable->numBuckets() && !m_hashTable->buckets()[m_currentIndex].isValid())
          m_currentIndex++;
      }

      AGX_FORCE_INLINE void gotoPreviousElement()
      {
        m_currentIndex--;
        while (!m_hashTable->buckets()[m_currentIndex].isValid())
          m_currentIndex--;
      }
    private:
      friend class QuadraticProbingHashTableImplementation;
      QuadraticProbingHashTableT* m_hashTable;
      size_t m_currentIndex;
    };

    /** Iterators */
    typedef template_iterator< QuadraticProbingHashTableImplementation<KeyT, DataT, HashT, AllocatorT> > iterator;
    typedef template_iterator< const QuadraticProbingHashTableImplementation<KeyT, DataT, HashT, AllocatorT> > const_iterator;

    class insert_iterator : public iterator
    {
    public:
      typedef typename iterator::QuadraticProbingHashTableT QuadraticProbingHashTableT;

    public:

      AGX_FORCE_INLINE insert_iterator() : iterator(), m_overwrite(false)
      {
      }

      AGX_FORCE_INLINE insert_iterator(QuadraticProbingHashTableT* hashTable, size_t startIndex, bool overwrite) : iterator(hashTable, startIndex), m_overwrite(overwrite)
      {
      }

      /**
      \return True if the insert operation did overwrite an existing entry.
      */
      AGX_FORCE_INLINE bool didOverwrite() const
      {
        return m_overwrite;
      }

    private:
      bool m_overwrite;
    };


    /** Constructor */
    AGX_FORCE_INLINE QuadraticProbingHashTableImplementation(const AllocatorT& allocator = AllocatorT()) : m_smoothingAverage(0), m_maxProbeLength(0), m_allocator(allocator)
    { m_allocator.setContainer(this); }

    /** Copy constructor */
    AGX_FORCE_INLINE QuadraticProbingHashTableImplementation(const QuadraticProbingHashTableImplementation<KeyT, DataT, HashT, AllocatorT>& other) : m_smoothingAverage(0), m_maxProbeLength(0)
    {
      *this = other;
    }

    /** Assignment operator */
    AGX_FORCE_INLINE QuadraticProbingHashTableImplementation<KeyT, DataT, HashT, AllocatorT>& operator= (const QuadraticProbingHashTableImplementation<KeyT, DataT, HashT, AllocatorT>& other)
    {
      m_allocator = other.m_allocator;
      m_allocator.setContainer(this);

      this->clear(MAINTAIN_BUFFER);
      this->reserve(other.size());

      for (const_iterator it = other.begin(); it != other.end(); ++it)
        this->insert(it->first, it->second);

      return *this;
    }


    /** Destructor */
    AGX_FORCE_INLINE ~QuadraticProbingHashTableImplementation()
    {
      for (size_t i = 0; i < numBuckets(); ++i)
        buckets()[i].~Bucket();

      m_allocator.deallocateBytes(m_buffer, m_capacity * sizeof(Bucket));
    }


    /**
    Index operator for accessing elements in the hash, inserting it if not found.
    \param key - The key to look for
    \return reference to the matching data.
    */
    AGX_FORCE_INLINE DataT& operator[](const KeyT& key )
    {
      iterator it = find( key );
      if (it == end() ){
        it = insert( key, value_type() );
      }
      return it->second;
    }

    AGX_FORCE_INLINE const DataT& operator[](const KeyT& key ) const
    {
      const_iterator it = find( key );
      agxAssert( it != end() );
      return it->second;
    }

    /**
    Returns value given key. If the key doesn't exist, the key
    is inserted and the default value is returned.
    \param key - key
    \return data for key
    */
    DataT& getOrCreate( const KeyT& key )
    {
      return (*this)[ key ];
    }

    /**
    Insert an element, overwriting if key already exists.
    \return An iterator to the inserted pair
    */
    insert_iterator insert(const KeyT& key, const DataT& data)
    {
      UInt32 hashValue = m_hashFunctor(key);
      Bucket *bucket;
      Bucket *targetBucket = 0;
      size_t offset = 0;

      /* Check if we should grow and rebuild hash table */
      if (numFilled() >= (size_t)((float)numBuckets() * AGX_HASH_TABLE_GROW_THRESHOLD))
        this->rebuild((size_t)((float)(numBuckets()+1) * AGX_HASH_TABLE_GROW_FACTOR));

      /* Iterate until target bucket is found */
      bool alwaysTrue = true;
      do
      {
        bucket = &buckets()[(hashValue + offset * offset) % numBuckets()];

        /* Check if inactive bucket or overwriting previous insertion with same key */
        if (!bucket->isActive() || (bucket->isValid() && hashKeyEqual(bucket->first, key)))
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
          this->rebuild((size_t)((float)numBuckets() * AGX_HASH_TABLE_GROW_FACTOR));
          return this->insert(key, data);
        }
      } while (alwaysTrue);

      agxAssert(targetBucket);

      if (offset > m_maxProbeLength)
        m_maxProbeLength = (UInt16)offset;


      targetBucket->first = key;
      targetBucket->second = data;


      /* Check if inserting a new value rather than overwriting an old entry */
      bool overwrite = targetBucket->isValid();
      if (!overwrite)
      {
        numFilled()++;
        targetBucket->state = Bucket::VALID;
      }

      return insert_iterator(this, targetBucket - buckets(), overwrite);
    }

    /**
    Boolean query to test existence of an element.
    */
    AGX_FORCE_INLINE bool contains(const KeyT& key) const
    {
      return this->findBucket(key) != numBuckets();
    }

    /**
    Find an element.
    */
    AGX_FORCE_INLINE iterator find(const KeyT& key)
    {
      return iterator(this, this->findBucket(key));
    }

    AGX_FORCE_INLINE const_iterator find(const KeyT& key) const
    {
      return const_iterator(this, this->findBucket(key));
    }


    /**
    Iterator to first element in hash table.
    */
    AGX_FORCE_INLINE iterator begin()
    { return iterator(this); }

    AGX_FORCE_INLINE const_iterator begin() const
    { return const_iterator(this); }


    /**
    Iterator marking end of hash table.
    */
    AGX_FORCE_INLINE iterator end()
    { return iterator(this, numBuckets()); }

    AGX_FORCE_INLINE const_iterator end() const
    { return const_iterator(this, numBuckets()); }


    /**
    Erase an element from the hash table.
    \return false if element was not found
    */
    AGX_FORCE_INLINE bool erase(const KeyT& key)
    {
      return this->eraseBucket(this->findBucket(key));
    }

    /**
    Erase an element using an iterator.
    \return An iterator to the next element
    */
    AGX_FORCE_INLINE iterator erase(const iterator& it)
    {
      this->eraseBucket(it.m_currentIndex);
      return iterator(this, it.m_currentIndex);
    }

    /**
    Remove all elements.
    */
    void clear(ClearPolicy policy = SHRINK_BUFFER_AVERAGED)
    {
      if (policy == MAINTAIN_BUFFER)
      {
        for (size_t i = 0; i < numBuckets(); i++)
        {
          buckets()[i].first.~KeyT();
          buckets()[i].second.~DataT();
          ::new (&(buckets()[i])) Bucket();

          //buckets()[i] = Bucket();
        }
      }
      else if (policy == SHRINK_BUFFER)
      {
        for (size_t i = 0; i < numBuckets(); ++i)
          buckets()[i].~Bucket();

        m_allocator.deallocateBytes(m_buffer, m_capacity * sizeof(Bucket));
        m_buffer = nullptr;
        numBuckets() = 0;

        m_smoothingAverage = 0.0;
      }
      else //if (policy == SHRINK_BUFFER_AVERAGED)
      {
        /* Smoothing average is never less than current size */
        if ((Real32)numFilled() > m_smoothingAverage)
          m_smoothingAverage = Real32(numFilled());

        /* Update smoothing average */
        m_smoothingAverage = Real32(AGX_HASH_TABLE_SMOOTHING_FACTOR * m_smoothingAverage + (1.0-AGX_HASH_TABLE_SMOOTHING_FACTOR) * (Real32)numFilled());

        /* Reallocate buffer if the smoothing average is sufficiently below the current capacity */
        if (m_smoothingAverage < AGX_HASH_TABLE_SHRINK_THRESHOLD * (float)numBuckets() && m_smoothingAverage >= AGX_HASH_TABLE_MIN_SIZE)
        {
          for (size_t i = 0; i < numBuckets(); ++i)
            buckets()[i].~Bucket();

          void *oldBuffer = m_buffer;
          size_t oldCapacity = m_capacity;

          numBuckets() = agx::nextPrime((int)m_smoothingAverage);
          m_buffer = m_allocator.allocateBytes(numBuckets() * sizeof(Bucket));

          for (size_t i = 0; i < numBuckets(); ++i)
            ::new (&(buckets()[i])) Bucket();

          m_allocator.deallocateBytes(oldBuffer, oldCapacity * sizeof(Bucket));
        }
        else
        {
          for (size_t i = 0; i < numBuckets(); i++)
          {
            buckets()[i].first.~KeyT();
            buckets()[i].second.~DataT();
            ::new (&(buckets()[i])) Bucket();
          }
        }
      }


      numFilled() = 0;
      m_maxProbeLength = 0;
    }


    /**
    Reserve capacity in the hash table.
    */
    AGX_FORCE_INLINE void reserve(size_t size)
    {
      if (size <= (size_t)((float)numBuckets() * AGX_HASH_TABLE_GROW_THRESHOLD))
        return;

      this->rebuild((size_t)((float)size / AGX_HASH_TABLE_GROW_THRESHOLD));
    }

    /**
    Purge the hash table. Good to use when removing multiple items, since iterators
    are invalidated after any modification. The purger should be a functor class
    which takes the key and value as parameters and return true if the pair should
    be removed from the hash table.
    */
    template<typename T>
    void purge(T purger)
    {
      for (size_t i = 0; i < numBuckets(); i++)
      {
        Bucket& bucket = buckets()[i];

        if (bucket.isValid() && purger(bucket.first, bucket.second))
        {
          //bucket = Bucket();
          bucket.first.~KeyT();
          bucket.second.~DataT();
          ::new (&bucket) Bucket();

          bucket.state = Bucket::ACTIVE;
          numFilled()--;
        }
      }

      if ((float)numBuckets() * AGX_HASH_TABLE_SHRINK_FACTOR >= AGX_HASH_TABLE_MIN_SIZE && (float)numFilled()/(float)numBuckets() < AGX_HASH_TABLE_SHRINK_THRESHOLD)
        this->rebuild( size_t((float)numFilled() * AGX_HASH_TABLE_GROW_FACTOR) );
    }

    AllocatorT& allocator() { return m_allocator; }
    const AllocatorT& allocator() const { return m_allocator; }

    DOXYGEN_START_INTERNAL_BLOCK()
    AGX_FORCE_INLINE size_t getNumActiveElements() const { return numBuckets(); }
    DOXYGEN_END_INTERNAL_BLOCK()

  public:
    AGX_FORCE_INLINE const Bucket *getBucketBuffer() const { return buckets(); }
    AGX_FORCE_INLINE size_t getMaxProbeLength() const { return m_maxProbeLength; }
    AGX_FORCE_INLINE size_t getNumBuckets() const { return numBuckets(); }


  protected:
    template <typename T1, typename T2, typename T3>
    friend class HashTableComponent;

    // Access to Container members
    AGX_FORCE_INLINE size_t& numBuckets() { return m_capacity; }
    AGX_FORCE_INLINE size_t& numFilled() { return m_size; }

    AGX_FORCE_INLINE Bucket *buckets() const { return static_cast<Bucket *>(m_buffer); }
    AGX_FORCE_INLINE size_t numBuckets() const { return m_capacity; }
    AGX_FORCE_INLINE size_t numFilled() const { return m_size; }


    /**
    Rebuild, and optionally resize hash table.
    */
    void rebuild(size_t size)
    {
      size = std::max( size, size_t(AGX_HASH_TABLE_MIN_SIZE) );
      //LOGGER_DEBUG() << "QuadraticProbingHashTable: Rebuilding and resizing from " << numBuckets() << " to " << agx::nextPrime((int)(numBuckets() * resizeFactor)) << " buckets.\n" << LOGGER_END();
      agxAssert(size >= numFilled());
      size_t oldNumBuckets = numBuckets();
      Bucket *oldBuckets = buckets();

      // Do not set numBuckets directly, since allocateBytes might Throw an exception,
      // after which we want to be able to recover.
      const size_t desiredNumBuckets = (size_t)agx::nextPrime((int)size);
      const size_t desiredNumBytes = desiredNumBuckets * sizeof(Bucket);

      m_buffer = m_allocator.allocateBytes(desiredNumBytes);
      for (size_t i = 0; i < desiredNumBuckets; ++i)
        ::new (&(buckets()[i])) Bucket();

      numBuckets() = desiredNumBuckets;

      numFilled() = 0;
      m_maxProbeLength = 0;

      for (size_t i = 0; i < oldNumBuckets; i++)
      {
        Bucket& bucket = oldBuckets[i];

        if (bucket.isValid())
          this->insert(bucket.first, bucket.second);
      }

      for (size_t i = 0; i < oldNumBuckets; ++i)
        oldBuckets[i].~Bucket();

      m_allocator.deallocateBytes((void *)oldBuckets, oldNumBuckets * sizeof(Bucket));
    }


    template <typename T2>
    AGX_FORCE_INLINE size_t findBucket(const T2& key) const
    {
      if (!buckets())
        return numBuckets();

      UInt32 hashValue = m_hashFunctor(key);
      size_t index = hashValue % numBuckets();
      const Bucket *bucket = &buckets()[index];

      /* First test is outside for-loop, hopefully getting better branch prediction */
      if (bucket->isValid() && hashKeyEqual(bucket->first, key))
        return index;


      for (size_t offset = 1; bucket->isActive() && offset <= m_maxProbeLength; offset++)
      {
        index  = (hashValue + offset * offset) % numBuckets();
        bucket = &buckets()[index];

        if (bucket->isValid() && hashKeyEqual(bucket->first, key))
          return index;
      }

      /* Not found */
      return numBuckets();
    }

    bool eraseBucket(size_t index)
    {
      if (index >= numBuckets())
        return false;

      Bucket *bucket = &buckets()[index];
      if (!bucket->isValid())
        return false;

      numFilled()--;

      // Replace bucket by a new one (otherwise, if key or data is ref_ptr their ref count won't decrease)
      bucket->first.~KeyT();
      bucket->second.~DataT();
      ::new (bucket) Bucket();

      // Bucket does not become inactive since it might be in the middle of a probing chain
      bucket->state = Bucket::ACTIVE;

      return true;
    }

  private:
    Real32 m_smoothingAverage;
    UInt16 m_maxProbeLength;
    AllocatorT m_allocator;
    HashT m_hashFunctor;
  };



  /**
  Inheritance with partial specialization due to bug with ref_ptr containers. And partial specialization only works on
  classes, not methods (which would have been a cleaner solution).

  See full comment in HashSet.h
  */

  template <typename KeyT, typename DataT, typename HashT = HashFn<KeyT>, typename AllocatorT = ByteAllocator>
  class QuadraticProbingHashTable : public QuadraticProbingHashTableImplementation<KeyT, DataT, HashT, AllocatorT>
  {
  public:
    typedef QuadraticProbingHashTableImplementation<KeyT, DataT, HashT, AllocatorT> Implementation;

    AGX_FORCE_INLINE QuadraticProbingHashTable(const AllocatorT& allocator = AllocatorT()) : Implementation(allocator)
    {}

  };

  template <typename KeyT, typename DataT, typename HashT, typename AllocatorT>
  class QuadraticProbingHashTable< agx::ref_ptr<KeyT>, DataT, HashT, AllocatorT > : public QuadraticProbingHashTableImplementation<agx::ref_ptr<KeyT>, DataT, HashT, AllocatorT>
  {
  public:
    typedef QuadraticProbingHashTableImplementation<agx::ref_ptr<KeyT>, DataT, HashT, AllocatorT> Implementation;
    typedef typename Implementation::iterator iterator;
    typedef typename Implementation::const_iterator const_iterator;

    AGX_FORCE_INLINE QuadraticProbingHashTable(const AllocatorT& allocator = AllocatorT()) : Implementation(allocator)
    {}

    using Implementation::contains;
    AGX_FORCE_INLINE bool contains(const KeyT *key) const
    {
      return this->findBucket(key) != this->numBuckets();
    }

    using Implementation::find;
    AGX_FORCE_INLINE iterator find(const KeyT *key)
    {
      return iterator(this, this->findBucket(key));
    }

    AGX_FORCE_INLINE const_iterator find(const KeyT *key) const
    {
      return const_iterator(this, this->findBucket(key));
    }

    using Implementation::erase;
    AGX_FORCE_INLINE bool erase(const KeyT *key)
    {
      return this->eraseBucket(this->findBucket(key));
    }

  };

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
