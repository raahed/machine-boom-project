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

#ifndef AGX_HASHSET_H
#define AGX_HASHSET_H

#include <agx/agx.h>
#include <agx/ref_ptr.h>

#include <agx/Math.h>

#include <agx/HashFunction.h>
#include <agx/Container.h>
#include <agx/Allocator.h>
#include <agx/Integer.h>

#include <algorithm> // For std::max

#define AGX_DECLARE_HASHSET_TYPES(type)                     \
typedef agx::HashSet<type ## Ref> type ## RefSet;           \
typedef agx::HashSet<type ## Observer> type ## ObserverSet; \
typedef agx::HashSet<type *> type ## PtrSet

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 6011) // Disable warningC6011: dereferencing nullptr pointer
#endif


namespace agx
{
  /**
  Same as hash table, but only containing keys, not key-value pairs.
  */
  template <typename KeyT, typename HashT = HashFn<KeyT>, typename AllocatorT = ByteAllocator>
  class HashSetImplementation : public Container
  {
  public:
      typedef KeyT                                  value_type;
      typedef KeyT*                                 pointer;
      typedef const KeyT*                           const_pointer;
      typedef KeyT&                                 reference;
      typedef const KeyT&                           const_reference;
      typedef size_t                                size_type;
      typedef std::bidirectional_iterator_tag       iterator_category;
      typedef size_t                                difference_type;


  private:
    #define AGX_HASH_SET_MIN_SIZE 1

    #define AGX_HASH_SET_GROW_THRESHOLD 0.8
    #define AGX_HASH_SET_SHRINK_THRESHOLD 0.25
    #define AGX_HASH_SET_GROW_FACTOR 2.0
    #define AGX_HASH_SET_SHRINK_FACTOR 0.5
    #define AGX_HASH_SET_SMOOTHING_FACTOR 0.8
    #define AGX_HASH_SET_MAX_PROBE_LENGTH 10

  public:

    template<typename T>
    class template_iterator
    {
    public:
      typedef T HashSetT;
      typedef typename std::iterator_traits<T>::iterator_category iterator_category;
      typedef typename std::iterator_traits<T>::value_type value_type;
      typedef typename std::iterator_traits<T>::difference_type difference_type;
      typedef typename std::iterator_traits<T>::reference reference;
      typedef typename std::iterator_traits<T>::pointer pointer;

    public:
      /** Constructors */
      AGX_FORCE_INLINE template_iterator() : m_hashSet(0), m_currentIndex(0) {}

      AGX_FORCE_INLINE template_iterator(T* hashSet, size_t startIndex = 0) : m_hashSet(hashSet), m_currentIndex(startIndex-1)
      { this->gotoNextElement(); }

      /** Cast operator from non-const to const */
      AGX_FORCE_INLINE operator template_iterator<const T>() const
      {
        template_iterator<const T> constIt;
        constIt.m_hashSet = m_hashSet;
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

        return template_iterator<T>(m_hashSet, oldIndex);
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

        return template_iterator<T>(m_hashSet, oldIndex);
      }

      AGX_FORCE_INLINE bool operator==(const template_iterator<T>& rhs)
      { return this->m_currentIndex == rhs.m_currentIndex && this->m_hashSet == rhs.m_hashSet; }

      AGX_FORCE_INLINE bool operator!=(const template_iterator<T>& rhs)
      { return this->m_currentIndex != rhs.m_currentIndex || this->m_hashSet != rhs.m_hashSet; }


      AGX_FORCE_INLINE KeyT& operator*() const
      {
        agxAssert1(m_hashSet && m_currentIndex != m_hashSet->numBuckets(), "Can not dereference an invalid bucket!");
        return m_hashSet->buckets()[m_currentIndex].key;
      }

      AGX_FORCE_INLINE KeyT *operator->() const
      { return &(operator*()); }


      AGX_FORCE_INLINE template_iterator<T>& operator =( const template_iterator<T>& copy)
      {
        m_hashSet = copy.m_hashSet;
        m_currentIndex = copy.m_currentIndex;
        return *this;
      }

    private:
      AGX_FORCE_INLINE void gotoNextElement()
      {
        m_currentIndex++;
        while (m_currentIndex < m_hashSet->numBuckets() && !m_hashSet->buckets()[m_currentIndex].isValid())
          m_currentIndex++;
      }

      AGX_FORCE_INLINE void gotoPreviousElement()
      {
        if (m_currentIndex > 0) {
          m_currentIndex--;
          while (m_currentIndex > 0 && !m_hashSet->buckets()[m_currentIndex].isValid())
            m_currentIndex--;
        }
      }
    private:
      friend class HashSetImplementation;
      HashSetT* m_hashSet;
      size_t m_currentIndex;
    };

    /** Iterators */
    typedef template_iterator< HashSetImplementation<KeyT, HashT, AllocatorT> > iterator;
    typedef template_iterator< const HashSetImplementation<KeyT, HashT, AllocatorT> > const_iterator;

    class insert_iterator : public iterator
    {
    public:
      typedef typename iterator::HashSetT HashSetT;

    public:

      AGX_FORCE_INLINE insert_iterator() : iterator(), m_overwrite(false)
      {
      }

      AGX_FORCE_INLINE insert_iterator(HashSetT* hashSet, size_t startIndex, bool overwrite) : iterator(hashSet, startIndex), m_overwrite(overwrite)
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
    AGX_FORCE_INLINE HashSetImplementation(const AllocatorT& allocator = AllocatorT()) : m_smoothingAverage(0), m_maxProbeLength(0), m_allocator(allocator)
    { m_allocator.setContainer(this); }

    /** Copy constructor */
    AGX_FORCE_INLINE HashSetImplementation(const HashSetImplementation<KeyT, HashT, AllocatorT>& other) : m_smoothingAverage(0), m_maxProbeLength(0)
    {
      *this = other;
    }

    /** Assignment operator */
    AGX_FORCE_INLINE HashSetImplementation<KeyT, HashT, AllocatorT>& operator= (const HashSetImplementation<KeyT, HashT, AllocatorT>& other)
    {
      m_allocator = other.m_allocator;
      m_allocator.setContainer(this);

      this->clear(MAINTAIN_BUFFER);
      this->reserve(other.size());

      for (const_iterator it = other.begin(); it != other.end(); ++it)
        this->insert(*it);

      return *this;
    }

    /** Destructor */
    AGX_FORCE_INLINE ~HashSetImplementation()
    {
      for (size_t i = 0; i < numBuckets(); ++i)
        buckets()[i].~Bucket();

      m_allocator.deallocateBytes(m_buffer, m_capacity * sizeof(Bucket));
    }


    /**
    Insert an element, overwriting if key already exists.
    \return An iterator to the inserted key
    */
    insert_iterator insert(const KeyT& key)
    {
      UInt32 hashValue = m_hashFunctor(key);
      Bucket *bucket;
      Bucket *targetBucket = 0;
      size_t offset = 0;

      /* Check if we should grow and rebuild hash table */
      if (numFilled() >= (size_t)((float)numBuckets() * AGX_HASH_SET_GROW_THRESHOLD))
        this->rebuild((size_t)((float)(numBuckets()+1) * AGX_HASH_SET_GROW_FACTOR));

      /* Iterate until target bucket is found */
      bool goon=true;
      do
      {
        bucket = &buckets()[(hashValue + offset * offset) % numBuckets()];

        /* Check if inactive bucket or overwriting previous insertion with same key */
        if (!bucket->isActive() || (bucket->isValid() && hashKeyEqual(bucket->key, key)))
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
        if (offset > AGX_HASH_SET_MAX_PROBE_LENGTH)
        {
          this->rebuild((size_t)((float)numBuckets() * AGX_HASH_SET_GROW_FACTOR));
          return this->insert(key);
        }
      } while (goon);

      agxAssert(targetBucket);

      if (offset > m_maxProbeLength)
        m_maxProbeLength = (UInt16)offset;

      targetBucket->key = key;

      /* Check if inserting a new value rather than overwriting an old entry*/
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
    Index operator for accessing elements in the hash, inserting it if not found.
    \param key - The key to look for
    \return reference to the matching data.
    */
    AGX_FORCE_INLINE KeyT& operator[](const KeyT& key )
    {
      iterator it = find( key );
      if (it == end() ){
        it = insert( key );
      }
      return *it;
    }

    AGX_FORCE_INLINE const KeyT& operator[](const KeyT& key ) const
    {
      iterator it = find( key );
      if (it == end() ){
        it = insert( key );
      }
      return *it;
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
    { return iterator(this, numBuckets()); }

    AGX_FORCE_INLINE const_iterator end() const
    { return const_iterator(this, numBuckets()); }


    /**
    Erase an element from the hash set.
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
          buckets()[i] = Bucket();
      }
      else if (policy == SHRINK_BUFFER)
      {
        for (size_t i = 0; i < numBuckets(); ++i)
          buckets()[i].~Bucket();

        m_allocator.deallocateBytes(m_buffer, m_capacity * sizeof(Bucket));
        numBuckets() = agx::nextPrime(AGX_HASH_SET_MIN_SIZE);
        m_buffer = m_allocator.allocateBytes(numBuckets() * sizeof(Bucket));
        for (size_t i = 0; i < numBuckets(); ++i)
          ::new (&(buckets()[i])) Bucket();

        m_smoothingAverage = 0.0;
      }
      else //if (policy == SHRINK_BUFFER_AVERAGED)
      {
        /* Smoothing average is never less than current size */
        if ((Real32)numFilled() > m_smoothingAverage)
          m_smoothingAverage = Real32(numFilled());

        /* Update smoothing average */
        m_smoothingAverage = Real32(AGX_HASH_SET_SMOOTHING_FACTOR * m_smoothingAverage + (1.0-AGX_HASH_SET_SMOOTHING_FACTOR) * (float)numFilled());

        /* Reallocate buffer if the smoothing average is sufficiently below the current capacity */
        if (m_smoothingAverage < AGX_HASH_SET_SHRINK_THRESHOLD * (float)numBuckets() && m_smoothingAverage >= AGX_HASH_SET_MIN_SIZE)
        {
          for (size_t i = 0; i < numBuckets(); ++i)
            buckets()[i].~Bucket();

          m_allocator.deallocateBytes(m_buffer, m_capacity * sizeof(Bucket));
          numBuckets() = agx::nextPrime((int)m_smoothingAverage);
          m_buffer = m_allocator.allocateBytes(numBuckets() * sizeof(Bucket));
          for (size_t i = 0; i < numBuckets(); ++i)
            ::new (&(buckets()[i])) Bucket();
        }
        else
        {
          for (size_t i = 0; i < numBuckets(); i++)
            buckets()[i] = Bucket();
        }
      }


      numFilled() = 0;
      m_maxProbeLength = 0;
    }

    /**
    Reserve capacity in the hash set.
    */
    void reserve(size_t size)
    {
      if ((double)size <= (double)numBuckets() * AGX_HASH_SET_GROW_THRESHOLD)
        return;

      this->rebuild((size_t)((double)size / AGX_HASH_SET_GROW_THRESHOLD));
    }

    /**
    Purge the hash set. Good to use when removing multiple items, since iterators
    are invalidated after any modification. The purger should be a functor class
    which takes the key as parameter and return true if the entry should
    be removed from the hash set.
    */
    template<typename T>
    void purge(T purger)
    {
      for (size_t i = 0; i < numBuckets(); i++)
      {
        Bucket& bucket = buckets()[i];

        if (bucket.isValid() && purger(bucket.key))
        {
          bucket = Bucket();
          bucket.state = Bucket::ACTIVE;
          numFilled()--;
        }
      }

      if (numBuckets() * AGX_HASH_SET_SHRINK_FACTOR >= AGX_HASH_SET_MIN_SIZE && numFilled()/(Real)numBuckets() < AGX_HASH_SET_SHRINK_THRESHOLD)
        this->rebuild( size_t(numFilled() * AGX_HASH_SET_GROW_FACTOR) );
    }

    AllocatorT& allocator() { return m_allocator; }
    const AllocatorT& allocator() const { return m_allocator; }


    DOXYGEN_START_INTERNAL_BLOCK()
    AGX_FORCE_INLINE size_t getNumActiveElements() const { return numBuckets(); }
    DOXYGEN_END_INTERNAL_BLOCK()

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

      AGX_FORCE_INLINE Bucket() : key(), state(INACTIVE)
      {
      }

      KeyT key;
      agx::UInt8 state;

      AGX_FORCE_INLINE bool isActive() const
      { return state >= ACTIVE; }

      AGX_FORCE_INLINE bool isValid() const
      { return state == VALID; }
    };


  protected:
    // Access to Container members
    AGX_FORCE_INLINE size_t& numBuckets() { return m_capacity; }
    AGX_FORCE_INLINE size_t& numFilled() { return m_size; }

    AGX_FORCE_INLINE Bucket *buckets() const { return static_cast<Bucket *>(m_buffer); }
    AGX_FORCE_INLINE size_t numBuckets() const { return m_capacity; }
    AGX_FORCE_INLINE size_t numFilled() const { return m_size; }

    /**
    Rebuild, and optionally resize hash set.
    */
    void rebuild(size_t size)
    {
      size = std::max( size, size_t(AGX_HASH_SET_MIN_SIZE) );
      //LOGGER_DEBUG() << "HashTable: Rebuilding and resizing from " << numBuckets() << " to " << agx::nextPrime((int)(numBuckets() * resizeFactor)) << " buckets.\n" << LOGGER_END();
      agxAssert(size >= numFilled());
      const size_t oldNumBuckets = numBuckets();
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
          this->insert(bucket.key);
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
      Bucket *bucket = &buckets()[index];

      /* First test is outside for-loop, hopefully getting better branch prediction */
      if (bucket->isValid() && hashKeyEqual(bucket->key, key))
        return index;


      for (size_t offset = 1; bucket->isActive() && offset <= m_maxProbeLength; offset++)
      {
        index  = (hashValue + offset * offset) % numBuckets();
        bucket = &buckets()[index];

        if (bucket->isValid() && hashKeyEqual(bucket->key, key))
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
      *bucket = Bucket();

      // Bucket does not become inactive since it might be in the middle of a probing chain
      bucket->state = Bucket::ACTIVE;

      /* Check if we should shrink and rebuild hash set */
      // if (numBuckets() * AGX_HASH_SET_SHRINK_FACTOR >= AGX_HASH_SET_MIN_SIZE && numFilled()/(Real)numBuckets() < AGX_HASH_SET_SHRINK_THRESHOLD)
        // this->rebuild(AGX_HASH_SET_SHRINK_FACTOR);

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

  example:

  HashSet<RigidBodyRef> bodySet;
  RigidBody *body = new RigidBody();
  if (bodySet.contains(body)) // <------------- the call to contains will deallocate body due to creation of temporary ref_ptr
  {
    // ...
  }


  Also, even if we have another reference somewhere which will prevent the deallocation, we still create a temporary ref_ptr
  and perform two atomic operations just for a table lookup.

  And note that we can not use the same solution as in Vector, which is to add another template for the type being compared.
  so instead of

  bool contains(const KeyT& key) const

  we would have

  template <typename T2>
  bool contains(const T2& key) const

  but this is not safe for hash tables since another hash function may be called, example:

  HashSet<UInt64> set;
  set.insert(123);
  if (set.contains(UInt32(123))) <--------- would be false due to a different hash function being used
  {
    // ...
  }
  */

  template <typename KeyT, typename HashT = HashFn<KeyT>, typename AllocatorT = ByteAllocator>
  class HashSet : public HashSetImplementation<KeyT, HashT, AllocatorT>
  {
  public:
    typedef HashSetImplementation<KeyT, HashT, AllocatorT> Implementation;

    AGX_FORCE_INLINE HashSet(const AllocatorT& allocator = AllocatorT()) : Implementation(allocator)
    {}
  };

  template <typename KeyT, typename HashT, typename AllocatorT>
  class HashSet< agx::ref_ptr<KeyT>, HashT, AllocatorT > : public HashSetImplementation<agx::ref_ptr<KeyT>, HashT, AllocatorT>
  {
  public:
    typedef HashSetImplementation<agx::ref_ptr<KeyT>, HashT, AllocatorT> Implementation;
    typedef typename Implementation::iterator iterator;
    typedef typename Implementation::const_iterator const_iterator;

    AGX_FORCE_INLINE HashSet(const AllocatorT& allocator = AllocatorT()) : Implementation(allocator)
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

#endif /* _AGX_HASHSET_H_ */
