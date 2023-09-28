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

#ifndef AGX_MEMORYPOOL_H
#define AGX_MEMORYPOOL_H

#include <agx/config.h>
#include <cstring>
#include <agx/Allocator.h>
#include <agx/Vector.h>
#include <limits>

namespace agx { template <typename T> class MemoryPool; }


// template <typename T>
// inline void *operator new(size_t size, agx::MemoryPool<T>& pool)
// {
//   agxAssert(size == sizeof(T));
//   return pool.allocate();
// }


namespace agx
{
  #define AGX_CHUNK_ALIGNMENT 128
  #define AGX_CHUNK_GROW_FACTOR 1.5

  /**
  Batch allocator.
  */
  class AGXCORE_EXPORT BatchAllocator : public ByteAllocator
  {
  public:
    BatchAllocator(const char *name = "BatchAllocator", uint32_t initialChunkSize = 64);
    ~BatchAllocator();
    BatchAllocator(const BatchAllocator& other);
    BatchAllocator& operator=(const BatchAllocator& other);

    void *allocate(uint32_t numBytes);
    void *allocate(uint32_t numBytes, uint32_t alignment);
    void deallocate(void *buffer);
    void clear();

    size_t size();

  protected:
    void createChunk(uint32_t size);
    size_t deleteAllChunks();

    struct Chunk
    {
      char *buffer;
      char *end;
      char *head;
      Chunk *next;
    };

    Chunk *m_firstChunk;
    uint32_t m_chunkSize;
  };


  class AGXCORE_EXPORT AbstractMemoryPool
  {
  public:
    virtual void deallocateVirtual(void *ptr) = 0;
    virtual ~AbstractMemoryPool() {}
  };

  /**
  Memory pool.
  */
  template <typename T>
  class MemoryPool : public BatchAllocator, public AbstractMemoryPool
  {
  public:
    /**
    Constructor.
    */
    MemoryPool(const char *name = "MemoryPool", uint32_t initialChunkSize = 64);

    /**
    Destructor
    */
    virtual ~MemoryPool();


    /**
    Allocates one element, no constructor called.
    */
    T *allocate();

    /**
    Deallocates one element, no destructor called.
    */
    void deallocate(T *element);

    /**
    Allocates an element and calls its default constructor.
    */
    T *create();

    /**
    Deallocates an element and calls its destructor.
    */
    void destroy(T *element);


    /**
    Clears the memory pool and runs the destructor on every allocated object.
    */
    void clear();

    /**
    Clears the memory pool without calling destructors.
    */
    void clearFast();

    // void *allocateArray(uint32_t numElements);
    // void deallocateArray(T *buffer);


    virtual void deallocateVirtual(void *ptr);

    size_t size() const { return m_size; }

  private:
    size_t m_size;
    T *m_freeHead;
  };



  //---------------------------------------------------------------

  template <typename T>
  class MemoryPoolOld : public ByteAllocator
  {
  public:
    MemoryPoolOld(size_t initialSize = 0, size_t chunkSize = 64);
    ~MemoryPoolOld();

    /**
    Clear all memory allocated by this MemoryPoolOld
    */
    void clear();

    /**
    Set the new size of the chunk. Chunks created after this call will have this size.
    */
    void setChunkSize( size_t size );

    /// \return the current chunkSize
    size_t getChunkSize() const;

    /**
    \return an element of type T from the pool.
    */
    T *getElement();

    /**
    \return an element of type T from the pool. (same as getElement()
    */
    AGX_FORCE_INLINE T *acquire() { return getElement(); }

    /**
    Get an element of type T, if we have ran out, then allocate a new chunks.
    \return a pointer to an unused element of type T.
    */
    T *getElement(const T& initializer);


    /// \return number of elements currently in use
    size_t size() const;

    ///\ return the total number of elements allocated
    size_t capacity() const;

    /// \return the total memory allocated in the chunks
    size_t memUsed() const;

    /**
    Return all elements to container.
    */
    void flush();

    /**
    Free memory not in use. Based on the average allocation size (measured at each call to flush), only
    memory enough to store the average number of elements will be left.
    */
    void garbageCollect();


    /**
    Set the number of calls to flush which will trigger a call to garbageCollect
    \param n - Number of calls to flush to get garbageCollect, default: -1 (never)
    */
    void setAutoGarbageCollect( int n );

    /// \return number of calls to flush which will trigger a garbageCollect
    int getAutoGarbageCollect( ) const;

    /**
    Set to true if the constructor of the elements should be called BEFORE they are returned in getElement();
    */
    void setEnableCallConstructor( bool flag );

    /**
    Set the largest number of bytes a call to new will be.
    */
    void setMaxAllocationSize( size_t numBytes );

  private:
    void createChunk();
    void deleteChunk();

  private:
    typedef Vector<T *> ChunkVector;
    ChunkVector m_chunks;
    size_t m_numUsed;
    size_t m_chunkSize;
    size_t m_numFlushes;
    size_t m_lastAllocatedNumElements;
    float m_averageNumElements;
    int m_flushCount;
    bool m_callConstructor;
    size_t m_maxAllocationSize;
    size_t m_capacity;
    size_t m_chunkIndex;
    size_t m_elementIndex;

    typedef agx::Vector<size_t> SizeVector;
    SizeVector m_sizeVector;
  };

  /* Implementation */

  template <typename T>
  MemoryPoolOld<T>::MemoryPoolOld(size_t initialSize, size_t chunkSize) :
    m_numUsed(0), m_chunkSize(chunkSize), m_numFlushes(0), m_lastAllocatedNumElements(0), m_averageNumElements(0),
    m_flushCount( std::numeric_limits<int>::max() ), m_callConstructor(true),
    m_maxAllocationSize( std::numeric_limits<size_t>::max() ), m_capacity( 0 ), m_chunkIndex(0), m_elementIndex(0)
  {
    size_t numChunks = initialSize ? (initialSize / chunkSize) + 1 : 0;
    for (size_t i = 0; i < numChunks; i++)
      this->createChunk();
  }

  template <typename T>
  MemoryPoolOld<T>::~MemoryPoolOld()
  {
    clear();
  }

  template <typename T>
  void MemoryPoolOld<T>::setChunkSize( size_t size )
  {
    m_chunkSize = size;
  }

  template <typename T>
  size_t MemoryPoolOld<T>::getChunkSize( ) const
  {
    return m_chunkSize;
  }

  template <typename T>
  size_t MemoryPoolOld<T>::memUsed() const
  {
    return sizeof(T)*capacity();
  }


  template <typename T>
  AGX_FORCE_INLINE size_t MemoryPoolOld<T>::size() const
  {
    return m_numUsed;
  }

  template <typename T>
  inline void MemoryPoolOld<T>::setEnableCallConstructor( bool flag )
  {
    m_callConstructor = flag;
  }

  template <typename T>
  inline void MemoryPoolOld<T>::setMaxAllocationSize(size_t numBytes )
  {
    m_maxAllocationSize = numBytes;
  }


  template <typename T>
  AGX_FORCE_INLINE size_t MemoryPoolOld<T>::capacity() const
  {
    return m_capacity;
  }

  template <typename T>
  inline void MemoryPoolOld<T>::setAutoGarbageCollect( int n )
  {
    m_flushCount = n;
  }

  template <typename T>
  AGX_FORCE_INLINE int MemoryPoolOld<T>::getAutoGarbageCollect( ) const
  {
    return m_flushCount;
  }

  template <typename T>
  inline T *MemoryPoolOld<T>::getElement()
  {
    m_numUsed++;

    if ( m_numUsed > this->capacity() )
      this->createChunk();
    else if ( m_elementIndex >= m_sizeVector[ m_chunkIndex] )
    {
      m_elementIndex =0;
      m_chunkIndex++;
    }


    T* element = m_chunks[m_chunkIndex]+m_elementIndex;
    m_elementIndex++;

    if (m_callConstructor)
      ::new((void *)element) T();

    return element;
  }


  template <typename T>
  inline T *MemoryPoolOld<T>::getElement(const T& initializer)
  {
    // size_t numElementsInChunk = m_sizeVector[m_chunkIndex];

    m_numUsed++;
    m_elementIndex++;

    if (m_numUsed > this->capacity() || m_elementIndex >= m_sizeVector[ m_chunkIndex] )
      this->createChunk();


    T* element = m_chunks[m_chunkIndex]+m_elementIndex;

    if (m_callConstructor)
      ::new((void *)element) T(initializer);

    return element;
  }


  template <typename T>
  void MemoryPoolOld<T>::clear()
  {
    // Free ALL memory used
    while (!m_chunks.empty())
      this->deleteChunk();

    m_capacity = 0;
    m_chunkIndex=0;
    m_numUsed = 0;
    m_elementIndex=0;
    m_lastAllocatedNumElements=0;
    m_sizeVector.clear(Container::SHRINK_BUFFER);
  }

  template <typename T>
  void MemoryPoolOld<T>::createChunk()
  {
    // allocate..
    T* ptr = static_cast<T*>( allocateBytes( sizeof(T) * m_chunkSize ) );

    // .. and call default constructors
    for (size_t i=0; i < m_chunkSize; ++i)
      ::new( (void*) &ptr[i]) T();

    // Create one chunk, increase capacity
    m_chunks.push_back(ptr);
    m_sizeVector.push_back( m_chunkSize );
    m_capacity += m_chunkSize;
    m_chunkIndex = m_chunks.size()-1;
    m_elementIndex=0;
  }

  template <typename T>
  void MemoryPoolOld<T>::deleteChunk()
  {
    T* ptr = m_chunks.back();

    for (size_t i=0; i < m_sizeVector.back(); ++i)
      ptr[i].~T();

    deallocateBytes( ptr );

    // reduce capacity
    m_capacity -= m_sizeVector.back();

    m_sizeVector.pop_back();

    m_chunks.pop_back();
    m_chunkIndex = m_chunks.size();

    if (m_chunks.size())
      m_chunkIndex = m_chunks.size()-1;

    m_elementIndex=0;
  }

  template <typename T>
  void MemoryPoolOld<T>::flush()
  {
    // Store the average number of elements used
    m_lastAllocatedNumElements = m_numUsed;
    m_averageNumElements = m_averageNumElements*0.3f + 0.7f*m_lastAllocatedNumElements;

    m_numFlushes++;
    m_numUsed=0;
    m_chunkIndex=0;
    m_elementIndex=0;

    // Should we do a garbageCollect now?
    if (m_flushCount > 0 && (int)m_numFlushes > m_flushCount)
      garbageCollect();
  }

  template <typename T>
  void MemoryPoolOld<T>::garbageCollect() {

    // If no flushes are done, we cannot determine the average size
    if (!m_numFlushes || !m_chunks.size())
      return;

    // Remove any chunks exceeding the average used size
    size_t sum=0;
    size_t endIdx=m_sizeVector.size()-1;

    for(size_t i=0; i < m_sizeVector.size(); i++)
    {
      sum += m_sizeVector[i];
      if (sum >= m_averageNumElements) {
        endIdx = i;
        break;
      }
    }

    size_t n=m_chunks.size();
    for(size_t i=n-1; i > endIdx; i--)
      deleteChunk();

    m_sizeVector.resize( m_chunks.size() );

    m_capacity = sum;

    m_chunkIndex=0;
    m_numUsed = 0;
    m_elementIndex=0;
  }



  //---------------------------------------------------------------
















  /* Implementation */
  inline void *BatchAllocator::allocate(uint32_t numBytes)
  {
    if (m_firstChunk->head + numBytes > m_firstChunk->end)
      this->createChunk( std::max(m_chunkSize, numBytes) );

    void* ret = m_firstChunk->head;
    m_firstChunk->head += numBytes;

    return ret;
  }

  inline void *BatchAllocator::allocate(uint32_t numBytes, uint32_t alignment)
  {
    agxAssert(isPowerOfTwo(alignment));
    // m_firstChunk->head += (alignment-1) & (alignment - ((size_t)m_firstChunk->head & (alignment-1)));
    m_firstChunk->head += (alignment-1);
    m_firstChunk->head = (char *) ((size_t)m_firstChunk->head & ~((size_t)alignment-1));
    return this->allocate(numBytes);
  }

  inline void BatchAllocator::deallocate(void *buffer)
  {
    agxVerify(buffer); // Just for the unused variable warning.
    /* We can only reuse memory if deallocating the last allocated block */
    /*
    agxAssert(buffer);

    if (buffer == m_prev && m_prev >= m_firstChunk->buffer && m_prev < m_firstChunk->end)
    {
      m_head = m_prev;
      m_prev = 0;
    }
    */
  }



  inline size_t BatchAllocator::size()
  {
    size_t totSize = 0;

    Chunk *chunk = m_firstChunk;
    while (chunk)
    {
      totSize += chunk->end - chunk->buffer;
      chunk = chunk->next;
    }

    return totSize;
  }





  template <typename T>
  MemoryPool<T>::MemoryPool(const char *name, uint32_t initialChunkSize)
    : BatchAllocator(name, (uint32_t)(initialChunkSize * sizeof(T))), m_size(0), m_freeHead(0)
  {
    #ifdef AGX_DEBUG
    size_t warningHack = 0;
    #endif
    agxAssert( sizeof(void*) <= (sizeof(T)+warningHack) );
  }

  template <typename T>
  MemoryPool<T>::~MemoryPool()
  {
    this->clear();
  }



  template <typename T>
  inline T *MemoryPool<T>::allocate()
  {
    m_size++;
    if ( m_freeHead != 0 )
    {
      T* ret = m_freeHead;
      m_freeHead = (T*) ( *((void**)( m_freeHead )) );
      return ret;
    }

    return static_cast<T *>( BatchAllocator::allocate(sizeof(T)) );
  }

  template <typename T>
  inline T *MemoryPool<T>::create()
  {
    void *element = this->allocate();
    return ::new(element) T();
  }


  template <typename T>
  inline void MemoryPool<T>::deallocate(T *element)
  {
    m_size--;
    // store previous head
    *((void**)element) = m_freeHead;
    m_freeHead = element;
  }


  template <typename T>
  inline void MemoryPool<T>::destroy(T *element)
  {
    element->~T();
    this->deallocate(element);
  }

  template <typename T>
  void MemoryPool<T>::deallocateVirtual(void *ptr)
  {
    this->deallocate(static_cast<T *>(ptr));
  }

  template <typename T>
  void MemoryPool<T>::clear()
  {
    m_size = 0;
    // If we have free elements, there is a linked list within the chunks to the free slots
    if ( this->m_freeHead )
    {
      agx::Vector<size_t> chunkSizes;

      char *bitmap;
      chunkSizes.push_back( 0 );

      for( BatchAllocator::Chunk *c = this->m_firstChunk; c != 0; c= c->next )
      {
        chunkSizes.push_back( chunkSizes.back() + ((c->end - c->buffer) / sizeof(T)) );
      }

      // keep a bitmap, to avoid / and % calculations, use 1 byte/element
      bitmap = (char *)this->allocateBytes( chunkSizes.back() );
      memset( bitmap, 0, chunkSizes.back() );

      // traverse linked list..
      for ( void* l = this->m_freeHead; l != 0; l = *((void**)l) )
      {
        // find chunk,
        int idx = 0;
        BatchAllocator::Chunk *tmp = this->m_firstChunk;
        while ( tmp && !( tmp->buffer <= l && l <= tmp->head ) )
        {
          ++idx;
          tmp = tmp->next;
        }

        agxAssert(tmp);
        // tag in bitmap
        if (tmp) {
          bitmap[ chunkSizes[idx] + ((char*)l - tmp->buffer) / sizeof(T)  ] = 1;
        }
      }

      // for every chunk,
      // check bitmap and call d-tors.
      int idx = 0;
      for ( BatchAllocator::Chunk* tmp = this->m_firstChunk; tmp!=0;  tmp=tmp->next, ++idx )
      {
        size_t end = (tmp->head - tmp->buffer) / sizeof(T);
        T* buff = reinterpret_cast<T*>(tmp->buffer);

        for (size_t e = 0; e < end; ++e)
          if ( !bitmap[ chunkSizes[idx] + e] )
            buff[e].~T();
      }


      this->deallocateBytes( bitmap );
    }
    else
    {
      //
      // All allocated elements are in use:
      //   for every chunk, for every element, run destructor

      BatchAllocator::Chunk *c = this->m_firstChunk;

      while ( c )
      {
        T* element = reinterpret_cast<T *>(c->buffer);
        while ( (char *)element < c->head )
        {
          element->~T();
          ++element;
        }
        c = c->next;
      }
    }

    this->m_freeHead = 0;
    BatchAllocator::clear();
  }


  template <typename T>
  void MemoryPool<T>::clearFast()
  {
    m_size = 0;
    ///\todo
    ///\todo
    agxAssert1(!this->m_freeHead, "TODO");
    this->m_freeHead = 0;
    BatchAllocator::clear();
  }


}


#endif /* _AGX_MEMORYPOOL_H_ */

