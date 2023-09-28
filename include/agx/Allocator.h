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

#ifndef AGX_ALLOCATOR_H
#define AGX_ALLOCATOR_H

#include <agx/config/AGX_DEBUG_INTRUSIVE.h>
#include <stdlib.h>
#include <agx/agxCore_export.h>
#include <agx/Math.h>
#include <agx/Integer.h>
#include <agx/debug.h>


#if defined(AGX_DEBUG)
//#define AGX_TRACK_ALLOCATIONS
#endif

#if defined(AGX_TRACK_ALLOCATIONS)

#ifdef _MSC_VER
//Visual Studio does not implement declarations using exception specification. Turn off the warning
# pragma warning(push)
# pragma warning( disable : 4290 )
#endif

/**
Override new, placement new, and delete.
*/
void *operator new(size_t size) throw(std::bad_alloc);
void *operator new(size_t size, void *ptr) throw();
void operator delete(void *ptr) throw();
#endif

// #define agxNew(type) new (Thread::getCurrentThread()->allocate<type>())

namespace agx
{

  /**
  Byte allocator.
  */
  class AGXCORE_EXPORT ByteAllocator
  {
  public:
    ByteAllocator(const char *name = "ByteAllocator");

    ByteAllocator(const ByteAllocator& other);
    ByteAllocator& operator=(const ByteAllocator& other);

    /// \return The name
    const char *getName() const;

    /// Allocate a memory section of specified alignment
    void *allocateBytes(size_t numBytes, size_t alignment = 16);

    /// Deallocate a section previously allocated using allocateBytes method
    void deallocateBytes(void *buffer, size_t numBytes);
    void deallocateBytes(void *buffer);
    AGX_FORCE_INLINE void setContainer(class Container *) {}


    // Allocation and deallocation that isn't tracked.
    static void* standaloneAllocateBytes(size_t numBytes, size_t alignment = 16);
    static void standaloneDeallocateTypes(void* buffer);

    static bool initMemoryFlag; // Set true if allocated memory should be initialized
    static agx::UInt32 allocatedMemoryPattern; // The byte pattern which the initialized memory is given
    static agx::UInt32 deallocatedMemoryPattern; // The byte pattern which the initialized memory is given
    static void initMemory(void *data, size_t numBytes, agx::UInt32 pattern);


    #ifdef AGX_TRACK_ALLOCATIONS
    size_t getNumActiveAllocations() const;
    size_t getTotalActiveAllocationSize() const;
    #endif

  private:
    #if AGX_DEBUG_INTRUSIVE()
    const char *m_name;
    #endif


    #ifdef AGX_TRACK_ALLOCATIONS
    size_t m_numActiveAllocations;
    size_t m_totalActiveAllocationSize;
    #endif
  };


  /**
  Templated allocator.
  */
  template <typename T>
  class Allocator : public ByteAllocator
  {
  public:
    Allocator(const char *name = "Allocator");

    /// Allocate/deallocate raw memory
    T *allocate(size_t numElements);
    void deallocate(T *buffer);

    /* Allocate/deallocate and run constructors/destructors */
    // T *create(size_t numElements);
    // void destroy(T *buffer);
  };






  /**
  Aligned allocator.
  */
  template <typename T>
  class AlignedAllocator : public Allocator<T>
  {
  public:
    AlignedAllocator(const char *name = "AlignedAllocator", agx::UInt32 aligment = 128);

    /* Allocate/deallocate raw memory */
    T *allocate(size_t numElements);
    void deallocate(T *buffer);

    /* Allocate/deallocate and run constructors/destructors */
    //T *create(size_t numElements);
    //void destroy(T *buffer);


  private:
    agx::UInt32 m_alignment;
  };




  /* Implementation */
  #if AGX_DEBUG_INTRUSIVE()
  inline ByteAllocator::ByteAllocator(const char *name)
  : m_name(name)
  #else
  inline ByteAllocator::ByteAllocator(const char * /*name*/)
  #endif
  {
    #ifdef AGX_TRACK_ALLOCATIONS
    m_numActiveAllocations = 0;
    m_totalActiveAllocationSize = 0;
    #endif
  }

  AGX_FORCE_INLINE const char *ByteAllocator::getName() const
  {
    #if AGX_DEBUG_INTRUSIVE()
    return m_name;
    #else
    return "AllocatorNameOnlyAvaiableInDebugBuild";
    #endif
  }

  AGX_FORCE_INLINE void ByteAllocator::deallocateBytes(void *buffer) { this->deallocateBytes(buffer, 0); }


  //-----------------------------------------------------------------------------------------------------

  template <typename T>
  inline Allocator<T>::Allocator(const char *name) : ByteAllocator(name)
  {
  }


  template <typename T>
  AGX_FORCE_INLINE T *Allocator<T>::allocate(size_t numElements)
  {
    return reinterpret_cast<T*>(ByteAllocator::allocateBytes(numElements * sizeof(T)));
  }

  template <typename T>
  AGX_FORCE_INLINE void Allocator<T>::deallocate(T *buffer)
  {
    ByteAllocator::deallocateBytes((void *)buffer);
  }


  //-----------------------------------------------------------------------------------------------------

  template <typename T>
  inline AlignedAllocator<T>::AlignedAllocator(const char *name, agx::UInt32 aligment) : Allocator<T>(name), m_alignment(aligment)
  {
  }


  template <typename T>
  AGX_FORCE_INLINE T *AlignedAllocator<T>::allocate(size_t numElements)
  {
    return reinterpret_cast<T*>(ByteAllocator::allocateBytes(numElements * sizeof(T), m_alignment));
  }

  template <typename T>
  AGX_FORCE_INLINE void AlignedAllocator<T>::deallocate(T *buffer)
  {
    ByteAllocator::deallocateBytes((void *)buffer);
  }

}

#if defined(AGX_TRACK_ALLOCATIONS)
# ifdef _MSC_VER
#   pragma warning(pop)
# endif
#endif

#endif /* _AGX_ALLOCATOR_H_ */
