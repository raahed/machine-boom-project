/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_CONTAINER_H
#define AGX_CONTAINER_H

#include <agx/macros.h>
#include <agx/agxCore_export.h>

namespace agxData { class Buffer; }

namespace agx
{
  class ThreadScopeAllocator;


  /**
  The Container is the base class for several of the container classes proided
  by AGX, such as agx::Vector and agx::HashSet.
  */
  class AGXCORE_EXPORT Container
  {
  public:

    /**
    agxData::Values from this enumeration is passed to the subclasses' 'clear' method in
    order to control what should become of the container's memory.
    */
    enum ClearPolicy
    {
      SHRINK_BUFFER,           //!< Buffer is deallocated and replaced by an newly allocated empty buffer.
      SHRINK_BUFFER_AVERAGED,  //!< Buffer is shrunk if a smoothing average (which is updated each clear call) goes below a threshold.
      MAINTAIN_BUFFER          //!< Buffer is maintained (normal stl behavior).
    };

  public:

    /// \return The number of elements in the container.
    size_t size() const;

    /**
    Returns the size of the memory are used by the container to store its
    elements. Some types of containers, e.g. the agx::Vector, will not do a
    reallocation until the size of the container reaches the capacity. However,
    this is not the case for e.g., agx::HashTable, which may perform a
    reallocation earlier.

    \return The size of the allocated memory area, in number of elements.
    */
    size_t capacity() const;

    /// \return True if the container contains no elements, i.e., its size is zero. False otherwise.
    bool empty() const;

    /// \return Pointer to the container's memory area.
    void* ptr();
    const void* ptr() const;


  public:

    DOXYGEN_START_INTERNAL_BLOCK()

    /**
    Only use if you know what you are doing...
    */
    void _setBuffer(void* buffer) { m_buffer = buffer; }
    void _setCapacity(size_t capacity) { m_capacity = capacity; }
    void _setSize(size_t size) { m_size = size; }
    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    Container();
    Container(Container&& other);
    Container(const Container& other);
    ~Container();

  protected:
    friend class agxData::Buffer;
    friend class Thread;
    size_t m_size;
    size_t m_capacity;
    void* m_buffer;
  };



  /* Implementation */
  AGX_FORCE_INLINE Container::Container() : m_size(0), m_capacity(0), m_buffer(nullptr) {}
  AGX_FORCE_INLINE Container::Container(const Container& other)
    : m_size(0), m_capacity(0), m_buffer(nullptr)
  {
    m_size = other.m_size;
    m_buffer = other.m_buffer;
    m_capacity = other.m_capacity;
  }

  AGX_FORCE_INLINE Container::Container(Container&& other):
    m_size(0),
    m_capacity(0),
    m_buffer(nullptr)
  {
    // Move values
    m_buffer = other.m_buffer;
    m_capacity = other.m_capacity;
    m_size = other.m_size;

    // Set default values to the rvalue reference.
    other.m_buffer = nullptr;
    other.m_capacity = 0;
    other.m_size = 0;
  }

  AGX_FORCE_INLINE Container::~Container()
  {
    m_buffer = nullptr;
    m_size = 0;
    m_capacity = 0;
  }

  AGX_FORCE_INLINE size_t Container::size() const { return m_size; }
  AGX_FORCE_INLINE size_t Container::capacity() const { return m_capacity; }
  AGX_FORCE_INLINE bool Container::empty() const { return m_size == 0; }
  AGX_FORCE_INLINE void *Container::ptr() { return m_buffer; }
  AGX_FORCE_INLINE const void *Container::ptr() const { return m_buffer; }
}


#endif /* _AGX_CONTAINER_H_ */
