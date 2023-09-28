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


#ifndef AGXDATA_VECTOR_H
#define AGXDATA_VECTOR_H

#include <agxData/Array.h>
#include <agxData/Buffer.h>

namespace agxData
{
  template <typename T>
  class Vector : public Array<T>
  {
  public:
    typedef T                                  Type;
    typedef T                                  value_type;
    typedef T*                                 pointer;
    typedef const T*                           const_pointer;
    typedef T&                                 reference;
    typedef const T&                           const_reference;
    typedef size_t                             size_type;
    typedef T*                                 iterator;
    typedef const T*                           const_iterator;
    typedef std::random_access_iterator_tag    iterator_category;
    typedef ptrdiff_t                          difference_type;

  public:
    explicit Vector();
    explicit Vector(agxData::Buffer* buffer);
    ~Vector();

    /**
    Commit all changes.
    */
    void commit();

    /**
    Resize the vector, which then enables direct addressing using the bracket '[]' operator.
    */
    void resize(size_t size, const T& value = T());

    /**
    Reserve capacity in the vector.
    */
    void reserve(size_t size);

    /**
    Resize using a increment.
    */
    T* increment(size_t numElements = 1, const T& value = T());
    void decrement(size_t numElements = 1);


    /**
    Append a new data element to the vector.
    */
    template <typename T2>
    void push_back(const T2& value);

    /**
    Remove the back element.
    */
    void pop_back();


    /**
    STL erase functionality.
    */
    iterator erase(iterator position);
    iterator erase(iterator start, iterator end);

    /**
    Erase elements using indices instead of iterators.
    */
    void erase(size_t index);
    void erase(size_t start, size_t end);

    /**
    Fast erase, replacing the erased element with the last element.
    */
    iterator eraseFast(const_iterator position);
    void eraseFast(size_t index);

    /**
    Remove all elements, optionally with maintained buffer allocation.
    */
    void clear();

    /**
    Find and erase an element.
    \param element The element to search for
    \param searchMultiple Set to true if search should continue after element is found, otherwise the search terminates on first match.
    \return true if the element was found, else false
    */
    template <typename T2>
    bool findAndErase(const T2& element, bool searchMultiple = false);

    iterator insert(iterator position, const T& value);
    void insert(size_t index, const T& value);

    template <typename InputIterator>
    void insert(iterator it, InputIterator first, InputIterator last);

  private:
    void reallocate(size_t size);
  };

  typedef Vector<agx::UInt32> GenericVector;



  /* Implementation */

  template <typename T>
  AGX_FORCE_INLINE Vector<T>::Vector()
  {
  }

  template <typename T>
  AGX_FORCE_INLINE Vector<T>::Vector(agxData::Buffer* buffer) : Array<T>(buffer)
  {
    agxVerify(!buffer->getStorage());
  }

  template <typename T>
  AGX_FORCE_INLINE Vector<T>::~Vector()
  {
    this->commit();
  }

  template <typename T>
  AGX_FORCE_INLINE void Vector<T>::commit()
  {
    agxAssert(this->buffer());
    size_t size = this->buffer()->size();
    size_t oldSize = this->buffer()->sizeParameter->get();

    if (oldSize != size)
    {
      this->buffer()->sizeParameter->updateEvent.removeCallback(&this->buffer()->m_resizeCallback);
      this->buffer()->sizeParameter->set(size);
      this->buffer()->sizeParameter->updateEvent.addCallbackFirst(&this->buffer()->m_resizeCallback);

      this->buffer()->m_eventDispatch.triggerResizeEvent(this->buffer(), (agx::Index)size, (agx::Index)oldSize);
    }
  }


  template <typename T>
  AGX_FORCE_INLINE void Vector<T>::resize(size_t size, const T& value)
  {
    agxAssert(this->buffer());

    if (size > this->buffer()->m_size)
    {
      this->increment(size - this->buffer()->m_size, value);
    }
    else
    {
      if (size < (size_t)((double)this->buffer()->m_capacity * AGX_VECTOR_SHRINK_THRESHOLD))
        this->reallocate(size);

      this->destroyElements(size, this->buffer()->m_size);
    }
  }

  template <typename T>
  AGX_FORCE_INLINE void Vector<T>::reallocate(size_t size)
  {
    this->buffer()->reserve(std::max((size_t)4, size));
    this->sync();
  }


  template <typename T>
  AGX_FORCE_INLINE void Vector<T>::reserve(size_t size)
  {
    agxAssert(this->buffer());
    this->reallocate(size);
  }

  template <typename T>
  AGX_FORCE_INLINE T* Vector<T>::increment(size_t numElements, const T& value)
  {
    agxAssert(this->buffer());

    size_t currentSize = this->buffer()->m_size;
    size_t newSize = currentSize + numElements;
    if (newSize > this->buffer()->m_capacity)
      this->reallocate((size_t)((agx::Real)newSize * Buffer::CAPACITY_MULTIPLIER));

    this->buffer()->m_size = newSize;
    this->range().end() = newSize;

    for(size_t i = currentSize; i < newSize; ++i)
      ::new((void *)(this->ptr() + i)) T(value);

    return this->ptr() + currentSize;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vector<T>::decrement(size_t numElements)
  {
    agxAssert(this->buffer());
    agxAssert(this->buffer()->size() >= numElements);

    size_t newSize = this->buffer()->m_size - numElements;
    for (size_t i = newSize; i < this->buffer()->m_size; ++i)
      this->ptr()[i].~T();

    this->range().end() = newSize;
    this->buffer()->m_size = newSize;
  }

  template <typename T> template <typename T2>
  AGX_FORCE_INLINE void Vector<T>::push_back(const T2& value)
  {
    agxAssert(this->buffer());

    if (this->buffer()->m_size == this->buffer()->m_capacity)
      this->reallocate((size_t)((agx::Real)this->buffer()->m_size * Buffer::CAPACITY_MULTIPLIER));

    this->buffer()->m_size++;
    this->range().end()++;
    ::new((void *)&this->back()) T(value);
  }

  template <typename T>
  AGX_FORCE_INLINE void Vector<T>::pop_back()
  {
    agxAssert(this->buffer());
    agxAssert(!this->buffer()->empty());

    this->back().~T();
    this->range().end()--;
    this->buffer()->m_size--;
  }

  template <typename T>
  AGX_FORCE_INLINE typename Vector<T>::iterator Vector<T>::erase(iterator position)
  {
    agxAssert(this->buffer());

    size_t index = position - this->begin();
    this->erase(index);
    return this->begin() + index;
  }

  template <typename T>
  AGX_FORCE_INLINE typename Vector<T>::iterator Vector<T>::erase(iterator start, iterator end)
  {
    agxAssert(this->buffer());

    size_t startIndex = start - this->begin();
    size_t endIndex = end - this->begin();
    this->erase(startIndex, endIndex);

    return this->begin() + endIndex;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vector<T>::erase(size_t index)
  {
    agxAssert(this->buffer());

    for (size_t i = index + 1; i < this->buffer()->m_size; i++)
      this->ptr()[i-1] = this->ptr()[i];

    /* Run destructor on trailing elements */
    this->ptr()[this->buffer()->m_size - 1].~T();

    this->buffer()->m_size--;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vector<T>::erase(size_t start, size_t end)
  {
    agxAssert(this->buffer());

    agxAssert1(start < end && end <= this->buffer()->m_size, "Erase bounds are not with array bounds!");
    size_t numElements = end - start;

    // TODO We should first run destructor and constructor on element before copying? Again we assume copy operator semantics to be 'proper'
    for (size_t i = end; i < this->buffer()->m_size; i++)
      this->ptr()[i-numElements] = this->ptr()[i];

    /* Run destructor on trailing elements */
    for(size_t i = this->buffer()->m_size - numElements; i < this->buffer()->m_size; ++i)
      this->ptr()[i].~T();

    this->buffer()->m_size -= numElements;
  }

  template <typename T>
  AGX_FORCE_INLINE typename Vector<T>::iterator Vector<T>::eraseFast(const_iterator position)
  {
    agxAssert(this->buffer());

    size_t index = position - this->begin();
    this->eraseFast(index);
    return this->begin() + index;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vector<T>::eraseFast(size_t index)
  {
    agxAssert(this->buffer());
    agxAssertN(index < this->buffer()->m_size, "Array index %llu is out of bounds, current size is %llu", (long long unsigned)index, (long long unsigned)this->buffer()->m_size);

    this->ptr()[index] = this->ptr()[this->buffer()->m_size-1];
    this->ptr()[this->buffer()->m_size - 1].~T();
    this->buffer()->m_size--;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vector<T>::clear()
  {
    agxAssert(this->buffer());
    this->decrement(this->buffer()->m_size);
  }

  template <typename T> template <typename T2>
  AGX_FORCE_INLINE bool Vector<T>::findAndErase(const T2& element, bool searchMultiple)
  {
    agxAssert(this->buffer());

    bool found = false;

    for (size_t i = 0; i < this->buffer()->m_size; i++)
    {
      if (this->ptr()[i] == element)
      {
        found = true;
        this->erase(i);

        if (!searchMultiple)
          return true;

        i--;
      }
    }

    return found;
  }

  template <typename T>
  AGX_FORCE_INLINE typename Vector<T>::iterator Vector<T>::insert(iterator position, const T& value)
  {
    agxAssert(this->buffer());

    size_t index = position - this->begin();
    this->insert(index, value);
    return this->begin() + index;
  }

  template <typename T>
  AGX_FORCE_INLINE void Vector<T>::insert(size_t index, const T& value)
  {
    agxAssert(this->buffer());

    agxAssertN(index <= this->buffer()->m_size, "Insert with invalid index %llu, size is %llu", (long long unsigned)index, (long long unsigned)this->buffer()->m_size);

    if (this->buffer()->m_size == this->buffer()->m_capacity)
      this->reallocate((size_t)((agx::Real)this->buffer()->m_size * Buffer::CAPACITY_MULTIPLIER));

    for (size_t i = this->buffer()->m_size; i > index; i--) {
      ::new((void *)&this->ptr()[i]) T(this->ptr()[i-1]);
      this->ptr()[i-1].~T();
    }

    ::new((void *)&this->ptr()[index]) T(value);
    this->buffer()->m_size++;
  }

  template <typename T> template <typename InputIterator>
  AGX_FORCE_INLINE void Vector<T>::insert(iterator it, InputIterator first, InputIterator last)
  {
    agxAssert(this->buffer());

    size_t numElements = std::distance( first, last );

    if (numElements == 0)
      return;

    size_t index = it - this->begin();

    if (this->buffer()->m_size + numElements > this->buffer()->m_capacity)
      this->reallocate((size_t)((agx::Real)(this->buffer()->m_size + numElements) * Buffer::CAPACITY_MULTIPLIER));

    for (size_t i = this->buffer()->m_size; i > index; i--) {
      ::new((void *)&this->ptr()[i + numElements - 1]) T(this->ptr()[i-1]);
      this->ptr()[i-1].~T();
    }

    for (; first != last; first++)
      ::new((void *)&this->ptr()[index++]) T(*first);

    this->buffer()->m_size += numElements;
  }
}

AGX_TEMPLATED_TYPE_BINDING(agxData::Vector, "Vector")

#endif /* AGXDATA_VECTOR_H */
