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

#ifndef AGX_VECTOR_H
#define AGX_VECTOR_H

#include <type_traits>

#include <agx/agx.h>
// #include <agx/Referenced.h>
#include <agx/Allocator.h>
#include <agx/Container.h>
#include <ostream>
#include <cstring>
#include <iterator>

#ifdef __linux__
#include <cstddef>
#endif

#define AGX_DECLARE_VECTOR_TYPES(type)              \
typedef agx::Vector<type ## Ref> type ## RefVector; \
typedef agx::Vector<type ## Observer> type ## ObserverVector; \
typedef agx::VectorPOD<type *> type ## PtrVector

#ifdef _MSC_VER
#pragma warning (push)
# pragma warning( disable: 4345) // behavior change: an object of POD type constructed with an initializer of the form () will be default-initialized
#endif



namespace agx
{
  /**
  Templated vector class.
  */
  template<typename T, typename Allocator = ByteAllocator >
  class Vector : public Container
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
    explicit Vector(const Allocator& allocator = Allocator());
    explicit Vector(size_t size, const T& value = T(), const Allocator& allocator = Allocator());
    explicit Vector(const_iterator first, const_iterator end, const Allocator& allocator = Allocator());
    Vector(std::initializer_list<T> values, const Allocator& allocator = Allocator());

    Vector(const Vector<T, Allocator>& other);
    Vector(Vector<T, Allocator>&& other);

    ~Vector();

    Vector<T, Allocator>& operator= (const Vector<T, Allocator>& other);


    /**
    Compare with other vector, return true if same size and all pairs are equal.
    */
    bool operator == ( const Vector<T, Allocator>& other ) const;
    bool operator != ( const Vector<T, Allocator>& other ) const;


    /**
    Resize the vector, which then enables direct addressing using the bracket '[]' operator.
    */
    void resize(size_t size);
    void resize(size_t size, const T& value);


    /**
    Reduce the capacity of the vector to the actual size (number of elements)
    */
    void shrink_to_fit();

    /**
    Reserve capacity in the vector.
    */
    void reserve(size_t size);


    /**
    Reserve capacity in the vector. Let vector grow in same way as push_back() does.
    */
    void reserveAtLeast(size_t size);


    /**
    Resize using a increment.
    */
    T* increment(size_t numElements = 1);

    /**
    Get access to the internal data buffer.
    */
    T* ptr();
    const T* ptr() const;

    T& operator[] (size_t i) const;
    T& at(size_t index) const;
    T& front() const;
    T& back() const;

    template <typename T2>
    void push_back(const T2& value);

    void push_back(const T& value);

    void push_back(T&& value);

    void pop_back();

    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    reverse_iterator rbegin();
    reverse_iterator rend();
    const_reverse_iterator rbegin() const;
    const_reverse_iterator rend() const;

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
    void clear(ClearPolicy policy = SHRINK_BUFFER_AVERAGED);

    /**
    Test if the vector contains a certain element.
    */
    template <typename T2>
    bool contains(const T2& element) const;

    bool contains(const T& element) const;


    /**
    Find the index to a matching element, return size() if not found.
    */
    template <typename T2>
    size_t find(const T2& element) const;

    /**
    Find and erase an element.
    \param element - Element to search for
    \param searchMultiple Set to true if search should continue after element is found, otherwise the search terminates on first match.
    \return true if the element was found, else false
    */
    template <typename T2>
    bool findAndErase(const T2& element, bool searchMultiple = false);

    iterator insert(iterator position, const T& value);
    void insert(size_t index, const T& value);

    template <typename InputIterator>
    void insert(const_iterator it, InputIterator first, InputIterator last);

    void insert(const_iterator it, std::initializer_list<T> ilist);

    void swap(Vector& other);


    Allocator& allocator();
    const Allocator& allocator() const;

    DOXYGEN_START_INTERNAL_BLOCK()
    size_t getNumActiveElements() const;
    DOXYGEN_END_INTERNAL_BLOCK()

  private:
    void reallocate(size_t size);
    void constructElements(size_t startIndex, size_t endIndex);
    void constructElements(size_t startIndex, size_t endIndex, const T& value);
    void destroyElements(size_t startIndex, size_t endIndex);

    Allocator m_allocator;
    Real32 m_smoothingAverage;
  };

  template<typename T, typename A>
  typename Vector<T, A>::iterator begin(Vector<T, A>& v);

  template<typename T, typename A>
  typename Vector<T, A>::const_iterator begin(const Vector<T, A>& v);

  template<typename T, typename A>
  typename Vector<T, A>::iterator end(Vector<T, A>& v);

  template<typename T, typename A>
  typename Vector<T, A>::const_iterator end(const Vector<T, A>& v);



  /**
  Vector containing 'raw' data. Same interface as normal vectors, but no element constructors
  or destructors are executed. New elements are however initialized by basic memory copying
  from a default instance.
  */
  template<typename T, typename Allocator = ByteAllocator>
  class VectorPOD : public Container
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
      typedef ptrdiff_t                          difference_type;



    public:
      explicit VectorPOD(const Allocator& allocator = Allocator());
      explicit VectorPOD(size_t size, const T& value = T(), const Allocator& allocator = Allocator());
      explicit VectorPOD(const_iterator first, const_iterator end, const Allocator& allocator = Allocator());
      VectorPOD(std::initializer_list<T> values, const Allocator& allocator = Allocator());

      VectorPOD(const VectorPOD<T, Allocator>& other);
      VectorPOD(VectorPOD<T, Allocator>&& other);

      ~VectorPOD();

      VectorPOD<T, Allocator>& operator= (const VectorPOD<T, Allocator>& other);


      /**
      Compare with other vector, return true if same size and all pairs are equal.
      */
      bool operator == ( const VectorPOD<T, Allocator>& other ) const;
      bool operator != ( const VectorPOD<T, Allocator>& other ) const;

      /**
      Resize the vector, which then enables direct addressing using the bracket '[]' operator.
      */
      void resize(size_t size);
      void resize(size_t size, const T& value);

      /**
      Reduce the capacity of the vector to the actual size (number of elements)
      */
      void shrink_to_fit();

      /**
      Reserve capacity in the vector.
      */
      void reserve(size_t size);

      /**
      Reserve capacity in the vector. Let vector grow in same way as push_back() does.
      */
      void reserveAtLeast(size_t size);

      /**
      Resize using a increment.
      */
      T* increment(size_t numElements = 1);

      /**
      Get access to the internal data buffer.
      */
      T* ptr();
      const T* ptr() const;

      T& operator[] (size_t i) const;
      T& at(size_t index) const;
      T& front() const;
      T& back() const;

      void push_back(const T& value);
      void push_back(T&& value);
      void pop_back();

      typedef std::reverse_iterator<iterator> reverse_iterator;
      typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

      iterator begin();
      iterator end();
      const_iterator begin() const;
      const_iterator end() const;

      reverse_iterator rbegin();
      reverse_iterator rend();
      const_reverse_iterator rbegin() const;
      const_reverse_iterator rend() const;

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
      void clear(ClearPolicy policy = SHRINK_BUFFER_AVERAGED);

      /**
      Test if the vector contains a certain element.
      */
      template <typename T2>
      bool contains(const T2& element) const;

      bool contains(const T& element) const;

      /**
      Find the index to a matching element, return size() if not found.
      */
      template <typename T2>
      size_t find(const T2& element) const;

      /**
      Find and erase an element.
      \param element - Element to search for and erase
      \param searchMultiple Set to true if search should continue after element is found, otherwise the search terminates on first match.
      \return true if the element was found, else false
      */
      bool findAndErase(const T& element, bool searchMultiple = false);

      iterator insert(iterator position, const T& value);
      void insert(size_t index, const T& value);

      template <typename InputIterator>
      void insert(const_iterator it, InputIterator first, InputIterator last);

      void insert(const_iterator it, std::initializer_list<T> ilist);

      void swap(VectorPOD& other);

      Allocator& allocator();
      const Allocator& allocator() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      size_t getNumActiveElements() const;
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      void reallocate(size_t size);
      void constructElements(size_t startIndex, size_t endIndex);
      void constructElements(size_t startIndex, size_t endIndex, const T& value);

      static T s_initializer;

      Allocator m_allocator;
      Real32 m_smoothingAverage;
  };

  template<typename T, typename A>
  typename VectorPOD<T, A>::iterator begin(VectorPOD<T, A>& v);

  template<typename T, typename A>
  typename VectorPOD<T, A>::const_iterator begin(const VectorPOD<T, A>& v);

  template<typename T, typename A>
  typename VectorPOD<T, A>::iterator end(VectorPOD<T, A>& v);

  template<typename T, typename A>
  typename VectorPOD<T, A>::const_iterator end(const VectorPOD<T, A>& v);


  /* Implementation */
  #define AGX_VECTOR_RESIZE_FACTOR 2.0f
  #define AGX_VECTOR_SMOOTHING_FACTOR 0.8f
  #define AGX_VECTOR_SHRINK_THRESHOLD 0.25f
  #define AGX_VECTOR_MIN_SIZE 4

  #define m_elements static_cast<T *>(this->m_buffer)
  #define m_size this->m_size
  #define m_capacity this->m_capacity


  template<class T>
  AGX_FORCE_INLINE void swap(Vector<T>& x, Vector<T>& y) { x.swap(y); }

  template<class T>
  AGX_FORCE_INLINE void swap(VectorPOD<T>& x, VectorPOD<T>& y) { x.swap(y); }


  template <typename T, typename A>
  Vector<T, A>::Vector(const A& allocator) : m_allocator(allocator), m_smoothingAverage(0.0)
  {
    m_allocator.setContainer(this);
  }

  template <typename T, typename A>
  Vector<T, A>::Vector(size_t size, const T& value, const A& allocator) : m_allocator(allocator), m_smoothingAverage(0.0)
  {
    m_allocator.setContainer(this);
    this->resize(size, value);
  }

  template <typename T, typename A>
  Vector<T, A>::Vector(const Vector<T, A>& other) : m_smoothingAverage(0.0)
  {
    *this = other;
  }

  template <typename T, typename Allocator>
  Vector<T, Allocator>::Vector(Vector<T, Allocator>&& other)
    :Container(std::move(other)), m_smoothingAverage(0.0)
  {
    // Move values
    m_allocator = other.m_allocator;
    m_allocator.setContainer(this),
    m_smoothingAverage = other.m_smoothingAverage;

    // Assign default values to the rvalue reference.
    other.m_allocator = Allocator();
    other.m_smoothingAverage = Real32(0.0);
  }

  template <typename T, typename A>
  Vector<T, A>::Vector(const_iterator first, const_iterator end, const A& allocator) : m_allocator(allocator), m_smoothingAverage(0.0)
  {
    m_allocator.setContainer(this);
    this->insert(this->begin(), first, end);
  }

  template <typename T, typename A>
  Vector<T, A>::Vector(std::initializer_list<T> values, const A& allocator)
    : m_allocator(allocator), m_smoothingAverage(0.0)
  {
    m_allocator.setContainer(this);
    using std::begin;
    using std::end;
    this->insert(begin(*this), begin(values), end(values));
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::clear(ClearPolicy policy)
  {
    if (this->empty())
      return;

    this->destroyElements(0, m_size);

    if (policy == MAINTAIN_BUFFER)
    {
      m_size = 0;
    }
    else if (policy == SHRINK_BUFFER)
    {
      m_allocator.deallocateBytes(m_buffer, m_capacity * sizeof(T));
      m_buffer = nullptr;
      m_size = 0;
      m_capacity = 0;
      m_smoothingAverage = 0.0;
    }
    else // SHRINK_BUFFER_AVERAGED
    {
      /* Smoothing average is never less than current size */
      if ((Real32)m_size > m_smoothingAverage)
        m_smoothingAverage = (Real32)m_size;

      /* Update smoothing average */
      m_smoothingAverage = (Real32)(AGX_VECTOR_SMOOTHING_FACTOR * m_smoothingAverage + (1.0f-AGX_VECTOR_SMOOTHING_FACTOR) * (float)m_size);

      /* Clear size before reallocate so it does not copy elements from old buffer. */
      m_size = 0;

      /* Reallocate buffer if the smoothing average is sufficiently below the current capacity */
      if (m_smoothingAverage/(Real32)m_capacity < AGX_VECTOR_SHRINK_THRESHOLD)
        this->reallocate((size_t)m_smoothingAverage);
    }
  }



  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::shrink_to_fit()
  {
    if (m_capacity > m_size )
    {
      this->reallocate(m_size);
    }
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::resize(size_t size)
  {
    if (size > m_size)
    {
      if (size > m_capacity)
        this->reallocate(size);

      this->constructElements(m_size, size);
    }
    else
    {
      if (size < (size_t)((double)m_capacity * AGX_VECTOR_SHRINK_THRESHOLD))
        this->reallocate(size);

      this->destroyElements(size, m_size);
    }

    m_size = size;
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::resize(size_t size, const T& value)
  {
    if (size > m_size)
    {
      if (size > m_capacity)
        this->reallocate(size);

      this->constructElements(m_size, size, value);
    }
    else
    {
      if (size < (size_t)((double)m_capacity * AGX_VECTOR_SHRINK_THRESHOLD))
        this->reallocate(size);

      this->destroyElements(size, m_size);
    }

    m_size = size;
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::reserve(size_t size)
  {
    if (size > m_capacity)
      this->reallocate(size);
  }



  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::reserveAtLeast(size_t size)
  {
    if (size > m_capacity) {
      size_t newCapacity = std::max(m_capacity, size_t(AGX_VECTOR_MIN_SIZE));
      while (size > newCapacity)
        newCapacity = size_t((double)newCapacity * AGX_VECTOR_RESIZE_FACTOR);
      this->reallocate(newCapacity);
    }
  }



  template <typename T, typename A>
  AGX_FORCE_INLINE T* Vector<T, A>::increment(size_t numElements)
  {
    size_t newSize = m_size + numElements;
    if (newSize > m_capacity)
      this->reallocate((size_t)((double)newSize * AGX_VECTOR_RESIZE_FACTOR));

    this->constructElements(m_size, newSize);
    m_size = newSize;
    return m_elements + m_size - numElements;
  }



  template <typename T, typename A>
  AGX_FORCE_INLINE bool Vector<T, A>::operator == ( const Vector<T, A>& other ) const
  {
    if (m_size != other.size())
      return false;

    for (size_t i = 0; i < m_size; i++)
      if (m_elements[i] != other[i])
        return false;

    return true;
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE bool Vector<T, A>::operator != ( const Vector<T, A>& other ) const
  {
    if (m_size != other.size())
      return true;

    for (size_t i = 0; i < m_size; i++)
      if (m_elements[i] != other[i])
        return true;

    return false;
  }


  template <typename T, typename A>
  Vector<T, A>& Vector<T, A>::operator= (const Vector<T, A>& other)
  {
    if (this == &other)
      return *this;

    this->clear(MAINTAIN_BUFFER);
    m_allocator = other.m_allocator;
    m_allocator.setContainer(this);

    // this->reserve(other.capacity());
    // this->resize(other.size());
    this->reallocate(other.capacity());

    /* Copy elements from other buffer */
    for (size_t i = 0; i < other.size(); i++)
      ::new((void *)&m_elements[i]) T(other[i]);

    m_size = other.size();

    return *this;
  }

  template <typename T, typename A>
  Vector<T, A>::~Vector()
  {
    this->destroyElements(0, m_size);
    m_allocator.deallocateBytes(m_elements, m_capacity * sizeof(T));
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE T* Vector<T, A>::ptr() { return m_elements; }

  template <typename T, typename A>
  AGX_FORCE_INLINE const T* Vector<T, A>::ptr() const { return m_elements; }


  template <typename T, typename A>
  AGX_FORCE_INLINE T& Vector<T, A>::operator[] (size_t i) const
  {
    agxAssertN(i < m_size, "Array index %llu is out of bounds, current size is %llu", (long long unsigned)i, (long long unsigned)m_size);
    return m_elements[i];
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE T& Vector<T, A>::at(size_t index) const
  {
    agxVerifyN(index < m_size, "Array index %llu is out of bounds, current size is %llu", (long long unsigned)index, (long long unsigned)m_size);
    return m_elements[index];
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE T& Vector<T, A>::front() const
  {
    agxAssert1(m_size > 0, "Can not take front element of an empty vector!");
    return m_elements[0];
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE T& Vector<T, A>::back() const
  {
    agxAssert1(m_size > 0, "Can not take back element of an empty vector!");
    return m_elements[m_size-1];
  }


  template <typename T, typename A> template <typename T2>
  AGX_FORCE_INLINE void Vector<T, A>::push_back(const T2& value)
  {
    if (m_size == m_capacity)
      this->reallocate(std::max((size_t)AGX_VECTOR_MIN_SIZE, (size_t)((float)m_size * AGX_VECTOR_RESIZE_FACTOR)));


    ::new((void *)&m_elements[m_size]) T(value); // removed unnecessary loop by replacing line below by this one
    // this->_constructElements(m_size, m_size+1, value);
    m_size++;
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::push_back(const T& value)
  {
    if (m_size == m_capacity)
      this->reallocate(std::max((size_t)AGX_VECTOR_MIN_SIZE, (size_t)((float)m_size * AGX_VECTOR_RESIZE_FACTOR)));


    ::new((void *)&m_elements[m_size]) T(value); // removed unnecessary loop by replacing line below by this one
    // this->_constructElements(m_size, m_size+1, value);
    m_size++;
  }



  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::push_back(T&& value)
  {
    if (m_size == m_capacity)
    {
      this->reallocate(std::max((size_t)AGX_VECTOR_MIN_SIZE, (size_t)((float)m_size * AGX_VECTOR_RESIZE_FACTOR)));
    }

    ::new((void*)&m_elements[m_size]) T(std::move(value));
    m_size++;
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::pop_back()
  {
    agxAssert( m_size );
    m_elements[m_size-1].~T();
    m_size--;

//    if (m_size < m_capacity * AGX_VECTOR_SHRINK_THRESHOLD)
//      this->reallocate((size_t)(m_size * AGX_VECTOR_RESIZE_FACTOR));
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename Vector<T, A>::iterator Vector<T, A>::erase(iterator position)
  {
    size_t index = position - this->begin();
    this->erase(index);
    return this->begin() + index;
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename Vector<T, A>::iterator Vector<T, A>::erase(iterator start, iterator end)
  {
    size_t startIndex = start - this->begin();
    size_t endIndex = end - this->begin();
    this->erase(startIndex, endIndex);

    return this->begin() + endIndex;
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::erase(size_t index)
  {
    agxAssert( m_size );

    // removed unnecessary loop by replacing line below by this block
    {
      // TODO We should first run destructor and constructor on element before copying? Again we assume copy operator semantics to be 'proper'
      for (size_t i = index + 1; i < m_size; i++)
        m_elements[i-1] = m_elements[i];

      /* Run destructor on trailing elements */
      m_elements[m_size - 1].~T();

      m_size--;

      // if (m_size < m_capacity * AGX_VECTOR_SHRINK_THRESHOLD)
      // this->reallocate((size_t)(m_size * AGX_VECTOR_RESIZE_FACTOR));
    }
    // this->erase(index, index+1);
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::erase(size_t start, size_t end)
  {
    agxAssert1(start < end && end <= m_size, "Erase bounds are not with array bounds!");
    size_t numElements = end - start;

    // TODO We should first run destructor and constructor on element before copying? Again we assume copy operator semantics to be 'proper'
    for (size_t i = end; i < m_size; i++)
      m_elements[i-numElements] = m_elements[i];

    /* Run destructor on trailing elements */
    this->destroyElements(m_size-numElements, m_size);

    m_size -= numElements;

    // if (m_size < m_capacity * AGX_VECTOR_SHRINK_THRESHOLD)
      // this->reallocate((size_t)(m_size * AGX_VECTOR_RESIZE_FACTOR));
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename Vector<T, A>::iterator Vector<T, A>::eraseFast(const_iterator position)
  {
    size_t index = position - this->begin();
    this->eraseFast(index);
    return this->begin() + index;
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::eraseFast(size_t index)
  {
    agxAssertN(index < m_size, "Array index %llu is out of bounds, current size is %llu", (long long unsigned)index, (long long unsigned)m_size);
    m_elements[index] = m_elements[m_size-1];
    m_elements[m_size - 1].~T(); // removed unnecessary loop by replacing line below by this one
    // this->_destroyElements(m_size-1, m_size);
    m_size--;
  }

  template <typename T, typename A> template <typename T2>
  AGX_FORCE_INLINE bool Vector<T, A>::contains(const T2& element) const
  {
    return this->find(element) != m_size;
  }

  template <typename T, typename A> //template <typename T2>
  AGX_FORCE_INLINE bool Vector<T, A>::contains(const T& element) const
  {
    return this->find(element) != m_size;
  }


  template <typename T, typename A> template <typename T2>
  AGX_FORCE_INLINE size_t Vector<T, A>::find(const T2& element) const
  {
    for (size_t i = 0; i < m_size; i++)
    {
      if (m_elements[i] == element)
        return i;
    }

    return m_size;
  }

  template <typename T, typename A> template <typename T2>
  AGX_FORCE_INLINE bool Vector<T, A>::findAndErase(const T2& element, bool searchMultiple)
  {
    bool found = false;

    for (size_t i = 0; i < m_size; i++)
    {
      if (m_elements[i] == element)
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


  template <typename T, typename A>
  AGX_FORCE_INLINE typename Vector<T, A>::iterator Vector<T, A>::insert(iterator position, const T& value)
  {
    size_t index = position - this->begin();
    this->insert(index, value);
    return this->begin() + index;
  }

  template <typename T, typename A> template <typename InputIterator>
  AGX_FORCE_INLINE void Vector<T, A>::insert(const_iterator it, InputIterator first, InputIterator last)
  {
    // TODO Difference between iterators is always number of elements between them??
    size_t numElements = std::distance( first, last );

    if (numElements == 0)
      return;

    size_t index = it - this->begin();

    if (m_size + numElements > m_capacity)
      this->reallocate((size_t)((float)(m_size + numElements) * AGX_VECTOR_RESIZE_FACTOR));

    for (size_t i = m_size; i > index; i--) {
      ::new((void *)&m_elements[i + numElements - 1]) T(m_elements[i-1]);
      m_elements[i-1].~T();
    }

    for (; first != last; first++)
      ::new((void *)&m_elements[index++]) T(*first);

    m_size += numElements;
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void agx::Vector<T, A>::insert(const_iterator it, std::initializer_list<T> ilist)
  {
    this->insert(it, ilist.begin(), ilist.end());
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::insert(size_t index, const T& value)
  {
    agxAssertN(index <= m_size, "Insert with invalid index %llu, size is %llu", (long long unsigned)index, (long long unsigned)m_size);

    if (m_size == m_capacity)
      this->reallocate(std::max((size_t)AGX_VECTOR_MIN_SIZE, (size_t)((double)m_size * AGX_VECTOR_RESIZE_FACTOR)));

    for (size_t i = m_size; i > index; i--) {
      ::new((void *)&m_elements[i]) T(m_elements[i-1]);
      m_elements[i-1].~T();
    }

    ::new((void *)&m_elements[index]) T(value);
    m_size++;
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::reallocate(size_t size)
  {
    /* Check if new size is less than current size, which will remove trailing elements */
    if (size < m_size)
    {
      this->destroyElements(size, m_size);
      m_size = size;
    }


    /* Create new buffer */
    size = std::max(size, (size_t)AGX_VECTOR_MIN_SIZE);
    T *newBuf = reinterpret_cast<T*>(m_allocator.allocateBytes(size * sizeof(T)));

    /* Initialize elements with old values. NOTE This only works as long as the copy constructor
      has the same semantics as the assignment operator, which we will assume. Otherwise this has to
      be done in two passes, first running default constructors, then assigning old values. */
    for (size_t i = 0; i < m_size; i++)
      ::new((void *)&newBuf[i]) T(m_elements[i]);


    /* Destroy old elements and free buffer */
    this->destroyElements(0, m_size);
    m_allocator.deallocateBytes(m_elements, m_capacity * sizeof(T));

    m_buffer = (void *)newBuf;
    m_capacity = size;
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::constructElements(size_t startIndex, size_t endIndex)
  {
    for (size_t i = startIndex; i < endIndex; i++)
      ::new((void *)&m_elements[i]) T();
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::constructElements(size_t startIndex, size_t endIndex, const T& value)
  {
    for (size_t i = startIndex; i < endIndex; i++)
      ::new((void *)&m_elements[i]) T(value);
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::destroyElements(size_t startIndex, size_t endIndex)
  {
    for (size_t i = startIndex; i < endIndex; i++)
      m_elements[i].~T();
  }



  template <typename T, typename A>
  AGX_FORCE_INLINE A& Vector<T, A>::allocator() { return m_allocator; }

  template <typename T, typename A>
  AGX_FORCE_INLINE const A& Vector<T, A>::allocator() const { return m_allocator; }

  DOXYGEN_START_INTERNAL_BLOCK()
  template <typename T, typename A>
  AGX_FORCE_INLINE size_t Vector<T, A>::getNumActiveElements() const { return m_size; }
  DOXYGEN_END_INTERNAL_BLOCK()

  template <typename VectorT>
  std::ostream& printVector( std::ostream& output, const VectorT& vec)
  {
    for (size_t i = 0; i < vec.size(); i++)
    {
      output << vec[i];

      if (i < vec.size()-1)
        output << ", ";
    }

    return output;
  }

  template <typename T, typename A>
  std::ostream& operator << ( std::ostream& output, const Vector<T, A>& vec)
  {
    return printVector(output, vec);
  }


  template <typename T, typename A>
  std::ostream& operator << ( std::ostream& output, const VectorPOD<T, A>& vec)
  {
    return printVector(output, vec);
  }


  /* Iterators */
  template <typename T, typename A>
  AGX_FORCE_INLINE typename Vector<T, A>::iterator Vector<T, A>::begin() { return m_elements; }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename Vector<T, A>::iterator Vector<T, A>::end() { return m_elements + m_size; }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename Vector<T, A>::const_iterator Vector<T, A>::begin() const { return m_elements; }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename Vector<T, A>::const_iterator Vector<T, A>::end() const { return m_elements + m_size; }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename Vector<T, A>::reverse_iterator Vector<T, A>::rbegin() { return reverse_iterator(this->end()); }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename Vector<T, A>::reverse_iterator Vector<T, A>::rend() { return reverse_iterator(this->begin()); }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename Vector<T, A>::const_reverse_iterator Vector<T, A>::rbegin() const { return const_reverse_iterator(this->end()); }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename Vector<T, A>::const_reverse_iterator Vector<T, A>::rend() const { return const_reverse_iterator(this->begin()); }




  template<typename T, typename A>
  typename Vector<T, A>::iterator begin(Vector<T, A>& v)  { return v.begin(); }

  template<typename T, typename A>
  typename Vector<T, A>::const_iterator begin(const Vector<T, A>& v) { return v.begin(); }


  template <typename T, typename A>
  typename Vector<T, A>::iterator end(Vector<T, A>& v) { return v.end(); }

  template <typename T, typename A>
  typename Vector<T, A>::const_iterator end(const Vector<T, A>& v) { return v.end(); }







  //-----------------------------------------------------------------------------------------------------
  //--------------- VectorPOD ---------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------------------------



  template <typename T, typename A>
  VectorPOD<T, A>::VectorPOD(const A& allocator) : m_allocator(allocator), m_smoothingAverage(0.0)
  {
    static_assert( std::is_trivially_destructible<T>::value, "Non-POD type used in VectorPOD");
    m_allocator.setContainer(this);
  }

  template <typename T, typename A>
  VectorPOD<T, A>::VectorPOD(size_t size, const T& value, const A& allocator) : m_allocator(allocator), m_smoothingAverage(0.0)
  {
    m_allocator.setContainer(this);
    this->resize(size, value);
  }

  template <typename T, typename A>
  VectorPOD<T, A>::VectorPOD(const VectorPOD<T, A>& other) : m_smoothingAverage(0.0)
  {
    *this = other;
  }



  template <typename T, typename A>
  VectorPOD<T, A>::VectorPOD(VectorPOD<T, A>&& other)
    : Container(std::move(other)), m_smoothingAverage(0.0)
  {
    // Move values
    m_allocator = std::move(other.m_allocator);
    m_allocator.setContainer(this);
    m_smoothingAverage = other.m_smoothingAverage;

    // Assign default values to rvalue reference
    other.m_allocator = A();
    other.m_smoothingAverage = Real32(0);
  }



  template <typename T, typename A>
  VectorPOD<T, A>::VectorPOD(const_iterator first, const_iterator end, const A& allocator) : m_allocator(allocator), m_smoothingAverage(0.0)
  {
    m_allocator.setContainer(this);
    this->insert(this->begin(), first, end);
  }

  template <typename T, typename A>
  VectorPOD<T, A>::VectorPOD(std::initializer_list<T> values, const A& allocator)
    : m_allocator(allocator), m_smoothingAverage(0.0)
  {
    m_allocator.setContainer(this);
    using std::begin;
    using std::end;
    this->insert(begin(*this), begin(values), end(values));
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::clear(ClearPolicy policy)
  {
    if (this->empty())
      return;

    if (policy == MAINTAIN_BUFFER)
    {
      m_size = 0;
    }
    else if (policy == SHRINK_BUFFER)
    {
      m_allocator.deallocateBytes(m_buffer, m_capacity * sizeof(T));
      m_buffer = nullptr;
      m_size = 0;
      m_capacity = 0;
      m_smoothingAverage = 0.0;
    }
    else // SHRINK_BUFFER_AVERAGED
    {
      /* Smoothing average is never less than current size */
      if ((Real32)m_size > m_smoothingAverage)
        m_smoothingAverage = (Real32)m_size;

      /* Update smoothing average */
      m_smoothingAverage = (Real32)(AGX_VECTOR_SMOOTHING_FACTOR * m_smoothingAverage + (1.0-AGX_VECTOR_SMOOTHING_FACTOR) * (float)m_size);

      /* Clear size before reallocate so it does not copy elements from old buffer. */
      m_size = 0;

      /* Reallocate buffer if the smoothing average is sufficiently below the current capacity */
      if (m_smoothingAverage/(Real32)m_capacity < AGX_VECTOR_SHRINK_THRESHOLD)
        this->reallocate((size_t)m_smoothingAverage);
    }
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::shrink_to_fit()
  {
    if (m_capacity > m_size)
    {
      this->reallocate(m_size);
    }
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::resize(size_t size)
  {
    if (size > m_size)
    {
      if (size > m_capacity)
        this->reallocate(size);

      this->constructElements(m_size, size);
    }
    else
    {
      if (size < (size_t)((double)m_capacity * AGX_VECTOR_SHRINK_THRESHOLD))
        this->reallocate(size);
    }

    m_size = size;
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::resize(size_t size, const T& value)
  {
    if (size > m_size)
    {
      if (size > m_capacity)
        this->reallocate(size);

      this->constructElements(m_size, size, value);
    }
    else
    {
      if ((Real32)size < (Real32)m_capacity * AGX_VECTOR_SHRINK_THRESHOLD)
        this->reallocate(size);
    }

    m_size = size;
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::reserve(size_t size)
  {
    if (size > m_capacity)
      this->reallocate(size);
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::reserveAtLeast(size_t size)
  {
    if (size > m_capacity) {
      size_t newCapacity = std::max(m_capacity, size_t(AGX_VECTOR_MIN_SIZE));
      while (size > newCapacity)
        newCapacity = size_t((float)newCapacity * AGX_VECTOR_RESIZE_FACTOR);
      this->reallocate(newCapacity);
    }
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE T* VectorPOD<T, A>::increment(size_t numElements)
  {
    size_t newSize = m_size + numElements;
    if (newSize > m_capacity)
      this->reallocate((size_t)((float)newSize * AGX_VECTOR_RESIZE_FACTOR));

    this->constructElements(m_size, newSize);
    m_size = newSize;
    return m_elements + m_size - numElements;
  }



  template <typename T, typename A>
  AGX_FORCE_INLINE bool VectorPOD<T, A>::operator == ( const VectorPOD<T, A>& other ) const
  {
    if (m_size != other.size())
      return false;

    for (size_t i = 0; i < m_size; i++)
      if (m_elements[i] != other[i])
        return false;

    return true;
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE bool VectorPOD<T, A>::operator != ( const VectorPOD<T, A>& other ) const
  {
    if (m_size != other.size())
      return true;

    for (size_t i = 0; i < m_size; i++)
      if (m_elements[i] != other[i])
        return true;

    return false;
  }


  template <typename T, typename A>
  VectorPOD<T, A>& VectorPOD<T, A>::operator= (const VectorPOD<T, A>& other)
  {
    if (this == &other)
      return *this;

    m_allocator = other.m_allocator;
    m_allocator.setContainer(this);

    if (other.size() > m_capacity || other.size() < (size_t)((double)m_capacity * AGX_VECTOR_SHRINK_THRESHOLD))
      this->reallocate(other.size());

    if (other.size() > 0) {
      memcpy(m_elements, other.ptr(), other.size() * sizeof(T));
    }

    m_size = other.size();

    return *this;
  }

  template <typename T, typename A>
  VectorPOD<T, A>::~VectorPOD()
  {
    m_allocator.deallocateBytes(m_elements, m_capacity * sizeof(T));
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE T* VectorPOD<T, A>::ptr() { return m_elements; }

  template <typename T, typename A>
  AGX_FORCE_INLINE const T* VectorPOD<T, A>::ptr() const { return m_elements; }

  template <typename T, typename A>
  AGX_FORCE_INLINE T& VectorPOD<T, A>::operator[] (size_t i) const
  {
    agxAssertN(i < m_size, "Array index %llu is out of bounds, current size is %llu", (long long unsigned)i, (long long unsigned)m_size);
    return m_elements[i];
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE T& VectorPOD<T, A>::at(size_t index) const
  {
    agxVerifyN(index < m_size, "Array index %llu is out of bounds, current size is %llu", (long long unsigned)index, (long long unsigned)m_size);
    return m_elements[index];
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE T& VectorPOD<T, A>::front() const
  {
    agxAssert1(m_size > 0, "Can not take front element of an empty vector!");
    return m_elements[0];
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE T& VectorPOD<T, A>::back() const
  {
    agxAssert1(m_size > 0, "Can not take back element of an empty vector!");
    return m_elements[m_size-1];
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::push_back(const T& value)
  {
    if (m_size == m_capacity)
      this->reallocate(std::max((size_t)AGX_VECTOR_MIN_SIZE, (size_t)( (float)m_size * AGX_VECTOR_RESIZE_FACTOR)));

    m_elements[m_size] = value;
    m_size++;
  }



  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::push_back(T&& value)
  {
    if (m_size == m_capacity)
      this->reallocate(std::max((size_t)AGX_VECTOR_MIN_SIZE, (size_t)( (float)m_size * AGX_VECTOR_RESIZE_FACTOR)));

    m_elements[m_size] = std::move(value);
    m_size++;
  }



  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::pop_back()
  {
    m_size--;

//    if (m_size < m_capacity * AGX_VECTOR_SHRINK_THRESHOLD)
//      this->reallocate((size_t)(m_size * AGX_VECTOR_RESIZE_FACTOR));
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename VectorPOD<T, A>::iterator VectorPOD<T, A>::erase(iterator position)
  {
    size_t index = position - this->begin();
    this->erase(index);
    return this->begin() + index;
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename VectorPOD<T, A>::iterator VectorPOD<T, A>::erase(iterator start, iterator end)
  {
    size_t startIndex = start - this->begin();
    size_t endIndex = end - this->begin();
    this->erase(startIndex, endIndex);

    return this->begin() + endIndex;
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::erase(size_t index)
  {
    this->erase(index, index+1);
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::erase(size_t start, size_t end)
  {
    agxAssert1(start < end && end <= m_size, "Erase bounds are not with array bounds!");
    size_t numElements = end - start;

    // TODO We should first run destructor and constructor on element before copying? Again we assume copy operator semantics to be 'proper'
    for (size_t i = end; i < m_size; i++)
      m_elements[i-numElements] = m_elements[i];

    m_size -= numElements;

    // if (m_size < m_capacity * AGX_VECTOR_SHRINK_THRESHOLD)
      // this->reallocate((size_t)(m_size * AGX_VECTOR_RESIZE_FACTOR));
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename VectorPOD<T, A>::iterator VectorPOD<T, A>::eraseFast(const_iterator position)
  {
    size_t index = position - this->begin();
    this->eraseFast(index);
    return this->begin() + index;
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::eraseFast(size_t index)
  {
    agxAssertN(index < m_size, "Array index %llu is out of bounds, current size is %llu", (long long unsigned)index, (long long unsigned)m_size);
    m_elements[index] = m_elements[m_size-1];
    m_size--;
  }

  template <typename T, typename A> template <typename T2>
  AGX_FORCE_INLINE bool VectorPOD<T, A>::contains(const T2& element) const
  {
    return this->find(element) != m_size;
  }

  template <typename T, typename A> //template <typename T2>
  AGX_FORCE_INLINE bool VectorPOD<T, A>::contains(const T& element) const
  {
    return this->find(element) != m_size;
  }


  template <typename T, typename A> template <typename T2>
  AGX_FORCE_INLINE size_t VectorPOD<T, A>::find(const T2& element) const
  {
    for (size_t i = 0; i < m_size; i++)
    {
      if (m_elements[i] == element)
        return i;
    }

    return m_size;
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE bool VectorPOD<T, A>::findAndErase(const T& element, bool searchMultiple)
  {
    bool found = false;

    for (size_t i = 0; i < m_size; i++)
    {
      if (m_elements[i] == element)
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


  template <typename T, typename A>
  AGX_FORCE_INLINE typename VectorPOD<T, A>::iterator VectorPOD<T, A>::insert(iterator position, const T& value)
  {
    agxAssert( position >= begin() );
    size_t index = position - this->begin();
    this->insert(index, value);
    return this->begin() + index;
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::insert(const_iterator it, std::initializer_list<T> ilist)
  {
    this->insert(it, ilist.begin(), ilist.end());
  }

  template <typename T, typename A> template <typename InputIterator>
  AGX_FORCE_INLINE void VectorPOD<T, A>::insert(const_iterator it, InputIterator first, InputIterator last)
  {
    // TODO Difference between iterators is always number of elements between them??
    size_t numElements = std::distance( first, last );
    if (numElements == 0)
      return;

    size_t index = it - this->begin();

    if (m_size + numElements > m_capacity)
      this->reallocate((size_t)((float)(m_size + numElements) * AGX_VECTOR_RESIZE_FACTOR));

    for (size_t i = m_size; i > index; i--)
      memcpy(&m_elements[i + numElements - 1], &m_elements[i-1], sizeof(T));

    for (; first != last; first++)
      memcpy(&m_elements[index++], &(*first), sizeof(T));

    m_size += numElements;
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::insert(size_t index, const T& value)
  {
    if (m_size == m_capacity)
      this->reallocate(std::max((size_t)AGX_VECTOR_MIN_SIZE, (size_t)((double)m_size * AGX_VECTOR_RESIZE_FACTOR)));

    for (size_t i = m_size; i > index; i--)
      memcpy(&m_elements[i], &m_elements[i-1], sizeof(T));

    memcpy(&m_elements[index], &value, sizeof(T));
    m_size++;
  }



  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::reallocate(size_t size)
  {
    /* Check if new size is less than current size, which will remove trailing elements */
    if (size < m_size)
      m_size = size;

    /* Create new buffer */
    size = std::max(size, (size_t)AGX_VECTOR_MIN_SIZE);
    void *newBuf = m_allocator.allocateBytes(size * sizeof(T));
    void *oldBuf = m_elements;

    if (m_size > 0) {
      memcpy(newBuf, oldBuf, m_size * sizeof(T));
    }

    m_buffer = newBuf;

    /* Free old buffer */
    m_allocator.deallocateBytes(oldBuf, m_capacity * sizeof(T));
    m_capacity = size;
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::constructElements(size_t startIndex, size_t endIndex)
  {
    for (size_t i = startIndex; i < endIndex; i++)
      memcpy(&m_elements[i], &s_initializer, sizeof(T));
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::constructElements(size_t startIndex, size_t endIndex, const T& value)
  {
    for (size_t i = startIndex; i < endIndex; i++)
      memcpy((void*)&m_elements[i], (void*)&value, sizeof(T));
  }


  template <typename T, typename A>
  AGX_FORCE_INLINE A& VectorPOD<T, A>::allocator() { return m_allocator; }

  template <typename T, typename A>
  AGX_FORCE_INLINE const A& VectorPOD<T, A>::allocator() const { return m_allocator; }

  DOXYGEN_START_INTERNAL_BLOCK()
  template <typename T, typename A>
  AGX_FORCE_INLINE size_t VectorPOD<T, A>::getNumActiveElements() const { return m_size; }
  DOXYGEN_END_INTERNAL_BLOCK()

  /* Initializer instance */
  template <typename T, typename A>
  T VectorPOD<T, A>::s_initializer;

  /* Iterators */
  template <typename T, typename A>
  AGX_FORCE_INLINE typename VectorPOD<T, A>::iterator VectorPOD<T, A>::begin() { return m_elements; }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename VectorPOD<T, A>::iterator VectorPOD<T, A>::end() { return m_elements + m_size; }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename VectorPOD<T, A>::const_iterator VectorPOD<T, A>::begin() const { return m_elements; }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename VectorPOD<T, A>::const_iterator VectorPOD<T, A>::end() const { return m_elements + m_size; }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename VectorPOD<T, A>::reverse_iterator VectorPOD<T, A>::rbegin() { return reverse_iterator(this->end()); }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename VectorPOD<T, A>::reverse_iterator VectorPOD<T, A>::rend() { return reverse_iterator(this->begin()); }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename VectorPOD<T, A>::const_reverse_iterator VectorPOD<T, A>::rbegin() const { return const_reverse_iterator(this->end()); }

  template <typename T, typename A>
  AGX_FORCE_INLINE typename VectorPOD<T, A>::const_reverse_iterator VectorPOD<T, A>::rend() const { return const_reverse_iterator(this->begin()); }


  template<typename T, typename A>
  typename VectorPOD<T, A>::iterator begin(VectorPOD<T, A>& v) { return v.begin(); }

  template<typename T, typename A>
  typename VectorPOD<T, A>::const_iterator begin(const VectorPOD<T, A>& v) { return v.begin(); }

  template <typename T, typename A>
  typename VectorPOD<T, A>::iterator end(VectorPOD<T, A>& v) { return v.end(); }

  template <typename T, typename A>
  typename VectorPOD<T, A>::const_iterator end(const VectorPOD<T, A>& v) { return v.end(); }

  #undef m_elements
  #undef m_size
  #undef m_capacity


  template <typename T, typename A>
  AGX_FORCE_INLINE void Vector<T, A>::swap(Vector& other)
  {
    std::swap(this->m_buffer, other.m_buffer);
    std::swap(this->m_size, other.m_size);
    std::swap(this->m_capacity, other.m_capacity);
    std::swap(this->m_smoothingAverage, other.m_smoothingAverage);
  }

  template <typename T, typename A>
  AGX_FORCE_INLINE void VectorPOD<T, A>::swap(VectorPOD& other)
  {
    std::swap(this->m_buffer, other.m_buffer);
    std::swap(this->m_size, other.m_size);
    std::swap(this->m_capacity, other.m_capacity);
    std::swap(this->m_smoothingAverage, other.m_smoothingAverage);
  }

}

#ifdef _MSC_VER
#pragma warning (pop)
#endif

#endif /* _AGX_VECTOR_H_ */
