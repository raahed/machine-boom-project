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

#ifndef AGX_OBSERVER_PTR
#define AGX_OBSERVER_PTR

#include <agx/macros.h>
#include <agx/hash.h>

namespace agx
{

  DOXYGEN_START_INTERNAL_BLOCK()


  class Observer
  {
    public:
      virtual ~Observer() {}
      virtual void objectDeleted(void*) {}
  };

  DOXYGEN_END_INTERNAL_BLOCK()

  /**
  Smart pointer for observed objects, that automatically set pointers to them to null when they deleted.

  \code
  class MyClass : public Referenced
  {};

  /// Instantiate the class and let a ref_ptr reference it (increment #references with one)
  ref_ptr<MyClass> r = new MyClass;

  // Keep an observer pointer to the object too.
  observer_ptr<MyClass> o = r.get();

  /// No one is any longer referencing the object, it will be deleted.
  // AND all observer pointers will be informed about this and hence set to 0.
  r = null;

  if (o.isValid()) // Now the observer_pointer is no longer valid.
    // should not happen
  \endcode
  */
  template<class T>
  class observer_ptr : public Observer
  {
    public:

      /// The type of the referenced class
      typedef T element_type;

      /// Constructor
      observer_ptr();

      /// Constructor, create an observer_ptr and listen to the object \p t for when it is deleted.
      observer_ptr(T* t);

      /// Copy constructor
      observer_ptr(const observer_ptr& rp);

      /// Destructor
      virtual ~observer_ptr();

      /// Assignment operator
      observer_ptr& operator = (const observer_ptr& rp);

      observer_ptr& operator = (T* ptr);

      virtual void objectDeleted(void*);

      /// Cast operator that will cast the observer_ptr to its native type
      operator T* () const;

      /// \return true if the two observer pointers are referencing the same object (address)
      template <typename U>
      bool operator == (observer_ptr<U> const& rp) const;

      /// \return true if the two observer pointers are NOT referencing the same object (address)
      template <typename U>
      bool operator != (observer_ptr<U> const& rp) const;

      /// \return true if the pointer \p is pointing at the same object as this observer pointer
      template <typename U>
      bool operator == (U const* p) const;

      /// \return true if the pointer \p is NOT pointing at the same object as this observer pointer
      template <typename U>
      bool operator != (U const* p) const;

      /// \return true if the pointer \p is pointing at the same object as this observer pointer
      template <typename U>
      bool operator == (U* p) const;

      /// \return true if the pointer \p is NOT pointing at the same object as this observer pointer
      template <typename U>
      bool operator != (U* p) const;

      /// \return true if the pointer this is pointing to an address which is less than the one \p rp is referencing.
      bool operator < (const observer_ptr& rp) const;

      /// \return a reference to the referenced object
      T& operator*() const;

      /// automatic cast to native pointer and use the dereferencing operator
      T* operator->() const;

      /// \return a native pointer to the referenced object
      T* get() const;

      /// \return true if the referenced pointer is == null
      bool operator!() const;

      /// \return true if the referenced pointer is == null
      bool isValid() const;

    private:
      T* m_ptr;
  };

  template<typename T>
  AGX_FORCE_INLINE observer_ptr<T>::observer_ptr() : m_ptr(nullptr)
  {

  }

  template<typename T>
  AGX_FORCE_INLINE observer_ptr<T>::observer_ptr(T* t): m_ptr(t)
  {
    if (m_ptr) m_ptr->addObserver(this);
  }

  template<typename T>
  AGX_FORCE_INLINE observer_ptr<T>::observer_ptr(const observer_ptr<T>& rp): Observer(), m_ptr(rp.m_ptr)
  {
    if (m_ptr) m_ptr->addObserver(this);
  }

  template<typename T>
  AGX_FORCE_INLINE observer_ptr<T>::~observer_ptr()
  {
    if (m_ptr) m_ptr->removeObserver(this);
    m_ptr = 0;

  }

  template<typename T>
  AGX_FORCE_INLINE observer_ptr<T>& observer_ptr<T>::operator = (const observer_ptr<T>& rp)
  {
    if (m_ptr == rp.m_ptr) return *this;
    if (m_ptr) m_ptr->removeObserver(this);

    m_ptr = rp.m_ptr;
    if (m_ptr) m_ptr->addObserver(this);
    return *this;
  }

  template<typename T>
  AGX_FORCE_INLINE observer_ptr<T>& observer_ptr<T>::operator = (T* ptr)
  {
    if (m_ptr == ptr) return *this;
    if (m_ptr) m_ptr->removeObserver(this);

    m_ptr = ptr;
    if (m_ptr) m_ptr->addObserver(this);

    return *this;
  }

  template<typename T>
  void observer_ptr<T>::objectDeleted(void*)
  {
    m_ptr = 0;
  }

#if 0 // Disable any hashing on observer_pointer, it can only be bad
  // Hash function
  template<typename T>
  struct HashFn< observer_ptr<T> >
  {
    AGX_FORCE_INLINE UInt32 operator()(const observer_ptr<T>& key) const
    {
      return agx::hash(key.get());
    }

    typedef const T *PtrT;
    AGX_FORCE_INLINE UInt32 operator()(const PtrT key) const
    {
      return agx::hash(key);
    }
  };
#endif

  /// Cast operator that will cast the observer_ptr to its native type
  template<typename T>
  AGX_FORCE_INLINE observer_ptr<T>::operator T* () const
  {
    return m_ptr;
  }

  template<typename T>
  template <typename U>
  AGX_FORCE_INLINE bool observer_ptr<T>::operator == (observer_ptr<U> const& rp) const
  {
    return m_ptr == rp.m_ptr;
  }

  template<typename T>
  template <typename U>
  AGX_FORCE_INLINE bool observer_ptr<T>::operator != (observer_ptr<U> const& rp) const
  {
    return m_ptr != rp.m_ptr;
  }

  template<typename T>
  template <typename U>
  AGX_FORCE_INLINE bool observer_ptr<T>::operator == (U const* p) const
  {
    return m_ptr == p;
  }

  template<typename T>
  template <typename U>
  AGX_FORCE_INLINE bool observer_ptr<T>::operator != (U const* p) const
  {
    return m_ptr != p;
  }

  template<typename T>
  template <typename U>
  AGX_FORCE_INLINE bool observer_ptr<T>::operator == (U* p) const
  {
    return m_ptr == p;
  }

  template<typename T>
  template <typename U>
  AGX_FORCE_INLINE bool observer_ptr<T>::operator != (U* p) const
  {
    return m_ptr != p;
  }

  template<typename T>
  AGX_FORCE_INLINE bool observer_ptr<T>::operator < (const observer_ptr& rp) const
  {
    return (m_ptr < rp.m_ptr);
  }

  template<typename T>
  AGX_FORCE_INLINE T& observer_ptr<T>::operator*() const
  {
    return *m_ptr;
  }

  template<typename T>
  AGX_FORCE_INLINE T* observer_ptr<T>::operator->() const
  {
    return m_ptr;
  }

  template<typename T>
  AGX_FORCE_INLINE T* observer_ptr<T>::get() const
  {
    return m_ptr;
  }

  template<typename T>
  AGX_FORCE_INLINE bool observer_ptr<T>::operator!() const
  {
    return m_ptr == 0;  // not required
  }

  template<typename T>
  AGX_FORCE_INLINE bool observer_ptr<T>::isValid() const
  {
    return m_ptr != 0;
  }



}

#endif
