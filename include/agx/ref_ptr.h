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

#ifndef AGX_REF_PTR_H
#define AGX_REF_PTR_H

#include <agx/macros.h>
#include <agx/hash.h>
#include <agx/HashFunction.h>

namespace agx
{

  /** Smart pointer for handling referenced counted objects.*/
  template<class T>
  class ref_ptr
  {
  public:

#if !(defined(AGXJAVA) || defined(AGXPYTHON))

      /// The type of the referenced class
      typedef T element_type;

      /// Constructor
      ref_ptr();

      /// Will take ownership and reference \p ptr
      ref_ptr(T* ptr);

      /// Copy constructor. Will increment reference count with one
      ref_ptr(const ref_ptr& rp);

      /// Destructor, will decrement reference count with one
      ~ref_ptr();

      /// Assignment operator, will increment reference count
      ref_ptr& operator = (const ref_ptr& rp);

      /// Assignment operator, will reference \p and increment its reference count with one
      ref_ptr& operator = (T* ptr);

      /// Cast operator that will cast the ref_ptr to its native type
      operator T* () const;

      /// \return true if the reference pointer is pointing to the same object as \p rp
      template <typename U>
      bool operator == (ref_ptr<U> const& rp) const;

      /// \return true if the reference pointer is NOT pointing to the same object as \p rp
      template <typename U>
      bool operator != (ref_ptr<U> const& rp) const;

      /// \return true if the reference pointer is pointing to the same object \p p
      template <typename U>
      bool operator == (U const* p) const;


      /// \return true if the reference pointer is NOT pointing to the same object as \p p
      template <typename U>
      bool operator != (U const* p) const;

      /// \return true if the reference pointer is pointing to the same object as \p p
      template <typename U>
      bool operator == (U* p) const;

      /// \return true if the reference pointer is NOT pointing to the same object as \p p
      template <typename U>
      bool operator != (U* p) const;

      /// \return true if the pointer this is referencing to is less than the one \p rp is referencing.
      bool operator < (const ref_ptr& rp) const;

      /// \return a reference to the referenced object
      T& operator*() const;

      /// automatic cast to native pointer and use the dereferencing operator
      T* operator->() const;

      /// \return a native pointer to the referenced object
      T* get() const;

      /// \return true if the referenced pointer is == null
      bool operator!() const;

      /// \return true if the referenced pointer is != null
      bool isValid() const;

      /// Release the reference (without decrementing) to the referenced object and return a native pointer.
      T* release();

      /// Force the pointer to the referenced object to be null. Without deleting or decrement counters. Use with caution!
      void forceClear();

      /**
      Swap \p rb with this reference without changing reference count.
      */
      void swap(ref_ptr& rp);

#endif //AGXJAVA || AGXPYTHON

    private:
      T* m_ptr;
  };

  template <typename T, typename U>
  AGX_FORCE_INLINE  bool operator == (T const* p, ref_ptr<U> const& rp)
  {
    return (rp == p);
  }

  template <typename T, typename U>
  AGX_FORCE_INLINE bool operator != (T const* p, ref_ptr<U> const& rp)
  {
    return !(rp == p);
  }

  template <typename T, typename U>
  AGX_FORCE_INLINE bool operator == (T* p, ref_ptr<U> const& rp)
  {
    return (rp == p);
  }

  template <typename T, typename U>
  AGX_FORCE_INLINE bool operator != (T* p, ref_ptr<U> const& rp)
  {
    return !(rp == p);
  }


  template <typename T>
  AGX_FORCE_INLINE ref_ptr<T>::ref_ptr() : m_ptr(0) {}

  template <typename T>
  AGX_FORCE_INLINE ref_ptr<T>::ref_ptr(T* ptr) : m_ptr(ptr)
  {
    if (m_ptr) m_ptr->reference(this);
  }

  template <typename T>
  AGX_FORCE_INLINE ref_ptr<T>::ref_ptr(const ref_ptr<T>& rp) : m_ptr(rp.m_ptr)
  {
    if (m_ptr) m_ptr->reference(this);
  }

  template <typename T>
  AGX_FORCE_INLINE ref_ptr<T>::~ref_ptr()
  {
    if (m_ptr) m_ptr->unreference(this);
    m_ptr = 0;
  }

  template <typename T>
  AGX_FORCE_INLINE ref_ptr<T>& ref_ptr<T>::operator = (const ref_ptr<T>& rp)
  {
    if (m_ptr == rp.m_ptr) return *this;
    T* tmp_ptr = m_ptr;
    m_ptr = rp.m_ptr;
    if (m_ptr) m_ptr->reference(this);
    // unref second to prevent any deletion of any object which might
    // be referenced by the other object. i.e rp is child of the
    // original m_ptr.
    if (tmp_ptr) tmp_ptr->unreference(this);
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE ref_ptr<T>& ref_ptr<T>::operator = (T* ptr)
  {
    if (m_ptr == ptr) return *this;
    T* tmp_ptr = m_ptr;
    m_ptr = ptr;
    if (m_ptr) m_ptr->reference(this);
    // unref second to prevent any deletion of any object which might
    // be referenced by the other object. i.e rp is child of the
    // original m_ptr.
    if (tmp_ptr) tmp_ptr->unreference(this);
    return *this;
  }

  /// Cast operator that will cast the ref_ptr to its native type
  template <typename T>
  AGX_FORCE_INLINE  ref_ptr<T>::operator T* () const
  {
    return m_ptr;
  }

  template <typename T> template <typename U>
  AGX_FORCE_INLINE  bool ref_ptr<T>::operator == (ref_ptr<U> const& rp) const
  {
    return m_ptr == rp.get();
  }

  template <typename T> template <typename U>
  AGX_FORCE_INLINE  bool ref_ptr<T>::operator != (ref_ptr<U> const& rp) const
  {
    return m_ptr != rp.get();
  }

  template <typename T> template <typename U>
  AGX_FORCE_INLINE  bool ref_ptr<T>::operator == (U const* p) const
  {
    return m_ptr == p;
  }

  template <typename T> template <typename U>
  AGX_FORCE_INLINE  bool ref_ptr<T>::operator != (U const* p) const
  {
    return m_ptr != p;
  }

  template <typename T> template <typename U>
  AGX_FORCE_INLINE  bool ref_ptr<T>::operator == (U* p) const
  {
    return m_ptr == p;
  }

  template <typename T> template <typename U>
  AGX_FORCE_INLINE  bool ref_ptr<T>::operator != (U* p) const
  {
    return m_ptr != p;
  }

  template <typename T>
  AGX_FORCE_INLINE bool ref_ptr<T>::operator < (const ref_ptr<T>& rp) const
  {
    return (m_ptr < rp.m_ptr);
  }

  template <typename T>
  AGX_FORCE_INLINE T& ref_ptr<T>::operator*() const
  {
    return *m_ptr;
  }

  template <typename T>
  AGX_FORCE_INLINE T* ref_ptr<T>::operator->() const
  {
    return m_ptr;
  }

  template <typename T>
  AGX_FORCE_INLINE T* ref_ptr<T>::get() const
  {
    return m_ptr;
  }

  template <typename T>
  AGX_FORCE_INLINE bool ref_ptr<T>::operator!() const
  {
    return m_ptr == 0;  // not required
  }

  template <typename T>
  AGX_FORCE_INLINE bool ref_ptr<T>::isValid() const
  {
    return m_ptr != 0;
  }

  template <typename T>
  AGX_FORCE_INLINE T* ref_ptr<T>::release()
  {
    T* tmp = m_ptr;
    if (m_ptr) m_ptr->unreference_nodelete();
    m_ptr = 0;
    return tmp;
  }

  template <typename T>
  AGX_FORCE_INLINE void ref_ptr<T>::forceClear()
  {
    m_ptr = nullptr;
  }

  template <typename T>
  AGX_FORCE_INLINE void ref_ptr<T>::swap(ref_ptr<T>& rp)
  {
    T* tmp = m_ptr;
    m_ptr = rp.m_ptr;
    rp.m_ptr = tmp;
  }

  // Hash function
  template<typename T>
  struct HashFn< ref_ptr<T> >
  {
    AGX_FORCE_INLINE UInt32 operator()(const ref_ptr<T>& key) const
    {
      return agx::hash(key.get());
    }

    typedef const T *PtrT;
    AGX_FORCE_INLINE UInt32 operator()(const PtrT key) const
    {
      return agx::hash(key);
    }
  };


  template<class Type>
  struct RefPtrAccessor {
    const Type* operator()( agx::ref_ptr<Type> a ) const {
      return a.get();
    }
  };


  template<class T> AGX_FORCE_INLINE
  T* get_pointer(const ref_ptr<T>& rp)
  {
    return rp.get();
  }

  template<class T, class Y> AGX_FORCE_INLINE
  ref_ptr<T> static_pointer_cast(const ref_ptr<Y>& rp)
  {
    return static_cast<T*>(rp.get());
  }

  template<class T, class Y> AGX_FORCE_INLINE
  ref_ptr<T> dynamic_pointer_cast(const ref_ptr<Y>& rp)
  {
    return dynamic_cast<T*>(rp.get());
  }

  template<class T, class Y> AGX_FORCE_INLINE
  ref_ptr<T> const_pointer_cast(const ref_ptr<Y>& rp)
  {
    return const_cast<T*>(rp.get());
  }



}

namespace std
{
  template<class T> AGX_FORCE_INLINE
  void swap(agx::ref_ptr<T>& rp1, agx::ref_ptr<T>& rp2)
  {
    rp1.swap(rp2);
  }
}


#endif
