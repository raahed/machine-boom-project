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

#ifndef AGXDATA_SCALAR_H
#define AGXDATA_SCALAR_H

#include <agxData/Data.h>
#include <agxData/Type.h>
#include <agx/config/AGX_DEBUG_INTRUSIVE.h>

namespace agxData
{
  /**
  Abstract representation of a scalar variable.
  */
  class AbstractScalar : public Data
  {
  public:
    AbstractScalar();
    AbstractScalar(void *storage);
    AbstractScalar(const AbstractScalar& other, agx::IndexRange localRange);

    /// \return The data value
    template <typename T>
    T& get();

    /// \return The data value
    template <typename T>
    const T& get() const;

    /// \return 1 (Used for template programming)
    size_t size() const;

    /// Set the memory area for the scalar
    void setStorage(void *storage);
  };

  //---------------------------------------------------------------

  /// Templated scalar
  template <typename T>
  class Scalar : public AbstractScalar
  {
  public:
    typedef T Type;
    static const bool IsScalar = true;

  public:
    Scalar(const T& value = T());
    Scalar(void *storage);
    Scalar(const Scalar<T>& other, agx::IndexRange localRange);

    /// \return The data value
    T& get();
    const T& get() const;

    operator T& ();
    operator const T& () const;

    Scalar<T>& operator= (const Scalar<T>& value);
    Scalar<T>& operator= (const T& value);

    /// \return The data value (Used for template programming)
    T& operator[] (size_t index);
    const T& operator[] (size_t index) const;

  private:
    T m_value;
  };




  /* Implementation */

  //---------------------------------------------------------------

  AGX_FORCE_INLINE AbstractScalar& Data::asScalar() { return *static_cast<AbstractScalar *>(this); }
  AGX_FORCE_INLINE const AbstractScalar& Data::asScalar() const { return *static_cast<const AbstractScalar *>(this); }

  template <typename T>
  AGX_FORCE_INLINE Scalar<T>& Data::asScalar() { return *static_cast<Scalar<T> *>(this); }

  template <typename T>
  AGX_FORCE_INLINE const Scalar<T>& Data::asScalar() const { return *static_cast<const Scalar<T> *>(this); }

  //---------------------------------------------------------------


  AGX_FORCE_INLINE AbstractScalar::AbstractScalar()
  {}

  AGX_FORCE_INLINE AbstractScalar::AbstractScalar(void *storage) : Data(storage)
  {}

  AGX_FORCE_INLINE AbstractScalar::AbstractScalar(const AbstractScalar& other, agx::IndexRange) : Data(other.m_ptr)
  {}

  AGX_FORCE_INLINE size_t AbstractScalar::size() const { return 1; }

  // AGX_FORCE_INLINE void *AbstractScalar::get() { return m_ptr; }
  // AGX_FORCE_INLINE const void *AbstractScalar::get() const { return m_ptr; }

  template <typename T>
  AGX_FORCE_INLINE T& AbstractScalar::get()
  {
    agxAssert(m_ptr);
    #if AGX_DEBUG_INTRUSIVE()
    agxAssert(agxData::getFormat<T>() == this->getFormat());
    #endif

    return *static_cast<T *>(m_ptr);
  }

  template <typename T>
  AGX_FORCE_INLINE const T& AbstractScalar::get() const
  {
    agxAssert(m_ptr);
    #if AGX_DEBUG_INTRUSIVE()
    agxAssert(agxData::getFormat<T>() == this->getFormat());
    #endif

    return *static_cast<const T *>(m_ptr);
  }


  AGX_FORCE_INLINE void AbstractScalar::setStorage(void *storage) { m_ptr = storage; }

  //---------------------------------------------------------------

  template <typename T>
  AGX_FORCE_INLINE Scalar<T>::Scalar(const T& value) : AbstractScalar(&m_value, agxData::getFormat<T>()), m_value(value) {}

  template <typename T>
  AGX_FORCE_INLINE Scalar<T>::Scalar(void *storage) : AbstractScalar(storage, agxData::getFormat<T>()) {}

  template <typename T>
  AGX_FORCE_INLINE Scalar<T>::Scalar(const Scalar<T>& other, agx::IndexRange localRange) : AbstractScalar(other, localRange)
  {
  }

  template <typename T>
  AGX_FORCE_INLINE T& Scalar<T>::get() { return *(T *)m_ptr; }

  template <typename T>
  AGX_FORCE_INLINE const T& Scalar<T>::get() const { return *(T *)m_ptr; }

  template <typename T>
  AGX_FORCE_INLINE T& Scalar<T>::operator[] (size_t /* index */) { return *(T *)m_ptr; }

  template <typename T>
  AGX_FORCE_INLINE const T& Scalar<T>::operator[] (size_t /* index */) const { return *(T *)m_ptr; }

  template <typename T>
  AGX_FORCE_INLINE Scalar<T>::operator T& () { return *(T *)m_ptr; }

  template <typename T>
  AGX_FORCE_INLINE Scalar<T>::operator const T& () const { return *(T *)m_ptr; }

  template <typename T>
  AGX_FORCE_INLINE Scalar<T>& Scalar<T>::operator= (const Scalar<T>& value) { *(T *)m_ptr = *(T *)value.m_ptr; return *this; }

  template <typename T>
  AGX_FORCE_INLINE Scalar<T>& Scalar<T>::operator= (const T& value) { *(T *)m_ptr = value; return *this; }

}

AGX_TEMPLATED_TYPE_BINDING(agxData::Scalar, "Scalar")



#endif /* AGXDATA_SCALAR_H */
