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

#ifndef AGX_OBJECTWRAPPER_H
#define AGX_OBJECTWRAPPER_H

#include <agx/Object.h>

namespace agx
{
  template <typename T>
  class ObjectWrapper : public agx::Object
  {
  public:
    ObjectWrapper(const agx::Name& name, const T& value = T());

    T& get();
    const T& get() const;

    void set(const T& value);

  protected:
    virtual ~ObjectWrapper() {}

  private:
    T m_object;
  };

  #if 0
  template <typename T>
  class ObjectRefWrapper : public agx::Object
  {
  public:
    ObjectRefWrapper(const agx::Name& name);

    T *get();
    const T *get() const;

    void set(const T *value);

  protected:
    virtual ~ObjectRefWrapper() {}

  private:
    agx::ref_ptr<T> m_object;
  };
  #endif


  /* Implementation */
  template <typename T>
  ObjectWrapper<T>::ObjectWrapper(const agx::Name& name, const T& value) : Object(name), m_object(value)
  {
  }

  template <typename T>
  AGX_FORCE_INLINE T& ObjectWrapper<T>::get() { return m_object; }

  template <typename T>
  AGX_FORCE_INLINE const T& ObjectWrapper<T>::get() const { return m_object; }

  template <typename T>
  AGX_FORCE_INLINE void ObjectWrapper<T>::set(const T& value) { m_object = value; }


  /////////////////////////////////////////////////////
  #if 0
  template <typename T>
  ObjectWrapperRef<T>::ObjectWrapperRef(const agx::Name& name, T *object) : Object(name), m_object(object)
  {
  }

  template <typename T>
  AGX_FORCE_INLINE T *ObjectWrapperRef<T>::get() { return m_object; }

  template <typename T>
  AGX_FORCE_INLINE const T *ObjectWrapperRef<T>::get() const { return m_object; }

  template <typename T>
  AGX_FORCE_INLINE void ObjectWrapperRef<T>::set(T *object) { m_object = object; }
  #endif
}


#endif /* AGX_OBJECTWRAPPER_H */
