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

#ifndef AGX_POINTER_H
#define AGX_POINTER_H



#include <agx/Object.h>
#include <agxData/Value.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(Pointer);
  AGX_DECLARE_VECTOR_TYPES(Pointer);

  /**
  A pointer to an agx::Object. The pointer itself is an object
  and can have a name and be inserted into the system hierarchy.
  */
  class AGXCORE_EXPORT Pointer : public agxData::Value, public Object::EventListener
  {
  public:
    static agx::Model *ClassModel();
    static Pointer *load(TiXmlElement *ePointer, Device *device);
    virtual void configure(TiXmlElement *ePointer) override;

  public:
    Pointer(const agx::Name& name, const agx::String& autoBindPath = "");
    Pointer(const agx::Name& name, agx::Object* target, const agx::String& autoBindPath = "");

    using agxData::Value::bind;

    /**
    Bind the pointer to an object.
    */
    virtual bool bind(agx::Object *object) override;

    /**
    Unbind the pointer.
    */
    void unbind();

    /**
    \return The current binding.
    */
    Object *get();
    const Object *get() const;

    template <typename T>
    T *get();

    template <typename T>
    const T *get() const;

    virtual void buildNavigationTree(agxJson::Value& eNode) const override;

  protected:
    virtual Object *getResourceImpl(const Path& path, agx::Model *model) override;
    virtual ~Pointer();

    // From Object::EventListener
    virtual void destroyCallback(agx::Object* object) override;

  private:
    bool m_hasReference;
  };



  //---------------------------------------------------------------

  template <typename T>
  class PointerT : public Pointer
  {
  public:
    PointerT(const Name& name, const String& autoBindPath = "");
    PointerT(const Name& name, T *ptr, const String& autoBindPath = "");

    T *get();
    const T *get() const;

    T *operator-> ();
    const T *operator-> () const;

    PointerT<T>& operator= (const PointerT<T>& pointer);
    PointerT<T>& operator= (T *pointer);

    operator T* ();

  protected:
    virtual ~PointerT();
  };

  #if 0
  /**
  Only use as member allocated variable.
  */
  template <typename T>
  class Ptr : public PointerT<T>
  {
  public:
    Ptr(const Name& name, const String& autoBindPath = "") : PointerT<T>(name, autoBindPath)
    {
      this->reference();
    }

    Ptr(const Name& name, T *ptr, const String& autoBindPath = "") : PointerT<T>(name, ptr, autoBindPath)
    {
      this->reference();
    }

    virtual ~Ptr()
    {
      this->forceUnbindChildren();
      // agxAssert1(this->getReferenceCount() == 1, "Can not have active reference to member allocated value!");
      this->unreference_nodelete();
      if (this->getReferenceCount() > 0)
      {
        // std::cerr << "Warning: " << this->getReferenceCount() << " active references to member allocated value \'" << this->getName() << "\' (" << this << ") in component \'" << (this->getContext() ? this->getContext()->getName() : String("nullptr")) << "\' in explicit destructor!"<< std::endl;
        this->setReferenceCount(AGX_EXPIRED_SCOPE_ALLOCATION);
      }

      if (dynamic_cast<Component *>(this->getContext()))
        static_cast<Component *>(this->getContext())->removeObject(this);
    }

    Ptr<T>& operator= (const Ptr<T>& pointer) { return this->operator=(pointer.get()); }
    Ptr<T>& operator= (T *pointer) { this->set((Object *)pointer); return *this; }
  };
  #endif



  /* Implementation */

  AGX_FORCE_INLINE Object *Pointer::get() { return agxData::Value::get<Object *>(); }
  AGX_FORCE_INLINE const Object *Pointer::get() const { return agxData::Value::get<Object *>(); }

  template <typename T>
  AGX_FORCE_INLINE T *Pointer::get() { return dynamic_cast<T *>(agxData::Value::get<Object *>()); }

  template <typename T>
  AGX_FORCE_INLINE const T *Pointer::get() const { return dynamic_cast<const T *>(agxData::Value::get<Object *>()); }


  //-----------------------------------------------------------------------------------------------------

  template <typename T>
  PointerT<T>::PointerT(const Name& name, const String& autoBindPath) : Pointer(name, autoBindPath)
  {}

  template <typename T>
  PointerT<T>::PointerT(const Name& name, T *ptr, const String& autoBindPath) : Pointer(name, ptr, autoBindPath)
  {}

  template <typename T>
  PointerT<T>::~PointerT()
  {
  }


  template <typename T>
  AGX_FORCE_INLINE T *PointerT<T>::get() { return m_value.get<T *>(); }

  template <typename T>
  AGX_FORCE_INLINE const T *PointerT<T>::get() const { return m_value.get<T *>(); }

  template <typename T>
  AGX_FORCE_INLINE PointerT<T>::operator T* ()
  {
    return this->get();
  }

  template <typename T>
  AGX_FORCE_INLINE T *PointerT<T>::operator-> () { return this->get(); }

  template <typename T>
  AGX_FORCE_INLINE const T *PointerT<T>::operator-> () const { return this->get(); }

  template <typename T>
  AGX_FORCE_INLINE PointerT<T>& PointerT<T>::operator= (const PointerT<T>& pointer)
  {
    this->set((Object *)pointer.get());
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE PointerT<T>& PointerT<T>::operator= (T *pointer)
  {
    this->set((Object *)pointer);
    return *this;
  }


  //-----------------------------------------------------------------------------------------------------

}

#endif /* _AGX_POINTER_H_ */
