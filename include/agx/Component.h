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

#ifndef AGX_COMPONENT_H
#define AGX_COMPONENT_H

#include <iostream>
#include <agx/Object.h>
#include <typeinfo>
#include <agx/Event.h>

namespace agx
{
  class ComponentVisitor;
  class Device;


  AGX_DECLARE_POINTER_TYPES(Component);
  AGX_DECLARE_VECTOR_TYPES(Component);

  /**
  A component is an object containing other objects, enabling hierarchical structuring.
  Allows multiple child objects with same name.
  */
  class AGXCORE_EXPORT Component : public Object
  {
  public:
    static agx::Model* ClassModel();

    static Component* load(const String& path, const Path& _namespace = Path());
    static Component* load(const String& path, Device* device, const Path& _namespace = Path());
    static Component* load(const Path& path, const Name& implementation, Device* device, const Path& _namespace = Path());

#ifndef SWIG
    static Component* load(TiXmlElement* eComponent, Device* device);
    virtual void configure(TiXmlElement* eComponent) override;
    virtual void snapshot(TiXmlNode* eParent, const String& directory) const override;
#endif

  public:

    /**
    Event when adding removing child objects. NOTE: Events are triggered recursively up the
    parent structure
    Example: B is added to root node A, one event is triggered (A.addObjectEvent(A, B)). Then
    when C is added to B two events are triggered (B.addObjectEvent(B, C), and A.addObjectEvent(B,C))
    */
    typedef Event2<Component *, Object *> ObjectEvent;

    ObjectEvent addObjectEvent;
    ObjectEvent removeObjectEvent;

  public:
    Component(const agx::Name& name = agx::Name(), agx::Model *model = agx::Component::ClassModel(), agx::Device *device = nullptr);

    /**
    Add an object to the component.
    \param object The object to be added
    \param assignContext True if the child get the component as the context, set to false to allow 'weak' coupled children
    */
    virtual void addObject(agx::Object *object, bool assignContext = true);

    /**
    Remove an object from the component.
    */
    virtual void removeObject(agx::Object *object);

    /**
    Remove an object with a specified name (and optional index) from the component.
    */
    void removeObject(const agx::Name& name, size_t index = 0);

    /**
    Remove all components.
    */
    void removeAllObjects();

    /**
    \return The child object with specified name (and optional index).
    */
    agx::Object *getObject(const agx::Name& name, size_t index = 0);
    const agx::Object *getObject(const agx::Name& name, size_t index = 0) const;

    agx::Object *getObject(const agx::Name& name, agx::Model *model, size_t index = 0);
    const agx::Object *getObject(const agx::Name& name, agx::Model *model, size_t index = 0) const;

    template <typename T>
    T *getObject(const agx::Name& name, size_t index = 0);

    template <typename T>
    const T *getObject(const agx::Name& name, size_t index = 0) const;

    /**
    \return The number of child objects with a specified name.
    */
    size_t getNumObjects(const agx::Name& name) const;

    /**
    \return All child objects.
    */
    const agx::ObjectRefVector& getObjects() const;

    /**
    \return All child objects of a specified type.
    */
    template <typename T>
    void getObjects(agx::ObjectPtrVector& result, bool recursive = false) const;

    void getObjects(agx::ObjectPtrVector& result, const agx::Model *model, bool recursive = false) const;


    /**
    \return The device which the component stored on.
    */
    agx::Device *getDevice();
    const agx::Device *getDevice() const;

    template <typename T> T *getDevice();
    template <typename T> const T *getDevice() const;


    virtual void rebind() override;

#ifndef SWIG
    virtual String autoComplete(const String& partialName, StringVector& matchingNames) const;
#endif

    typedef Callback1<Object*> TraverseCallback;
    void traverse( const TraverseCallback& callback );

    void traverse( ComponentVisitor* );

    virtual void printSubtree() const override;
    void printSubtree(std::ostream& stream, int depth=0) const;

    void configure(Model *model);
#ifndef SWIG
    virtual Object *getResourceImpl(const Path& path, agx::Model *model) override;

    virtual void buildNavigationTree(agxJson::Value& eNode) const override;
#endif

  protected:
    virtual ~Component();
    void setDevice(Device* device);
    String expandAutoCompletionMatch(const String& query, const StringVector& matchingNames) const;

  private:
    Object *findObject(const Name& name, Model *model, size_t index) const;
    void triggerAddEvents(Object *child);
    void triggerRemoveEvents(Object *child);

  public:
    // Same as load, resolves XmlLoaderRegistrator ambiguity
    static Component *_load(TiXmlElement *eComponent, Device *device);

  private:
    ref_ptr<Referenced> m_device; // Referenced instead of Device to avoid include loop under Windows
    ObjectRefVector m_childObjects;
    ObjectPtrTable m_objectTable;
    UInt m_rebindState;
  };




  /* Implementation */
  #if 0
  AGX_FORCE_INLINE Component *Component::getContext()
  {
    Object *parent = Object::getContext();
    agxAssert(!parent || dynamic_cast<Component *>(parent));
    return static_cast<Component *>(parent);
  }

  AGX_FORCE_INLINE const Component *Component::getContext() const { return const_cast<Component *>(this)->getContext(); }
  #endif

  AGX_FORCE_INLINE const ObjectRefVector& Component::getObjects() const { return m_childObjects; }

  AGX_FORCE_INLINE Object *Component::getObject(const Name& name, size_t index) { return this->findObject(name, nullptr, index); }
  AGX_FORCE_INLINE const Object *Component::getObject(const Name& name, size_t index) const { return this->findObject(name, nullptr, index); }

  AGX_FORCE_INLINE Object *Component::getObject(const Name& name, Model *model, size_t index) { return this->findObject(name, model, index); }
  AGX_FORCE_INLINE const Object *Component::getObject(const Name& name, Model *model, size_t index) const { return this->findObject(name, model, index); }


  template <typename T>
  AGX_FORCE_INLINE T *Component::getObject(const Name& name, size_t index)
  {
    Object *object = this->findObject(name, T::ClassModel(), index);
    T *result = dynamic_cast<T *>(object);

    #ifdef AGX_DEBUG
    if (object && !result && this->getNumObjects(name) > 1)
      std::cout << this->getPath() << ": Unable to cast child object \'" << name << "\' (" << object << ") to type \'" << typeid(T).name() << "\', there are however other children with the same name so an unintended name clash could be the problem." << std::endl;
    #endif

    return result;
  }

  template <typename T>
  AGX_FORCE_INLINE const T *Component::getObject(const Name& name, size_t index) const
  {
    return const_cast<Component *>(this)->getObject<T>(name, index);
  }


  AGX_FORCE_INLINE Device *Component::getDevice() { return reinterpret_cast<Device *>(m_device.get()); }
  AGX_FORCE_INLINE const Device *Component::getDevice() const { return const_cast<Component *>(this)->getDevice(); }

  template <typename T>
  AGX_FORCE_INLINE T *Component::getDevice()
  {
    agxAssert(dynamic_cast<T *>(m_device.get()));
    return static_cast<T *>(m_device.get());
  }

  template <typename T>
  AGX_FORCE_INLINE const T *Component::getDevice() const
  {
    agxAssert(dynamic_cast<const T *>(m_device.get()));
    return static_cast<const T *>(m_device.get());
  }


  template <typename T>
  void Component::getObjects(ObjectPtrVector& result, bool recursive) const
  {
    Component::getObjects(result, T::ClassModel(), recursive);
  }




}


#endif /* AGX_COMPONENT_H */
