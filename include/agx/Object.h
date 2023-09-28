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

#ifndef AGX_OBJECT_H
#define AGX_OBJECT_H

#include <agx/Referenced.h>
#include <agx/Name.h>
#include <agx/Path.h>
#include <agx/Vector.h>
#include <agx/HashTable.h>
#include <agx/HashSet.h>

namespace agxData
{
  class Type;
}

namespace agxJson
{
  class Value;
}

namespace agx
{
  class TiXmlElement;
  class TiXmlNode;
  class TiXmlDocument;

  class Model;
  class Device;
  class Component;
  class Thread;

  // Cannot include agx/ThreadSynchronization.h because of include loop with Type.h and Model.h.
  using ReentrantMutex = std::recursive_mutex;

  AGX_DECLARE_POINTER_TYPES(Object);
  AGX_DECLARE_VECTOR_TYPES(Object);
  AGX_DECLARE_HASHSET_TYPES(Object);

  /**
  agx::Object is a refcounted object with a name. It is used together with the subclass agx::Component
  to build hierarchies of objects, which can be accessed dynamically at runtime using string paths.
  */
  class AGXCORE_EXPORT Object : public Referenced
  {
  public:
    AGX_DECLARE_POINTER_TYPES(Event);
    AGX_DECLARE_VECTOR_TYPES(Event);
    class AGXCORE_EXPORT Event : public agx::Referenced
    {
    public:
      // virtual void serialize() = 0;

    protected:
      virtual ~Event();

    private:
    };

    // Generated event interface, defined in Object.agxEvent
    #include <agx/ObjectEvent.Declaration.h>

#ifndef SWIG
    void addListener(EventListener* listener);
    void removeListener(EventListener* listener);
    bool hasListener(EventListener* listener);
#endif

  public:
    /**
    Sets if rebinding of objects should be allowed (Right now a global setting).
    Allowing it when adding larger parts to the system tree might hit performance,
    so it might be smarter to disable it first, then add, and then enable again.
    */
    static void setEnableRebind(bool flag);

    /**
    Should rebinding be allowed? Right now a global setting.
    */
    static bool getEnableRebind();

    static agx::Model *ClassModel();

    static agx::String generateName(const agx::String& bindPath);


#ifndef SWIG
    // void configure(const String& path);
    virtual void configure(TiXmlElement* /*eObject*/) {}
    virtual void save(TiXmlElement* /* eParent */) const {}
    virtual void snapshot(TiXmlNode* /* eParent */, const String& /* directory */) const {}

    static Object *load(TiXmlElement *eObject, Device *device);

    template <typename T>
    static T *load(TiXmlElement *eObject, Device *device);
#endif

    /* Load from file */
    static Object *load(const String& path, Device *device, const String& type = "", size_t instance = 0);
    static Object *load(const String& path, Device *device, const String& type, const String& attribute, const String& value);
    static void load(const String& path, Device *device, ObjectPtrVector& loadedObjects);


    /*
    typedef Callback1<TiXmlElement *> LoadCallback;
    static void load(const String& path, LoadCallback& callback);
    */

    static TiXmlDocument openDocument(const String& path);


    /* Auto-cast return value */
    template <typename T>
    static T *load(const String& path, Device *device, const String& type = "", size_t instance = 0);

    template <typename T>
    static T *load(const String& path, Device *device, const String& type, const String& attribute, const String& value);

    static const HashSet<Object *>& getActiveObjects();
  public:
    Object(const Name& name = Name(), Model *model = Object::ClassModel());

    // virtual Object *clone() const;

    /**
    \return Then name of the object.
    */
    const agx::Name& getName() const;

    /**
    Set the name of the object.
    */
    void setName(const agx::Name& name);

    /**
    \return The implementation name of the model. Alias for getModel()->getImplementationName()
    */
    const agx::Name& getImplementationName() const;

    /**
    \return name:implementation
    */
    agx::String fullName() const;

    /**
    \return The path to the object in the system tree.
    */
    agx::Path getPath() const;

    /**
    \return path:implementation
    */
    agx::String fullPath() const;

    /**
    \return A relative path from a specified root
    \param root A node along the parent path from this object.
    */
    agx::Path getPath(const agx::Object *root) const;
    agx::String fullPath(const agx::Object *root) const;


    /**
    \return The global object id.
    */
    agx::UInt32 getId() const;

    /**
    Set the object model (handled automatically).
    */
    void setModel(agx::Model *model);

    /**
    \return The object model.
    */
    agx::Model *getModel();
    const agx::Model *getModel() const;

    /**
    \return The parent node in the system tree.
    */
    agx::Object *getContext();
    const agx::Object *getContext() const;

    /**
    \return The root context.
    */
    agx::Object *getRootContext();
    const agx::Object *getRootContext() const;

    /**
    \return True if the specified node is located somewhere in the path above the object.
    */
    bool hasParent(const agx::Object *node) const;

    /**
    \return The next sibling with identical name.
    */
    agx::Object *getNextSibling();
    const agx::Object *getNextSibling() const;

    /**
    \return A resource in the subtree below this object.
    */
    agx::Object *getResource(const agx::Path& path, agx::Model *model = nullptr);
    const agx::Object *getResource(const agx::Path& path, agx::Model *model = nullptr) const;

    /**
    \return The resource with the first matching subpath starting from this node and traversing upwards to the root.
    */
    agx::Object *getAutoScopedResource(const agx::Path& path, agx::Model *model = nullptr);
    const agx::Object *getAutoScopedResource(const agx::Path& path, agx::Model *model = nullptr) const;

    ///// Templated convenience methods /////////////////////////////////////////
    template <typename T>
    T *getContext();

    template <typename T>
    const T *getContext() const;

    template <typename T>
    T *getResource(const agx::Path& path);

    template <typename T>
    const T *getResource(const agx::Path& path) const;

    template <typename T>
    T *getAutoScopedResource(const agx::Path& path);

    template <typename T>
    const T *getAutoScopedResource(const agx::Path& path) const;

    //---------------------------------------------------------------


    bool isUnique() const;

    virtual void printSubtree() const {}

#ifndef SWIG
    virtual void buildNavigationTree(agxJson::Value& eNode) const;
#endif

    // Normally handled automatically...
    virtual void setContext(agx::Object *context);
    virtual void rebind() {}
#ifndef SWIG
    virtual agx::Object *getResourceImpl(const agx::Path& path, agx::Model *model);
#endif

    void setId(UInt32 id);

  protected:
    friend class Alias;
    friend class Pointer;
    friend class Component;
    virtual ~Object();

  private:
    static bool g_shouldRebind;

  private:
    UInt32 m_id;
    Name m_name;
    Model *m_model;
    Object *m_context;
    Object *m_next;
    EventDispatch *m_eventDispatch;
  };

  typedef HashTable<Name, Object *> ObjectPtrTable;


  /**
  \return The root component.
  */
  AGXCORE_EXPORT Component* root();
  // AGXCORE_EXPORT void setRoot(Component *root);

  /**
  \return The model for a specified path. Path may be relative if a namespace is provided.
  */
  AGXCORE_EXPORT Model* getModel(const Path& path, const Path& _namespace = Path());
  AGXCORE_EXPORT Model* getModel(const Path& path, const Name& implementation, const Path& _namespace = Path());

  template <typename T>
  T* getModel(const Path& path, const Path& _namespace = Path());

  template <typename T>
  T* getModel(const Path& path, const Name& implementation, const Path& _namespace = Path());


  // May contain both name and implementation using name:implementation format
  AGXCORE_EXPORT Model* getModel(const String& path, const Path& _namespace = Path());
  AGXCORE_EXPORT Model* getModel(const char* path, const Path& _namespace = Path());


  /**
  \return The model for an xml element.
  */
  AGXCORE_EXPORT Model* getModel(TiXmlElement* element);


  /// Debugging
  AGXCORE_EXPORT void printNode(const char* path);

  // Internal use only.

  // The mutex returned by getTypeSystemMutex() must be held when calling the
  // create.+Model functions. An alternative to always taking the mutex, in case
  // that turns out be too costly, is to use the Double-Checked Locking Pattern.
  // See https://preshing.com/20130930/double-checked-locking-is-fixed-in-cpp11/
  AGXCORE_EXPORT void createBuiltinModel(const char* path, Model* parent, Model*& result);  // NOLINT
  AGXCORE_EXPORT void createImplementedModel(const char* path, Model*& result);
  AGXCORE_EXPORT agx::ReentrantMutex& getTypeSystemMutex();

  #define AGX_MODEL_IMPLEMENTATION(_class, _path)                               \
  agx::Model *_class::ClassModel()                                              \
  {                                                                             \
    ScopeLock<ReentrantMutex> lock(agx::getTypeSystemMutex());                  \
    static agx::Model *s_model = nullptr;                                       \
                                                                                \
    if (!s_model)                                                               \
      agx::createImplementedModel(_path, s_model);                              \
                                                                                \
    if (!s_model)                                                               \
      LOGGER_ERROR() << agx::String::format("No model found for path \'%s\'. Make sure AGX environment is properly setup", _path) << LOGGER_ENDL(); \
    return s_model;                                                             \
  }


  #define AGX_BUILTIN_MODEL(_class, _path, _parent)             \
  agx::Model *_class::ClassModel()                              \
  {                                                             \
    ScopeLock<ReentrantMutex> lock(agx::getTypeSystemMutex());  \
    static agx::Model *s_model = nullptr;                       \
                                                                \
    if (!s_model)                                               \
      agx::createBuiltinModel(_path, _parent, s_model);         \
                                                                \
    return s_model;                                             \
  }



  /* Implementation */
  template <typename T>
  T *getModel(const Path& path, const Path& _namespace) { return dynamic_cast<T *>(agx::getModel(path, _namespace)); }

  template <typename T>
  T *getModel(const Path& path, const Name& implementation, const Path& _namespace) { return dynamic_cast<T *>(agx::getModel(path, implementation, _namespace)); }

  AGX_FORCE_INLINE const Name& Object::getName() const { return m_name; }
  AGX_FORCE_INLINE UInt32 Object::getId() const { return m_id; }
  AGX_FORCE_INLINE Model *Object::getModel() { return m_model; }
  AGX_FORCE_INLINE const Model *Object::getModel() const { return m_model; }
  AGX_FORCE_INLINE Object *Object::getContext() { return m_context; }
  AGX_FORCE_INLINE const Object *Object::getContext() const { return m_context; }
  AGX_FORCE_INLINE bool Object::isUnique() const { return !m_next; }


  template <typename T>
  T *Object::load(const String& path, Device *device, const String& type, size_t instance)
  {
    Object *object = Object::load(path, device, type, instance);
    return dynamic_cast<T *>(object);
  }

  template <typename T>
  T *Object::load(const String& path, Device *device, const String& type, const String& attribute, const String& value)
  {
    Object *object = Object::load(path, device, type, attribute, value);
    return dynamic_cast<T *>(object);
  }

  template <typename T>
  T *Object::load(TiXmlElement *eObject, Device *device)
  {
    Object *object = Object::load(eObject, device);
    return dynamic_cast<T *>(object);
  }


  template <typename T>
  AGX_FORCE_INLINE T *Object::getContext()
  {
    return dynamic_cast<T *>(this->getContext());
  }

  template <typename T>
  AGX_FORCE_INLINE const T *Object::getContext() const
  {
    return dynamic_cast<const T *>(this->getContext());
  }


  template <typename T>
  T *Object::getResource(const Path& path)
  {
    Object *object = this->getResource(path, T::ClassModel());
    return (object != nullptr && object->is<T>()) ? object->as<T>() : nullptr;
  }

  template <typename T>
  const T *Object::getResource(const Path& path) const
  {
    return const_cast<Object *>(this)->getResource<T>(path);
  }


  template <typename T>
  T *Object::getAutoScopedResource(const Path& path)
  {
    Object *object = this->getAutoScopedResource(path, T::ClassModel());
    return (object != nullptr && object->is<T>()) ? object->as<T>() : nullptr;
  }

  template <typename T>
  const T *Object::getAutoScopedResource(const Path& path) const
  {
    return const_cast<Object *>(this)->getAutoScopedResource<T>(path);
  }
}

#include <agx/ObjectEvent.Implementation.h>

// AGX_TYPE_BINDING(agx::Object *, "ObjectPointer")

#endif /* _AGX_OBJECT_H_ */
