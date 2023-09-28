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

#ifndef AGXDATA_ENTITY_MODEL_H
#define AGXDATA_ENTITY_MODEL_H


#include <agx/Vector.h>
#include <agx/HashTable.h>
#include <agx/String.h>
#include <agx/Model.h>
#include <agx/agxCore_export.h>
#include <agxData/EntityPtr.h>
#include <agxData/EntityInstance.h>
#include <agxData/AttributePtr.h>
#include <agxData/Attribute.h>


#define AGX_ENTITY_NAMESPACE ""

namespace agx
{
  class TiXmlElement;
  class Device;
  class EntityParameter;
  class NameSpace;
}

namespace agxData
{
  class EntityData;

  typedef agx::Vector<EntityModel *> EntityModelPtrVector;
  typedef agx::Vector<Attribute *> AttributePtrVector;
  typedef agx::HashTable<Attribute *, EntityModel *> EntityDependencyTable;

  /** Get a entity with a specified path */
  AGXCORE_EXPORT EntityModel *getEntity(const agx::Path& path);

  /** \return True if an entity with the specified name exists */
  AGXCORE_EXPORT bool hasEntity(const agx::Path& path);

  AGX_DECLARE_POINTER_TYPES(EntityModel);

  /**
  An abstract description of a data entity stored using SOA (structure of arrays) pattern
  in a EntityStorage. The entity model contains a set of attributes defined in a
  .agxEntity XML header file.
  */
  class AGXCORE_EXPORT EntityModel : public agx::Model
  {
  public:
    static agx::Model *ClassModel();
    static ScalarAttributeT<EntityPtr> *instanceAttribute;

  public:
    static EntityModel *load(agx::TiXmlElement *eEntity, agx::Device *device);
    virtual void configure(agx::TiXmlElement *eEntity);

    static bool findEntityPath(const agx::Path& entityPath, agx::Path& fullEntityPath, agx::String& filePath, agx::String& classPath, agx::TiXmlElement *relativeRoot = nullptr, agxIO::Environment::Type envType = agxIO::Environment::RUNTIME_PATH);

    static agx::Namespace *root();




    static void setThreadSafe(bool flag);
    static bool getThreadSafe();
  public:
    typedef agx::Event1<EntityStorage *> StorageEvent;

    StorageEvent storageCreationEvent;

  public:
    EntityModel(const agx::Name& name);

    /**
    \return The parent entity.
    */
    EntityModel* getSource();
    const EntityModel* getSource() const;

    /**
    Add an attribute to the entity.
    \param attribute The attribute.
    */
    void addAttribute(Attribute *attribute);

    /**
    \return The registered attributes.
    */
    const AttributePtrVector& getAttributes() const;

    /**
    \return The registered attribute with the given name. nullptr if no such attribute is found;
    */
    Attribute* getAttribute(const agx::Name& name);
    const Attribute* getAttribute(const agx::Name& name) const;


    void setUseInstanceTable(bool flag);
    bool useInstanceTable() const;

    void setHasConstructor(bool flag);
    bool hasConstructor() const;

    /**
    \return The default storage for this entity.
    */
    EntityStorage *getDefaultStorage();


    /**
    \return The number of bytes for one entity instance.
    */
    size_t getNumBytes();
    size_t getNumBytes(agx::Device *device);

    /**
    Print the registered attributes.
    */
    void print(std::ostream& stream) const;
    void print() const;

    static EntityModel *getRootModel();
    void setBaseEntity(EntityModel *base);

    virtual void shutdownCleanup() {}

  public:
    virtual EntityData *createData(EntityStorage *storage);

    class Loader;

    template <typename T>
    class LoaderT;

    virtual void initAttributeAccessors() {}
    virtual void construct(agxData::EntityPtr) {}

  protected:
    virtual ~EntityModel();

    // void registerStorage(EntityStorage *storage);
    // void unregisterStorage(EntityStorage *storage);
    void propagateNewAttribute(Attribute *attribute);
    void addChild(EntityModel *child);
    void removeChild(EntityModel *child);
    EntityPtr createInstanceWrapper();

  private:
    friend class EntityStorage;

    struct SharedBuffer
    {
      agx::Name name;
      Format *format;
    };

    struct SharedStorage
    {
      EntityModel *entity;
      agx::Name name;
    };

    typedef agx::Vector<SharedBuffer> SharedBufferVector;
    typedef agx::Vector<SharedStorage> SharedStorageVector;
    const SharedBufferVector& getSharedBuffers() const;
    const SharedStorageVector& getSharedStorages() const;

    void addSharedBuffer(SharedBuffer buffer);
    void addSharedStorage(SharedStorage storage);

  private:
    static void initDefaultEntity();

    static bool s_threadSafe;

    // EntityModel *m_parent;
    EntityModelPtrVector m_children;
    AttributePtrVector m_attributes;
    SharedBufferVector m_sharedBuffers;
    SharedStorageVector m_sharedStorages;
    EntityStorage *m_defaultStorage;
    bool m_useInstanceTable;
    bool m_hasConstructor;
    // EntityStoragePtrVector m_activeStorages;
  };

  class AGXCORE_EXPORT EntityModel::Loader
  {
  public:
    Loader(const char *modelPath);
    virtual EntityModel *load() = 0;
    virtual ~Loader();
  };

  template <typename T>
  class EntityModel::LoaderT : public EntityModel::Loader
  {
  public:
    LoaderT(const char *modelPath) : EntityModel::Loader(modelPath) {}
    virtual EntityModel *load() { return new T(); }
    virtual ~LoaderT() {}
  };



  /* Implementation */
  AGX_FORCE_INLINE bool EntityModel::getThreadSafe() { return s_threadSafe; }
  AGX_FORCE_INLINE bool EntityModel::hasConstructor() const { return m_hasConstructor; }


  AGX_FORCE_INLINE const AttributePtrVector& EntityModel::getAttributes() const { return m_attributes; }
  AGX_FORCE_INLINE const EntityModel::SharedBufferVector& EntityModel::getSharedBuffers() const { return m_sharedBuffers; }
  AGX_FORCE_INLINE const EntityModel::SharedStorageVector& EntityModel::getSharedStorages() const { return m_sharedStorages; }

  AGX_FORCE_INLINE const EntityModel* EntityModel::getSource() const { return dynamic_cast<const EntityModel *>(Model::getSource()); }
  AGX_FORCE_INLINE EntityModel* EntityModel::getSource() { return dynamic_cast<EntityModel *>(Model::getSource()); }


  // Implementation of EntityPtr is located in EntityStorage.h due to include loops
}

AGX_TYPE_BINDING(agxData::EntityPtr, "EntityPtr")
AGX_TYPE_BINDING(agxData::EntityInstance, "EntityInstance")
AGX_TYPE_BINDING(agxData::EntityRange, "EntityRange")

#undef AGX_ENTITY_NAMESPACE
#endif /* _AGXDATA_ENTITY_MODEL_H_ */
