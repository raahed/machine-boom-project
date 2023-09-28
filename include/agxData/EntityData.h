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

#ifndef AGXDATA_ENTITYDATA_H
#define AGXDATA_ENTITYDATA_H

#include <agx/Object.h>
#include <agx/GlobalResult.h>
#include <agxData/EntityPtr.h>
#include <agxData/EntityInstance.h>
#include <agxData/Array.h>
#include <agxData/Buffer.h>

namespace agx
{
  class TiXmlElement;
  class EntityDataParameter;
}


namespace agxData
{
  class EntityModel;

  DOXYGEN_START_INTERNAL_BLOCK()

  AGX_DECLARE_POINTER_TYPES(EntityData);

  /**
  Accessor for entity data in a EntityStorage. Mainly used in compute kernels.
  */
  class AGXCORE_EXPORT EntityData : public agx::Object
  {
  public:
    static agx::Model *ClassModel();

  public:

    /**
    \return The number of elements in the data set.
    */
    agx::UInt numElements();

    /**
    \return The storage where data is stored.
    */
    EntityStorage *getStorage();

    /**
    \return The data model.
    */
    EntityModel *getModel();

    /**
    \return The active global result, otherwise nullptr.
    */
    agx::GlobalResult *getGlobalResult();

    /**
    \return The global, deterministic, ordering, when iterating this
    dataset. Only valid if previously written using a global result.
    */
    agx::Index getGlobalOrder(size_t index);


    /**
    Convert id to index.
    */
    size_t idToIndex(size_t id) const;

    /**
    Convert index to id.
    */
    size_t indexToId(size_t index) const;

    /**
    Explicitly sync the data set. Normally handled automatically.
    */
    void sync();


    // Global result convenience wrapper
    agx::GlobalResult::Transaction allocateResult(size_t numElements);


    // Access to the instance attribute
    Array< EntityPtr > instance;



    // Create a new instance
    agx::Index createInstance();
    agx::Index createInstances(agx::Index numInstances);
    void presize(agx::Index size);

    // Set the batch size for creating instances
    void setInstanceBatchSize(agx::Index size);

    // Commit the created instances, automatically done when a task/kernel completes
    void commitInstanceBatch();

  protected:
    friend class EntityStorage;
    friend class EntityModel;
    friend class agx::EntityDataParameter;
    friend class agx::GlobalResultStorage;

    EntityData(EntityStorage *storage);
    EntityData();
    virtual ~EntityData();

    void setGlobalResult(agx::GlobalResult *globalResult);
    void setGlobalOrderBuffer(Buffer *buffer);

    void bind(const Attribute *attribute, Buffer *buffer);
    void bind(const SharedAttribute *, agxData::Value *value);
    void realloactionCallback(Buffer *buffer);
    void registerArray(const Attribute *attribute, AbstractArray *array);
    void unregisterArray(const Attribute *attribute);

    virtual void setNumElements(agx::UInt32 numElements);
    virtual void synchronizeSharedAttribute(agxData::Value *) {};

  private:
    agx::UInt m_numElements;
    EntityStorage *m_storage;
    agx::GlobalResult *m_globalResult;
    agx::Vector<AbstractArray *> m_bindings;
    agxData::Value::Event::CallbackType m_sharedAttributeUpdateCallback;
    Buffer::Event::CallbackType m_bufferReallocationCallback;
    IndexArray m_globalOrder;
    agx::Index m_instanceBatchSize;
    bool m_hasBatchInstances;
  };

  DOXYGEN_END_INTERNAL_BLOCK()


  typedef agxData::EntityData *EntityDataPtr;


  /* Implementation */
  DOXYGEN_START_INTERNAL_BLOCK()

  AGX_FORCE_INLINE agx::UInt EntityData::numElements() { return m_numElements; }
  AGX_FORCE_INLINE EntityStorage *EntityData::getStorage() { return m_storage; }
  AGX_FORCE_INLINE agx::GlobalResult *EntityData::getGlobalResult() { return m_globalResult; }
  AGX_FORCE_INLINE void EntityData::setGlobalResult(agx::GlobalResult *globalResult) { m_globalResult = globalResult; }
  AGX_FORCE_INLINE agx::GlobalResult::Transaction EntityData::allocateResult(size_t numElements)
  {
    agxVerifyN(m_globalResult, "%s: No global result is active", getPath().c_str());
    return m_globalResult->allocateResult(numElements);
  }

  AGX_FORCE_INLINE agx::Index EntityData::getGlobalOrder(size_t index)
  {
    agxAssertN(m_globalOrder.buffer(), "%s: Data set does not have a global ordering buffer", getPath().c_str());
    return m_globalOrder[index];
  }
  DOXYGEN_END_INTERNAL_BLOCK()

  // indexToId, idToIndex, and createInstance implemented in EntityStorage.h due to include loop
}


#endif /* AGXDATA_ENTITYDATA_H */
