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

#ifndef AGXDATA_ENTITYREF_H
#define AGXDATA_ENTITYREF_H

#include <agx/AtomicValue.h>
#include <agxData/EntityPtr.h>

namespace agxData
{
  /**
  An entity reference is a safe/referencing handle to an entity instance whose model
  is descendant from Referenced.agxEntity

  1. Provides shared ownership of an entity instance in a storage.
  2. It also holds a strong retain to the storage in which the entity is allocated.
  3. It is automatically updated when an entity is transferred from one storage to another.
  */
  template <typename PtrT>
  class EntityRef : public PtrT
  {
  public:
    EntityRef();
    EntityRef(agxData::EntityStorage* storage, agx::Index id);
    EntityRef(const agxData::EntityRef<PtrT>& other);
    EntityRef(const agxData::EntityPtr& other);
    ~EntityRef();

    void init(agxData::EntityStorage* storage, agx::Index id);
    EntityRef<PtrT>& operator= (const EntityRef<PtrT>& other);

  private:
    void retain();
    void release();
  };





  /* Implementation */
  template <typename PtrT>
  AGX_FORCE_INLINE EntityRef<PtrT>::EntityRef() {}

  template <typename PtrT>
  AGX_FORCE_INLINE EntityRef<PtrT>::EntityRef(agxData::EntityStorage* storage, agx::Index id) : PtrT(storage, id)
  {
    this->retain();
  }

  template <typename PtrT>
  AGX_FORCE_INLINE EntityRef<PtrT>::EntityRef(const agxData::EntityRef<PtrT>& other) : PtrT(other)
  {
    this->retain();
  }

  template <typename PtrT>
  AGX_FORCE_INLINE EntityRef<PtrT>::EntityRef(const agxData::EntityPtr& other) : PtrT(other)
  {
    this->retain();
  }

  template <typename PtrT>
  AGX_FORCE_INLINE EntityRef<PtrT>::~EntityRef()
  {
    this->release();
  }

  template <typename PtrT>
  AGX_FORCE_INLINE void EntityRef<PtrT>::retain()
  {
    if (this->isValid())
    {
      this->getStorage()->reference();

      typename PtrT::InstanceType instance = this->instance();
      agx::ScopeLock<agx::SpinMutex> scopeLock(instance.observerMutex());
      instance.referenceCount()++;
      instance.observers().push_back(this);
    }
  }

  template <typename PtrT>
  AGX_FORCE_INLINE void EntityRef<PtrT>::release()
  {
    if (this->isValid())
    {
      typename PtrT::InstanceType instance = this->instance();

      agxData::EntityStorage* storage = nullptr;
      agx::UInt32 refCounter = 0;
      {
        agx::ScopeLock<agx::SpinMutex> scopeLock(instance.observerMutex());

        agx::UInt32& refCount = instance.referenceCount();
        refCount--;

        size_t index = instance.observers().find(this);
        agxAssert(index != instance.observers().size());
        instance.observers().eraseFast(index);

        storage = this->getStorage();
        refCounter = refCount;
      }

      if (refCounter == 0)
        this->destroy();

      storage->unreference();
    }
  }


  template <typename PtrT>
  AGX_FORCE_INLINE void EntityRef<PtrT>::init(agxData::EntityStorage* storage, agx::Index id)
  {
    this->release();
    PtrT::init(storage, id);
    this->retain();
  }


  template <typename PtrT>
  EntityRef<PtrT>& EntityRef<PtrT>::operator= (const EntityRef<PtrT>& other)
  {
    this->release();
    PtrT::operator=(other);
    this->retain();
    return* this;
  }

}


#endif /* AGXDATA_ENTITYREF_H */
