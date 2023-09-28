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

#ifndef AGXDATA_ENTITYPTR_H
#define AGXDATA_ENTITYPTR_H

#ifdef _MSC_VER
# pragma warning( disable: 4714 )// const' marked as __forceinline not inlined
#endif

#include <agx/agx.h>
#include <agx/Vector.h>
#include <agx/IndexRange.h>
#include <agxData/EntityInstance.h>
#include <iosfwd>

namespace agxData
{
  class EntityStorage;
  class EntityModel;
  class EntityData;

  class Attribute;
  template <typename T> class ScalarAttributeT;
  template <typename T> class PointerAttributeT;
  template <typename T> class VectorAttributeT;
  template <typename T> class HashSetAttributeT;
  template <typename T> class ArrayAttributeT;
  template <typename T> class SharedAttributeT;

  /**
  Accessor to a entity instance stored in a EntityStorage, to allow SOA (structure of
  arrays) data to be accessed using a AOS (array of structures) pattern. The entity
  pointer has a indirect id to its data inside the storage. The actual array index where
  the data is stored may change as the storage is reordered, but the EntityPtr is valid
  as long as the instance is not destroyed or transferred to another storage.
  */
  class AGXCORE_EXPORT EntityPtr
  {
  public:
    EntityPtr();
    EntityPtr(agxData::EntityStorage *storage, agx::Index id);
    EntityPtr(const agxData::EntityPtr& other) = default;
    EntityPtr(const agxData::EntityInstance& instance);

    ~EntityPtr() = default;

    EntityPtr& operator=( const EntityPtr& other ) = default;
    EntityPtr& operator=( EntityPtr&& ) = default;

    static EntityPtr createWithIndex(EntityStorage *storage, agx::Index index);

    /**
    \return The storage in which the entity is instantiated. nullptr if invalid pointer.
    */
    agxData::EntityStorage *getStorage() const;

    /**
    \return The data set for this entity.
    */
    agxData::EntityData *getData();
    const agxData::EntityData *getData() const;

    template <typename T>
    T& dataSet();

    template <typename T>
    const T& dataSet() const;

    /**
    \return an entity instance accessor.
    */
    agxData::EntityInstance instance();
    const agxData::EntityInstance instance() const;

    /**
    \return The id of the instance.
    */
    agx::Index getId() const;

    /**
    Set the target storage and id.
    */
    void init(agxData::EntityStorage *storage, agx::Index id);

    /**
    Invalidate the entity pointer.
    */
    void invalidate();

    /**
    Calculate the index using the indirect id.
    */
    agx::Index calculateIndex() const;



    /**
    \return True if the pointer is valid (non-nullptr)
    */
    bool isValid() const;

    /**
    Check for broken ptr, by verifying that the id is active in the storage. Only used in debug mode.
    */
    void verifyIndex() const;

    /**
    \return The entity model for this instance.
    */
    agxData::EntityModel* getModel();
    const agxData::EntityModel* getModel() const;

    /**
    \return True if the pointer is an instance of a specified model or sub-model
    */
    bool isInstanceOf(agxData::EntityModel *model) const;

    /**
    Remove the entity from the active storage.
    */
    void destroy();

    /**
    Transfer instance to another storage.
    */
    void transfer(agxData::EntityStorage *target);


    /**
    Create a copy instance in the same storage.
    */
    agxData::EntityPtr copy();

    /**
    Copy instance to another storage.
    */
    agxData::EntityPtr copy(agxData::EntityStorage *target);

    /**
    Copy instance attributes to an existing target instance.
    */
    void copyAttributes(agxData::EntityPtr source);

    /**
    Swap id with another instance. Both must be part of the same storage.
    */
    void swapId(EntityPtr& other);

    /**
    Swap index with another instance. Both must be part of the same storage.
    */
    void swapIndex(EntityPtr& other);

    /**
    Access the value of a specific attribute.
    */
    template <typename T>
    T& operator() (const agxData::ScalarAttributeT<T> *attribute);

    template <typename T>
    const T& operator() (const agxData::ScalarAttributeT<T> *attribute) const;

    /** Pointer attribute */
    template <typename T>
    T& operator() (const agxData::PointerAttributeT<T> *attribute);

    template <typename T>
    const T& operator() (const agxData::PointerAttributeT<T> *attribute) const;

    /** Vector attribute */
    template <typename T>
    typename agxData::VectorAttributeT<T>::Type& operator() (const agxData::VectorAttributeT<T> *attribute);

    template <typename T>
    const typename agxData::VectorAttributeT<T>::Type& operator() (const agxData::VectorAttributeT<T> *attribute) const;

    /** HashSet attribute */
    template <typename T>
    typename agxData::HashSetAttributeT<T>::Type& operator() (const agxData::HashSetAttributeT<T> *attribute);

    template <typename T>
    const typename agxData::HashSetAttributeT<T>::Type& operator() (const agxData::HashSetAttributeT<T> *attribute) const;

    /** Array attribute */
    template <typename T>
    typename agxData::ArrayAttributeT<T>::Type& operator() (const agxData::ArrayAttributeT<T> *attribute);

    template <typename T>
    const typename agxData::ArrayAttributeT<T>::Type& operator() (const agxData::ArrayAttributeT<T> *attribute) const;

    /** Shared attribute */
    template <typename T>
    T& operator() (const agxData::SharedAttributeT<T> *attribute);

    template <typename T>
    const T& operator() (const agxData::SharedAttributeT<T> *attribute) const;

    // Hash function
    agx::UInt32 hash() const;


    bool operator!() const;
    operator bool() const;

    bool operator<( const EntityPtr& rhs ) const;
    bool operator>( const EntityPtr& rhs ) const;

    bool operator==( const EntityPtr& rhs ) const;
    bool operator!=( const EntityPtr& rhs ) const;

    void print(std::ostream& stream) const;
    void print() const;

    static void TransformToAscii(void *targetBuffer, const void *sourceBuffer, size_t numElements);

  private:
    template <typename T>
    friend class AttributePtr;

    template <typename T>
    T& getElement(const Attribute *attribute) const;

  private:
    EntityStorage *m_storage;
    agx::Index m_id;
  };

  typedef agx::VectorPOD<EntityPtr> EntityPtrVector;

  AGXCORE_EXPORT std::ostream& operator << (std::ostream& output, const EntityPtr& entity);


  class AGXCORE_EXPORT EntityRange : public agx::IndexRange
  {
  public:
    EntityRange();
    EntityRange(EntityStorage *storage, size_t start, size_t end);

    EntityPtr operator[] (size_t index);
    EntityPtr at(size_t index);
    EntityRange operator[] (agx::IndexRange subRange);

  private:
    EntityStorage *m_storage;
  };



  /* Implementation */
  AGX_FORCE_INLINE EntityPtr::EntityPtr() : m_storage(0), m_id(agx::InvalidIndex) {}
  AGX_FORCE_INLINE EntityPtr::EntityPtr(EntityStorage *storage, agx::Index id) { this->init(storage, id); }



  inline EntityInstance EntityPtr::instance() { return EntityInstance(getData(), calculateIndex()); }
  inline const EntityInstance EntityPtr::instance() const { return EntityInstance(const_cast<EntityData *>(getData()), calculateIndex()); }

  AGX_FORCE_INLINE EntityStorage *EntityPtr::getStorage() const { return m_storage; }
  AGX_FORCE_INLINE agx::Index EntityPtr::getId() const { return m_id; }

  AGX_FORCE_INLINE void EntityPtr::invalidate()
  {
    m_storage = nullptr;
    m_id = agx::InvalidIndex;
  }

  AGX_FORCE_INLINE agx::UInt32 EntityPtr::hash() const
  {
    return agx::hash( agx::hash( m_storage ), agx::hash( m_id ) );
  }


  AGX_FORCE_INLINE bool EntityPtr::isValid() const { return m_storage != nullptr; }
  AGX_FORCE_INLINE bool EntityPtr::operator!() const { return !this->isValid(); }
  AGX_FORCE_INLINE EntityPtr::operator bool() const { return this->isValid(); }

  AGX_FORCE_INLINE bool EntityPtr::operator<( const EntityPtr& rhs ) const
  {
    return this->getId() < rhs.getId();
  }

  AGX_FORCE_INLINE bool EntityPtr::operator>( const EntityPtr& rhs ) const
  {
    return this->getId() > rhs.getId();
  }

  AGX_FORCE_INLINE bool EntityPtr::operator==( const EntityPtr& rhs ) const
  {
    return (this->getStorage() == rhs.getStorage()) && (this->getId() == rhs.getId());
  }

  AGX_FORCE_INLINE bool EntityPtr::operator!=( const EntityPtr& rhs ) const
  {
    return (this->getStorage() != rhs.getStorage()) || (this->getId() != rhs.getId());
  }

}


#endif /* AGXDATA_ENTITYPTR_H */
