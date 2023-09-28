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

#ifndef AGXDATA_ENTITYINSTANCE_H
#define AGXDATA_ENTITYINSTANCE_H

#include <agx/agx.h>
#include <agx/Integer.h>
#include <agx/hash.h>
// #include <agxData/EntityData.h>

namespace agxData
{
  class EntityStorage;
  class EntityData;
  class EntityModel;
  class EntityPtr;
  typedef agxData::EntityData* EntityDataPtr;


  /**
  Entity instance provides access to a single instance in an EntityStorage. Behaves like
  a struct, but stored using SOA buffers. The EntityInstance is temporal, it stores a direct
  index to its position in the storage. As the storage is modified and instances are
  added/removed, existing EntityInstance variables may be invalidated. For this reason
  EntityInstance should only be used in local scopes where the instance is known to be valid.
  For other cases, use EntityPtr instead, which use indirect addressing, and is valid even
  when the storage is permuted. The downside is that data access is a bit more costly.
  */
  class AGXCORE_EXPORT EntityInstance
  {
  public:
    /// Constructors
    EntityInstance();
    EntityInstance(agxData::EntityStorage* storage, agx::Index index);
    EntityInstance(agxData::EntityData* data, agx::Index index);
    EntityInstance(const agxData::EntityInstance& other);
    EntityInstance(const agxData::EntityPtr& other);

    /// Destructor
    ~EntityInstance() = default;

    /**
    Initialize an instance.
    \param data The data set
    \param index The instance index
    */
    void init(agxData::EntityData* data, agx::Index index);

    /**
    \return The index of the instance.
    */
    agx::Index getIndex() const;

    /**
    \return The id of the instance.
    */
    agx::Index calculateId() const;

    /**
    \return The data set which the instance is part of.
    */
    agxData::EntityData* getData();
    const agxData::EntityData* getData() const;

    /**
    Alias for getData()->getStorage()
    \return The storage which the instance is part of.
    */
    agxData::EntityStorage* getStorage();
    const agxData::EntityStorage* getStorage() const;

    /**
    Alias for getData()->getStorage()->getModel()
    \return The entity model for this instance.
    */
    agxData::EntityModel* getModel();
    const agxData::EntityModel* getModel() const;

    /**
    \return True if the pointer is an instance of a specified model or sub-model
    */
    bool isInstanceOf(agxData::EntityModel* model) const;

    /**
    \return True if the instance is valid (non-nullptr)
    */
    bool isValid() const;

    /**
    Check for broken ptr, by verifying that the index is active in the storage. Only used in debug mode.
    */
    void verifyIndex() const;


    // Hash function
    agx::UInt32 hash() const;

    /// Boolean operators, implemented using isValid
    bool operator!() const;
    operator bool() const;

    bool operator<( const EntityInstance& rhs ) const;
    bool operator>( const EntityInstance& rhs ) const;

    bool operator==( const EntityInstance& rhs ) const;
    bool operator!=( const EntityInstance& rhs ) const;

    static void TransformToAscii(void *targetBuffer, const void *sourceBuffer, size_t numElements);

  private:

    EntityDataPtr m_data;
    agx::Index m_index;
  };


  AGXCORE_EXPORT std::ostream& operator << (std::ostream& output, const EntityInstance& entity);

  /* Implementation */
  AGX_FORCE_INLINE EntityInstance::EntityInstance() : m_data(nullptr), m_index(agx::InvalidIndex)
  {}

  AGX_FORCE_INLINE EntityInstance::EntityInstance(agxData::EntityData* data, agx::Index index) : m_data(data), m_index(index)
  {}


  AGX_FORCE_INLINE EntityInstance::EntityInstance(const agxData::EntityInstance& other) : m_data(other.m_data), m_index(other.m_index)
  {}


  AGX_FORCE_INLINE void EntityInstance::init(agxData::EntityData* data, agx::Index index) { m_data = data; m_index = index; }

  AGX_FORCE_INLINE agx::Index EntityInstance::getIndex() const { return m_index; }

  AGX_FORCE_INLINE agxData::EntityData* EntityInstance::getData() { return m_data; }
  AGX_FORCE_INLINE const agxData::EntityData* EntityInstance::getData() const { return m_data; }

  AGX_FORCE_INLINE agx::UInt32 EntityInstance::hash() const
  {
    return agx::hash( agx::hash( m_data ), agx::hash( m_index ) );
  }

  AGX_FORCE_INLINE bool EntityInstance::isValid() const { return m_data != nullptr; }
  AGX_FORCE_INLINE bool EntityInstance::operator!() const { return !this->isValid(); }
  AGX_FORCE_INLINE EntityInstance::operator bool() const { return this->isValid(); }

  AGX_FORCE_INLINE bool EntityInstance::operator<( const EntityInstance& rhs ) const
  {
    return this->getIndex() < rhs.getIndex();
  }

  AGX_FORCE_INLINE bool EntityInstance::operator>( const EntityInstance& rhs ) const
  {
    return this->getIndex() > rhs.getIndex();
  }

  AGX_FORCE_INLINE bool EntityInstance::operator==( const EntityInstance& rhs ) const
  {
    return (this->getStorage() == rhs.getStorage()) && (this->getIndex() == rhs.getIndex());
  }

  AGX_FORCE_INLINE bool EntityInstance::operator!=( const EntityInstance& rhs ) const
  {
    return (this->getStorage() != rhs.getStorage()) || (this->getIndex() != rhs.getIndex());
  }


}


#endif /* AGXDATA_ENTITYINSTANCE_H */
