/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

//////////////////////////////////////////////////
// AUTOMATICALLY GENERATED ENTITY, DO NOT EDIT! //
//////////////////////////////////////////////////

#ifndef GENERATED_AGX_PHYSICS_COLLISIONGROUPSET_H_PLUGIN
#define GENERATED_AGX_PHYSICS_COLLISIONGROUPSET_H_PLUGIN

#define AGX_ENTITY_WRAPPER 1


#ifdef _MSC_VER
# pragma warning(push)
// warning C4505: 'agxData::VectorAttributeT<T>::print' : unreferenced local function has been removed
# pragma warning( disable : 4505 )
//  warning C4251:  'X' : class 'Y' needs to have dll-interface to be used by clients of class 'Z'
# pragma warning( disable : 4251 )
//  warning C4355: 'this' : used in base member initializer list
# pragma warning( disable : 4355 )
//  marked as __forceinline not inlined
# pragma warning( disable: 4714 )
#endif

#include <agxData/EntityModel.h>
#include <agxData/EntityStorage.h>
#include <agxData/EntityRef.h>
#include <agxData/EntityPtr.h>
#include <agxData/EntityInstance.h>
#include <agx/Integer.h>
#include <agx/Real.h>
#include <agx/macros.h>
#include <agx/Physics/CollisionGroupEntity.h>
#include <agx/Integer.h>
namespace agxCollide { class CollisionGroupManager; }

namespace agx { namespace Physics { class CollisionGroupPtr; }}

namespace agx
{
  namespace Physics
  {

    class CollisionGroupSetModel;
    class CollisionGroupSetData;
    class CollisionGroupSetPtr;
    class CollisionGroupSetInstance;
    class CollisionGroupSetSemantics;


    AGX_DECLARE_POINTER_TYPES(CollisionGroupSetModel);

    /** 
    Abstract description of the data attributes for the Physics.CollisionGroupSet entity.
    */ 
    class AGXPHYSICS_EXPORT CollisionGroupSetModel : public agxData::EntityModel
    {
    public:
      typedef CollisionGroupSetPtr PtrT;

      CollisionGroupSetModel(const agx::String& name = "CollisionGroupSet");

      /// \return The entity model singleton.
      static CollisionGroupSetModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static CollisionGroupSetPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Vector< agx::Physics::CollisionGroupPtr > >* groupsAttribute;
      static agxData::ScalarAttributeT< agx::HashSet< agx::UInt32 > >* groupIdHashAttribute;
      static agxData::ScalarAttributeT< agx::UInt >* refCountAttribute;
      static agxData::PointerAttributeT< agxCollide::CollisionGroupManager*>* managerAttribute;

    protected:
      virtual ~CollisionGroupSetModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::CollisionGroupSetPtr collisionGroupSet);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_COLLISIONGROUPSET_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_COLLISIONGROUPSET_DATA_SET
    class AGXPHYSICS_EXPORT CollisionGroupSetData : public agxData::EntityData
    {
    public:
      CollisionGroupSetInstance operator[] (size_t index);

    public:
      agxData::Array< CollisionGroupSetPtr >& instance;
      agxData::Array< agx::Vector< agx::Physics::CollisionGroupPtr > > groups;
      agxData::Array< agx::HashSet< agx::UInt32 > > groupIdHash;
      agxData::Array< agx::UInt > refCount;
      agxData::Array< agxCollide::CollisionGroupManager* > manager;

    public:
      typedef agx::Vector< agx::Physics::CollisionGroupPtr > groupsType;
      typedef agx::HashSet< agx::UInt32 > groupIdHashType;
      typedef agx::UInt refCountType;
      typedef agxCollide::CollisionGroupManager* managerType;

    public:
      CollisionGroupSetData(agxData::EntityStorage* storage);
      CollisionGroupSetData();

    protected:
      virtual ~CollisionGroupSetData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      CollisionGroupSetData& operator= (const CollisionGroupSetData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT CollisionGroupSetSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::Vector< agx::Physics::CollisionGroupPtr > const& getGroups() const;
      agx::HashSet< agx::UInt32 > const& getGroupIdHash() const;
      agx::UInt const& getRefCount() const;
      agxCollide::CollisionGroupManager* const& getManager() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setGroups(agx::Vector< agx::Physics::CollisionGroupPtr > const& value);
      void setGroupIdHash(agx::HashSet< agx::UInt32 > const& value);
      void setRefCount(agx::UInt const& value);
      void setManager(agxCollide::CollisionGroupManager* const& value);


    protected:
      friend class CollisionGroupSetPtr;
      friend class CollisionGroupSetInstance;
      CollisionGroupSetSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.CollisionGroupSet
    */
    class CALLABLE CollisionGroupSetPtr : public agxData::EntityPtr
    {
    public:
      typedef CollisionGroupSetModel ModelType;
      typedef CollisionGroupSetData DataType;
      typedef CollisionGroupSetInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT CollisionGroupSetPtr();
      AGXPHYSICS_EXPORT CollisionGroupSetPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT CollisionGroupSetPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT CollisionGroupSetPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT CollisionGroupSetPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT CollisionGroupSetPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT CollisionGroupSetInstance instance();
      AGXPHYSICS_EXPORT const CollisionGroupSetInstance instance() const;

      AGXPHYSICS_EXPORT CollisionGroupSetSemantics* operator->();
      AGXPHYSICS_EXPORT const CollisionGroupSetSemantics* operator->() const;

      CollisionGroupSetData* getData();
      const CollisionGroupSetData* getData() const;


      /// \return reference to the groups attribute
      AGXPHYSICS_EXPORT agx::Vector< agx::Physics::CollisionGroupPtr >& groups();
      /// \return const reference to the groups attribute
      AGXPHYSICS_EXPORT agx::Vector< agx::Physics::CollisionGroupPtr > const& groups() const;

      /// \return reference to the groupIdHash attribute
      AGXPHYSICS_EXPORT agx::HashSet< agx::UInt32 >& groupIdHash();
      /// \return const reference to the groupIdHash attribute
      AGXPHYSICS_EXPORT agx::HashSet< agx::UInt32 > const& groupIdHash() const;

      /// \return reference to the refCount attribute
      AGXPHYSICS_EXPORT agx::UInt& refCount();
      /// \return const reference to the refCount attribute
      AGXPHYSICS_EXPORT agx::UInt const& refCount() const;

      /// \return reference to the manager attribute
      AGXPHYSICS_EXPORT agxCollide::CollisionGroupManager*& manager();
      /// \return const reference to the manager attribute
      AGXPHYSICS_EXPORT agxCollide::CollisionGroupManager* const& manager() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Vector< agx::Physics::CollisionGroupPtr > const& getGroups() const { return (*this)->getGroups(); }
      inline agx::HashSet< agx::UInt32 > const& getGroupIdHash() const { return (*this)->getGroupIdHash(); }
      inline agx::UInt const& getRefCount() const { return (*this)->getRefCount(); }
      inline agxCollide::CollisionGroupManager* const& getManager() const { return (*this)->getManager(); }


      inline void setGroups(agx::Vector< agx::Physics::CollisionGroupPtr > const& value) { (*this)->setGroups(value); }
      inline void setGroupIdHash(agx::HashSet< agx::UInt32 > const& value) { (*this)->setGroupIdHash(value); }
      inline void setRefCount(agx::UInt const& value) { (*this)->setRefCount(value); }
      inline void setManager(agxCollide::CollisionGroupManager* const& value) { (*this)->setManager(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT CollisionGroupSetInstance : public agxData::EntityInstance
    {
    public:
      CollisionGroupSetInstance();
      CollisionGroupSetInstance(CollisionGroupSetData* data, agx::Index index);
      CollisionGroupSetInstance(agxData::EntityStorage *storage, agx::Index index);
      CollisionGroupSetInstance(const agxData::EntityInstance& other);
      CollisionGroupSetInstance(const agxData::EntityPtr& ptr);

      CollisionGroupSetData* getData();
      const CollisionGroupSetData* getData() const;

    public:
      /// \return reference to the groups attribute
      agx::Vector< agx::Physics::CollisionGroupPtr >& groups();
      /// \return const reference to the groups attribute
      agx::Vector< agx::Physics::CollisionGroupPtr > const& groups() const;

      /// \return reference to the groupIdHash attribute
      agx::HashSet< agx::UInt32 >& groupIdHash();
      /// \return const reference to the groupIdHash attribute
      agx::HashSet< agx::UInt32 > const& groupIdHash() const;

      /// \return reference to the refCount attribute
      agx::UInt& refCount();
      /// \return const reference to the refCount attribute
      agx::UInt const& refCount() const;

      /// \return reference to the manager attribute
      agxCollide::CollisionGroupManager*& manager();
      /// \return const reference to the manager attribute
      agxCollide::CollisionGroupManager* const& manager() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<CollisionGroupSetPtr> CollisionGroupSetPtrVector;
    typedef agxData::Array<CollisionGroupSetPtr> CollisionGroupSetPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline CollisionGroupSetInstance agx::Physics::CollisionGroupSetData::operator[] (size_t index) { return CollisionGroupSetInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE CollisionGroupSetPtr::CollisionGroupSetPtr() {}
    AGX_FORCE_INLINE CollisionGroupSetPtr::CollisionGroupSetPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE CollisionGroupSetPtr::CollisionGroupSetPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(CollisionGroupSetModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), CollisionGroupSetModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE CollisionGroupSetPtr::CollisionGroupSetPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(CollisionGroupSetModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), CollisionGroupSetModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE CollisionGroupSetPtr& CollisionGroupSetPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(CollisionGroupSetModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), CollisionGroupSetModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE CollisionGroupSetPtr& CollisionGroupSetPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(CollisionGroupSetModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), CollisionGroupSetModel::instance()->fullPath().c_str());
      return *this;
    }

    inline CollisionGroupSetInstance CollisionGroupSetPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const CollisionGroupSetInstance CollisionGroupSetPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE CollisionGroupSetSemantics* CollisionGroupSetPtr::operator->() { return (CollisionGroupSetSemantics* )this; }
    AGX_FORCE_INLINE const CollisionGroupSetSemantics* CollisionGroupSetPtr::operator->() const { return (const CollisionGroupSetSemantics* )this; }
    AGX_FORCE_INLINE CollisionGroupSetData* CollisionGroupSetPtr::getData() { return static_cast<CollisionGroupSetData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const CollisionGroupSetData* CollisionGroupSetPtr::getData() const { return static_cast<const CollisionGroupSetData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Vector< agx::Physics::CollisionGroupPtr >& CollisionGroupSetPtr::groups() { verifyIndex(); return getData()->groups[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vector< agx::Physics::CollisionGroupPtr > const& CollisionGroupSetPtr::groups() const { verifyIndex(); return getData()->groups[calculateIndex()]; }

    AGX_FORCE_INLINE agx::HashSet< agx::UInt32 >& CollisionGroupSetPtr::groupIdHash() { verifyIndex(); return getData()->groupIdHash[calculateIndex()]; }
    AGX_FORCE_INLINE agx::HashSet< agx::UInt32 > const& CollisionGroupSetPtr::groupIdHash() const { verifyIndex(); return getData()->groupIdHash[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt& CollisionGroupSetPtr::refCount() { verifyIndex(); return getData()->refCount[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& CollisionGroupSetPtr::refCount() const { verifyIndex(); return getData()->refCount[calculateIndex()]; }

    AGX_FORCE_INLINE agxCollide::CollisionGroupManager*& CollisionGroupSetPtr::manager() { verifyIndex(); return getData()->manager[calculateIndex()]; }
    AGX_FORCE_INLINE agxCollide::CollisionGroupManager* const& CollisionGroupSetPtr::manager() const { verifyIndex(); return getData()->manager[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE CollisionGroupSetInstance::CollisionGroupSetInstance() {}
    AGX_FORCE_INLINE CollisionGroupSetInstance::CollisionGroupSetInstance(CollisionGroupSetData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE CollisionGroupSetInstance::CollisionGroupSetInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE CollisionGroupSetInstance::CollisionGroupSetInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(CollisionGroupSetModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), CollisionGroupSetModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE CollisionGroupSetInstance::CollisionGroupSetInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(CollisionGroupSetModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), CollisionGroupSetModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE CollisionGroupSetData* CollisionGroupSetInstance::getData() { return static_cast<CollisionGroupSetData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const CollisionGroupSetData* CollisionGroupSetInstance::getData() const { return static_cast<const CollisionGroupSetData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Vector< agx::Physics::CollisionGroupPtr >& CollisionGroupSetInstance::groups() { verifyIndex(); return getData()->groups[getIndex()]; }
    AGX_FORCE_INLINE agx::Vector< agx::Physics::CollisionGroupPtr > const& CollisionGroupSetInstance::groups() const { verifyIndex(); return getData()->groups[getIndex()]; }

    AGX_FORCE_INLINE agx::HashSet< agx::UInt32 >& CollisionGroupSetInstance::groupIdHash() { verifyIndex(); return getData()->groupIdHash[getIndex()]; }
    AGX_FORCE_INLINE agx::HashSet< agx::UInt32 > const& CollisionGroupSetInstance::groupIdHash() const { verifyIndex(); return getData()->groupIdHash[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt& CollisionGroupSetInstance::refCount() { verifyIndex(); return getData()->refCount[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& CollisionGroupSetInstance::refCount() const { verifyIndex(); return getData()->refCount[getIndex()]; }

    AGX_FORCE_INLINE agxCollide::CollisionGroupManager*& CollisionGroupSetInstance::manager() { verifyIndex(); return getData()->manager[getIndex()]; }
    AGX_FORCE_INLINE agxCollide::CollisionGroupManager* const& CollisionGroupSetInstance::manager() const { verifyIndex(); return getData()->manager[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE CollisionGroupSetSemantics::CollisionGroupSetSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::CollisionGroupSetPtr, "Physics.CollisionGroupSetPtr")
AGX_TYPE_BINDING(agx::Physics::CollisionGroupSetInstance, "Physics.CollisionGroupSetInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

