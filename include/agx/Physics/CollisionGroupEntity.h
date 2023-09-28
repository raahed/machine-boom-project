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

#ifndef GENERATED_AGX_PHYSICS_COLLISIONGROUP_H_PLUGIN
#define GENERATED_AGX_PHYSICS_COLLISIONGROUP_H_PLUGIN

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
#include <agx/Integer.h>
#include <agx/Name.h>
#include <agx/Physics/CollisionGroupSetEntity.h>
namespace agxCollide { class CollisionGroupManager; }

namespace agx { namespace Physics { class CollisionGroupSetPtr; }}
namespace agx { namespace Physics { class CollisionGroupPtr; }}

namespace agx
{
  namespace Physics
  {

    class CollisionGroupModel;
    class CollisionGroupData;
    class CollisionGroupPtr;
    class CollisionGroupInstance;
    class CollisionGroupSemantics;


    AGX_DECLARE_POINTER_TYPES(CollisionGroupModel);

    /** 
    Abstract description of the data attributes for the Physics.CollisionGroup entity.
    */ 
    class AGXPHYSICS_EXPORT CollisionGroupModel : public agxData::EntityModel
    {
    public:
      typedef CollisionGroupPtr PtrT;

      CollisionGroupModel(const agx::String& name = "CollisionGroup");

      /// \return The entity model singleton.
      static CollisionGroupModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static CollisionGroupPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::UInt32 >* idAttribute;
      static agxData::ScalarAttributeT< agx::Name >* nameAttribute;
      static agxData::ScalarAttributeT< agx::HashSet< agx::Physics::CollisionGroupSetPtr > >* setsAttribute;
      static agxData::ScalarAttributeT< agx::HashSet< agx::Physics::CollisionGroupPtr > >* disabledGroupsAttribute;
      static agxData::PointerAttributeT< agxCollide::CollisionGroupManager*>* managerAttribute;

    protected:
      virtual ~CollisionGroupModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::CollisionGroupPtr collisionGroup);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_COLLISIONGROUP_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_COLLISIONGROUP_DATA_SET
    class AGXPHYSICS_EXPORT CollisionGroupData : public agxData::EntityData
    {
    public:
      CollisionGroupInstance operator[] (size_t index);

    public:
      agxData::Array< CollisionGroupPtr >& instance;
      agxData::Array< agx::UInt32 > id;
      agxData::Array< agx::Name > name;
      agxData::Array< agx::HashSet< agx::Physics::CollisionGroupSetPtr > > sets;
      agxData::Array< agx::HashSet< agx::Physics::CollisionGroupPtr > > disabledGroups;
      agxData::Array< agxCollide::CollisionGroupManager* > manager;

    public:
      typedef agx::UInt32 idType;
      typedef agx::Name nameType;
      typedef agx::HashSet< agx::Physics::CollisionGroupSetPtr > setsType;
      typedef agx::HashSet< agx::Physics::CollisionGroupPtr > disabledGroupsType;
      typedef agxCollide::CollisionGroupManager* managerType;

    public:
      CollisionGroupData(agxData::EntityStorage* storage);
      CollisionGroupData();

    protected:
      virtual ~CollisionGroupData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      CollisionGroupData& operator= (const CollisionGroupData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT CollisionGroupSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::UInt32 const& getId() const;
      agx::Name const& getName() const;
      agx::HashSet< agx::Physics::CollisionGroupSetPtr > const& getSets() const;
      agx::HashSet< agx::Physics::CollisionGroupPtr > const& getDisabledGroups() const;
      agxCollide::CollisionGroupManager* const& getManager() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setId(agx::UInt32 const& value);
      void setName(agx::Name const& value);
      void setSets(agx::HashSet< agx::Physics::CollisionGroupSetPtr > const& value);
      void setDisabledGroups(agx::HashSet< agx::Physics::CollisionGroupPtr > const& value);
      void setManager(agxCollide::CollisionGroupManager* const& value);


    protected:
      friend class CollisionGroupPtr;
      friend class CollisionGroupInstance;
      CollisionGroupSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.CollisionGroup
    */
    class CALLABLE CollisionGroupPtr : public agxData::EntityPtr
    {
    public:
      typedef CollisionGroupModel ModelType;
      typedef CollisionGroupData DataType;
      typedef CollisionGroupInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT CollisionGroupPtr();
      AGXPHYSICS_EXPORT CollisionGroupPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT CollisionGroupPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT CollisionGroupPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT CollisionGroupPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT CollisionGroupPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT CollisionGroupInstance instance();
      AGXPHYSICS_EXPORT const CollisionGroupInstance instance() const;

      AGXPHYSICS_EXPORT CollisionGroupSemantics* operator->();
      AGXPHYSICS_EXPORT const CollisionGroupSemantics* operator->() const;

      CollisionGroupData* getData();
      const CollisionGroupData* getData() const;


      /// \return reference to the id attribute
      AGXPHYSICS_EXPORT agx::UInt32& id();
      /// \return const reference to the id attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& id() const;

      /// \return reference to the name attribute
      AGXPHYSICS_EXPORT agx::Name& name();
      /// \return const reference to the name attribute
      AGXPHYSICS_EXPORT agx::Name const& name() const;

      /// \return reference to the sets attribute
      AGXPHYSICS_EXPORT agx::HashSet< agx::Physics::CollisionGroupSetPtr >& sets();
      /// \return const reference to the sets attribute
      AGXPHYSICS_EXPORT agx::HashSet< agx::Physics::CollisionGroupSetPtr > const& sets() const;

      /// \return reference to the disabledGroups attribute
      AGXPHYSICS_EXPORT agx::HashSet< agx::Physics::CollisionGroupPtr >& disabledGroups();
      /// \return const reference to the disabledGroups attribute
      AGXPHYSICS_EXPORT agx::HashSet< agx::Physics::CollisionGroupPtr > const& disabledGroups() const;

      /// \return reference to the manager attribute
      AGXPHYSICS_EXPORT agxCollide::CollisionGroupManager*& manager();
      /// \return const reference to the manager attribute
      AGXPHYSICS_EXPORT agxCollide::CollisionGroupManager* const& manager() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::UInt32 const& getId() const { return (*this)->getId(); }
      inline agx::Name const& getName() const { return (*this)->getName(); }
      inline agx::HashSet< agx::Physics::CollisionGroupSetPtr > const& getSets() const { return (*this)->getSets(); }
      inline agx::HashSet< agx::Physics::CollisionGroupPtr > const& getDisabledGroups() const { return (*this)->getDisabledGroups(); }
      inline agxCollide::CollisionGroupManager* const& getManager() const { return (*this)->getManager(); }


      inline void setId(agx::UInt32 const& value) { (*this)->setId(value); }
      inline void setName(agx::Name const& value) { (*this)->setName(value); }
      inline void setSets(agx::HashSet< agx::Physics::CollisionGroupSetPtr > const& value) { (*this)->setSets(value); }
      inline void setDisabledGroups(agx::HashSet< agx::Physics::CollisionGroupPtr > const& value) { (*this)->setDisabledGroups(value); }
      inline void setManager(agxCollide::CollisionGroupManager* const& value) { (*this)->setManager(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT CollisionGroupInstance : public agxData::EntityInstance
    {
    public:
      CollisionGroupInstance();
      CollisionGroupInstance(CollisionGroupData* data, agx::Index index);
      CollisionGroupInstance(agxData::EntityStorage *storage, agx::Index index);
      CollisionGroupInstance(const agxData::EntityInstance& other);
      CollisionGroupInstance(const agxData::EntityPtr& ptr);

      CollisionGroupData* getData();
      const CollisionGroupData* getData() const;

    public:
      /// \return reference to the id attribute
      agx::UInt32& id();
      /// \return const reference to the id attribute
      agx::UInt32 const& id() const;

      /// \return reference to the name attribute
      agx::Name& name();
      /// \return const reference to the name attribute
      agx::Name const& name() const;

      /// \return reference to the sets attribute
      agx::HashSet< agx::Physics::CollisionGroupSetPtr >& sets();
      /// \return const reference to the sets attribute
      agx::HashSet< agx::Physics::CollisionGroupSetPtr > const& sets() const;

      /// \return reference to the disabledGroups attribute
      agx::HashSet< agx::Physics::CollisionGroupPtr >& disabledGroups();
      /// \return const reference to the disabledGroups attribute
      agx::HashSet< agx::Physics::CollisionGroupPtr > const& disabledGroups() const;

      /// \return reference to the manager attribute
      agxCollide::CollisionGroupManager*& manager();
      /// \return const reference to the manager attribute
      agxCollide::CollisionGroupManager* const& manager() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<CollisionGroupPtr> CollisionGroupPtrVector;
    typedef agxData::Array<CollisionGroupPtr> CollisionGroupPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline CollisionGroupInstance agx::Physics::CollisionGroupData::operator[] (size_t index) { return CollisionGroupInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE CollisionGroupPtr::CollisionGroupPtr() {}
    AGX_FORCE_INLINE CollisionGroupPtr::CollisionGroupPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE CollisionGroupPtr::CollisionGroupPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(CollisionGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), CollisionGroupModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE CollisionGroupPtr::CollisionGroupPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(CollisionGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), CollisionGroupModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE CollisionGroupPtr& CollisionGroupPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(CollisionGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), CollisionGroupModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE CollisionGroupPtr& CollisionGroupPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(CollisionGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), CollisionGroupModel::instance()->fullPath().c_str());
      return *this;
    }

    inline CollisionGroupInstance CollisionGroupPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const CollisionGroupInstance CollisionGroupPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE CollisionGroupSemantics* CollisionGroupPtr::operator->() { return (CollisionGroupSemantics* )this; }
    AGX_FORCE_INLINE const CollisionGroupSemantics* CollisionGroupPtr::operator->() const { return (const CollisionGroupSemantics* )this; }
    AGX_FORCE_INLINE CollisionGroupData* CollisionGroupPtr::getData() { return static_cast<CollisionGroupData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const CollisionGroupData* CollisionGroupPtr::getData() const { return static_cast<const CollisionGroupData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::UInt32& CollisionGroupPtr::id() { verifyIndex(); return getData()->id[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& CollisionGroupPtr::id() const { verifyIndex(); return getData()->id[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Name& CollisionGroupPtr::name() { verifyIndex(); return getData()->name[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Name const& CollisionGroupPtr::name() const { verifyIndex(); return getData()->name[calculateIndex()]; }

    AGX_FORCE_INLINE agx::HashSet< agx::Physics::CollisionGroupSetPtr >& CollisionGroupPtr::sets() { verifyIndex(); return getData()->sets[calculateIndex()]; }
    AGX_FORCE_INLINE agx::HashSet< agx::Physics::CollisionGroupSetPtr > const& CollisionGroupPtr::sets() const { verifyIndex(); return getData()->sets[calculateIndex()]; }

    AGX_FORCE_INLINE agx::HashSet< agx::Physics::CollisionGroupPtr >& CollisionGroupPtr::disabledGroups() { verifyIndex(); return getData()->disabledGroups[calculateIndex()]; }
    AGX_FORCE_INLINE agx::HashSet< agx::Physics::CollisionGroupPtr > const& CollisionGroupPtr::disabledGroups() const { verifyIndex(); return getData()->disabledGroups[calculateIndex()]; }

    AGX_FORCE_INLINE agxCollide::CollisionGroupManager*& CollisionGroupPtr::manager() { verifyIndex(); return getData()->manager[calculateIndex()]; }
    AGX_FORCE_INLINE agxCollide::CollisionGroupManager* const& CollisionGroupPtr::manager() const { verifyIndex(); return getData()->manager[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE CollisionGroupInstance::CollisionGroupInstance() {}
    AGX_FORCE_INLINE CollisionGroupInstance::CollisionGroupInstance(CollisionGroupData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE CollisionGroupInstance::CollisionGroupInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE CollisionGroupInstance::CollisionGroupInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(CollisionGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), CollisionGroupModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE CollisionGroupInstance::CollisionGroupInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(CollisionGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), CollisionGroupModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE CollisionGroupData* CollisionGroupInstance::getData() { return static_cast<CollisionGroupData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const CollisionGroupData* CollisionGroupInstance::getData() const { return static_cast<const CollisionGroupData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::UInt32& CollisionGroupInstance::id() { verifyIndex(); return getData()->id[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& CollisionGroupInstance::id() const { verifyIndex(); return getData()->id[getIndex()]; }

    AGX_FORCE_INLINE agx::Name& CollisionGroupInstance::name() { verifyIndex(); return getData()->name[getIndex()]; }
    AGX_FORCE_INLINE agx::Name const& CollisionGroupInstance::name() const { verifyIndex(); return getData()->name[getIndex()]; }

    AGX_FORCE_INLINE agx::HashSet< agx::Physics::CollisionGroupSetPtr >& CollisionGroupInstance::sets() { verifyIndex(); return getData()->sets[getIndex()]; }
    AGX_FORCE_INLINE agx::HashSet< agx::Physics::CollisionGroupSetPtr > const& CollisionGroupInstance::sets() const { verifyIndex(); return getData()->sets[getIndex()]; }

    AGX_FORCE_INLINE agx::HashSet< agx::Physics::CollisionGroupPtr >& CollisionGroupInstance::disabledGroups() { verifyIndex(); return getData()->disabledGroups[getIndex()]; }
    AGX_FORCE_INLINE agx::HashSet< agx::Physics::CollisionGroupPtr > const& CollisionGroupInstance::disabledGroups() const { verifyIndex(); return getData()->disabledGroups[getIndex()]; }

    AGX_FORCE_INLINE agxCollide::CollisionGroupManager*& CollisionGroupInstance::manager() { verifyIndex(); return getData()->manager[getIndex()]; }
    AGX_FORCE_INLINE agxCollide::CollisionGroupManager* const& CollisionGroupInstance::manager() const { verifyIndex(); return getData()->manager[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE CollisionGroupSemantics::CollisionGroupSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::CollisionGroupPtr, "Physics.CollisionGroupPtr")
AGX_TYPE_BINDING(agx::Physics::CollisionGroupInstance, "Physics.CollisionGroupInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

