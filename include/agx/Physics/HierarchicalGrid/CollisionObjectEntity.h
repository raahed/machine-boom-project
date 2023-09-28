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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_COLLISIONOBJECT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_COLLISIONOBJECT_H_PLUGIN

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
#include <agx/Real.h>
#include <agx/Vec3.h>


namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class CollisionObjectModel;
      class CollisionObjectData;
      class CollisionObjectPtr;
      class CollisionObjectInstance;
      class CollisionObjectSemantics;


      AGX_DECLARE_POINTER_TYPES(CollisionObjectModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.CollisionObject entity.
      */ 
      class AGXPHYSICS_EXPORT CollisionObjectModel : public agxData::EntityModel
      {
      public:
        typedef CollisionObjectPtr PtrT;

        CollisionObjectModel(const agx::String& name = "CollisionObject");

        /// \return The entity model singleton.
        static CollisionObjectModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static CollisionObjectPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::UInt32 >* sourceIndexAttribute;
        static agxData::ScalarAttributeT< agx::UInt16 >* subsystemAttribute;
        static agxData::ScalarAttributeT< agx::Real >* radiusAttribute;
        static agxData::ScalarAttributeT< agx::Vec3 >* positionAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* obbIndexAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* collisionGroupSetAttribute;

      protected:
        virtual ~CollisionObjectModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::CollisionObjectPtr collisionObject);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_COLLISIONOBJECT_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_COLLISIONOBJECT_DATA_SET
      class AGXPHYSICS_EXPORT CollisionObjectData : public agxData::EntityData
      {
      public:
        CollisionObjectInstance operator[] (size_t index);

      public:
        agxData::Array< CollisionObjectPtr >& instance;
        agxData::Array< agx::UInt32 > sourceIndex;
        agxData::Array< agx::UInt16 > subsystem;
        agxData::Array< agx::Real > radius;
        agxData::Array< agx::Vec3 > position;
        agxData::Array< agx::UInt32 > obbIndex;
        agxData::Array< agx::UInt32 > collisionGroupSet;

      public:
        typedef agx::UInt32 sourceIndexType;
        typedef agx::UInt16 subsystemType;
        typedef agx::Real radiusType;
        typedef agx::Vec3 positionType;
        typedef agx::UInt32 obbIndexType;
        typedef agx::UInt32 collisionGroupSetType;

      public:
        CollisionObjectData(agxData::EntityStorage* storage);
        CollisionObjectData();

      protected:
        virtual ~CollisionObjectData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        CollisionObjectData& operator= (const CollisionObjectData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CollisionObjectSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::UInt32 const& getSourceIndex() const;
        agx::UInt16 const& getSubsystem() const;
        agx::Real const& getRadius() const;
        agx::Vec3 const& getPosition() const;
        agx::UInt32 const& getObbIndex() const;
        agx::UInt32 const& getCollisionGroupSet() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setSourceIndex(agx::UInt32 const& value);
        void setSubsystem(agx::UInt16 const& value);
        void setRadius(agx::Real const& value);
        void setPosition(agx::Vec3 const& value);
        void setObbIndex(agx::UInt32 const& value);
        void setCollisionGroupSet(agx::UInt32 const& value);


      protected:
        friend class CollisionObjectPtr;
        friend class CollisionObjectInstance;
        CollisionObjectSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.CollisionObject
      */
      class CALLABLE CollisionObjectPtr : public agxData::EntityPtr
      {
      public:
        typedef CollisionObjectModel ModelType;
        typedef CollisionObjectData DataType;
        typedef CollisionObjectInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT CollisionObjectPtr();
        AGXPHYSICS_EXPORT CollisionObjectPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT CollisionObjectPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CollisionObjectPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CollisionObjectPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CollisionObjectPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CollisionObjectInstance instance();
        AGXPHYSICS_EXPORT const CollisionObjectInstance instance() const;

        AGXPHYSICS_EXPORT CollisionObjectSemantics* operator->();
        AGXPHYSICS_EXPORT const CollisionObjectSemantics* operator->() const;

        CollisionObjectData* getData();
        const CollisionObjectData* getData() const;


        /// \return reference to the sourceIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32& sourceIndex();
        /// \return const reference to the sourceIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& sourceIndex() const;

        /// \return reference to the subsystem attribute
        AGXPHYSICS_EXPORT agx::UInt16& subsystem();
        /// \return const reference to the subsystem attribute
        AGXPHYSICS_EXPORT agx::UInt16 const& subsystem() const;

        /// \return reference to the radius attribute
        AGXPHYSICS_EXPORT agx::Real& radius();
        /// \return const reference to the radius attribute
        AGXPHYSICS_EXPORT agx::Real const& radius() const;

        /// \return reference to the position attribute
        AGXPHYSICS_EXPORT agx::Vec3& position();
        /// \return const reference to the position attribute
        AGXPHYSICS_EXPORT agx::Vec3 const& position() const;

        /// \return reference to the obbIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32& obbIndex();
        /// \return const reference to the obbIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& obbIndex() const;

        /// \return reference to the collisionGroupSet attribute
        AGXPHYSICS_EXPORT agx::UInt32& collisionGroupSet();
        /// \return const reference to the collisionGroupSet attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& collisionGroupSet() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::UInt32 const& getSourceIndex() const { return (*this)->getSourceIndex(); }
        inline agx::UInt16 const& getSubsystem() const { return (*this)->getSubsystem(); }
        inline agx::Real const& getRadius() const { return (*this)->getRadius(); }
        inline agx::Vec3 const& getPosition() const { return (*this)->getPosition(); }
        inline agx::UInt32 const& getObbIndex() const { return (*this)->getObbIndex(); }
        inline agx::UInt32 const& getCollisionGroupSet() const { return (*this)->getCollisionGroupSet(); }


        inline void setSourceIndex(agx::UInt32 const& value) { (*this)->setSourceIndex(value); }
        inline void setSubsystem(agx::UInt16 const& value) { (*this)->setSubsystem(value); }
        inline void setRadius(agx::Real const& value) { (*this)->setRadius(value); }
        inline void setPosition(agx::Vec3 const& value) { (*this)->setPosition(value); }
        inline void setObbIndex(agx::UInt32 const& value) { (*this)->setObbIndex(value); }
        inline void setCollisionGroupSet(agx::UInt32 const& value) { (*this)->setCollisionGroupSet(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CollisionObjectInstance : public agxData::EntityInstance
      {
      public:
        CollisionObjectInstance();
        CollisionObjectInstance(CollisionObjectData* data, agx::Index index);
        CollisionObjectInstance(agxData::EntityStorage *storage, agx::Index index);
        CollisionObjectInstance(const agxData::EntityInstance& other);
        CollisionObjectInstance(const agxData::EntityPtr& ptr);

        CollisionObjectData* getData();
        const CollisionObjectData* getData() const;

      public:
        /// \return reference to the sourceIndex attribute
        agx::UInt32& sourceIndex();
        /// \return const reference to the sourceIndex attribute
        agx::UInt32 const& sourceIndex() const;

        /// \return reference to the subsystem attribute
        agx::UInt16& subsystem();
        /// \return const reference to the subsystem attribute
        agx::UInt16 const& subsystem() const;

        /// \return reference to the radius attribute
        agx::Real& radius();
        /// \return const reference to the radius attribute
        agx::Real const& radius() const;

        /// \return reference to the position attribute
        agx::Vec3& position();
        /// \return const reference to the position attribute
        agx::Vec3 const& position() const;

        /// \return reference to the obbIndex attribute
        agx::UInt32& obbIndex();
        /// \return const reference to the obbIndex attribute
        agx::UInt32 const& obbIndex() const;

        /// \return reference to the collisionGroupSet attribute
        agx::UInt32& collisionGroupSet();
        /// \return const reference to the collisionGroupSet attribute
        agx::UInt32 const& collisionGroupSet() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<CollisionObjectPtr> CollisionObjectPtrVector;
      typedef agxData::Array<CollisionObjectPtr> CollisionObjectPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline CollisionObjectInstance agx::Physics::HierarchicalGrid::CollisionObjectData::operator[] (size_t index) { return CollisionObjectInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CollisionObjectPtr::CollisionObjectPtr() {}
      AGX_FORCE_INLINE CollisionObjectPtr::CollisionObjectPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE CollisionObjectPtr::CollisionObjectPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CollisionObjectModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CollisionObjectModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CollisionObjectPtr::CollisionObjectPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(CollisionObjectModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CollisionObjectModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CollisionObjectPtr& CollisionObjectPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(CollisionObjectModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CollisionObjectModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE CollisionObjectPtr& CollisionObjectPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(CollisionObjectModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CollisionObjectModel::instance()->fullPath().c_str());
        return *this;
      }

      inline CollisionObjectInstance CollisionObjectPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const CollisionObjectInstance CollisionObjectPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE CollisionObjectSemantics* CollisionObjectPtr::operator->() { return (CollisionObjectSemantics* )this; }
      AGX_FORCE_INLINE const CollisionObjectSemantics* CollisionObjectPtr::operator->() const { return (const CollisionObjectSemantics* )this; }
      AGX_FORCE_INLINE CollisionObjectData* CollisionObjectPtr::getData() { return static_cast<CollisionObjectData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const CollisionObjectData* CollisionObjectPtr::getData() const { return static_cast<const CollisionObjectData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::UInt32& CollisionObjectPtr::sourceIndex() { verifyIndex(); return getData()->sourceIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObjectPtr::sourceIndex() const { verifyIndex(); return getData()->sourceIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt16& CollisionObjectPtr::subsystem() { verifyIndex(); return getData()->subsystem[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt16 const& CollisionObjectPtr::subsystem() const { verifyIndex(); return getData()->subsystem[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& CollisionObjectPtr::radius() { verifyIndex(); return getData()->radius[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& CollisionObjectPtr::radius() const { verifyIndex(); return getData()->radius[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vec3& CollisionObjectPtr::position() { verifyIndex(); return getData()->position[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& CollisionObjectPtr::position() const { verifyIndex(); return getData()->position[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CollisionObjectPtr::obbIndex() { verifyIndex(); return getData()->obbIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObjectPtr::obbIndex() const { verifyIndex(); return getData()->obbIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CollisionObjectPtr::collisionGroupSet() { verifyIndex(); return getData()->collisionGroupSet[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObjectPtr::collisionGroupSet() const { verifyIndex(); return getData()->collisionGroupSet[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CollisionObjectInstance::CollisionObjectInstance() {}
      AGX_FORCE_INLINE CollisionObjectInstance::CollisionObjectInstance(CollisionObjectData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE CollisionObjectInstance::CollisionObjectInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE CollisionObjectInstance::CollisionObjectInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(CollisionObjectModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CollisionObjectModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CollisionObjectInstance::CollisionObjectInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CollisionObjectModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CollisionObjectModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE CollisionObjectData* CollisionObjectInstance::getData() { return static_cast<CollisionObjectData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const CollisionObjectData* CollisionObjectInstance::getData() const { return static_cast<const CollisionObjectData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::UInt32& CollisionObjectInstance::sourceIndex() { verifyIndex(); return getData()->sourceIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObjectInstance::sourceIndex() const { verifyIndex(); return getData()->sourceIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt16& CollisionObjectInstance::subsystem() { verifyIndex(); return getData()->subsystem[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt16 const& CollisionObjectInstance::subsystem() const { verifyIndex(); return getData()->subsystem[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& CollisionObjectInstance::radius() { verifyIndex(); return getData()->radius[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& CollisionObjectInstance::radius() const { verifyIndex(); return getData()->radius[getIndex()]; }

      AGX_FORCE_INLINE agx::Vec3& CollisionObjectInstance::position() { verifyIndex(); return getData()->position[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& CollisionObjectInstance::position() const { verifyIndex(); return getData()->position[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CollisionObjectInstance::obbIndex() { verifyIndex(); return getData()->obbIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObjectInstance::obbIndex() const { verifyIndex(); return getData()->obbIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CollisionObjectInstance::collisionGroupSet() { verifyIndex(); return getData()->collisionGroupSet[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObjectInstance::collisionGroupSet() const { verifyIndex(); return getData()->collisionGroupSet[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CollisionObjectSemantics::CollisionObjectSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::CollisionObjectPtr, "Physics.HierarchicalGrid.CollisionObjectPtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::CollisionObjectInstance, "Physics.HierarchicalGrid.CollisionObjectInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

