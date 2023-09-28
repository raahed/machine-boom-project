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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_COLLISIONOBJECT2D_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_COLLISIONOBJECT2D_H_PLUGIN

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
#include <agx/Vec2.h>


namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class CollisionObject2DModel;
      class CollisionObject2DData;
      class CollisionObject2DPtr;
      class CollisionObject2DInstance;
      class CollisionObject2DSemantics;


      AGX_DECLARE_POINTER_TYPES(CollisionObject2DModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.CollisionObject2D entity.
      */ 
      class AGXPHYSICS_EXPORT CollisionObject2DModel : public agxData::EntityModel
      {
      public:
        typedef CollisionObject2DPtr PtrT;

        CollisionObject2DModel(const agx::String& name = "CollisionObject2D");

        /// \return The entity model singleton.
        static CollisionObject2DModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static CollisionObject2DPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::UInt32 >* sourceIndexAttribute;
        static agxData::ScalarAttributeT< agx::UInt16 >* subsystemAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* cellIndexAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* radiusAttribute;
        static agxData::ScalarAttributeT< agx::Vec2f >* positionAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* oobIndexAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* collisionGroupSetAttribute;

      protected:
        virtual ~CollisionObject2DModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::CollisionObject2DPtr collisionObject2D);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_COLLISIONOBJECT2D_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_COLLISIONOBJECT2D_DATA_SET
      class AGXPHYSICS_EXPORT CollisionObject2DData : public agxData::EntityData
      {
      public:
        CollisionObject2DInstance operator[] (size_t index);

      public:
        agxData::Array< CollisionObject2DPtr >& instance;
        agxData::Array< agx::UInt32 > sourceIndex;
        agxData::Array< agx::UInt16 > subsystem;
        agxData::Array< agx::UInt32 > cellIndex;
        agxData::Array< agx::Real32 > radius;
        agxData::Array< agx::Vec2f > position;
        agxData::Array< agx::UInt32 > oobIndex;
        agxData::Array< agx::UInt32 > collisionGroupSet;

      public:
        typedef agx::UInt32 sourceIndexType;
        typedef agx::UInt16 subsystemType;
        typedef agx::UInt32 cellIndexType;
        typedef agx::Real32 radiusType;
        typedef agx::Vec2f positionType;
        typedef agx::UInt32 oobIndexType;
        typedef agx::UInt32 collisionGroupSetType;

      public:
        CollisionObject2DData(agxData::EntityStorage* storage);
        CollisionObject2DData();

      protected:
        virtual ~CollisionObject2DData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        CollisionObject2DData& operator= (const CollisionObject2DData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CollisionObject2DSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::UInt32 const& getSourceIndex() const;
        agx::UInt16 const& getSubsystem() const;
        agx::UInt32 const& getCellIndex() const;
        agx::Real32 const& getRadius() const;
        agx::Vec2f const& getPosition() const;
        agx::UInt32 const& getOobIndex() const;
        agx::UInt32 const& getCollisionGroupSet() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setSourceIndex(agx::UInt32 const& value);
        void setSubsystem(agx::UInt16 const& value);
        void setCellIndex(agx::UInt32 const& value);
        void setRadius(agx::Real32 const& value);
        void setPosition(agx::Vec2f const& value);
        void setOobIndex(agx::UInt32 const& value);
        void setCollisionGroupSet(agx::UInt32 const& value);


      protected:
        friend class CollisionObject2DPtr;
        friend class CollisionObject2DInstance;
        CollisionObject2DSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.CollisionObject2D
      */
      class CALLABLE CollisionObject2DPtr : public agxData::EntityPtr
      {
      public:
        typedef CollisionObject2DModel ModelType;
        typedef CollisionObject2DData DataType;
        typedef CollisionObject2DInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT CollisionObject2DPtr();
        AGXPHYSICS_EXPORT CollisionObject2DPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT CollisionObject2DPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CollisionObject2DPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CollisionObject2DPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CollisionObject2DPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CollisionObject2DInstance instance();
        AGXPHYSICS_EXPORT const CollisionObject2DInstance instance() const;

        AGXPHYSICS_EXPORT CollisionObject2DSemantics* operator->();
        AGXPHYSICS_EXPORT const CollisionObject2DSemantics* operator->() const;

        CollisionObject2DData* getData();
        const CollisionObject2DData* getData() const;


        /// \return reference to the sourceIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32& sourceIndex();
        /// \return const reference to the sourceIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& sourceIndex() const;

        /// \return reference to the subsystem attribute
        AGXPHYSICS_EXPORT agx::UInt16& subsystem();
        /// \return const reference to the subsystem attribute
        AGXPHYSICS_EXPORT agx::UInt16 const& subsystem() const;

        /// \return reference to the cellIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32& cellIndex();
        /// \return const reference to the cellIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& cellIndex() const;

        /// \return reference to the radius attribute
        AGXPHYSICS_EXPORT agx::Real32& radius();
        /// \return const reference to the radius attribute
        AGXPHYSICS_EXPORT agx::Real32 const& radius() const;

        /// \return reference to the position attribute
        AGXPHYSICS_EXPORT agx::Vec2f& position();
        /// \return const reference to the position attribute
        AGXPHYSICS_EXPORT agx::Vec2f const& position() const;

        /// \return reference to the oobIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32& oobIndex();
        /// \return const reference to the oobIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& oobIndex() const;

        /// \return reference to the collisionGroupSet attribute
        AGXPHYSICS_EXPORT agx::UInt32& collisionGroupSet();
        /// \return const reference to the collisionGroupSet attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& collisionGroupSet() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::UInt32 const& getSourceIndex() const { return (*this)->getSourceIndex(); }
        inline agx::UInt16 const& getSubsystem() const { return (*this)->getSubsystem(); }
        inline agx::UInt32 const& getCellIndex() const { return (*this)->getCellIndex(); }
        inline agx::Real32 const& getRadius() const { return (*this)->getRadius(); }
        inline agx::Vec2f const& getPosition() const { return (*this)->getPosition(); }
        inline agx::UInt32 const& getOobIndex() const { return (*this)->getOobIndex(); }
        inline agx::UInt32 const& getCollisionGroupSet() const { return (*this)->getCollisionGroupSet(); }


        inline void setSourceIndex(agx::UInt32 const& value) { (*this)->setSourceIndex(value); }
        inline void setSubsystem(agx::UInt16 const& value) { (*this)->setSubsystem(value); }
        inline void setCellIndex(agx::UInt32 const& value) { (*this)->setCellIndex(value); }
        inline void setRadius(agx::Real32 const& value) { (*this)->setRadius(value); }
        inline void setPosition(agx::Vec2f const& value) { (*this)->setPosition(value); }
        inline void setOobIndex(agx::UInt32 const& value) { (*this)->setOobIndex(value); }
        inline void setCollisionGroupSet(agx::UInt32 const& value) { (*this)->setCollisionGroupSet(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CollisionObject2DInstance : public agxData::EntityInstance
      {
      public:
        CollisionObject2DInstance();
        CollisionObject2DInstance(CollisionObject2DData* data, agx::Index index);
        CollisionObject2DInstance(agxData::EntityStorage *storage, agx::Index index);
        CollisionObject2DInstance(const agxData::EntityInstance& other);
        CollisionObject2DInstance(const agxData::EntityPtr& ptr);

        CollisionObject2DData* getData();
        const CollisionObject2DData* getData() const;

      public:
        /// \return reference to the sourceIndex attribute
        agx::UInt32& sourceIndex();
        /// \return const reference to the sourceIndex attribute
        agx::UInt32 const& sourceIndex() const;

        /// \return reference to the subsystem attribute
        agx::UInt16& subsystem();
        /// \return const reference to the subsystem attribute
        agx::UInt16 const& subsystem() const;

        /// \return reference to the cellIndex attribute
        agx::UInt32& cellIndex();
        /// \return const reference to the cellIndex attribute
        agx::UInt32 const& cellIndex() const;

        /// \return reference to the radius attribute
        agx::Real32& radius();
        /// \return const reference to the radius attribute
        agx::Real32 const& radius() const;

        /// \return reference to the position attribute
        agx::Vec2f& position();
        /// \return const reference to the position attribute
        agx::Vec2f const& position() const;

        /// \return reference to the oobIndex attribute
        agx::UInt32& oobIndex();
        /// \return const reference to the oobIndex attribute
        agx::UInt32 const& oobIndex() const;

        /// \return reference to the collisionGroupSet attribute
        agx::UInt32& collisionGroupSet();
        /// \return const reference to the collisionGroupSet attribute
        agx::UInt32 const& collisionGroupSet() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<CollisionObject2DPtr> CollisionObject2DPtrVector;
      typedef agxData::Array<CollisionObject2DPtr> CollisionObject2DPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline CollisionObject2DInstance agx::Physics::HierarchicalGrid::CollisionObject2DData::operator[] (size_t index) { return CollisionObject2DInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CollisionObject2DPtr::CollisionObject2DPtr() {}
      AGX_FORCE_INLINE CollisionObject2DPtr::CollisionObject2DPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE CollisionObject2DPtr::CollisionObject2DPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CollisionObject2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CollisionObject2DModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CollisionObject2DPtr::CollisionObject2DPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(CollisionObject2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CollisionObject2DModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CollisionObject2DPtr& CollisionObject2DPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(CollisionObject2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CollisionObject2DModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE CollisionObject2DPtr& CollisionObject2DPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(CollisionObject2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CollisionObject2DModel::instance()->fullPath().c_str());
        return *this;
      }

      inline CollisionObject2DInstance CollisionObject2DPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const CollisionObject2DInstance CollisionObject2DPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE CollisionObject2DSemantics* CollisionObject2DPtr::operator->() { return (CollisionObject2DSemantics* )this; }
      AGX_FORCE_INLINE const CollisionObject2DSemantics* CollisionObject2DPtr::operator->() const { return (const CollisionObject2DSemantics* )this; }
      AGX_FORCE_INLINE CollisionObject2DData* CollisionObject2DPtr::getData() { return static_cast<CollisionObject2DData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const CollisionObject2DData* CollisionObject2DPtr::getData() const { return static_cast<const CollisionObject2DData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::UInt32& CollisionObject2DPtr::sourceIndex() { verifyIndex(); return getData()->sourceIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObject2DPtr::sourceIndex() const { verifyIndex(); return getData()->sourceIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt16& CollisionObject2DPtr::subsystem() { verifyIndex(); return getData()->subsystem[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt16 const& CollisionObject2DPtr::subsystem() const { verifyIndex(); return getData()->subsystem[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CollisionObject2DPtr::cellIndex() { verifyIndex(); return getData()->cellIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObject2DPtr::cellIndex() const { verifyIndex(); return getData()->cellIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& CollisionObject2DPtr::radius() { verifyIndex(); return getData()->radius[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& CollisionObject2DPtr::radius() const { verifyIndex(); return getData()->radius[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vec2f& CollisionObject2DPtr::position() { verifyIndex(); return getData()->position[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec2f const& CollisionObject2DPtr::position() const { verifyIndex(); return getData()->position[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CollisionObject2DPtr::oobIndex() { verifyIndex(); return getData()->oobIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObject2DPtr::oobIndex() const { verifyIndex(); return getData()->oobIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CollisionObject2DPtr::collisionGroupSet() { verifyIndex(); return getData()->collisionGroupSet[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObject2DPtr::collisionGroupSet() const { verifyIndex(); return getData()->collisionGroupSet[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CollisionObject2DInstance::CollisionObject2DInstance() {}
      AGX_FORCE_INLINE CollisionObject2DInstance::CollisionObject2DInstance(CollisionObject2DData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE CollisionObject2DInstance::CollisionObject2DInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE CollisionObject2DInstance::CollisionObject2DInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(CollisionObject2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CollisionObject2DModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CollisionObject2DInstance::CollisionObject2DInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CollisionObject2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CollisionObject2DModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE CollisionObject2DData* CollisionObject2DInstance::getData() { return static_cast<CollisionObject2DData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const CollisionObject2DData* CollisionObject2DInstance::getData() const { return static_cast<const CollisionObject2DData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::UInt32& CollisionObject2DInstance::sourceIndex() { verifyIndex(); return getData()->sourceIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObject2DInstance::sourceIndex() const { verifyIndex(); return getData()->sourceIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt16& CollisionObject2DInstance::subsystem() { verifyIndex(); return getData()->subsystem[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt16 const& CollisionObject2DInstance::subsystem() const { verifyIndex(); return getData()->subsystem[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CollisionObject2DInstance::cellIndex() { verifyIndex(); return getData()->cellIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObject2DInstance::cellIndex() const { verifyIndex(); return getData()->cellIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& CollisionObject2DInstance::radius() { verifyIndex(); return getData()->radius[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& CollisionObject2DInstance::radius() const { verifyIndex(); return getData()->radius[getIndex()]; }

      AGX_FORCE_INLINE agx::Vec2f& CollisionObject2DInstance::position() { verifyIndex(); return getData()->position[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec2f const& CollisionObject2DInstance::position() const { verifyIndex(); return getData()->position[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CollisionObject2DInstance::oobIndex() { verifyIndex(); return getData()->oobIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObject2DInstance::oobIndex() const { verifyIndex(); return getData()->oobIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CollisionObject2DInstance::collisionGroupSet() { verifyIndex(); return getData()->collisionGroupSet[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CollisionObject2DInstance::collisionGroupSet() const { verifyIndex(); return getData()->collisionGroupSet[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CollisionObject2DSemantics::CollisionObject2DSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::CollisionObject2DPtr, "Physics.HierarchicalGrid.CollisionObject2DPtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::CollisionObject2DInstance, "Physics.HierarchicalGrid.CollisionObject2DInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

