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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_GRIDOVERLAP_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_GRIDOVERLAP_H_PLUGIN

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


namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class GridOverlapModel;
      class GridOverlapData;
      class GridOverlapPtr;
      class GridOverlapInstance;
      class GridOverlapSemantics;


      AGX_DECLARE_POINTER_TYPES(GridOverlapModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.GridOverlap entity.
      */ 
      class AGXPHYSICS_EXPORT GridOverlapModel : public agxData::EntityModel
      {
      public:
        typedef GridOverlapPtr PtrT;

        GridOverlapModel(const agx::String& name = "GridOverlap");

        /// \return The entity model singleton.
        static GridOverlapModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static GridOverlapPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::UInt32 >* collisionObject1Attribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* collisionObject2Attribute;

      protected:
        virtual ~GridOverlapModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::GridOverlapPtr gridOverlap);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_GRIDOVERLAP_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_GRIDOVERLAP_DATA_SET
      class AGXPHYSICS_EXPORT GridOverlapData : public agxData::EntityData
      {
      public:
        GridOverlapInstance operator[] (size_t index);

      public:
        agxData::Array< GridOverlapPtr >& instance;
        agxData::Array< agx::UInt32 > collisionObject1;
        agxData::Array< agx::UInt32 > collisionObject2;

      public:
        typedef agx::UInt32 collisionObject1Type;
        typedef agx::UInt32 collisionObject2Type;

      public:
        GridOverlapData(agxData::EntityStorage* storage);
        GridOverlapData();

      protected:
        virtual ~GridOverlapData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        GridOverlapData& operator= (const GridOverlapData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT GridOverlapSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::UInt32 const& getCollisionObject1() const;
        agx::UInt32 const& getCollisionObject2() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setCollisionObject1(agx::UInt32 const& value);
        void setCollisionObject2(agx::UInt32 const& value);


      protected:
        friend class GridOverlapPtr;
        friend class GridOverlapInstance;
        GridOverlapSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.GridOverlap
      */
      class CALLABLE GridOverlapPtr : public agxData::EntityPtr
      {
      public:
        typedef GridOverlapModel ModelType;
        typedef GridOverlapData DataType;
        typedef GridOverlapInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT GridOverlapPtr();
        AGXPHYSICS_EXPORT GridOverlapPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT GridOverlapPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT GridOverlapPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT GridOverlapPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT GridOverlapPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT GridOverlapInstance instance();
        AGXPHYSICS_EXPORT const GridOverlapInstance instance() const;

        AGXPHYSICS_EXPORT GridOverlapSemantics* operator->();
        AGXPHYSICS_EXPORT const GridOverlapSemantics* operator->() const;

        GridOverlapData* getData();
        const GridOverlapData* getData() const;


        /// \return reference to the collisionObject1 attribute
        AGXPHYSICS_EXPORT agx::UInt32& collisionObject1();
        /// \return const reference to the collisionObject1 attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& collisionObject1() const;

        /// \return reference to the collisionObject2 attribute
        AGXPHYSICS_EXPORT agx::UInt32& collisionObject2();
        /// \return const reference to the collisionObject2 attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& collisionObject2() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::UInt32 const& getCollisionObject1() const { return (*this)->getCollisionObject1(); }
        inline agx::UInt32 const& getCollisionObject2() const { return (*this)->getCollisionObject2(); }


        inline void setCollisionObject1(agx::UInt32 const& value) { (*this)->setCollisionObject1(value); }
        inline void setCollisionObject2(agx::UInt32 const& value) { (*this)->setCollisionObject2(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT GridOverlapInstance : public agxData::EntityInstance
      {
      public:
        GridOverlapInstance();
        GridOverlapInstance(GridOverlapData* data, agx::Index index);
        GridOverlapInstance(agxData::EntityStorage *storage, agx::Index index);
        GridOverlapInstance(const agxData::EntityInstance& other);
        GridOverlapInstance(const agxData::EntityPtr& ptr);

        GridOverlapData* getData();
        const GridOverlapData* getData() const;

      public:
        /// \return reference to the collisionObject1 attribute
        agx::UInt32& collisionObject1();
        /// \return const reference to the collisionObject1 attribute
        agx::UInt32 const& collisionObject1() const;

        /// \return reference to the collisionObject2 attribute
        agx::UInt32& collisionObject2();
        /// \return const reference to the collisionObject2 attribute
        agx::UInt32 const& collisionObject2() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<GridOverlapPtr> GridOverlapPtrVector;
      typedef agxData::Array<GridOverlapPtr> GridOverlapPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline GridOverlapInstance agx::Physics::HierarchicalGrid::GridOverlapData::operator[] (size_t index) { return GridOverlapInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE GridOverlapPtr::GridOverlapPtr() {}
      AGX_FORCE_INLINE GridOverlapPtr::GridOverlapPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE GridOverlapPtr::GridOverlapPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(GridOverlapModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), GridOverlapModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE GridOverlapPtr::GridOverlapPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(GridOverlapModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), GridOverlapModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE GridOverlapPtr& GridOverlapPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(GridOverlapModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), GridOverlapModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE GridOverlapPtr& GridOverlapPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(GridOverlapModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), GridOverlapModel::instance()->fullPath().c_str());
        return *this;
      }

      inline GridOverlapInstance GridOverlapPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const GridOverlapInstance GridOverlapPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE GridOverlapSemantics* GridOverlapPtr::operator->() { return (GridOverlapSemantics* )this; }
      AGX_FORCE_INLINE const GridOverlapSemantics* GridOverlapPtr::operator->() const { return (const GridOverlapSemantics* )this; }
      AGX_FORCE_INLINE GridOverlapData* GridOverlapPtr::getData() { return static_cast<GridOverlapData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const GridOverlapData* GridOverlapPtr::getData() const { return static_cast<const GridOverlapData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::UInt32& GridOverlapPtr::collisionObject1() { verifyIndex(); return getData()->collisionObject1[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& GridOverlapPtr::collisionObject1() const { verifyIndex(); return getData()->collisionObject1[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& GridOverlapPtr::collisionObject2() { verifyIndex(); return getData()->collisionObject2[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& GridOverlapPtr::collisionObject2() const { verifyIndex(); return getData()->collisionObject2[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE GridOverlapInstance::GridOverlapInstance() {}
      AGX_FORCE_INLINE GridOverlapInstance::GridOverlapInstance(GridOverlapData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE GridOverlapInstance::GridOverlapInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE GridOverlapInstance::GridOverlapInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(GridOverlapModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), GridOverlapModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE GridOverlapInstance::GridOverlapInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(GridOverlapModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), GridOverlapModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE GridOverlapData* GridOverlapInstance::getData() { return static_cast<GridOverlapData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const GridOverlapData* GridOverlapInstance::getData() const { return static_cast<const GridOverlapData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::UInt32& GridOverlapInstance::collisionObject1() { verifyIndex(); return getData()->collisionObject1[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& GridOverlapInstance::collisionObject1() const { verifyIndex(); return getData()->collisionObject1[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& GridOverlapInstance::collisionObject2() { verifyIndex(); return getData()->collisionObject2[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& GridOverlapInstance::collisionObject2() const { verifyIndex(); return getData()->collisionObject2[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE GridOverlapSemantics::GridOverlapSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::GridOverlapPtr, "Physics.HierarchicalGrid.GridOverlapPtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::GridOverlapInstance, "Physics.HierarchicalGrid.GridOverlapInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

