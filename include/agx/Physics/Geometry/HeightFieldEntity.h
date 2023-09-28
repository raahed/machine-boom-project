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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_HEIGHTFIELD_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_HEIGHTFIELD_H_PLUGIN

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
#include <agx/Physics/Geometry/MeshEntity.h>


namespace agx
{
  namespace Physics
  {
    namespace Geometry
    {

      class HeightFieldModel;
      class HeightFieldData;
      class HeightFieldPtr;
      class HeightFieldInstance;
      class HeightFieldSemantics;


      AGX_DECLARE_POINTER_TYPES(HeightFieldModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.HeightField entity.
      */ 
      class AGXPHYSICS_EXPORT HeightFieldModel : public agx::Physics::Geometry::MeshModel
      {
      public:
        typedef HeightFieldPtr PtrT;

        HeightFieldModel(const agx::String& name = "HeightField");

        /// \return The entity model singleton.
        static HeightFieldModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static HeightFieldPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */

      protected:
        virtual ~HeightFieldModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::HeightFieldPtr heightField);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_HEIGHTFIELD_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_HEIGHTFIELD_DATA_SET
      class AGXPHYSICS_EXPORT HeightFieldData : public agx::Physics::Geometry::MeshData
      {
      public:
        HeightFieldInstance operator[] (size_t index);

      public:
        agxData::Array< HeightFieldPtr >& instance;

      public:

      public:
        HeightFieldData(agxData::EntityStorage* storage);
        HeightFieldData();

      protected:
        virtual ~HeightFieldData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        HeightFieldData& operator= (const HeightFieldData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT HeightFieldSemantics : public agx::Physics::Geometry::MeshSemantics
      {
      public:

        // Automatic getters

        // Semantics defined by explicit kernels

        // Automatic setters


      protected:
        friend class HeightFieldPtr;
        friend class HeightFieldInstance;
        HeightFieldSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.HeightField
      */
      class CALLABLE HeightFieldPtr : public agx::Physics::Geometry::MeshPtr
      {
      public:
        typedef HeightFieldModel ModelType;
        typedef HeightFieldData DataType;
        typedef HeightFieldInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT HeightFieldPtr();
        AGXPHYSICS_EXPORT HeightFieldPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT HeightFieldPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT HeightFieldPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT HeightFieldPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT HeightFieldPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT HeightFieldInstance instance();
        AGXPHYSICS_EXPORT const HeightFieldInstance instance() const;

        AGXPHYSICS_EXPORT HeightFieldSemantics* operator->();
        AGXPHYSICS_EXPORT const HeightFieldSemantics* operator->() const;

        HeightFieldData* getData();
        const HeightFieldData* getData() const;


        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER


        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< HeightFieldPtr > HeightFieldRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT HeightFieldInstance : public agx::Physics::Geometry::MeshInstance
      {
      public:
        HeightFieldInstance();
        HeightFieldInstance(HeightFieldData* data, agx::Index index);
        HeightFieldInstance(agxData::EntityStorage *storage, agx::Index index);
        HeightFieldInstance(const agxData::EntityInstance& other);
        HeightFieldInstance(const agxData::EntityPtr& ptr);

        HeightFieldData* getData();
        const HeightFieldData* getData() const;

      public:
      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<HeightFieldPtr> HeightFieldPtrVector;
      typedef agxData::Array<HeightFieldPtr> HeightFieldPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline HeightFieldInstance agx::Physics::Geometry::HeightFieldData::operator[] (size_t index) { return HeightFieldInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE HeightFieldPtr::HeightFieldPtr() {}
      AGX_FORCE_INLINE HeightFieldPtr::HeightFieldPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::MeshPtr(storage, id) {}
      AGX_FORCE_INLINE HeightFieldPtr::HeightFieldPtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::MeshPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(HeightFieldModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), HeightFieldModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE HeightFieldPtr::HeightFieldPtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::MeshPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(HeightFieldModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), HeightFieldModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE HeightFieldPtr& HeightFieldPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(HeightFieldModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), HeightFieldModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE HeightFieldPtr& HeightFieldPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(HeightFieldModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), HeightFieldModel::instance()->fullPath().c_str());
        return *this;
      }

      inline HeightFieldInstance HeightFieldPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const HeightFieldInstance HeightFieldPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE HeightFieldSemantics* HeightFieldPtr::operator->() { return (HeightFieldSemantics* )this; }
      AGX_FORCE_INLINE const HeightFieldSemantics* HeightFieldPtr::operator->() const { return (const HeightFieldSemantics* )this; }
      AGX_FORCE_INLINE HeightFieldData* HeightFieldPtr::getData() { return static_cast<HeightFieldData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const HeightFieldData* HeightFieldPtr::getData() const { return static_cast<const HeightFieldData* >(agxData::EntityPtr::getData()); }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE HeightFieldInstance::HeightFieldInstance() {}
      AGX_FORCE_INLINE HeightFieldInstance::HeightFieldInstance(HeightFieldData* data, agx::Index index) : agx::Physics::Geometry::MeshInstance(data, index) {}
      AGX_FORCE_INLINE HeightFieldInstance::HeightFieldInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::MeshInstance(storage, index) {}
      AGX_FORCE_INLINE HeightFieldInstance::HeightFieldInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::MeshInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(HeightFieldModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), HeightFieldModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE HeightFieldInstance::HeightFieldInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::MeshInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(HeightFieldModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), HeightFieldModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE HeightFieldData* HeightFieldInstance::getData() { return static_cast<HeightFieldData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const HeightFieldData* HeightFieldInstance::getData() const { return static_cast<const HeightFieldData* >(agxData::EntityInstance::getData()); }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE HeightFieldSemantics::HeightFieldSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::HeightFieldPtr, "Physics.Geometry.HeightFieldPtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::HeightFieldInstance, "Physics.Geometry.HeightFieldInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

