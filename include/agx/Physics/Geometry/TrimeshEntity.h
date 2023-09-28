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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_TRIMESH_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_TRIMESH_H_PLUGIN

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

      class TrimeshModel;
      class TrimeshData;
      class TrimeshPtr;
      class TrimeshInstance;
      class TrimeshSemantics;


      AGX_DECLARE_POINTER_TYPES(TrimeshModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.Trimesh entity.
      */ 
      class AGXPHYSICS_EXPORT TrimeshModel : public agx::Physics::Geometry::MeshModel
      {
      public:
        typedef TrimeshPtr PtrT;

        TrimeshModel(const agx::String& name = "Trimesh");

        /// \return The entity model singleton.
        static TrimeshModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static TrimeshPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */

      protected:
        virtual ~TrimeshModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::TrimeshPtr trimesh);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_TRIMESH_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_TRIMESH_DATA_SET
      class AGXPHYSICS_EXPORT TrimeshData : public agx::Physics::Geometry::MeshData
      {
      public:
        TrimeshInstance operator[] (size_t index);

      public:
        agxData::Array< TrimeshPtr >& instance;

      public:

      public:
        TrimeshData(agxData::EntityStorage* storage);
        TrimeshData();

      protected:
        virtual ~TrimeshData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        TrimeshData& operator= (const TrimeshData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT TrimeshSemantics : public agx::Physics::Geometry::MeshSemantics
      {
      public:

        // Automatic getters

        // Semantics defined by explicit kernels

        // Automatic setters


      protected:
        friend class TrimeshPtr;
        friend class TrimeshInstance;
        TrimeshSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.Trimesh
      */
      class CALLABLE TrimeshPtr : public agx::Physics::Geometry::MeshPtr
      {
      public:
        typedef TrimeshModel ModelType;
        typedef TrimeshData DataType;
        typedef TrimeshInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT TrimeshPtr();
        AGXPHYSICS_EXPORT TrimeshPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT TrimeshPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT TrimeshPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT TrimeshPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT TrimeshPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT TrimeshInstance instance();
        AGXPHYSICS_EXPORT const TrimeshInstance instance() const;

        AGXPHYSICS_EXPORT TrimeshSemantics* operator->();
        AGXPHYSICS_EXPORT const TrimeshSemantics* operator->() const;

        TrimeshData* getData();
        const TrimeshData* getData() const;


        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER


        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< TrimeshPtr > TrimeshRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT TrimeshInstance : public agx::Physics::Geometry::MeshInstance
      {
      public:
        TrimeshInstance();
        TrimeshInstance(TrimeshData* data, agx::Index index);
        TrimeshInstance(agxData::EntityStorage *storage, agx::Index index);
        TrimeshInstance(const agxData::EntityInstance& other);
        TrimeshInstance(const agxData::EntityPtr& ptr);

        TrimeshData* getData();
        const TrimeshData* getData() const;

      public:
      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<TrimeshPtr> TrimeshPtrVector;
      typedef agxData::Array<TrimeshPtr> TrimeshPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline TrimeshInstance agx::Physics::Geometry::TrimeshData::operator[] (size_t index) { return TrimeshInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE TrimeshPtr::TrimeshPtr() {}
      AGX_FORCE_INLINE TrimeshPtr::TrimeshPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::MeshPtr(storage, id) {}
      AGX_FORCE_INLINE TrimeshPtr::TrimeshPtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::MeshPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(TrimeshModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), TrimeshModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE TrimeshPtr::TrimeshPtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::MeshPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(TrimeshModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), TrimeshModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE TrimeshPtr& TrimeshPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(TrimeshModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), TrimeshModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE TrimeshPtr& TrimeshPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(TrimeshModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), TrimeshModel::instance()->fullPath().c_str());
        return *this;
      }

      inline TrimeshInstance TrimeshPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const TrimeshInstance TrimeshPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE TrimeshSemantics* TrimeshPtr::operator->() { return (TrimeshSemantics* )this; }
      AGX_FORCE_INLINE const TrimeshSemantics* TrimeshPtr::operator->() const { return (const TrimeshSemantics* )this; }
      AGX_FORCE_INLINE TrimeshData* TrimeshPtr::getData() { return static_cast<TrimeshData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const TrimeshData* TrimeshPtr::getData() const { return static_cast<const TrimeshData* >(agxData::EntityPtr::getData()); }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE TrimeshInstance::TrimeshInstance() {}
      AGX_FORCE_INLINE TrimeshInstance::TrimeshInstance(TrimeshData* data, agx::Index index) : agx::Physics::Geometry::MeshInstance(data, index) {}
      AGX_FORCE_INLINE TrimeshInstance::TrimeshInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::MeshInstance(storage, index) {}
      AGX_FORCE_INLINE TrimeshInstance::TrimeshInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::MeshInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(TrimeshModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), TrimeshModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE TrimeshInstance::TrimeshInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::MeshInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(TrimeshModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), TrimeshModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE TrimeshData* TrimeshInstance::getData() { return static_cast<TrimeshData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const TrimeshData* TrimeshInstance::getData() const { return static_cast<const TrimeshData* >(agxData::EntityInstance::getData()); }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE TrimeshSemantics::TrimeshSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::TrimeshPtr, "Physics.Geometry.TrimeshPtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::TrimeshInstance, "Physics.Geometry.TrimeshInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

